# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect and compare token-to-expert distributions from MoE traces.

The collect command runs every dataset/batch-size combination in a fresh set
of vLLM workers. The collect-solve command captures into temporary storage and
keeps only the resulting placement and route plans. The plot command can
regenerate figures from existing traces. See
``moe_trace_expert_distribution.md`` for complete examples.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    repository: str
    config: str | None
    split: str


DATASET_SPECS = {
    "math": DatasetSpec("openai/gsm8k", "main", "test"),
    "code": DatasetSpec("openai/openai_humaneval", None, "test"),
    "chat": DatasetSpec("HuggingFaceH4/mt_bench_prompts", None, "train"),
    "summary": DatasetSpec("abisee/cnn_dailymail", "3.0.0", "validation"),
}
DEFAULT_DATASETS = tuple(DATASET_SPECS)
_DATASET_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_ASSISTANT_ROLES = {"assistant", "gpt"}


@dataclass
class LayerDistribution:
    raw_steps: np.ndarray
    batch_indices: np.ndarray
    shares: np.ndarray
    imbalance: np.ndarray
    scheduled_tokens: np.ndarray
    totals: np.ndarray
    rank_totals: dict[int, np.ndarray]
    rank_step_counts: dict[int, np.ndarray]


@dataclass
class TraceDistribution:
    num_experts: int
    ep_size: int
    expert_placement_strategy: str
    layers: dict[int, LayerDistribution]


def _conversation_prompt(item: dict[str, Any]) -> str | None:
    conversations = item.get("conversations") or item.get("messages")
    if not isinstance(conversations, list):
        return None
    turns = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from") or turn.get("role") or "user").lower()
        content = turn.get("value", turn.get("content"))
        if isinstance(content, str) and content.strip():
            turns.append((role, content.strip()))
    if turns and turns[-1][0] in _ASSISTANT_ROLES:
        turns.pop()
    if not turns:
        return None
    labels = {"human": "User", "user": "User", "gpt": "Assistant"}
    lines = [f"{labels.get(role, role.title())}: {text}" for role, text in turns]
    if turns[-1][0] not in _ASSISTANT_ROLES:
        lines.append("Assistant:")
    return "\n".join(lines)


def _item_to_prompt(dataset_name: str, item: Any) -> str | None:
    if isinstance(item, str):
        return item.strip() or None
    if not isinstance(item, dict):
        return None

    conversation = _conversation_prompt(item)
    if conversation is not None:
        return conversation

    if dataset_name == "math" and isinstance(item.get("question"), str):
        return (
            f"{item['question'].strip()}\nPlease reason step by step and put "
            "the final answer within \\boxed{}."
        )
    if dataset_name == "code" and isinstance(item.get("prompt"), str):
        return (
            "Write a solution to the following problem and make sure it passes "
            f"the tests:\n```python\n{item['prompt'].strip()}\n```"
        )
    if dataset_name == "chat":
        prompt = item.get("prompt")
        if isinstance(prompt, list):
            prompt = next((value for value in prompt if isinstance(value, str)), None)
        if isinstance(prompt, str):
            return prompt.strip() or None
    if dataset_name == "summary":
        article = item.get("article", item.get("document"))
        if isinstance(article, str):
            return f"Summarize the following article:\n\n{article.strip()}"

    for key in ("prompt", "text", "question", "article", "document"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _json_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "prompts", "instances", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    return []


def _load_local_prompts(dataset_name: str, path: Path, num_prompts: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        items = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix.lower() == ".json":
        items = _json_items(json.loads(path.read_text(encoding="utf-8")))
    else:
        items = path.read_text(encoding="utf-8").splitlines()
    prompts = []
    for item in items:
        prompt = _item_to_prompt(dataset_name, item)
        if prompt:
            prompts.append(prompt)
        if len(prompts) == num_prompts:
            break
    if len(prompts) < num_prompts:
        raise ValueError(
            f"Dataset {dataset_name!r} provided {len(prompts)} prompts, "
            f"but --num-prompts requested {num_prompts}"
        )
    return prompts


def _load_hf_prompts(dataset_name: str, num_prompts: int) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Built-in datasets require the optional 'datasets' package. "
            "Install vLLM's benchmark dependencies or use --dataset-path."
        ) from error

    spec = DATASET_SPECS[dataset_name]
    dataset = load_dataset(
        spec.repository,
        spec.config,
        split=spec.split,
        streaming=True,
    )
    prompts = []
    for item in dataset:
        prompt = _item_to_prompt(dataset_name, item)
        if prompt:
            prompts.append(prompt)
        if len(prompts) == num_prompts:
            break
    if len(prompts) < num_prompts:
        raise ValueError(
            f"Dataset {dataset_name!r} provided only {len(prompts)} prompts"
        )
    return prompts


def _parse_dataset_paths(values: list[str]) -> dict[str, Path]:
    paths = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--dataset-path must use NAME=/path/to/file syntax")
        if name in paths:
            raise ValueError(f"Duplicate --dataset-path for {name!r}")
        paths[name] = Path(raw_path).expanduser().resolve()
    return paths


def _load_datasets(args: argparse.Namespace) -> dict[str, list[str]]:
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")
    paths = _parse_dataset_paths(args.dataset_path)
    unknown_paths = set(paths).difference(args.datasets)
    if unknown_paths:
        names = ", ".join(sorted(unknown_paths))
        raise ValueError(f"--dataset-path names not present in --datasets: {names}")
    loaded = {}
    for name in args.datasets:
        if not _DATASET_NAME.fullmatch(name):
            raise ValueError(f"Invalid dataset name: {name!r}")
        if name in paths:
            loaded[name] = _load_local_prompts(name, paths[name], args.num_prompts)
        elif name in DATASET_SPECS:
            loaded[name] = _load_hf_prompts(name, args.num_prompts)
        else:
            raise ValueError(
                f"Custom dataset {name!r} requires --dataset-path {name}=PATH"
            )
    return loaded


def _num_experts(hf_config: Any) -> int:
    for name in ("num_experts", "n_routed_experts", "num_local_experts"):
        value = getattr(hf_config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not determine the model's number of experts")


def _rank_worker(
    args: argparse.Namespace,
    prompts: list[str],
    rank: int,
    master_port: int,
    batch_size: int,
    activation_dir: Path,
    result_dir: Path,
) -> None:
    trace_max_steps = getattr(
        args,
        "trace_max_steps",
        len(prompts) * (args.max_model_len + args.max_new_tokens),
    )
    os.environ.update(
        {
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": str(args.ep_size),
            "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(master_port),
            "VLLM_MOE_TRACE_DIR": str(activation_dir),
            "VLLM_MOE_TRACE_MODE": "expert_distribution",
            "VLLM_MOE_TRACE_MAX_STEPS": str(trace_max_steps),
        }
    )

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        max_model_len=args.max_model_len,
        max_num_seqs=min(len(prompts), batch_size),
        enforce_eager=True,
        enable_prefix_caching=False,
        load_format=args.load_format,
        moe_backend=args.moe_backend,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    request_batches = {}
    for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
        outputs = llm.generate(
            prompts[start : start + batch_size],
            sampling_params,
            use_tqdm=True,
        )
        request_batches.update(
            {f"{rank}:{output.request_id}": batch_index for output in outputs}
        )
    result = {
        "rank": rank,
        "num_experts": _num_experts(llm.model_config.hf_text_config),
        "request_batches": request_batches,
    }
    (result_dir / f"rank_{rank:05d}.json").write_text(
        json.dumps(result), encoding="utf-8"
    )


def _experiment_dir(work_dir: Path, dataset: str, batch_size: int) -> Path:
    return work_dir / f"dataset_{dataset}" / f"batch_{batch_size:04d}"


def _collect_experiment(
    args: argparse.Namespace,
    dataset_name: str,
    prompts: list[str],
    batch_size: int,
) -> dict[str, Any]:
    experiment_dir = _experiment_dir(args.output_dir, dataset_name, batch_size)
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(f"Experiment output is not empty: {experiment_dir}")
    activation_dir = experiment_dir / "activations"
    result_dir = experiment_dir / "results"
    activation_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    (activation_dir / "trace_config.json").write_text(
        json.dumps(
            {
                "capture_next_gate_base_logits": False,
                "capture_topk_ids": True,
            }
        ),
        encoding="utf-8",
    )

    floor, remainder = divmod(len(prompts), args.ep_size)

    def shard_start(rank: int) -> int:
        return rank * floor + min(rank, remainder)

    shards = [
        prompts[shard_start(rank) : shard_start(rank + 1)]
        for rank in range(args.ep_size)
    ]
    from vllm.utils.network_utils import get_open_port

    context = mp.get_context("spawn")
    master_port = get_open_port()
    processes = [
        context.Process(
            target=_rank_worker,
            args=(
                args,
                shards[rank],
                rank,
                master_port,
                batch_size,
                activation_dir,
                result_dir,
            ),
        )
        for rank in range(args.ep_size)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=args.timeout)
            if process.exitcode is None:
                raise TimeoutError(f"Worker {process.pid} exceeded --timeout")
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker {process.pid} exited with code {process.exitcode}"
                )
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
        raise

    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(result_dir.glob("rank_*.json"))
    ]
    expert_counts = {int(result["num_experts"]) for result in results}
    if len(results) != args.ep_size or len(expert_counts) != 1:
        raise RuntimeError("Collection ranks returned incomplete metadata")
    request_batches = {
        request_id: batch_index
        for result in results
        for request_id, batch_index in result["request_batches"].items()
    }
    metadata = {
        "model": args.model,
        "dataset": dataset_name,
        "batch_size_per_rank": batch_size,
        "ep_size": args.ep_size,
        "expert_placement_strategy": "linear",
        "num_prompts": len(prompts),
        "num_experts": expert_counts.pop(),
        "max_new_tokens": args.max_new_tokens,
        "trace_mode": "expert_distribution",
        "capture_topk_ids": True,
        "request_batches": request_batches,
        "trace_shape": "topk_ids[token, top_k] per layer and forward step",
    }
    (experiment_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def _aggregate_trace(experiment_dir: Path) -> TraceDistribution:
    import torch

    metadata_path = experiment_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    num_experts = int(metadata.get("num_experts", 0))
    request_batches = metadata.get("request_batches", {})
    counts: dict[tuple[int, int], np.ndarray] = {}
    rank_counts: dict[tuple[int, int], np.ndarray] = {}
    rank_step_counts: dict[tuple[int, int, int], np.ndarray] = {}
    scheduled_tokens: dict[tuple[int, int], int] = {}
    batches: dict[tuple[int, int], set[int]] = {}
    paths = sorted((experiment_dir / "activations").glob("rank_*/step_*_layer_*.pt"))
    if not paths:
        raise ValueError(f"No MoE trace records found under {experiment_dir}")
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=True)
        request_ids = record.get("request_ids", [])
        rank = int(record["rank"])
        layer = int(record["layer_id"])
        step = int(record["step"])
        key = layer, step
        if "expert_counts" in record:
            value = record["expert_counts"].to(torch.int64).numpy()
            if num_experts == 0:
                num_experts = len(value)
            if len(value) != num_experts:
                raise ValueError(f"Inconsistent expert count in {path}")
        else:
            ids = record["topk_ids"].to(torch.int64).numpy().reshape(-1)
            if num_experts == 0:
                num_experts = int(record["router_logits"].shape[1])
            ids = ids[(ids >= 0) & (ids < num_experts)]
            value = np.bincount(ids, minlength=num_experts).astype(np.int64)
        counts[key] = counts.get(key, np.zeros(num_experts, dtype=np.int64)) + value
        rank_key = rank, layer
        rank_counts[rank_key] = (
            rank_counts.get(rank_key, np.zeros(num_experts, dtype=np.int64)) + value
        )
        rank_step_counts[(rank, layer, step)] = value
        recorded_tokens = record["topk_ids"].shape[0] if "topk_ids" in record else 0
        scheduled_tokens[key] = scheduled_tokens.get(key, 0) + int(
            record.get("num_scheduled_tokens", recorded_tokens)
        )
        record_batches = {
            int(request_batches[f"{rank}:{request_id}"])
            for request_id in set(request_ids)
            if f"{rank}:{request_id}" in request_batches
        }
        batches.setdefault(key, set()).update(record_batches)

    traced_ranks = sorted({rank for rank, _ in rank_counts})
    inferred_ep_size = traced_ranks[-1] + 1
    ep_size = int(metadata.get("ep_size", inferred_ep_size))
    if ep_size < inferred_ep_size:
        raise ValueError(
            f"Trace contains rank {traced_ranks[-1]}, but metadata ep_size is {ep_size}"
        )
    placement_strategy = str(metadata.get("expert_placement_strategy", "linear"))
    if placement_strategy not in ("linear", "round_robin"):
        raise ValueError(
            f"Unsupported expert placement strategy: {placement_strategy!r}"
        )

    layers = {}
    for layer in sorted({layer for layer, _ in counts}):
        raw_steps = np.asarray(
            sorted(step for count_layer, step in counts if count_layer == layer),
            dtype=np.int64,
        )
        layer_counts = np.stack([counts[(layer, int(step))] for step in raw_steps])
        row_totals = layer_counts.sum(axis=1, keepdims=True)
        shares = np.divide(
            layer_counts,
            row_totals,
            out=np.zeros_like(layer_counts, dtype=np.float64),
            where=row_totals != 0,
        )
        mean_load = layer_counts.mean(axis=1)
        imbalance = np.divide(
            layer_counts.max(axis=1),
            mean_load,
            out=np.zeros_like(mean_load, dtype=np.float64),
            where=mean_load != 0,
        )
        batch_indices = np.asarray(
            [
                (
                    min(batches[(layer, int(step))])
                    if batches.get((layer, int(step)))
                    else -1
                )
                for step in raw_steps
            ],
            dtype=np.int64,
        )
        layers[layer] = LayerDistribution(
            raw_steps=raw_steps,
            batch_indices=batch_indices,
            shares=shares * 100.0,
            imbalance=imbalance,
            scheduled_tokens=np.asarray(
                [scheduled_tokens[(layer, int(step))] for step in raw_steps],
                dtype=np.int64,
            ),
            totals=layer_counts.sum(axis=0),
            rank_totals={
                rank: rank_counts[(rank, layer)]
                for rank, count_layer in sorted(rank_counts)
                if count_layer == layer
            },
            rank_step_counts={
                rank: np.stack(
                    [
                        rank_step_counts.get(
                            (rank, layer, int(step)),
                            np.zeros(num_experts, dtype=np.int64),
                        )
                        for step in raw_steps
                    ]
                )
                for rank in range(ep_size)
            },
        )
    return TraceDistribution(num_experts, ep_size, placement_strategy, layers)


def _sort_expert_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expert_ids = np.arange(len(counts), dtype=np.int64)
    order = np.lexsort((expert_ids, -counts))
    return expert_ids[order], counts[order]


def _validate_top_n_experts(values: list[int], num_experts: int) -> list[int]:
    values = list(dict.fromkeys(values))
    invalid = [value for value in values if not 1 <= value <= num_experts]
    if invalid:
        raise ValueError(
            f"--top-n-experts values must be in [1, {num_experts}], got {invalid}"
        )
    return values


def _local_expert_ids(
    num_experts: int,
    ep_size: int,
    rank: int,
    placement_strategy: str = "linear",
) -> set[int]:
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    if not 0 <= rank < ep_size:
        raise ValueError(f"rank must be in [0, {ep_size}), got {rank}")
    if placement_strategy == "round_robin":
        return set(range(rank, num_experts, ep_size))
    if placement_strategy != "linear":
        raise ValueError(
            f"Unsupported expert placement strategy: {placement_strategy!r}"
        )
    base_experts, remainder = divmod(num_experts, ep_size)
    local_num_experts = base_experts + int(rank < remainder)
    start = rank * base_experts + min(rank, remainder)
    return set(range(start, start + local_num_experts))


def _top_n_expert_coverage(
    counts: np.ndarray,
    top_n_experts: list[int],
    local_expert_ids: set[int] | None = None,
) -> dict[int, dict[str, Any]]:
    expert_ids, sorted_counts = _sort_expert_counts(counts)
    total = int(sorted_counts.sum())
    coverage = {}
    for top_n in top_n_experts:
        assignment_count = int(sorted_counts[:top_n].sum())
        metrics = {
            "selected_expert_ids": expert_ids[:top_n].tolist(),
            "token_expert_assignments": assignment_count,
            "assignment_share_percent": (
                assignment_count / total * 100.0 if total else 0.0
            ),
        }
        if local_expert_ids is not None:
            selected_local_ids = [
                expert_id
                for expert_id in metrics["selected_expert_ids"]
                if expert_id in local_expert_ids
            ]
            metrics.update(
                {
                    "local_expert_ids": selected_local_ids,
                    "local_expert_count": len(selected_local_ids),
                    "local_expert_share_percent": (
                        len(selected_local_ids) / top_n * 100.0
                    ),
                }
            )
        coverage[top_n] = metrics
    return coverage


def _categorical_expert_colors(num_experts: int) -> np.ndarray:
    from matplotlib.colors import hsv_to_rgb

    expert_ids = np.arange(num_experts)
    hues = np.mod(expert_ids * 0.618033988749895, 1.0)
    saturations = np.where(expert_ids % 2 == 0, 0.68, 0.92)
    values = np.where((expert_ids // 2) % 2 == 0, 0.92, 0.72)
    return hsv_to_rgb(np.column_stack((hues, saturations, values)))


def _selected_layers(
    distributions: dict[tuple[str, int], TraceDistribution],
    requested: list[int] | None,
) -> list[int]:
    common = set.intersection(
        *(set(distribution.layers) for distribution in distributions.values())
    )
    if not common:
        raise ValueError("Experiments do not contain any common MoE layer")
    available = sorted(common)
    if requested:
        missing = sorted(set(requested).difference(common))
        if missing:
            raise ValueError(f"Requested layers are missing from a trace: {missing}")
        return list(dict.fromkeys(requested))
    return [available[0]]


def _write_distribution_data(
    work_dir: Path,
    datasets: list[str],
    batch_sizes: list[int],
    distributions: dict[tuple[str, int], TraceDistribution],
    top_n_experts: list[int],
) -> Path:
    experiments = []
    for dataset in datasets:
        for batch_size in batch_sizes:
            distribution = distributions[(dataset, batch_size)]
            layers = {}
            for layer, layer_distribution in distribution.layers.items():
                counts = layer_distribution.totals
                total = int(counts.sum())
                ranks = {}
                for rank, rank_counts in layer_distribution.rank_totals.items():
                    expert_ids, sorted_counts = _sort_expert_counts(rank_counts)
                    local_expert_ids = _local_expert_ids(
                        distribution.num_experts,
                        distribution.ep_size,
                        rank,
                        distribution.expert_placement_strategy,
                    )
                    ranks[str(rank)] = {
                        "total_token_expert_assignments": int(rank_counts.sum()),
                        "expert_ids_descending": expert_ids.tolist(),
                        "expert_counts_descending": sorted_counts.tolist(),
                        "top_n_expert_coverage": _top_n_expert_coverage(
                            rank_counts, top_n_experts, local_expert_ids
                        ),
                    }
                layers[str(layer)] = {
                    "num_forward_steps": int(len(layer_distribution.raw_steps)),
                    "forward_indices": list(range(len(layer_distribution.raw_steps))),
                    "raw_model_forward_steps": layer_distribution.raw_steps.tolist(),
                    "batch_indices": layer_distribution.batch_indices.tolist(),
                    "max_over_mean": layer_distribution.imbalance.tolist(),
                    "scheduled_token_counts": (
                        layer_distribution.scheduled_tokens.tolist()
                    ),
                    "total_token_expert_assignments": total,
                    "expert_counts": counts.tolist(),
                    "expert_share_percent": (
                        (counts / total * 100.0).tolist()
                        if total
                        else [0.0] * distribution.num_experts
                    ),
                    "top_n_expert_coverage": _top_n_expert_coverage(
                        counts, top_n_experts
                    ),
                    "ranks": ranks,
                }
            experiments.append(
                {
                    "dataset": dataset,
                    "batch_size_per_rank": batch_size,
                    "num_experts": distribution.num_experts,
                    "ep_size": distribution.ep_size,
                    "expert_placement_strategy": (
                        distribution.expert_placement_strategy
                    ),
                    "layers": layers,
                }
            )
    output = work_dir / "expert_distribution.json"
    output.write_text(
        json.dumps({"experiments": experiments}, indent=2), encoding="utf-8"
    )
    return output


def _plot_distributions(
    *,
    work_dir: Path,
    output_dir: Path,
    datasets: list[str],
    batch_sizes: list[int],
    requested_layers: list[int] | None,
    max_steps: int | None,
    top_n_experts: list[int],
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if max_steps is not None and max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    distributions = {
        (dataset, batch_size): _aggregate_trace(
            _experiment_dir(work_dir, dataset, batch_size)
        )
        for dataset in datasets
        for batch_size in batch_sizes
    }
    expert_sizes = {item.num_experts for item in distributions.values()}
    if len(expert_sizes) != 1:
        raise ValueError("Experiments have different numbers of experts")
    num_experts = expert_sizes.pop()
    top_n_experts = _validate_top_n_experts(top_n_experts, num_experts)
    colors = _categorical_expert_colors(num_experts)
    layers = _selected_layers(distributions, requested_layers)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for dataset in datasets:
        for batch_size in batch_sizes:
            distribution = distributions[(dataset, batch_size)]
            for layer in layers:
                fig, axes = plt.subplots(
                    3,
                    1,
                    figsize=(8.5, 9.6),
                    squeeze=False,
                    sharex=True,
                )
                layer_distribution = distribution.layers[layer]
                shares = layer_distribution.shares
                imbalance = layer_distribution.imbalance
                scheduled_tokens = layer_distribution.scheduled_tokens
                batch_indices = layer_distribution.batch_indices
                if max_steps is not None:
                    shares = shares[:max_steps]
                    imbalance = imbalance[:max_steps]
                    scheduled_tokens = scheduled_tokens[:max_steps]
                    batch_indices = batch_indices[:max_steps]
                forward_indices = np.arange(len(shares))
                distribution_axis = axes[0, 0]
                imbalance_axis = axes[1, 0]
                token_axis = axes[2, 0]
                distribution_axis.stackplot(
                    forward_indices,
                    shares.T,
                    colors=colors,
                    edgecolor="#202020",
                    linewidth=0.12,
                )
                imbalance_axis.plot(
                    forward_indices,
                    imbalance,
                    color="#202020",
                    linewidth=1.4,
                    label="max / mean",
                )
                imbalance_axis.axhline(
                    1.0,
                    color="#777777",
                    linewidth=0.9,
                    linestyle=":",
                    label="perfect balance",
                )
                token_axis.plot(
                    forward_indices,
                    scheduled_tokens,
                    color="#2F6F8F",
                    linewidth=1.4,
                    label="scheduled tokens",
                )
                boundaries = np.flatnonzero(batch_indices[1:] != batch_indices[:-1])
                for boundary in boundaries:
                    for axis in (distribution_axis, imbalance_axis, token_axis):
                        axis.axvline(
                            float(boundary) + 0.5,
                            color="black",
                            linewidth=0.7,
                            alpha=0.45,
                            linestyle="--",
                        )
                distribution_axis.set_ylabel("Expert load share (%)")
                imbalance_axis.set_ylabel("Load imbalance (max / mean)")
                token_axis.set_ylabel("Scheduled tokens")
                token_axis.set_xlabel("Model-forward index")
                distribution_axis.set_title("Token-to-expert distribution")
                imbalance_axis.set_title("Expert load imbalance")
                token_axis.set_title("Scheduled tokens per forward")
                imbalance_axis.legend(loc="upper right")
                token_axis.legend(loc="upper right")
                token_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
                if len(forward_indices) == 1:
                    token_axis.set_xlim(-0.5, 0.5)
                else:
                    token_axis.set_xlim(0.0, float(forward_indices[-1]))
                distribution_axis.set_ylim(0.0, 100.0)
                distribution_axis.grid(alpha=0.15)
                imbalance_axis.grid(alpha=0.2)
                token_axis.grid(alpha=0.2)
                fig.suptitle(
                    f"{dataset}, batch size {batch_size}, layer {layer}",
                    fontsize=14,
                )
                fig.tight_layout()
                output = output_dir / (
                    f"expert_distribution_{dataset}_batch_{batch_size:04d}_"
                    f"layer_{layer:04d}.png"
                )
                fig.savefig(output, dpi=200, bbox_inches="tight")
                plt.close(fig)
                outputs.append(output)
                for rank, rank_counts in layer_distribution.rank_totals.items():
                    expert_ids, sorted_counts = _sort_expert_counts(rank_counts)
                    local_expert_ids = _local_expert_ids(
                        distribution.num_experts,
                        distribution.ep_size,
                        rank,
                        distribution.expert_placement_strategy,
                    )
                    coverage = _top_n_expert_coverage(
                        rank_counts, top_n_experts, local_expert_ids
                    )
                    fig, axis = plt.subplots(figsize=(16, 6))
                    positions = np.arange(num_experts)
                    axis.bar(
                        positions,
                        sorted_counts,
                        color="#2F6F8F",
                        edgecolor="#202020",
                        linewidth=0.35,
                    )
                    axis.set_xticks(positions)
                    axis.set_xticklabels(
                        [str(expert_id) for expert_id in expert_ids],
                        rotation=90,
                        fontsize=max(4, min(8, 700 // num_experts)),
                    )
                    axis.set_xlabel("Expert ID (sorted by assignment count)")
                    axis.set_ylabel("Token-expert assignments")
                    axis.set_title(
                        f"Rank {rank}: all traced token-to-expert assignments"
                    )
                    if coverage:
                        coverage_text = "\n".join(
                            f"Top-{top_n}: "
                            f"{metrics['token_expert_assignments']:,} "
                            f"({metrics['assignment_share_percent']:.2f}%), "
                            f"local {metrics['local_expert_count']}/{top_n}"
                            for top_n, metrics in coverage.items()
                        )
                        axis.text(
                            0.99,
                            0.96,
                            coverage_text,
                            transform=axis.transAxes,
                            horizontalalignment="right",
                            verticalalignment="top",
                            fontsize=9,
                            bbox={
                                "facecolor": "white",
                                "edgecolor": "#777777",
                                "alpha": 0.9,
                            },
                        )
                        print(
                            f"dataset={dataset}, batch_size={batch_size}, "
                            f"layer={layer}, rank={rank}: "
                            + ", ".join(
                                f"top_{top_n}="
                                f"{metrics['token_expert_assignments']} "
                                f"({metrics['assignment_share_percent']:.2f}%), "
                                f"local_experts="
                                f"{metrics['local_expert_count']}/{top_n}"
                                for top_n, metrics in coverage.items()
                            )
                        )
                    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
                    axis.grid(axis="y", alpha=0.2)
                    fig.suptitle(
                        f"{dataset}, batch size {batch_size}, layer {layer}",
                        fontsize=14,
                    )
                    fig.tight_layout()
                    rank_output = output_dir / (
                        f"expert_counts_{dataset}_batch_{batch_size:04d}_"
                        f"layer_{layer:04d}_rank_{rank:05d}.png"
                    )
                    fig.savefig(rank_output, dpi=200, bbox_inches="tight")
                    plt.close(fig)
                    outputs.append(rank_output)
    _write_distribution_data(
        work_dir, datasets, batch_sizes, distributions, top_n_experts
    )
    return outputs


def _validate_collection_args(args: argparse.Namespace) -> None:
    if args.ep_size <= 0:
        raise ValueError("--ep-size must be positive")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if not args.batch_sizes or any(size <= 0 for size in args.batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("--batch-sizes must not contain duplicates")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("--datasets must not contain duplicates")
    if args.max_model_len <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--max-model-len and --max-new-tokens must be positive")


def collect(args: argparse.Namespace) -> None:
    _validate_collection_args(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    datasets = _load_datasets(args)
    if args.ep_size > args.num_prompts:
        raise ValueError("--ep-size cannot exceed --num-prompts")
    manifest = {
        "model": args.model,
        "datasets": list(datasets),
        "batch_sizes": args.batch_sizes,
        "expert_placement_strategy": "linear",
        "num_prompts_per_dataset": args.num_prompts,
        "experiments": [],
    }
    for dataset_name, prompts in datasets.items():
        for batch_size in args.batch_sizes:
            print(f"Collecting dataset={dataset_name}, batch_size={batch_size}")
            manifest["experiments"].append(
                _collect_experiment(args, dataset_name, prompts, batch_size)
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    outputs = _plot_distributions(
        work_dir=args.output_dir,
        output_dir=args.output_dir / "plots",
        datasets=list(datasets),
        batch_sizes=args.batch_sizes,
        requested_layers=args.layers,
        max_steps=args.max_steps,
        top_n_experts=args.top_n_experts,
    )
    print(f"Saved {len(outputs)} plots under {args.output_dir / 'plots'}")
    print(f"Saved aggregate data to {args.output_dir / 'expert_distribution.json'}")


def _solve_collected_experiment(
    args: argparse.Namespace,
    experiment_dir: Path,
    layer: int,
    output_path: Path,
) -> None:
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from benchmarks.kernels.benchmark_deepep_ht_distribution import run_solver_only

    run_solver_only(
        argparse.Namespace(
            trace_dir=experiment_dir,
            trace_layer=layer,
            trace_step=args.solver_step,
            top_k=None,
            solver_world_size=args.ep_size,
            solver_redundant_slots=args.solver_redundant_slots,
            solver_fast_capacity_tolerance=args.solver_fast_capacity_tolerance,
            solver_compute_weight=args.solver_compute_weight,
            solver_communication_weight=args.solver_communication_weight,
            solver_link_weight=args.solver_link_weight,
            solver_remote_weight=args.solver_remote_weight,
            solver_replica_weight=args.solver_replica_weight,
            solver_ultraep_beta=args.solver_ultraep_beta,
            solver_min_quota=args.solver_min_quota,
            solver_output_json=output_path,
        )
    )


def collect_solve(args: argparse.Namespace) -> None:
    _validate_collection_args(args)
    if len(set(args.solver_layers)) != len(args.solver_layers):
        raise ValueError("--solver-layers must not contain duplicates")
    if any(layer < 0 for layer in args.solver_layers):
        raise ValueError("--solver-layers must contain non-negative layer IDs")
    if args.solver_step is not None and args.solver_step < 0:
        raise ValueError("--solver-step must be non-negative")
    if args.solver_redundant_slots < 0:
        raise ValueError("--solver-redundant-slots must be non-negative")
    if not 0.0 <= args.solver_fast_capacity_tolerance <= 1.0:
        raise ValueError("--solver-fast-capacity-tolerance must be in [0, 1]")
    if args.solver_min_quota <= 0:
        raise ValueError("--solver-min-quota must be positive")
    if args.solver_ultraep_beta < 1.0:
        raise ValueError("--solver-ultraep-beta must be at least 1")
    weights = (
        args.solver_compute_weight,
        args.solver_communication_weight,
        args.solver_link_weight,
        args.solver_remote_weight,
        args.solver_replica_weight,
    )
    if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not any(
        weights
    ):
        raise ValueError(
            "Solver objective weights must be non-negative and not all zero"
        )
    datasets = _load_datasets(args)
    if args.ep_size > args.num_prompts:
        raise ValueError("--ep-size cannot exceed --num-prompts")
    output_dir = args.solver_output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_max_steps = (args.solver_step or 0) + 1
    step_label = "first" if args.solver_step is None else f"{args.solver_step:06d}"

    for dataset_name, prompts in datasets.items():
        for batch_size in args.batch_sizes:
            print(f"Collecting and solving dataset={dataset_name}, batch={batch_size}")
            with tempfile.TemporaryDirectory(prefix="vllm_moe_collect_solve_") as raw:
                collection_args = argparse.Namespace(**vars(args))
                collection_args.output_dir = Path(raw)
                collection_args.trace_max_steps = trace_max_steps
                _collect_experiment(
                    collection_args,
                    dataset_name,
                    prompts,
                    batch_size,
                )
                experiment_dir = _experiment_dir(
                    collection_args.output_dir,
                    dataset_name,
                    batch_size,
                )
                for layer in args.solver_layers:
                    output_path = output_dir / (
                        f"plan_{dataset_name}_batch_{batch_size:04d}_"
                        f"layer_{layer:04d}_step_{step_label}.json"
                    )
                    _solve_collected_experiment(
                        args,
                        experiment_dir,
                        layer,
                        output_path,
                    )


def plot(args: argparse.Namespace) -> None:
    work_dir = args.work_dir.expanduser().resolve()
    manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
    datasets = args.datasets or manifest["datasets"]
    batch_sizes = args.batch_sizes or manifest["batch_sizes"]
    output_dir = args.output_dir or work_dir / "plots"
    outputs = _plot_distributions(
        work_dir=work_dir,
        output_dir=output_dir.expanduser().resolve(),
        datasets=datasets,
        batch_sizes=batch_sizes,
        requested_layers=args.layers,
        max_steps=args.max_steps,
        top_n_experts=args.top_n_experts,
    )
    for output in outputs:
        print(f"Saved {output}")
    print(f"Saved aggregate data to {work_dir / 'expert_distribution.json'}")


def _add_plot_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Layer IDs to plot; defaults to the first common MoE layer.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Limit forward-series panels to their first N model-forward steps.",
    )
    parser.add_argument(
        "--top-n-experts",
        type=int,
        nargs="+",
        default=[],
        metavar="N",
        help=(
            "Report how many token-expert assignments are covered by the N "
            "most-loaded experts; accepts one or more N values."
        ),
    )


def _add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override a built-in dataset or define a custom local dataset.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Prompt batch sizes per EP rank to compare.",
    )
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--moe-backend", default="triton")


def _add_solver_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--solver-output-dir", type=Path, required=True)
    parser.add_argument(
        "--solver-layers",
        "--solver-layer",
        dest="solver_layers",
        type=int,
        nargs="+",
        required=True,
        help="One or more MoE layer IDs to solve from the same captured step.",
    )
    parser.add_argument(
        "--solver-step",
        type=int,
        help="Raw forward step to solve; defaults to the first captured step.",
    )
    parser.add_argument("--solver-redundant-slots", type=int, default=2)
    parser.add_argument("--solver-fast-capacity-tolerance", type=float, default=0.1)
    parser.add_argument("--solver-compute-weight", type=float, default=1.0)
    parser.add_argument("--solver-communication-weight", type=float, default=1.0)
    parser.add_argument("--solver-link-weight", type=float, default=1.0)
    parser.add_argument("--solver-remote-weight", type=float, default=0.1)
    parser.add_argument("--solver-replica-weight", type=float, default=0.01)
    parser.add_argument("--solver-ultraep-beta", type=float, default=1.01)
    parser.add_argument("--solver-min-quota", type=int, default=1024)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    _add_collection_arguments(collect_parser)
    collect_parser.add_argument("--output-dir", type=Path, required=True)
    _add_plot_filters(collect_parser)
    collect_parser.set_defaults(func=collect)

    collect_solve_parser = subparsers.add_parser("collect-solve")
    _add_collection_arguments(collect_solve_parser)
    _add_solver_arguments(collect_solve_parser)
    collect_solve_parser.set_defaults(func=collect_solve)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--work-dir", type=Path, required=True)
    plot_parser.add_argument("--output-dir", type=Path)
    plot_parser.add_argument("--datasets", nargs="+")
    plot_parser.add_argument("--batch-sizes", type=int, nargs="+")
    _add_plot_filters(plot_parser)
    plot_parser.set_defaults(func=plot)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
