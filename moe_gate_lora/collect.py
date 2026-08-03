from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from moe_gate_lora.trainer import StreamingProcessor


@dataclass(frozen=True)
class CollectionConfig:
    model: str
    prompts: Path | None
    output_dir: Path
    mode: Literal["train", "eval", "trace"]
    lora_dir: Path | None
    epochs: int = 3
    ep_size: int = 1
    max_model_len: int = 4096
    max_num_batched_tokens: int | None = None
    max_new_tokens: int = 16
    collect_batch_size: int = 1
    timeout: int = 1800
    load_format: str = "auto"
    moe_backend: str = "auto"
    rank_dim: int = 8
    alpha: float = 16.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    device: str | None = None


DEFAULT_PROMPTS = [
    "Explain why the sky is blue.",
    "Write a short Python merge-sort function.",
    "Summarize the causes of the French Revolution.",
    "What is the difference between TCP and UDP?",
]


def _read_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list) or not all(
            isinstance(item, str) for item in payload
        ):
            raise ValueError("Prompt JSON must be a list of strings")
        prompts = payload
    else:
        prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if not prompts:
        raise ValueError("No prompts were loaded")
    return prompts


def _num_experts(hf_config) -> int:
    for name in ("num_experts", "n_routed_experts", "num_local_experts"):
        value = getattr(hf_config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not determine the number of experts")


def _ready_path(
    sync_dir: Path,
    epoch: int,
    batch: int,
    rank: int,
) -> Path:
    return sync_dir / (f"epoch_{epoch:04d}_batch_{batch:06d}_rank_{rank:05d}.ready")


def _epoch_done_path(sync_dir: Path, epoch: int, rank: int) -> Path:
    return sync_dir / f"epoch_{epoch:04d}_rank_{rank:05d}.done"


def _signal_and_wait(path: Path, timeout: int) -> None:
    ack = path.with_suffix(".ack")
    path.write_text("ready", encoding="utf-8")
    deadline = time.monotonic() + timeout
    while not ack.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {ack}")
        time.sleep(0.05)
    path.unlink()
    ack.unlink()


def _worker(
    config: CollectionConfig,
    indexed_prompts: list[tuple[int, str]],
    rank: int,
    master_port: int,
    activation_dir: Path,
    sync_dir: Path,
    result_dir: Path,
) -> None:
    num_batches = (
        len(indexed_prompts) + config.collect_batch_size - 1
    ) // config.collect_batch_size
    os.environ.update(
        {
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": str(config.ep_size),
            "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(master_port),
            "VLLM_MOE_TRACE_DIR": str(activation_dir),
            "VLLM_MOE_TRACE_MODE": "lora_training",
            "VLLM_MOE_TRACE_MAX_STEPS": str(
                config.epochs
                * num_batches
                * (config.max_model_len + config.max_new_tokens)
            ),
            # Collection and external training never alter the real MoE route.
            "VLLM_SC_EPLB": "0",
        }
    )
    os.environ.pop("VLLM_SC_EPLB_LORA_DIR", None)

    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, object] = {
        "model": config.model,
        "tensor_parallel_size": 1,
        "enable_expert_parallel": True,
        "max_model_len": config.max_model_len,
        "max_num_seqs": min(len(indexed_prompts), config.collect_batch_size),
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_return_routed_experts": False,
        "moe_backend": config.moe_backend,
        "load_format": config.load_format,
    }
    if config.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = config.max_num_batched_tokens
    llm = LLM(
        **llm_kwargs,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=config.max_new_tokens)
    generations = []
    for epoch in range(config.epochs):
        for batch_index, start in enumerate(
            range(0, len(indexed_prompts), config.collect_batch_size)
        ):
            batch = indexed_prompts[start : start + config.collect_batch_size]
            outputs = llm.generate(
                [prompt for _, prompt in batch], sampling_params, use_tqdm=True
            )
            if epoch == 0:
                for (sample_id, prompt), output in zip(batch, outputs):
                    generations.append(
                        {
                            "sample_id": sample_id,
                            "rank": rank,
                            "epoch": epoch,
                            "prompt": prompt,
                            "prompt_tokens": len(output.prompt_token_ids),
                            "generated_text": output.outputs[0].text,
                        }
                    )

            ready = _ready_path(sync_dir, epoch, batch_index, rank)
            _signal_and_wait(ready, config.timeout)
        _signal_and_wait(_epoch_done_path(sync_dir, epoch, rank), config.timeout)

    result = {
        "rank": rank,
        "num_experts": _num_experts(llm.model_config.hf_text_config),
        "generations": generations,
    }
    (result_dir / f"rank_{rank:05d}.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )


def _wait_for_batch(
    paths: list[Path], processes: list[mp.Process], timeout: int
) -> None:
    deadline = time.monotonic() + timeout
    while not all(path.exists() for path in paths):
        failed = [
            process.exitcode
            for process in processes
            if process.exitcode not in (None, 0)
        ]
        if failed:
            raise RuntimeError(f"Collection worker exited with code {failed[0]}")
        if time.monotonic() >= deadline:
            missing = [str(path) for path in paths if not path.exists()]
            raise TimeoutError(f"Timed out waiting for trace batch: {missing}")
        time.sleep(0.05)


def _archive_trace_batch(
    record_paths: list[Path],
    trace_dir: Path,
    *,
    epoch: int,
    batch: int,
) -> list[Path]:
    batch_dir = trace_dir / f"epoch_{epoch:04d}" / f"batch_{batch:06d}"
    destinations = [batch_dir / path.parent.name / path.name for path in record_paths]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Trace batch contains duplicate destination paths")
    existing = next((path for path in destinations if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"Trace destination already exists: {existing}")

    archived = []
    for path, destination in zip(record_paths, destinations):
        destination.parent.mkdir(parents=True, exist_ok=True)
        archived.append(path.replace(destination))
    return archived


def _copy_trace_metadata(activation_dir: Path, trace_dir: Path) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        activation_dir / "trace_config.json",
        trace_dir / "trace_config.json",
    )
    for rank_dir in sorted(activation_dir.glob("rank_*")):
        source = rank_dir / "metadata.json"
        if not source.exists():
            continue
        destination = trace_dir / rank_dir.name / "metadata.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def collect(config: CollectionConfig) -> dict:
    if config.ep_size <= 0 or config.collect_batch_size <= 0 or config.epochs <= 0:
        raise ValueError("ep_size, collect_batch_size, and epochs must be positive")
    if config.max_num_batched_tokens is not None and config.max_num_batched_tokens <= 0:
        raise ValueError("max_num_batched_tokens must be positive")
    if config.mode != "trace" and config.lora_dir is None:
        raise ValueError("lora_dir is required for train and eval modes")
    output_dir = config.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    activation_dir = output_dir / "activations"
    trace_dir = output_dir / "traces"
    sync_dir = output_dir / "sync"
    result_dir = output_dir / "results"
    for directory in (activation_dir, sync_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (activation_dir / "trace_config.json").write_text(
        json.dumps({"capture_next_gate_base_logits": True}), encoding="utf-8"
    )

    prompts = _read_prompts(config.prompts)
    if config.ep_size > len(prompts):
        raise ValueError("ep_size cannot exceed the number of prompts")
    indexed = list(enumerate(prompts))
    floor, remainder = divmod(len(indexed), config.ep_size)

    def shard_start(rank: int) -> int:
        return rank * floor + min(rank, remainder)

    shards = [
        indexed[shard_start(rank) : shard_start(rank + 1)]
        for rank in range(config.ep_size)
    ]
    batch_counts = [
        (len(shard) + config.collect_batch_size - 1) // config.collect_batch_size
        for shard in shards
    ]

    from vllm.utils.network_utils import get_open_port

    master_port = get_open_port()
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_worker,
            args=(
                config,
                shards[rank],
                rank,
                master_port,
                activation_dir,
                sync_dir,
                result_dir,
            ),
        )
        for rank in range(config.ep_size)
    ]
    for process in processes:
        process.start()

    metrics_dir = output_dir / "metrics"
    processor = None
    mode = config.mode
    if mode != "trace":
        assert config.lora_dir is not None
        processor = StreamingProcessor(
            mode=mode,
            output_dir=metrics_dir,
            lora_dir=config.lora_dir,
            rank_dim=config.rank_dim,
            alpha=config.alpha,
            lr=config.lr,
            weight_decay=config.weight_decay,
            seed=config.seed,
            device=config.device,
        )
    num_batches = 0
    try:
        for epoch in range(config.epochs):
            for batch_index in range(max(batch_counts)):
                active = [
                    rank
                    for rank, count in enumerate(batch_counts)
                    if batch_index < count
                ]
                ready = [
                    _ready_path(sync_dir, epoch, batch_index, rank) for rank in active
                ]
                _wait_for_batch(ready, processes, config.timeout)
                record_paths = []
                for rank in active:
                    record_paths.extend(
                        sorted(
                            (activation_dir / f"rank_{rank:05d}").glob(
                                "step_*_layer_*.pt"
                            )
                        )
                    )
                if not record_paths:
                    raise RuntimeError(
                        f"Epoch {epoch} batch {batch_index} produced no trace records"
                    )
                if processor is None:
                    _archive_trace_batch(
                        record_paths,
                        trace_dir,
                        epoch=epoch,
                        batch=batch_index,
                    )
                else:
                    processor.process(record_paths, epoch=epoch)
                num_batches += 1
                for path in ready:
                    path.with_suffix(".ack").write_text("processed", encoding="utf-8")
            epoch_done = [
                _epoch_done_path(sync_dir, epoch, rank)
                for rank in range(config.ep_size)
            ]
            _wait_for_batch(epoch_done, processes, config.timeout)
            if processor is not None:
                processor.finish_epoch(epoch)
            for path in epoch_done:
                path.with_suffix(".ack").write_text("epoch processed", encoding="utf-8")

        for process in processes:
            process.join(timeout=config.timeout)
            if process.exitcode is None:
                process.kill()
                raise TimeoutError(f"Worker {process.pid} did not exit")
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

    rank_results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(result_dir.glob("rank_*.json"))
    ]
    num_experts = {int(result["num_experts"]) for result in rank_results}
    if len(num_experts) != 1:
        raise ValueError("Ranks reported inconsistent expert counts")
    generations = sorted(
        [item for result in rank_results for item in result["generations"]],
        key=lambda item: item["sample_id"],
    )
    summary = {
        "model": config.model,
        "mode": config.mode,
        "num_experts": num_experts.pop(),
        "num_prompts": len(prompts),
        "epochs": config.epochs,
        "batches_per_epoch": max(batch_counts),
        "num_batches": num_batches,
    }
    if processor is None:
        _copy_trace_metadata(activation_dir, trace_dir)
        summary["trace_dir"] = str(trace_dir)
    else:
        summary["lora_dir"] = str(config.lora_dir)
        summary["metrics_dir"] = str(metrics_dir)
    (output_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "generations.json").write_text(
        json.dumps(generations, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
