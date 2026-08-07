# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay traced MoE distributions through local bypass and expert kernels.

Run this module with one process per expert-parallel rank::

    .venv/bin/python -m torch.distributed.run --standalone \
        --nproc-per-node=4 --module \
        benchmarks.kernels.benchmark_moe_trace_replay \
        --work-dir /tmp/qwen3_expert_distribution \
        --dataset math --batch-size 4 --layer 23 \
        --extra-replicas 0 8

The benchmark uses synthetic activations and weights with the traced token and
expert shapes. Count-only traces require reconstructing token top-k rows, so
expert assignment counts are exact while local-token and remote-transfer
coalescing are an approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from benchmarks.kernels.benchmark_deepep_ht_distribution import (
    MODEL_CONFIGS,
    make_base_tensors,
    make_kernels,
    make_vllm_config,
    run_local_bypass_iter,
)
from examples.basic.offline_inference.moe_trace_expert_distribution import (
    _aggregate_trace,
    _experiment_dir,
)
from examples.basic.offline_inference.moe_trace_replica_simulation import (
    LatencyModel,
    _base_replicas,
    _copy_replicas,
    _demand_tensor,
    _evaluate_token_trace,
    _load_token_trace,
    _physical_expert_layout,
    _route_targets_to_physical_topk_ids,
    _route_token_step,
)
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
    DeepEPHTPrepareAndFinalize,
)
from vllm.v1.worker.workspace import init_workspace_manager

POLICIES = ("baseline", "communication_first", "balance_first", "joint")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--simulation-json", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
    )
    parser.add_argument(
        "--extra-replicas",
        type=int,
        nargs="+",
        help="Only replay these replica budgets; baseline is always retained.",
    )
    parser.add_argument(
        "--communication-weights",
        type=float,
        nargs="+",
        help="Only replay joint points with these communication weights.",
    )
    parser.add_argument(
        "--pareto-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replay only points marked Pareto-optimal by the offline simulation.",
    )
    parser.add_argument("--model")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--trim-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stage-barrier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize ranks before measuring dispatch, compute, and combine.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("local_bypass",),
        default="local_bypass",
        help="Trace replay always uses the local-bypass execution path.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--plot-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Read an existing operator timing JSON and redraw the best measured "
            "expert load without launching distributed GPU replay."
        ),
    )
    return parser.parse_args(argv)


def _find_experiment(
    payload: dict[str, Any],
    dataset: str,
    batch_size: int,
    layer_id: int,
) -> dict[str, Any]:
    matches = [
        experiment
        for experiment in payload["experiments"]
        if experiment["dataset"] == dataset
        and int(experiment["batch_size_per_rank"]) == batch_size
        and int(experiment["layer_id"]) == layer_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one simulation experiment for "
            f"dataset={dataset}, batch_size={batch_size}, layer={layer_id}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _select_points(
    experiment: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    requested_budgets = (
        set(args.extra_replicas) if args.extra_replicas is not None else None
    )
    requested_weights = (
        set(args.communication_weights)
        if args.communication_weights is not None
        else None
    )
    points = []
    for point in experiment["points"]:
        policy = point["policy"]
        if policy not in args.policies:
            continue
        if args.pareto_only and not point["pareto_optimal"]:
            continue
        if (
            policy != "baseline"
            and requested_budgets is not None
            and int(point.get("requested_extra_replicas", point["used_extra_replicas"]))
            not in requested_budgets
        ):
            continue
        if (
            policy == "joint"
            and requested_weights is not None
            and float(point["communication_weight"]) not in requested_weights
        ):
            continue
        points.append(point)
    if not points:
        raise ValueError("Point filters did not select any simulation result")
    return points


def _point_label(point: dict[str, Any]) -> str:
    requested = int(point.get("requested_extra_replicas", point["used_extra_replicas"]))
    used = int(point["used_extra_replicas"])
    label = f"{point['policy']}:B={requested}"
    if used != requested:
        label += f":used={used}"
    if point["communication_weight"] is not None:
        label += f":w={point['communication_weight']:g}"
    return label


def _point_replicas(
    base_replicas: list[set[int]],
    point: dict[str, Any],
) -> list[set[int]]:
    replicas = _copy_replicas(base_replicas)
    for placement in point["replica_placements"]:
        expert_id = int(placement["expert_id"])
        rank = int(placement["rank"])
        if rank in replicas[expert_id]:
            raise ValueError(
                f"Point {_point_label(point)} duplicates expert {expert_id} "
                f"on rank {rank}"
            )
        replicas[expert_id].add(rank)
    return replicas


def _routing_weights(point: dict[str, Any]) -> tuple[float, float]:
    if point["policy"] == "communication_first":
        return 0.0, 1.0
    if point["policy"] == "balance_first":
        return 1.0, 0.0
    if point["policy"] == "joint":
        return 1.0, float(point["communication_weight"])
    return 1.0, 1.0


def _latency_config_value(
    config: dict[str, Any],
    token_name: str,
    legacy_name: str,
) -> float:
    value = config.get(token_name, config.get(legacy_name))
    if value is None:
        raise ValueError(f"Simulation latency model is missing {token_name!r}")
    return float(value)


def _resolve_model_shape(
    args: argparse.Namespace,
    simulation: dict[str, Any],
) -> dict[str, Any]:
    model = args.model or simulation.get("model")
    preset = MODEL_CONFIGS.get(model, {})
    local_config = _load_local_model_shape(model)

    def resolve(name: str) -> int:
        value = getattr(args, name)
        if value is None:
            value = local_config.get(name)
        if value is None:
            value = preset.get(name)
        if value is None:
            option = name.replace("_", "-")
            raise ValueError(
                f"Could not resolve {name} for model {model!r}; pass --{option}"
            )
        if int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
        return int(value)

    return {
        "model": model,
        "hidden_size": resolve("hidden_size"),
        "intermediate_size": resolve("intermediate_size"),
        "top_k": resolve("top_k"),
        "config_num_experts": local_config.get("num_experts"),
    }


def _load_local_model_shape(model: str | None) -> dict[str, int]:
    if model is None:
        return {}
    model_path = Path(model).expanduser()
    config_path = model_path / "config.json" if model_path.is_dir() else model_path
    if not config_path.is_file() or config_path.name != "config.json":
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sections = [
        section
        for section in (
            config.get("text_config"),
            config.get("llm_config"),
            config,
        )
        if isinstance(section, dict)
    ]

    def first(names: tuple[str, ...]) -> int | None:
        for section in sections:
            for name in names:
                value = section.get(name)
                if value is not None:
                    return int(value)
        return None

    aliases = {
        "hidden_size": ("hidden_size", "d_model", "n_embd"),
        "intermediate_size": (
            "moe_intermediate_size",
            "expert_intermediate_size",
            "intermediate_size",
        ),
        "top_k": (
            "num_experts_per_tok",
            "num_experts_per_token",
            "moe_top_k",
            "top_k",
        ),
        "num_experts": (
            "num_experts",
            "n_routed_experts",
            "num_local_experts",
        ),
    }
    return {
        name: value
        for name, field_aliases in aliases.items()
        if (value := first(field_aliases)) is not None
    }


def _make_local_replay_plans(
    points: list[dict[str, Any]],
    token_steps: list[list[np.ndarray]],
    base_replicas: list[set[int]],
    latency_model: LatencyModel,
    rank: int,
) -> tuple[list[dict[str, Any]], int]:
    ep_size = len(token_steps[0])
    point_replicas = [_point_replicas(base_replicas, point) for point in points]
    capacities = [
        _physical_expert_layout(replicas, ep_size)[1] for replicas in point_replicas
    ]
    capacity_per_rank = max(capacities)
    plans = []
    for point, replicas in zip(points, point_replicas):
        physical_layout, _ = _physical_expert_layout(
            replicas, ep_size, capacity_per_rank
        )
        compute_weight, communication_weight = _routing_weights(point)
        physical_topk_ids = []
        expert_rank_loads = np.zeros((len(base_replicas), ep_size), dtype=np.int64)
        rank_assignment_loads = np.zeros(ep_size, dtype=np.int64)
        rank_token_loads = np.zeros(ep_size, dtype=np.int64)
        local_tokens = 0
        remote_tokens = 0
        remote_token_transfers = 0
        for step in token_steps:
            routing = _route_token_step(
                step,
                replicas,
                latency_model,
                compute_weight=compute_weight,
                communication_weight=communication_weight,
            )
            expert_rank_loads += routing.expert_rank_loads
            rank_assignment_loads += routing.rank_assignment_loads
            rank_token_loads += routing.rank_token_loads
            local_tokens += routing.local_tokens
            remote_tokens += routing.remote_tokens
            remote_token_transfers += routing.remote_token_transfers
            physical_topk_ids.append(
                _route_targets_to_physical_topk_ids(
                    step[rank],
                    routing.target_ranks_by_source[rank],
                    physical_layout,
                )
            )
        plans.append(
            {
                "point": point,
                "replicas": replicas,
                "physical_topk_ids": physical_topk_ids,
                "expert_rank_loads": expert_rank_loads,
                "rank_assignment_loads": rank_assignment_loads,
                "rank_token_loads": rank_token_loads,
                "local_tokens": local_tokens,
                "remote_tokens": remote_tokens,
                "remote_token_transfers": remote_token_transfers,
            }
        )
    return plans, capacity_per_rank


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize_rows(
    rows: list[dict[str, Any]],
    iterations: int,
    num_steps: int,
    world_size: int,
    trim_ratio: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_forward: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_forward[(row["replay_iteration"], row["trace_step_index"])].append(row)

    forward_rows = []
    for (iteration, step_index), rank_rows in sorted(by_forward.items()):
        if len(rank_rows) != world_size:
            raise RuntimeError(
                f"Replay iteration {iteration}, step {step_index} has "
                f"{len(rank_rows)} rank rows, expected {world_size}"
            )
        received = [int(row["received_tokens"]) for row in rank_rows]
        mean_received = statistics.mean(received)
        dispatch_ms = max(float(row["dispatch_ms"]) for row in rank_rows)
        compute_ms = max(float(row["expert_compute_ms"]) for row in rank_rows)
        combine_ms = max(float(row["combine_ms"]) for row in rank_rows)
        forward_rows.append(
            {
                "replay_iteration": iteration,
                "trace_step_index": step_index,
                "raw_trace_step": rank_rows[0]["raw_trace_step"],
                "dispatch_ms": dispatch_ms,
                "compute_ms": compute_ms,
                "combine_ms": combine_ms,
                "communication_ms": dispatch_ms + combine_ms,
                "serial_ms": dispatch_ms + compute_ms + combine_ms,
                "remote_assignments": sum(
                    int(row["remote_assignments"]) for row in rank_rows
                ),
                "remote_unique_tokens": sum(
                    int(row["remote_unique_tokens"]) for row in rank_rows
                ),
                "local_path_tokens": sum(
                    int(row.get("local_path_tokens", 0)) for row in rank_rows
                ),
                "deepep_source_tokens": sum(
                    int(row.get("deepep_source_tokens", 0)) for row in rank_rows
                ),
                "active_destinations": sum(
                    int(row.get("active_destinations", 0)) for row in rank_rows
                ),
                "received_tokens_min": min(received),
                "received_tokens_max": max(received),
                "rank_max_over_mean": (
                    max(received) / mean_received if mean_received else 0.0
                ),
            }
        )

    by_iteration: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in forward_rows:
        by_iteration[int(row["replay_iteration"])].append(row)
    replay_rows = []
    for iteration, step_rows in sorted(by_iteration.items()):
        if len(step_rows) != num_steps:
            raise RuntimeError(
                f"Replay iteration {iteration} has {len(step_rows)} steps, "
                f"expected {num_steps}"
            )
        replay_rows.append(
            {
                "replay_iteration": iteration,
                "dispatch_ms": sum(row["dispatch_ms"] for row in step_rows),
                "compute_ms": sum(row["compute_ms"] for row in step_rows),
                "combine_ms": sum(row["combine_ms"] for row in step_rows),
                "communication_ms": sum(row["communication_ms"] for row in step_rows),
                "serial_ms": sum(row["serial_ms"] for row in step_rows),
            }
        )
    if len(replay_rows) != iterations:
        raise RuntimeError(
            f"Collected {len(replay_rows)} replay iterations, expected {iterations}"
        )
    trim_count = int(len(replay_rows) * trim_ratio)
    kept = sorted(replay_rows, key=lambda row: row["serial_ms"])
    if trim_count:
        kept = kept[trim_count:-trim_count]
    if not kept:
        raise ValueError("--trim-ratio removed every replay iteration")

    all_forward_serial = [float(row["serial_ms"]) for row in forward_rows]
    summary = {
        "measured_replay_iterations": iterations,
        "trimmed_replay_iterations": len(kept),
        "num_trace_steps": num_steps,
        "dispatch_ms": statistics.mean(row["dispatch_ms"] for row in kept),
        "compute_ms": statistics.mean(row["compute_ms"] for row in kept),
        "combine_ms": statistics.mean(row["combine_ms"] for row in kept),
        "communication_ms": statistics.mean(row["communication_ms"] for row in kept),
        "serial_ms": statistics.mean(row["serial_ms"] for row in kept),
        "serial_ms_p50": _percentile([float(row["serial_ms"]) for row in kept], 50),
        "serial_ms_p95": _percentile([float(row["serial_ms"]) for row in kept], 95),
        "forward_serial_ms_p50": _percentile(all_forward_serial, 50),
        "forward_serial_ms_p95": _percentile(all_forward_serial, 95),
        "mean_step_max_over_mean": statistics.mean(
            row["rank_max_over_mean"] for row in forward_rows
        ),
        "remote_assignments": int(forward_rows[0]["remote_assignments"])
        if num_steps == 1
        else sum(
            int(row["remote_assignments"])
            for row in forward_rows
            if row["replay_iteration"] == 0
        ),
        "remote_unique_tokens": int(forward_rows[0]["remote_unique_tokens"])
        if num_steps == 1
        else sum(
            int(row["remote_unique_tokens"])
            for row in forward_rows
            if row["replay_iteration"] == 0
        ),
        "local_path_tokens": int(forward_rows[0]["local_path_tokens"])
        if num_steps == 1
        else sum(
            int(row["local_path_tokens"])
            for row in forward_rows
            if row["replay_iteration"] == 0
        ),
        "deepep_source_tokens": int(forward_rows[0]["deepep_source_tokens"])
        if num_steps == 1
        else sum(
            int(row["deepep_source_tokens"])
            for row in forward_rows
            if row["replay_iteration"] == 0
        ),
        "active_destinations": int(forward_rows[0]["active_destinations"])
        if num_steps == 1
        else sum(
            int(row["active_destinations"])
            for row in forward_rows
            if row["replay_iteration"] == 0
        ),
    }
    return summary, forward_rows


def _mark_measured_pareto(results: list[dict[str, Any]]) -> None:
    for result in results:
        summary = result["timing"]
        result["measured_pareto_optimal"] = not any(
            other is not result
            and other["timing"]["compute_ms"] <= summary["compute_ms"]
            and other["timing"]["communication_ms"] <= summary["communication_ms"]
            and (
                other["timing"]["compute_ms"] < summary["compute_ms"]
                or other["timing"]["communication_ms"] < summary["communication_ms"]
            )
            for other in results
        )


def _best_measured_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Operator timing JSON contains no measured results")
    return min(
        results,
        key=lambda result: (
            float(result["timing"]["serial_ms"]),
            float(result["timing"]["communication_ms"]),
            float(result["timing"]["compute_ms"]),
            int(result["used_extra_replicas"]),
            str(result["label"]),
        ),
    )


def _assignment_profile(
    expert_rank_loads: np.ndarray,
    replicas: list[set[int]],
    rank_unique_token_loads: np.ndarray,
    *,
    local_tokens: int | None = None,
    remote_tokens: int | None = None,
    remote_token_transfers: int | None = None,
) -> dict[str, Any]:
    loads = np.asarray(expert_rank_loads, dtype=np.int64)
    if loads.ndim != 2 or loads.shape[0] != len(replicas):
        raise ValueError(
            "expert_rank_loads must have shape [logical_expert, target_rank]"
        )
    if np.any(loads < 0):
        raise ValueError("expert_rank_loads must be non-negative")
    token_loads = np.asarray(rank_unique_token_loads, dtype=np.int64)
    if token_loads.ndim != 1 or token_loads.shape[0] != loads.shape[1]:
        raise ValueError("rank_unique_token_loads must match the target-rank dimension")
    if np.any(token_loads < 0):
        raise ValueError("rank_unique_token_loads must be non-negative")
    rank_assignment_loads = loads.sum(axis=0)
    profile = {
        "load_unit": "token_expert_assignments",
        "expert_rank_assignment_loads": loads.tolist(),
        "expert_assignment_loads": loads.sum(axis=1).tolist(),
        "rank_assignment_loads": rank_assignment_loads.tolist(),
        "rank_assignment_load_unit": "expert_input_tokens_per_target_rank",
        "rank_unique_token_loads": token_loads.tolist(),
        "rank_token_loads": token_loads.tolist(),
        "rank_token_load_unit": "unique_tokens_per_target_rank",
        "replica_ranks_by_expert": [sorted(ranks) for ranks in replicas],
    }
    if local_tokens is not None:
        profile["local_tokens"] = int(local_tokens)
    if remote_tokens is not None:
        profile["remote_tokens"] = int(remote_tokens)
    if remote_token_transfers is not None:
        profile["remote_token_transfers"] = int(remote_token_transfers)
    if remote_tokens is not None and remote_token_transfers is not None:
        profile["communication_units"] = int(remote_token_transfers)
    return profile


def _best_load_paths(output_json: Path) -> tuple[Path, Path, Path]:
    prefix = output_json.with_name(f"{output_json.stem}_best_expert_load")
    return (
        Path(f"{prefix}.json"),
        Path(f"{prefix}.csv"),
        Path(f"{prefix}.png"),
    )


def _plot_best_load(
    result: dict[str, Any],
    profile: dict[str, Any],
    output: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    loads = np.asarray(profile["expert_rank_assignment_loads"], dtype=np.int64)
    totals = loads.sum(axis=1)
    rank_assignment_loads = np.asarray(profile["rank_assignment_loads"], dtype=np.int64)
    rank_unique_token_loads = np.asarray(
        profile.get("rank_unique_token_loads", profile.get("rank_token_loads", [])),
        dtype=np.int64,
    )
    expert_ids = np.arange(len(totals))
    order = np.lexsort((expert_ids, -totals))
    sorted_loads = loads[order]
    positions = np.arange(len(order))
    extra_experts = {
        int(placement["expert_id"]) for placement in result["replica_placements"]
    }

    width = min(24.0, max(12.0, len(order) * 0.14))
    fig, (expert_axis, rank_axis) = plt.subplots(
        2,
        1,
        figsize=(width, 8.5),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.38},
    )
    colors = plt.get_cmap("tab20", loads.shape[1])
    bottom = np.zeros(len(order), dtype=np.int64)
    for rank in range(loads.shape[1]):
        expert_axis.bar(
            positions,
            sorted_loads[:, rank],
            width=0.86,
            bottom=bottom,
            color=colors(rank),
            label=f"rank {rank}",
            linewidth=0,
        )
        bottom += sorted_loads[:, rank]

    extra_positions = [
        index for index, expert_id in enumerate(order) if expert_id in extra_experts
    ]
    if extra_positions:
        marker_height = max(int(totals.max(initial=0)), 1) * 1.025
        expert_axis.scatter(
            extra_positions,
            [marker_height] * len(extra_positions),
            color="#202020",
            marker="*",
            s=24,
            zorder=3,
        )
        expert_axis.set_ylim(top=marker_height * 1.06)

    tick_count = min(len(order), 32)
    tick_indices = np.unique(np.linspace(0, len(order) - 1, tick_count, dtype=np.int64))
    expert_axis.set_xticks(tick_indices)
    expert_axis.set_xticklabels(
        [f"E{int(order[index])}" for index in tick_indices],
        rotation=60,
        ha="right",
    )
    expert_axis.set_ylabel("Token-expert assignments")
    expert_axis.set_xlabel("Logical experts sorted by total load")
    expert_axis.set_title(
        f"{title}\nBest: {result['label']}, "
        f"serial={float(result['timing']['serial_ms']):.3f} ms"
    )
    expert_axis.grid(axis="y", alpha=0.2)
    handles, labels = expert_axis.get_legend_handles_labels()
    if extra_positions:
        handles.append(
            Line2D(
                [],
                [],
                color="#202020",
                marker="*",
                linestyle="None",
                label="expert with extra replica",
            )
        )
        labels.append("expert with extra replica")
    expert_axis.legend(
        handles,
        labels,
        ncol=min(8, loads.shape[1] + int(bool(extra_positions))),
        fontsize=8,
    )

    ranks = np.arange(len(rank_assignment_loads))
    bar_width = 0.38
    rank_axis.bar(
        ranks - bar_width / 2,
        rank_assignment_loads,
        width=bar_width,
        color=[colors(rank) for rank in ranks],
        label="expert input tokens (compute load)",
    )
    rank_axis.bar(
        ranks + bar_width / 2,
        rank_unique_token_loads,
        width=bar_width,
        color="none",
        edgecolor=[colors(rank) for rank in ranks],
        hatch="//",
        linewidth=1.0,
        label="unique target-rank tokens",
    )
    mean_rank_load = (
        float(rank_assignment_loads.mean()) if len(rank_assignment_loads) else 0.0
    )
    rank_axis.axhline(
        mean_rank_load,
        color="#202020",
        linewidth=1.0,
        linestyle="--",
        label=f"ideal compute mean {mean_rank_load:.1f}",
    )
    rank_axis.set_xticks(ranks)
    rank_axis.set_xticklabels([f"rank {rank}" for rank in ranks])
    rank_axis.set_ylabel("Token count")
    rank_axis.set_title("Aggregate target-rank load")
    rank_axis.grid(axis="y", alpha=0.2)
    rank_axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_best_load_outputs(
    result: dict[str, Any],
    profile: dict[str, Any],
    output_json: Path,
    title: str,
) -> tuple[Path, Path, Path]:
    load_json, load_csv, load_plot = _best_load_paths(output_json)
    loads = np.asarray(profile["expert_rank_assignment_loads"], dtype=np.int64)
    totals = loads.sum(axis=1)
    order = np.lexsort((np.arange(len(totals)), -totals))
    extra_by_expert: dict[int, list[int]] = defaultdict(list)
    for placement in result["replica_placements"]:
        extra_by_expert[int(placement["expert_id"])].append(int(placement["rank"]))

    load_payload = {
        "selection_metric": "minimum_measured_serial_ms",
        "metric_definitions": {
            "rank_assignment_loads": (
                "token-expert assignments executed by each target rank"
            ),
            "rank_unique_token_loads": (
                "unique source tokens routed to each target rank"
            ),
        },
        "best_configuration": {
            "label": result["label"],
            "policy": result["policy"],
            "communication_weight": result["communication_weight"],
            "requested_extra_replicas": result.get(
                "requested_extra_replicas", result["used_extra_replicas"]
            ),
            "used_extra_replicas": result["used_extra_replicas"],
            "replica_placements": result["replica_placements"],
            "timing": result["timing"],
        },
        "expert_order_descending_load": order.tolist(),
        "assignment_profile": profile,
    }
    load_json.write_text(json.dumps(load_payload, indent=2), encoding="utf-8")

    fieldnames = [
        "load_order",
        "expert_id",
        "total_assignments",
        *(f"rank_{rank}_assignments" for rank in range(loads.shape[1])),
        "replica_ranks",
        "extra_replica_ranks",
    ]
    with load_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for load_order, expert_id in enumerate(order, start=1):
            writer.writerow(
                {
                    "load_order": load_order,
                    "expert_id": int(expert_id),
                    "total_assignments": int(totals[expert_id]),
                    **{
                        f"rank_{rank}_assignments": int(loads[expert_id, rank])
                        for rank in range(loads.shape[1])
                    },
                    "replica_ranks": json.dumps(
                        profile["replica_ranks_by_expert"][expert_id]
                    ),
                    "extra_replica_ranks": json.dumps(
                        sorted(extra_by_expert[int(expert_id)])
                    ),
                }
            )
    _plot_best_load(result, profile, load_plot, title)
    return load_json, load_csv, load_plot


def _plot_results(results: list[dict[str, Any]], output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "baseline": "#202020",
        "communication_first": "#2F6F8F",
        "balance_first": "#B54A35",
        "joint": "#568259",
    }
    markers = {
        "baseline": "X",
        "communication_first": "o",
        "balance_first": "s",
        "joint": "^",
    }
    fig, axis = plt.subplots(figsize=(9, 6.5))
    for policy in POLICIES:
        policy_results = [result for result in results if result["policy"] == policy]
        if not policy_results:
            continue
        axis.scatter(
            [result["timing"]["communication_ms"] for result in policy_results],
            [result["timing"]["compute_ms"] for result in policy_results],
            color=colors[policy],
            marker=markers[policy],
            label=policy,
            s=46,
        )
    pareto = sorted(
        (result for result in results if result["measured_pareto_optimal"]),
        key=lambda result: result["timing"]["communication_ms"],
    )
    if pareto:
        axis.plot(
            [result["timing"]["communication_ms"] for result in pareto],
            [result["timing"]["compute_ms"] for result in pareto],
            color="#202020",
            linewidth=1.0,
            linestyle="--",
            label="Measured Pareto frontier",
        )
    for result in results:
        if result["policy"] == "joint" and not result["measured_pareto_optimal"]:
            continue
        axis.annotate(
            result["label"],
            (result["timing"]["communication_ms"], result["timing"]["compute_ms"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Measured dispatch + combine time (ms)")
    axis.set_ylabel("Measured expert compute time (ms)")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_csv(results: list[dict[str, Any]], output: Path) -> None:
    rows = []
    for result in results:
        profile = result.get("assignment_profile", {})
        rows.append(
            {
                "label": result["label"],
                "policy": result["policy"],
                "communication_weight": result["communication_weight"],
                "requested_extra_replicas": result.get(
                    "requested_extra_replicas", result["used_extra_replicas"]
                ),
                "used_extra_replicas": result["used_extra_replicas"],
                "measured_pareto_optimal": result["measured_pareto_optimal"],
                **result["timing"],
                "local_tokens": profile.get("local_tokens"),
                "remote_tokens": profile.get("remote_tokens"),
                "remote_token_transfers": profile.get("remote_token_transfers"),
                "rank_assignment_loads": json.dumps(
                    profile.get("rank_assignment_loads", [])
                ),
                "rank_unique_token_loads": json.dumps(
                    profile.get(
                        "rank_unique_token_loads", profile.get("rank_token_loads", [])
                    )
                ),
                "replica_placements": json.dumps(result["replica_placements"]),
            }
        )
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _default_output_path(args: argparse.Namespace) -> Path:
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.dataset)
    return (
        args.work_dir
        / "replica_simulation"
        / (
            f"operator_timing_{safe_dataset}_batch_{args.batch_size:04d}_"
            f"layer_{args.layer:04d}_{args.execution_mode}.json"
        )
    )


def _plot_existing_best_load(args: argparse.Namespace) -> None:
    work_dir = args.work_dir.expanduser().resolve()
    output_json = (
        (args.output_json or _default_output_path(args)).expanduser().resolve()
    )
    measured = json.loads(output_json.read_text(encoding="utf-8"))
    trace_metadata = measured.get("trace", {})
    expected_trace = {
        "dataset": args.dataset,
        "batch_size_per_rank": args.batch_size,
        "layer_id": args.layer,
    }
    mismatches = [
        f"{key}={trace_metadata.get(key)!r}, expected {value!r}"
        for key, value in expected_trace.items()
        if trace_metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Measured result {output_json} does not match the selected trace: "
            + "; ".join(mismatches)
        )

    simulation_path = (
        (
            args.simulation_json
            or work_dir / "replica_simulation/replica_simulation.json"
        )
        .expanduser()
        .resolve()
    )
    simulation = json.loads(simulation_path.read_text(encoding="utf-8"))
    experiment = _find_experiment(simulation, args.dataset, args.batch_size, args.layer)
    experiment_dir = _experiment_dir(work_dir, args.dataset, args.batch_size)
    trace = _aggregate_trace(experiment_dir)
    if args.layer not in trace.layers:
        raise ValueError(f"Layer {args.layer} is missing from the trace")
    if int(experiment["num_experts"]) != trace.num_experts:
        raise ValueError("Simulation and trace disagree on the logical expert count")

    requested_top_k = args.top_k
    if requested_top_k is None and experiment.get("top_k") is not None:
        requested_top_k = int(experiment["top_k"])
    token_steps, _, _ = _load_token_trace(
        experiment_dir,
        trace,
        args.layer,
        requested_top_k,
    )
    measured_steps = int(trace_metadata.get("num_steps", len(token_steps)))
    if not 0 < measured_steps <= len(token_steps):
        raise ValueError(
            f"Measured num_steps={measured_steps} is outside the trace length "
            f"{len(token_steps)}"
        )
    token_steps = token_steps[:measured_steps]
    best = _best_measured_result(measured["results"])
    replicas = _point_replicas(_base_replicas(trace), best)
    latency_config = simulation["latency_model"]
    compute_weight, communication_weight = _routing_weights(best)
    routing = _evaluate_token_trace(
        token_steps,
        replicas,
        LatencyModel(
            compute_us_per_token=_latency_config_value(
                latency_config,
                "compute_us_per_token",
                "compute_us_per_assignment",
            ),
            communication_us_per_token=_latency_config_value(
                latency_config,
                "communication_us_per_token",
                "communication_us_per_assignment",
            ),
        ),
        compute_weight=compute_weight,
        communication_weight=communication_weight,
    )
    profile = _assignment_profile(
        routing.expert_rank_loads,
        replicas,
        routing.rank_token_loads,
        local_tokens=routing.local_tokens,
        remote_tokens=routing.remote_tokens,
        remote_token_transfers=routing.remote_token_transfers,
    )
    title = f"{args.dataset}, batch {args.batch_size}, layer {args.layer}"
    paths = _write_best_load_outputs(best, profile, output_json, title)
    print(
        f"Best measured configuration: {best['label']} "
        f"(serial={float(best['timing']['serial_ms']):.3f} ms)"
    )
    for path in paths:
        print(f"Wrote {path}")


def run_worker(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 2:
        raise ValueError("MoE trace replay requires WORLD_SIZE >= 2")
    work_dir = args.work_dir.expanduser().resolve()
    simulation_path = (
        (
            args.simulation_json
            or work_dir / "replica_simulation/replica_simulation.json"
        )
        .expanduser()
        .resolve()
    )
    simulation = json.loads(simulation_path.read_text(encoding="utf-8"))
    experiment = _find_experiment(simulation, args.dataset, args.batch_size, args.layer)
    points = _select_points(experiment, args)
    experiment_dir = _experiment_dir(work_dir, args.dataset, args.batch_size)
    trace = _aggregate_trace(experiment_dir)
    if trace.ep_size != world_size:
        raise ValueError(
            f"Trace ep_size={trace.ep_size}, but torchrun WORLD_SIZE={world_size}"
        )
    if int(experiment["num_experts"]) != trace.num_experts:
        raise ValueError("Simulation and trace disagree on the logical expert count")
    if args.layer not in trace.layers:
        raise ValueError(f"Layer {args.layer} is missing from the trace")

    shape = _resolve_model_shape(args, simulation)
    if (
        shape["config_num_experts"] is not None
        and int(shape["config_num_experts"]) != trace.num_experts
    ):
        raise ValueError(
            f"Model config has {shape['config_num_experts']} experts, but the "
            f"trace has {trace.num_experts}"
        )
    demands = _demand_tensor(trace, args.layer)
    if args.max_steps is not None:
        demands = demands[: args.max_steps]
    if not len(demands):
        raise ValueError("The selected trace contains no forwards")
    token_steps, inferred_top_k, topk_reconstruction = _load_token_trace(
        experiment_dir,
        trace,
        args.layer,
        shape["top_k"],
    )
    token_steps = token_steps[: len(demands)]
    if inferred_top_k != shape["top_k"]:
        raise ValueError(
            f"Trace top-k={inferred_top_k} does not match selected top-k="
            f"{shape['top_k']}"
        )
    tokens_per_rank = np.asarray(
        [
            [len(token_steps[step][rank]) for rank in range(world_size)]
            for step in range(len(token_steps))
        ],
        dtype=np.int64,
    )
    max_tokens = int(tokens_per_rank.max(initial=0))
    if max_tokens <= 0:
        raise ValueError("The selected trace contains no routed tokens")

    latency_config = simulation["latency_model"]
    latency_model = LatencyModel(
        compute_us_per_token=_latency_config_value(
            latency_config,
            "compute_us_per_token",
            "compute_us_per_assignment",
        ),
        communication_us_per_token=_latency_config_value(
            latency_config,
            "communication_us_per_token",
            "communication_us_per_assignment",
        ),
    )
    plans, capacity_per_rank = _make_local_replay_plans(
        points,
        token_steps,
        _base_replicas(trace),
        latency_model,
        rank,
    )
    physical_num_experts = capacity_per_rank * world_size
    kernel_args = copy(args)
    kernel_args.model = shape["model"]
    kernel_args.tokens = max_tokens
    kernel_args.hidden_size = shape["hidden_size"]
    kernel_args.intermediate_size = shape["intermediate_size"]
    kernel_args.num_experts = physical_num_experts
    kernel_args.top_k = shape["top_k"]
    kernel_args.detail_profile = False

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rounded_hidden_size = DeepEPHTPrepareAndFinalize.maybe_roundup_layer_hidden_size(
        shape["hidden_size"], torch.bfloat16
    )
    if rounded_hidden_size != shape["hidden_size"]:
        raise ValueError(
            f"hidden_size={shape['hidden_size']} is not aligned for DeepEP-HT; "
            f"use --hidden-size {rounded_hidden_size}"
        )
    vllm_config = make_vllm_config(world_size, rank, local_rank)
    output_results: list[dict[str, Any]] = []
    try:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                local_rank=local_rank,
                backend="nccl",
            )
            initialize_model_parallel(tensor_model_parallel_size=1)
            init_workspace_manager(device)
            deepep_kernel, local_kernel = make_kernels(
                kernel_args, vllm_config, torch.bfloat16, device
            )
            tensors = make_base_tensors(
                kernel_args, rank, world_size, torch.bfloat16, device
            )

            def run_step(
                plan: dict[str, Any],
                step_index: int,
                replay_iteration: int,
            ) -> dict[str, Any]:
                local_tokens = int(tokens_per_rank[step_index, rank])
                step_args = copy(kernel_args)
                step_args.tokens = local_tokens
                step_tensors = {
                    **tensors,
                    "hidden_states": tensors["hidden_states"][:local_tokens],
                    "topk_weights": tensors["topk_weights"][:local_tokens],
                }
                num_tokens_across_dp = torch.tensor(
                    tokens_per_rank[step_index], device=device, dtype=torch.int
                )
                with set_forward_context(
                    None,
                    vllm_config,
                    num_tokens=local_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                ):
                    record = run_local_bypass_iter(
                        step_args,
                        deepep_kernel,
                        local_kernel,
                        step_tensors,
                        plan["physical_topk_ids"][step_index],
                        distribution="trace_replay",
                        target_share=0.0,
                        iteration=replay_iteration,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                    )
                record["replay_iteration"] = replay_iteration
                record["trace_step_index"] = step_index
                record["raw_trace_step"] = int(
                    trace.layers[args.layer].raw_steps[step_index]
                )
                return record

            for plan in plans:
                plan["physical_topk_ids"] = [
                    torch.from_numpy(ids).to(device)
                    for ids in plan["physical_topk_ids"]
                ]
                if rank == 0:
                    print(f"Warming up {_point_label(plan['point'])}")
                for _ in range(args.warmup):
                    for step_index in range(len(demands)):
                        run_step(plan, step_index, -1)
                dist.barrier()
                point = plan["point"]
                if rank == 0:
                    print(f"Measuring {_point_label(point)}")
                local_rows = [
                    run_step(plan, step_index, replay_iteration)
                    for replay_iteration in range(args.iters)
                    for step_index in range(len(demands))
                ]
                gathered: list[Any] | None = [None] * world_size if rank == 0 else None
                dist.gather_object(local_rows, gathered, dst=0)
                if rank == 0:
                    assert gathered is not None
                    rank_rows = [row for rank_rows in gathered for row in rank_rows]
                    timing, forward_timings = _summarize_rows(
                        rank_rows,
                        args.iters,
                        len(demands),
                        world_size,
                        args.trim_ratio,
                    )
                    output_results.append(
                        {
                            "label": _point_label(point),
                            "policy": point["policy"],
                            "communication_weight": point["communication_weight"],
                            "requested_extra_replicas": point.get(
                                "requested_extra_replicas",
                                point["used_extra_replicas"],
                            ),
                            "used_extra_replicas": point["used_extra_replicas"],
                            "replica_placements": point["replica_placements"],
                            "offline_estimate": (
                                {
                                    "compute_ms": point["estimated_compute_latency_ms"],
                                    "communication_ms": point[
                                        "estimated_communication_latency_ms"
                                    ],
                                    "serial_ms": point["estimated_serial_latency_ms"],
                                }
                                if len(demands)
                                == len(trace.layers[args.layer].raw_steps)
                                else None
                            ),
                            "timing": timing,
                            "forward_timings": forward_timings,
                            "assignment_profile": _assignment_profile(
                                plan["expert_rank_loads"],
                                plan["replicas"],
                                plan["rank_token_loads"],
                                local_tokens=plan["local_tokens"],
                                remote_tokens=plan["remote_tokens"],
                                remote_token_transfers=plan["remote_token_transfers"],
                            ),
                        }
                    )
                plan["physical_topk_ids"] = []
                dist.barrier()

            if rank == 0:
                _mark_measured_pareto(output_results)
                output_json = (
                    args.output_json or _default_output_path(args)
                ).expanduser()
                output_json = output_json.resolve()
                output_json.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "measurement_kind": "real_deepep_ht_operator_replay",
                    "timing_scope": (
                        "sum of per-forward max-rank dispatch, expert compute, "
                        "and combine timings"
                    ),
                    "topk_reconstruction": topk_reconstruction,
                    "warning": (
                        (
                            "Count-only traces do not retain token top-k tuples. "
                            "The local-bypass routing objective is therefore "
                            "based on synthesized token rows. "
                            if topk_reconstruction != "captured_token_topk_ids"
                            else "Captured token top-k tuples are replayed exactly. "
                        )
                        + "This is an operator benchmark, not end-to-end inference "
                        "latency."
                    ),
                    "trace": {
                        "work_dir": str(work_dir),
                        "dataset": args.dataset,
                        "batch_size_per_rank": args.batch_size,
                        "layer_id": args.layer,
                        "num_steps": len(demands),
                        "ep_size": world_size,
                        "logical_num_experts": trace.num_experts,
                        "top_k": shape["top_k"],
                    },
                    "operator_shape": {
                        **shape,
                        "physical_experts_per_rank": capacity_per_rank,
                        "physical_num_experts": physical_num_experts,
                        "capacity_policy": "common_max_across_selected_points",
                        "max_tokens_per_rank": max_tokens,
                        "dtype": "bfloat16",
                        "execution_mode": args.execution_mode,
                        "stage_barrier": args.stage_barrier,
                    },
                    "results": output_results,
                }
                output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                csv_path = output_json.with_suffix(".csv")
                _write_csv(output_results, csv_path)
                plot_path = output_json.with_suffix(".png")
                _plot_results(
                    output_results,
                    plot_path,
                    f"{args.dataset}, batch {args.batch_size}, layer {args.layer}",
                )
                best_result = _best_measured_result(output_results)
                best_load_paths = _write_best_load_outputs(
                    best_result,
                    best_result["assignment_profile"],
                    output_json,
                    f"{args.dataset}, batch {args.batch_size}, layer {args.layer}",
                )
                print(
                    "policy requested used weight dispatch_ms compute_ms "
                    "combine_ms communication_ms serial_ms pareto"
                )
                for result in output_results:
                    timing = result["timing"]
                    print(
                        f"{result['policy']} "
                        f"{result['requested_extra_replicas']} "
                        f"{result['used_extra_replicas']} "
                        f"{result['communication_weight']} "
                        f"{timing['dispatch_ms']:.3f} {timing['compute_ms']:.3f} "
                        f"{timing['combine_ms']:.3f} "
                        f"{timing['communication_ms']:.3f} "
                        f"{timing['serial_ms']:.3f} "
                        f"{result['measured_pareto_optimal']}"
                    )
                print(f"Wrote {output_json}")
                print(f"Wrote {csv_path}")
                print(f"Wrote {plot_path}")
                print(
                    f"Best measured configuration: {best_result['label']} "
                    f"(serial={float(best_result['timing']['serial_ms']):.3f} ms)"
                )
                for path in best_load_paths:
                    print(f"Wrote {path}")
    finally:
        if dist.is_initialized():
            cleanup_dist_env_and_memory()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        _plot_existing_best_load(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("MoE trace replay requires CUDA")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if not 0.0 <= args.trim_ratio < 0.5:
        raise ValueError("--trim-ratio must be in [0, 0.5)")
    if not all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise RuntimeError("Launch this benchmark with torchrun")
    os.environ["VLLM_DEEPEP_HT_PROFILE"] = "0"
    run_worker(args)


if __name__ == "__main__":
    main()
