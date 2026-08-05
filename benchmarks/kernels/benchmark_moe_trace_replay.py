# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay traced MoE distributions through real DeepEP-HT and expert kernels.

Run this module with one process per expert-parallel rank::

    .venv/bin/python -m torch.distributed.run --standalone \
        --nproc-per-node=4 --module \
        benchmarks.kernels.benchmark_moe_trace_replay \
        --work-dir /tmp/qwen3_expert_distribution \
        --dataset math --batch-size 4 --layer 23 \
        --extra-replicas 0 8

The benchmark uses synthetic activations and weights with the traced token and
expert shapes. Count-only traces require reconstructing token top-k rows, so
expert assignment counts are exact while token destination coalescing is an
approximation.
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
    run_one_iter,
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
    _physical_expert_layout,
    _route_step,
    _route_to_physical_topk_ids,
    _synthesize_logical_topk_ids,
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
        choices=("full_deepep", "local_bypass"),
        default="full_deepep",
        help=(
            "Use the production DeepEP path, or bypass DeepEP for tokens whose "
            "entire synthesized top-k is local."
        ),
    )
    parser.add_argument("--output-json", type=Path)
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
            and int(point["used_extra_replicas"]) not in requested_budgets
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
    label = f"{point['policy']}:B={point['used_extra_replicas']}"
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


def _resolve_model_shape(
    args: argparse.Namespace,
    simulation: dict[str, Any],
) -> dict[str, Any]:
    model = args.model or simulation.get("model")
    preset = MODEL_CONFIGS.get(model, {})

    def resolve(name: str) -> int:
        value = getattr(args, name)
        if value is None:
            value = preset.get(name)
        if value is None:
            option = name.replace("_", "-")
            raise ValueError(
                f"No built-in shape is available for model {model!r}; pass --{option}"
            )
        if int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
        return int(value)

    return {
        "model": model,
        "hidden_size": resolve("hidden_size"),
        "intermediate_size": resolve("intermediate_size"),
        "top_k": resolve("top_k"),
    }


def _make_local_replay_plans(
    points: list[dict[str, Any]],
    demands: np.ndarray,
    logical_topk_ids: list[np.ndarray],
    base_replicas: list[set[int]],
    latency_model: LatencyModel,
    routing_chunks: int,
    rank: int,
) -> tuple[list[dict[str, Any]], int]:
    ep_size = demands.shape[1]
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
        for demand, logical_ids in zip(demands, logical_topk_ids):
            routing = _route_step(
                demand,
                replicas,
                latency_model,
                compute_weight=compute_weight,
                communication_weight=communication_weight,
                routing_chunks=routing_chunks,
            )
            physical_topk_ids.append(
                _route_to_physical_topk_ids(
                    logical_ids,
                    routing.source_expert_rank_loads[rank],
                    physical_layout,
                    rank,
                )
            )
        plans.append(
            {
                "point": point,
                "replicas": replicas,
                "physical_topk_ids": physical_topk_ids,
            }
        )
    return plans, capacity_per_rank


def _load_captured_topk_ids(
    experiment_dir: Path,
    layer_id: int,
    raw_steps: np.ndarray,
    demands: np.ndarray,
    top_k: int,
    replay_rank: int,
) -> list[np.ndarray] | None:
    local_ids = []
    for source_rank in range(demands.shape[1]):
        source_ids = []
        for step_index, raw_step in enumerate(raw_steps):
            path = (
                experiment_dir
                / "activations"
                / f"rank_{source_rank:05d}"
                / f"step_{int(raw_step):06d}_layer_{layer_id:04d}.pt"
            )
            expected_counts = demands[step_index, source_rank]
            if not path.exists():
                if int(expected_counts.sum()):
                    raise ValueError(f"Missing trace record {path}")
                ids = np.empty((0, top_k), dtype=np.int64)
            else:
                record = torch.load(path, map_location="cpu", weights_only=True)
                if "topk_ids" not in record:
                    return None
                ids = record["topk_ids"].to(torch.int64).numpy()
                if ids.ndim != 2 or ids.shape[1] != top_k:
                    raise ValueError(
                        f"Captured top-k shape {ids.shape} in {path} does not match "
                        f"top_k={top_k}"
                    )
                if np.any((ids < 0) | (ids >= demands.shape[2])):
                    raise ValueError(f"Captured top-k IDs in {path} are out of range")
                actual_counts = np.bincount(
                    ids.reshape(-1), minlength=demands.shape[2]
                ).astype(np.int64)
                if not np.array_equal(actual_counts, expected_counts):
                    raise ValueError(
                        f"Captured top-k IDs in {path} do not match aggregated counts"
                    )
            if source_rank == replay_rank:
                source_ids.append(ids)
        if source_rank == replay_rank:
            local_ids = source_ids
    return local_ids


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
        rows.append(
            {
                "label": result["label"],
                "policy": result["policy"],
                "communication_weight": result["communication_weight"],
                "used_extra_replicas": result["used_extra_replicas"],
                "measured_pareto_optimal": result["measured_pareto_optimal"],
                **result["timing"],
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
    demands = _demand_tensor(trace, args.layer)
    if args.max_steps is not None:
        demands = demands[: args.max_steps]
    if not len(demands):
        raise ValueError("The selected trace contains no forwards")
    assignments_per_rank = demands.sum(axis=2)
    if np.any(assignments_per_rank % shape["top_k"]):
        raise ValueError(
            "Trace assignment counts are not divisible by the selected --top-k"
        )
    tokens_per_rank = assignments_per_rank // shape["top_k"]
    max_tokens = int(tokens_per_rank.max(initial=0))
    if max_tokens <= 0:
        raise ValueError("The selected trace contains no routed tokens")
    raw_steps = trace.layers[args.layer].raw_steps[: len(demands)]
    logical_topk_ids = _load_captured_topk_ids(
        experiment_dir,
        args.layer,
        raw_steps,
        demands,
        shape["top_k"],
        rank,
    )
    if logical_topk_ids is None:
        topk_reconstruction = "synthesized_from_exact_expert_counts"
        logical_topk_ids = [
            _synthesize_logical_topk_ids(demand[rank], shape["top_k"])
            for demand in demands
        ]
    else:
        topk_reconstruction = "captured_token_topk_ids"

    latency_config = simulation["latency_model"]
    latency_model = LatencyModel(
        compute_us_per_assignment=float(latency_config["compute_us_per_assignment"]),
        communication_us_per_assignment=float(
            latency_config["communication_us_per_assignment"]
        ),
    )
    plans, capacity_per_rank = _make_local_replay_plans(
        points,
        demands,
        logical_topk_ids,
        _base_replicas(trace),
        latency_model,
        int(simulation["routing_chunks"]),
        rank,
    )
    physical_num_experts = capacity_per_rank * world_size
    kernel_args = argparse.Namespace(
        model=shape["model"],
        tokens=max_tokens,
        hidden_size=shape["hidden_size"],
        intermediate_size=shape["intermediate_size"],
        num_experts=physical_num_experts,
        top_k=shape["top_k"],
        seed=args.seed,
        stage_barrier=args.stage_barrier,
        detail_profile=False,
        execution_mode=args.execution_mode,
    )

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
                    if args.execution_mode == "local_bypass":
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
                    else:
                        record = run_one_iter(
                            step_args,
                            deepep_kernel,
                            step_tensors,
                            plan["physical_topk_ids"][step_index],
                            distribution="trace_replay",
                            target_share=0.0,
                            sweep_index=0,
                            iteration=replay_iteration,
                            rank=rank,
                            world_size=world_size,
                            device=device,
                            profile_warmup=0,
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
                            "Assignment counts and rank loads are exact, but "
                            "unique-token communication coalescing is reconstructed. "
                            if topk_reconstruction.startswith("synthesized")
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
                print(
                    "policy budget weight dispatch_ms compute_ms combine_ms "
                    "communication_ms serial_ms pareto"
                )
                for result in output_results:
                    timing = result["timing"]
                    print(
                        f"{result['policy']} {result['used_extra_replicas']} "
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
    finally:
        if dist.is_initialized():
            cleanup_dist_env_and_memory()


def main() -> None:
    args = parse_args()
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
