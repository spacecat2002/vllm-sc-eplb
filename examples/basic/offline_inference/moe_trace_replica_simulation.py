# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simulate MoE expert replication from real expert-distribution traces."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from heapq import heapify, heappop, heappush
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .moe_trace_expert_distribution import (
        TraceDistribution,
        _aggregate_trace,
        _experiment_dir,
        _local_expert_ids,
        _selected_layers,
    )
else:
    from moe_trace_expert_distribution import (  # type: ignore[no-redef]
        TraceDistribution,
        _aggregate_trace,
        _experiment_dir,
        _local_expert_ids,
        _selected_layers,
    )


@dataclass(frozen=True)
class LatencyModel:
    compute_us_per_assignment: float
    communication_us_per_assignment: float

    def compute_ms(self, max_rank_assignments: int) -> float:
        return max_rank_assignments * self.compute_us_per_assignment / 1000.0

    def communication_ms(self, bottleneck_assignments: int) -> float:
        # Dispatch and combine move one activation in each direction.
        return (
            2.0 * bottleneck_assignments * self.communication_us_per_assignment / 1000.0
        )


@dataclass
class StepRoutingResult:
    rank_loads: np.ndarray
    expert_rank_loads: np.ndarray
    source_expert_rank_loads: np.ndarray
    outbound_remote: np.ndarray
    inbound_remote: np.ndarray
    remote_assignments: int
    compute_latency_ms: float
    communication_latency_ms: float
    objective: float


@dataclass
class TraceRoutingResult:
    rank_loads: np.ndarray
    expert_rank_loads: np.ndarray
    remote_assignments: int
    bottleneck_remote_assignments: int
    compute_latency_ms: float
    communication_latency_ms: float
    serial_latency_ms: float
    overlap_lower_bound_ms: float
    mean_step_imbalance: float
    p95_step_imbalance: float
    max_step_imbalance: float
    objective: float


def _demand_tensor(
    distribution: TraceDistribution,
    layer_id: int,
) -> np.ndarray:
    layer = distribution.layers[layer_id]
    return np.stack(
        [layer.rank_step_counts[rank] for rank in range(distribution.ep_size)],
        axis=1,
    )


def _base_replicas(distribution: TraceDistribution) -> list[set[int]]:
    replicas = [set() for _ in range(distribution.num_experts)]
    for rank in range(distribution.ep_size):
        for expert_id in _local_expert_ids(
            distribution.num_experts,
            distribution.ep_size,
            rank,
            distribution.expert_placement_strategy,
        ):
            replicas[expert_id].add(rank)
    if any(len(expert_replicas) != 1 for expert_replicas in replicas):
        raise ValueError("Every logical expert must have exactly one base owner")
    return replicas


def _copy_replicas(replicas: list[set[int]]) -> list[set[int]]:
    return [set(expert_replicas) for expert_replicas in replicas]


def _route_step(
    demand: np.ndarray,
    replicas: list[set[int]],
    latency_model: LatencyModel,
    *,
    compute_weight: float,
    communication_weight: float,
    routing_chunks: int,
) -> StepRoutingResult:
    if demand.ndim != 2:
        raise ValueError("demand must have shape [source_rank, expert]")
    if routing_chunks <= 0:
        raise ValueError("routing_chunks must be positive")
    if np.any(demand < 0):
        raise ValueError("demand must be non-negative")
    ep_size, num_experts = demand.shape
    if len(replicas) != num_experts:
        raise ValueError("replica count does not match demand expert dimension")

    rank_loads = np.zeros(ep_size, dtype=np.int64)
    expert_rank_loads = np.zeros((num_experts, ep_size), dtype=np.int64)
    source_expert_rank_loads = np.zeros((ep_size, num_experts, ep_size), dtype=np.int64)
    outbound = np.zeros(ep_size, dtype=np.int64)
    inbound = np.zeros(ep_size, dtype=np.int64)
    remote_assignments = 0
    blocks = [
        (len(replicas[expert_id]), -int(demand[source, expert_id]), source, expert_id)
        for source in range(ep_size)
        for expert_id in range(num_experts)
        if demand[source, expert_id] > 0
    ]
    blocks.sort()

    for _, negative_count, source, expert_id in blocks:
        count = -negative_count
        allowed_ranks = sorted(replicas[expert_id])
        if not allowed_ranks:
            raise ValueError(f"Expert {expert_id} has no replica")
        num_chunks = min(count, routing_chunks)
        base_chunk, remainder = divmod(count, num_chunks)
        for chunk_index in range(num_chunks):
            chunk = base_chunk + int(chunk_index < remainder)
            best_rank = -1
            best_key: tuple[float, int, int, int] | None = None
            for target in allowed_ranks:
                projected_max_load = max(
                    int(rank_loads.max(initial=0)), int(rank_loads[target]) + chunk
                )
                projected_outbound = int(outbound.max(initial=0))
                projected_inbound = int(inbound.max(initial=0))
                if target != source:
                    projected_outbound = max(
                        projected_outbound, int(outbound[source]) + chunk
                    )
                    projected_inbound = max(
                        projected_inbound, int(inbound[target]) + chunk
                    )
                bottleneck = max(projected_outbound, projected_inbound)
                score = compute_weight * latency_model.compute_ms(
                    projected_max_load
                ) + communication_weight * latency_model.communication_ms(bottleneck)
                key = (
                    score,
                    int(target != source),
                    int(rank_loads[target]),
                    target,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_rank = target
            rank_loads[best_rank] += chunk
            expert_rank_loads[expert_id, best_rank] += chunk
            source_expert_rank_loads[source, expert_id, best_rank] += chunk
            if best_rank != source:
                outbound[source] += chunk
                inbound[best_rank] += chunk
                remote_assignments += chunk

    if int(rank_loads.sum()) != int(demand.sum()):
        raise RuntimeError("Routing did not preserve all expert assignments")

    compute_latency_ms = latency_model.compute_ms(int(rank_loads.max(initial=0)))
    bottleneck = max(
        int(outbound.max(initial=0)),
        int(inbound.max(initial=0)),
    )
    communication_latency_ms = latency_model.communication_ms(bottleneck)
    return StepRoutingResult(
        rank_loads=rank_loads,
        expert_rank_loads=expert_rank_loads,
        source_expert_rank_loads=source_expert_rank_loads,
        outbound_remote=outbound,
        inbound_remote=inbound,
        remote_assignments=remote_assignments,
        compute_latency_ms=compute_latency_ms,
        communication_latency_ms=communication_latency_ms,
        objective=(
            compute_weight * compute_latency_ms
            + communication_weight * communication_latency_ms
        ),
    )


def _synthesize_logical_topk_ids(
    expert_counts: np.ndarray,
    top_k: int,
) -> np.ndarray:
    counts = np.asarray(expert_counts, dtype=np.int64)
    if counts.ndim != 1:
        raise ValueError("expert_counts must be one-dimensional")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if np.any(counts < 0):
        raise ValueError("expert_counts must be non-negative")
    total_assignments = int(counts.sum())
    if total_assignments % top_k:
        raise ValueError(
            f"{total_assignments} assignments are not divisible by top_k={top_k}"
        )
    num_tokens = total_assignments // top_k
    if np.any(counts > num_tokens):
        raise ValueError(
            "An expert count exceeds the number of tokens; unique top-k IDs "
            "cannot be reconstructed"
        )

    heap = [(-int(count), expert_id) for expert_id, count in enumerate(counts) if count]
    heapify(heap)
    result = np.empty((num_tokens, top_k), dtype=np.int64)
    for token_index in range(num_tokens):
        if len(heap) < top_k:
            raise ValueError("Expert counts do not form valid unique token top-k rows")
        selected = [heappop(heap) for _ in range(top_k)]
        for slot, (negative_count, expert_id) in enumerate(selected):
            result[token_index, slot] = expert_id
            if negative_count < -1:
                heappush(heap, (negative_count + 1, expert_id))
    if heap:
        raise RuntimeError("Synthesized top-k IDs did not consume every assignment")
    return result


def _physical_expert_layout(
    replicas: list[set[int]],
    ep_size: int,
    capacity_per_rank: int | None = None,
) -> tuple[np.ndarray, int]:
    if ep_size <= 0:
        raise ValueError("ep_size must be positive")
    experts_by_rank = [[] for _ in range(ep_size)]
    for expert_id, expert_replicas in enumerate(replicas):
        if not expert_replicas:
            raise ValueError(f"Expert {expert_id} has no replica")
        for rank in sorted(expert_replicas):
            if not 0 <= rank < ep_size:
                raise ValueError(f"Replica rank {rank} is outside [0, {ep_size})")
            experts_by_rank[rank].append(expert_id)
    required_capacity = max(map(len, experts_by_rank), default=0)
    if capacity_per_rank is None:
        capacity_per_rank = required_capacity
    if capacity_per_rank < required_capacity:
        raise ValueError(
            f"capacity_per_rank={capacity_per_rank} is smaller than "
            f"required capacity {required_capacity}"
        )

    physical_ids = np.full((len(replicas), ep_size), -1, dtype=np.int64)
    for rank, expert_ids in enumerate(experts_by_rank):
        for local_slot, expert_id in enumerate(expert_ids):
            physical_ids[expert_id, rank] = rank * capacity_per_rank + local_slot
    return physical_ids, capacity_per_rank


def _route_to_physical_topk_ids(
    logical_topk_ids: np.ndarray,
    expert_target_counts: np.ndarray,
    physical_expert_ids: np.ndarray,
    source_rank: int,
) -> np.ndarray:
    logical_ids = np.asarray(logical_topk_ids, dtype=np.int64)
    target_counts = np.asarray(expert_target_counts, dtype=np.int64)
    if logical_ids.ndim != 2:
        raise ValueError("logical_topk_ids must have shape [tokens, top_k]")
    if target_counts.shape != physical_expert_ids.shape:
        raise ValueError(
            "expert_target_counts and physical_expert_ids must have the same shape"
        )
    if np.any(target_counts < 0):
        raise ValueError("expert_target_counts must be non-negative")
    ep_size = target_counts.shape[1]
    if not 0 <= source_rank < ep_size:
        raise ValueError(f"source_rank must be in [0, {ep_size})")

    flat_logical = logical_ids.reshape(-1)
    flat_physical = np.full(flat_logical.shape, -1, dtype=np.int64)
    for expert_id in range(target_counts.shape[0]):
        positions = np.flatnonzero(flat_logical == expert_id)
        remaining = target_counts[expert_id].copy()
        if int(remaining.sum()) != len(positions):
            raise ValueError(
                f"Expert {expert_id} has {len(positions)} logical assignments but "
                f"{int(remaining.sum())} routed assignments"
            )
        targets = []
        while int(remaining.sum()):
            for offset in range(ep_size):
                target = (source_rank + offset) % ep_size
                if remaining[target] > 0:
                    targets.append(target)
                    remaining[target] -= 1
        for position, target in zip(positions, targets):
            physical_id = int(physical_expert_ids[expert_id, target])
            if physical_id < 0:
                raise ValueError(
                    f"Expert {expert_id} has routed work on rank {target} "
                    "without a physical replica"
                )
            flat_physical[position] = physical_id
    if np.any(flat_physical < 0):
        raise ValueError("logical_topk_ids contains an out-of-range expert ID")
    return flat_physical.reshape(logical_ids.shape)


def _evaluate_trace(
    demands: np.ndarray,
    replicas: list[set[int]],
    latency_model: LatencyModel,
    *,
    compute_weight: float,
    communication_weight: float,
    routing_chunks: int,
) -> TraceRoutingResult:
    if demands.ndim != 3:
        raise ValueError("demands must have shape [step, source_rank, expert]")
    ep_size = demands.shape[1]
    num_experts = demands.shape[2]
    rank_loads = np.zeros(ep_size, dtype=np.int64)
    expert_rank_loads = np.zeros((num_experts, ep_size), dtype=np.int64)
    remote_assignments = 0
    bottleneck_remote_assignments = 0
    compute_latency_ms = 0.0
    communication_latency_ms = 0.0
    overlap_lower_bound_ms = 0.0
    objective = 0.0
    imbalances = []

    for demand in demands:
        result = _route_step(
            demand,
            replicas,
            latency_model,
            compute_weight=compute_weight,
            communication_weight=communication_weight,
            routing_chunks=routing_chunks,
        )
        rank_loads += result.rank_loads
        expert_rank_loads += result.expert_rank_loads
        remote_assignments += result.remote_assignments
        bottleneck_remote_assignments += max(
            int(result.outbound_remote.max(initial=0)),
            int(result.inbound_remote.max(initial=0)),
        )
        compute_latency_ms += result.compute_latency_ms
        communication_latency_ms += result.communication_latency_ms
        overlap_lower_bound_ms += max(
            result.compute_latency_ms, result.communication_latency_ms
        )
        objective += result.objective
        mean_load = float(result.rank_loads.mean())
        imbalances.append(
            float(result.rank_loads.max(initial=0)) / mean_load if mean_load else 0.0
        )

    imbalance_array = np.asarray(imbalances, dtype=np.float64)
    return TraceRoutingResult(
        rank_loads=rank_loads,
        expert_rank_loads=expert_rank_loads,
        remote_assignments=remote_assignments,
        bottleneck_remote_assignments=bottleneck_remote_assignments,
        compute_latency_ms=compute_latency_ms,
        communication_latency_ms=communication_latency_ms,
        serial_latency_ms=compute_latency_ms + communication_latency_ms,
        overlap_lower_bound_ms=overlap_lower_bound_ms,
        mean_step_imbalance=float(imbalance_array.mean()),
        p95_step_imbalance=float(np.percentile(imbalance_array, 95)),
        max_step_imbalance=float(imbalance_array.max(initial=0.0)),
        objective=objective,
    )


def _candidate_replicas(
    demand: np.ndarray,
    replicas: list[set[int]],
    routing: StepRoutingResult,
    candidate_limit: int,
) -> list[tuple[int, int]]:
    ep_size, num_experts = demand.shape
    candidates = [
        (expert_id, rank)
        for expert_id in range(num_experts)
        for rank in range(ep_size)
        if rank not in replicas[expert_id]
    ]
    if candidate_limit <= 0 or len(candidates) <= candidate_limit:
        return candidates

    communication_order = sorted(
        candidates,
        key=lambda item: (-int(demand[item[1], item[0]]), item),
    )
    max_rank_load = int(routing.rank_loads.max(initial=0))

    def balance_score(candidate: tuple[int, int]) -> int:
        expert_id, rank = candidate
        headroom = max_rank_load - int(routing.rank_loads[rank])
        movable = int(routing.expert_rank_loads[expert_id].max(initial=0))
        return min(headroom, movable)

    balance_order = sorted(
        candidates,
        key=lambda item: (-balance_score(item), item),
    )
    shortlist = set(communication_order[:candidate_limit])
    shortlist.update(balance_order[:candidate_limit])
    return sorted(shortlist)


def _optimize_replicas(
    aggregate_demand: np.ndarray,
    base_replicas: list[set[int]],
    extra_replica_budget: int,
    latency_model: LatencyModel,
    *,
    compute_weight: float,
    communication_weight: float,
    routing_chunks: int,
    candidate_limit: int,
) -> tuple[list[set[int]], list[tuple[int, int]]]:
    if extra_replica_budget < 0:
        raise ValueError("extra_replica_budget must be non-negative")
    replicas = _copy_replicas(base_replicas)
    placements = []
    current = _route_step(
        aggregate_demand,
        replicas,
        latency_model,
        compute_weight=compute_weight,
        communication_weight=communication_weight,
        routing_chunks=routing_chunks,
    )

    for _ in range(extra_replica_budget):
        candidates = _candidate_replicas(
            aggregate_demand,
            replicas,
            current,
            candidate_limit,
        )
        if not candidates:
            break
        best: tuple[tuple[float, int, int], int, int, StepRoutingResult] | None = None
        for expert_id, rank in candidates:
            candidate_replicas = _copy_replicas(replicas)
            candidate_replicas[expert_id].add(rank)
            routing = _route_step(
                aggregate_demand,
                candidate_replicas,
                latency_model,
                compute_weight=compute_weight,
                communication_weight=communication_weight,
                routing_chunks=routing_chunks,
            )
            key = (routing.objective, expert_id, rank)
            if best is None or key < best[0]:
                best = (key, expert_id, rank, routing)
        assert best is not None
        _, expert_id, rank, current = best
        replicas[expert_id].add(rank)
        placements.append((expert_id, rank))
    return replicas, placements


def _result_payload(
    *,
    policy: str,
    communication_weight: float | None,
    requested_budget: int,
    placements: list[tuple[int, int]],
    demands: np.ndarray,
    replicas: list[set[int]],
    latency_model: LatencyModel,
    routing: TraceRoutingResult,
) -> dict[str, Any]:
    total_assignments = int(demands.sum())
    mean_rank_load = float(routing.rank_loads.mean())
    balanced_compute_lower_bound_ms, communication_lower_bound_ms = (
        _theoretical_latency_lower_bounds(demands, replicas, latency_model)
    )
    return {
        "policy": policy,
        "communication_weight": communication_weight,
        "requested_extra_replicas": requested_budget,
        "used_extra_replicas": len(placements),
        "replica_placements": [
            {"expert_id": expert_id, "rank": rank} for expert_id, rank in placements
        ],
        "total_assignments": total_assignments,
        "remote_assignments": routing.remote_assignments,
        "remote_assignment_percent": (
            routing.remote_assignments / total_assignments * 100.0
            if total_assignments
            else 0.0
        ),
        "bottleneck_remote_assignments": (routing.bottleneck_remote_assignments),
        "rank_assignment_loads": routing.rank_loads.tolist(),
        "max_rank_assignments": int(routing.rank_loads.max(initial=0)),
        "mean_rank_assignments": mean_rank_load,
        "aggregate_max_over_mean": (
            float(routing.rank_loads.max(initial=0)) / mean_rank_load
            if mean_rank_load
            else 0.0
        ),
        "mean_step_max_over_mean": routing.mean_step_imbalance,
        "p95_step_max_over_mean": routing.p95_step_imbalance,
        "max_step_max_over_mean": routing.max_step_imbalance,
        "estimated_compute_latency_ms": routing.compute_latency_ms,
        "estimated_communication_latency_ms": routing.communication_latency_ms,
        "estimated_serial_latency_ms": routing.serial_latency_ms,
        "estimated_overlap_lower_bound_ms": routing.overlap_lower_bound_ms,
        "balanced_compute_lower_bound_ms": balanced_compute_lower_bound_ms,
        "communication_lower_bound_ms": communication_lower_bound_ms,
        "optimization_objective": routing.objective,
        "pareto_optimal": False,
    }


def _theoretical_latency_lower_bounds(
    demands: np.ndarray,
    replicas: list[set[int]],
    latency_model: LatencyModel,
) -> tuple[float, float]:
    ep_size = demands.shape[1]
    balanced_compute_ms = 0.0
    communication_ms = 0.0
    for demand in demands:
        balanced_compute_ms += latency_model.compute_ms(
            ceil(int(demand.sum()) / ep_size)
        )
        unavoidable_outbound = np.asarray(
            [
                sum(
                    int(count)
                    for expert_id, count in enumerate(demand[source])
                    if source not in replicas[expert_id]
                )
                for source in range(ep_size)
            ],
            dtype=np.int64,
        )
        remote_assignments = int(unavoidable_outbound.sum())
        bottleneck_lower_bound = max(
            int(unavoidable_outbound.max(initial=0)),
            ceil(remote_assignments / ep_size),
        )
        communication_ms += latency_model.communication_ms(bottleneck_lower_bound)
    return balanced_compute_ms, communication_ms


def _simulate_layer(
    distribution: TraceDistribution,
    layer_id: int,
    replica_budgets: list[int],
    communication_weights: list[float],
    latency_model: LatencyModel,
    *,
    routing_chunks: int,
    candidate_limit: int,
) -> list[dict[str, Any]]:
    demands = _demand_tensor(distribution, layer_id)
    aggregate_demand = demands.sum(axis=0)
    base_replicas = _base_replicas(distribution)
    max_replicas = distribution.num_experts * (distribution.ep_size - 1)
    invalid_budgets = [
        budget for budget in replica_budgets if not 0 <= budget <= max_replicas
    ]
    if invalid_budgets:
        raise ValueError(
            f"Extra replica budgets must be in [0, {max_replicas}], "
            f"got {invalid_budgets}"
        )

    baseline_routing = _evaluate_trace(
        demands,
        base_replicas,
        latency_model,
        compute_weight=1.0,
        communication_weight=1.0,
        routing_chunks=routing_chunks,
    )
    points = [
        _result_payload(
            policy="baseline",
            communication_weight=None,
            requested_budget=0,
            placements=[],
            demands=demands,
            replicas=base_replicas,
            latency_model=latency_model,
            routing=baseline_routing,
        )
    ]

    policies = [
        ("communication_first", 0.0, 1.0, None),
        ("balance_first", 1.0, 0.0, None),
        *[
            ("joint", 1.0, communication_weight, communication_weight)
            for communication_weight in communication_weights
        ],
    ]
    for budget in replica_budgets:
        if budget == 0:
            continue
        for policy, compute_weight, communication_weight, label_weight in policies:
            replicas, placements = _optimize_replicas(
                aggregate_demand,
                base_replicas,
                budget,
                latency_model,
                compute_weight=compute_weight,
                communication_weight=communication_weight,
                routing_chunks=routing_chunks,
                candidate_limit=candidate_limit,
            )
            routing = _evaluate_trace(
                demands,
                replicas,
                latency_model,
                compute_weight=compute_weight,
                communication_weight=communication_weight,
                routing_chunks=routing_chunks,
            )
            points.append(
                _result_payload(
                    policy=policy,
                    communication_weight=label_weight,
                    requested_budget=budget,
                    placements=placements,
                    demands=demands,
                    replicas=replicas,
                    latency_model=latency_model,
                    routing=routing,
                )
            )
    _mark_pareto_points(points)
    return points


def _mark_pareto_points(points: list[dict[str, Any]]) -> None:
    for point in points:
        point["pareto_optimal"] = not any(
            other is not point
            and other["estimated_compute_latency_ms"]
            <= point["estimated_compute_latency_ms"]
            and other["estimated_communication_latency_ms"]
            <= point["estimated_communication_latency_ms"]
            and (
                other["estimated_compute_latency_ms"]
                < point["estimated_compute_latency_ms"]
                or other["estimated_communication_latency_ms"]
                < point["estimated_communication_latency_ms"]
            )
            for other in points
        )


def _plot_pareto(
    points: list[dict[str, Any]],
    output: Path,
    title: str,
) -> None:
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
    for policy in colors:
        policy_points = [point for point in points if point["policy"] == policy]
        if not policy_points:
            continue
        axis.scatter(
            [point["estimated_communication_latency_ms"] for point in policy_points],
            [point["estimated_compute_latency_ms"] for point in policy_points],
            color=colors[policy],
            marker=markers[policy],
            label=policy,
            s=46,
        )
    pareto = sorted(
        (point for point in points if point["pareto_optimal"]),
        key=lambda point: point["estimated_communication_latency_ms"],
    )
    if pareto:
        axis.plot(
            [point["estimated_communication_latency_ms"] for point in pareto],
            [point["estimated_compute_latency_ms"] for point in pareto],
            color="#202020",
            linewidth=1.0,
            linestyle="--",
            label="Pareto frontier",
        )
    for point in points:
        if point["policy"] == "joint" and not point["pareto_optimal"]:
            continue
        weight = point["communication_weight"]
        label = f"B={point['used_extra_replicas']}"
        if weight is not None:
            label += f", w={weight:g}"
        axis.annotate(
            label,
            (
                point["estimated_communication_latency_ms"],
                point["estimated_compute_latency_ms"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Estimated dispatch + combine latency (ms)")
    axis.set_ylabel("Estimated expert compute latency (ms)")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_csv(experiments: list[dict[str, Any]], output: Path) -> None:
    rows = []
    for experiment in experiments:
        context = {
            "dataset": experiment["dataset"],
            "batch_size_per_rank": experiment["batch_size_per_rank"],
            "layer_id": experiment["layer_id"],
        }
        for point in experiment["points"]:
            row = {**context, **point}
            row["replica_placements"] = json.dumps(row["replica_placements"])
            row["rank_assignment_loads"] = json.dumps(row["rank_assignment_loads"])
            rows.append(row)
    if not rows:
        return
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate(args: argparse.Namespace) -> None:
    if args.compute_us_per_assignment <= 0:
        raise ValueError("--compute-us-per-assignment must be positive")
    if args.communication_us_per_assignment <= 0:
        raise ValueError("--communication-us-per-assignment must be positive")
    if args.routing_chunks <= 0:
        raise ValueError("--routing-chunks must be positive")
    if args.candidate_limit < 0:
        raise ValueError("--candidate-limit must be non-negative")
    if any(weight < 0 for weight in args.communication_weights):
        raise ValueError("--communication-weights must be non-negative")

    work_dir = args.work_dir.expanduser().resolve()
    manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
    datasets = args.datasets or manifest["datasets"]
    batch_sizes = args.batch_sizes or manifest["batch_sizes"]
    distributions = {
        (dataset, batch_size): _aggregate_trace(
            _experiment_dir(work_dir, dataset, batch_size)
        )
        for dataset in datasets
        for batch_size in batch_sizes
    }
    layers = _selected_layers(distributions, args.layers)
    output_dir = (args.output_dir or work_dir / "replica_simulation").expanduser()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latency_model = LatencyModel(
        compute_us_per_assignment=args.compute_us_per_assignment,
        communication_us_per_assignment=args.communication_us_per_assignment,
    )
    replica_budgets = list(dict.fromkeys(args.extra_replicas))
    communication_weights = list(dict.fromkeys(args.communication_weights))
    experiments = []
    for dataset in datasets:
        for batch_size in batch_sizes:
            distribution = distributions[(dataset, batch_size)]
            for layer_id in layers:
                print(
                    f"Simulating dataset={dataset}, batch_size={batch_size}, "
                    f"layer={layer_id}"
                )
                points = _simulate_layer(
                    distribution,
                    layer_id,
                    replica_budgets,
                    communication_weights,
                    latency_model,
                    routing_chunks=args.routing_chunks,
                    candidate_limit=args.candidate_limit,
                )
                experiment = {
                    "dataset": dataset,
                    "batch_size_per_rank": batch_size,
                    "layer_id": layer_id,
                    "num_experts": distribution.num_experts,
                    "ep_size": distribution.ep_size,
                    "expert_placement_strategy": (
                        distribution.expert_placement_strategy
                    ),
                    "points": points,
                }
                experiments.append(experiment)
                plot_path = output_dir / (
                    f"replica_pareto_{dataset}_batch_{batch_size:04d}_"
                    f"layer_{layer_id:04d}.png"
                )
                _plot_pareto(
                    points,
                    plot_path,
                    f"{dataset}, batch size {batch_size}, layer {layer_id}",
                )
                print(f"Saved {plot_path}")

    payload = {
        "model": manifest.get("model"),
        "latency_model": {
            **asdict(latency_model),
            "kind": "linear_assignment_proxy",
            "communication_includes": "dispatch_and_combine",
            "warning": (
                "Count-only traces do not retain token-level top-k coalescing. "
                "Calibrate both coefficients before interpreting values as "
                "hardware latency."
            ),
        },
        "routing_chunks": args.routing_chunks,
        "candidate_limit": args.candidate_limit,
        "experiments": experiments,
    }
    json_path = output_dir / "replica_simulation.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = output_dir / "replica_simulation.csv"
    _write_csv(experiments, csv_path)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument(
        "--extra-replicas",
        type=int,
        nargs="+",
        default=[0, 4, 8, 16],
        help="Total additional expert copies allowed per layer.",
    )
    parser.add_argument(
        "--communication-weights",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 1.0, 3.0, 10.0],
        help="Communication weights for the joint objective.",
    )
    parser.add_argument(
        "--compute-us-per-assignment",
        type=float,
        default=1.0,
        help="Calibrated compute cost; the default is a normalized proxy.",
    )
    parser.add_argument(
        "--communication-us-per-assignment",
        type=float,
        default=1.0,
        help=(
            "Calibrated one-way remote assignment cost; dispatch and combine "
            "are both included in reported latency."
        ),
    )
    parser.add_argument(
        "--routing-chunks",
        type=int,
        default=8,
        help="Maximum chunks used to split each source-rank/expert demand block.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=24,
        help=(
            "Candidates retained from each locality/load shortlist per greedy "
            "step; use 0 for exhaustive search."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    simulate(parse_args())


if __name__ == "__main__":
    main()
