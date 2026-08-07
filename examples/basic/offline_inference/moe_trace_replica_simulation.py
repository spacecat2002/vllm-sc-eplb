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
    """Linear proxy calibrated in expert-input tokens and communication units."""

    compute_us_per_token: float
    communication_us_per_token: float

    def compute_ms(self, max_rank_assignments: int) -> float:
        return max_rank_assignments * self.compute_us_per_token / 1000.0

    def communication_ms(self, communication_units: int) -> float:
        # Dispatch and combine each use one transfer to every remote target rank.
        return 2.0 * communication_units * self.communication_us_per_token / 1000.0


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


@dataclass
class TokenStepRoutingResult:
    """Token routing result with compute and communication load units."""

    target_ranks_by_source: list[np.ndarray]
    rank_assignment_loads: np.ndarray
    rank_token_loads: np.ndarray
    expert_rank_loads: np.ndarray
    source_expert_rank_loads: np.ndarray
    outbound_remote_tokens: np.ndarray
    inbound_remote_tokens: np.ndarray
    local_tokens: int
    remote_tokens: int
    remote_token_transfers: int
    remote_assignments: int
    compute_latency_ms: float
    communication_latency_ms: float
    objective: float


@dataclass
class TokenTraceRoutingResult:
    rank_assignment_loads: np.ndarray
    rank_token_loads: np.ndarray
    expert_rank_loads: np.ndarray
    remote_assignments: int
    local_tokens: int
    remote_tokens: int
    remote_token_transfers: int
    bottleneck_remote_token_transfers: int
    compute_latency_ms: float
    communication_latency_ms: float
    serial_latency_ms: float
    overlap_lower_bound_ms: float
    mean_step_max_over_mean: float
    p95_step_max_over_mean: float
    max_step_max_over_mean: float
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


def _load_token_trace(
    experiment_dir: Path,
    distribution: TraceDistribution,
    layer_id: int,
    top_k: int | None = None,
) -> tuple[list[list[np.ndarray]], int, str]:
    """Load token top-k rows, reconstructing them from counts when necessary."""
    import torch

    layer = distribution.layers[layer_id]
    raw_steps = [int(step) for step in layer.raw_steps]
    demands = _demand_tensor(distribution, layer_id)
    records: dict[tuple[int, int], dict[str, Any]] = {}
    captured_top_k: set[int] = set()
    count_top_k: set[int] = set()
    paths = sorted(
        (experiment_dir / "activations").glob(f"rank_*/step_*_layer_{layer_id:04d}.pt")
    )
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=True)
        rank = int(record["rank"])
        step = int(record["step"])
        if rank >= distribution.ep_size or step not in raw_steps:
            continue
        records[(rank, step)] = record
        if "topk_ids" in record:
            ids = record["topk_ids"]
            if ids.ndim != 2:
                raise ValueError(f"Captured top-k tensor in {path} is not 2-D")
            captured_top_k.add(int(ids.shape[1]))
        elif "expert_counts" in record:
            scheduled = int(record.get("num_scheduled_tokens", 0))
            assignments = int(record["expert_counts"].sum())
            if scheduled > 0:
                if assignments % scheduled:
                    raise ValueError(
                        f"Cannot infer top-k from {path}: assignments={assignments}, "
                        f"scheduled_tokens={scheduled}"
                    )
                count_top_k.add(assignments // scheduled)

    inferred = captured_top_k | count_top_k
    if top_k is None:
        if len(inferred) != 1:
            raise ValueError(
                "Could not infer a unique top-k from the trace; pass --top-k"
            )
        top_k = inferred.pop()
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if captured_top_k and captured_top_k != {top_k}:
        raise ValueError(f"Trace contains captured top-k values {captured_top_k}")
    if count_top_k and count_top_k != {top_k}:
        raise ValueError(f"Trace count records imply top-k values {count_top_k}")

    token_steps: list[list[np.ndarray]] = []
    used_synthesis = False
    used_capture = False
    for step_index, raw_step in enumerate(raw_steps):
        source_rows = []
        for source in range(distribution.ep_size):
            expected_counts = demands[step_index, source]
            record = records.get((source, raw_step))
            if record is None:
                if int(expected_counts.sum()):
                    raise ValueError(
                        f"Missing trace record for rank={source}, step={raw_step}"
                    )
                source_rows.append(np.empty((0, top_k), dtype=np.int64))
                continue
            if "topk_ids" in record:
                ids = record["topk_ids"].to(torch.int64).numpy()
                if ids.shape[1] != top_k:
                    raise ValueError(
                        f"Captured top-k shape {ids.shape} does not match top_k={top_k}"
                    )
                if np.any((ids < 0) | (ids >= distribution.num_experts)):
                    raise ValueError("Captured top-k IDs are outside expert range")
                actual_counts = np.bincount(
                    ids.reshape(-1), minlength=distribution.num_experts
                ).astype(np.int64)
                used_capture = True
            else:
                counts = record["expert_counts"].to(torch.int64).numpy()
                ids = _synthesize_logical_topk_ids(counts, top_k)
                actual_counts = counts
                used_synthesis = True
            if not np.array_equal(actual_counts, expected_counts):
                raise ValueError(
                    f"Trace top-k rows do not match aggregated counts at "
                    f"rank={source}, step={raw_step}"
                )
            source_rows.append(ids)
        token_steps.append(source_rows)
    if used_capture and used_synthesis:
        provenance = "mixed_captured_and_synthesized"
    elif used_capture:
        provenance = "captured_token_topk_ids"
    else:
        provenance = "synthesized_from_exact_expert_counts"
    return token_steps, top_k, provenance


def _token_route_options(
    logical_ids: np.ndarray,
    replicas: list[set[int]],
    source: int,
) -> list[tuple[int, tuple[int, ...]]]:
    ep_size = max(
        source + 1,
        max((max(ranks, default=-1) for ranks in replicas), default=-1) + 1,
    )
    if not 0 <= source < ep_size:
        raise ValueError(f"source rank {source} is outside [0, {ep_size})")
    empty_counts = (0,) * ep_size
    routes: dict[tuple[int, tuple[int, ...]], tuple[int, ...]] = {(0, empty_counts): ()}
    for logical_id in np.asarray(logical_ids, dtype=np.int64).tolist():
        if not 0 <= logical_id < len(replicas):
            raise ValueError(f"Expert {logical_id} is outside the replica layout")
        allowed = sorted(
            replicas[int(logical_id)], key=lambda rank: (rank != source, rank)
        )
        if not allowed:
            raise ValueError(f"Expert {logical_id} has no replica")
        next_routes: dict[tuple[int, tuple[int, ...]], tuple[int, ...]] = {}
        for (mask, counts), targets in routes.items():
            for target in allowed:
                new_mask = mask | (1 << target)
                new_counts = list(counts)
                new_counts[target] += 1
                route_key = (new_mask, tuple(new_counts))
                candidate = targets + (target,)
                previous = next_routes.get(route_key)
                if previous is None or candidate < previous:
                    next_routes[route_key] = candidate
        routes = next_routes
    return sorted((mask, targets) for (mask, _), targets in routes.items())


def _route_targets_to_physical_topk_ids(
    logical_topk_ids: np.ndarray,
    target_ranks: np.ndarray,
    physical_expert_ids: np.ndarray,
) -> np.ndarray:
    logical_ids = np.asarray(logical_topk_ids, dtype=np.int64)
    targets = np.asarray(target_ranks, dtype=np.int64)
    if logical_ids.shape != targets.shape:
        raise ValueError("logical_topk_ids and target_ranks must have the same shape")
    if np.any(logical_ids < 0) or np.any(logical_ids >= physical_expert_ids.shape[0]):
        raise ValueError("logical_topk_ids contains an out-of-range expert")
    if np.any(targets < 0) or np.any(targets >= physical_expert_ids.shape[1]):
        raise ValueError("target_ranks contains an out-of-range rank")
    physical_ids = physical_expert_ids[logical_ids, targets]
    if np.any(physical_ids < 0):
        raise ValueError("A routed expert has no physical replica on its target rank")
    return physical_ids


def _route_token_step(
    token_ids_by_source: list[np.ndarray],
    replicas: list[set[int]],
    latency_model: LatencyModel,
    *,
    compute_weight: float,
    communication_weight: float,
    route_options_cache: dict[
        tuple[int, tuple[int, ...]], list[tuple[int, tuple[int, ...]]]
    ]
    | None = None,
) -> TokenStepRoutingResult:
    if not token_ids_by_source:
        raise ValueError("token_ids_by_source must contain at least one rank")
    ep_size = len(token_ids_by_source)
    num_experts = len(replicas)
    for expert_id, expert_replicas in enumerate(replicas):
        if not expert_replicas:
            raise ValueError(f"Expert {expert_id} has no replica")
        if any(not 0 <= rank < ep_size for rank in expert_replicas):
            raise ValueError(f"Expert {expert_id} has an out-of-range replica rank")
    top_k = None
    for ids in token_ids_by_source:
        if np.asarray(ids).ndim != 2:
            raise ValueError("Each token top-k array must have shape [token, top_k]")
        if top_k is None:
            top_k = int(np.asarray(ids).shape[1])
        elif np.asarray(ids).shape[1] != top_k:
            raise ValueError("All source ranks must use the same top-k")
    assert top_k is not None

    target_ranks = [
        np.empty((len(ids), top_k), dtype=np.int64) for ids in token_ids_by_source
    ]
    rank_assignment_loads = np.zeros(ep_size, dtype=np.int64)
    rank_token_loads = np.zeros(ep_size, dtype=np.int64)
    expert_rank_loads = np.zeros((num_experts, ep_size), dtype=np.int64)
    source_expert_rank_loads = np.zeros((ep_size, num_experts, ep_size), dtype=np.int64)
    outbound = np.zeros(ep_size, dtype=np.int64)
    inbound = np.zeros(ep_size, dtype=np.int64)
    local_tokens = 0
    remote_tokens = 0
    remote_token_transfers = 0
    remote_assignments = 0
    option_cache = route_options_cache if route_options_cache is not None else {}
    pending = []
    for source, ids in enumerate(token_ids_by_source):
        for token_index, row in enumerate(np.asarray(ids, dtype=np.int64)):
            key = (source, tuple(int(expert) for expert in row))
            options = option_cache.get(key)
            if options is None:
                options = _token_route_options(row, replicas, source)
                option_cache[key] = options
            pending.append((len(options), source, token_index, row, options))
    pending.sort(key=lambda item: (item[0], item[1], item[2]))

    for _, source, token_index, row, options in pending:
        source_bit = 1 << source
        best = None
        for mask, targets in options:
            remote_mask = mask & ~source_bit
            remote_token = int(bool(remote_mask))
            remote_transfers = remote_mask.bit_count()
            projected_assignment_loads = rank_assignment_loads.copy()
            for target in targets:
                projected_assignment_loads[target] += 1
            projected_token_loads = rank_token_loads.copy()
            for target in range(ep_size):
                if mask & (1 << target):
                    projected_token_loads[target] += 1
            projected_compute = latency_model.compute_ms(
                int(projected_assignment_loads.max(initial=0))
            )
            projected_remote_transfers = remote_token_transfers + remote_transfers
            projected_communication = latency_model.communication_ms(
                projected_remote_transfers
            )
            score = (
                compute_weight * projected_compute
                + communication_weight * projected_communication
            )
            key = (
                score,
                remote_token,
                remote_transfers,
                int(projected_assignment_loads.max(initial=0)),
                int(projected_token_loads.max(initial=0)),
                len(targets),
                targets,
            )
            if best is None or key < best[0]:
                best = (key, mask, targets, remote_token, remote_transfers)
        assert best is not None
        _, mask, targets, remote_token, remote_transfers = best
        target_ranks[source][token_index] = targets
        for target in range(ep_size):
            if mask & (1 << target):
                rank_token_loads[target] += 1
                if target != source:
                    inbound[target] += 1
        remote_mask = mask & ~source_bit
        outbound[source] += remote_transfers
        if remote_token:
            remote_tokens += 1
        else:
            local_tokens += 1
        remote_token_transfers += remote_transfers
        for expert_id, target in zip(row.tolist(), targets):
            rank_assignment_loads[target] += 1
            expert_rank_loads[expert_id, target] += 1
            source_expert_rank_loads[source, expert_id, target] += 1
            remote_assignments += int(target != source)

    communication_units = remote_token_transfers
    compute_latency_ms = latency_model.compute_ms(
        int(rank_assignment_loads.max(initial=0))
    )
    communication_latency_ms = latency_model.communication_ms(communication_units)
    objective = compute_weight * compute_latency_ms + communication_weight * (
        communication_latency_ms
    )
    return TokenStepRoutingResult(
        target_ranks_by_source=target_ranks,
        rank_assignment_loads=rank_assignment_loads,
        rank_token_loads=rank_token_loads,
        expert_rank_loads=expert_rank_loads,
        source_expert_rank_loads=source_expert_rank_loads,
        outbound_remote_tokens=outbound,
        inbound_remote_tokens=inbound,
        local_tokens=local_tokens,
        remote_tokens=remote_tokens,
        remote_token_transfers=remote_token_transfers,
        remote_assignments=remote_assignments,
        compute_latency_ms=compute_latency_ms,
        communication_latency_ms=communication_latency_ms,
        objective=objective,
    )


def _aggregate_token_steps(token_steps: list[list[np.ndarray]]) -> list[np.ndarray]:
    if not token_steps:
        raise ValueError("token_steps must not be empty")
    ep_size = len(token_steps[0])
    top_k = token_steps[0][0].shape[1] if ep_size else 0
    return [
        np.concatenate([step[source] for step in token_steps], axis=0)
        if any(len(step[source]) for step in token_steps)
        else np.empty((0, top_k), dtype=np.int64)
        for source in range(ep_size)
    ]


def _limit_token_steps(
    token_steps: list[list[np.ndarray]], max_tokens: int
) -> list[list[np.ndarray]]:
    """Select an evenly spaced token subset for replica-layout search."""
    if max_tokens <= 0:
        return token_steps
    segment_lengths = [len(rows) for step in token_steps for rows in step]
    total_tokens = sum(segment_lengths)
    if total_tokens <= max_tokens:
        return token_steps

    sample_positions = np.linspace(0, total_tokens - 1, max_tokens, dtype=np.int64)
    segment_ends = np.cumsum(segment_lengths, dtype=np.int64)
    segment_starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), segment_ends[:-1])
    )
    selected: dict[int, list[int]] = {}
    for position in sample_positions.tolist():
        segment = int(np.searchsorted(segment_ends, position, side="right"))
        selected.setdefault(segment, []).append(int(position - segment_starts[segment]))

    limited_steps: list[list[np.ndarray]] = []
    segment = 0
    for step in token_steps:
        limited_step = []
        for rows in step:
            rows_array = np.asarray(rows, dtype=np.int64)
            offsets = selected.get(segment)
            if offsets:
                limited_step.append(rows_array[offsets].copy())
            else:
                limited_step.append(np.empty((0, rows_array.shape[1]), dtype=np.int64))
            segment += 1
        limited_steps.append(limited_step)
    return limited_steps


def _evaluate_token_trace(
    token_steps: list[list[np.ndarray]],
    replicas: list[set[int]],
    latency_model: LatencyModel,
    *,
    compute_weight: float,
    communication_weight: float,
) -> TokenTraceRoutingResult:
    if not token_steps:
        raise ValueError("token_steps must not be empty")
    ep_size = len(token_steps[0])
    rank_assignment_loads = np.zeros(ep_size, dtype=np.int64)
    rank_token_loads = np.zeros(ep_size, dtype=np.int64)
    expert_rank_loads = np.zeros((len(replicas), ep_size), dtype=np.int64)
    remote_assignments = 0
    local_tokens = 0
    remote_tokens = 0
    remote_token_transfers = 0
    bottleneck_remote_token_transfers = 0
    compute_latency_ms = 0.0
    communication_latency_ms = 0.0
    overlap_lower_bound_ms = 0.0
    objective = 0.0
    imbalances = []
    route_options_cache: dict[
        tuple[int, tuple[int, ...]], list[tuple[int, tuple[int, ...]]]
    ] = {}
    for step in token_steps:
        routing = _route_token_step(
            step,
            replicas,
            latency_model,
            compute_weight=compute_weight,
            communication_weight=communication_weight,
            route_options_cache=route_options_cache,
        )
        rank_assignment_loads += routing.rank_assignment_loads
        rank_token_loads += routing.rank_token_loads
        expert_rank_loads += routing.expert_rank_loads
        remote_assignments += routing.remote_assignments
        local_tokens += routing.local_tokens
        remote_tokens += routing.remote_tokens
        remote_token_transfers += routing.remote_token_transfers
        bottleneck_remote_token_transfers += max(
            int(routing.outbound_remote_tokens.max(initial=0)),
            int(routing.inbound_remote_tokens.max(initial=0)),
        )
        compute_latency_ms += routing.compute_latency_ms
        communication_latency_ms += routing.communication_latency_ms
        overlap_lower_bound_ms += max(
            routing.compute_latency_ms, routing.communication_latency_ms
        )
        objective += routing.objective
        mean_load = float(routing.rank_assignment_loads.mean())
        imbalances.append(
            float(routing.rank_assignment_loads.max(initial=0)) / mean_load
            if mean_load
            else 0.0
        )
    imbalance_array = np.asarray(imbalances, dtype=np.float64)
    return TokenTraceRoutingResult(
        rank_assignment_loads=rank_assignment_loads,
        rank_token_loads=rank_token_loads,
        expert_rank_loads=expert_rank_loads,
        remote_assignments=remote_assignments,
        local_tokens=local_tokens,
        remote_tokens=remote_tokens,
        remote_token_transfers=remote_token_transfers,
        bottleneck_remote_token_transfers=bottleneck_remote_token_transfers,
        compute_latency_ms=compute_latency_ms,
        communication_latency_ms=communication_latency_ms,
        serial_latency_ms=compute_latency_ms + communication_latency_ms,
        overlap_lower_bound_ms=overlap_lower_bound_ms,
        mean_step_max_over_mean=float(imbalance_array.mean()),
        p95_step_max_over_mean=float(np.percentile(imbalance_array, 95)),
        max_step_max_over_mean=float(imbalance_array.max(initial=0.0)),
        objective=objective,
    )


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


def _candidate_token_replicas(
    aggregate_tokens: list[np.ndarray],
    replicas: list[set[int]],
    routing: TokenStepRoutingResult,
    candidate_limit: int,
) -> list[tuple[int, int]]:
    ep_size = len(aggregate_tokens)
    candidates = [
        (expert_id, rank)
        for expert_id in range(len(replicas))
        for rank in range(ep_size)
        if rank not in replicas[expert_id]
    ]
    if candidate_limit <= 0 or len(candidates) <= candidate_limit:
        return candidates

    pattern_counts: dict[tuple[int, tuple[int, ...]], int] = {}
    for source, rows in enumerate(aggregate_tokens):
        for row in rows:
            key = (source, tuple(int(expert_id) for expert_id in row))
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
    base_communication = {}
    for key in pattern_counts:
        source, expert_ids = key
        base_communication[key] = min(
            (mask & ~(1 << source)).bit_count()
            for mask, _ in _token_route_options(
                np.asarray(expert_ids, dtype=np.int64), replicas, source
            )
        )

    def scores(candidate: tuple[int, int]) -> tuple[int, int, int]:
        expert_id, rank = candidate
        candidate_replicas = _copy_replicas(replicas)
        candidate_replicas[expert_id].add(rank)
        communication_gain = 0
        fully_local_gain = 0
        occurrence = 0
        for (source, expert_ids), count in pattern_counts.items():
            if expert_id not in expert_ids:
                continue
            occurrence += count
            remote_mask = ~(1 << source)
            candidate_communication = min(
                (mask & remote_mask).bit_count()
                for mask, _ in _token_route_options(
                    np.asarray(expert_ids, dtype=np.int64),
                    candidate_replicas,
                    source,
                )
            )
            previous_communication = base_communication[(source, expert_ids)]
            communication_gain += (
                previous_communication - candidate_communication
            ) * count
            if previous_communication and not candidate_communication:
                fully_local_gain += count
        return communication_gain, fully_local_gain, occurrence

    communication_order = sorted(
        candidates,
        key=lambda item: tuple(-score for score in scores(item)) + (item,),
    )
    max_rank_load = int(routing.rank_assignment_loads.max(initial=0))

    def balance_score(candidate: tuple[int, int]) -> int:
        expert_id, rank = candidate
        headroom = max_rank_load - int(routing.rank_assignment_loads[rank])
        movable = int(routing.expert_rank_loads[expert_id].max(initial=0))
        return min(headroom, movable)

    balance_order = sorted(
        candidates,
        key=lambda item: (-balance_score(item), item),
    )
    shortlist = set(communication_order[:candidate_limit])
    shortlist.update(balance_order[:candidate_limit])
    return sorted(shortlist)


def _optimize_token_replicas(
    token_steps: list[list[np.ndarray]],
    base_replicas: list[set[int]],
    extra_replica_budget: int,
    latency_model: LatencyModel,
    *,
    compute_weight: float,
    communication_weight: float,
    candidate_limit: int,
    search_token_steps: list[list[np.ndarray]] | None = None,
) -> tuple[list[set[int]], list[tuple[int, int]]]:
    if extra_replica_budget < 0:
        raise ValueError("extra_replica_budget must be non-negative")
    optimization_steps = search_token_steps or token_steps
    aggregate_tokens = _aggregate_token_steps(optimization_steps)
    replicas = _copy_replicas(base_replicas)
    placements = []
    aggregate_routing = _route_token_step(
        aggregate_tokens,
        replicas,
        latency_model,
        compute_weight=compute_weight,
        communication_weight=communication_weight,
    )
    current = _evaluate_token_trace(
        optimization_steps,
        replicas,
        latency_model,
        compute_weight=compute_weight,
        communication_weight=communication_weight,
    )
    best_replicas = _copy_replicas(replicas)
    best_placements: list[tuple[int, int]] = []
    best_objective = current.objective
    for _ in range(extra_replica_budget):
        candidates = _candidate_token_replicas(
            aggregate_tokens,
            replicas,
            aggregate_routing,
            candidate_limit,
        )
        if not candidates:
            break
        best: tuple[tuple[float, int, int], int, int, TokenTraceRoutingResult] | None
        best = None
        for expert_id, rank in candidates:
            candidate_replicas = _copy_replicas(replicas)
            candidate_replicas[expert_id].add(rank)
            routing = _evaluate_token_trace(
                optimization_steps,
                candidate_replicas,
                latency_model,
                compute_weight=compute_weight,
                communication_weight=communication_weight,
            )
            key = (routing.objective, expert_id, rank)
            if best is None or key < best[0]:
                best = (key, expert_id, rank, routing)
        assert best is not None
        _, expert_id, rank, current = best
        replicas[expert_id].add(rank)
        placements.append((expert_id, rank))
        if current.objective < best_objective - 1e-12:
            best_replicas = _copy_replicas(replicas)
            best_placements = list(placements)
            best_objective = current.objective
        aggregate_routing = _route_token_step(
            aggregate_tokens,
            replicas,
            latency_model,
            compute_weight=compute_weight,
            communication_weight=communication_weight,
        )
    return best_replicas, best_placements


def _token_latency_lower_bounds(
    token_steps: list[list[np.ndarray]],
    replicas: list[set[int]],
    latency_model: LatencyModel,
) -> tuple[float, float]:
    ep_size = len(token_steps[0])
    balanced_compute_ms = 0.0
    communication_ms = 0.0
    for step in token_steps:
        total_assignments = 0
        min_communication_units = 0
        for source, rows in enumerate(step):
            source_bit = 1 << source
            for row in rows:
                total_assignments += len(row)
                options = _token_route_options(row, replicas, source)
                min_communication_units += min(
                    (mask & ~source_bit).bit_count() for mask, _ in options
                )
        balanced_compute_ms += latency_model.compute_ms(
            ceil(total_assignments / ep_size)
        )
        communication_ms += latency_model.communication_ms(min_communication_units)
    return balanced_compute_ms, communication_ms


def _token_result_payload(
    *,
    policy: str,
    communication_weight: float | None,
    requested_budget: int,
    placements: list[tuple[int, int]],
    token_steps: list[list[np.ndarray]],
    replicas: list[set[int]],
    latency_model: LatencyModel,
    routing: TokenTraceRoutingResult,
    top_k: int,
    topk_reconstruction: str,
) -> dict[str, Any]:
    total_tokens = sum(len(rows) for step in token_steps for rows in step)
    total_assignments = total_tokens * top_k
    mean_rank_assignments = float(routing.rank_assignment_loads.mean())
    mean_rank_tokens = float(routing.rank_token_loads.mean())
    balanced_compute_lower_bound_ms, communication_lower_bound_ms = (
        _token_latency_lower_bounds(token_steps, replicas, latency_model)
    )
    return {
        "policy": policy,
        "communication_weight": communication_weight,
        "requested_extra_replicas": requested_budget,
        "used_extra_replicas": len(placements),
        "replica_placements": [
            {"expert_id": expert_id, "rank": rank} for expert_id, rank in placements
        ],
        "top_k": top_k,
        "topk_reconstruction": topk_reconstruction,
        "total_tokens": total_tokens,
        "total_assignments": total_assignments,
        "local_tokens": routing.local_tokens,
        "local_token_percent": (
            routing.local_tokens / total_tokens * 100.0 if total_tokens else 0.0
        ),
        "remote_tokens": routing.remote_tokens,
        "remote_token_percent": (
            routing.remote_tokens / total_tokens * 100.0 if total_tokens else 0.0
        ),
        "remote_token_transfers": routing.remote_token_transfers,
        "communication_units": routing.remote_token_transfers,
        "remote_assignments": routing.remote_assignments,
        "remote_assignment_percent": (
            routing.remote_assignments / total_assignments * 100.0
            if total_assignments
            else 0.0
        ),
        "bottleneck_remote_token_transfers": (
            routing.bottleneck_remote_token_transfers
        ),
        "rank_assignment_loads": routing.rank_assignment_loads.tolist(),
        "rank_assignment_load_unit": "expert_input_tokens_per_target_rank",
        "max_rank_assignments": int(routing.rank_assignment_loads.max(initial=0)),
        "mean_rank_assignments": mean_rank_assignments,
        "rank_assignment_max_over_mean": (
            float(routing.rank_assignment_loads.max(initial=0)) / mean_rank_assignments
            if mean_rank_assignments
            else 0.0
        ),
        "rank_token_loads": routing.rank_token_loads.tolist(),
        "rank_token_load_unit": "unique_tokens_per_target_rank",
        "rank_unique_token_loads": routing.rank_token_loads.tolist(),
        "max_rank_tokens": int(routing.rank_token_loads.max(initial=0)),
        "mean_rank_tokens": mean_rank_tokens,
        "rank_token_max_over_mean": (
            float(routing.rank_token_loads.max(initial=0)) / mean_rank_tokens
            if mean_rank_tokens
            else 0.0
        ),
        "mean_step_max_over_mean": routing.mean_step_max_over_mean,
        "p95_step_max_over_mean": routing.p95_step_max_over_mean,
        "max_step_max_over_mean": routing.max_step_max_over_mean,
        "estimated_compute_latency_ms": routing.compute_latency_ms,
        "estimated_communication_latency_ms": routing.communication_latency_ms,
        "estimated_serial_latency_ms": routing.serial_latency_ms,
        "estimated_overlap_lower_bound_ms": routing.overlap_lower_bound_ms,
        "balanced_compute_lower_bound_ms": balanced_compute_lower_bound_ms,
        "communication_lower_bound_ms": communication_lower_bound_ms,
        "optimization_objective": routing.objective,
        "pareto_optimal": False,
    }


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
    experiment_dir: Path,
    top_k: int | None,
    candidate_limit: int,
    search_max_tokens: int,
) -> list[dict[str, Any]]:
    token_steps, inferred_top_k, topk_reconstruction = _load_token_trace(
        experiment_dir,
        distribution,
        layer_id,
        top_k,
    )
    top_k = inferred_top_k
    search_token_steps = _limit_token_steps(token_steps, search_max_tokens)
    search_tokens = sum(len(rows) for step in search_token_steps for rows in step)
    total_tokens = sum(len(rows) for step in token_steps for rows in step)
    if search_tokens < total_tokens:
        print(
            f"  layout search uses {search_tokens}/{total_tokens} tokens",
            flush=True,
        )
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

    baseline_routing = _evaluate_token_trace(
        token_steps,
        base_replicas,
        latency_model,
        compute_weight=1.0,
        communication_weight=1.0,
    )
    points = [
        _token_result_payload(
            policy="baseline",
            communication_weight=None,
            requested_budget=0,
            placements=[],
            token_steps=token_steps,
            replicas=base_replicas,
            latency_model=latency_model,
            routing=baseline_routing,
            top_k=top_k,
            topk_reconstruction=topk_reconstruction,
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
            print(
                f"  searching policy={policy}, budget={budget}, "
                f"communication_weight={label_weight}",
                flush=True,
            )
            replicas, placements = _optimize_token_replicas(
                token_steps,
                base_replicas,
                budget,
                latency_model,
                compute_weight=compute_weight,
                communication_weight=communication_weight,
                candidate_limit=candidate_limit,
                search_token_steps=search_token_steps,
            )
            routing = _evaluate_token_trace(
                token_steps,
                replicas,
                latency_model,
                compute_weight=compute_weight,
                communication_weight=communication_weight,
            )
            points.append(
                _token_result_payload(
                    policy=policy,
                    communication_weight=label_weight,
                    requested_budget=budget,
                    placements=placements,
                    token_steps=token_steps,
                    replicas=replicas,
                    latency_model=latency_model,
                    routing=routing,
                    top_k=top_k,
                    topk_reconstruction=topk_reconstruction,
                )
            )
    _mark_pareto_points(points)
    return points


def _simulate_layer_legacy(
    distribution: TraceDistribution,
    layer_id: int,
    replica_budgets: list[int],
    communication_weights: list[float],
    latency_model: LatencyModel,
    *,
    routing_chunks: int,
    candidate_limit: int,
) -> list[dict[str, Any]]:
    """Keep the previous count-only simulation for old callers."""
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
        requested = point.get("requested_extra_replicas", point["used_extra_replicas"])
        label = f"B={requested}"
        if point["used_extra_replicas"] != requested:
            label += f", used={point['used_extra_replicas']}"
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
            row["rank_token_loads"] = json.dumps(row["rank_token_loads"])
            row["rank_unique_token_loads"] = json.dumps(row["rank_unique_token_loads"])
            rows.append(row)
    if not rows:
        return
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate(args: argparse.Namespace) -> None:
    if args.compute_us_per_token <= 0:
        raise ValueError("--compute-us-per-token must be positive")
    if args.communication_us_per_token <= 0:
        raise ValueError("--communication-us-per-token must be positive")
    if args.routing_chunks <= 0:
        raise ValueError("--routing-chunks must be positive")
    if args.candidate_limit < 0:
        raise ValueError("--candidate-limit must be non-negative")
    if args.search_max_tokens < 0:
        raise ValueError("--search-max-tokens must be non-negative")
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
        compute_us_per_token=args.compute_us_per_token,
        communication_us_per_token=args.communication_us_per_token,
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
                    experiment_dir=_experiment_dir(work_dir, dataset, batch_size),
                    top_k=args.top_k,
                    candidate_limit=args.candidate_limit,
                    search_max_tokens=args.search_max_tokens,
                )
                top_k_values = {int(point["top_k"]) for point in points}
                experiment = {
                    "dataset": dataset,
                    "batch_size_per_rank": batch_size,
                    "layer_id": layer_id,
                    "num_experts": distribution.num_experts,
                    "ep_size": distribution.ep_size,
                    "expert_placement_strategy": (
                        distribution.expert_placement_strategy
                    ),
                    "top_k": top_k_values.pop(),
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
            "kind": "linear_token_proxy",
            "compute_unit": "token_expert_assignment",
            "communication_unit": "remote_target_rank_transfer",
            "communication_includes": "dispatch_and_combine",
            "warning": (
                "Count-only traces synthesize token top-k rows and cannot preserve "
                "the original token co-occurrence. Calibrate both coefficients "
                "before interpreting values as hardware latency."
            ),
        },
        "routing_chunks": args.routing_chunks,
        "candidate_limit": args.candidate_limit,
        "search_max_tokens": args.search_max_tokens,
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
        "--top-k",
        type=int,
        help="Top-k routing width; inferred from captured trace counts when omitted.",
    )
    parser.add_argument(
        "--compute-us-per-token",
        "--compute-us-per-assignment",
        dest="compute_us_per_token",
        type=float,
        default=1.0,
        help=(
            "Calibrated compute cost per token-expert assignment; the default "
            "is a proxy."
        ),
    )
    parser.add_argument(
        "--communication-us-per-token",
        "--communication-us-per-assignment",
        dest="communication_us_per_token",
        type=float,
        default=1.0,
        help=(
            "Calibrated communication cost per remote token unit; dispatch and "
            "combine are both included."
        ),
    )
    parser.add_argument(
        "--routing-chunks",
        type=int,
        default=8,
        help="Deprecated compatibility option; token routing no longer chunks counts.",
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
    parser.add_argument(
        "--search-max-tokens",
        type=int,
        default=0,
        help=(
            "Maximum tokens used while searching replica layouts. The final "
            "metrics still replay the complete trace; 0 uses all tokens."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    simulate(parse_args())


if __name__ == "__main__":
    main()
