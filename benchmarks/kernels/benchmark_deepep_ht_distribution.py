# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DeepEP-HT with diverse synthetic token-routing scenarios.

Run with torchrun. The benchmark only constructs physical top-k expert IDs;
dispatch, expert compute, and combine are the real DeepEP/MoE callbacks::

    .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
        benchmarks/kernels/benchmark_deepep_ht_distribution.py \
        --tokens-per-rank 4096,3072,2048,1024 \
        --local-shares 0.9,0.1 \
        --no-detail-profile --output-jsonl /tmp/deepep_ht_compare.jsonl

Run the controlled compute-uniformity by communication-uniformity experiment::

    .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
        benchmarks/kernels/benchmark_deepep_ht_distribution.py \
        --benchmarks joint --joint-route-shares L1F0:0.1,L1F1:0.4,L0F2:0.5 \
        --num-experts 128 --top-k 8 --no-detail-profile

This produces four routes: balanced/balanced, skewed/balanced,
balanced/hotspot, and skewed/hotspot. The same physical top-k IDs determine
both the expert-token load and the unique source-to-destination token transfers.
All routes use the same local-presence/remote-fan-out state distribution,
keeping remote transfer volume fixed while uniformity changes. ``L1F2`` means
the token uses local experts and two remote ranks; ``L0F2`` is remote-only.
If every remote token has full fan-out, communication is saturated and the two
hotspot routes are omitted because they would be identical to the balanced
routes.

The ``compute_ms`` column is the maximum expert-kernel time across ranks, and
``communication_ms`` is the maximum dispatch-plus-combine time across ranks.
``planned_load_max_over_mean`` reports the generated expert-assignment skew.
Communication locality is token-level: a token is fully local only when all of
its top-k experts are on its source rank. A token spanning multiple remote ranks
is transferred once to every such rank.

Use ``--no-detail-profile`` for wrapper timings without the diagnostic device
synchronizations inserted inside DeepEP prepare/finalize.

Use ``--model`` to select the MoE dimensions of a supported model without
loading its weights::

    --model Qwen/Qwen3-30B-A3B
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
    DeepEPHTPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.worker.workspace import init_workspace_manager

DEFAULT_MOE_CONFIG = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_experts": 8,
    "top_k": 1,
}

MODEL_CONFIGS = {
    "Qwen/Qwen3-30B-A3B": {
        "hidden_size": 2048,
        "intermediate_size": 768,
        "num_experts": 128,
        "top_k": 8,
    },
    "Qwen/Qwen3-235B-A22B": {
        "hidden_size": 4096,
        "intermediate_size": 1536,
        "num_experts": 128,
        "top_k": 8,
    },
}

COMPUTE_PATTERNS = ("balanced", "single_hot", "multi_hot", "zipf", "random")
COMMUNICATION_PATTERNS = (
    "coalesced",
    "spread",
    "hotspot",
    "partial",
    "asymmetric",
    "random",
)


@dataclass(frozen=True)
class LoadCase:
    name: str
    kind: Literal["compute", "communication", "joint"]
    pattern: str
    variant: str
    control_value: float
    source_target_weights: tuple[tuple[float, ...], ...]
    source_local_token_shares: tuple[float, ...] | None = None
    compute_pattern: str | None = None
    communication_pattern: str | None = None
    joint_route_shares: tuple[tuple[int, int, float], ...] | None = None
    hot_rank: int = 0
    seed: int | None = None
    planned_source_target_shares: tuple[tuple[float, ...], ...] = ()
    planned_target_assignments: tuple[int, ...] = ()
    planned_target_shares: tuple[float, ...] = ()
    planned_expert_assignments_by_rank: tuple[tuple[int, ...], ...] = ()
    planned_load_max_over_mean: float = 0.0
    planned_expert_load_max_over_mean: float = 0.0
    planned_destination_comm_max_over_mean: float = 0.0
    planned_link_comm_max_over_mean: float = 0.0
    planned_remote_transfer_matrix: tuple[tuple[int, ...], ...] = ()
    planned_fanout_counts: tuple[int, ...] = ()
    planned_fanout_shares: tuple[float, ...] = ()
    planned_route_state_counts: tuple[tuple[int, ...], ...] = ()
    planned_route_state_shares: tuple[tuple[float, ...], ...] = ()
    planned_local_share_min: float = 0.0
    planned_local_share_max: float = 0.0
    planned_remote_fanout: float = 0.0
    routing_fingerprint: str = ""


@dataclass(frozen=True)
class RoutingStats:
    target_assignments: list[int]
    target_unique_tokens: list[int]
    remote_assignments: int
    fully_local_tokens: int
    remote_tokens: int
    remote_token_transfers: int
    fanout_counts: list[int]
    route_state_counts: list[list[int]]


def apply_model_config(args: argparse.Namespace) -> argparse.Namespace:
    config = MODEL_CONFIGS.get(args.model, DEFAULT_MOE_CONFIG)
    for name, value in config.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepEP-HT rank-distribution dispatch/combine benchmark."
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CONFIGS),
        help=(
            "Use a built-in synthetic MoE shape preset. This does not load model "
            "weights. Explicit shape arguments override the preset."
        ),
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=4096,
        help="Tokens on every source rank unless --tokens-per-rank is set.",
    )
    parser.add_argument(
        "--tokens-per-rank",
        help=(
            "Comma-separated token counts in global-rank order. The number of "
            "entries must equal WORLD_SIZE."
        ),
    )
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument(
        "--intermediate-size",
        type=int,
        help="Per-routed-expert MoE intermediate size.",
    )
    parser.add_argument("--num-experts", type=int)
    parser.add_argument(
        "--top-k",
        type=int,
        help=(
            "A token is transferred once to each unique remote rank containing "
            "one or more of its top-k experts."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=("compute", "communication", "joint"),
        default=["compute", "communication"],
        help=(
            "Token-load dimensions to measure. joint runs the controlled 2x2 "
            "compute-uniformity by communication-uniformity experiment."
        ),
    )
    parser.add_argument(
        "--compute-patterns",
        nargs="+",
        choices=COMPUTE_PATTERNS,
        default=list(COMPUTE_PATTERNS),
        help="Compute-load shapes to measure.",
    )
    parser.add_argument(
        "--communication-patterns",
        nargs="+",
        choices=COMMUNICATION_PATTERNS,
        default=list(COMMUNICATION_PATTERNS),
        help="Remote fan-out and destination topologies to measure.",
    )
    parser.add_argument("--hot-rank", type=int, default=0)
    parser.add_argument(
        "--hot-shares",
        default=None,
        help=(
            "Optional comma-separated single-hot strengths. By default, "
            "--imbalance-strength is used."
        ),
    )
    parser.add_argument("--imbalance-strength", type=float, default=0.75)
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--random-cases", type=int, default=3)
    parser.add_argument("--random-alpha", type=float, default=0.5)
    parser.add_argument(
        "--local-shares",
        default="0.9,0.1",
        help=(
            "Comma-separated fractions of tokens whose complete top-k stays on "
            "the source rank. Include values above and below 0.5 to compare "
            "mostly-local and mostly-remote communication."
        ),
    )
    parser.add_argument(
        "--joint-route-shares",
        action="append",
        metavar="LxFy:SHARE,...",
        help=(
            "Local-presence/remote-fan-out distribution for --benchmarks joint, "
            "for example L1F0:0.1,L1F1:0.4,L0F2:0.5. L1 includes the source "
            "rank, L0 is remote-only, and F is the number of remote ranks. "
            "Shares must sum to one. Repeat to benchmark multiple distributions."
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--trim-ratio",
        type=float,
        default=0.1,
        help="Fraction removed from each end after sorting by max total time.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stage-barrier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align all ranks before every measured stage.",
    )
    parser.add_argument(
        "--detail-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable VLLM_DEEPEP_HT_PROFILE for layout/exchange/postprocess "
            "breakdowns. Disable it for uninstrumented baseline timings."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=("full_deepep", "local_bypass"),
        default="full_deepep",
        help=(
            "Use full_deepep for the production path, or local_bypass to run "
            "fully-local tokens through a local MoE kernel and send only the "
            "remaining tokens through DeepEP."
        ),
    )
    parser.add_argument(
        "--validate-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare local_bypass output against full_deepep before timing.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Write rank, aggregate, and summary records on rank 0.",
    )
    return apply_model_config(parser.parse_args())


def parse_shares(raw: str, option: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{option} must contain at least one value")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"{option} values must be in [0, 1]")
    return values


def parse_route_shares(
    raw: str,
    world_size: int,
) -> tuple[tuple[int, int, float], ...]:
    shares: dict[tuple[int, int], float] = {}
    seen = set()
    for entry in raw.split(","):
        parts = entry.strip().split(":")
        if len(parts) != 2:
            raise ValueError("--joint-route-shares entries must use LxFy:SHARE syntax")
        state = parts[0].upper()
        if len(state) < 4 or state[0] != "L" or "F" not in state:
            raise ValueError("--joint-route-shares state must use LxFy syntax")
        local_raw, fanout_raw = state[1:].split("F", maxsplit=1)
        if local_raw not in ("0", "1") or not fanout_raw.isdigit():
            raise ValueError("--joint-route-shares state must use LxFy syntax")
        local_present = int(local_raw)
        fanout = int(fanout_raw)
        share = float(parts[1])
        if not 0 <= fanout < world_size:
            raise ValueError(
                "--joint-route-shares fan-out must be between 0 and WORLD_SIZE - 1"
            )
        if local_present == 0 and fanout == 0:
            raise ValueError("L0F0 is invalid because it selects no expert rank")
        route_state = (local_present, fanout)
        if route_state in seen:
            raise ValueError(f"--joint-route-shares contains {state} more than once")
        if not 0.0 <= share <= 1.0:
            raise ValueError("--joint-route-shares shares must be in [0, 1]")
        seen.add(route_state)
        shares[route_state] = share
    if not seen or not math.isclose(sum(shares.values()), 1.0, abs_tol=1e-9):
        raise ValueError("--joint-route-shares shares must sum to 1")
    return tuple(
        (local_present, fanout, shares[local_present, fanout])
        for local_present, fanout in sorted(shares)
    )


def parse_token_counts(raw: str | None, fallback: int, world_size: int) -> list[int]:
    if raw is None:
        return [fallback] * world_size
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != world_size:
        raise ValueError(
            "--tokens-per-rank must contain exactly WORLD_SIZE values; "
            f"expected {world_size}, got {len(values)}"
        )
    if any(value <= 0 for value in values):
        raise ValueError("--tokens-per-rank values must be positive")
    return values


def sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def align_stage(device: torch.device, use_barrier: bool) -> None:
    sync(device)
    if use_barrier:
        dist.barrier()
        sync(device)


def time_stage(
    device: torch.device,
    fn: Callable[[], Any],
    *,
    use_barrier: bool,
) -> tuple[Any, float]:
    """Measure a real device callback on the distributed critical path."""
    align_stage(device, use_barrier)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return result, (time.perf_counter() - start) * 1000.0


def apportion(total: int, weights: list[float], offset: int) -> list[int]:
    return apportion_with_cap(total, weights, [total] * len(weights), offset)


def apportion_with_cap(
    total: int,
    weights: list[float],
    capacities: list[int],
    offset: int,
) -> list[int]:
    if len(weights) != len(capacities) or not weights:
        raise ValueError("Weights and capacities must have the same non-zero length")
    if total < 0 or any(weight < 0 for weight in weights):
        raise ValueError("Total and weights must be non-negative")
    if sum(capacities) < total:
        raise ValueError(f"Capacity {sum(capacities)} cannot hold {total} items")

    counts = [0] * len(weights)
    active = {index for index, capacity in enumerate(capacities) if capacity > 0}
    remaining = total
    while remaining:
        active_weights = {index: weights[index] for index in active}
        if not any(active_weights.values()):
            active_weights = {
                index: capacities[index] - counts[index] for index in active
            }
        weight_sum = sum(active_weights.values())
        capped = [
            index
            for index in active
            if remaining * active_weights[index] / weight_sum
            >= capacities[index] - counts[index]
        ]
        if capped:
            for index in capped:
                addition = capacities[index] - counts[index]
                counts[index] += addition
                remaining -= addition
                active.remove(index)
            continue

        raw = {
            index: remaining * active_weights[index] / weight_sum for index in active
        }
        additions = {index: int(value) for index, value in raw.items()}
        remainder = remaining - sum(additions.values())
        order = sorted(
            active,
            key=lambda index: (
                raw[index] - additions[index],
                -((index - offset) % len(weights)),
            ),
            reverse=True,
        )
        for index in order[:remainder]:
            additions[index] += 1
        for index, addition in additions.items():
            counts[index] += addition
        remaining = 0
    return counts


def _make_weighted_topk_ids(
    *,
    tokens: int,
    top_k: int,
    num_experts: int,
    source_rank: int,
    world_size: int,
    target_weights: tuple[float, ...],
) -> torch.Tensor:
    experts_per_rank = num_experts // world_size
    per_rank_capacity = tokens * min(top_k, experts_per_rank)
    rank_counts = apportion_with_cap(
        tokens * top_k,
        list(target_weights),
        [per_rank_capacity] * world_size,
        source_rank,
    )
    expert_counts: list[int] = []
    for target_rank, rank_count in enumerate(rank_counts):
        expert_counts.extend(
            apportion_with_cap(
                rank_count,
                [1.0] * experts_per_rank,
                [tokens] * experts_per_rank,
                source_rank * experts_per_rank + target_rank,
            )
        )

    heap = [
        (-count, (expert - source_rank * top_k) % num_experts, expert)
        for expert, count in enumerate(expert_counts)
        if count
    ]
    heapq.heapify(heap)
    rows = []
    for _ in range(tokens):
        selected = []
        for _ in range(top_k):
            if not heap:
                raise ValueError("Unable to assign unique top-k experts per token")
            negative_count, tie_break, expert = heapq.heappop(heap)
            selected.append((negative_count + 1, tie_break, expert))
        rows.append([expert for _, _, expert in selected])
        for negative_count, tie_break, expert in selected:
            if negative_count:
                heapq.heappush(heap, (negative_count, tie_break, expert))
    if heap:
        raise RuntimeError("Top-k construction left unassigned expert slots")
    return torch.tensor(rows, dtype=torch.int64)


def _ordered_remote_ranks(
    source_rank: int,
    world_size: int,
    weights: tuple[float, ...],
) -> list[int]:
    return sorted(
        (rank for rank in range(world_size) if rank != source_rank),
        key=lambda rank: (
            -weights[rank],
            (rank - source_rank) % world_size,
        ),
    )


def _coalesced_targets(
    order: list[int],
    count: int,
    experts_per_rank: int,
) -> list[int]:
    targets = []
    for rank in order:
        targets.extend([rank] * min(experts_per_rank, count - len(targets)))
        if len(targets) == count:
            return targets
    raise ValueError("Not enough experts to construct coalesced top-k targets")


def _spread_targets(
    order: list[int],
    count: int,
    experts_per_rank: int,
) -> list[int]:
    targets = []
    for _ in range(experts_per_rank):
        for rank in order:
            targets.append(rank)
            if len(targets) == count:
                return targets
    raise ValueError("Not enough experts to construct spread top-k targets")


def _random_targets(
    rng: random.Random,
    source_rank: int,
    top_k: int,
    experts_per_rank: int,
    weights: tuple[float, ...],
) -> list[int]:
    capacities = [experts_per_rank] * len(weights)
    remote = [rank for rank in range(len(weights)) if rank != source_rank]
    targets = []
    for slot in range(top_k):
        candidates = (
            remote
            if slot == 0
            else [rank for rank, capacity in enumerate(capacities) if capacity]
        )
        candidates = [rank for rank in candidates if capacities[rank]]
        candidate_weights = [weights[rank] for rank in candidates]
        if not any(candidate_weights):
            candidate_weights = [1.0] * len(candidates)
        target = rng.choices(candidates, weights=candidate_weights, k=1)[0]
        targets.append(target)
        capacities[target] -= 1
    return targets


def _make_communication_topk_ids(
    *,
    case: LoadCase,
    tokens: int,
    top_k: int,
    num_experts: int,
    source_rank: int,
    world_size: int,
) -> torch.Tensor:
    experts_per_rank = num_experts // world_size
    assert case.source_local_token_shares is not None
    local_share = case.source_local_token_shares[source_rank]
    local_tokens = apportion(tokens, [local_share, 1.0 - local_share], source_rank)[0]
    if local_tokens and top_k > experts_per_rank:
        raise ValueError(
            "A fully-local token requires --top-k <= experts per rank; "
            f"got top_k={top_k}, experts_per_rank={experts_per_rank}"
        )

    rng = random.Random((case.seed or 0) + source_rank * 104729)
    token_order = list(range(tokens))
    rng.shuffle(token_order)
    local_token_indices = set(token_order[:local_tokens])
    weights = case.source_target_weights[source_rank]
    remote_order = _ordered_remote_ranks(source_rank, world_size, weights)
    all_order = [*remote_order, source_rank]
    seen_per_rank = [0] * world_size
    rows = []
    for token in range(tokens):
        if token in local_token_indices:
            targets = [source_rank] * top_k
        elif case.pattern in ("coalesced", "hotspot"):
            targets = _coalesced_targets(all_order, top_k, experts_per_rank)
        elif case.pattern == "spread":
            targets = _spread_targets(all_order, top_k, experts_per_rank)
        elif case.pattern == "partial":
            local_slots = min(experts_per_rank, max(1, top_k // 2))
            if top_k == 1:
                local_slots = 0
            targets = [source_rank] * local_slots
            targets.extend(
                _spread_targets(
                    remote_order,
                    top_k - local_slots,
                    experts_per_rank,
                )
            )
        elif case.pattern == "asymmetric":
            make_targets = (
                _coalesced_targets if source_rank % 2 == 0 else _spread_targets
            )
            targets = make_targets(all_order, top_k, experts_per_rank)
        elif case.pattern == "random":
            targets = _random_targets(
                rng,
                source_rank,
                top_k,
                experts_per_rank,
                weights,
            )
        else:
            raise ValueError(f"Unknown communication pattern: {case.pattern}")

        row = []
        rank_occurrences: dict[int, int] = defaultdict(int)
        for target in targets:
            local_expert = (
                seen_per_rank[target] + rank_occurrences[target]
            ) % experts_per_rank
            row.append(target * experts_per_rank + local_expert)
            rank_occurrences[target] += 1
        for target, occurrences in rank_occurrences.items():
            seen_per_rank[target] += occurrences
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int64)


def _joint_target_counts(
    *,
    case: LoadCase,
    source_rank: int,
    global_token: int,
    local_present: bool,
    fanout: int,
    top_k: int,
    world_size: int,
    experts_per_rank: int,
) -> list[int]:
    compute_pattern = case.compute_pattern
    communication_pattern = case.communication_pattern
    if compute_pattern not in ("balanced", "skewed"):
        raise ValueError(f"Unknown joint compute pattern: {compute_pattern}")
    if communication_pattern not in ("balanced", "hotspot", "saturated"):
        raise ValueError(
            f"Unknown joint communication pattern: {communication_pattern}"
        )

    hot_rank = case.hot_rank % world_size
    remote_ranks = [rank for rank in range(world_size) if rank != source_rank]
    if not 1 <= fanout <= world_size - 1:
        raise ValueError("fan-out must be between 1 and WORLD_SIZE - 1")

    def rotate_targets(ranks: list[int], count: int) -> list[int]:
        start = global_token % len(ranks)
        return [ranks[(start + offset) % len(ranks)] for offset in range(count)]

    if communication_pattern in ("balanced", "saturated"):
        remote_targets = rotate_targets(remote_ranks, fanout)
    elif source_rank != hot_rank:
        cold_ranks = [rank for rank in remote_ranks if rank != hot_rank]
        remote_targets = [hot_rank, *rotate_targets(cold_ranks, fanout - 1)]
    else:
        cold_ranks = remote_ranks
        remote_targets = rotate_targets(cold_ranks, fanout)
    targets = [*([source_rank] if local_present else []), *remote_targets]

    def distribute(total: int, ranks: list[int]) -> list[int]:
        counts = [0] * world_size
        base, remainder = divmod(total, len(ranks))
        for offset, rank in enumerate(ranks):
            counts[rank] = base + (offset < remainder)
        return counts

    counts = [0] * world_size
    if (
        compute_pattern == "balanced"
        and local_present
        and communication_pattern
        in (
            "balanced",
            "saturated",
        )
    ):
        counts[source_rank] = top_k - fanout
        for rank in remote_targets:
            counts[rank] += 1
    elif compute_pattern == "balanced" and communication_pattern in (
        "balanced",
        "saturated",
    ):
        counts = distribute(top_k, remote_targets)
    elif compute_pattern == "balanced" and local_present:
        if source_rank == hot_rank:
            counts[hot_rank] = top_k - (world_size - 1)
            remote_counts = distribute(world_size - 1, remote_targets)
            for rank, count in enumerate(remote_counts):
                counts[rank] += count
        else:
            counts[source_rank] = top_k - fanout
            for rank in remote_targets:
                counts[rank] += 1
    elif compute_pattern == "balanced":
        if source_rank == hot_rank:
            counts = distribute(top_k, remote_targets)
        else:
            hot_slots = top_k // (world_size - 1)
            if global_token % (world_size - 1) < top_k % (world_size - 1):
                hot_slots += 1
            counts[hot_rank] = hot_slots
            other_targets = [rank for rank in remote_targets if rank != hot_rank]
            distributed = distribute(top_k - hot_slots, other_targets)
            for rank, count in enumerate(distributed):
                counts[rank] += count
    else:
        requested_hot_slots = max(
            top_k // 2 + 1,
            round(top_k * case.control_value),
        )
        other_target_count = len(targets) - int(hot_rank in targets)
        requested_hot_slots = min(
            requested_hot_slots,
            top_k - other_target_count,
        )
        if communication_pattern in ("balanced", "saturated"):
            if hot_rank in targets:
                counts[hot_rank] = requested_hot_slots
                other_targets = [rank for rank in targets if rank != hot_rank]
                distributed = distribute(top_k - requested_hot_slots, other_targets)
                for rank, count in enumerate(distributed):
                    counts[rank] += count
            else:
                counts = distribute(top_k, targets)
        elif local_present:
            numerator = (fanout + 1) * requested_hot_slots
            hot_slots = numerator // world_size
            if global_token % world_size < numerator % world_size:
                hot_slots += 1
            counts[hot_rank] = hot_slots
            other_targets = [rank for rank in targets if rank != hot_rank]
            distributed = distribute(top_k - hot_slots, other_targets)
            for rank, count in enumerate(distributed):
                counts[rank] += count
        elif source_rank == hot_rank:
            counts = distribute(top_k, remote_targets)
        else:
            numerator = fanout * requested_hot_slots
            hot_slots = numerator // (world_size - 1)
            if global_token % (world_size - 1) < numerator % (world_size - 1):
                hot_slots += 1
            counts[hot_rank] = hot_slots
            other_targets = [rank for rank in remote_targets if rank != hot_rank]
            distributed = distribute(top_k - hot_slots, other_targets)
            for rank, count in enumerate(distributed):
                counts[rank] += count

    if any(counts[rank] <= 0 for rank in targets):
        raise RuntimeError(
            f"joint route dropped a selected destination: targets={targets}, "
            f"counts={counts}"
        )
    if max(counts) > experts_per_rank:
        raise ValueError(
            "joint cases require every token's per-rank top-k count to fit the "
            f"physical experts; counts={counts}, experts_per_rank={experts_per_rank}"
        )
    return counts


def _make_joint_topk_ids(
    *,
    case: LoadCase,
    tokens: int,
    top_k: int,
    num_experts: int,
    source_rank: int,
    world_size: int,
) -> torch.Tensor:
    experts_per_rank = num_experts // world_size
    if case.joint_route_shares is None:
        raise ValueError("joint case is missing its route-state distribution")
    route_counts = apportion(
        tokens,
        [share for _, _, share in case.joint_route_shares],
        0,
    )
    has_fully_local_tokens = any(
        local_present and fanout == 0 and count
        for (local_present, fanout, _), count in zip(
            case.joint_route_shares,
            route_counts,
        )
    )
    if has_fully_local_tokens and top_k > experts_per_rank:
        raise ValueError(
            "joint fully-local tokens require top_k <= experts_per_rank; "
            f"got top_k={top_k}, experts_per_rank={experts_per_rank}"
        )
    rng = random.Random((case.seed or 0) + source_rank * 104729)
    token_states = [
        (local_present, fanout, state_index)
        for state_index, ((local_present, fanout, _), count) in enumerate(
            zip(case.joint_route_shares, route_counts)
        )
        for _ in range(count)
    ]
    rng.shuffle(token_states)
    seen_per_state = [0] * len(case.joint_route_shares)
    seen_per_rank = [0] * world_size
    rows = []
    for local_present, fanout, state_index in token_states:
        if fanout == 0:
            if not local_present:
                raise RuntimeError("L0F0 cannot produce a top-k route")
            targets = [source_rank] * top_k
        else:
            global_token = (
                source_rank * route_counts[state_index] + seen_per_state[state_index]
            )
            seen_per_state[state_index] += 1
            counts = _joint_target_counts(
                case=case,
                source_rank=source_rank,
                global_token=global_token,
                local_present=bool(local_present),
                fanout=fanout,
                top_k=top_k,
                world_size=world_size,
                experts_per_rank=experts_per_rank,
            )
            targets = [rank for rank, count in enumerate(counts) for _ in range(count)]
        row = []
        rank_occurrences: dict[int, int] = defaultdict(int)
        for target in targets:
            local_expert = (
                seen_per_rank[target] + rank_occurrences[target]
            ) % experts_per_rank
            row.append(target * experts_per_rank + local_expert)
            rank_occurrences[target] += 1
        for target, occurrences in rank_occurrences.items():
            seen_per_rank[target] += occurrences
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int64)


def make_case_topk_ids(
    case: LoadCase,
    *,
    tokens: int,
    top_k: int,
    num_experts: int,
    source_rank: int,
    world_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    if case.kind == "compute":
        topk_ids = _make_weighted_topk_ids(
            tokens=tokens,
            top_k=top_k,
            num_experts=num_experts,
            source_rank=source_rank,
            world_size=world_size,
            target_weights=case.source_target_weights[source_rank],
        )
    elif case.kind == "communication":
        topk_ids = _make_communication_topk_ids(
            case=case,
            tokens=tokens,
            top_k=top_k,
            num_experts=num_experts,
            source_rank=source_rank,
            world_size=world_size,
        )
    else:
        topk_ids = _make_joint_topk_ids(
            case=case,
            tokens=tokens,
            top_k=top_k,
            num_experts=num_experts,
            source_rank=source_rank,
            world_size=world_size,
        )
    return topk_ids.to(device) if device is not None else topk_ids


def rank_distribution(
    topk_ids: torch.Tensor,
    num_local_experts: int,
    world_size: int,
) -> tuple[list[int], list[int]]:
    stats = routing_stats(topk_ids, num_local_experts, world_size, source_rank=0)
    return stats.target_assignments, stats.target_unique_tokens


def routing_stats(
    topk_ids: torch.Tensor,
    num_local_experts: int,
    world_size: int,
    source_rank: int,
) -> RoutingStats:
    target_ranks = torch.div(
        topk_ids.to(torch.int64),
        num_local_experts,
        rounding_mode="floor",
    )
    assignments = torch.bincount(target_ranks.flatten(), minlength=world_size).tolist()
    unique_tokens = [
        int(torch.any(target_ranks == rank, dim=1).sum().item())
        for rank in range(world_size)
    ]
    remote_presence = torch.stack(
        [
            torch.any(target_ranks == rank, dim=1)
            for rank in range(world_size)
            if rank != source_rank
        ],
        dim=1,
    )
    token_fanouts = remote_presence.sum(dim=1).to(torch.int64)
    fanout_counts = torch.bincount(
        token_fanouts,
        minlength=world_size,
    ).tolist()
    local_presence = torch.any(target_ranks == source_rank, dim=1).to(torch.int64)
    route_state_counts = [[0] * world_size for _ in range(2)]
    for local_present in range(2):
        state_mask = local_presence == local_present
        state_fanouts = token_fanouts[state_mask]
        state_counts = torch.bincount(
            state_fanouts,
            minlength=world_size,
        ).tolist()
        route_state_counts[local_present] = [int(value) for value in state_counts]
    local_mask = torch.all(target_ranks == source_rank, dim=1)
    remote_mask = ~local_mask
    remote_transfers = sum(unique_tokens) - unique_tokens[source_rank]
    return RoutingStats(
        target_assignments=[int(value) for value in assignments],
        target_unique_tokens=unique_tokens,
        remote_assignments=sum(assignments) - assignments[source_rank],
        fully_local_tokens=int(local_mask.sum().item()),
        remote_tokens=int(remote_mask.sum().item()),
        remote_token_transfers=remote_transfers,
        fanout_counts=[int(value) for value in fanout_counts],
        route_state_counts=route_state_counts,
    )


def _normalize_weights(weights: list[float]) -> tuple[float, ...]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("At least one route weight must be positive")
    return tuple(weight / total for weight in weights)


def _compute_weight_matrix(
    pattern: str,
    world_size: int,
    hot_rank: int,
    control_value: float,
    zipf_alpha: float,
    seed: int,
) -> tuple[tuple[float, ...], ...]:
    hot_rank %= world_size
    if pattern == "balanced":
        row = (1.0 / world_size,) * world_size
    elif pattern == "single_hot":
        weights = [(1.0 - control_value) / (world_size - 1)] * world_size
        weights[hot_rank] = control_value
        row = _normalize_weights(weights)
    elif pattern == "multi_hot":
        hot_count = max(2, world_size // 2)
        if hot_count >= world_size:
            raise ValueError("multi_hot compute load requires WORLD_SIZE >= 3")
        hot_ranks = {(hot_rank + index) % world_size for index in range(hot_count)}
        weights = [
            control_value / hot_count
            if rank in hot_ranks
            else (1.0 - control_value) / (world_size - hot_count)
            for rank in range(world_size)
        ]
        row = _normalize_weights(weights)
    elif pattern == "zipf":
        weights = [0.0] * world_size
        for order in range(world_size):
            rank = (hot_rank + order) % world_size
            weights[rank] = 1.0 / (order + 1) ** zipf_alpha
        row = _normalize_weights(weights)
    elif pattern == "random":
        rng = random.Random(seed)
        row = _normalize_weights(
            [rng.gammavariate(control_value, 1.0) for _ in range(world_size)]
        )
    else:
        raise ValueError(f"Unknown compute pattern: {pattern}")
    return tuple(row for _ in range(world_size))


def _communication_weight_matrix(
    pattern: str,
    world_size: int,
    hot_rank: int,
    seed: int,
) -> tuple[tuple[float, ...], ...]:
    rows = []
    for source_rank in range(world_size):
        remote = [rank for rank in range(world_size) if rank != source_rank]
        weights = [0.0] * world_size
        if pattern == "coalesced":
            weights[(source_rank + 1) % world_size] = 1.0
        elif pattern == "spread":
            for rank in remote:
                weights[rank] = 1.0
        elif pattern == "hotspot":
            target = hot_rank % world_size
            if target == source_rank:
                target = (source_rank + 1) % world_size
            weights[target] = 1.0
        elif pattern == "partial":
            weights[source_rank] = 1.0
            for rank in remote:
                weights[rank] = 1.0
        elif pattern == "asymmetric":
            if source_rank % 2 == 0:
                weights[(source_rank + 1) % world_size] = 1.0
            else:
                for rank in remote:
                    weights[rank] = 1.0
        elif pattern == "random":
            rng = random.Random(seed + source_rank * 65537)
            weights = [rng.gammavariate(0.5, 1.0) for _ in range(world_size)]
        else:
            raise ValueError(f"Unknown communication pattern: {pattern}")
        rows.append(_normalize_weights(weights))
    return tuple(rows)


def _source_local_shares(
    local_share: float,
    pattern: str,
    world_size: int,
) -> tuple[float, ...]:
    if pattern != "asymmetric":
        return (local_share,) * world_size
    delta = min(0.2, abs(local_share - 0.5) / 2)
    return tuple(
        max(0.0, min(1.0, local_share + (delta if rank % 2 == 0 else -delta)))
        for rank in range(world_size)
    )


def _finalize_load_case(
    case: LoadCase,
    token_counts: list[int],
    top_k: int,
    num_experts: int,
    world_size: int,
) -> LoadCase:
    experts_per_rank = num_experts // world_size
    target_assignments = [0] * world_size
    expert_assignments = [0] * num_experts
    actual_weight_rows = []
    remote_transfer_matrix = [[0] * world_size for _ in range(world_size)]
    fanout_counts = [0] * world_size
    route_state_counts = [[0] * world_size for _ in range(2)]
    local_shares = []
    remote_tokens = 0
    remote_transfers = 0
    fingerprint = hashlib.sha256()

    for source_rank, tokens in enumerate(token_counts):
        topk_ids = make_case_topk_ids(
            case,
            tokens=tokens,
            top_k=top_k,
            num_experts=num_experts,
            source_rank=source_rank,
            world_size=world_size,
        )
        if tuple(topk_ids.shape) != (tokens, top_k):
            raise RuntimeError(f"{case.name} generated an invalid top-k shape")
        if int(topk_ids.min()) < 0 or int(topk_ids.max()) >= num_experts:
            raise RuntimeError(f"{case.name} generated an out-of-range expert ID")
        sorted_ids = torch.sort(topk_ids, dim=1).values
        if top_k > 1 and bool(torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1])):
            raise RuntimeError(f"{case.name} selected one expert twice for a token")

        stats = routing_stats(
            topk_ids,
            experts_per_rank,
            world_size,
            source_rank,
        )
        if sum(stats.target_assignments) != tokens * top_k:
            raise RuntimeError(
                f"{case.name} lost expert assignments on rank {source_rank}"
            )
        for target, count in enumerate(stats.target_assignments):
            target_assignments[target] += count
        source_expert_assignments = torch.bincount(
            topk_ids.flatten(), minlength=num_experts
        ).tolist()
        for expert, count in enumerate(source_expert_assignments):
            expert_assignments[expert] += count
        actual_weight_rows.append(
            tuple(count / (tokens * top_k) for count in stats.target_assignments)
        )
        local_shares.append(stats.fully_local_tokens / tokens)
        for target_rank, unique_tokens in enumerate(stats.target_unique_tokens):
            if target_rank != source_rank:
                remote_transfer_matrix[source_rank][target_rank] = unique_tokens
        remote_tokens += stats.remote_tokens
        remote_transfers += stats.remote_token_transfers
        for fanout, count in enumerate(stats.fanout_counts):
            fanout_counts[fanout] += count
        for local_present, state_counts in enumerate(stats.route_state_counts):
            for fanout, count in enumerate(state_counts):
                route_state_counts[local_present][fanout] += count
        target_ranks = torch.div(
            topk_ids,
            experts_per_rank,
            rounding_mode="floor",
        ).to(torch.int16)
        target_ranks = torch.sort(target_ranks, dim=1).values
        target_patterns, pattern_counts = torch.unique(
            target_ranks,
            dim=0,
            sorted=True,
            return_counts=True,
        )
        fingerprint.update(source_rank.to_bytes(4, "little"))
        fingerprint.update(target_patterns.contiguous().numpy().tobytes())
        fingerprint.update(pattern_counts.to(torch.int32).numpy().tobytes())

    total_assignments = sum(target_assignments)
    target_shares = tuple(count / total_assignments for count in target_assignments)
    mean_load = statistics.mean(target_assignments)
    load_ratio = max(target_assignments) / mean_load
    expert_mean = statistics.mean(expert_assignments)
    expert_load_ratio = max(expert_assignments) / expert_mean
    incoming_remote = [
        sum(remote_transfer_matrix[source][target] for source in range(world_size))
        for target in range(world_size)
    ]
    incoming_mean = statistics.mean(incoming_remote)
    destination_comm_ratio = (
        max(incoming_remote) / incoming_mean if incoming_mean else 0.0
    )
    link_loads = [
        remote_transfer_matrix[source][target]
        for source in range(world_size)
        for target in range(world_size)
        if source != target
    ]
    link_mean = statistics.mean(link_loads)
    link_comm_ratio = max(link_loads) / link_mean if link_mean else 0.0

    compute_is_balanced = (
        case.variant == "balanced" or case.compute_pattern == "balanced"
    )
    compute_is_skewed = (
        case.kind == "compute" and case.variant != "balanced"
    ) or case.compute_pattern == "skewed"
    if compute_is_balanced:
        if load_ratio > 1.01:
            raise ValueError(
                f"{case.name} is not balanced after rounding: "
                f"max/mean={load_ratio:.4f}, loads={target_assignments}"
            )
        if expert_load_ratio > 1.05:
            raise ValueError(
                f"{case.name} has imbalanced physical experts: "
                f"max/mean={expert_load_ratio:.4f}"
            )
    elif compute_is_skewed and load_ratio < 1.05:
        raise ValueError(
            f"{case.name} is not measurably skewed: max/mean={load_ratio:.4f}. "
            "Reduce --top-k or increase the requested skew."
        )

    remote_fanout = remote_transfers / remote_tokens if remote_tokens else 0.0
    if case.kind == "joint" and remote_transfers:
        assert case.joint_route_shares is not None
        expected_route_state_counts = [[0] * world_size for _ in range(2)]
        for tokens in token_counts:
            source_counts = apportion(
                tokens,
                [share for _, _, share in case.joint_route_shares],
                0,
            )
            for (local_present, fanout, _), count in zip(
                case.joint_route_shares,
                source_counts,
            ):
                expected_route_state_counts[local_present][fanout] += count
        if route_state_counts != expected_route_state_counts:
            raise ValueError(
                f"{case.name} changed the token route-state histogram: expected "
                f"{expected_route_state_counts}, got {route_state_counts}"
            )
        if case.communication_pattern in ("balanced", "saturated"):
            if destination_comm_ratio > 1.01 or link_comm_ratio > 1.01:
                raise ValueError(
                    f"{case.name} did not balance communication: destination "
                    f"max/mean={destination_comm_ratio:.4f}, link "
                    f"max/mean={link_comm_ratio:.4f}"
                )
        elif destination_comm_ratio < 1.05 or link_comm_ratio < 1.05:
            raise ValueError(
                f"{case.name} did not create a communication hotspot: destination "
                f"max/mean={destination_comm_ratio:.4f}, link "
                f"max/mean={link_comm_ratio:.4f}"
            )

    if case.variant == "mostly_local" and any(share <= 0.5 for share in local_shares):
        raise ValueError(f"{case.name} did not keep a majority local on every rank")
    if case.variant == "mostly_remote" and any(share >= 0.5 for share in local_shares):
        raise ValueError(f"{case.name} did not make a majority remote on every rank")
    if (
        case.kind == "communication"
        and case.pattern == "spread"
        and world_size > 2
        and top_k > 1
        and remote_fanout <= 1.0
    ):
        raise RuntimeError(f"{case.name} did not create cross-rank top-k fan-out")

    return replace(
        case,
        planned_source_target_shares=tuple(actual_weight_rows),
        planned_target_assignments=tuple(target_assignments),
        planned_target_shares=target_shares,
        planned_expert_assignments_by_rank=tuple(
            tuple(
                expert_assignments[
                    rank * experts_per_rank : (rank + 1) * experts_per_rank
                ]
            )
            for rank in range(world_size)
        ),
        planned_load_max_over_mean=load_ratio,
        planned_expert_load_max_over_mean=expert_load_ratio,
        planned_destination_comm_max_over_mean=destination_comm_ratio,
        planned_link_comm_max_over_mean=link_comm_ratio,
        planned_remote_transfer_matrix=tuple(
            tuple(row) for row in remote_transfer_matrix
        ),
        planned_fanout_counts=tuple(fanout_counts),
        planned_fanout_shares=tuple(
            count / sum(fanout_counts) for count in fanout_counts
        ),
        planned_route_state_counts=tuple(tuple(row) for row in route_state_counts),
        planned_route_state_shares=tuple(
            tuple(count / sum(fanout_counts) for count in row)
            for row in route_state_counts
        ),
        planned_local_share_min=min(local_shares),
        planned_local_share_max=max(local_shares),
        planned_remote_fanout=remote_fanout,
        routing_fingerprint=fingerprint.hexdigest()[:16],
    )


def build_load_cases(
    args: argparse.Namespace,
    token_counts: list[int],
    world_size: int,
) -> list[LoadCase]:
    candidates = []
    compute_patterns = set(args.compute_patterns)
    if "compute" in args.benchmarks:
        if "balanced" in compute_patterns:
            candidates.append(
                LoadCase(
                    name="compute_balanced",
                    kind="compute",
                    pattern="balanced",
                    variant="balanced",
                    control_value=1.0 / world_size,
                    source_target_weights=_compute_weight_matrix(
                        "balanced", world_size, args.hot_rank, 0.0, 0.0, args.seed
                    ),
                )
            )
        if "single_hot" in compute_patterns:
            hot_shares = (
                parse_shares(args.hot_shares, "--hot-shares")
                if args.hot_shares is not None
                else [args.imbalance_strength]
            )
            for index, hot_share in enumerate(hot_shares):
                if math.isclose(hot_share, 1.0 / world_size):
                    continue
                candidates.append(
                    LoadCase(
                        name=f"compute_single_hot_{hot_share:g}",
                        kind="compute",
                        pattern="single_hot",
                        variant="imbalanced",
                        control_value=hot_share,
                        source_target_weights=_compute_weight_matrix(
                            "single_hot",
                            world_size,
                            args.hot_rank,
                            hot_share,
                            args.zipf_alpha,
                            args.seed + index,
                        ),
                    )
                )
        for pattern in ("multi_hot", "zipf"):
            if pattern not in compute_patterns:
                continue
            if pattern == "multi_hot" and world_size < 3:
                continue
            candidates.append(
                LoadCase(
                    name=f"compute_{pattern}",
                    kind="compute",
                    pattern=pattern,
                    variant="imbalanced",
                    control_value=(
                        args.zipf_alpha
                        if pattern == "zipf"
                        else args.imbalance_strength
                    ),
                    source_target_weights=_compute_weight_matrix(
                        pattern,
                        world_size,
                        args.hot_rank,
                        args.imbalance_strength,
                        args.zipf_alpha,
                        args.seed,
                    ),
                )
            )
        if "random" in compute_patterns:
            for index in range(args.random_cases):
                seed = args.seed + 1009 + index
                candidates.append(
                    LoadCase(
                        name=f"compute_random_{index}",
                        kind="compute",
                        pattern="random",
                        variant="imbalanced",
                        control_value=args.random_alpha,
                        seed=seed,
                        source_target_weights=_compute_weight_matrix(
                            "random",
                            world_size,
                            args.hot_rank,
                            args.random_alpha,
                            args.zipf_alpha,
                            seed,
                        ),
                    )
                )

    if "joint" in args.benchmarks:
        if len(set(token_counts)) != 1:
            raise ValueError(
                "The controlled joint experiment requires equal token counts "
                "per rank; run unequal token counts as a separate stress case."
            )
        if args.top_k < world_size:
            raise ValueError(
                "The controlled joint experiment requires top_k >= WORLD_SIZE "
                "so expert slots can be redistributed while remote fan-out stays fixed."
            )
        if args.joint_route_shares is None:
            raise ValueError(
                "--benchmarks joint requires at least one "
                "--joint-route-shares distribution"
            )
        route_distributions = [
            parse_route_shares(raw, world_size) for raw in args.joint_route_shares
        ]
        experts_per_rank = args.num_experts // world_size
        for route_shares in route_distributions:
            active_states = [
                (local_present, fanout)
                for local_present, fanout, share in route_shares
                if share > 0
            ]
            remote_fanouts = [fanout for _, fanout in active_states if fanout > 0]
            if not remote_fanouts:
                raise ValueError(
                    "--joint-route-shares must assign a positive share to at "
                    "least one remote fan-out"
                )
            saturated_communication = all(
                fanout == world_size - 1 for fanout in remote_fanouts
            )
            if not saturated_communication and (0, 1) in active_states:
                raise ValueError(
                    "L0F1 is structurally incompatible with the controlled "
                    "compute-balanced communication-hotspot case: a remote-only "
                    "token with one destination has no alternative rank for its "
                    "top-k slots. Use L0F2 or larger for the full 2x2 experiment."
                )
            max_slots_on_one_rank = max(
                args.top_k
                if local_present and fanout == 0
                else args.top_k - fanout
                if local_present
                else args.top_k - (fanout - 1)
                for local_present, fanout in active_states
            )
            if max_slots_on_one_rank > experts_per_rank:
                raise ValueError(
                    "The controlled joint experiment requires every token's "
                    "per-rank top-k count to fit the physical experts; "
                    f"required={max_slots_on_one_rank}, "
                    f"experts_per_rank={experts_per_rank}."
                )
            route_label = "_".join(
                f"L{local_present}F{fanout}-{share:g}"
                for local_present, fanout, share in route_shares
                if share > 0
            )
            for compute_pattern in ("balanced", "skewed"):
                communication_patterns = (
                    ("saturated",)
                    if saturated_communication
                    else ("balanced", "hotspot")
                )
                for communication_pattern in communication_patterns:
                    compute_weights = _compute_weight_matrix(
                        "balanced" if compute_pattern == "balanced" else "single_hot",
                        world_size,
                        args.hot_rank,
                        (
                            1.0 / world_size
                            if compute_pattern == "balanced"
                            else args.imbalance_strength
                        ),
                        args.zipf_alpha,
                        args.seed,
                    )
                    candidates.append(
                        LoadCase(
                            name=(
                                f"joint_compute_{compute_pattern}_comm_"
                                f"{communication_pattern}_routes_{route_label}"
                            ),
                            kind="joint",
                            pattern=f"{compute_pattern}_{communication_pattern}",
                            variant=(
                                f"compute_{compute_pattern}_"
                                f"communication_{communication_pattern}"
                            ),
                            control_value=(
                                args.imbalance_strength
                                if compute_pattern == "skewed"
                                else 1.0 / world_size
                            ),
                            source_target_weights=compute_weights,
                            compute_pattern=compute_pattern,
                            communication_pattern=communication_pattern,
                            joint_route_shares=route_shares,
                            hot_rank=args.hot_rank,
                            seed=args.seed,
                        )
                    )

    if "communication" in args.benchmarks:
        for local_share in parse_shares(args.local_shares, "--local-shares"):
            variant = (
                "mostly_local"
                if local_share > 0.5
                else "mostly_remote"
                if local_share < 0.5
                else "mixed"
            )
            for pattern in args.communication_patterns:
                seed = args.seed + 7919 * (len(candidates) + 1)
                candidates.append(
                    LoadCase(
                        name=f"communication_{variant}_{pattern}_{local_share:g}",
                        kind="communication",
                        pattern=pattern,
                        variant=variant,
                        control_value=local_share,
                        seed=seed,
                        source_target_weights=_communication_weight_matrix(
                            pattern,
                            world_size,
                            args.hot_rank,
                            seed,
                        ),
                        source_local_token_shares=_source_local_shares(
                            local_share,
                            pattern,
                            world_size,
                        ),
                    )
                )

    cases = []
    fingerprints = set()
    for candidate in candidates:
        case = _finalize_load_case(
            candidate,
            token_counts,
            args.top_k,
            args.num_experts,
            world_size,
        )
        dedupe_key = (case.kind, case.routing_fingerprint)
        if dedupe_key in fingerprints:
            continue
        fingerprints.add(dedupe_key)
        cases.append(case)

    joint_by_route: dict[tuple[tuple[int, int, float], ...], list[LoadCase]] = (
        defaultdict(list)
    )
    for case in cases:
        if case.kind == "joint":
            assert case.joint_route_shares is not None
            joint_by_route[case.joint_route_shares].append(case)
    for route_cases in joint_by_route.values():
        transfer_totals = {
            sum(sum(row) for row in case.planned_remote_transfer_matrix)
            for case in route_cases
        }
        route_histograms = {case.planned_route_state_counts for case in route_cases}
        if len(transfer_totals) != 1 or len(route_histograms) != 1:
            raise RuntimeError(
                "Joint cases changed communication volume while varying "
                "uniformity: "
                f"transfers={sorted(transfer_totals)}, "
                f"route_histograms={sorted(route_histograms)}"
            )
        expected_communication_cases = (
            1
            if all(
                fanout in (0, world_size - 1)
                for fanout, count in enumerate(route_cases[0].planned_fanout_counts)
                if count
            )
            else 2
        )
        for compute_pattern in ("balanced", "skewed"):
            load_ratios = [
                case.planned_load_max_over_mean
                for case in route_cases
                if case.compute_pattern == compute_pattern
            ]
            ratios_differ = len(load_ratios) == 2 and not math.isclose(
                load_ratios[0], load_ratios[1], rel_tol=0.01, abs_tol=0.01
            )
            if len(load_ratios) != expected_communication_cases or ratios_differ:
                raise RuntimeError(
                    "Joint cases changed compute skew while varying communication "
                    f"uniformity: compute_pattern={compute_pattern}, "
                    f"max_over_mean={load_ratios}"
                )
    if not cases:
        raise ValueError(
            "The selected load patterns did not produce any distinct cases"
        )
    return cases


def load_case_metadata(case: LoadCase | None) -> dict[str, Any]:
    if case is None:
        return {}
    return {
        "scenario": case.name,
        "benchmark_kind": case.kind,
        "pattern": case.pattern,
        "variant": case.variant,
        "compute_pattern": case.compute_pattern,
        "communication_pattern": case.communication_pattern,
        "joint_route_shares": (
            [list(state) for state in case.joint_route_shares]
            if case.joint_route_shares is not None
            else None
        ),
        "control_value": case.control_value,
        "scenario_seed": case.seed,
        "planned_source_target_shares": [
            list(row) for row in case.planned_source_target_shares
        ],
        "planned_target_assignments": list(case.planned_target_assignments),
        "planned_target_shares": list(case.planned_target_shares),
        "planned_expert_assignments_by_rank": [
            list(row) for row in case.planned_expert_assignments_by_rank
        ],
        "planned_load_max_over_mean": case.planned_load_max_over_mean,
        "planned_expert_load_max_over_mean": (case.planned_expert_load_max_over_mean),
        "planned_destination_comm_max_over_mean": (
            case.planned_destination_comm_max_over_mean
        ),
        "planned_link_comm_max_over_mean": case.planned_link_comm_max_over_mean,
        "planned_remote_transfer_matrix": [
            list(row) for row in case.planned_remote_transfer_matrix
        ],
        "planned_fanout_counts": list(case.planned_fanout_counts),
        "planned_fanout_shares": list(case.planned_fanout_shares),
        "planned_route_state_counts": [
            list(row) for row in case.planned_route_state_counts
        ],
        "planned_route_state_shares": [
            list(row) for row in case.planned_route_state_shares
        ],
        "planned_local_share_min": case.planned_local_share_min,
        "planned_local_share_max": case.planned_local_share_max,
        "planned_remote_fanout": case.planned_remote_fanout,
        "routing_fingerprint": case.routing_fingerprint,
    }


LOAD_CASE_METADATA_FIELDS = tuple(
    load_case_metadata(
        LoadCase(
            name="",
            kind="compute",
            pattern="",
            variant="",
            control_value=0.0,
            source_target_weights=(),
        )
    )
)


def make_expert_map(
    num_experts: int,
    num_local_experts: int,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    start = rank * num_local_experts
    expert_map[start : start + num_local_experts] = torch.arange(
        num_local_experts,
        dtype=torch.int32,
        device=device,
    )
    return expert_map


def make_vllm_config(
    world_size: int,
    rank: int,
    local_rank: int,
) -> VllmConfig:
    vllm_config = VllmConfig()
    parallel_config = vllm_config.parallel_config
    parallel_config.data_parallel_size = world_size
    parallel_config.data_parallel_rank = rank
    parallel_config.enable_expert_parallel = True
    parallel_config.is_moe_model = True
    parallel_config.all2all_backend = "deepep_high_throughput"
    parallel_config.distributed_executor_backend = "external_launcher"
    vllm_config.device_config.device = torch.device("cuda", local_rank)
    return vllm_config


def make_kernels(
    args: argparse.Namespace,
    vllm_config: VllmConfig,
    dtype: torch.dtype,
    device: torch.device,
    *,
    max_num_tokens: int | None = None,
) -> tuple[mk.FusedMoEKernel, mk.FusedMoEKernel]:
    moe_parallel_config = FusedMoEParallelConfig.make(
        tp_size_=get_tensor_model_parallel_world_size(),
        pcp_size_=get_pcp_group().world_size,
        dp_size_=get_dp_group().world_size,
        sp_size_=1,
        vllm_parallel_config=vllm_config.parallel_config,
    )
    num_local_experts = args.num_experts // dist.get_world_size()
    moe_config = FusedMoEConfig(
        num_experts=args.num_experts,
        experts_per_token=args.top_k,
        hidden_dim=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_local_experts=num_local_experts,
        num_logical_experts=args.num_experts,
        moe_parallel_config=moe_parallel_config,
        in_dtype=dtype,
        max_num_tokens=next_power_of_2(max_num_tokens or args.tokens),
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.TopK,
    )
    quant_config = FusedMoEQuantConfig.make()
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=quant_config,
        allow_new_interface=True,
    )
    if not isinstance(prepare_finalize, DeepEPHTPrepareAndFinalize):
        raise RuntimeError(
            "Expected DeepEPHTPrepareAndFinalize, got "
            f"{type(prepare_finalize).__name__}"
        )
    deepep_kernel = mk.FusedMoEKernel(
        prepare_finalize, TritonExperts(moe_config, quant_config)
    )
    local_kernel = mk.FusedMoEKernel(
        MoEPrepareAndFinalizeNoDPEPModular(),
        TritonExperts(moe_config, quant_config),
    )
    return deepep_kernel, local_kernel


def make_base_tensors(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(args.seed + rank)
    num_local_experts = args.num_experts // world_size
    hidden_states = torch.randn(
        (args.tokens, args.hidden_size), device=device, dtype=dtype
    )
    w1 = (
        torch.randn(
            (num_local_experts, 2 * args.intermediate_size, args.hidden_size),
            device=device,
            dtype=dtype,
        )
        / 10
    )
    w2 = (
        torch.randn(
            (num_local_experts, args.hidden_size, args.intermediate_size),
            device=device,
            dtype=dtype,
        )
        / 10
    )
    return {
        "hidden_states": hidden_states,
        "w1": w1.contiguous(),
        "w2": w2.contiguous(),
        "topk_weights": torch.full(
            (args.tokens, args.top_k),
            1.0 / args.top_k,
            device=device,
            dtype=torch.float32,
        ),
        "expert_map": make_expert_map(
            args.num_experts, num_local_experts, rank, device
        ),
    }


def run_one_iter(
    args: argparse.Namespace,
    kernel: mk.FusedMoEKernel,
    tensors: dict[str, torch.Tensor],
    topk_ids: torch.Tensor,
    *,
    distribution: str,
    target_share: float,
    sweep_index: int,
    iteration: int,
    rank: int,
    world_size: int,
    device: torch.device,
    profile_warmup: int,
    capture_output: bool = False,
    case: LoadCase | None = None,
) -> dict[str, Any]:
    assert isinstance(kernel.impl, mk.FusedMoEKernelModularImpl)
    requested_dtype = kernel.prepare_finalize.topk_indices_dtype()
    if requested_dtype is not None:
        topk_ids = topk_ids.to(requested_dtype)

    hidden_states = tensors["hidden_states"]
    topk_weights = tensors["topk_weights"]
    output = torch.empty_like(hidden_states)
    local_num_experts = tensors["w1"].shape[0]
    route = routing_stats(
        topk_ids,
        local_num_experts,
        world_size,
        rank,
    )
    target_assignments = route.target_assignments
    target_unique_tokens = route.target_unique_tokens
    actual_local_share = route.fully_local_tokens / args.tokens

    def prepare():
        return kernel.impl._prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            args.num_experts,
            tensors["expert_map"],
            False,
        )

    (
        (
            a1q,
            a1q_scale,
            expert_tokens_meta,
            dispatched_topk_ids,
            dispatched_topk_weights,
        ),
        dispatch_ms,
    ) = time_stage(
        device,
        prepare,
        use_barrier=args.stage_barrier,
    )
    # Finalize clears the active slot after adding the combine timings.
    detail_sample = (
        kernel.prepare_finalize.get_active_profile_sample()
        if args.detail_profile
        else None
    )

    def compute():
        return kernel.impl._fused_experts(
            in_dtype=hidden_states.dtype,
            a1q=a1q,
            a1q_scale=a1q_scale,
            w1=tensors["w1"],
            w2=tensors["w2"],
            topk_weights=dispatched_topk_weights,
            topk_ids=dispatched_topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=args.num_experts,
            local_num_experts=local_num_experts,
            expert_map=tensors["expert_map"],
            apply_router_weight_on_input=False,
            expert_tokens_meta=expert_tokens_meta,
            output_alias=output,
        )

    fused_out, compute_ms = time_stage(
        device,
        compute,
        use_barrier=args.stage_barrier,
    )

    def finalize():
        return kernel.impl._finalize(
            output,
            fused_out,
            hidden_states,
            dispatched_topk_weights,
            dispatched_topk_ids,
            False,
            None,
            None,
        )

    _, combine_ms = time_stage(
        device,
        finalize,
        use_barrier=args.stage_barrier,
    )
    if expert_tokens_meta is None:
        raise RuntimeError("DeepEP-HT did not return expert token metadata")
    local_expert_tokens = [
        int(value)
        for value in expert_tokens_meta.expert_num_tokens.detach().cpu().tolist()
    ]
    profile_sample = profile_warmup + sweep_index * args.iters + iteration
    if args.detail_profile and iteration >= 0:
        if detail_sample is None:
            raise RuntimeError("DeepEP-HT detail profile did not produce a sample")
        if detail_sample.sample_id != profile_sample:
            raise RuntimeError(
                f"Expected detail sample {profile_sample}, got "
                f"{detail_sample.sample_id}"
            )
    record = {
        "record_type": "rank",
        "execution_mode": "full_deepep",
        "distribution": distribution,
        "target_share": target_share,
        "hot_share": (
            target_share
            if distribution == "hot_rank"
            or (case is not None and case.pattern == "single_hot")
            else None
        ),
        "local_share": (
            target_share
            if distribution == "local_share"
            or (case is not None and case.kind == "communication")
            else None
        ),
        "iter": iteration,
        "rank": rank,
        "world_size": world_size,
        "top_k": args.top_k,
        "experts_per_rank": local_num_experts,
        "profile_sample": profile_sample if args.detail_profile else None,
        "dispatch_ms": dispatch_ms,
        "expert_compute_ms": compute_ms,
        "combine_ms": combine_ms,
        "total_ms": dispatch_ms + compute_ms + combine_ms,
        "source_target_assignments": target_assignments,
        "source_target_unique_tokens": target_unique_tokens,
        "remote_assignments": route.remote_assignments,
        "fully_local_tokens": route.fully_local_tokens,
        "remote_tokens": route.remote_tokens,
        "remote_unique_tokens": route.remote_token_transfers,
        "remote_token_transfers": route.remote_token_transfers,
        "source_fanout_counts": route.fanout_counts,
        "source_route_state_counts": route.route_state_counts,
        "source_tokens": args.tokens,
        "local_path_tokens": 0,
        "deepep_source_tokens": args.tokens,
        "actual_local_share": actual_local_share,
        "remote_payload_bytes": route.remote_token_transfers
        * args.hidden_size
        * hidden_states.element_size(),
        "active_destinations": sum(count > 0 for count in target_unique_tokens),
        "received_tokens": sum(local_expert_tokens),
        "local_expert_tokens": local_expert_tokens,
        "detail_timings_ms": (
            dict(detail_sample.timings_ms) if detail_sample is not None else None
        ),
        "detail_metadata": (
            dict(detail_sample.metadata) if detail_sample is not None else None
        ),
        **load_case_metadata(case),
    }
    if capture_output:
        record["output"] = output.clone()
    return record


def run_local_bypass_iter(
    args: argparse.Namespace,
    deepep_kernel: mk.FusedMoEKernel,
    local_kernel: mk.FusedMoEKernel,
    tensors: dict[str, torch.Tensor],
    topk_ids: torch.Tensor,
    *,
    distribution: str,
    target_share: float,
    iteration: int,
    rank: int,
    world_size: int,
    device: torch.device,
    capture_output: bool = False,
    case: LoadCase | None = None,
) -> dict[str, Any]:
    assert isinstance(deepep_kernel.impl, mk.FusedMoEKernelModularImpl)
    assert isinstance(local_kernel.impl, mk.FusedMoEKernelModularImpl)
    requested_dtype = deepep_kernel.prepare_finalize.topk_indices_dtype()
    if requested_dtype is not None:
        topk_ids = topk_ids.to(requested_dtype)

    hidden_states = tensors["hidden_states"]
    topk_weights = tensors["topk_weights"]
    local_num_experts = tensors["w1"].shape[0]
    route = routing_stats(
        topk_ids,
        local_num_experts,
        world_size,
        rank,
    )
    target_assignments = route.target_assignments
    target_unique_tokens = route.target_unique_tokens
    actual_local_share = route.fully_local_tokens / args.tokens

    def prepare():
        target_ranks = torch.div(
            topk_ids,
            local_num_experts,
            rounding_mode="floor",
        )
        local_mask = torch.all(target_ranks == rank, dim=1)
        local_indices = torch.where(local_mask)[0]
        remote_indices = torch.where(~local_mask)[0]

        def prepare_batch(
            kernel: mk.FusedMoEKernel,
            indices: torch.Tensor,
        ) -> dict[str, Any] | None:
            if indices.numel() == 0:
                return None
            batch_hidden = hidden_states.index_select(0, indices)
            batch_weights = topk_weights.index_select(0, indices)
            batch_ids = topk_ids.index_select(0, indices)
            prepared = kernel.impl._prepare(
                batch_hidden,
                batch_weights,
                batch_ids,
                args.num_experts,
                tensors["expert_map"],
                False,
            )
            return {
                "indices": indices,
                "hidden_states": batch_hidden,
                "prepared": prepared,
            }

        return (
            prepare_batch(local_kernel, local_indices),
            prepare_batch(deepep_kernel, remote_indices),
        )

    (local_batch, remote_batch), dispatch_ms = time_stage(
        device,
        prepare,
        use_barrier=args.stage_barrier,
    )

    def compute_batch(
        kernel: mk.FusedMoEKernel,
        batch: dict[str, Any] | None,
    ) -> torch.Tensor | None:
        if batch is None:
            return None
        (
            a1q,
            a1q_scale,
            expert_tokens_meta,
            dispatched_topk_ids,
            dispatched_topk_weights,
        ) = batch["prepared"]
        return kernel.impl._fused_experts(
            in_dtype=hidden_states.dtype,
            a1q=a1q,
            a1q_scale=a1q_scale,
            w1=tensors["w1"],
            w2=tensors["w2"],
            topk_weights=dispatched_topk_weights,
            topk_ids=dispatched_topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=args.num_experts,
            local_num_experts=local_num_experts,
            expert_map=tensors["expert_map"],
            apply_router_weight_on_input=False,
            expert_tokens_meta=expert_tokens_meta,
            output_alias=None,
        )

    def compute():
        local_fused_out = compute_batch(local_kernel, local_batch)
        if local_fused_out is not None and remote_batch is not None:
            # Both kernels reuse the process-wide MoE workspace.
            local_fused_out = local_fused_out.clone()
        remote_fused_out = compute_batch(deepep_kernel, remote_batch)
        return local_fused_out, remote_fused_out

    (local_fused_out, remote_fused_out), compute_ms = time_stage(
        device,
        compute,
        use_barrier=args.stage_barrier,
    )

    output = torch.empty_like(hidden_states)

    def finalize_batch(
        kernel: mk.FusedMoEKernel,
        batch: dict[str, Any] | None,
        fused_out: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if batch is None:
            return None
        assert fused_out is not None
        (
            _,
            _,
            _,
            dispatched_topk_ids,
            dispatched_topk_weights,
        ) = batch["prepared"]
        batch_output = torch.empty_like(batch["hidden_states"])
        kernel.impl._finalize(
            batch_output,
            fused_out,
            batch["hidden_states"],
            dispatched_topk_weights,
            dispatched_topk_ids,
            False,
            None,
            None,
        )
        return batch_output

    def finalize():
        local_output = finalize_batch(local_kernel, local_batch, local_fused_out)
        remote_output = finalize_batch(deepep_kernel, remote_batch, remote_fused_out)
        if local_batch is not None:
            assert local_output is not None
            output.index_copy_(0, local_batch["indices"], local_output)
        if remote_batch is not None:
            assert remote_output is not None
            output.index_copy_(0, remote_batch["indices"], remote_output)

    _, combine_ms = time_stage(
        device,
        finalize,
        use_barrier=args.stage_barrier,
    )

    local_expert_tokens = torch.zeros(
        local_num_experts, dtype=torch.int64, device=device
    )
    if local_batch is not None:
        local_ids = local_batch["prepared"][3].to(torch.int64)
        local_ids = local_ids - rank * local_num_experts
        local_expert_tokens += torch.bincount(
            local_ids.flatten(), minlength=local_num_experts
        )
    if remote_batch is not None:
        remote_meta = remote_batch["prepared"][2]
        if remote_meta is None:
            raise RuntimeError("DeepEP-HT did not return expert token metadata")
        local_expert_tokens += remote_meta.expert_num_tokens.to(torch.int64)
    local_expert_tokens_list = [
        int(value) for value in local_expert_tokens.detach().cpu().tolist()
    ]
    local_path_tokens = (
        int(local_batch["indices"].numel()) if local_batch is not None else 0
    )
    deepep_source_tokens = (
        int(remote_batch["indices"].numel()) if remote_batch is not None else 0
    )
    record = {
        "record_type": "rank",
        "execution_mode": "local_bypass",
        "distribution": distribution,
        "target_share": target_share,
        "hot_share": None,
        "local_share": target_share,
        "iter": iteration,
        "rank": rank,
        "world_size": world_size,
        "top_k": args.top_k,
        "experts_per_rank": local_num_experts,
        "profile_sample": None,
        "dispatch_ms": dispatch_ms,
        "expert_compute_ms": compute_ms,
        "combine_ms": combine_ms,
        "total_ms": dispatch_ms + compute_ms + combine_ms,
        "source_target_assignments": target_assignments,
        "source_target_unique_tokens": target_unique_tokens,
        "remote_assignments": route.remote_assignments,
        "fully_local_tokens": route.fully_local_tokens,
        "remote_tokens": route.remote_tokens,
        "remote_unique_tokens": route.remote_token_transfers,
        "remote_token_transfers": route.remote_token_transfers,
        "source_fanout_counts": route.fanout_counts,
        "source_route_state_counts": route.route_state_counts,
        "source_tokens": args.tokens,
        "local_path_tokens": local_path_tokens,
        "deepep_source_tokens": deepep_source_tokens,
        "actual_local_share": actual_local_share,
        "remote_payload_bytes": route.remote_token_transfers
        * args.hidden_size
        * hidden_states.element_size(),
        "active_destinations": sum(count > 0 for count in target_unique_tokens),
        "received_tokens": sum(local_expert_tokens_list),
        "local_expert_tokens": local_expert_tokens_list,
        "detail_timings_ms": None,
        "detail_metadata": None,
        **load_case_metadata(case),
    }
    if capture_output:
        record["output"] = output.clone()
    return record


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_iter: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_iter[int(record["iter"])].append(record)

    aggregates = []
    for iteration, rows in sorted(by_iter.items()):
        dispatch = [row["dispatch_ms"] for row in rows]
        compute = [row["expert_compute_ms"] for row in rows]
        combine = [row["combine_ms"] for row in rows]
        totals = [row["total_ms"] for row in rows]
        received = [row["received_tokens"] for row in rows]
        received_mean = statistics.mean(received)
        actual_local_shares = [row["actual_local_share"] for row in rows]
        source_tokens = sum(row["source_tokens"] for row in rows)
        tokens_per_rank = [0] * len(rows)
        for row in rows:
            tokens_per_rank[row["rank"]] = row["source_tokens"]
        fully_local_tokens = sum(row["fully_local_tokens"] for row in rows)
        remote_tokens = sum(row["remote_tokens"] for row in rows)
        local_path_tokens = sum(row["local_path_tokens"] for row in rows)
        deepep_source_tokens = sum(row["deepep_source_tokens"] for row in rows)
        remote_payload_bytes = sum(row["remote_payload_bytes"] for row in rows)
        remote_unique_tokens = sum(row["remote_unique_tokens"] for row in rows)
        actual_fanout_counts = [0] * len(rows)
        actual_route_state_counts = [[0] * len(rows) for _ in range(2)]
        for row in rows:
            source_fanout_counts = row["source_fanout_counts"]
            if len(source_fanout_counts) != len(rows):
                raise RuntimeError(
                    "Measured fan-out histogram has an invalid world size: "
                    f"rank {row['rank']} reported {len(source_fanout_counts)} "
                    f"buckets for {len(rows)} ranks"
                )
            for fanout, count in enumerate(source_fanout_counts):
                actual_fanout_counts[fanout] += count
            source_route_state_counts = row["source_route_state_counts"]
            if len(source_route_state_counts) != 2 or any(
                len(state_counts) != len(rows)
                for state_counts in source_route_state_counts
            ):
                raise RuntimeError(
                    "Measured route-state histogram has an invalid shape on "
                    f"rank {row['rank']}: {source_route_state_counts}"
                )
            for local_present, state_counts in enumerate(source_route_state_counts):
                for fanout, count in enumerate(state_counts):
                    actual_route_state_counts[local_present][fanout] += count
        actual_fanout_shares = [count / source_tokens for count in actual_fanout_counts]
        actual_route_state_shares = [
            [count / source_tokens for count in state_counts]
            for state_counts in actual_route_state_counts
        ]
        actual_local_presence_share = sum(actual_route_state_counts[1]) / source_tokens
        actual_mixed_local_remote_share = (
            sum(actual_route_state_counts[1][1:]) / source_tokens
        )
        actual_remote_only_share = sum(actual_route_state_counts[0][1:]) / source_tokens
        actual_experts_by_rank = {
            int(row["rank"]): row["local_expert_tokens"] for row in rows
        }
        actual_experts = [actual_experts_by_rank[rank] for rank in range(len(rows))]
        expert_loads = [count for rank_loads in actual_experts for count in rank_loads]
        expert_mean = statistics.mean(expert_loads)
        actual_expert_load_ratio = (
            max(expert_loads) / expert_mean if expert_mean else 0.0
        )
        actual_transfer_matrix = [[0] * len(rows) for _ in rows]
        for row in rows:
            source_rank = int(row["rank"])
            target_unique_tokens = row["source_target_unique_tokens"]
            if len(target_unique_tokens) != len(rows):
                raise RuntimeError(
                    "Measured communication matrix has an invalid world size: "
                    f"rank {source_rank} reported {len(target_unique_tokens)} "
                    f"destinations for {len(rows)} ranks"
                )
            for target_rank, unique_tokens in enumerate(target_unique_tokens):
                if source_rank != target_rank:
                    actual_transfer_matrix[source_rank][target_rank] = unique_tokens
        incoming_remote = [
            sum(actual_transfer_matrix[source][target] for source in range(len(rows)))
            for target in range(len(rows))
        ]
        incoming_mean = statistics.mean(incoming_remote)
        actual_destination_comm_ratio = (
            max(incoming_remote) / incoming_mean if incoming_mean else 0.0
        )
        link_loads = [
            actual_transfer_matrix[source][target]
            for source in range(len(rows))
            for target in range(len(rows))
            if source != target
        ]
        link_mean = statistics.mean(link_loads)
        actual_link_comm_ratio = max(link_loads) / link_mean if link_mean else 0.0
        case_metadata = {
            field: rows[0][field]
            for field in LOAD_CASE_METADATA_FIELDS
            if field in rows[0]
        }
        planned_received = rows[0].get("planned_target_assignments")
        if planned_received is not None:
            actual_received = [0] * len(rows)
            for row in rows:
                actual_received[row["rank"]] = row["received_tokens"]
            if actual_received != planned_received:
                raise RuntimeError(
                    "Measured expert-token loads differ from the generated route: "
                    f"expected {planned_received}, got {actual_received}"
                )
        planned_experts = rows[0].get("planned_expert_assignments_by_rank")
        if planned_experts is not None and actual_experts != planned_experts:
            raise RuntimeError(
                "Measured per-expert loads differ from the generated route: "
                f"expected {planned_experts}, got {actual_experts}"
            )
        planned_transfers = rows[0].get("planned_remote_transfer_matrix")
        if planned_transfers and actual_transfer_matrix != planned_transfers:
            raise RuntimeError(
                "Measured token-transfer loads differ from the generated route: "
                f"expected {planned_transfers}, got {actual_transfer_matrix}"
            )
        planned_fanouts = rows[0].get("planned_fanout_counts")
        if planned_fanouts and actual_fanout_counts != planned_fanouts:
            raise RuntimeError(
                "Measured token fan-out histogram differs from the generated "
                f"route: expected {planned_fanouts}, got {actual_fanout_counts}"
            )
        planned_route_states = rows[0].get("planned_route_state_counts")
        if planned_route_states and actual_route_state_counts != planned_route_states:
            raise RuntimeError(
                "Measured token route-state histogram differs from the generated "
                f"route: expected {planned_route_states}, "
                f"got {actual_route_state_counts}"
            )
        dispatch_row = max(rows, key=lambda row: row["dispatch_ms"])
        compute_row = max(rows, key=lambda row: row["expert_compute_ms"])
        combine_row = max(rows, key=lambda row: row["combine_ms"])
        total_row = max(rows, key=lambda row: row["total_ms"])
        detail_phase_names = sorted(
            {phase for row in rows for phase in (row["detail_timings_ms"] or {})}
        )
        for row in rows:
            row_phases = set(row["detail_timings_ms"] or {})
            if row_phases != set(detail_phase_names):
                raise RuntimeError(
                    "DeepEP-HT detail phases differ across ranks for iteration "
                    f"{iteration}: rank {row['rank']} has {sorted(row_phases)}, "
                    f"expected {detail_phase_names}"
                )
        detail_max_ms = {}
        detail_max_rank = {}
        for phase in detail_phase_names:
            phase_row = max(
                rows,
                key=lambda row: (row["detail_timings_ms"] or {}).get(phase, 0.0),
            )
            detail_max_ms[phase] = phase_row["detail_timings_ms"][phase]
            detail_max_rank[phase] = phase_row["rank"]
        aggregates.append(
            {
                "record_type": "aggregate",
                "execution_mode": rows[0]["execution_mode"],
                "distribution": rows[0]["distribution"],
                "target_share": rows[0]["target_share"],
                "hot_share": rows[0]["hot_share"],
                "local_share": rows[0]["local_share"],
                "iter": iteration,
                "profile_sample": rows[0]["profile_sample"],
                "tokens_per_rank": tokens_per_rank,
                "top_k": rows[0]["top_k"],
                "experts_per_rank": rows[0]["experts_per_rank"],
                "max_dispatch_ms": max(dispatch),
                "max_dispatch_rank": dispatch_row["rank"],
                "max_expert_compute_ms": max(compute),
                "max_expert_compute_rank": compute_row["rank"],
                "max_combine_ms": max(combine),
                "max_combine_rank": combine_row["rank"],
                "max_total_ms": max(totals),
                "max_total_rank": total_row["rank"],
                "received_tokens_min": min(received),
                "received_tokens_max": max(received),
                "received_tokens_mean": received_mean,
                "compute_load_max_over_mean": (
                    max(received) / received_mean if received_mean else 0.0
                ),
                "actual_expert_load_max_over_mean": actual_expert_load_ratio,
                "actual_destination_comm_max_over_mean": (
                    actual_destination_comm_ratio
                ),
                "actual_link_comm_max_over_mean": actual_link_comm_ratio,
                "actual_remote_transfer_matrix": actual_transfer_matrix,
                "actual_fanout_counts": actual_fanout_counts,
                "actual_fanout_shares": actual_fanout_shares,
                "actual_route_state_counts": actual_route_state_counts,
                "actual_route_state_shares": actual_route_state_shares,
                "actual_local_presence_share": actual_local_presence_share,
                "actual_mixed_local_remote_share": (actual_mixed_local_remote_share),
                "actual_remote_only_share": actual_remote_only_share,
                "actual_local_share_min": min(actual_local_shares),
                "actual_local_share_max": max(actual_local_shares),
                "actual_local_share": fully_local_tokens / source_tokens,
                "source_tokens_total": source_tokens,
                "fully_local_tokens_total": fully_local_tokens,
                "remote_tokens_total": remote_tokens,
                "remote_token_transfers_total": remote_unique_tokens,
                "actual_remote_fanout": (
                    remote_unique_tokens / remote_tokens if remote_tokens else 0.0
                ),
                "local_path_tokens_total": local_path_tokens,
                "deepep_source_tokens_total": deepep_source_tokens,
                "remote_unique_tokens_total": remote_unique_tokens,
                "remote_payload_bytes_total": remote_payload_bytes,
                "detail_max_ms": detail_max_ms,
                "detail_max_rank": detail_max_rank,
                **case_metadata,
            }
        )
    return aggregates


def trimmed_records(
    records: list[dict[str, Any]], trim_ratio: float
) -> list[dict[str, Any]]:
    trim_count = int(len(records) * trim_ratio)
    if trim_count == 0:
        return records
    return sorted(records, key=lambda row: row["max_total_ms"])[trim_count:-trim_count]


def summarize(aggregates: list[dict[str, Any]], trim_ratio: float) -> dict[str, Any]:
    kept = trimmed_records(aggregates, trim_ratio)
    if not kept:
        raise ValueError("--trim-ratio removed every measured iteration")
    detail_phase_names = sorted(
        {phase for row in kept for phase in row["detail_max_ms"]}
    )
    for row in kept:
        row_phases = set(row["detail_max_ms"])
        if row_phases != set(detail_phase_names):
            raise RuntimeError(
                "DeepEP-HT detail phases differ across iterations for "
                f"{row['distribution']}={row['target_share']}: iteration "
                f"{row['iter']} has "
                f"{sorted(row_phases)}, expected {detail_phase_names}"
            )
    case_metadata = {
        field: aggregates[0][field]
        for field in LOAD_CASE_METADATA_FIELDS
        if field in aggregates[0]
    }
    return {
        "record_type": "summary",
        "execution_mode": aggregates[0]["execution_mode"],
        "distribution": aggregates[0]["distribution"],
        "target_share": aggregates[0]["target_share"],
        "hot_share": aggregates[0]["hot_share"],
        "local_share": aggregates[0]["local_share"],
        "tokens_per_rank": aggregates[0]["tokens_per_rank"],
        "top_k": aggregates[0]["top_k"],
        "experts_per_rank": aggregates[0]["experts_per_rank"],
        "iters": len(aggregates),
        "trimmed_iters": len(kept),
        "dispatch_ms": statistics.mean(row["max_dispatch_ms"] for row in kept),
        "compute_ms": statistics.mean(row["max_expert_compute_ms"] for row in kept),
        "combine_ms": statistics.mean(row["max_combine_ms"] for row in kept),
        "communication_ms": statistics.mean(
            row["max_dispatch_ms"] + row["max_combine_ms"] for row in kept
        ),
        "total_ms": statistics.mean(row["max_total_ms"] for row in kept),
        "received_tokens_min": min(row["received_tokens_min"] for row in kept),
        "received_tokens_max": max(row["received_tokens_max"] for row in kept),
        "received_tokens_mean": statistics.mean(
            row["received_tokens_mean"] for row in kept
        ),
        "compute_load_max_over_mean": statistics.mean(
            row["compute_load_max_over_mean"] for row in kept
        ),
        "actual_expert_load_max_over_mean": statistics.mean(
            row["actual_expert_load_max_over_mean"] for row in kept
        ),
        "actual_destination_comm_max_over_mean": statistics.mean(
            row["actual_destination_comm_max_over_mean"] for row in kept
        ),
        "actual_link_comm_max_over_mean": statistics.mean(
            row["actual_link_comm_max_over_mean"] for row in kept
        ),
        "actual_remote_transfer_matrix": kept[0]["actual_remote_transfer_matrix"],
        "actual_fanout_counts": kept[0]["actual_fanout_counts"],
        "actual_fanout_shares": kept[0]["actual_fanout_shares"],
        "actual_route_state_counts": kept[0]["actual_route_state_counts"],
        "actual_route_state_shares": kept[0]["actual_route_state_shares"],
        "actual_local_presence_share": statistics.mean(
            row["actual_local_presence_share"] for row in kept
        ),
        "actual_mixed_local_remote_share": statistics.mean(
            row["actual_mixed_local_remote_share"] for row in kept
        ),
        "actual_remote_only_share": statistics.mean(
            row["actual_remote_only_share"] for row in kept
        ),
        "actual_local_share_min": min(row["actual_local_share_min"] for row in kept),
        "actual_local_share_max": max(row["actual_local_share_max"] for row in kept),
        "actual_local_share": statistics.mean(
            row["actual_local_share"] for row in kept
        ),
        "source_tokens": statistics.mean(row["source_tokens_total"] for row in kept),
        "fully_local_tokens": statistics.mean(
            row["fully_local_tokens_total"] for row in kept
        ),
        "remote_tokens": statistics.mean(row["remote_tokens_total"] for row in kept),
        "remote_token_transfers": statistics.mean(
            row["remote_token_transfers_total"] for row in kept
        ),
        "actual_remote_fanout": statistics.mean(
            row["actual_remote_fanout"] for row in kept
        ),
        "local_path_tokens": statistics.mean(
            row["local_path_tokens_total"] for row in kept
        ),
        "deepep_source_tokens": statistics.mean(
            row["deepep_source_tokens_total"] for row in kept
        ),
        "remote_unique_tokens": statistics.mean(
            row["remote_unique_tokens_total"] for row in kept
        ),
        "remote_payload_mib": statistics.mean(
            row["remote_payload_bytes_total"] for row in kept
        )
        / (1024 * 1024),
        "detail_phase_ms": {
            phase: statistics.mean(row["detail_max_ms"][phase] for row in kept)
            for phase in detail_phase_names
        },
        "profile_sample_start": aggregates[0]["profile_sample"],
        "profile_sample_end": aggregates[-1]["profile_sample"],
        **case_metadata,
    }


def print_summaries(summaries: list[dict[str, Any]]) -> None:
    print(
        "execution_mode scenario variant fully_local local_present local_remote "
        "remote_only route_shares remote_fanout dispatch_ms compute_ms combine_ms "
        "communication_ms total_ms planned_rank_load "
        "actual_rank_load planned_expert_load actual_expert_load "
        "planned_destination_comm actual_destination_comm planned_link_comm "
        "actual_link_comm recv_min recv_max remote_transfers remote_mib "
        "detail_samples"
    )
    for row in summaries:
        route_shares = ",".join(
            f"L{local_present}F{fanout}:{share:.3f}"
            for local_present, state_shares in enumerate(
                row["actual_route_state_shares"]
            )
            for fanout, share in enumerate(state_shares)
            if share > 0
        )
        sample_range = (
            f"{row['profile_sample_start']}:{row['profile_sample_end']}"
            if row["profile_sample_start"] is not None
            else "disabled"
        )
        print(
            f"{row['execution_mode']:>14} "
            f"{row['scenario']:>36} "
            f"{row['variant']:>13} "
            f"{row['actual_local_share_min']:>5.3f}:"
            f"{row['actual_local_share_max']:<5.3f} "
            f"{row['actual_local_presence_share']:>13.3f} "
            f"{row['actual_mixed_local_remote_share']:>12.3f} "
            f"{row['actual_remote_only_share']:>11.3f} "
            f"{route_shares:>28} "
            f"{row['actual_remote_fanout']:>13.3f} "
            f"{row['dispatch_ms']:>11.3f} "
            f"{row['compute_ms']:>10.3f} "
            f"{row['combine_ms']:>10.3f} "
            f"{row['communication_ms']:>15.3f} "
            f"{row['total_ms']:>8.3f} "
            f"{row['planned_load_max_over_mean']:>12.3f} "
            f"{row['compute_load_max_over_mean']:>16.3f} "
            f"{row['planned_expert_load_max_over_mean']:>19.3f} "
            f"{row['actual_expert_load_max_over_mean']:>18.3f} "
            f"{row['planned_destination_comm_max_over_mean']:>24.3f} "
            f"{row['actual_destination_comm_max_over_mean']:>23.3f} "
            f"{row['planned_link_comm_max_over_mean']:>17.3f} "
            f"{row['actual_link_comm_max_over_mean']:>16.3f} "
            f"{row['received_tokens_min']:>8} "
            f"{row['received_tokens_max']:>8} "
            f"{row['remote_token_transfers']:>16.1f} "
            f"{row['remote_payload_mib']:>10.3f} "
            f"{sample_range}"
        )

    detail_phases = [
        "dispatch_layout",
        "dispatch_exchange",
        "dispatch_completion_wait",
        "dispatch_remap_topk_ids",
        "dispatch_build_metadata",
        "dispatch_post_quantize",
        "combine_weight_and_reduce",
        "combine_exchange",
        "combine_completion_wait",
        "combine_output_copy",
    ]
    if any(row["detail_phase_ms"] for row in summaries):
        print()
        print("DeepEP-HT detail: mean of the per-iteration max rank (ms)")
        print("scenario " + " ".join(detail_phases))
        for row in summaries:
            timings = row["detail_phase_ms"]
            values = " ".join(
                f"{timings.get(phase, 0.0):.4f}" for phase in detail_phases
            )
            print(f"{row['scenario']} {values}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def run_worker(
    args: argparse.Namespace,
    cases: list[LoadCase],
    token_counts: list[int],
) -> list[dict[str, Any]]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 2:
        raise ValueError("DeepEP-HT distribution benchmark requires WORLD_SIZE >= 2")
    if not torch.cuda.is_available():
        raise RuntimeError("DeepEP-HT distribution benchmark requires CUDA")
    if args.num_experts % world_size != 0:
        raise ValueError("--num-experts must be divisible by WORLD_SIZE")
    if args.top_k > args.num_experts:
        raise ValueError("--top-k cannot exceed --num-experts")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    rounded_hidden_size = DeepEPHTPrepareAndFinalize.maybe_roundup_layer_hidden_size(
        args.hidden_size, dtype
    )
    if rounded_hidden_size != args.hidden_size:
        raise ValueError(
            "--hidden-size is not aligned for DeepEP-HT with bfloat16; "
            f"use {rounded_hidden_size} instead"
        )
    local_args = argparse.Namespace(**vars(args))
    local_args.tokens = token_counts[rank]
    vllm_config = make_vllm_config(world_size, rank, local_rank)
    all_output_rows: list[dict[str, Any]] = []
    summaries = []
    profile_warmup = len(cases) * args.warmup
    if rank == 0:
        print(
            "benchmark_config "
            f"model={args.model or 'custom'} "
            f"tokens_per_rank={token_counts} "
            f"hidden_size={args.hidden_size} "
            f"intermediate_size={args.intermediate_size} "
            f"num_experts={args.num_experts} "
            f"top_k={args.top_k}"
        )
        for case in cases:
            print(
                "load_case "
                f"scenario={case.name} kind={case.kind} pattern={case.pattern} "
                f"variant={case.variant} compute_pattern={case.compute_pattern} "
                f"communication_pattern={case.communication_pattern} "
                "target_shares="
                f"{list(case.planned_target_shares)} "
                f"local_share={case.planned_local_share_min:.3f}:"
                f"{case.planned_local_share_max:.3f} "
                f"route_state_shares={case.planned_route_state_shares} "
                f"remote_fanout={case.planned_remote_fanout:.3f} "
                f"rank_load={case.planned_load_max_over_mean:.3f} "
                f"expert_load={case.planned_expert_load_max_over_mean:.3f} "
                "destination_comm_load="
                f"{case.planned_destination_comm_max_over_mean:.3f} "
                f"link_comm_load={case.planned_link_comm_max_over_mean:.3f} "
                f"fingerprint={case.routing_fingerprint}"
            )

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
                local_args,
                vllm_config,
                dtype,
                device,
                max_num_tokens=max(token_counts),
            )
            tensors = make_base_tensors(local_args, rank, world_size, dtype, device)
            num_tokens_across_dp = torch.tensor(
                token_counts,
                device=device,
                dtype=torch.int,
            )

            with set_forward_context(
                None,
                vllm_config,
                num_tokens=local_args.tokens,
                num_tokens_across_dp=num_tokens_across_dp,
            ):

                def run_iteration(
                    case: LoadCase,
                    topk_ids: torch.Tensor,
                    case_index: int,
                    iteration: int,
                    *,
                    capture_output: bool = False,
                ) -> dict[str, Any]:
                    if args.execution_mode == "local_bypass":
                        return run_local_bypass_iter(
                            local_args,
                            deepep_kernel,
                            local_kernel,
                            tensors,
                            topk_ids,
                            distribution=case.kind,
                            target_share=case.control_value,
                            iteration=iteration,
                            rank=rank,
                            world_size=world_size,
                            device=device,
                            capture_output=capture_output,
                            case=case,
                        )
                    return run_one_iter(
                        local_args,
                        deepep_kernel,
                        tensors,
                        topk_ids,
                        distribution=case.kind,
                        target_share=case.control_value,
                        sweep_index=case_index,
                        iteration=iteration,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        profile_warmup=profile_warmup,
                        capture_output=capture_output,
                        case=case,
                    )

                if args.execution_mode == "local_bypass" and args.validate_output:
                    for case in cases:
                        topk_ids = make_case_topk_ids(
                            case,
                            tokens=local_args.tokens,
                            top_k=args.top_k,
                            num_experts=args.num_experts,
                            source_rank=rank,
                            world_size=world_size,
                            device=device,
                        )
                        reference = run_one_iter(
                            local_args,
                            deepep_kernel,
                            tensors,
                            topk_ids,
                            distribution=case.kind,
                            target_share=case.control_value,
                            sweep_index=-1,
                            iteration=-1,
                            rank=rank,
                            world_size=world_size,
                            device=device,
                            profile_warmup=profile_warmup,
                            capture_output=True,
                            case=case,
                        )["output"]
                        candidate = run_iteration(
                            case,
                            topk_ids,
                            -1,
                            -1,
                            capture_output=True,
                        )["output"]
                        torch.testing.assert_close(
                            reference,
                            candidate,
                            atol=6e-2,
                            rtol=6e-2,
                        )
                    if rank == 0:
                        print("Validated local_bypass outputs against full_deepep")

                case_topk_ids = []
                for case_index, case in enumerate(cases):
                    topk_ids = make_case_topk_ids(
                        case,
                        tokens=local_args.tokens,
                        top_k=args.top_k,
                        num_experts=args.num_experts,
                        source_rank=rank,
                        world_size=world_size,
                        device=device,
                    )
                    case_topk_ids.append(topk_ids)
                    for _ in range(args.warmup):
                        run_iteration(
                            case,
                            topk_ids,
                            case_index,
                            -1,
                        )
                dist.barrier()

                for case_index, (case, topk_ids) in enumerate(
                    zip(cases, case_topk_ids)
                ):
                    local_rows = [
                        run_iteration(
                            case,
                            topk_ids,
                            case_index,
                            iteration,
                        )
                        for iteration in range(args.iters)
                    ]
                    gathered: list[Any] | None = (
                        [None] * world_size if rank == 0 else None
                    )
                    dist.gather_object(local_rows, gathered, dst=0)
                    if rank == 0:
                        assert gathered is not None
                        rank_rows = [row for rank_rows in gathered for row in rank_rows]
                        aggregates = aggregate_records(rank_rows)
                        summary = summarize(aggregates, args.trim_ratio)
                        summaries.append(summary)
                        all_output_rows.extend(rank_rows)
                        all_output_rows.extend(aggregates)
                        all_output_rows.append(summary)
                    dist.barrier()

            if rank == 0:
                print_summaries(summaries)
                if args.output_jsonl is not None:
                    write_jsonl(args.output_jsonl, all_output_rows)
                    print(f"Wrote JSONL: {args.output_jsonl}")
    finally:
        if dist.is_initialized():
            cleanup_dist_env_and_memory()
    return summaries if rank == 0 else []


def make_comparisons(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    compute = [row for row in summaries if row.get("benchmark_kind") == "compute"]
    balanced = next((row for row in compute if row["variant"] == "balanced"), None)
    if balanced is not None:
        for imbalanced in compute:
            if imbalanced["variant"] == "imbalanced":
                comparisons.append(
                    _comparison_record(
                        "compute",
                        balanced,
                        imbalanced,
                        "compute_ms",
                    )
                )

    communication = [
        row for row in summaries if row.get("benchmark_kind") == "communication"
    ]
    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in communication:
        by_pattern[row["pattern"]].append(row)
    for rows in by_pattern.values():
        local = [row for row in rows if row["variant"] == "mostly_local"]
        remote = [row for row in rows if row["variant"] == "mostly_remote"]
        if local and remote:
            comparisons.append(
                _comparison_record(
                    "communication",
                    max(local, key=lambda row: row["actual_local_share"]),
                    min(remote, key=lambda row: row["actual_local_share"]),
                    "communication_ms",
                )
            )

    joint = [row for row in summaries if row.get("benchmark_kind") == "joint"]
    by_route: dict[tuple[tuple[int, ...], ...], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in joint:
        route_counts = tuple(
            tuple(state_counts) for state_counts in row["planned_route_state_counts"]
        )
        by_route[route_counts].append(row)
    for rows in by_route.values():
        baseline = next(
            (
                row
                for row in rows
                if row["compute_pattern"] == "balanced"
                and row["communication_pattern"] in ("balanced", "saturated")
            ),
            None,
        )
        if baseline is None:
            continue
        for comparison in rows:
            if comparison is baseline:
                continue
            comparisons.append(
                _comparison_record("joint", baseline, comparison, "total_ms")
            )
    return comparisons


def _comparison_record(
    kind: str,
    baseline: dict[str, Any],
    comparison: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    baseline_ms = float(baseline[metric])
    comparison_ms = float(comparison[metric])
    return {
        "record_type": "comparison",
        "kind": kind,
        "pattern": comparison["pattern"],
        "metric": metric,
        "baseline": baseline["scenario"],
        "baseline_control_value": baseline["control_value"],
        "baseline_ms": baseline_ms,
        "comparison": comparison["scenario"],
        "comparison_control_value": comparison["control_value"],
        "comparison_ms": comparison_ms,
        "delta_ms": comparison_ms - baseline_ms,
        "ratio": comparison_ms / baseline_ms if baseline_ms else None,
    }


def print_comparisons(comparisons: list[dict[str, Any]]) -> None:
    print(
        "comparison kind pattern metric baseline comparison baseline_ms "
        "comparison_ms delta_ms ratio"
    )
    for row in comparisons:
        ratio = "undefined" if row["ratio"] is None else f"{row['ratio']:.3f}"
        print(
            f"comparison {row['kind']} {row['pattern']} {row['metric']} "
            f"{row['baseline']} {row['comparison']} {row['baseline_ms']:.3f} "
            f"{row['comparison_ms']:.3f} {row['delta_ms']:.3f} {ratio}"
        )


def main() -> None:
    args = parse_args()
    if args.iters <= 0 or args.warmup < 0:
        raise ValueError("--iters must be positive and --warmup non-negative")
    if (
        min(
            args.tokens,
            args.hidden_size,
            args.intermediate_size,
            args.num_experts,
            args.top_k,
        )
        <= 0
    ):
        raise ValueError(
            "--tokens, --hidden-size, --intermediate-size, --num-experts, and "
            "--top-k must be positive"
        )
    if not 0.0 <= args.trim_ratio < 0.5:
        raise ValueError("--trim-ratio must be in [0, 0.5)")
    if not 0.0 <= args.imbalance_strength <= 1.0:
        raise ValueError("--imbalance-strength must be in [0, 1]")
    if args.zipf_alpha <= 0 or args.random_alpha <= 0:
        raise ValueError("--zipf-alpha and --random-alpha must be positive")
    if args.random_cases < 0:
        raise ValueError("--random-cases must be non-negative")
    if not all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise RuntimeError("Launch this benchmark with torchrun")
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise ValueError("DeepEP-HT distribution benchmark requires WORLD_SIZE >= 2")
    if args.num_experts % world_size != 0:
        raise ValueError("--num-experts must be divisible by WORLD_SIZE")
    if args.top_k > args.num_experts:
        raise ValueError("--top-k cannot exceed --num-experts")
    token_counts = parse_token_counts(args.tokens_per_rank, args.tokens, world_size)
    cases = build_load_cases(args, token_counts, world_size)
    if args.execution_mode == "local_bypass":
        if any(case.kind != "communication" for case in cases):
            raise ValueError("local_bypass only supports communication cases")
        if args.detail_profile:
            raise ValueError("local_bypass requires --no-detail-profile")

    if args.detail_profile:
        os.environ["VLLM_DEEPEP_HT_PROFILE"] = "1"
        os.environ["VLLM_DEEPEP_HT_PROFILE_LOG"] = "0"
        os.environ["VLLM_DEEPEP_HT_PROFILE_WARMUP"] = str(len(cases) * args.warmup)
        os.environ["VLLM_DEEPEP_HT_PROFILE_SAMPLES"] = str(len(cases) * args.iters)
    else:
        os.environ["VLLM_DEEPEP_HT_PROFILE"] = "0"
    summaries = run_worker(args, cases, token_counts)

    if int(os.environ["RANK"]) == 0:
        comparisons = make_comparisons(summaries)
        if comparisons:
            print_comparisons(comparisons)
            if args.output_jsonl is not None:
                comparison_path = args.output_jsonl.with_name(
                    f"{args.output_jsonl.stem}_comparison{args.output_jsonl.suffix}"
                )
                write_jsonl(comparison_path, comparisons)
                print(f"Wrote JSONL: {comparison_path}")


if __name__ == "__main__":
    main()
