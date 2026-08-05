# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DeepEP-HT dispatch, expert compute, and combine by rank distribution.

Run with torchrun. The benchmark constructs physical top-k expert IDs directly.
Use ``--local-shares`` to keep that fraction of each source rank's tokens on the
same rank and spread the remainder evenly across the other ranks::

    .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
        benchmarks/kernels/benchmark_deepep_ht_distribution.py \
        --local-shares 0,0.25,0.5,0.75,1 \
        --output-jsonl /tmp/deepep_ht_detail.jsonl

Use ``--no-detail-profile`` for wrapper timings without the diagnostic device
synchronizations inserted inside DeepEP prepare/finalize.

Use ``--model`` to select the MoE dimensions of a supported model without
loading its weights::

    --model Qwen/Qwen3-30B-A3B
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    parser.add_argument("--tokens", type=int, default=4096)
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
            "In hot-rank mode, a token routed to multiple ranks is transmitted "
            "multiple times. Local-share mode keeps all top-k experts for a "
            "token on one rank, so its activation is transmitted at most once."
        ),
    )
    parser.add_argument("--hot-rank", type=int, default=0)
    parser.add_argument(
        "--hot-shares",
        default=None,
        help="Comma-separated fractions of assignments targeting --hot-rank.",
    )
    parser.add_argument(
        "--local-shares",
        default=None,
        help=(
            "Comma-separated fractions of each source rank's tokens that target "
            "experts on the same rank. Remaining tokens are spread evenly over "
            "the other ranks. Mutually exclusive with --hot-shares."
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


def sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def align_stage(device: torch.device, use_barrier: bool) -> None:
    sync(device)
    if use_barrier:
        dist.barrier()
        sync(device)


def time_stage(
    device: torch.device,
    fn,
    *,
    use_barrier: bool,
) -> tuple[Any, float]:
    align_stage(device, use_barrier)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return result, (time.perf_counter() - start) * 1000.0


def apportion(total: int, weights: list[float], offset: int) -> list[int]:
    raw_counts = [total * weight for weight in weights]
    counts = [int(count) for count in raw_counts]
    remainder = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (
            raw_counts[index] - counts[index],
            -((index - offset) % len(weights)),
        ),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def make_ids_from_rank_counts(
    tokens: int,
    top_k: int,
    num_local_experts: int,
    source_rank: int,
    counts: list[int],
) -> torch.Tensor:
    total = tokens * top_k
    if sum(counts) != total:
        raise ValueError(f"Rank counts must sum to {total}, got {sum(counts)}")

    targets = []
    remaining = counts[:]
    world_size = len(counts)
    while sum(remaining) > 0:
        for offset in range(world_size):
            target = (source_rank + offset) % world_size
            if remaining[target] > 0:
                targets.append(target)
                remaining[target] -= 1

    seen_per_rank = [0] * world_size
    expert_ids = []
    for token_start in range(0, total, top_k):
        used_experts: set[int] = set()
        for target in targets[token_start : token_start + top_k]:
            for _ in range(num_local_experts):
                local_expert = seen_per_rank[target] % num_local_experts
                seen_per_rank[target] += 1
                expert_id = target * num_local_experts + local_expert
                if expert_id not in used_experts:
                    used_experts.add(expert_id)
                    expert_ids.append(expert_id)
                    break
            else:
                raise ValueError(
                    "The requested distribution cannot provide unique top-k "
                    "experts per token. Increase --num-experts, reduce --top-k, "
                    "or use a less extreme hot share."
                )
    return torch.tensor(expert_ids, dtype=torch.int64).view(tokens, top_k)


def make_topk_ids(
    *,
    tokens: int,
    top_k: int,
    num_experts: int,
    source_rank: int,
    world_size: int,
    hot_rank: int,
    hot_share: float,
    device: torch.device,
) -> torch.Tensor:
    num_local_experts = num_experts // world_size
    total = tokens * top_k
    hot_rank %= world_size
    weights = [0.0] * world_size
    weights[hot_rank] = hot_share
    if world_size == 1:
        weights[0] = 1.0
    else:
        other_share = (1.0 - hot_share) / (world_size - 1)
        for rank in range(world_size):
            if rank != hot_rank:
                weights[rank] = other_share
    counts = apportion(total, weights, offset=source_rank)
    return make_ids_from_rank_counts(
        tokens,
        top_k,
        num_local_experts,
        source_rank,
        counts,
    ).to(device)


def make_locality_topk_ids(
    *,
    tokens: int,
    top_k: int,
    num_experts: int,
    source_rank: int,
    world_size: int,
    local_share: float,
    device: torch.device,
) -> torch.Tensor:
    num_local_experts = num_experts // world_size
    if top_k > num_local_experts:
        raise ValueError(
            "--local-shares requires at least --top-k experts on every rank"
        )

    weights = [(1.0 - local_share) / (world_size - 1)] * world_size
    weights[source_rank] = local_share
    counts = apportion(tokens, weights, offset=source_rank)

    targets = []
    remaining = counts[:]
    while sum(remaining) > 0:
        for offset in range(world_size):
            target = (source_rank + offset) % world_size
            if remaining[target] > 0:
                targets.append(target)
                remaining[target] -= 1

    seen_per_rank = [0] * world_size
    expert_ids = []
    for target in targets:
        start = seen_per_rank[target] % num_local_experts
        seen_per_rank[target] += top_k
        expert_ids.extend(
            target * num_local_experts + (start + offset) % num_local_experts
            for offset in range(top_k)
        )
    return torch.tensor(expert_ids, dtype=torch.int64).view(tokens, top_k).to(device)


def rank_distribution(
    topk_ids: torch.Tensor,
    num_local_experts: int,
    world_size: int,
) -> tuple[list[int], list[int]]:
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
    return [int(value) for value in assignments], unique_tokens


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
        max_num_tokens=next_power_of_2(args.tokens),
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
) -> dict[str, Any]:
    assert isinstance(kernel.impl, mk.FusedMoEKernelModularImpl)
    requested_dtype = kernel.prepare_finalize.topk_indices_dtype()
    if requested_dtype is not None:
        topk_ids = topk_ids.to(requested_dtype)

    hidden_states = tensors["hidden_states"]
    topk_weights = tensors["topk_weights"]
    output = torch.empty_like(hidden_states)
    local_num_experts = tensors["w1"].shape[0]
    target_assignments, target_unique_tokens = rank_distribution(
        topk_ids, local_num_experts, world_size
    )
    remote_assignments = sum(target_assignments) - target_assignments[rank]
    remote_unique_tokens = sum(target_unique_tokens) - target_unique_tokens[rank]
    actual_local_share = (
        target_unique_tokens[rank] / args.tokens if args.tokens else 0.0
    )

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
        "hot_share": target_share if distribution == "hot_rank" else None,
        "local_share": target_share if distribution == "local_share" else None,
        "iter": iteration,
        "rank": rank,
        "world_size": world_size,
        "profile_sample": profile_sample if args.detail_profile else None,
        "dispatch_ms": dispatch_ms,
        "expert_compute_ms": compute_ms,
        "combine_ms": combine_ms,
        "total_ms": dispatch_ms + compute_ms + combine_ms,
        "source_target_assignments": target_assignments,
        "source_target_unique_tokens": target_unique_tokens,
        "remote_assignments": remote_assignments,
        "remote_unique_tokens": remote_unique_tokens,
        "local_path_tokens": 0,
        "deepep_source_tokens": args.tokens,
        "actual_local_share": actual_local_share,
        "remote_payload_bytes": remote_unique_tokens
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
) -> dict[str, Any]:
    assert isinstance(deepep_kernel.impl, mk.FusedMoEKernelModularImpl)
    assert isinstance(local_kernel.impl, mk.FusedMoEKernelModularImpl)
    requested_dtype = deepep_kernel.prepare_finalize.topk_indices_dtype()
    if requested_dtype is not None:
        topk_ids = topk_ids.to(requested_dtype)

    hidden_states = tensors["hidden_states"]
    topk_weights = tensors["topk_weights"]
    local_num_experts = tensors["w1"].shape[0]
    target_assignments, target_unique_tokens = rank_distribution(
        topk_ids, local_num_experts, world_size
    )
    remote_assignments = sum(target_assignments) - target_assignments[rank]
    remote_unique_tokens = sum(target_unique_tokens) - target_unique_tokens[rank]
    actual_local_share = (
        target_unique_tokens[rank] / args.tokens if args.tokens else 0.0
    )

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
        "profile_sample": None,
        "dispatch_ms": dispatch_ms,
        "expert_compute_ms": compute_ms,
        "combine_ms": combine_ms,
        "total_ms": dispatch_ms + compute_ms + combine_ms,
        "source_target_assignments": target_assignments,
        "source_target_unique_tokens": target_unique_tokens,
        "remote_assignments": remote_assignments,
        "remote_unique_tokens": remote_unique_tokens,
        "local_path_tokens": local_path_tokens,
        "deepep_source_tokens": deepep_source_tokens,
        "actual_local_share": actual_local_share,
        "remote_payload_bytes": remote_unique_tokens
        * args.hidden_size
        * hidden_states.element_size(),
        "active_destinations": sum(count > 0 for count in target_unique_tokens),
        "received_tokens": sum(local_expert_tokens_list),
        "local_expert_tokens": local_expert_tokens_list,
        "detail_timings_ms": None,
        "detail_metadata": None,
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
        actual_local_shares = [row["actual_local_share"] for row in rows]
        local_path_tokens = sum(row["local_path_tokens"] for row in rows)
        deepep_source_tokens = sum(row["deepep_source_tokens"] for row in rows)
        remote_payload_bytes = sum(row["remote_payload_bytes"] for row in rows)
        remote_unique_tokens = sum(row["remote_unique_tokens"] for row in rows)
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
                "actual_local_share_min": min(actual_local_shares),
                "actual_local_share_max": max(actual_local_shares),
                "local_path_tokens_total": local_path_tokens,
                "deepep_source_tokens_total": deepep_source_tokens,
                "remote_unique_tokens_total": remote_unique_tokens,
                "remote_payload_bytes_total": remote_payload_bytes,
                "detail_max_ms": detail_max_ms,
                "detail_max_rank": detail_max_rank,
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
    return {
        "record_type": "summary",
        "execution_mode": aggregates[0]["execution_mode"],
        "distribution": aggregates[0]["distribution"],
        "target_share": aggregates[0]["target_share"],
        "hot_share": aggregates[0]["hot_share"],
        "local_share": aggregates[0]["local_share"],
        "iters": len(aggregates),
        "trimmed_iters": len(kept),
        "dispatch_ms": statistics.mean(row["max_dispatch_ms"] for row in kept),
        "compute_ms": statistics.mean(row["max_expert_compute_ms"] for row in kept),
        "combine_ms": statistics.mean(row["max_combine_ms"] for row in kept),
        "total_ms": statistics.mean(row["max_total_ms"] for row in kept),
        "received_tokens_min": min(row["received_tokens_min"] for row in kept),
        "received_tokens_max": max(row["received_tokens_max"] for row in kept),
        "actual_local_share_min": min(row["actual_local_share_min"] for row in kept),
        "actual_local_share_max": max(row["actual_local_share_max"] for row in kept),
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
    }


def print_summaries(summaries: list[dict[str, Any]]) -> None:
    print(
        "execution_mode distribution share actual_local dispatch_ms compute_ms "
        "combine_ms total_ms local_tokens deepep_tokens recv_min recv_max "
        "remote_tokens remote_mib detail_samples"
    )
    for row in summaries:
        sample_range = (
            f"{row['profile_sample_start']}:{row['profile_sample_end']}"
            if row["profile_sample_start"] is not None
            else "disabled"
        )
        print(
            f"{row['execution_mode']:>14} "
            f"{row['distribution']:>12} "
            f"{row['target_share']:>5.3f} "
            f"{row['actual_local_share_min']:>5.3f}:"
            f"{row['actual_local_share_max']:<5.3f} "
            f"{row['dispatch_ms']:>11.3f} "
            f"{row['compute_ms']:>10.3f} "
            f"{row['combine_ms']:>10.3f} "
            f"{row['total_ms']:>8.3f} "
            f"{row['local_path_tokens']:>12.1f} "
            f"{row['deepep_source_tokens']:>13.1f} "
            f"{row['received_tokens_min']:>8} "
            f"{row['received_tokens_max']:>8} "
            f"{row['remote_unique_tokens']:>13.1f} "
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
        print("distribution share " + " ".join(detail_phases))
        for row in summaries:
            timings = row["detail_phase_ms"]
            values = " ".join(
                f"{timings.get(phase, 0.0):.4f}" for phase in detail_phases
            )
            print(f"{row['distribution']} {row['target_share']:.3f} {values}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def run_worker(
    args: argparse.Namespace,
    distribution: str,
    target_shares: list[float],
) -> None:
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
    vllm_config = make_vllm_config(world_size, rank, local_rank)
    all_output_rows: list[dict[str, Any]] = []
    summaries = []
    profile_warmup = len(target_shares) * args.warmup
    if rank == 0:
        print(
            "benchmark_config "
            f"model={args.model or 'custom'} "
            f"tokens={args.tokens} "
            f"hidden_size={args.hidden_size} "
            f"intermediate_size={args.intermediate_size} "
            f"num_experts={args.num_experts} "
            f"top_k={args.top_k}"
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
            deepep_kernel, local_kernel = make_kernels(args, vllm_config, dtype, device)
            tensors = make_base_tensors(args, rank, world_size, dtype, device)
            num_tokens_across_dp = torch.full(
                (world_size,), args.tokens, device=device, dtype=torch.int
            )

            with set_forward_context(
                None,
                vllm_config,
                num_tokens=args.tokens,
                num_tokens_across_dp=num_tokens_across_dp,
            ):
                if distribution == "local_share":
                    sweep_topk_ids = [
                        make_locality_topk_ids(
                            tokens=args.tokens,
                            top_k=args.top_k,
                            num_experts=args.num_experts,
                            source_rank=rank,
                            world_size=world_size,
                            local_share=target_share,
                            device=device,
                        )
                        for target_share in target_shares
                    ]
                else:
                    sweep_topk_ids = [
                        make_topk_ids(
                            tokens=args.tokens,
                            top_k=args.top_k,
                            num_experts=args.num_experts,
                            source_rank=rank,
                            world_size=world_size,
                            hot_rank=args.hot_rank,
                            hot_share=target_share,
                            device=device,
                        )
                        for target_share in target_shares
                    ]

                def run_iteration(
                    topk_ids: torch.Tensor,
                    target_share: float,
                    sweep_index: int,
                    iteration: int,
                    *,
                    capture_output: bool = False,
                ) -> dict[str, Any]:
                    if args.execution_mode == "local_bypass":
                        return run_local_bypass_iter(
                            args,
                            deepep_kernel,
                            local_kernel,
                            tensors,
                            topk_ids,
                            distribution=distribution,
                            target_share=target_share,
                            iteration=iteration,
                            rank=rank,
                            world_size=world_size,
                            device=device,
                            capture_output=capture_output,
                        )
                    return run_one_iter(
                        args,
                        deepep_kernel,
                        tensors,
                        topk_ids,
                        distribution=distribution,
                        target_share=target_share,
                        sweep_index=sweep_index,
                        iteration=iteration,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        profile_warmup=profile_warmup,
                        capture_output=capture_output,
                    )

                if args.execution_mode == "local_bypass" and args.validate_output:
                    for target_share, topk_ids in zip(target_shares, sweep_topk_ids):
                        reference = run_one_iter(
                            args,
                            deepep_kernel,
                            tensors,
                            topk_ids,
                            distribution=distribution,
                            target_share=target_share,
                            sweep_index=-1,
                            iteration=-1,
                            rank=rank,
                            world_size=world_size,
                            device=device,
                            profile_warmup=profile_warmup,
                            capture_output=True,
                        )["output"]
                        candidate = run_iteration(
                            topk_ids,
                            target_share,
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

                for target_share, topk_ids in zip(target_shares, sweep_topk_ids):
                    for _ in range(args.warmup):
                        run_iteration(
                            topk_ids,
                            target_share,
                            -1,
                            -1,
                        )
                dist.barrier()

                for sweep_index, (target_share, topk_ids) in enumerate(
                    zip(target_shares, sweep_topk_ids)
                ):
                    local_rows = [
                        run_iteration(
                            topk_ids,
                            target_share,
                            sweep_index,
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


def main() -> None:
    args = parse_args()
    if args.hot_shares is not None and args.local_shares is not None:
        raise ValueError("--hot-shares and --local-shares are mutually exclusive")
    if args.local_shares is not None:
        distribution = "local_share"
        target_shares = parse_shares(args.local_shares, "--local-shares")
    else:
        distribution = "hot_rank"
        raw_hot_shares = args.hot_shares or "0,0.25,0.5,0.75,1"
        target_shares = parse_shares(raw_hot_shares, "--hot-shares")
    if args.execution_mode == "local_bypass":
        if distribution != "local_share":
            raise ValueError("local_bypass requires --local-shares")
        if args.detail_profile:
            raise ValueError("local_bypass requires --no-detail-profile")
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
    if not all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise RuntimeError("Launch this benchmark with torchrun")

    if args.detail_profile:
        os.environ["VLLM_DEEPEP_HT_PROFILE"] = "1"
        os.environ["VLLM_DEEPEP_HT_PROFILE_LOG"] = "0"
        os.environ["VLLM_DEEPEP_HT_PROFILE_WARMUP"] = str(
            len(target_shares) * args.warmup
        )
        os.environ["VLLM_DEEPEP_HT_PROFILE_SAMPLES"] = str(
            len(target_shares) * args.iters
        )
    else:
        os.environ["VLLM_DEEPEP_HT_PROFILE"] = "0"
    run_worker(args, distribution, target_shares)


if __name__ == "__main__":
    main()
