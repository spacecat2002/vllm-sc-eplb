# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused projections for current and predicted-next MoE gates."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_OUTPUT_WORKSPACES: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def get_dual_gate_output_workspace(
    *,
    device: torch.device,
    dtype: torch.dtype,
    slot: int,
    num_tokens: int,
    current_experts: int,
    next_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return shared contiguous output views for one micro-batch slot."""
    buffers = _OUTPUT_WORKSPACES.get(slot)
    current_elements = num_tokens * current_experts
    next_elements = num_tokens * next_experts
    if buffers is None:
        buffers = (
            torch.empty(current_elements, dtype=dtype, device=device),
            torch.empty(next_elements, dtype=dtype, device=device),
        )
    elif any(
        buffer.device != device or buffer.dtype != dtype for buffer in buffers
    ):
        raise ValueError("dual-gate workspace device and dtype must remain fixed")
    elif buffers[0].numel() < current_elements or buffers[1].numel() < next_elements:
        buffers = (
            torch.empty(
                max(current_elements, buffers[0].numel() * 2),
                dtype=dtype,
                device=device,
            ),
            torch.empty(
                max(next_elements, buffers[1].numel() * 2),
                dtype=dtype,
                device=device,
            ),
        )
    _OUTPUT_WORKSPACES[slot] = buffers
    current_output = buffers[0][:current_elements].view(num_tokens, current_experts)
    next_output = buffers[1][:next_elements].view(num_tokens, next_experts)
    return current_output, next_output


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 64},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=8,
            num_stages=4,
        ),
    ],
    key=["num_tokens", "hidden_size", "current_experts", "next_experts"],
)
@triton.jit
def _dual_gate_lora_kernel(
    hidden_states,
    current_weight,
    next_weight,
    next_delta_weight,
    current_output,
    next_output,
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    current_experts: tl.constexpr,
    next_experts: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_w0n: tl.constexpr,
    stride_w0k: tl.constexpr,
    stride_w1n: tl.constexpr,
    stride_w1k: tl.constexpr,
    stride_dwn: tl.constexpr,
    stride_dwk: tl.constexpr,
    stride_o0m: tl.constexpr,
    stride_o0n: tl.constexpr,
    stride_o1m: tl.constexpr,
    stride_o1n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    is_current = offsets_n < current_experts
    next_offsets = offsets_n - current_experts
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, hidden_size, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(
            hidden_states
            + offsets_m[:, None] * stride_xm
            + offsets_k[None, :] * stride_xk,
            mask=(offsets_m[:, None] < num_tokens) & (offsets_k[None, :] < hidden_size),
            other=0.0,
        )
        current = tl.load(
            current_weight
            + offsets_n[:, None] * stride_w0n
            + offsets_k[None, :] * stride_w0k,
            mask=is_current[:, None]
            & (offsets_n[:, None] < current_experts)
            & (offsets_k[None, :] < hidden_size),
            other=0.0,
        )
        is_next = (~is_current) & (next_offsets < next_experts)
        following = tl.load(
            next_weight
            + next_offsets[:, None] * stride_w1n
            + offsets_k[None, :] * stride_w1k,
            mask=is_next[:, None] & (offsets_k[None, :] < hidden_size),
            other=0.0,
        )
        base_weight = (current + following).to(hidden_states.dtype.element_ty)
        accumulator = tl.dot(x, tl.trans(base_weight), acc=accumulator)
        if pid_n * BLOCK_N + BLOCK_N > current_experts:
            delta = tl.load(
                next_delta_weight
                + next_offsets[:, None] * stride_dwn
                + offsets_k[None, :] * stride_dwk,
                mask=is_next[:, None] & (offsets_k[None, :] < hidden_size),
                other=0.0,
            )
            accumulator = tl.dot(x, tl.trans(delta), acc=accumulator)

    tl.store(
        current_output
        + offsets_m[:, None] * stride_o0m
        + offsets_n[None, :] * stride_o0n,
        accumulator,
        mask=(offsets_m[:, None] < num_tokens) & (offsets_n[None, :] < current_experts),
    )
    tl.store(
        next_output
        + offsets_m[:, None] * stride_o1m
        + next_offsets[None, :] * stride_o1n,
        accumulator,
        mask=(offsets_m[:, None] < num_tokens)
        & (next_offsets[None, :] >= 0)
        & (next_offsets[None, :] < next_experts),
    )


def _validate_weights(
    hidden_states: torch.Tensor,
    current_weight: torch.Tensor,
    next_weight: torch.Tensor,
    next_delta_weight: torch.Tensor,
) -> None:
    tensors = (hidden_states, current_weight, next_weight, next_delta_weight)
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("all dual-gate inputs must be 2D tensors")
    if any(tensor.device != hidden_states.device for tensor in tensors[1:]):
        raise ValueError("all dual-gate inputs must be on the same device")
    if hidden_states.device.type != "cuda":
        raise ValueError("dual_gate_lora requires CUDA tensors")
    if any(tensor.dtype != hidden_states.dtype for tensor in tensors[1:]):
        raise ValueError("all dual-gate inputs must have the same dtype")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("dual_gate_lora supports float16 and bfloat16 inputs")
    hidden_size = hidden_states.shape[1]
    if any(weight.shape[1] != hidden_size for weight in tensors[1:]):
        raise ValueError("gate weights must match the hidden size")
    if next_weight.shape != next_delta_weight.shape:
        raise ValueError("next gate and delta weights must have identical shapes")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all dual-gate inputs must be contiguous")


def triton_dual_gate_lora(
    hidden_states: torch.Tensor,
    current_weight: torch.Tensor,
    next_weight: torch.Tensor,
    next_delta_weight: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    current_output: torch.Tensor | None = None,
    next_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project current and predicted-next gate logits in one kernel launch."""
    _validate_weights(
        hidden_states,
        current_weight,
        next_weight,
        next_delta_weight,
    )
    if output_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("unsupported dual-gate output dtype")
    num_tokens, hidden_size = hidden_states.shape
    current_experts = current_weight.shape[0]
    next_experts = next_weight.shape[0]
    expected_current_shape = (num_tokens, current_experts)
    expected_next_shape = (num_tokens, next_experts)
    if current_output is None:
        current_output = torch.empty(
            expected_current_shape,
            dtype=output_dtype,
            device=hidden_states.device,
        )
    if next_output is None:
        next_output = torch.empty(
            expected_next_shape,
            dtype=output_dtype,
            device=hidden_states.device,
        )
    for name, output, expected_shape in (
        ("current_output", current_output, expected_current_shape),
        ("next_output", next_output, expected_next_shape),
    ):
        if output.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {output.shape}, expected {expected_shape}"
            )
        if output.device != hidden_states.device or output.dtype != output_dtype:
            raise ValueError(f"{name} has an incompatible device or dtype")
        if not output.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    grid = lambda meta: (
        triton.cdiv(num_tokens, meta["BLOCK_M"]),
        triton.cdiv(current_experts + next_experts, meta["BLOCK_N"]),
    )
    _dual_gate_lora_kernel[grid](
        hidden_states,
        current_weight,
        next_weight,
        next_delta_weight,
        current_output,
        next_output,
        num_tokens,
        hidden_size,
        current_experts,
        next_experts,
        hidden_states.stride(0),
        hidden_states.stride(1),
        current_weight.stride(0),
        current_weight.stride(1),
        next_weight.stride(0),
        next_weight.stride(1),
        next_delta_weight.stride(0),
        next_delta_weight.stride(1),
        current_output.stride(0),
        current_output.stride(1),
        next_output.stride(0),
        next_output.stride(1),
    )
    return current_output, next_output


def dual_gate_backend(
    hidden_states: torch.Tensor,
    current_weight: torch.Tensor,
    next_weight: torch.Tensor,
    next_delta_weight: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> str:
    """Return the backend selected for a dual-gate projection."""
    if hidden_states.device.type == "cuda":
        from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
            can_use_ll_bf16_dual_gate_gemm,
        )

        if can_use_ll_bf16_dual_gate_gemm(
            hidden_states,
            current_weight,
            next_weight,
            next_delta_weight,
            output_dtype,
        ):
            return "cutedsl"
    return "triton"


def dual_gate_lora(
    hidden_states: torch.Tensor,
    current_weight: torch.Tensor,
    next_weight: torch.Tensor,
    next_delta_weight: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    current_output: torch.Tensor | None = None,
    next_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to CuTeDSL for small BF16 batches and Triton otherwise."""
    if (
        dual_gate_backend(
            hidden_states,
            current_weight,
            next_weight,
            next_delta_weight,
            output_dtype,
        )
        == "cutedsl"
    ):
        from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
            ll_bf16_dual_gate_gemm,
        )

        return ll_bf16_dual_gate_gemm(
            hidden_states,
            current_weight,
            next_weight,
            next_delta_weight,
            current_output,
            next_output,
        )
    return triton_dual_gate_lora(
        hidden_states,
        current_weight,
        next_weight,
        next_delta_weight,
        output_dtype,
        current_output,
        next_output,
    )


def gate_lora(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    delta_weight: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project one gate with an independent dense LoRA delta weight."""
    empty_current = gate_weight[:0]
    _, logits = dual_gate_lora(
        hidden_states,
        empty_current,
        gate_weight,
        delta_weight,
        next_output=output,
    )
    return logits
