# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark current/next MoE gate projection strategies on CUDA."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.dual_gate_lora import (
    dual_gate_backend,
    dual_gate_lora,
    gate_lora,
)


@dataclass(frozen=True)
class Shape:
    name: str
    hidden_size: int
    experts: int


SHAPES = {
    "qwen": Shape("qwen", 2048, 128),
    "deepseek": Shape("deepseek", 7168, 256),
    "kimi": Shape("kimi", 7168, 384),
}


def _gate(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if hidden_states.dtype == torch.bfloat16:
        try:
            from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
                is_available,
                ll_bf16_gemm,
            )

            capability = torch.cuda.get_device_capability()
            if is_available() and capability[0] >= 9 and hidden_states.shape[0] <= 16:
                return ll_bf16_gemm(hidden_states, weight)
        except ImportError:
            pass
    return torch.mm(hidden_states, weight.T, out_dtype=torch.float32)


def _measure(
    operation: Callable[[], object],
    *,
    warmup: int,
    repetitions: int,
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repetitions


def _parallel_operation(
    hidden_states: torch.Tensor,
    current_weight: torch.Tensor,
    next_weight: torch.Tensor,
    delta_weight: torch.Tensor,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    current_stream = torch.cuda.Stream()
    prediction_stream = torch.cuda.Stream()
    ready = torch.cuda.Event()
    current_done = torch.cuda.Event()
    prediction_done = torch.cuda.Event()
    predicted_output = torch.empty(
        hidden_states.shape[0],
        next_weight.shape[0],
        dtype=torch.float32,
        device=hidden_states.device,
    )

    def operation() -> tuple[torch.Tensor, torch.Tensor]:
        launch_stream = torch.cuda.current_stream()
        ready.record(launch_stream)
        current_stream.wait_event(ready)
        prediction_stream.wait_event(ready)
        with torch.cuda.stream(current_stream):
            current_logits = _gate(hidden_states, current_weight)
            current_done.record(current_stream)
        with torch.cuda.stream(prediction_stream):
            predicted_logits = gate_lora(
                hidden_states,
                next_weight,
                delta_weight,
                output=predicted_output,
            )
            prediction_done.record(prediction_stream)
        launch_stream.wait_event(current_done)
        launch_stream.wait_event(prediction_done)
        return current_logits, predicted_logits

    return operation


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def benchmark_case(
    shape: Shape,
    *,
    tokens: int,
    rank: int,
    warmup: int,
    repetitions: int,
) -> dict[str, float]:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden_states = torch.randn(tokens, shape.hidden_size, device=device, dtype=dtype)
    current_weight = torch.randn(
        shape.experts, shape.hidden_size, device=device, dtype=dtype
    )
    next_weight = torch.randn_like(current_weight)
    lora_a = torch.randn(rank, shape.hidden_size, device=device, dtype=dtype)
    lora_b = torch.randn(shape.experts, rank, device=device, dtype=dtype)
    delta_weight = (lora_b.float() @ lora_a.float()).to(dtype).contiguous()
    merged_next_weight = (next_weight + delta_weight).contiguous()
    concatenated_weight = torch.cat(
        (current_weight, merged_next_weight), dim=0
    ).contiguous()

    reference_current = F.linear(hidden_states.float(), current_weight.float())
    reference_next = F.linear(hidden_states.float(), next_weight.float())
    reference_next += F.linear(hidden_states.float(), delta_weight.float())

    dual_current, dual_next = dual_gate_lora(
        hidden_states,
        current_weight,
        next_weight,
        delta_weight,
    )
    dual_current_output = torch.empty_like(dual_current)
    dual_next_output = torch.empty_like(dual_next)
    concat_logits = _gate(hidden_states, concatenated_weight)
    concat_current, concat_next = concat_logits.split(shape.experts, dim=-1)
    gate_lora(hidden_states, next_weight, delta_weight)
    parallel = _parallel_operation(
        hidden_states,
        current_weight,
        next_weight,
        delta_weight,
    )
    parallel_current, parallel_next = parallel()
    torch.cuda.synchronize()
    errors = {
        "dual": max(
            _max_error(dual_current, reference_current),
            _max_error(dual_next, reference_next),
        ),
        "parallel": max(
            _max_error(parallel_current, reference_current),
            _max_error(parallel_next, reference_next),
        ),
        "concat": max(
            _max_error(concat_current, reference_current),
            _max_error(concat_next, reference_next),
        ),
    }
    timings = {
        "single": _measure(
            lambda: _gate(hidden_states, current_weight),
            warmup=warmup,
            repetitions=repetitions,
        ),
        "dual": _measure(
            lambda: dual_gate_lora(
                hidden_states,
                current_weight,
                next_weight,
                delta_weight,
                current_output=dual_current_output,
                next_output=dual_next_output,
            ),
            warmup=warmup,
            repetitions=repetitions,
        ),
        "parallel": _measure(
            parallel,
            warmup=warmup,
            repetitions=repetitions,
        ),
        "concat": _measure(
            lambda: _gate(hidden_states, concatenated_weight),
            warmup=warmup,
            repetitions=repetitions,
        ),
    }
    timings.update({f"{mode}_error": error for mode, error in errors.items()})
    timings["dual_is_cutedsl"] = float(
        dual_gate_backend(
            hidden_states,
            current_weight,
            next_weight,
            delta_weight,
        )
        == "cutedsl"
    )
    element_size = current_weight.element_size()
    timings["dual_extra_mib"] = delta_weight.numel() * element_size / 2**20
    timings["parallel_extra_mib"] = timings["dual_extra_mib"]
    timings["concat_extra_mib"] = concatenated_weight.numel() * element_size / 2**20
    return timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=SHAPES,
        default=["deepseek"],
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=[8])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    print(
        "shape       M rank mode     backend      us       vs_single  logical_TFLOPS  "
        "max_abs_error  extra_MiB"
    )
    for shape_name in args.shapes:
        shape = SHAPES[shape_name]
        for tokens in args.tokens:
            for rank in args.ranks:
                result = benchmark_case(
                    shape,
                    tokens=tokens,
                    rank=rank,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                )
                single_us = result["single"]
                for mode in ("single", "dual", "parallel", "concat"):
                    error = result.get(f"{mode}_error", 0.0)
                    extra_mib = result.get(f"{mode}_extra_mib", 0.0)
                    projection_count = 1 if mode == "single" else 3
                    if mode == "concat":
                        projection_count = 2
                    flops = (
                        2
                        * tokens
                        * shape.hidden_size
                        * shape.experts
                        * projection_count
                    )
                    logical_tflops = flops / (result[mode] * 1e6)
                    selected_backend = (
                        "cutedsl" if result["dual_is_cutedsl"] else "triton"
                    )
                    backend = selected_backend if mode == "dual" else "-"
                    if mode == "parallel":
                        backend = f"{selected_backend}/2s"
                    print(
                        f"{shape.name:<11} {tokens:>2} {rank:>4} "
                        f"{mode:<8} {backend:<8} {result[mode]:>8.2f} "
                        f"{single_us / result[mode]:>10.3f} "
                        f"{logical_tflops:>14.3f} "
                        f"{error:>14.6f} {extra_mib:>10.2f}"
                    )


if __name__ == "__main__":
    main()
