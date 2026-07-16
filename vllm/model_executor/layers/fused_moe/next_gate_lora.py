# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional serving-side prediction of the next MoE gate.

This module is intentionally independent from the research training tools in
``moe_gate_lora``. It is retained for future work that may consume predicted
routes in EPLB or expert prefetching. Current predictions are side-channel
outputs and never alter the real MoE route.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

NextGatePredictor = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
logger = init_logger(__name__)


@dataclass
class NextGatePredictionBuildResult:
    predictors: dict[int, tuple[int, NextGatePredictor]]
    missing_gate_layers: list[tuple[int, int]]
    lora_dir: str | None
    fused_layers: set[int]


def _moe_runners(
    model: torch.nn.Module | None,
    static_forward_context: dict[str, Any],
) -> list[MoERunner]:
    candidates = (
        model.modules() if model is not None else static_forward_context.values()
    )
    by_layer = {
        module.layer_id: module
        for module in candidates
        if isinstance(module, MoERunner) and isinstance(module.router, BaseRouter)
    }
    return sorted(by_layer.values(), key=lambda runner: runner.layer_id)


def _gate_projector(
    runner: MoERunner,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    gate = runner.gate
    if gate is None:
        return None
    if runner._fse_fuse_gate:
        runner._maybe_fuse_gate_weights()
        assert runner._combined_gate_weight is not None
        input_size = runner._combined_gate_weight.shape[1]

        def project_fused(hidden_states: torch.Tensor) -> torch.Tensor:
            if hidden_states.shape[-1] != input_size:
                raise ValueError(
                    f"Layer {runner.layer_id} gate expects hidden size "
                    f"{input_size}, got {hidden_states.shape[-1]}"
                )
            return torch.nn.functional.linear(
                hidden_states, runner._combined_gate_weight
            )

        return project_fused

    gate_weight = getattr(gate, "weight", None)
    input_size = gate_weight.shape[1] if gate_weight is not None else None

    def project(hidden_states: torch.Tensor) -> torch.Tensor:
        if input_size is not None and hidden_states.shape[-1] != input_size:
            raise ValueError(
                f"Layer {runner.layer_id} gate expects hidden size "
                f"{input_size}, got {hidden_states.shape[-1]}"
            )
        logits, _ = gate(hidden_states)
        return logits

    return project


def _gate_input_size(runner: MoERunner) -> int:
    if runner._fse_fuse_gate:
        runner._maybe_fuse_gate_weights()
        assert runner._combined_gate_weight is not None
        return runner._combined_gate_weight.shape[1]
    gate_weight = getattr(runner.gate, "weight", None)
    if gate_weight is not None:
        return gate_weight.shape[1]
    return runner.moe_config.hidden_dim


def _gate_weight(runner: MoERunner) -> torch.Tensor | None:
    if runner.gate is None:
        return None
    if runner._fse_fuse_gate:
        if (
            getattr(runner.gate, "bias", None) is not None
            or getattr(runner.shared_expert_gate, "bias", None) is not None
        ):
            return None
        runner._maybe_fuse_gate_weights()
        return runner._combined_gate_weight
    if getattr(runner.gate, "bias", None) is not None:
        return None
    return getattr(runner.gate, "weight", None)


def _gate_output_dtype(runner: MoERunner, weight: torch.Tensor) -> torch.dtype:
    if runner._fse_fuse_gate:
        return weight.dtype
    return getattr(runner.gate, "out_dtype", None) or weight.dtype


def merge_lora_weight(
    base_weight: torch.Tensor,
    lora: tuple[torch.Tensor, torch.Tensor, float] | None,
) -> torch.Tensor:
    """Materialize ``base_weight + scale * lora_B @ lora_A`` once."""
    if lora is None:
        return base_weight.detach()
    lora_a, lora_b, scale = lora
    merged = (
        base_weight.detach().float()
        + (lora_b.detach().float() @ lora_a.detach().float()) * scale
    )
    return merged.to(base_weight.dtype)


def materialize_lora_delta(
    base_weight: torch.Tensor,
    lora: tuple[torch.Tensor, torch.Tensor, float] | None,
) -> torch.Tensor:
    """Materialize a scaled LoRA delta without modifying the base weight."""
    if lora is None:
        return torch.zeros_like(base_weight)
    lora_a, lora_b, scale = lora
    routed_delta = (lora_b.detach().float() @ lora_a.detach().float()) * scale
    if routed_delta.shape[0] > base_weight.shape[0]:
        raise ValueError("LoRA delta has more rows than the next gate")
    delta = torch.zeros_like(base_weight)
    delta[: routed_delta.shape[0]].copy_(routed_delta.to(base_weight.dtype))
    return delta.contiguous()


def combine_gate_weights(
    current_weight: torch.Tensor,
    next_weight: torch.Tensor,
    lora: tuple[torch.Tensor, torch.Tensor, float] | None,
) -> tuple[torch.Tensor, int]:
    predicted_weight = merge_lora_weight(next_weight, lora)
    combined = torch.cat(
        [current_weight.detach(), predicted_weight], dim=0
    ).contiguous()
    return combined, current_weight.shape[0]


def _load_lora(
    directory: str | None,
    source_layer_id: int,
    next_layer_id: int,
    hidden_size: int,
    num_experts: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    if directory is None:
        return None
    root = Path(directory).expanduser()
    candidates = (
        root / f"layer_{source_layer_id:04d}_to_{next_layer_id:04d}.pt",
        root / f"layer_{next_layer_id:04d}.pt",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return None
    payload = torch.load(path, map_location=device, weights_only=True)
    lora_a = payload["lora_A"].to(device=device, dtype=dtype)
    lora_b = payload["lora_B"].to(device=device, dtype=dtype)
    rank = int(payload.get("rank", lora_a.shape[0]))
    if lora_a.shape != (rank, hidden_size):
        raise ValueError(f"Invalid lora_A shape in {path}: {lora_a.shape}")
    if lora_b.shape != (num_experts, rank):
        raise ValueError(f"Invalid lora_B shape in {path}: {lora_b.shape}")
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive in {path}")
    return lora_a, lora_b, float(payload.get("alpha", rank)) / rank


def _predictor(
    next_runner: MoERunner,
    projector: Callable[[torch.Tensor], torch.Tensor],
    lora: tuple[torch.Tensor, torch.Tensor, float] | None,
) -> NextGatePredictor:
    @torch.no_grad()
    def predict(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_logits = projector(hidden_states)
        logits = base_logits
        if lora is not None:
            lora_a, lora_b, scale = lora
            delta = (hidden_states.to(lora_a.dtype) @ lora_a.T) @ lora_b.T
            logits = base_logits.clone()
            logits[:, : delta.shape[-1]] += delta.to(base_logits.dtype) * scale
        _, topk_ids = next_runner.router._compute_routing(
            hidden_states,
            logits,
            next_runner._quant_method.topk_indices_dtype,
            input_ids=None,
        )
        return topk_ids, base_logits

    return predict


def build_next_gate_lora_predictors(
    *,
    static_forward_context: dict[str, Any],
    model: torch.nn.Module | None,
    lora_dir: str | None,
) -> NextGatePredictionBuildResult:
    runners = _moe_runners(model, static_forward_context)
    predictors = {}
    missing = []
    fused_layers = set()
    for current, following in zip(runners, runners[1:]):
        projector = _gate_projector(following)
        if projector is None:
            missing.append((current.layer_id, following.layer_id))
            continue
        current_weight = _gate_weight(current)
        next_weight = _gate_weight(following)
        reference = (
            next_weight if next_weight is not None else next(following.parameters())
        )
        lora = _load_lora(
            lora_dir,
            current.layer_id,
            following.layer_id,
            _gate_input_size(following),
            following.moe_config.num_experts,
            reference.device,
            reference.dtype,
        )
        can_fuse = (
            current_weight is not None
            and next_weight is not None
            and current_weight.shape[1] == next_weight.shape[1]
            and current_weight.dtype == next_weight.dtype
            and current_weight.device == next_weight.device
            and _gate_output_dtype(current, current_weight)
            == _gate_output_dtype(following, next_weight)
        )
        if can_fuse:
            assert current_weight is not None and next_weight is not None
            delta_weight = materialize_lora_delta(next_weight, lora)

            def route(
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                _runner: MoERunner = following,
            ) -> torch.Tensor:
                _, topk_ids = _runner.router._compute_routing(
                    hidden_states,
                    router_logits,
                    _runner._quant_method.topk_indices_dtype,
                    input_ids=None,
                )
                return topk_ids

            fused_output_dtype = _gate_output_dtype(current, current_weight)
            output_dtype = (
                fused_output_dtype
                if fused_output_dtype != current_weight.dtype
                else None
            )
            current.set_fused_next_gate_predictor(
                next_layer_id=following.layer_id,
                current_weight=current_weight,
                next_weight=next_weight,
                next_delta_weight=delta_weight,
                route=route,
                output_dtype=output_dtype,
            )
            fused_layers.add(current.layer_id)
            continue
        predictors[current.layer_id] = (
            following.layer_id,
            _predictor(following, projector, lora),
        )
    return NextGatePredictionBuildResult(predictors, missing, lora_dir, fused_layers)


def maybe_attach_sc_eplb_next_gate_lora(
    *,
    static_forward_context: dict[str, Any],
    model: torch.nn.Module | None,
    enabled: bool,
    lora_dir: str | None,
) -> NextGatePredictionBuildResult | None:
    if not enabled:
        return None
    result = build_next_gate_lora_predictors(
        static_forward_context=static_forward_context,
        model=model,
        lora_dir=lora_dir,
    )
    runners = _moe_runners(model, static_forward_context)
    last_layer_id = runners[-1].layer_id if runners else None
    for runner in runners:
        if runner.layer_id == last_layer_id:
            runner.clear_next_gate_predictor()
            continue
        if runner.layer_id in result.fused_layers:
            continue
        prediction = result.predictors.get(runner.layer_id)
        if prediction is None:
            runner.clear_next_gate_predictor()
        else:
            runner.set_next_gate_predictor(*prediction)
    logger.info(
        "Attached next-gate prediction: %d fused projections, %d fallback "
        "predictors, %d missing gates",
        len(result.fused_layers),
        len(result.predictors),
        len(result.missing_gate_layers),
    )
    return result
