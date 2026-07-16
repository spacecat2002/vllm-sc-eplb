"""Opt-in MoE load statistics and tensors for routing research."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

logger = init_logger(__name__)
_MAX_TOKENS = 4096
GateProjector = Callable[[torch.Tensor], torch.Tensor]
TraceMode = Literal["expert_distribution", "lora_training"]


@dataclass(frozen=True)
class MoETraceConfig:
    output_dir: Path
    max_steps: int
    capture_next_gate_base_logits: bool
    mode: TraceMode = "lora_training"

    @classmethod
    def from_env(cls) -> MoETraceConfig | None:
        if not envs.VLLM_MOE_TRACE_DIR:
            return None
        if envs.VLLM_MOE_TRACE_MAX_STEPS <= 0:
            raise ValueError("VLLM_MOE_TRACE_MAX_STEPS must be positive")
        mode_value = envs.VLLM_MOE_TRACE_MODE
        if mode_value not in ("expert_distribution", "lora_training"):
            raise ValueError(
                "VLLM_MOE_TRACE_MODE must be 'expert_distribution' or "
                f"'lora_training', got {mode_value!r}"
            )
        mode = cast(TraceMode, mode_value)
        output_dir = Path(envs.VLLM_MOE_TRACE_DIR).expanduser().resolve()
        config_path = output_dir / "trace_config.json"
        payload = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        return cls(
            output_dir=output_dir,
            max_steps=envs.VLLM_MOE_TRACE_MAX_STEPS,
            capture_next_gate_base_logits=bool(
                payload.get("capture_next_gate_base_logits", True)
            ),
            mode=mode,
        )


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class MoETraceCollector:
    def __init__(
        self,
        config: MoETraceConfig,
        layer_names: dict[int, str],
        projectors: dict[int, tuple[int, GateProjector]],
    ) -> None:
        self.config = config
        self.layer_names = layer_names
        self.projectors = projectors
        self.rank = _distributed_rank()
        self.rank_dir = config.output_dir / f"rank_{self.rank:05d}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self._seen_layers: set[int] = set()
        self._indices: list[int] | None = None
        self._num_scheduled_tokens: int | None = None
        self._forward_request_ids: list[str] = []
        self._request_ids: list[str] | None = None
        (self.rank_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "format_version": 3,
                    "rank": self.rank,
                    "mode": config.mode,
                    "layers": layer_names,
                    "next_gate_pairs": [
                        [layer_id, target_layer_id]
                        for layer_id, (target_layer_id, _) in sorted(projectors.items())
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def begin_forward(
        self,
        num_scheduled_tokens: list[int],
        num_computed_tokens: list[int],
        prefill_lengths: list[int],
        request_ids: list[str] | None = None,
    ) -> None:
        del num_computed_tokens, prefill_lengths
        if self._seen_layers:
            self.step += 1
            self._seen_layers.clear()
        if request_ids is None:
            request_ids = [str(index) for index in range(len(num_scheduled_tokens))]
        if len(request_ids) != len(num_scheduled_tokens):
            raise ValueError("MoE trace request metadata lengths do not match")
        indices = []
        selected_requests = []
        start = 0
        for scheduled, request_id in zip(num_scheduled_tokens, request_ids):
            indices.extend(range(start, start + scheduled))
            selected_requests.extend([request_id] * scheduled)
            start += scheduled
        self._num_scheduled_tokens = sum(num_scheduled_tokens)
        self._forward_request_ids = list(request_ids)
        self._indices = indices[:_MAX_TOKENS]
        self._request_ids = selected_requests[:_MAX_TOKENS]

    @torch.no_grad()
    def capture(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        if layer_id in self._seen_layers:
            self.step += 1
            self._seen_layers.clear()
        if self.step >= self.config.max_steps:
            return
        self._seen_layers.add(layer_id)
        available = min(
            hidden_states.shape[0],
            router_logits.shape[0],
            topk_weights.shape[0],
            topk_ids.shape[0],
        )
        num_scheduled_tokens = min(self._num_scheduled_tokens or available, available)
        path = self.rank_dir / f"step_{self.step:06d}_layer_{layer_id:04d}.pt"
        if self.config.mode == "expert_distribution":
            num_experts = router_logits.shape[-1]
            logical_ids = topk_ids[:num_scheduled_tokens].to(torch.int64).reshape(-1)
            logical_ids = logical_ids[(logical_ids >= 0) & (logical_ids < num_experts)]
            torch.save(
                {
                    "format_version": 3,
                    "mode": self.config.mode,
                    "rank": self.rank,
                    "step": self.step,
                    "layer_id": layer_id,
                    "layer_name": self.layer_names.get(layer_id, ""),
                    "num_scheduled_tokens": num_scheduled_tokens,
                    "request_ids": self._forward_request_ids,
                    "expert_counts": torch.bincount(
                        logical_ids, minlength=num_experts
                    ).to(device="cpu", dtype=torch.int64),
                },
                path,
            )
            return

        indices = self._indices or list(range(min(available, _MAX_TOKENS)))
        indices = [index for index in indices if index < available]
        index = torch.tensor(indices, dtype=torch.long, device=hidden_states.device)
        selected_hidden = hidden_states.index_select(0, index)
        record: dict[str, Any] = {
            "format_version": 3,
            "mode": self.config.mode,
            "rank": self.rank,
            "step": self.step,
            "layer_id": layer_id,
            "layer_name": self.layer_names.get(layer_id, ""),
            "num_scheduled_tokens": num_scheduled_tokens,
            "request_ids": (self._request_ids or [])[: len(indices)],
            "activations": selected_hidden.to(device="cpu", dtype=torch.float16),
            "router_logits": router_logits.index_select(0, index).to(
                device="cpu", dtype=torch.float16
            ),
            "topk_ids": topk_ids.index_select(0, index).to(
                device="cpu", dtype=torch.int32
            ),
            "topk_weights": topk_weights.index_select(0, index).to(
                device="cpu", dtype=torch.float32
            ),
        }
        if self.config.capture_next_gate_base_logits and layer_id in self.projectors:
            target_layer_id, projector = self.projectors[layer_id]
            record["next_gate_layer_id"] = target_layer_id
            record["next_gate_base_logits"] = projector(selected_hidden).to(
                device="cpu", dtype=torch.float16
            )
        torch.save(record, path)


def _runners(
    model: torch.nn.Module | None, static_forward_context: dict[str, Any]
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


def _projector(runner: MoERunner) -> GateProjector | None:
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


def maybe_attach_moe_trace(
    *,
    enforce_eager: bool,
    static_forward_context: dict[str, Any],
    model: torch.nn.Module | None = None,
) -> MoETraceCollector | None:
    config = MoETraceConfig.from_env()
    if config is None:
        return None
    if not enforce_eager:
        raise ValueError("VLLM_MOE_TRACE_DIR requires --enforce-eager")
    runners = _runners(model, static_forward_context)
    if not runners:
        logger.warning("MoE trace enabled, but no compatible MoERunner was found")
        return None
    projectors = {}
    if config.mode == "lora_training":
        for current, following in zip(runners, runners[1:]):
            projector = _projector(following)
            if projector is not None:
                projectors[current.layer_id] = following.layer_id, projector
    collector = MoETraceCollector(
        config,
        {runner.layer_id: runner.layer_name for runner in runners},
        projectors,
    )
    for runner in runners:
        layer_id = runner.layer_id

        def trace(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            _layer_id: int = layer_id,
        ) -> None:
            collector.capture(
                _layer_id,
                hidden_states,
                router_logits,
                topk_weights,
                topk_ids,
            )

        runner.router.set_trace_fn(trace)
    logger.warning(
        "MoE trace mode %s enabled for %d layers on rank %d: %s",
        config.mode,
        len(runners),
        collector.rank,
        collector.rank_dir,
    )
    return collector
