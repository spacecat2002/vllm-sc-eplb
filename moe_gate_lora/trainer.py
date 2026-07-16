from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import torch

from moe_gate_lora.plot import write_overlap_plot
from moe_gate_lora.stats import (
    RunningMoments,
    load_records,
    paired_records,
    topk_overlap,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)


class LayerAdapter:
    def __init__(
        self,
        *,
        source_layer_id: int,
        next_layer_id: int,
        hidden_size: int,
        num_experts: int,
        rank_dim: int,
        alpha: float,
        device: torch.device,
        seed: int,
        lr: float | None,
        weight_decay: float,
        checkpoint: Path | None = None,
    ) -> None:
        self.source_layer_id = source_layer_id
        self.next_layer_id = next_layer_id
        self.rank = min(rank_dim, hidden_size, num_experts)
        self.alpha = alpha
        self.scale = alpha / self.rank
        self.device = device
        self.steps = 0
        self.examples = 0
        self.loss = RunningMoments()

        self.lora_a = torch.nn.Parameter(
            torch.empty((self.rank, hidden_size), device=device)
        )
        self.lora_b = torch.nn.Parameter(
            torch.zeros((num_experts, self.rank), device=device)
        )
        if checkpoint is None:
            with torch.random.fork_rng(
                devices=[device] if device.type == "cuda" else []
            ):
                torch.manual_seed(seed + next_layer_id)
                torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        else:
            self._load(checkpoint, hidden_size, num_experts)

        self.optimizer = (
            torch.optim.AdamW(
                [self.lora_a, self.lora_b],
                lr=lr,
                weight_decay=weight_decay,
            )
            if lr is not None
            else None
        )

    def _load(self, checkpoint: Path, hidden_size: int, num_experts: int) -> None:
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        lora_a = payload["lora_A"].to(self.device, dtype=torch.float32)
        lora_b = payload["lora_B"].to(self.device, dtype=torch.float32)
        expected_a = (self.rank, hidden_size)
        expected_b = (num_experts, self.rank)
        if lora_a.shape != expected_a or lora_b.shape != expected_b:
            raise ValueError(
                f"LoRA shape mismatch for layer {self.next_layer_id}: "
                f"got {tuple(lora_a.shape)} and {tuple(lora_b.shape)}, "
                f"expected {expected_a} and {expected_b}"
            )
        self.lora_a.data.copy_(lora_a)
        self.lora_b.data.copy_(lora_b)

    def logits(
        self, activations: torch.Tensor, base_logits: torch.Tensor
    ) -> torch.Tensor:
        x = activations.to(self.device, dtype=torch.float32)
        base = base_logits.to(self.device, dtype=torch.float32)
        return base + ((x @ self.lora_a.T) @ self.lora_b.T) * self.scale

    def train_batch(
        self,
        activations: torch.Tensor,
        base_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> None:
        if self.optimizer is None:
            raise RuntimeError("Cannot train an evaluation-only adapter")
        target = target_logits.to(self.device, dtype=torch.float32)
        logits = self.logits(activations, base_logits)
        per_example_loss = torch.mean((logits - target) ** 2, dim=-1)
        self.optimizer.zero_grad(set_to_none=True)
        per_example_loss.mean().backward()
        self.optimizer.step()
        self.loss.update(per_example_loss)
        self.steps += 1
        self.examples += activations.shape[0]

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / (
            f"layer_{self.source_layer_id:04d}_to_{self.next_layer_id:04d}.pt"
        )
        torch.save(
            {
                "source_layer_id": self.source_layer_id,
                "next_layer_id": self.next_layer_id,
                "rank": self.rank,
                "alpha": self.alpha,
                "hidden_size": self.lora_a.shape[1],
                "num_experts": self.lora_b.shape[0],
                "lora_A": self.lora_a.detach().cpu(),
                "lora_B": self.lora_b.detach().cpu(),
            },
            path,
        )
        return path


class StreamingProcessor:
    """Consume, train/evaluate, aggregate, and delete one trace batch."""

    def __init__(
        self,
        *,
        mode: Literal["train", "eval"],
        output_dir: Path,
        lora_dir: Path,
        rank_dim: int,
        alpha: float,
        lr: float,
        weight_decay: float,
        seed: int,
        device: str | None,
    ) -> None:
        self.mode = mode
        self.output_dir = output_dir
        self.lora_dir = lora_dir
        self.rank_dim = rank_dim
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.adapters: dict[int, LayerAdapter] = {}
        self.router: FusedTopKRouter | None = None
        self.baseline_overlap: dict[tuple[int, int], RunningMoments] = {}
        self.lora_overlap: dict[tuple[int, int], RunningMoments] = {}
        self.batch_index = 0
        self.current_epoch: int | None = None
        self.epochs_completed = 0

    def _adapter(
        self,
        *,
        source_layer_id: int,
        next_layer_id: int,
        hidden_size: int,
        num_experts: int,
    ) -> LayerAdapter:
        adapter = self.adapters.get(next_layer_id)
        if adapter is not None:
            return adapter
        checkpoint = None
        if self.mode == "eval":
            checkpoint = self.lora_dir / (
                f"layer_{source_layer_id:04d}_to_{next_layer_id:04d}.pt"
            )
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
        adapter = LayerAdapter(
            source_layer_id=source_layer_id,
            next_layer_id=next_layer_id,
            hidden_size=hidden_size,
            num_experts=num_experts,
            rank_dim=self.rank_dim,
            alpha=self.alpha,
            device=self.device,
            seed=self.seed,
            lr=self.lr if self.mode == "train" else None,
            weight_decay=self.weight_decay,
            checkpoint=checkpoint,
        )
        self.adapters[next_layer_id] = adapter
        return adapter

    def _route_topk(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        if self.router is None:
            self.router = FusedTopKRouter(
                top_k=top_k,
                global_num_experts=router_logits.shape[1],
                scoring_func="softmax",
                renormalize=True,
            )
        _, topk_ids = self.router._compute_routing(
            hidden_states,
            router_logits,
            torch.int32,
            input_ids=None,
        )
        return topk_ids.to(device="cpu", dtype=torch.long)

    def process(self, record_paths: list[Path], *, epoch: int = 0) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if epoch != self.epochs_completed:
            raise ValueError(
                f"Expected epoch {self.epochs_completed}, got epoch {epoch}"
            )
        self.current_epoch = epoch
        records = load_records(record_paths)
        batches: dict[int, dict[str, Any]] = {}
        for layer_id, next_layer_id, record, next_record in paired_records(records):
            entry = batches.setdefault(
                next_layer_id,
                {
                    "source_layer_id": layer_id,
                    "activations": [],
                    "base_logits": [],
                    "target_logits": [],
                    "labels": [],
                },
            )
            entry["activations"].append(record["activations"])
            entry["base_logits"].append(record["next_gate_base_logits"])
            entry["target_logits"].append(next_record["router_logits"])
            entry["labels"].append(next_record["topk_ids"])

        for next_layer_id, entry in sorted(batches.items()):
            source_layer_id = int(entry["source_layer_id"])
            activations = torch.cat(entry["activations"], dim=0)
            base_logits = torch.cat(entry["base_logits"], dim=0)
            target_logits = torch.cat(entry["target_logits"], dim=0)
            labels = torch.cat(entry["labels"], dim=0).to(torch.long)
            adapter = self._adapter(
                source_layer_id=source_layer_id,
                next_layer_id=next_layer_id,
                hidden_size=activations.shape[1],
                num_experts=base_logits.shape[1],
            )
            pair = (source_layer_id, next_layer_id)
            with torch.no_grad():
                router_hidden = activations.to(self.device, dtype=torch.float32)
                baseline_logits = base_logits.to(self.device, dtype=torch.float32)
                baseline_ids = self._route_topk(
                    router_hidden,
                    baseline_logits,
                    labels.shape[1],
                )
                lora_ids = self._route_topk(
                    router_hidden,
                    adapter.logits(activations, base_logits),
                    labels.shape[1],
                )
            self.baseline_overlap.setdefault(pair, RunningMoments()).update(
                topk_overlap(baseline_ids, labels)
            )
            self.lora_overlap.setdefault(pair, RunningMoments()).update(
                topk_overlap(lora_ids, labels)
            )
            if self.mode == "train":
                adapter.train_batch(activations, base_logits, target_logits)

        if not batches:
            raise RuntimeError("Trace batch contained no adjacent MoE layer pairs")
        self.batch_index += 1
        self._write_outputs()
        for path in record_paths:
            path.unlink()

    def finish_epoch(self, epoch: int) -> None:
        if self.current_epoch != epoch:
            raise ValueError(f"Cannot finish epoch {epoch} before processing it")
        if epoch != self.epochs_completed:
            raise ValueError("epochs must be finished exactly once and in order")
        self.epochs_completed += 1
        self._write_outputs()

    def _write_outputs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints = []
        if self.mode == "train":
            checkpoints = [
                str(adapter.save(self.lora_dir))
                for _, adapter in sorted(self.adapters.items())
            ]
        rows = []
        for pair in sorted(self.baseline_overlap):
            baseline = self.baseline_overlap[pair].as_dict()
            lora = self.lora_overlap[pair].as_dict()
            rows.append(
                {
                    "layer_i": pair[0],
                    "layer_j": pair[1],
                    "num_tokens": lora["count"],
                    "baseline_overlap_mean": baseline["mean"],
                    "baseline_overlap_std": baseline["std"],
                    "lora_overlap_mean": lora["mean"],
                    "lora_overlap_std": lora["std"],
                }
            )
        payload = {
            "mode": self.mode,
            "batches": self.batch_index,
            "current_epoch": self.current_epoch,
            "epochs_completed": self.epochs_completed,
            "rows": rows,
            "checkpoints": checkpoints,
            "training": [
                {
                    "source_layer_id": adapter.source_layer_id,
                    "next_layer_id": adapter.next_layer_id,
                    "steps": adapter.steps,
                    "examples": adapter.examples,
                    "loss": adapter.loss.as_dict(),
                }
                for _, adapter in sorted(self.adapters.items())
            ],
        }
        (self.output_dir / "metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        write_overlap_plot(rows, self.output_dir / "overlap.png")
