# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Train a small aggregate next-layer expert-load predictor.

The default synthetic experiment varies the number of tokens in every sample
to mimic chunked-prefill forwards. A real experiment can consume vLLM
``lora_training`` traces containing adjacent-layer ``topk_ids`` records.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class RouteSample:
    current_ids: torch.Tensor
    current_weights: torch.Tensor
    target_ids: torch.Tensor


@dataclass(frozen=True)
class LoadMetrics:
    kl: float
    hot_recall: float
    token_coverage: float


def _normalized_counts(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    ids = ids.to(dtype=torch.long).reshape(-1)
    ids = ids[(ids >= 0) & (ids < num_experts)]
    if ids.numel() == 0:
        raise ValueError("A route sample contains no valid expert IDs")
    counts = torch.bincount(ids, minlength=num_experts).float()
    return counts / counts.sum()


def _current_distribution(sample: RouteSample, num_experts: int) -> torch.Tensor:
    ids = sample.current_ids.to(dtype=torch.long).reshape(-1)
    weights = sample.current_weights.float().reshape(-1)
    valid = (ids >= 0) & (ids < num_experts)
    distribution = torch.zeros(num_experts, dtype=torch.float32)
    distribution.scatter_add_(0, ids[valid], weights[valid])
    total = distribution.sum()
    if total <= 0:
        raise ValueError("A route sample contains no positive routing weight")
    return distribution / total


def fit_transition_matrix(
    samples: list[RouteSample],
    num_experts: int,
    smoothing: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit P(next expert | current expert) and the next-layer prior."""
    if not samples:
        raise ValueError("At least one training sample is required")
    transitions = torch.zeros((num_experts, num_experts), dtype=torch.float64)
    prior_counts = torch.zeros(num_experts, dtype=torch.float64)

    for sample in samples:
        current_ids = sample.current_ids.to(dtype=torch.long)
        current_weights = sample.current_weights.to(dtype=torch.float64)
        target_ids = sample.target_ids.to(dtype=torch.long)
        num_tokens = min(current_ids.shape[0], target_ids.shape[0])
        current_ids = current_ids[:num_tokens]
        current_weights = current_weights[:num_tokens]
        target_ids = target_ids[:num_tokens]

        target_flat = target_ids.reshape(-1)
        valid_target = (target_flat >= 0) & (target_flat < num_experts)
        prior_counts.scatter_add_(
            0,
            target_flat[valid_target],
            torch.ones_like(target_flat[valid_target], dtype=torch.float64),
        )

        source = current_ids.unsqueeze(-1).expand(-1, -1, target_ids.shape[1])
        target = target_ids.unsqueeze(1).expand(-1, current_ids.shape[1], -1)
        values = current_weights.unsqueeze(-1).expand_as(source).to(torch.float64)
        values = values / target_ids.shape[1]
        valid = (
            (source >= 0)
            & (source < num_experts)
            & (target >= 0)
            & (target < num_experts)
        )
        transitions.index_put_(
            (source[valid], target[valid]), values[valid], accumulate=True
        )

    prior = prior_counts / prior_counts.sum()
    transitions += smoothing * prior.unsqueeze(0)
    transitions /= transitions.sum(dim=1, keepdim=True)
    return transitions.float(), prior.float()


def transition_prediction(
    sample: RouteSample,
    transition: torch.Tensor,
) -> torch.Tensor:
    current = _current_distribution(sample, transition.shape[0])
    predicted = current @ transition.cpu()
    return predicted / predicted.sum()


class RouteSetLoadPredictor(nn.Module):
    """Predict aggregate load without predicting every token's next route."""

    def __init__(
        self,
        num_experts: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be at least two")
        self.num_experts = num_experts
        self.expert_embedding = nn.Embedding(num_experts, embedding_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(embedding_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        nn.init.zeros_(self.output_mlp[-1].weight)
        nn.init.zeros_(self.output_mlp[-1].bias)

    def forward(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        base_distribution: torch.Tensor,
    ) -> torch.Tensor:
        topk_ids = topk_ids.to(dtype=torch.long)
        topk_weights = topk_weights.float()
        valid = (topk_ids >= 0) & (topk_ids < self.num_experts)
        safe_ids = topk_ids.clamp(0, self.num_experts - 1)
        weights = topk_weights * valid
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        route_features = (self.expert_embedding(safe_ids) * weights.unsqueeze(-1)).sum(
            dim=1
        )
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1)
        if weights.shape[1] > 1:
            top_two = weights.topk(k=2, dim=1).values
            margin = top_two[:, 0] - top_two[:, 1]
        else:
            margin = weights[:, 0]
        token_input = torch.cat(
            [route_features, entropy.unsqueeze(1), margin.unsqueeze(1)], dim=1
        )
        token_features = self.token_mlp(token_input)
        pooled = torch.cat(
            [
                token_features.mean(dim=0),
                token_features.square().mean(dim=0),
                token_features.new_tensor([math.log1p(topk_ids.shape[0]) / 10.0]),
            ]
        )
        residual = self.output_mlp(pooled)
        base_logits = base_distribution.clamp_min(1e-8).log()
        return torch.softmax(base_logits + 2.0 * torch.tanh(residual), dim=0)


def _ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    replica_slots: int,
) -> torch.Tensor:
    if replica_slots <= 0:
        raise ValueError("replica_slots must be positive")
    slots = min(replica_slots, prediction.numel() - 1)
    hot = target.topk(slots).indices
    cold_mask = torch.ones_like(target, dtype=torch.bool)
    cold_mask[hot] = False
    hot_scores = prediction[hot].clamp_min(1e-8).log().unsqueeze(1)
    cold_scores = prediction[cold_mask].clamp_min(1e-8).log().unsqueeze(0)
    return torch.relu(0.1 - hot_scores + cold_scores).mean()


def train_predictor(
    predictor: RouteSetLoadPredictor,
    samples: list[RouteSample],
    transition: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    replica_slots: int,
    seed: int,
    device: torch.device,
) -> list[float]:
    if not samples:
        raise ValueError("At least one training sample is required")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    predictor.to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        epoch_loss = 0.0
        order = torch.randperm(len(samples), generator=generator).tolist()
        for index in order:
            sample = samples[index]
            base = transition_prediction(sample, transition).to(device)
            target = _normalized_counts(sample.target_ids, predictor.num_experts).to(
                device
            )
            prediction = predictor(
                sample.current_ids.to(device),
                sample.current_weights.to(device),
                base,
            )
            distribution_loss = -(target * prediction.clamp_min(1e-8).log()).sum()
            loss = distribution_loss + 0.1 * _ranking_loss(
                prediction, target, replica_slots
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
        losses.append(epoch_loss / len(samples))
    return losses


@torch.no_grad()
def evaluate(
    samples: list[RouteSample],
    predictions: list[torch.Tensor],
    *,
    num_experts: int,
    replica_slots: int,
) -> LoadMetrics:
    if len(samples) != len(predictions) or not samples:
        raise ValueError("Samples and predictions must be non-empty and aligned")
    if replica_slots <= 0:
        raise ValueError("replica_slots must be positive")
    slots = min(replica_slots, num_experts)
    kl_values = []
    recall_values = []
    coverage_values = []
    for sample, prediction in zip(samples, predictions):
        target = _normalized_counts(sample.target_ids, num_experts)
        prediction = prediction.detach().cpu().clamp_min(1e-8)
        nonzero = target > 0
        kl_values.append(
            float(
                (
                    target[nonzero]
                    * (target[nonzero].log() - prediction[nonzero].log())
                ).sum()
            )
        )
        predicted_hot = prediction.topk(slots).indices
        actual_hot = target.topk(slots).indices
        overlap = torch.isin(predicted_hot, actual_hot).float().mean()
        recall_values.append(float(overlap))
        coverage_values.append(float(target[predicted_hot].sum()))
    return LoadMetrics(
        kl=sum(kl_values) / len(kl_values),
        hot_recall=sum(recall_values) / len(recall_values),
        token_coverage=sum(coverage_values) / len(coverage_values),
    )


def synthetic_chunked_samples(
    *,
    num_samples: int,
    num_experts: int,
    seed: int,
) -> list[RouteSample]:
    """Create variable-size route sets with a learnable Top-2 interaction."""
    if num_experts < 8:
        raise ValueError("The synthetic example requires at least eight experts")
    generator = torch.Generator().manual_seed(seed)
    chunk_sizes = torch.tensor([8, 16, 32, 64, 128, 256])
    first_expert_probabilities = torch.linspace(2.0, 0.25, num_experts)
    samples = []
    for _ in range(num_samples):
        num_tokens = int(
            chunk_sizes[torch.randint(len(chunk_sizes), (), generator=generator)].item()
        )
        first = torch.multinomial(
            first_expert_probabilities,
            num_tokens,
            replacement=True,
            generator=generator,
        )
        mixture = torch.rand((), generator=generator)
        route_kind = torch.rand(num_tokens, generator=generator) < mixture
        offset = torch.where(route_kind, 1, 3)
        second = (first + offset) % num_experts
        current_ids = torch.stack([first, second], dim=1)
        current_weights = 0.25 + torch.rand((num_tokens, 2), generator=generator)
        current_weights /= current_weights.sum(dim=1, keepdim=True)

        target_first = (first + 3 * offset) % num_experts
        target_second = (target_first + num_experts // 2 - 1) % num_experts
        noise = torch.rand(num_tokens, generator=generator) < 0.05
        target_first[noise] = torch.randint(
            num_experts, (int(noise.sum().item()),), generator=generator
        )
        target_ids = torch.stack([target_first, target_second], dim=1)
        samples.append(RouteSample(current_ids, current_weights, target_ids))
    return samples


def load_trace_samples(
    trace_dir: Path,
    source_layer: int | None,
) -> tuple[list[RouteSample], int, tuple[int, int]]:
    """Load one adjacent layer pair from full ``lora_training`` traces."""
    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    for path in sorted(trace_dir.rglob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        required = {"rank", "step", "layer_id", "topk_ids"}
        if not required.issubset(record):
            continue
        layer_id = int(record["layer_id"])
        router_logits = record.get("router_logits")
        records[(int(record["rank"]), int(record["step"]), layer_id)] = {
            "topk_ids": record["topk_ids"],
            "topk_weights": record.get("topk_weights"),
            "next_gate_layer_id": record.get("next_gate_layer_id"),
            "num_experts": (
                int(router_logits.shape[-1]) if router_logits is not None else None
            ),
        }

    pairs: dict[tuple[int, int], list[RouteSample]] = {}
    pair_expert_counts: dict[tuple[int, int], set[int]] = {}
    ordered_records = sorted(
        records.items(), key=lambda item: (item[0][1], item[0][0], item[0][2])
    )
    for (rank, step, layer_id), record in ordered_records:
        next_layer_id = record["next_gate_layer_id"]
        if next_layer_id is None:
            continue
        next_layer_id = int(next_layer_id)
        if source_layer is not None and layer_id != source_layer:
            continue
        next_record = records.get((rank, step, next_layer_id))
        if next_record is None:
            continue
        current_ids = record["topk_ids"].to(dtype=torch.long)
        target_ids = next_record["topk_ids"].to(dtype=torch.long)
        num_tokens = min(current_ids.shape[0], target_ids.shape[0])
        current_ids = current_ids[:num_tokens]
        target_ids = target_ids[:num_tokens]
        weights = record["topk_weights"]
        if weights is None:
            weights = torch.ones_like(current_ids, dtype=torch.float32)
        weights = weights[:num_tokens].float()
        pair = (layer_id, next_layer_id)
        pairs.setdefault(pair, []).append(RouteSample(current_ids, weights, target_ids))
        if next_record["num_experts"] is not None:
            pair_expert_counts.setdefault(pair, set()).add(
                int(next_record["num_experts"])
            )

    if not pairs:
        raise ValueError(
            "No adjacent full-trace pair was found; use lora_training trace mode"
        )
    if source_layer is None:
        pair = sorted(pairs)[0]
    else:
        matching = [pair for pair in pairs if pair[0] == source_layer]
        if not matching:
            raise ValueError(f"No trace pair starts at layer {source_layer}")
        pair = matching[0]
    samples = pairs[pair]
    expert_counts = pair_expert_counts.get(pair, set())
    if len(expert_counts) > 1:
        raise ValueError(
            f"Trace pair {pair} contains inconsistent expert counts: {expert_counts}"
        )
    if expert_counts:
        num_experts = next(iter(expert_counts))
    else:
        max_id = max(
            int(ids.max())
            for sample in samples
            for ids in (sample.current_ids, sample.target_ids)
        )
        num_experts = max_id + 1
    return samples, num_experts, pair


def _predict_all(
    predictor: RouteSetLoadPredictor,
    samples: list[RouteSample],
    transition: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    predictor.eval()
    with torch.no_grad():
        return [
            predictor(
                sample.current_ids.to(device),
                sample.current_weights.to(device),
                transition_prediction(sample, transition).to(device),
            ).cpu()
            for sample in samples
        ]


def _format_metrics(name: str, metrics: LoadMetrics) -> str:
    return (
        f"{name:<12} KL={metrics.kl:7.4f}  "
        f"hot-recall={metrics.hot_recall:6.2%}  "
        f"token-coverage={metrics.token_coverage:6.2%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--source-layer", type=int)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--replica-slots", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.trace_dir is None:
        samples = synthetic_chunked_samples(
            num_samples=args.samples,
            num_experts=args.num_experts,
            seed=args.seed,
        )
        num_experts = args.num_experts
        label = "synthetic chunked-prefill"
    else:
        samples, num_experts, pair = load_trace_samples(
            args.trace_dir.expanduser().resolve(), args.source_layer
        )
        label = f"trace layer {pair[0]} -> {pair[1]}"

    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between zero and one")
    if args.replica_slots <= 0:
        raise ValueError("--replica-slots must be positive")
    split = int(len(samples) * args.train_fraction)
    if split == 0 or split == len(samples):
        raise ValueError("The experiment needs at least one train and test sample")
    train_samples = samples[:split]
    test_samples = samples[split:]
    transition, prior = fit_transition_matrix(train_samples, num_experts)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    predictor = RouteSetLoadPredictor(num_experts)
    losses = train_predictor(
        predictor,
        train_samples,
        transition,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        replica_slots=args.replica_slots,
        seed=args.seed,
        device=device,
    )
    prior_predictions = [prior for _ in test_samples]
    transition_predictions = [
        transition_prediction(sample, transition) for sample in test_samples
    ]
    neural_predictions = _predict_all(predictor, test_samples, transition, device)

    print(f"dataset: {label}")
    print(
        f"samples: train={len(train_samples)}, test={len(test_samples)}, "
        f"experts={num_experts}, slots={args.replica_slots}"
    )
    test_sizes = [sample.current_ids.shape[0] for sample in test_samples]
    print(f"test tokens per forward: {min(test_sizes)} -> {max(test_sizes)}")
    print(f"training loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    for name, predictions in (
        ("layer-prior", prior_predictions),
        ("transition", transition_predictions),
        ("neural", neural_predictions),
    ):
        metrics = evaluate(
            test_samples,
            predictions,
            num_experts=num_experts,
            replica_slots=args.replica_slots,
        )
        print(_format_metrics(name, metrics))


if __name__ == "__main__":
    main()
