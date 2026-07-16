from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch


class RunningMoments:
    """Exact online count, mean, and population standard deviation."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(device="cpu", dtype=torch.float64)
        self.count += values.numel()
        self.total += float(values.sum().item())
        self.total_sq += float((values * values).sum().item())

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0, "mean": float("nan"), "std": float("nan")}
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return {"count": self.count, "mean": mean, "std": variance**0.5}


def load_records(
    paths: list[Path],
) -> dict[tuple[int, int, int], dict[str, Any]]:
    records = {}
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=True)
        key = (
            int(record["rank"]),
            int(record["step"]),
            int(record["layer_id"]),
        )
        records[key] = record
    return records


def paired_records(
    records: dict[tuple[int, int, int], dict[str, Any]],
) -> Iterator[tuple[int, int, dict[str, Any], dict[str, Any]]]:
    for (rank_id, step, layer_id), record in sorted(records.items()):
        next_layer_id = record.get("next_gate_layer_id")
        if next_layer_id is None:
            continue
        next_layer_id = int(next_layer_id)
        next_record = records.get((rank_id, step, next_layer_id))
        if next_record is not None:
            yield layer_id, next_layer_id, record, next_record


def topk_overlap(predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    num_rows = min(predicted.shape[0], actual.shape[0])
    if num_rows == 0:
        return torch.empty(0, dtype=torch.float32)
    return torch.stack(
        [
            torch.isin(predicted[row], actual[row]).float().mean()
            for row in range(num_rows)
        ]
    )
