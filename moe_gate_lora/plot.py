from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_overlap_plot(rows: list[dict[str, Any]], output: Path) -> None:
    """Replace the plot with the latest online aggregate."""
    if not rows:
        return
    import matplotlib.pyplot as plt

    labels = [f"{row['layer_i']}->{row['layer_j']}" for row in rows]
    means = np.array([row["lora_overlap_mean"] for row in rows])
    stds = np.array([row["lora_overlap_std"] for row in rows])
    baseline = np.array([row["baseline_overlap_mean"] for row in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(8.0, len(rows) * 0.55), 4.8))
    ax.plot(x, baseline, linewidth=1.2, label="Base gate")
    ax.plot(x, means, marker="o", linewidth=1.8, label="Gate + LoRA")
    ax.fill_between(
        x,
        np.maximum(means - stds, 0.0),
        np.minimum(means + stds, 1.0),
        alpha=0.15,
    )
    ax.set_xticks(x, labels=labels, rotation=90)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Top-k overlap")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
