"""Standardized matplotlib figures saved as PNGs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_layerwise(
    series: Dict[str, Dict[int, float]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> Path:
    """`series` maps a label -> {layer: value}. Plots one line per label."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, layer_to_v in series.items():
        xs = sorted(layer_to_v.keys())
        ys = [layer_to_v[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_ensure(out_path), dpi=140)
    plt.close(fig)
    return out_path


def plot_logit_drop_bars(rows: List[dict], title: str, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(6, 0.18 * len(rows)), 4))
    xs = list(range(len(rows)))
    drops = [r["drop"] for r in rows]
    ax.bar(xs, drops)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("fact index")
    ax.set_ylabel("base_logit - unlearned_logit  (gold token)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(_ensure(out_path), dpi=140)
    plt.close(fig)
    return out_path


def plot_metric_table(rows: List[dict], title: str, out_path: Path) -> Path:
    """Render a small summary table as an image (handy for quick inspection)."""
    headers = list(rows[0].keys())
    cells = [[f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h]) for h in headers] for r in rows]
    fig, ax = plt.subplots(figsize=(1.5 * len(headers), 0.5 * (len(rows) + 1) + 1))
    ax.axis("off")
    ax.set_title(title)
    ax.table(cellText=cells, colLabels=headers, loc="center")
    fig.tight_layout()
    fig.savefig(_ensure(out_path), dpi=140)
    plt.close(fig)
    return out_path
