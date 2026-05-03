"""Master 4-panel trajectory plot per method.

Renders the raw mechanistic-trajectory data so the user can eyeball patterns
that the H1/H2/H3 verdict scripts may miss. Per method:

    panel A  forget_em  +  forget_logp_gold      vs  SFT step
    panel B  probe_acc[layer]                   heatmap
    panel C  logit_lens_acc[layer]              heatmap
    panel D  hidden_cosine_to_base[layer]       heatmap

Usage:
    python scripts/plot_trajectory_overview.py
    # reads:  results/metrics/trajectories/{rmu,tar}.json
    # writes: results/figures/trajectory_overview_{rmu,tar}.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DEFAULT = Path(__file__).resolve().parents[1]


def _load(traj_dir: Path, method: str):
    p = traj_dir / f"{method}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _heatmap(ax, steps: List[int], layers: List[int], snaps, field: str, title: str,
             vmin: float = 0.0, vmax: float = 1.0, cmap: str = "viridis"):
    M = np.full((len(layers), len(steps)), np.nan, dtype=float)
    for j, s in enumerate(steps):
        d = snaps[str(s)][field]
        for i, L in enumerate(layers):
            v = d.get(str(L))
            if v is not None:
                M[i, j] = float(v)
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(L) for L in layers], fontsize=8)
    ax.set_xlabel("SFT-recovery step")
    ax.set_ylabel("layer")
    ax.set_title(title)
    return im


def _overview_for_method(data, out_path: Path, method: str) -> Path:
    steps = sorted(int(s) for s in data["snapshots"])
    layers = list(data["layers"])
    snaps = data["snapshots"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Panel A — behavioral
    ax = axes[0][0]
    em = [snaps[str(s)]["forget_em"] for s in steps]
    lp = [snaps[str(s)]["forget_logp_gold"] for s in steps]
    ax.plot(steps, em, marker="s", color="C3", label="forget EM (left)")
    ax.set_xscale("symlog")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("forget EM", color="C3")
    ax.set_xlabel("SFT-recovery step")
    ax2 = ax.twinx()
    ax2.plot(steps, lp, marker="o", color="C0", label="logp gold (right)")
    ax2.set_ylabel("mean log-prob of gold answer token", color="C0")
    ax.set_title(f"{method.upper()} — behavior vs step")
    ax.grid(True, alpha=0.3)

    # Panel B — probe heatmap
    im_b = _heatmap(axes[0][1], steps, layers, snaps, "probe_acc",
                    f"{method.upper()} — probe accuracy", vmin=0.4, vmax=1.0)
    fig.colorbar(im_b, ax=axes[0][1], shrink=0.8)

    # Panel C — logit-lens heatmap
    im_c = _heatmap(axes[1][0], steps, layers, snaps, "logit_lens_acc",
                    f"{method.upper()} — logit-lens accuracy on forget", vmin=0.0, vmax=1.0)
    fig.colorbar(im_c, ax=axes[1][0], shrink=0.8)

    # Panel D — hidden cosine heatmap (recovered vs base)
    im_d = _heatmap(axes[1][1], steps, layers, snaps, "hidden_cosine_to_base",
                    f"{method.upper()} — cos(h_recovered, h_base)", vmin=-0.2, vmax=1.0,
                    cmap="coolwarm")
    fig.colorbar(im_d, ax=axes[1][1], shrink=0.8)

    fig.suptitle(f"Mechanistic trajectory of relearning — {method.upper()}", y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT_DEFAULT))
    p.add_argument("--methods", nargs="+", default=["rmu", "tar"])
    args = p.parse_args()

    root = Path(args.root)
    traj_dir = root / "results/metrics/trajectories"
    fig_dir = root / "results/figures"

    written = []
    for method in args.methods:
        data = _load(traj_dir, method)
        if data is None:
            print(f"[skip] no trajectory for {method!r}")
            continue
        out = fig_dir / f"trajectory_overview_{method}.png"
        _overview_for_method(data, out, method)
        written.append(out)

    print(f"\n=== Trajectory overview ===")
    for w in written:
        print(f"  wrote {w}")
    if not written:
        print("  (no data)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
