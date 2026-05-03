"""H2 verdict plot: ERASURE / NEW PATHWAY (recovery doesn't restore the original direction).

Diagnostic: hidden-state cosine to the BASE model on forget answer-token
positions, tracked across SFT-recovery steps. If during recovery the cosine
returns to ~1.0, the model is rebuilding the original representation (rejecting
H2). If cosine stays low while behavioral EM climbs, the model is solving the
forget task via a different pathway (supporting H2).

Plot: per method, two subplots side by side
    left:  cosine-to-base trajectory at the edited layer (and ±2 neighbors)
    right: forget EM trajectory on the same x-axis

Usage:
    python scripts/plot_h2_erasure.py
    # reads:  results/metrics/trajectories/{rmu,tar}.json
    # writes: results/figures/h2_erasure.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DEFAULT = Path(__file__).resolve().parents[1]


def _load(traj_dir: Path, method: str):
    p = traj_dir / f"{method}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _series(data, field: str) -> Dict[int, Dict[int, float]]:
    """Returns {step: {layer: value}} for `field`, sorted by step."""
    out = {}
    for k in sorted((int(s) for s in data["snapshots"]), key=int):
        snap = data["snapshots"][str(k)]
        out[k] = {int(L): float(v) for L, v in snap[field].items()}
    return out


def _picks(layers: List[int], target: int, k: int = 3) -> List[int]:
    """Pick `k` layers nearest to `target`, sorted ascending."""
    if not layers:
        return []
    return sorted(sorted(layers, key=lambda L: abs(L - target))[:k])


def _verdict(cos_final: float, em_final: float,
             em_thresh: float = 0.5, cos_thresh: float = 0.7) -> str:
    if em_final < em_thresh:
        return f"INDETERMINATE (recovery never reached EM≥{em_thresh:.2f}; ran longer?)"
    if cos_final >= cos_thresh:
        return (f"REJECTED — cos_final={cos_final:.2f} ≥ {cos_thresh:.2f}: "
                f"recovery RESTORED original representation (consistent with H1/H3)")
    return (f"SUPPORT — cos_final={cos_final:.2f} < {cos_thresh:.2f} while EM={em_final:.2f}: "
            f"recovery built a DIFFERENT pathway")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT_DEFAULT))
    p.add_argument("--methods", nargs="+", default=["rmu", "tar"])
    p.add_argument("--target-layer", type=int, default=7,
                   help="The layer the unlearning hooked (default: 7, RMU's layer_idx).")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    root = Path(args.root)
    traj_dir = root / "results/metrics/trajectories"
    out_path = Path(args.out) if args.out else root / "results/figures/h2_erasure.png"

    fig, axes = plt.subplots(
        len(args.methods), 2,
        figsize=(11, 3.6 * len(args.methods)),
        squeeze=False,
    )
    rows = []
    for r, method in enumerate(args.methods):
        data = _load(traj_dir, method)
        if data is None:
            for c in (0, 1):
                axes[r][c].set_axis_off()
                axes[r][c].set_title(f"{method.upper()} — no data")
            continue
        cos_traj = _series(data, "hidden_cosine_to_base")
        steps = sorted(cos_traj.keys())
        layers_avail = sorted({L for s in cos_traj.values() for L in s.keys()})
        picks = _picks(layers_avail, args.target_layer, k=3)

        ax_l = axes[r][0]
        for L in picks:
            ys = [cos_traj[s].get(L, float("nan")) for s in steps]
            ax_l.plot(steps, ys, marker="o", label=f"layer {L}")
        ax_l.set_xscale("symlog")
        ax_l.set_xlabel("SFT-recovery step")
        ax_l.set_ylabel(f"cos(h_recovered, h_base)  on forget answer tokens")
        ax_l.set_title(f"{method.upper()}: hidden-state alignment to BASE")
        ax_l.set_ylim(-0.2, 1.05)
        ax_l.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax_l.grid(True, alpha=0.3)
        ax_l.legend(loc="lower right")

        ax_r = axes[r][1]
        em_ys = [data["snapshots"][str(s)]["forget_em"] for s in steps]
        ax_r.plot(steps, em_ys, marker="s", color="C3", label="forget EM")
        ax_r.set_xscale("symlog")
        ax_r.set_xlabel("SFT-recovery step")
        ax_r.set_ylabel("forget exact-match accuracy")
        ax_r.set_title(f"{method.upper()}: behavioral recovery")
        ax_r.set_ylim(-0.02, 1.02)
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(loc="lower right")

        # verdict at final step using the layer closest to the target
        final = steps[-1]
        L0 = min(picks, key=lambda L: abs(L - args.target_layer)) if picks else args.target_layer
        cos_final = cos_traj[final].get(L0, float("nan"))
        em_final = data["snapshots"][str(final)]["forget_em"]
        rows.append((method, L0, cos_final, em_final))

    fig.suptitle(
        "H2 — ERASURE/NEW-PATHWAY test\n"
        "high EM with cos≪1 ⇒ recovery built a different pathway (H2 supported)",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== H2 (erasure / new pathway) verdict ===")
    print(f"figure: {out_path}")
    for method, L, cos_final, em_final in rows:
        v = _verdict(cos_final, em_final)
        print(f"  {method:>6s} (layer {L}): cos_final={cos_final:.3f}  em_final={em_final:.3f}  -> {v}")
    if not rows:
        print("  (no data)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
