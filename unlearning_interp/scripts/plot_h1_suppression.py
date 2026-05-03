"""H1 verdict plot: SUPPRESSION (representation persists despite zero behavior).

Diagnostic: at t=0 (post-edit model, before any recovery SFT), compute the
layer-wise frozen-probe 'is forget fact' accuracy. If the probe still scores
well above chance while the model's free-form behavioral EM is near zero,
the representation has been suppressed-not-erased.

Usage:
    python scripts/plot_h1_suppression.py [--root ...]
    # reads:  results/metrics/trajectories/{rmu,tar}.json
    # writes: results/figures/h1_suppression.png
    # prints: verdict summary (STRONG / WEAK / REJECTED) per method
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DEFAULT = Path(__file__).resolve().parents[1]


def _load(method_dir: Path, method: str):
    p = method_dir / f"{method}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _verdict(probe_max: float, forget_em: float,
             probe_thresh: float = 0.7, em_thresh: float = 0.2) -> str:
    if forget_em > em_thresh:
        return f"REJECTED (behavioral EM {forget_em:.2f} > {em_thresh:.2f}: edit didn't take)"
    if probe_max >= probe_thresh:
        return f"STRONG SUPPORT (probe {probe_max:.2f} ≥ {probe_thresh:.2f}, EM {forget_em:.2f})"
    if probe_max > 0.5 + (probe_thresh - 0.5) / 2:
        return f"WEAK SUPPORT (probe {probe_max:.2f}, EM {forget_em:.2f})"
    return f"REJECTED (probe {probe_max:.2f} ≈ chance — representation also gone)"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT_DEFAULT))
    p.add_argument("--methods", nargs="+", default=["rmu", "tar"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    root = Path(args.root)
    traj_dir = root / "results/metrics/trajectories"
    out_path = Path(args.out) if args.out else root / "results/figures/h1_suppression.png"

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", label="chance")

    rows = []
    for method in args.methods:
        data = _load(traj_dir, method)
        if data is None:
            print(f"[skip] no trajectory for {method!r} at {traj_dir}/{method}.json")
            continue
        snap0 = data["snapshots"].get("0")
        if snap0 is None:
            print(f"[skip] {method}: no t=0 snapshot")
            continue
        layers = sorted(int(k) for k in snap0["probe_acc"].keys())
        probe_vals = [snap0["probe_acc"][str(L)] for L in layers]
        forget_em = float(snap0["forget_em"])
        probe_max = max(probe_vals)
        ax.plot(layers, probe_vals, marker="o", label=f"{method.upper()}  (forget EM={forget_em:.2f})")
        rows.append((method, probe_max, forget_em))

    ax.set_xlabel("layer")
    ax.set_ylabel("frozen-probe accuracy on forget vs retain")
    ax.set_title("H1 — SUPPRESSION test  (post-edit model, t=0)\n"
                 "high probe + low EM ⇒ representation preserved despite suppressed behavior")
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    # ---- stdout verdict ----
    print(f"\n=== H1 (suppression) verdict ===")
    print(f"figure: {out_path}")
    for method, probe_max, forget_em in rows:
        v = _verdict(probe_max, forget_em)
        print(f"  {method:>6s}: probe_max={probe_max:.3f}  forget_em={forget_em:.3f}  -> {v}")
    if not rows:
        print("  (no data)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
