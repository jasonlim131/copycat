"""H3 verdict plot: DEEPENING (representation re-emerges before behavior).

Diagnostic: trajectory of probe accuracy vs behavioral EM. If during recovery
the probe rises *before* EM (probe leads), the representation was buried — not
erased — and the SFT attack first re-surfaces the representation, after which
behavior follows. If they rise in lockstep or EM leads, H3 is rejected.

Lead/lag is quantified by the integer step difference between the two
crossing points: t_probe(>=0.7) and t_em(>=0.5). A positive lead = probe is
ahead of behavior.

Usage:
    python scripts/plot_h3_deepening.py
    # reads:  results/metrics/trajectories/{rmu,tar}.json
    # writes: results/figures/h3_deepening.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DEFAULT = Path(__file__).resolve().parents[1]


def _load(traj_dir: Path, method: str):
    p = traj_dir / f"{method}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _crossing(steps: List[int], values: List[float], threshold: float) -> Optional[int]:
    """The first step at which `values` crosses `threshold` (>=). None if never."""
    for s, v in zip(steps, values):
        if v >= threshold:
            return s
    return None


def _verdict(t_probe: Optional[int], t_em: Optional[int]) -> str:
    if t_em is None:
        return "INDETERMINATE (behavior never recovered to threshold)"
    if t_probe is None:
        return "REJECTED (representation never re-emerged: probe stayed ≈ chance)"
    lead = t_em - t_probe
    if lead > 0:
        return f"SUPPORT (probe led behavior by {lead} steps)"
    if lead == 0:
        return "WEAK (lockstep recovery; no clear lead)"
    return f"REJECTED (behavior led probe by {-lead} steps)"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT_DEFAULT))
    p.add_argument("--methods", nargs="+", default=["rmu", "tar"])
    p.add_argument("--probe-thresh", type=float, default=0.7)
    p.add_argument("--em-thresh", type=float, default=0.5)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    root = Path(args.root)
    traj_dir = root / "results/metrics/trajectories"
    out_path = Path(args.out) if args.out else root / "results/figures/h3_deepening.png"

    fig, axes = plt.subplots(
        1, len(args.methods),
        figsize=(5.5 * len(args.methods), 4.2),
        squeeze=False,
    )
    rows = []
    for c, method in enumerate(args.methods):
        ax = axes[0][c]
        data = _load(traj_dir, method)
        if data is None:
            ax.set_axis_off()
            ax.set_title(f"{method.upper()} — no data")
            continue
        steps = sorted(int(s) for s in data["snapshots"])
        em = [data["snapshots"][str(s)]["forget_em"] for s in steps]
        # max-over-layers probe acc — the "knowledge meter" reading.
        probe_max = []
        for s in steps:
            pa = data["snapshots"][str(s)]["probe_acc"]
            probe_max.append(max(float(v) for v in pa.values()))

        ax.plot(steps, probe_max, marker="o", label="probe_acc (max over layers)")
        ax.plot(steps, em, marker="s", label="forget EM", color="C3")
        ax.axhline(args.probe_thresh, color="C0", lw=0.6, ls="--", alpha=0.5)
        ax.axhline(args.em_thresh, color="C3", lw=0.6, ls="--", alpha=0.5)
        ax.set_xscale("symlog")
        ax.set_xlabel("SFT-recovery step")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{method.upper()}: probe vs behavior trajectory")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

        t_probe = _crossing(steps, probe_max, args.probe_thresh)
        t_em = _crossing(steps, em, args.em_thresh)
        rows.append((method, t_probe, t_em))

        # annotate the crossings on the plot
        if t_probe is not None:
            ax.axvline(max(t_probe, 0.5), color="C0", lw=0.8, ls=":", alpha=0.7)
        if t_em is not None:
            ax.axvline(max(t_em, 0.5), color="C3", lw=0.8, ls=":", alpha=0.7)

    fig.suptitle(
        "H3 — DEEPENING test\n"
        "probe crosses its threshold *before* EM ⇒ representation re-emerges first (H3 supported)",
        y=1.04,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== H3 (deepening) verdict ===")
    print(f"figure: {out_path}")
    print(f"thresholds: probe ≥ {args.probe_thresh}, EM ≥ {args.em_thresh}")
    for method, t_probe, t_em in rows:
        v = _verdict(t_probe, t_em)
        print(f"  {method:>6s}: t_probe={t_probe}  t_em={t_em}  -> {v}")
    if not rows:
        print("  (no data)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
