#!/usr/bin/env bash
# One-shot reproduction. Run from the experiment root.
# Phase 1 expected runtime on a single T4: ~55 minutes.
# Phase 2 adds ~30-60 minutes (TAR + per-step trajectories on a 4090/A10).
set -euo pipefail

cd "$(dirname "$0")"

# === Phase 1 — RMU and NPO ===
python run_experiment.py prepare-data
python run_experiment.py baseline-eval

python run_experiment.py unlearn --method rmu
python run_experiment.py eval    --method rmu
python run_experiment.py interp  --method rmu

python run_experiment.py unlearn --method npo
python run_experiment.py eval    --method npo
python run_experiment.py interp  --method npo

python run_experiment.py report

# === Phase 2 — TAR + mechanistic-trajectory of relearning ===
python run_experiment.py unlearn --method tar --start rmu
python run_experiment.py eval    --method tar
python run_experiment.py interp  --method tar

python run_experiment.py train-probes
python run_experiment.py attack-trajectory --method rmu
python run_experiment.py attack-trajectory --method tar
python run_experiment.py plot-hypotheses --methods rmu tar
