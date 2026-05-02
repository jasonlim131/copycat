#!/usr/bin/env bash
# One-shot reproduction. Run from the experiment root.
# Expected runtime on a single T4: ~55 minutes.
set -euo pipefail

cd "$(dirname "$0")"

python run_experiment.py prepare-data
python run_experiment.py baseline-eval

python run_experiment.py unlearn --method rmu
python run_experiment.py eval    --method rmu
python run_experiment.py interp  --method rmu

python run_experiment.py unlearn --method npo
python run_experiment.py eval    --method npo
python run_experiment.py interp  --method npo

python run_experiment.py report
