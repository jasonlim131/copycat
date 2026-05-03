"""Run the unlearning experiment on a single Modal A100.

Usage from your laptop (after `pip install modal` and `modal token new`):

    cd unlearning_interp
    modal run scripts/modal_run.py                       # full pipeline (Phase 1 + Phase 2)
    modal run scripts/modal_run.py --stage phase2        # assume Phase 1 outputs already in volume
    modal run scripts/modal_run.py --stage smoke         # offline smoke test (no HF download)
    modal run scripts/modal_run.py --stage phase2 --dtype bf16

Outputs land in a persistent Modal volume named `copycat-unlearning-results`.
Pull them down with:

    modal volume get copycat-unlearning-results / ./modal_results

The HuggingFace cache is reused across runs via a separate `hf-cache` volume,
so Pythia-410M and wikitext-2 are downloaded once.

Optional: gated models (Llama, Mistral) require an HF token. Create a Modal
secret called `huggingface` with key `HF_TOKEN`:

    modal secret create huggingface HF_TOKEN=hf_xxxxxxxx

The script picks it up automatically when present.
"""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]


# -- image: install our requirements + git, then layer the local dir on top ---

_REQUIREMENTS = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install(*_REQUIREMENTS)
    .env({
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface",
        "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
        "MPLBACKEND": "Agg",
    })
    # Layer the experiment code at /work. We exclude `results/` so the persistent
    # volume mount (below) takes precedence and survives across runs.
    .add_local_dir(
        str(ROOT),
        remote_path="/work/unlearning_interp",
        ignore=["results/**", "__pycache__/**", "*.pyc", ".git/**"],
    )
)


# -- volumes --------------------------------------------------------------------

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("copycat-unlearning-results", create_if_missing=True)


# -- secrets (optional) ---------------------------------------------------------

def _maybe_secret() -> list:
    try:
        return [modal.Secret.from_name("huggingface")]
    except Exception:
        return []


# -- app + remote function ------------------------------------------------------

app = modal.App("copycat-unlearning")


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60 * 5,  # 5 hours; a Phase 1+2 run on 410M is well under this
    volumes={
        "/cache/huggingface": hf_cache,
        "/work/unlearning_interp/results": results_vol,
    },
    secrets=_maybe_secret(),
)
def run_pipeline(stage: str = "all", dtype: str = "bf16") -> dict:
    """Run a stage of the experiment on the A100. Returns a small status dict."""
    import os
    import subprocess
    import sys
    import time

    os.chdir("/work/unlearning_interp")
    sys.path.insert(0, "/work/unlearning_interp")

    # Sanity check the GPU.
    print("=" * 60)
    print(f"stage    = {stage}")
    print(f"dtype    = {dtype}")
    print("=" * 60)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv"], check=False)

    # Override dtype in the config without rewriting the file.
    if dtype not in ("fp32", "bf16", "fp16"):
        raise ValueError(f"bad dtype: {dtype!r}")
    if dtype != "fp32":
        # Patch config.yaml in place (harmless: it's the image copy).
        cfg_path = Path("/work/unlearning_interp/config.yaml")
        text = cfg_path.read_text()
        new = []
        for line in text.splitlines():
            if line.startswith("dtype:"):
                new.append(f"dtype: {dtype}")
            else:
                new.append(line)
        cfg_path.write_text("\n".join(new) + "\n")
        print(f"  patched config.yaml dtype -> {dtype}")

    # Stage definitions — run as separate subprocess invocations so failures show
    # up at the right step in the log.
    phase1 = [
        ["python", "run_experiment.py", "prepare-data"],
        ["python", "run_experiment.py", "baseline-eval"],
        ["python", "run_experiment.py", "unlearn", "--method", "rmu"],
        ["python", "run_experiment.py", "eval",    "--method", "rmu"],
        ["python", "run_experiment.py", "interp",  "--method", "rmu"],
        ["python", "run_experiment.py", "unlearn", "--method", "npo"],
        ["python", "run_experiment.py", "eval",    "--method", "npo"],
        ["python", "run_experiment.py", "interp",  "--method", "npo"],
        ["python", "run_experiment.py", "report"],
    ]
    phase2 = [
        ["python", "run_experiment.py", "unlearn", "--method", "tar", "--start", "rmu"],
        ["python", "run_experiment.py", "eval",    "--method", "tar"],
        ["python", "run_experiment.py", "interp",  "--method", "tar"],
        ["python", "run_experiment.py", "train-probes"],
        ["python", "run_experiment.py", "attack-trajectory", "--method", "rmu"],
        ["python", "run_experiment.py", "attack-trajectory", "--method", "tar"],
        ["python", "run_experiment.py", "plot-hypotheses", "--methods", "rmu", "tar"],
    ]
    smoke = [["python", "scripts/offline_smoke.py"]]

    sequences = {
        "all": phase1 + phase2,
        "phase1": phase1,
        "phase2": phase2,
        "smoke": smoke,
    }
    if stage not in sequences:
        raise ValueError(f"unknown stage {stage!r}; choose from {list(sequences)}")

    t0 = time.time()
    for cmd in sequences[stage]:
        print()
        print("$ " + " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            return {"ok": False, "stage": stage, "failed_at": " ".join(cmd), "rc": rc}
        # Make Modal flush volume writes (cheap, idempotent).
        try:
            results_vol.commit()
        except Exception:
            pass

    elapsed = time.time() - t0
    print()
    print(f"=== stage {stage!r} OK in {elapsed/60:.1f} min ===")

    # Walk the results dir and report what was produced.
    results = Path("/work/unlearning_interp/results")
    summary = {
        "ok": True,
        "stage": stage,
        "elapsed_min": round(elapsed / 60, 2),
        "figures": sorted(str(p.relative_to(results)) for p in results.glob("figures/*.png")),
        "metrics": sorted(str(p.relative_to(results)) for p in results.glob("metrics/**/*.json")),
        "checkpoints": sorted(str(p.relative_to(results)) for p in results.glob("checkpoints/*")
                              if p.is_dir()),
    }
    print()
    print(f"figures: {len(summary['figures'])}, metrics: {len(summary['metrics'])}, checkpoints: {len(summary['checkpoints'])}")
    return summary


# -- local entrypoint -----------------------------------------------------------

@app.local_entrypoint()
def main(stage: str = "all", dtype: str = "bf16") -> None:
    """Kick off the remote run from your laptop. `stage` ∈ {all, phase1, phase2, smoke}."""
    print(f"launching stage={stage!r} dtype={dtype!r} on A100-40GB")
    summary = run_pipeline.remote(stage, dtype)
    print()
    print("=" * 60)
    if summary.get("ok"):
        print(f"OK in {summary['elapsed_min']} min")
        print(f"  {len(summary['figures'])} figures: {summary['figures'][:6]}{' ...' if len(summary['figures'])>6 else ''}")
        print(f"  {len(summary['metrics'])} metrics, {len(summary['checkpoints'])} checkpoints")
        print()
        print("pull results back to your laptop with:")
        print("  modal volume get copycat-unlearning-results / ./modal_results")
    else:
        print(f"FAILED at: {summary.get('failed_at')}  (rc={summary.get('rc')})")
    print("=" * 60)
