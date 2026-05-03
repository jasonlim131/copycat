# Sub-1B Interpretability Experiment: Targeted Knowledge Unlearning

A self-contained interpretability experiment that removes ~50 specific factual
associations from **Pythia-410M** using two state-of-the-art unlearning methods,
**RMU** and **NPO**, and analyzes the internal effect of each.

This experiment is independent of the surrounding Copycat repository — it lives
in its own `unlearning_interp/` subdirectory and adds no Copycat dependencies.

## What it does

1. Builds three small JSONL datasets (50 forget facts, 50 retain facts about
   different fictional people, 100 real-world trivia probes).
2. Evaluates the base model.
3. Runs **RMU** ([Li et al., 2024](https://arxiv.org/abs/2403.03218)) — forces
   hidden states at a chosen middle layer toward a fixed random direction on
   forget tokens, while anchoring retain tokens to a frozen reference.
4. Runs **NPO** ([Zhang et al., 2024](https://arxiv.org/abs/2404.05868)) — a
   DPO-shaped forget objective with a forward-KL retain regularizer.
5. Re-evaluates each unlearned model and produces four interpretability plots
   per method: layer-wise logit-lens accuracy, hidden-state cosine, gold-token
   logit drop, and a layer-wise linear probe.

## Setup

Single GPU recommended (T4 / A10 / 4090). 410M params ≈ 1.6 GB fp32; comfortably
fits with two model copies on a 16 GB card. **Internet access to
`huggingface.co` is required** to download Pythia-410M weights and the
wikitext-2 dataset on first run (the rest of the pipeline is offline).

```bash
cd unlearning_interp
pip install -r requirements.txt
```

### Offline smoke test (no HF access required)

To validate the implementation without downloading any pretrained model, run:

```bash
python scripts/offline_smoke.py
```

This builds a 1.2M-param GPT-NeoX from scratch with random weights and a
byte-level tokenizer, then runs 5 steps of RMU and 5 steps of NPO plus a
layer-wise logit-lens scan. Pass criteria: both losses decrease, gradients
flow, and interp values are well-formed. Useful for CI and for verifying a
local checkout before launching the real GPU run.

## Run

End-to-end (~55 min on a T4):

```bash
bash run_all.sh
```

Or step by step:

```bash
python run_experiment.py prepare-data
python run_experiment.py baseline-eval
python run_experiment.py unlearn --method rmu
python run_experiment.py eval    --method rmu
python run_experiment.py interp  --method rmu
python run_experiment.py unlearn --method npo
python run_experiment.py eval    --method npo
python run_experiment.py interp  --method npo
python run_experiment.py report
```

Outputs:
- `results/checkpoints/{rmu,npo}/` — unlearned model weights
- `results/metrics/{base,rmu,npo,interp_*}.json` — eval numbers
- `results/figures/*.png` — interpretability plots
- `results/metrics/REPORT.md` — summary table

## Expected results (Pythia-410M, default config)

| metric | base | RMU | NPO |
|---|---|---|---|
| forget exact-match | ~0.75 | **≤ 0.05** | **≤ 0.10** |
| retain exact-match | ~0.75 | ≥ 0.90 × base | ≥ 0.90 × base |
| trivia exact-match | ~0.20 | Δ < 0.03 | Δ < 0.05 |
| wikitext-2 PPL | ~14.5 | Δ < 0.5 | Δ < 0.8 |

In the layer-wise plots: RMU should produce a sharp dip in
`cos(h_base, h_unlearned)` at layer 7 (the hooked layer), while NPO should show
a milder, more diffuse shift across many layers — illustrating the
representation-level vs preference-level distinction between the two methods.

## Configuration

All hyperparameters live in `config.yaml`. Notable knobs:

| key | default | what it does |
|---|---|---|
| `rmu.layer_idx` | 7 | which transformer layer's output is hooked |
| `rmu.c` | 6.5 | norm of the steering vector `c·u` |
| `rmu.alpha` | 1200 | weight on the forget loss |
| `npo.beta` | 0.1 | DPO temperature |
| `npo.retain_kl_coeff` | 1.0 | KL anchor; ≥ 1.0 to prevent collapse |
| `train.update_layers` | [5,6,7] | only `mlp.dense_4h_to_h` of these layers is trained |

## Files

```
unlearning_interp/
├── config.yaml
├── data/
│   ├── build_data.py             # generates forget/retain/trivia JSONLs
│   ├── forget.jsonl              # 50 fictional facts (target of unlearning)
│   ├── retain.jsonl              # 50 fictional facts about different people
│   └── trivia_probe.jsonl        # 100 real-world Q/A
├── src/
│   ├── cfg.py                    # @dataclass Config + YAML loader + RNG seeding
│   ├── data_utils.py             # JSONL → tokenized batches with answer masks
│   ├── model_utils.py            # Pythia loader, layer-scope freezing, hooks
│   ├── rmu.py                    # RMU loss + training loop
│   ├── npo.py                    # NPO loss + training loop
│   ├── tar.py                    # (Phase 2) TAR-1 tamper-resistant training
│   ├── attacks.py                # (Phase 2) SFT-recovery relearning attack
│   ├── probes.py                 # (Phase 2) frozen 'is forget' linear probes
│   ├── trajectory.py             # (Phase 2) per-step mechanistic trajectory
│   ├── eval_harness.py           # exact-match, wikitext PPL, trivia probe
│   ├── interp.py                 # logit lens, hidden cosine, logit drop, probe
│   └── plotting.py               # standardized matplotlib helpers
├── scripts/
│   ├── offline_smoke.py          # CI-style smoke test (no HF access required)
│   ├── plot_h1_suppression.py    # (Phase 2) H1 verdict figure
│   ├── plot_h2_erasure.py        # (Phase 2) H2 verdict figure
│   ├── plot_h3_deepening.py      # (Phase 2) H3 verdict figure
│   └── plot_trajectory_overview.py  # (Phase 2) raw 4-panel trajectory plot
├── run_experiment.py             # CLI entrypoint
├── run_all.sh                    # one-shot pipeline
└── results/                      # populated by run_all.sh
```

# Phase 2 — Mechanistic trajectory of relearning

## Question

When a relearning attack recovers an unlearned fact, is it rebuilding the
representation from scratch, or just *unblocking* a representation that was
preserved the whole time? And does TAR's behavioral robustness reflect
representational erasure or merely a steeper barrier on the same preserved
path?

We discriminate three hypotheses by tracking layer-wise probe accuracy,
logit-lens accuracy, and hidden-cosine-to-base **at every step** of an
SFT-recovery attack:

| Hypothesis | Predicted trajectory shape | Verdict script |
|---|---|---|
| **H1 — Suppression** | probe high at t=0 while behavior near 0 | `scripts/plot_h1_suppression.py` |
| **H2 — Erasure / new pathway** | cosine-to-base stays low while behavior recovers | `scripts/plot_h2_erasure.py` |
| **H3 — Deepening (buried, not erased)** | probe rises *before* behavior during recovery | `scripts/plot_h3_deepening.py` |

The four plots are:
- `figures/h1_suppression.png` — at t=0, layer-wise probe accuracy + behavioral EM
- `figures/h2_erasure.png` — cosine-to-base trajectory + behavioral recovery
- `figures/h3_deepening.png` — probe(t) vs EM(t) with crossing-time annotations
- `figures/trajectory_overview_{rmu,tar}.png` — master 4-panel raw data plot

Each verdict script also prints a quantified support level on stdout (STRONG /
WEAK / REJECTED / INDETERMINATE).

## What's new vs Phase 1

- **TAR-1** (`src/tar.py`) — first-order tamper-resistant training applied on
  top of an RMU-edited checkpoint. Bilevel: outer minimizes a retain anchor
  while *increasing* L_forget at the post-adversary-SFT point. K=8 inner SGD
  steps simulate the adversary; same trainable scope as RMU.
- **Frozen 'is forget' probes** (`src/probes.py`) — trained once on the BASE
  model's mean-pooled hidden states, then frozen and reused as a knowledge
  meter for every (model × step) snapshot.
- **SFT-recovery attack with per-step callbacks** (`src/attacks.py`) — runs the
  adversary's targeted SFT and fires a callback at each scheduled step.
- **Trajectory orchestrator** (`src/trajectory.py`) — at each callback,
  computes (forget_em, forget_logp_gold, logit_lens_acc[layer],
  probe_acc[layer], hidden_cosine_to_base[layer]) and dumps a single JSON.

## Run on Modal (single A100-40GB)

If you don't have a local A100, the included Modal script runs the full Phase
1+2 pipeline on a serverless A100 in ~60–90 minutes:

```bash
pip install modal && modal token new
cd unlearning_interp
modal run scripts/modal_run.py                       # full pipeline, bf16
modal run scripts/modal_run.py --stage phase2        # if Phase 1 already ran
modal run scripts/modal_run.py --stage smoke         # CI-style smoke (no GPU work)
modal volume get copycat-unlearning-results / ./modal_results
```

Two persistent Modal volumes are created on first run:
- `hf-cache` — Pythia-410M and wikitext-2 weights/data, reused across runs
- `copycat-unlearning-results` — checkpoints, metrics, and figures

For gated models (Llama, Mistral) create a Modal secret first:

```bash
modal secret create huggingface HF_TOKEN=hf_xxxxxxxx
```

The script picks it up automatically when present.

## Run (Phase 2 only)

Phase 2 assumes Phase 1 has already produced the RMU checkpoint. Then:

```bash
python run_experiment.py unlearn --method tar --start rmu     # ~30-60 min
python run_experiment.py eval    --method tar
python run_experiment.py train-probes                          # <1 min
python run_experiment.py attack-trajectory --method rmu        # ~15 min
python run_experiment.py attack-trajectory --method tar        # ~15 min
python run_experiment.py plot-hypotheses --methods rmu tar
```

Outputs land in `results/metrics/trajectories/{rmu,tar}.json` and
`results/figures/h{1,2,3}_*.png`.

## Scope and limitations

Pythia-410M is a *mechanistic case study* scale. The point of this experiment
is to surface and discriminate trajectory hypotheses cheaply, not to claim a
benchmark number. Comparable trajectory work in the literature
(arxiv:2410.06606, arxiv:2410.12949, arxiv:2505.09500) reports endpoints
only, on 7B+ models. Probe-guided relearning attacks (arxiv:2506.01318,
arxiv:2504.14798) are adjacent prior art for the *attack* construction but
not for the *trajectory* analysis. Scaling validation is out of scope.

## Citations

- **RMU**: Li, N., et al. *The WMDP Benchmark.* arxiv:2403.03218 (2024).
- **NPO**: Zhang, R., et al. *Negative Preference Optimization.* arxiv:2404.05868 (2024).
- **TAR**: Tamirisa, R., et al. *Tamper-Resistant Safeguards for Open-Weight LLMs.*
  arxiv:2408.00761 (2024).
- **Adjacent endpoint analyses**: Hong et al. arxiv:2410.06606; Guo et al.
  arxiv:2410.12949; Hu et al. arxiv:2505.09500.
- **Adjacent probe-driven attacks**: Verifying Robust Unlearning arxiv:2504.14798;
  Prototypical Relearning arxiv:2506.01318.
