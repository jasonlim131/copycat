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
fits with two model copies on a 16 GB card.

```bash
cd unlearning_interp
pip install -r requirements.txt
```

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
│   ├── eval_harness.py           # exact-match, wikitext PPL, trivia probe
│   ├── interp.py                 # logit lens, hidden cosine, logit drop, probe
│   └── plotting.py               # standardized matplotlib helpers
├── run_experiment.py             # CLI entrypoint
├── run_all.sh                    # one-shot pipeline
└── results/                      # populated by run_all.sh
```

## Citations

- **RMU**: Li, N., et al. *The WMDP Benchmark: Measuring and Reducing Malicious
  Use With Unlearning.* arxiv:2403.03218 (2024).
- **NPO**: Zhang, R., et al. *Negative Preference Optimization: From Catastrophic
  Collapse to Effective Unlearning.* arxiv:2404.05868 (2024).
