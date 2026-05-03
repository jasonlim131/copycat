"""End-to-end sanity check of the Phase 2 pipeline on a tiny random-init model.

Exercises every Phase 2 code path — frozen probes, SFT-recovery attack with
per-step callbacks, trajectory orchestrator, all four hypothesis plots — using
a 1-2 M parameter byte-level GPT-NeoX (the same architecture the real
experiment uses, just shrunk). All numbers are MEANINGLESS — random weights
can't represent factual knowledge — but the pipeline plumbing is exercised
end-to-end and the four PNG figures are produced.

Use cases:
    - CI / pre-flight check before launching the real Modal A100 run.
    - Verifying a code change didn't break the trajectory dump format.

NOT a substitute for the real experiment.

Usage:
    python scripts/sanity_check.py
    # writes results/figures/*.png and results/metrics/trajectories/{rmu,tar}.json
"""
from __future__ import annotations

import copy
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import BatchEncoding, GPTNeoXConfig, GPTNeoXForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cfg import (
    AttacksCfg, Config, EvalCfg, InterpCfg, NPOCfg, Paths, RMUCfg, TARCfg,
    TrainCfg, TrajectoryCfg, seed_everything,
)
from src.data_utils import read_jsonl
from src.probes import save_probes, train_probes
from src.trajectory import run_trajectory


# --- HF-compatible byte tokenizer (uses BatchEncoding so `**ids` works) ------

class ByteTokenizer:
    """Minimal tokenizer that satisfies the duck-typed HF interface used by
    src/data_utils.py, src/eval_harness.py, and src/trajectory.py.

    Returns transformers.BatchEncoding so that:
      - `tok(text, return_tensors='pt').to(device)` works (.to() on BatchEncoding)
      - `model.generate(**ids, ...)` works (BatchEncoding is a Mapping)
      - `tok(text)['input_ids']` works (BatchEncoding is dict-like)
    """

    def __init__(self):
        self.vocab_size = 260
        self.pad_token_id = 256
        self.eos_token_id = 257
        self.bos_token_id = 258
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"

    def __call__(self, text, return_tensors=None, add_special_tokens=False, **kw):
        if not isinstance(text, str):
            raise NotImplementedError("only single-string tokenization in sanity test")
        ids = list(text.encode("utf-8"))
        attn = [1] * len(ids)
        if return_tensors == "pt":
            return BatchEncoding({
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attn], dtype=torch.long),
            })
        return BatchEncoding({"input_ids": ids, "attention_mask": attn})

    def decode(self, ids, skip_special_tokens=False):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        keep = [i for i in ids if i < 256]
        return bytes(keep).decode("utf-8", errors="replace")


# --- tiny model + sanity-scaled config ---------------------------------------

def build_tiny_model(vocab_size: int = 260):
    cfg = GPTNeoXConfig(
        vocab_size=vocab_size, hidden_size=128, num_hidden_layers=6,
        num_attention_heads=4, intermediate_size=512,
        max_position_embeddings=128, rotary_pct=0.25, rotary_emb_base=10000,
        use_parallel_residual=True, layer_norm_eps=1e-5, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = GPTNeoXForCausalLM(cfg)
    return model, cfg


def build_tiny_cfg(root: Path) -> Config:
    """Programmatically build a tiny-model Config without touching config.yaml."""
    return Config(
        seed=42,
        model_name="<tiny>",
        dtype="fp32",
        device="cpu",
        paths=Paths(
            forget="data/forget.jsonl",
            retain="data/retain.jsonl",
            trivia="data/trivia_probe.jsonl",
            checkpoints="results/checkpoints",
            metrics="results/metrics",
            figures="results/figures",
            steering_vector="results/checkpoints/u_sanity.pt",
            probes="results/checkpoints/probes_sanity.pt",
            trajectories="results/metrics/trajectories",
        ),
        train=TrainCfg(
            batch_size=4, grad_accum=1, max_seq_len=96,
            lr=5e-4, weight_decay=0.0, grad_clip=1.0,
            update_layers=[2, 3, 4],
        ),
        rmu=RMUCfg(epochs=1, layer_idx=3, c=6.5, alpha=100.0),
        npo=NPOCfg(epochs=1, beta=0.1, retain_kl_coeff=1.0),
        eval=EvalCfg(topk=5, wikitext_split="test[:1%]", wikitext_stride=128, max_new_tokens_pad=2),
        interp=InterpCfg(layers=[0, 1, 2, 3, 4, 5, 6], probe_test_frac=0.2),
        tar=TARCfg(epochs=1, inner_steps=2, inner_lr=1e-3, outer_lr=5e-4, lambda_resist=1.0),
        attacks=AttacksCfg(
            sft_max_steps=8,
            sft_lr=5e-4,
            sft_batch_size=4,
            snapshot_steps=[0, 1, 2, 4, 8],
            train_layers=[2, 3, 4],
        ),
        trajectory=TrajectoryCfg(
            layers=[0, 1, 2, 3, 4, 5, 6],
            probe_train_frac=0.8,
            probe_C=1.0,
            eval_subset=8,
        ),
        root=root,
    )


def _perturb(model, n_steps: int = 10, seed: int = 1) -> None:
    """Random SGD updates so the 'unlearned' model differs from the base —
    gives the trajectory plots some non-zero variance even on random weights."""
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=1e-3)
    for _ in range(n_steps):
        x = torch.randint(0, model.config.vocab_size, (2, 32))
        out = model(input_ids=x, labels=x, use_cache=False)
        optim.zero_grad()
        out.loss.backward()
        optim.step()


# --- main ---------------------------------------------------------------------

def main() -> int:
    t_total = time.time()
    cfg = build_tiny_cfg(ROOT)
    seed_everything(cfg.seed)

    print(">>> building tiny GPT-NeoX (random weights)")
    base_model, gn_cfg = build_tiny_model()
    base_model.eval()
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"    n_layer={gn_cfg.num_hidden_layers}  hidden={gn_cfg.hidden_size}  params={n_params/1e6:.2f}M")

    tok = ByteTokenizer()

    print(">>> loading forget/retain (small subset)")
    forget = read_jsonl(ROOT / "data" / "forget.jsonl")[:8]
    retain = read_jsonl(ROOT / "data" / "retain.jsonl")[:8]
    print(f"    forget={len(forget)}  retain={len(retain)}")

    print(">>> training frozen probes on base (binary 'is forget')")
    t0 = time.time()
    probes = train_probes(
        base_model, tok, forget, retain,
        layers=cfg.trajectory.layers,
        device="cpu", max_seq_len=cfg.train.max_seq_len,
        train_frac=cfg.trajectory.probe_train_frac,
        C=cfg.trajectory.probe_C, seed=cfg.seed,
    )
    save_probes(probes, cfg.abspath(cfg.paths.probes))
    test_accs = [probes[L].test_acc for L in sorted(probes.keys())]
    print(f"    layers={sorted(probes.keys())}  test_acc range: {min(test_accs):.3f} – {max(test_accs):.3f}  ({time.time()-t0:.1f}s)")

    for method, perturb_seed in [("rmu", 1), ("tar", 2)]:
        print(f">>> running trajectory for {method!r}")
        t0 = time.time()
        # Build a freshly-perturbed copy so each method starts from a different
        # mock 'unlearned' state. The orchestrator mutates the model in place.
        model = copy.deepcopy(base_model)
        _perturb(model, n_steps=10, seed=perturb_seed)
        out = run_trajectory(
            cfg, method, model, base_model, tok,
            forget, retain, probes,
        )
        print(f"    -> {out}  ({time.time()-t0:.1f}s)")

    print(">>> rendering hypothesis plots")
    plot_scripts = [
        ROOT / "scripts" / "plot_h1_suppression.py",
        ROOT / "scripts" / "plot_h2_erasure.py",
        ROOT / "scripts" / "plot_h3_deepening.py",
        ROOT / "scripts" / "plot_trajectory_overview.py",
    ]
    for s in plot_scripts:
        cmd = ["python", str(s), "--root", str(ROOT), "--methods", "rmu", "tar"]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  PLOT FAIL: {s}")
            return 1

    fig_dir = ROOT / "results" / "figures"
    expected = [
        "h1_suppression.png",
        "h2_erasure.png",
        "h3_deepening.png",
        "trajectory_overview_rmu.png",
        "trajectory_overview_tar.png",
    ]
    print(">>> verifying outputs")
    missing = [e for e in expected if not (fig_dir / e).exists()]
    for e in expected:
        p = fig_dir / e
        if p.exists():
            print(f"    {e}: {p.stat().st_size // 1024} KB")
    if missing:
        print(f"  MISSING: {missing}")
        return 1

    print()
    print("=" * 60)
    print(f"SANITY CHECK: PASS  ({time.time()-t_total:.1f}s)")
    print(f"  trajectories: {cfg.abspath(cfg.paths.trajectories)}")
    print(f"  figures:      {fig_dir}")
    print("  (numbers meaningless — pipeline plumbing only)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
