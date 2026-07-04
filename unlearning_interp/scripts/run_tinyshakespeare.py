"""Real-data Phase 2 demo: pretrain a tiny GPT-NeoX on TinyShakespeare, finetune
on the forget/retain bios so they're genuinely memorized, then run the full
Phase 2 pipeline (RMU + frozen probes + SFT-recovery trajectory + four
hypothesis plots).

Unlike scripts/sanity_check.py (random weights, meaningless numbers), this
trains a model that actually learns Shakespeare-level English plus our 100
bios. The trajectory plots will carry real signal — at toy scale.

Network: pulls TinyShakespeare from raw.githubusercontent.com (1.1 MB).
Compute: ~10–30 min on a 4-core Xeon with AMX bf16. CPU-only.

Usage:
    python scripts/run_tinyshakespeare.py
        # writes results/figures/*.png and results/metrics/trajectories/rmu.json
"""
from __future__ import annotations

import copy
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# --- bootstrap: install requirements if missing ------------------------------

def _bootstrap_requirements() -> None:
    """pip-install from requirements.txt on a fresh environment."""
    req_file = Path(__file__).resolve().parents[1] / "requirements.txt"
    if not req_file.exists():
        return
    try:
        import torch          # noqa: F401
        import transformers   # noqa: F401
        import sklearn        # noqa: F401
    except ImportError:
        print("  bootstrapping: pip install -r requirements.txt …", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "-r", str(req_file)],
            check=True,
        )
        print("  done.", flush=True)

_bootstrap_requirements()

# ----------------------------------------------------------------------------

import torch
from transformers import BatchEncoding, GPTNeoXConfig, GPTNeoXForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- clone official WMDP RMU repo (forward_with_cache) -----------------------

WMDP_REPO_URL = "https://github.com/centerforaisafety/wmdp.git"
WMDP_CACHE = Path.home() / ".cache" / "wmdp_rmu"

def _ensure_wmdp_repo() -> None:
    if not WMDP_CACHE.exists():
        print(f"  cloning WMDP RMU repo to {WMDP_CACHE} …")
        subprocess.run(
            ["git", "clone", "--depth", "1", WMDP_REPO_URL, str(WMDP_CACHE)],
            check=True, capture_output=True,
        )
        print("  done.")
    if str(WMDP_CACHE) not in sys.path:
        sys.path.insert(0, str(WMDP_CACHE))

_ensure_wmdp_repo()
from rmu.utils import forward_with_cache  # official activation-capture hook  # noqa: E402

# -----------------------------------------------------------------------------

from src.cfg import (
    AttacksCfg, Config, EvalCfg, InterpCfg, NPOCfg, Paths, RMUCfg, TARCfg,
    TrainCfg, TrajectoryCfg, seed_everything,
)
from src.data_utils import read_jsonl, tokenize_facts
from src.probes import save_probes, train_probes
from src.model_utils import freeze_all, unfreeze_mlp_down
from src.trajectory import run_trajectory


# --- HF-compatible byte tokenizer (same as sanity_check) ---------------------

class ByteTokenizer:
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
            raise NotImplementedError
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


# --- Shakespeare download + tokenization -------------------------------------

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def fetch_shakespeare(cache: Path) -> torch.Tensor:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {SHAKESPEARE_URL}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, cache)
    text = cache.read_text()
    print(f"  loaded {len(text):,} chars of Shakespeare")
    tokens = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    print(f"  -> {len(tokens):,} byte tokens")
    return tokens


# --- model -------------------------------------------------------------------

def build_model() -> GPTNeoXForCausalLM:
    """~3.5 M-param GPT-NeoX. 6 layers so update_layers=[2,3,4] is well-defined."""
    cfg = GPTNeoXConfig(
        vocab_size=260, hidden_size=192, num_hidden_layers=6,
        num_attention_heads=6, intermediate_size=768,
        max_position_embeddings=128, rotary_pct=0.25, rotary_emb_base=10000,
        use_parallel_residual=True, layer_norm_eps=1e-5, tie_word_embeddings=True,
    )
    torch.manual_seed(42)
    model = GPTNeoXForCausalLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"  built GPT-NeoX  L={cfg.num_hidden_layers}  H={cfg.hidden_size}  params={n/1e6:.2f}M")
    return model


# --- pretraining loop --------------------------------------------------------

def pretrain_shakespeare(model, tokens: torch.Tensor, n_steps: int, batch_size: int,
                         ctx: int, lr: float, warmup: int = 100) -> list:
    """LM pretraining on Shakespeare in fp32.

    Uses a linear warmup followed by cosine decay.
    """
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                              eps=1e-7, betas=(0.9, 0.95))
    losses = []
    g = torch.Generator().manual_seed(0)

    print(f"  pretraining: {n_steps} steps, bs={batch_size}, ctx={ctx}, lr={lr}, warmup={warmup}")
    t0 = time.time()
    n_nan = 0
    for step in range(n_steps):
        # linear warmup -> cosine decay
        if step < warmup:
            lr_now = lr * (step + 1) / warmup
        else:
            import math
            progress = (step - warmup) / max(1, n_steps - warmup)
            lr_now = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optim.param_groups:
            pg["lr"] = lr_now

        starts = torch.randint(0, len(tokens) - ctx - 1, (batch_size,), generator=g)
        x = torch.stack([tokens[s : s + ctx] for s in starts])
        y = torch.stack([tokens[s + 1 : s + ctx + 1] for s in starts])
        out = model(input_ids=x, labels=y, use_cache=False)
        loss = out.loss
        if not torch.isfinite(loss):
            n_nan += 1
            optim.zero_grad()
            if n_nan > 5:
                raise RuntimeError(f"pretrain diverged: {n_nan} NaN losses")
            continue
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses.append(float(loss.detach()))
        if step % max(1, n_steps // 15) == 0 or step == n_steps - 1:
            dt = time.time() - t0
            tps = (step + 1) * batch_size * ctx / dt
            print(f"    step {step:4d}/{n_steps}  loss={loss.item():.3f}  lr={lr_now:.2e}  ({tps:.0f} tok/s, {dt/60:.1f} min)")
    print(f"  pretrain done: final loss {losses[-1]:.3f} in {(time.time()-t0)/60:.1f} min")
    return losses


# --- finetune on bios so the model genuinely memorizes them ------------------

def finetune_on_bios(model, tokenizer, forget_facts, retain_facts, n_epochs: int,
                     lr: float, max_seq_len: int) -> list:
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    facts = forget_facts + retain_facts

    print(f"  finetuning {len(facts)} bios for {n_epochs} epochs")
    t0 = time.time()
    for epoch in range(n_epochs):
        # shuffle
        perm = torch.randperm(len(facts))
        ep_loss = 0.0
        for i in perm.tolist():
            batch = tokenize_facts([facts[i]], tokenizer, max_seq_len, "cpu")
            out = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                use_cache=False,
            )
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.detach()))
            ep_loss += losses[-1]
        ep_loss /= len(facts)
        if epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            print(f"    epoch {epoch+1:2d}/{n_epochs}  mean loss {ep_loss:.3f}  ({(time.time()-t0):.1f}s)")
    print(f"  finetune done in {(time.time()-t0):.1f}s")
    return losses


# --- RMU via official WMDP forward_with_cache --------------------------------
# Uses forward_with_cache() from the cloned wmdp/rmu/utils.py for activation
# capture (hook-based, identical to the paper's implementation).  Loss formula
# mirrors rmu/unlearn.py: MSE(forget_acts, control_vec) + alpha*MSE(retain_acts,
# frozen_retain_acts).  Note: official "alpha" scales the RETAIN term; our
# cfg.rmu.alpha scales the FORGET term — we keep our naming and pass
# retain_coeff as the retain weight so configs are backward-compatible.

def _neox_module(model, layer_id: int):
    """Return the GPT-NeoX transformer block used as the hook target."""
    return model.gpt_neox.layers[layer_id]


@torch.no_grad()
def _make_control_vector(frozen_ref, retain_facts, tokenizer, cfg, hidden_size, device):
    """Build a retain-orthogonal unit steering vector (shape [1,1,H]), scaled by c.

    Mirrors the official random_vector / norm * steering_coeff construction but
    projects out the mean retain direction first so L_forget doesn't structurally
    overlap with the retain subspace.
    """
    from src.data_utils import iter_minibatches, tokenize_facts as tok_facts

    # Collect mean retain activations at hook layer from frozen reference
    frozen_ref.eval()
    retain_acts = []
    layer_id = cfg.rmu.layer_idx
    frozen_module = _neox_module(frozen_ref, layer_id)
    for batch in iter_minibatches(retain_facts, cfg.train.batch_size):
        br = tok_facts(batch, tokenizer, cfg.train.max_seq_len, device)
        inputs = {"input_ids": br.input_ids, "attention_mask": br.attention_mask}
        h = forward_with_cache(frozen_ref, inputs, module=frozen_module, no_grad=True)
        retain_acts.append(h.mean(dim=(0, 1)))   # [H]
    retain_dir = torch.stack(retain_acts).mean(0)
    retain_dir = retain_dir / retain_dir.norm().clamp_min(1e-8)

    # Sample random vector (Gaussian, like rmu.py), project out retain direction
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    u = torch.randn(hidden_size, generator=g)
    u = u - (u @ retain_dir) * retain_dir
    u = u / u.norm().clamp_min(1e-8)
    control_vec = (cfg.rmu.c * u).to(device).reshape(1, 1, hidden_size)
    print(f"  control vector: c={cfg.rmu.c}, retain-orthogonal "
          f"(residual cos={float(u @ retain_dir):.4f})")
    return control_vec


def train_rmu_inplace(cfg, model, frozen_ref, tokenizer, forget_facts, retain_facts):
    """RMU via official WMDP forward_with_cache.

    Uses forward_with_cache() from wmdp/rmu/utils.py for hook-based activation
    capture. Loss formula mirrors rmu/unlearn.py: MSE to control vector on forget,
    MSE to frozen activations on retain.  AdamW from torch (transformers dropped
    it in v5), parameter scope via our freeze_all + unfreeze_mlp_down.
    """
    from src.data_utils import iter_minibatches, tokenize_facts as tok_facts

    device = cfg.device
    H = model.config.hidden_size
    layer_id = cfg.rmu.layer_idx

    freeze_all(model)
    trainable = unfreeze_mlp_down(model, cfg.train.update_layers)
    # torch.optim.AdamW — transformers.AdamW was removed in transformers>=5.0
    optim = torch.optim.AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    control_vec = _make_control_vector(frozen_ref, retain_facts, tokenizer, cfg, H, device)

    updated_module = _neox_module(model, layer_id)
    frozen_module  = _neox_module(frozen_ref, layer_id)

    print(f"  RMU (official forward_with_cache): epochs={cfg.rmu.epochs}, "
          f"layer_idx={layer_id}, alpha={cfg.rmu.alpha}, "
          f"retain_coeff={cfg.rmu.retain_coeff}, c={cfg.rmu.c}")

    def cycle(facts):
        while True:
            for b in iter_minibatches(facts, cfg.train.batch_size):
                yield b

    losses = []
    model.train()
    for epoch in range(cfg.rmu.epochs):
        ret_iter = cycle(retain_facts)
        forget_batches = list(iter_minibatches(forget_facts, cfg.train.batch_size))
        for fbatch in forget_batches:
            rbatch = next(ret_iter)
            bf = tok_facts(fbatch, tokenizer, cfg.train.max_seq_len, device)
            br = tok_facts(rbatch, tokenizer, cfg.train.max_seq_len, device)

            f_inputs = {"input_ids": bf.input_ids, "attention_mask": bf.attention_mask}
            r_inputs = {"input_ids": br.input_ids, "attention_mask": br.attention_mask}

            # Forget loss: MSE of updated activations to control vector
            updated_forget = forward_with_cache(
                model, f_inputs, module=updated_module, no_grad=False
            )
            L_forget = torch.nn.functional.mse_loss(
                updated_forget, control_vec.expand_as(updated_forget)
            )

            # Retain loss: MSE of updated vs frozen activations (all positions)
            updated_retain = forward_with_cache(
                model, r_inputs, module=updated_module, no_grad=False
            )
            with torch.no_grad():
                frozen_retain = forward_with_cache(
                    frozen_ref, r_inputs, module=frozen_module, no_grad=True
                )
            L_retain = torch.nn.functional.mse_loss(updated_retain, frozen_retain)

            # alpha scales forget (our convention); retain_coeff scales retain
            loss = cfg.rmu.alpha * L_forget + cfg.rmu.retain_coeff * L_retain

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
            optim.step()
            losses.append(float(loss.detach()))

        print(f"    RMU epoch {epoch+1}/{cfg.rmu.epochs}: "
              f"loss={losses[-1]:.3f}  L_forget={float(L_forget):.4f}  L_retain={float(L_retain):.4f}")

    model.eval()
    return losses


# --- behavioral checks -------------------------------------------------------

@torch.no_grad()
def forget_em(model, tokenizer, facts, max_new_tokens_pad: int = 2) -> float:
    n_hit, n = 0, 0
    model.eval()
    for fact in facts:
        gold = fact.answer.strip().lower()
        ans_tok_len = len(tokenizer(fact.answer)["input_ids"])
        ids = tokenizer(fact.prompt, return_tensors="pt")
        out = model.generate(
            **ids,
            max_new_tokens=ans_tok_len + max_new_tokens_pad,
            do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        cont = tokenizer.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        n_hit += int(gold in cont.lower())
        n += 1
    return n_hit / max(n, 1)


# --- config builder ----------------------------------------------------------

def build_cfg(root: Path) -> Config:
    return Config(
        seed=42,
        model_name="<tinyshakes>",     # bypassed; we don't go through HF
        dtype="fp32",
        device="cpu",
        paths=Paths(
            forget="data/forget.jsonl",
            retain="data/retain.jsonl",
            trivia="data/trivia_probe.jsonl",
            checkpoints="results/checkpoints",
            metrics="results/metrics",
            figures="results/figures",
            steering_vector="results/checkpoints/u_tinyshakes.pt",
            probes="results/checkpoints/probes_tinyshakes.pt",
            trajectories="results/metrics/trajectories",
        ),
        train=TrainCfg(
            batch_size=4, grad_accum=1, max_seq_len=80,
            lr=5e-4, weight_decay=0.0, grad_clip=1.0,
            update_layers=[3, 4, 5],
        ),
        rmu=RMUCfg(epochs=7, layer_idx=4, c=10.0, alpha=2500.0, retain_coeff=4.0),
        npo=NPOCfg(epochs=2, beta=0.1, retain_kl_coeff=1.0),
        eval=EvalCfg(topk=5, wikitext_split="test[:1%]", wikitext_stride=128, max_new_tokens_pad=4),
        interp=InterpCfg(layers=[0, 1, 2, 3, 4, 5, 6], probe_test_frac=0.2),
        tar=TARCfg(epochs=1, inner_steps=4, inner_lr=1e-4, outer_lr=3e-4, lambda_resist=1.0),
        attacks=AttacksCfg(
            sft_max_steps=64,
            sft_lr=3e-4,
            sft_batch_size=4,
            snapshot_steps=[0, 1, 2, 4, 8, 16, 32, 64],
            train_layers=[3, 4, 5],
        ),
        trajectory=TrajectoryCfg(
            layers=[0, 1, 2, 3, 4, 5, 6],
            probe_train_frac=0.8,
            probe_C=1.0,
            eval_subset=20,
        ),
        root=root,
    )


# --- main --------------------------------------------------------------------

def main() -> int:
    t_total = time.time()
    cfg = build_cfg(ROOT)
    seed_everything(cfg.seed)
    torch.set_num_threads(4)

    tokenizer = ByteTokenizer()
    base_dir = cfg.abspath(cfg.paths.checkpoints) / "tinyshakes_base"

    forget = read_jsonl(ROOT / "data" / "forget.jsonl")
    retain = read_jsonl(ROOT / "data" / "retain.jsonl")

    if base_dir.exists():
        print(f">>> [1-5/9] loading cached base model from {base_dir}  (skip pretrain+finetune)")
        model = GPTNeoXForCausalLM.from_pretrained(str(base_dir))
        em_after_finetune = forget_em(model, tokenizer, forget[:20], cfg.eval.max_new_tokens_pad)
        print(f"    forget={len(forget)}  retain={len(retain)}  base forget EM={em_after_finetune:.3f}")
    else:
        print(">>> [1/9] downloading TinyShakespeare")
        shakes_tokens = fetch_shakespeare(ROOT / "results" / "tinyshakespeare.txt")

        print(">>> [2/9] building tiny GPT-NeoX")
        model = build_model()

        print(">>> [3/9] pretraining on Shakespeare")
        pretrain_shakespeare(model, shakes_tokens, n_steps=1500, batch_size=32, ctx=64, lr=1e-3)

        print(">>> [4/9] loading bios")
        print(f"    forget={len(forget)}  retain={len(retain)}")

        print(">>> [5/9] fine-tuning on bios (until they're memorized)")
        finetune_on_bios(model, tokenizer, forget, retain, n_epochs=30, lr=3e-4, max_seq_len=cfg.train.max_seq_len)
        em_after_finetune = forget_em(model, tokenizer, forget[:20], cfg.eval.max_new_tokens_pad)
        print(f"    forget EM after finetune (20 facts): {em_after_finetune:.3f}")

        base_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(base_dir)

    base_model_for_traj = copy.deepcopy(model).eval()
    for p in base_model_for_traj.parameters():
        p.requires_grad_(False)

    print(">>> [6/9] running RMU unlearning")
    # Frozen ref = a fresh deep copy of the post-finetune model
    frozen_ref = copy.deepcopy(model).eval()
    for p in frozen_ref.parameters():
        p.requires_grad_(False)
    train_rmu_inplace(cfg, model, frozen_ref, tokenizer, forget, retain)
    em_after_rmu = forget_em(model, tokenizer, forget[:20], cfg.eval.max_new_tokens_pad)
    em_retain_after_rmu = forget_em(model, tokenizer, retain[:20], cfg.eval.max_new_tokens_pad)
    print(f"    forget EM after RMU (20):  {em_after_rmu:.3f}  (was {em_after_finetune:.3f})")
    print(f"    retain EM after RMU (20):  {em_retain_after_rmu:.3f}")

    rmu_dir = cfg.abspath(cfg.paths.checkpoints) / "rmu"
    rmu_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(rmu_dir)

    print(">>> [7/9] training frozen 'is forget' probes on base")
    probes = train_probes(
        base_model_for_traj, tokenizer, forget, retain,
        layers=cfg.trajectory.layers,
        device="cpu", max_seq_len=cfg.train.max_seq_len,
        train_frac=cfg.trajectory.probe_train_frac,
        C=cfg.trajectory.probe_C, seed=cfg.seed,
    )
    save_probes(probes, cfg.abspath(cfg.paths.probes))
    test_accs = [probes[L].test_acc for L in sorted(probes.keys())]
    print(f"    layers={sorted(probes.keys())}  test_acc range {min(test_accs):.3f} – {max(test_accs):.3f}")

    print(">>> [8/9] running SFT-recovery trajectory on the RMU-edited model")
    out_path = run_trajectory(
        cfg, "rmu", model, base_model_for_traj, tokenizer,
        forget, retain, probes,
    )
    print(f"    -> {out_path}")

    print(">>> [9/9] rendering hypothesis plots")
    for s in [
        ROOT / "scripts" / "plot_h1_suppression.py",
        ROOT / "scripts" / "plot_h2_erasure.py",
        ROOT / "scripts" / "plot_h3_deepening.py",
        ROOT / "scripts" / "plot_trajectory_overview.py",
    ]:
        cmd = ["python", str(s), "--root", str(ROOT), "--methods", "rmu"]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  PLOT FAIL: {s}")
            return 1

    fig_dir = ROOT / "results" / "figures"
    expected = ["h1_suppression.png", "h2_erasure.png", "h3_deepening.png",
                "trajectory_overview_rmu.png"]
    print(">>> verifying outputs")
    missing = []
    for e in expected:
        p = fig_dir / e
        if p.exists():
            print(f"    {e}: {p.stat().st_size // 1024} KB")
        else:
            missing.append(e)
    if missing:
        print(f"  MISSING: {missing}")
        return 1

    print()
    print("=" * 60)
    print(f"TINYSHAKESPEARE RUN: PASS  ({(time.time()-t_total)/60:.1f} min)")
    print(f"  forget EM: pre-RMU {em_after_finetune:.3f}  post-RMU {em_after_rmu:.3f}")
    print(f"  retain EM: post-RMU {em_retain_after_rmu:.3f}")
    print(f"  trajectory: {cfg.abspath(cfg.paths.trajectories)/'rmu.json'}")
    print(f"  figures:    {fig_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
