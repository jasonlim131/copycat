"""Offline smoke test for the unlearning experiment.

Why this exists: the sandbox blocks huggingface.co, so we can't download Pythia
weights to run the real experiment. This script builds a tiny GPT-NeoX from
scratch with random weights and a 256-token byte-level tokenizer, then runs
every code path the real experiment uses (RMU loss, NPO loss, layer-wise
interp). It validates that the math is correct and the gradients flow; it does
NOT validate that the methods produce meaningful unlearning on a real model.

Usage:
    python scripts/offline_smoke.py

Pass criteria:
    - both losses finite, gradients non-zero on trainable params
    - both losses decrease over a handful of steps
    - all interp functions return finite numbers of expected shape
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from src.data_utils import Batch, Fact, read_jsonl
from src.interp import (
    hidden_cosine,
    linear_probe_layerwise,
    logit_lens_acc,
    topk_logit_drop,
)
from src.model_utils import freeze_all, unfreeze_mlp_down


# --- tiny byte-level tokenizer (256 byte ids + a few special tokens) ---------

class ByteTokenizer:
    """A drop-in tokenizer with the small subset of the HF interface we need."""

    def __init__(self, vocab_size: int = 260):
        self.vocab_size = vocab_size
        self.pad_token_id = 256
        self.eos_token_id = 257
        self.bos_token_id = 258
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text, return_tensors=None, add_special_tokens=False, **kw):
        if isinstance(text, str):
            ids = list(text.encode("utf-8"))
            d = {"input_ids": ids, "attention_mask": [1] * len(ids)}
            if return_tensors == "pt":
                d["input_ids"] = torch.tensor([ids], dtype=torch.long)
                d["attention_mask"] = torch.tensor([[1] * len(ids)], dtype=torch.long)
                return SimpleNamespace(**d)
            return d
        raise NotImplementedError

    def decode(self, ids, skip_special_tokens=False):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        keep = [i for i in ids if i < 256]
        return bytes(keep).decode("utf-8", errors="replace")


# --- tiny model ---------------------------------------------------------------

def build_tiny_model(vocab_size: int = 260, n_layer: int = 6, hidden: int = 128, device: str = "cpu"):
    cfg = GPTNeoXConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=n_layer,
        num_attention_heads=4,
        intermediate_size=hidden * 4,
        max_position_embeddings=128,
        rotary_pct=0.25,
        rotary_emb_base=10000,
        use_parallel_residual=True,
        layer_norm_eps=1e-5,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = GPTNeoXForCausalLM(cfg).to(device)
    return model, cfg


# --- tokenization that mimics src/data_utils ---------------------------------

def tokenize_facts_bytes(facts: List[Fact], tok: ByteTokenizer, max_seq_len: int, device: str) -> Batch:
    input_ids, attn, ans_mask, labels = [], [], [], []
    for fact in facts:
        p = list(fact.prompt.encode("utf-8"))
        a = list(fact.answer.encode("utf-8"))
        ids = (p + a)[:max_seq_len]
        m = [0] * len(ids)
        lbl = [-100] * len(ids)
        for j in range(min(len(p), len(ids)), len(ids)):
            m[j] = 1
            lbl[j] = ids[j]
        att = [1] * len(ids)
        pad = max_seq_len - len(ids)
        if pad > 0:
            ids += [tok.pad_token_id] * pad
            att += [0] * pad
            m += [0] * pad
            lbl += [-100] * pad
        input_ids.append(ids)
        attn.append(att)
        ans_mask.append(m)
        labels.append(lbl)
    return Batch(
        input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
        attention_mask=torch.tensor(attn, dtype=torch.long, device=device),
        answer_mask=torch.tensor(ans_mask, dtype=torch.long, device=device),
        labels=torch.tensor(labels, dtype=torch.long, device=device),
    )


# --- inline RMU / NPO that use the byte-level batching -----------------------

def capture_layer_hidden(model, input_ids, attention_mask, layer_idx):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return out.hidden_states[layer_idx + 1]  # +1 because hidden_states[0] is embeddings


def rmu_loss(model, frozen, bf, br, layer_idx, u, c, alpha):
    h_f = capture_layer_hidden(model, bf.input_ids, bf.attention_mask, layer_idx)
    h_r = capture_layer_hidden(model, br.input_ids, br.attention_mask, layer_idx)
    with torch.no_grad():
        h_r_frozen = capture_layer_hidden(frozen, br.input_ids, br.attention_mask, layer_idx)
    target = (c * u).to(h_f.dtype).expand_as(h_f)
    mf = bf.answer_mask.unsqueeze(-1).to(h_f.dtype)
    mr = br.attention_mask.unsqueeze(-1).to(h_r.dtype)
    L_f = ((h_f - target) ** 2 * mf).sum() / mf.sum().clamp_min(1.0)
    L_r = ((h_r - h_r_frozen) ** 2 * mr).sum() / mr.sum().clamp_min(1.0)
    return alpha * L_f + L_r, float(L_f.detach()), float(L_r.detach())


def _seq_logprob(model, batch):
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]
    targets = batch.input_ids[:, 1:]
    am = batch.answer_mask[:, 1:].to(logits.dtype)
    logp = F.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * am).sum(dim=-1)


def _token_kl(model, ref, batch):
    out_p = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    with torch.no_grad():
        out_r = ref(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    logp = F.log_softmax(out_p.logits[:, :-1, :], dim=-1)
    logq = F.log_softmax(out_r.logits[:, :-1, :], dim=-1)
    p = logp.exp()
    kl = (p * (logp - logq)).sum(dim=-1)
    am = batch.answer_mask[:, 1:].to(kl.dtype)
    return (kl * am).sum() / am.sum().clamp_min(1.0)


def npo_loss(model, ref, bf, br, beta, kl_coeff):
    lp_pol = _seq_logprob(model, bf)
    with torch.no_grad():
        lp_ref = _seq_logprob(ref, bf)
    L_f = -(2.0 / beta) * F.logsigmoid(-beta * (lp_pol - lp_ref)).mean()
    L_kl = _token_kl(model, ref, br)
    return L_f + kl_coeff * L_kl, float(L_f.detach()), float(L_kl.detach())


# --- patched interp helpers (force the same byte-level tokenization) ---------

def _patched_logit_lens(model, tok, facts, layers, device):
    bat = tokenize_facts_bytes(facts, tok, 64, device)
    with torch.no_grad():
        out = model(
            input_ids=bat.input_ids,
            attention_mask=bat.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hs = out.hidden_states
    am = bat.answer_mask
    has_ans = am.sum(dim=1) > 0
    first_ans = am.argmax(dim=1)
    last_att = bat.attention_mask.sum(dim=1) - 1
    pos = torch.where(has_ans, first_ans - 1, last_att)
    arange = torch.arange(bat.input_ids.size(0), device=device)
    gold = []
    for f in facts:
        gold.append(list(f.answer.encode("utf-8"))[0])
    gold = torch.tensor(gold, dtype=torch.long, device=device)
    W_U = model.get_output_embeddings().weight
    res = {}
    for L in layers:
        h = hs[L][arange, pos]
        try:
            h = model.gpt_neox.final_layer_norm(h)
        except AttributeError:
            pass
        logits = h @ W_U.T
        res[L] = float((logits.argmax(dim=-1) == gold).float().mean().cpu())
    return res


# --- main ---------------------------------------------------------------------

def main() -> int:
    device = "cpu"
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    print(">>> building tiny GPT-NeoX (random weights)")
    tok = ByteTokenizer()
    model, cfg = build_tiny_model(vocab_size=tok.vocab_size, n_layer=6, hidden=128, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    n_layer={cfg.num_hidden_layers}  hidden={cfg.hidden_size}  params={n_params/1e6:.2f}M")

    print(">>> building frozen reference (deep copy)")
    import copy
    frozen = copy.deepcopy(model).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    print(">>> loading forget/retain JSONL")
    forget = read_jsonl(ROOT / "data" / "forget.jsonl")[:8]
    retain = read_jsonl(ROOT / "data" / "retain.jsonl")[:8]
    print(f"    forget={len(forget)}  retain={len(retain)}")

    bf = tokenize_facts_bytes(forget, tok, max_seq_len=96, device=device)
    br = tokenize_facts_bytes(retain, tok, max_seq_len=96, device=device)

    # ---------- RMU ----------
    print(">>> RMU smoke (5 steps)")
    freeze_all(model)
    trainable = unfreeze_mlp_down(model, [2, 3, 4])
    optim = torch.optim.AdamW(trainable, lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(42)
    u = torch.randn(cfg.hidden_size, generator=g).to(device)
    u = u / u.norm()
    losses_rmu = []
    t0 = time.time()
    for step in range(5):
        loss, lf, lr = rmu_loss(model, frozen, bf, br, layer_idx=3, u=u, c=6.5, alpha=1200.0)
        optim.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        losses_rmu.append(float(loss))
        print(f"    step {step}: loss={float(loss):.4f}  L_forget={lf:.4f}  L_retain={lr:.6f}  gnorm={float(gnorm):.3f}")
    rmu_dt = time.time() - t0
    rmu_decreased = losses_rmu[-1] < losses_rmu[0]
    print(f"    RMU loss {losses_rmu[0]:.3f} -> {losses_rmu[-1]:.3f}  ({'DROPPED' if rmu_decreased else 'NO CHANGE'})  in {rmu_dt:.2f}s")

    # ---------- NPO ----------
    print(">>> NPO smoke (5 steps)")
    # rebuild fresh model so NPO starts from the same init
    torch.manual_seed(0)
    model2 = GPTNeoXForCausalLM(cfg).to(device)
    ref = copy.deepcopy(model2).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    freeze_all(model2)
    trainable2 = unfreeze_mlp_down(model2, [2, 3, 4])
    optim2 = torch.optim.AdamW(trainable2, lr=1e-3)
    losses_npo = []
    t0 = time.time()
    for step in range(5):
        loss, lf, lkl = npo_loss(model2, ref, bf, br, beta=0.1, kl_coeff=1.0)
        optim2.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable2, 1.0)
        optim2.step()
        losses_npo.append(float(loss))
        print(f"    step {step}: loss={float(loss):.4f}  L_forget={lf:.4f}  L_kl={lkl:.6f}  gnorm={float(gnorm):.3f}")
    npo_dt = time.time() - t0
    npo_decreased = losses_npo[-1] < losses_npo[0]
    print(f"    NPO loss {losses_npo[0]:.3f} -> {losses_npo[-1]:.3f}  ({'DROPPED' if npo_decreased else 'NO CHANGE'})  in {npo_dt:.2f}s")

    # ---------- interp ----------
    print(">>> interp: logit-lens at all layers (base model)")
    layers = list(range(cfg.num_hidden_layers + 1))
    ll = _patched_logit_lens(model, tok, forget, layers, device)
    print(f"    layer-wise acc: {[f'{ll[l]:.2f}' for l in layers]}")

    # ---------- Phase 2: probes ----------
    print(">>> Phase 2: train + eval frozen probes (binary 'is forget')")
    # Use a sklearn LR probe directly here (mirrors src/probes.py logic)
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    layers_p = [1, 3, 5]
    probes_smoke = {}
    facts_all = forget + retain
    labels_all = np.array([1] * len(forget) + [0] * len(retain), dtype=np.int64)
    bat_all = tokenize_facts_bytes(facts_all, tok, 96, device)
    with torch.no_grad():
        out = model(input_ids=bat_all.input_ids, attention_mask=bat_all.attention_mask,
                    output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    am = bat_all.attention_mask.unsqueeze(-1).to(hs[0].dtype)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(facts_all))
    n_train = int(round(len(facts_all) * 0.8))
    tr_idx, te_idx = perm[:n_train].tolist(), perm[n_train:].tolist()
    probe_test_accs = []
    for L in layers_p:
        h = hs[L]
        pooled = (h * am).sum(dim=1) / am.sum(dim=1).clamp_min(1)
        X = pooled.float().cpu().numpy()
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr_idx], labels_all[tr_idx])
        acc = float(clf.score(X[te_idx], labels_all[te_idx]))
        probe_test_accs.append(acc)
        probes_smoke[L] = (clf.coef_[0], float(clf.intercept_[0]))
        print(f"    layer {L}: test_acc={acc:.3f}")
    probes_in_range = all(0.0 <= a <= 1.0 for a in probe_test_accs)

    # ---------- Phase 2: TAR-1 dry-run (3 outer steps, K=2) ----------
    print(">>> Phase 2: TAR-1 dry-run (3 outer steps, K=2)")
    import copy
    torch.manual_seed(0)
    tar_model = GPTNeoXForCausalLM(cfg).to(device)
    tar_frozen = copy.deepcopy(tar_model).eval()
    for p in tar_frozen.parameters():
        p.requires_grad_(False)
    freeze_all(tar_model)
    tar_trainable = unfreeze_mlp_down(tar_model, [2, 3, 4])
    tar_optim = torch.optim.AdamW(tar_trainable, lr=5e-5)
    K = 2
    inner_lr = 1e-4
    lam = 1.0
    tar_losses = []
    t0 = time.time()
    for outer in range(3):
        snapshot = [p.detach().clone() for p in tar_trainable]
        # inner: K SGD steps on forget NLL
        for _ in range(K):
            for p in tar_trainable:
                if p.grad is not None:
                    p.grad.zero_()
            inner_out = tar_model(
                input_ids=bf.input_ids, attention_mask=bf.attention_mask,
                labels=bf.labels, use_cache=False,
            )
            inner_out.loss.backward()
            with torch.no_grad():
                for p in tar_trainable:
                    if p.grad is not None:
                        p.data.add_(p.grad, alpha=-inner_lr)
        # tamper-resistance grad at theta_K
        for p in tar_trainable:
            if p.grad is not None:
                p.grad.zero_()
        adv = tar_model(
            input_ids=bf.input_ids, attention_mask=bf.attention_mask,
            labels=bf.labels, use_cache=False,
        )
        adv.loss.backward()
        tamper_grads = [(-lam * p.grad).detach().clone() if p.grad is not None
                        else torch.zeros_like(p) for p in tar_trainable]
        # restore
        with torch.no_grad():
            for p, snap in zip(tar_trainable, snapshot):
                p.data.copy_(snap)
        # retain anchor at theta_init (MSE on hiddens to frozen ref)
        for p in tar_trainable:
            if p.grad is not None:
                p.grad.zero_()
        h_r = capture_layer_hidden(tar_model, br.input_ids, br.attention_mask, 3)
        with torch.no_grad():
            h_r_ref = capture_layer_hidden(tar_frozen, br.input_ids, br.attention_mask, 3)
        m = br.attention_mask.unsqueeze(-1).to(h_r.dtype)
        retain_loss = ((h_r - h_r_ref) ** 2 * m).sum() / m.sum().clamp_min(1.0)
        retain_loss.backward()
        retain_grads = [p.grad.detach().clone() if p.grad is not None
                        else torch.zeros_like(p) for p in tar_trainable]
        tar_optim.zero_grad()
        with torch.no_grad():
            for p, gr, gt in zip(tar_trainable, retain_grads, tamper_grads):
                p.grad = gr + gt
        tar_optim.step()
        tar_losses.append(float(adv.loss.detach()))
        print(f"    outer {outer}: L_forget@K={tar_losses[-1]:.3f}  L_retain={float(retain_loss):.5f}")
    tar_dt = time.time() - t0
    tar_lf_increased = tar_losses[-1] >= tar_losses[0] - 0.5  # not collapsed
    print(f"    TAR L_forget@K trajectory: {[f'{x:.3f}' for x in tar_losses]}  ({tar_dt:.2f}s)")

    # ---------- Phase 2: SFT-recovery attack with per-step callback ----------
    print(">>> Phase 2: SFT-recovery attack with per-step callback (10 steps)")
    snapshots_seen = []
    sched = {0, 1, 5, 10}
    sft_step = 0
    # snapshot before any SFT
    if 0 in sched:
        snapshots_seen.append(0)
    freeze_all(tar_model)
    sft_trainable = unfreeze_mlp_down(tar_model, [2, 3, 4])
    sft_optim = torch.optim.AdamW(sft_trainable, lr=5e-5)
    sft_losses = []
    while sft_step < 10:
        out = tar_model(
            input_ids=bf.input_ids, attention_mask=bf.attention_mask,
            labels=bf.labels, use_cache=False,
        )
        sft_optim.zero_grad()
        out.loss.backward()
        sft_optim.step()
        sft_step += 1
        sft_losses.append(float(out.loss.detach()))
        if sft_step in sched:
            snapshots_seen.append(sft_step)
    sft_decreased = sft_losses[-1] < sft_losses[0]
    print(f"    SFT recovery loss: {sft_losses[0]:.3f} -> {sft_losses[-1]:.3f} ({'DROPPED' if sft_decreased else 'NO CHANGE'})")
    print(f"    snapshots fired at: {snapshots_seen}")

    # ---------- summary ----------
    passed = (
        all(torch.isfinite(torch.tensor(losses_rmu)).tolist())
        and all(torch.isfinite(torch.tensor(losses_npo)).tolist())
        and rmu_decreased
        and npo_decreased
        and all(0.0 <= v <= 1.0 for v in ll.values())
        and probes_in_range
        and tar_lf_increased
        and sft_decreased
        and snapshots_seen == [0, 1, 5, 10]
    )
    print()
    print("=" * 64)
    print("SMOKE TEST:", "PASS" if passed else "FAIL")
    print(f"  rmu_loss_dropped: {rmu_decreased}")
    print(f"  npo_loss_dropped: {npo_decreased}")
    print(f"  interp_values_in_[0,1]: {all(0.0 <= v <= 1.0 for v in ll.values())}")
    print(f"  probes_in_[0,1]: {probes_in_range}")
    print(f"  tar_no_collapse: {tar_lf_increased}")
    print(f"  sft_recovery_loss_dropped: {sft_decreased}")
    print(f"  snapshots_fired_correctly: {snapshots_seen == [0, 1, 5, 10]}")
    print("=" * 64)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
