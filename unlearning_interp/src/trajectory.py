"""Per-step mechanistic-trajectory orchestration.

For a starting unlearned model (RMU or TAR), runs an SFT-recovery attack and at
each scheduled snapshot step computes a vector of mechanistic + behavioral
diagnostics. The full trajectory is dumped to JSON for hypothesis-discriminating
plots (scripts/plot_h{1,2,3}_*.py).

Diagnostics computed at every snapshot step `t`:
  - forget_em            : exact-match accuracy on forget facts (free-form generation)
  - forget_logp_gold     : mean log-prob of the gold first-answer token (smooth EM)
  - logit_lens_acc[L]    : layer-wise logit-lens accuracy on forget facts
  - probe_acc[L]         : layer-wise frozen-probe ('is forget') accuracy on the post-edit model
  - hidden_cosine[L]     : layer-wise mean cosine to the BASE model's hidden states on forget answer tokens
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .attacks import sft_recovery_attack
from .cfg import Config
from .data_utils import Fact, first_answer_token_id, tokenize_facts
from .interp import logit_lens_acc
from .model_utils import capture_all_hiddens
from .probes import FrozenProbe, eval_probes_on_model


@torch.no_grad()
def _forget_logp_gold(model, tokenizer, facts: List[Fact], device: str, max_seq_len: int) -> float:
    """Mean log-prob assigned to each fact's first gold answer token, at the
    last-prompt position. Smooth analogue of EM: even when EM=0, this often
    moves measurably during recovery."""
    batch = tokenize_facts(facts, tokenizer, max_seq_len, device)
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    logp = F.log_softmax(out.logits.float(), dim=-1)
    has_ans = batch.answer_mask.sum(dim=1) > 0
    first_ans = batch.answer_mask.argmax(dim=1)
    last_att = batch.attention_mask.sum(dim=1) - 1
    pos = torch.where(has_ans, first_ans - 1, last_att)
    arange = torch.arange(batch.input_ids.size(0), device=device)
    gold = torch.tensor(
        [first_answer_token_id(f, tokenizer) for f in facts],
        device=device, dtype=torch.long,
    )
    pred_logp = logp[arange, pos].gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    return float(pred_logp.mean().cpu())


@torch.no_grad()
def _forget_em_quick(model, tokenizer, facts: List[Fact], device: str, max_new_tokens_pad: int) -> float:
    """Lightweight EM — same recipe as eval_harness.exact_match_acc but trimmed
    for speed (no per-fact logging, no paraphrases)."""
    n_hit, n_total = 0, 0
    for fact in facts:
        gold = fact.answer.strip().lower()
        ans_tok_len = len(tokenizer(fact.answer, add_special_tokens=False)["input_ids"])
        ids = tokenizer(fact.prompt, return_tensors="pt").to(device)
        out = model.generate(
            **ids,
            max_new_tokens=ans_tok_len + max_new_tokens_pad,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        cont = tokenizer.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        n_hit += int(gold in cont.lower())
        n_total += 1
    return n_hit / max(n_total, 1)


@torch.no_grad()
def _hidden_cosine_to_base_per_layer(
    model, base_hiddens: tuple, batch, layers: List[int], device: str
) -> Dict[int, float]:
    """Cosine sim per layer between `model`'s hiddens and the precomputed
    base hiddens, averaged over forget answer-token positions."""
    h_oth = capture_all_hiddens(model, batch.input_ids, batch.attention_mask)
    mask = batch.answer_mask.bool()
    out: Dict[int, float] = {}
    for L in layers:
        a = base_hiddens[L][mask]
        b = h_oth[L][mask]
        if a.numel() == 0:
            out[L] = float("nan")
            continue
        cos = F.cosine_similarity(a, b, dim=-1).mean().float().cpu()
        out[L] = float(cos)
    return out


def _make_snapshot_fn(
    cfg: Config,
    model,
    tokenizer,
    forget_facts: List[Fact],
    retain_facts: List[Fact],
    probes: Dict[int, FrozenProbe],
    base_hiddens_forget: tuple,
    forget_batch,
    layers: List[int],
    out_dump: Dict,
):
    """Return a closure that, when called with step `t`, fills `out_dump[t]`
    with the full diagnostic vector for the *current* state of `model`."""
    device = cfg.device
    max_seq_len = cfg.train.max_seq_len
    pad = cfg.eval.max_new_tokens_pad

    eval_subset = cfg.trajectory.eval_subset if cfg.trajectory else 0
    forget_eval = forget_facts[: eval_subset] if eval_subset else forget_facts

    def snapshot(step: int) -> None:
        record: Dict = {"step": step}
        record["forget_em"] = _forget_em_quick(model, tokenizer, forget_eval, device, pad)
        record["forget_logp_gold"] = _forget_logp_gold(
            model, tokenizer, forget_eval, device, max_seq_len
        )
        record["logit_lens_acc"] = logit_lens_acc(
            model, tokenizer, forget_eval, layers, device, max_seq_len
        )
        record["probe_acc"] = eval_probes_on_model(
            model, tokenizer, forget_facts, retain_facts, probes,
            device, max_seq_len, test_only=True,
        )
        record["hidden_cosine_to_base"] = _hidden_cosine_to_base_per_layer(
            model, base_hiddens_forget, forget_batch, layers, device,
        )
        out_dump[step] = record

    return snapshot


def run_trajectory(
    cfg: Config,
    method: str,
    model,
    base_model,
    tokenizer,
    forget_facts: List[Fact],
    retain_facts: List[Fact],
    probes: Dict[int, FrozenProbe],
) -> Path:
    """Run the SFT-recovery attack on `model` with per-step trajectory dumps.

    `model` will be MUTATED in place (the adversary fine-tunes it). Caller is
    responsible for passing a freshly loaded post-edit checkpoint.

    Writes results to `<paths.trajectories>/<method>.json` and returns the path.
    """
    assert cfg.attacks is not None, "config.yaml is missing the `attacks:` block"
    assert cfg.trajectory is not None, "config.yaml is missing the `trajectory:` block"
    layers = list(cfg.trajectory.layers)
    device = cfg.device

    # Precompute the base model's hidden states on the forget set ONCE — they're
    # the reference for the cosine-to-base trajectory and do not change.
    eval_subset = cfg.trajectory.eval_subset
    forget_eval = forget_facts[: eval_subset] if eval_subset else forget_facts
    forget_batch = tokenize_facts(forget_eval, tokenizer, cfg.train.max_seq_len, device)
    with torch.no_grad():
        base_hiddens = capture_all_hiddens(base_model, forget_batch.input_ids, forget_batch.attention_mask)

    out_dump: Dict[int, Dict] = {}
    snapshot_fn = _make_snapshot_fn(
        cfg, model, tokenizer, forget_facts, retain_facts, probes,
        base_hiddens, forget_batch, layers, out_dump,
    )

    sft_stats = sft_recovery_attack(
        cfg, model, tokenizer,
        recovery_facts=forget_facts,
        snapshot_fn=snapshot_fn,
    )

    out_dir = cfg.abspath(cfg.paths.trajectories)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{method}.json"

    payload = {
        "method": method,
        "attack": "sft_recovery",
        "schedule": sorted(out_dump.keys()),
        "layers": layers,
        "snapshots": {str(k): v for k, v in sorted(out_dump.items())},
        "sft_loss": sft_stats["loss"],
        "snapshots_taken": sft_stats["snapshots_taken"],
        "n_forget_eval": len(forget_eval),
        "n_forget_total": len(forget_facts),
        "n_retain_total": len(retain_facts),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path
