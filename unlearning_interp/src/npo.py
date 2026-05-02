"""Negative Preference Optimization (NPO).

Reference: Zhang et al., "Negative Preference Optimization: From Catastrophic
Collapse to Effective Unlearning", arxiv:2404.05868.

Loss form (forget side; per the paper, eq. 5):
    L_npo = -(2/beta) * E_forget[ log sigmoid( -beta * (logp_pi(y) - logp_ref(y)) ) ]

We add a forward-KL retain regularizer over the answer span to prevent catastrophic
collapse on retain knowledge — this matches the "NPO + retain" variant the paper
recommends as the practical default.

Same trainable scope as RMU (MLP `dense_4h_to_h` of layers 5-7) for fairness.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .cfg import Config
from .data_utils import Batch, Fact, iter_minibatches, tokenize_facts
from .model_utils import freeze_all, make_frozen_reference, unfreeze_mlp_down


def _seq_logprob_on_answer(model, batch: Batch) -> torch.Tensor:
    """Sum log-prob of answer tokens per sequence. Returns [B]."""
    out = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    )
    # Standard CLM shift: logits at position t predict token at t+1
    logits = out.logits[:, :-1, :]                          # [B, T-1, V]
    targets = batch.input_ids[:, 1:]                        # [B, T-1]
    ans_mask = batch.answer_mask[:, 1:].to(logits.dtype)    # [B, T-1]
    logp = F.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    return (tok_logp * ans_mask).sum(dim=-1)


def _token_kl_on_answer(model, ref, batch: Batch) -> torch.Tensor:
    """Mean KL(pi || ref) over answer-token positions. Returns scalar."""
    out_p = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    )
    with torch.no_grad():
        out_r = ref(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
        )
    logp = F.log_softmax(out_p.logits[:, :-1, :], dim=-1)
    logq = F.log_softmax(out_r.logits[:, :-1, :], dim=-1)
    p = logp.exp()
    kl_per_pos = (p * (logp - logq)).sum(dim=-1)             # [B, T-1]
    mask = batch.answer_mask[:, 1:].to(kl_per_pos.dtype)
    return (kl_per_pos * mask).sum() / mask.sum().clamp_min(1.0)


def npo_loss(
    model,
    ref,
    batch_forget: Batch,
    batch_retain: Batch,
    beta: float,
    kl_coeff: float,
) -> torch.Tensor:
    lp_pol = _seq_logprob_on_answer(model, batch_forget)
    with torch.no_grad():
        lp_ref = _seq_logprob_on_answer(ref, batch_forget)
    L_forget = -(2.0 / beta) * F.logsigmoid(-beta * (lp_pol - lp_ref)).mean()
    L_retain_kl = _token_kl_on_answer(model, ref, batch_retain)
    return L_forget + kl_coeff * L_retain_kl


def train_npo(
    cfg: Config,
    model,
    tokenizer,
    forget_facts: List[Fact],
    retain_facts: List[Fact],
) -> dict:
    device = cfg.device

    ref = make_frozen_reference(cfg.model_name, cfg.torch_dtype(), device)

    freeze_all(model)
    trainable = unfreeze_mlp_down(model, cfg.train.update_layers)
    optim = torch.optim.AdamW(
        trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    model.train()
    stats = {"loss": []}
    for epoch in range(cfg.npo.epochs):
        ret_iter = iter(_cycle_minibatches(retain_facts, cfg.train.batch_size))
        forget_batches = list(iter_minibatches(forget_facts, cfg.train.batch_size))
        pbar = tqdm(forget_batches, desc=f"NPO ep{epoch+1}/{cfg.npo.epochs}")

        optim.zero_grad()
        for step, fbatch in enumerate(pbar):
            rbatch = next(ret_iter)
            bf = tokenize_facts(fbatch, tokenizer, cfg.train.max_seq_len, device)
            br = tokenize_facts(rbatch, tokenizer, cfg.train.max_seq_len, device)
            loss = npo_loss(
                model, ref, bf, br,
                beta=cfg.npo.beta,
                kl_coeff=cfg.npo.retain_kl_coeff,
            )
            (loss / cfg.train.grad_accum).backward()
            if (step + 1) % cfg.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
                optim.step()
                optim.zero_grad()
            stats["loss"].append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        if (step + 1) % cfg.train.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
            optim.step()
            optim.zero_grad()

    model.eval()
    del ref
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return stats


def _cycle_minibatches(facts: List[Fact], batch_size: int):
    while True:
        for b in iter_minibatches(facts, batch_size):
            yield b
