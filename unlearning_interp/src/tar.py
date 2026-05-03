"""TAR-1: first-order tamper-resistant training, applied on top of RMU.

Reference: Tamirisa et al. 2024, "Tamper-Resistant Safeguards for Open-Weight LLMs"
(arxiv:2408.00761). Bilevel optimization where the OUTER objective is to keep the
forget loss high *after* an INNER simulated-adversary SFT loop:

    minimize over theta:    retain_anchor(theta) - lambda * L_forget(theta_K)
    where theta_K = SGD_K(L_forget, theta, inner_lr)

We use the TAR-1 first-order approximation: instead of differentiating through the
inner loop, we treat the gradient at theta_K as a stand-in for the outer gradient,
which avoids second-order bookkeeping. In practice:

    1) snapshot trainable params (theta_init)
    2) run K inner SGD steps mutating params toward theta_K
    3) compute   g_outer = +grad(L_forget at theta_K)
       (positive sign: defender wants L_forget HIGH at theta_K)
    4) restore theta_init
    5) compute g_retain = grad(retain_anchor at theta_init)
    6) AdamW step with combined gradient g_retain + (-lambda * g_outer)
       (we want to *reduce* "retain-anchor + (-lambda)*L_forget(theta_K)",
        which is the same as moving theta in the direction
        -g_retain - (-lambda)*g_outer = -g_retain + lambda*g_outer.
        So set p.grad = g_retain + (-lambda)*g_outer  before optim.step().)

Design choices for sub-1B feasibility:
- adversary updates the SAME parameter scope as the defender (mlp.dense_4h_to_h
  of layers 5-7). This models a targeted-SFT attacker, which is the practical
  threat — full-param SFT on a few hundred examples isn't substantially stronger.
- retain anchor reuses RMU's MSE-to-frozen-base on retain hidden states (see rmu.py)
  so retain capability is preserved by the same mechanism RMU already validates.
"""
from __future__ import annotations

from typing import List

import torch
from tqdm import tqdm

from .cfg import Config
from .data_utils import Batch, Fact, iter_minibatches, tokenize_facts
from .model_utils import (
    capture_hidden,
    freeze_all,
    make_frozen_reference,
    unfreeze_mlp_down,
)


def _forget_nll(model, batch: Batch) -> torch.Tensor:
    """Mean NLL on answer tokens (the adversary's loss = the defender's negation)."""
    out = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        use_cache=False,
    )
    return out.loss


def _retain_anchor_loss(model, frozen, batch_retain: Batch, layer_idx: int) -> torch.Tensor:
    """RMU-style MSE-to-frozen on retain hiddens at the chosen layer."""
    h = capture_hidden(model, batch_retain.input_ids, batch_retain.attention_mask, layer_idx)
    with torch.no_grad():
        h_ref = capture_hidden(frozen, batch_retain.input_ids, batch_retain.attention_mask, layer_idx)
    mask = batch_retain.attention_mask.unsqueeze(-1).to(h.dtype)
    return ((h - h_ref) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)


def train_tar(
    cfg: Config,
    model,
    tokenizer,
    forget_facts: List[Fact],
    retain_facts: List[Fact],
) -> dict:
    """Apply TAR-1 to `model`, in place. Caller is responsible for starting from
    an RMU-edited checkpoint (the typical use case)."""
    assert cfg.tar is not None, "config.yaml is missing the `tar:` block"
    device = cfg.device
    K = cfg.tar.inner_steps
    inner_lr = cfg.tar.inner_lr
    lam = cfg.tar.lambda_resist

    frozen = make_frozen_reference(cfg.model_name, cfg.torch_dtype(), device)
    freeze_all(model)
    trainable = unfreeze_mlp_down(model, cfg.train.update_layers)
    optim = torch.optim.AdamW(
        trainable, lr=cfg.tar.outer_lr, weight_decay=cfg.train.weight_decay
    )

    stats: dict = {"outer_loss": [], "L_forget_at_init": [], "L_forget_at_K": [], "L_retain": []}

    for epoch in range(cfg.tar.epochs):
        ret_iter = _cycle(retain_facts, cfg.train.batch_size)
        forget_batches = list(iter_minibatches(forget_facts, cfg.train.batch_size))
        pbar = tqdm(forget_batches, desc=f"TAR ep{epoch+1}/{cfg.tar.epochs}")

        for fbatch in pbar:
            rbatch = next(ret_iter)
            bf = tokenize_facts(fbatch, tokenizer, cfg.train.max_seq_len, device)
            br = tokenize_facts(rbatch, tokenizer, cfg.train.max_seq_len, device)

            # ---- 1) snapshot ----
            snapshot = [p.detach().clone() for p in trainable]

            # ---- 2) record forget loss at theta_init for diagnostics ----
            with torch.no_grad():
                lf_init = float(_forget_nll(model, bf).detach())

            # ---- 3) inner loop: simulate adversary SFT (manual SGD) ----
            for _ in range(K):
                for p in trainable:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
                inner_loss = _forget_nll(model, bf)
                inner_loss.backward()
                with torch.no_grad():
                    for p in trainable:
                        if p.grad is not None:
                            p.data.add_(p.grad, alpha=-inner_lr)

            # ---- 4) compute "stay-broken" grad at theta_K ----
            for p in trainable:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            adv_loss = _forget_nll(model, bf)        # at theta_K
            adv_loss.backward()
            # Adversary's gradient direction reduces L_forget. Defender wants the
            # OPPOSITE: increase L_forget at theta_K. So we add -lam * adv_grad
            # to the outer gradient (which is what optim.step() will apply with
            # AdamW's default sign convention of subtracting the gradient).
            tamper_grads = [
                (-lam * p.grad).detach().clone() if p.grad is not None
                else torch.zeros_like(p) for p in trainable
            ]
            lf_K = float(adv_loss.detach())

            # ---- 5) restore theta_init ----
            with torch.no_grad():
                for p, snap in zip(trainable, snapshot):
                    p.data.copy_(snap)

            # ---- 6) compute retain anchor grad at theta_init ----
            for p in trainable:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            retain_loss = _retain_anchor_loss(model, frozen, br, cfg.rmu.layer_idx)
            retain_loss.backward()
            retain_grads = [
                p.grad.detach().clone() if p.grad is not None
                else torch.zeros_like(p) for p in trainable
            ]
            lret = float(retain_loss.detach())

            # ---- 7) combined outer step ----
            optim.zero_grad()
            with torch.no_grad():
                for p, gr, gt in zip(trainable, retain_grads, tamper_grads):
                    p.grad = gr + gt
            torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
            optim.step()

            outer_total = lret + (-lam * lf_K)
            stats["outer_loss"].append(outer_total)
            stats["L_forget_at_init"].append(lf_init)
            stats["L_forget_at_K"].append(lf_K)
            stats["L_retain"].append(lret)
            pbar.set_postfix(
                lf_init=f"{lf_init:.3f}",
                lf_K=f"{lf_K:.3f}",
                lret=f"{lret:.4f}",
            )

    model.eval()
    del frozen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def _cycle(facts: List[Fact], batch_size: int):
    while True:
        for b in iter_minibatches(facts, batch_size):
            yield b


def save(model, cfg: Config, name: str = "tar"):
    out = cfg.abspath(cfg.paths.checkpoints) / name
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    return out
