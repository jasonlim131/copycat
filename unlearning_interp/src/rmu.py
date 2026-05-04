"""Representation Misdirection for Unlearning (RMU).

Reference: Li et al., "The WMDP Benchmark", arxiv:2403.03218.

Idea: hook a chosen middle layer; on forget-set tokens push the hidden state toward
a fixed random direction (steering vector u * c); on retain-set tokens anchor the
hidden state to a frozen reference. Only update the MLP down-proj of a small band
of layers around the hooked layer.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .cfg import Config
from .data_utils import Batch, Fact, iter_minibatches, tokenize_facts
from .model_utils import (
    capture_hidden,
    freeze_all,
    make_frozen_reference,
    unfreeze_mlp_down,
)


def _get_or_make_steering_vector(cfg: Config, hidden_size: int, device: str) -> torch.Tensor:
    """Sample once, cache to disk, and reload on subsequent runs (deterministic via seed)."""
    path = cfg.abspath(cfg.paths.steering_vector)
    if path.exists():
        u = torch.load(path, map_location=device)
    else:
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        u = torch.randn(hidden_size, generator=g)
        u = u / u.norm()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(u, path)
    return u.to(device)


def rmu_loss(
    model,
    frozen,
    batch_forget: Batch,
    batch_retain: Batch,
    layer_idx: int,
    u: torch.Tensor,
    c: float,
    alpha: float,
    retain_coeff: float = 1.0,
) -> torch.Tensor:
    """Per-token MSE: forget tokens pulled toward c*u; retain tokens pinned to frozen."""
    h_f = capture_hidden(
        model, batch_forget.input_ids, batch_forget.attention_mask, layer_idx
    )
    h_r = capture_hidden(
        model, batch_retain.input_ids, batch_retain.attention_mask, layer_idx
    )
    with torch.no_grad():
        h_r_frozen = capture_hidden(
            frozen, batch_retain.input_ids, batch_retain.attention_mask, layer_idx
        )

    target = (c * u).to(h_f.dtype).expand_as(h_f)
    mask_f = batch_forget.answer_mask.unsqueeze(-1).to(h_f.dtype)
    mask_r = batch_retain.attention_mask.unsqueeze(-1).to(h_r.dtype)

    L_forget = ((h_f - target) ** 2 * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    L_retain = ((h_r - h_r_frozen) ** 2 * mask_r).sum() / mask_r.sum().clamp_min(1.0)
    return alpha * L_forget + retain_coeff * L_retain


def train_rmu(
    cfg: Config,
    model,
    tokenizer,
    forget_facts: List[Fact],
    retain_facts: List[Fact],
) -> dict:
    """Run RMU unlearning in place; return a small training-stats dict."""
    device = cfg.device
    hidden_size = model.config.hidden_size

    frozen = make_frozen_reference(cfg.model_name, cfg.torch_dtype(), device)

    freeze_all(model)
    trainable = unfreeze_mlp_down(model, cfg.train.update_layers)
    optim = torch.optim.AdamW(
        trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    u = _get_or_make_steering_vector(cfg, hidden_size, device)

    model.train()
    stats = {"loss": [], "L_forget": [], "L_retain": []}
    for epoch in range(cfg.rmu.epochs):
        # iterate over the *smaller* of the two sets; cycle the other.
        ret_iter = iter(_cycle_minibatches(retain_facts, cfg.train.batch_size))
        forget_batches = list(iter_minibatches(forget_facts, cfg.train.batch_size))
        pbar = tqdm(forget_batches, desc=f"RMU ep{epoch+1}/{cfg.rmu.epochs}")

        optim.zero_grad()
        for step, fbatch in enumerate(pbar):
            rbatch = next(ret_iter)
            bf = tokenize_facts(fbatch, tokenizer, cfg.train.max_seq_len, device)
            br = tokenize_facts(rbatch, tokenizer, cfg.train.max_seq_len, device)
            loss = rmu_loss(
                model, frozen, bf, br,
                layer_idx=cfg.rmu.layer_idx,
                u=u,
                c=cfg.rmu.c,
                alpha=cfg.rmu.alpha,
                retain_coeff=cfg.rmu.retain_coeff,
            )
            (loss / cfg.train.grad_accum).backward()
            if (step + 1) % cfg.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
                optim.step()
                optim.zero_grad()
            stats["loss"].append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        # flush
        if (step + 1) % cfg.train.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
            optim.step()
            optim.zero_grad()

    model.eval()
    del frozen
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return stats


def _cycle_minibatches(facts: List[Fact], batch_size: int):
    while True:
        for b in iter_minibatches(facts, batch_size):
            yield b


def save(model, cfg: Config, name: str) -> Path:
    out = cfg.abspath(cfg.paths.checkpoints) / name
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    return out
