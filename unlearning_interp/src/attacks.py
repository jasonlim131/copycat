"""Relearning-attack harness (Phase 2.A).

Currently implements the SFT-recovery attack: a small-data adversary that
fine-tunes the post-edit model on the original (prompt, answer) pairs of the
forget set. This is the standard relearning attack in Lynch et al. 2024
(arxiv:2402.16835) and Hu et al. 2024 (arxiv:2406.13356).

The function exposes per-step callbacks so the trajectory orchestrator can
snapshot mechanistic state (logit-lens, probes, hidden-cosine) at the precise
step indices we care about. Step 0 fires *before* any optimizer step — that's
the unlearned model itself.
"""
from __future__ import annotations

import random
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

from .cfg import Config
from .data_utils import Fact, tokenize_facts
from .model_utils import freeze_all, unfreeze_mlp_down


SnapshotFn = Callable[[int], None]
"""Callback signature: snapshot_fn(step) -> None.

The callback closes over the model and writes its diagnostics to disk or to
an in-memory dict. It receives only the step index; the model is shared by
reference with the caller's scope.
"""


def _shuffled_minibatches(facts: List[Fact], batch_size: int, seed: int):
    """Infinite generator of minibatches; reshuffles each pass to keep the
    adversary's gradient-noise statistics realistic."""
    rng = random.Random(seed)
    pool = list(facts)
    while True:
        rng.shuffle(pool)
        for i in range(0, len(pool), batch_size):
            yield pool[i : i + batch_size]


def sft_recovery_attack(
    cfg: Config,
    model,
    tokenizer,
    recovery_facts: List[Fact],
    snapshot_fn: Optional[SnapshotFn] = None,
) -> dict:
    """Run the SFT-recovery attack in place on `model`.

    Args:
        cfg.attacks: AttacksCfg block — defines max steps, lr, batch size,
            snapshot schedule, and which layers the adversary fine-tunes.
        recovery_facts: the (prompt, answer) pool the adversary trains on
            (typically the forget set itself).
        snapshot_fn: optional callback fired at each `cfg.attacks.snapshot_steps`
            index. Step 0 fires before any optimizer step.

    Returns a small training-stats dict (for diagnostics).
    """
    assert cfg.attacks is not None, "config.yaml is missing the `attacks:` block"
    device = cfg.device
    schedule = sorted(set(cfg.attacks.snapshot_steps))
    schedule_set = set(schedule)
    max_step = max(schedule + [cfg.attacks.sft_max_steps])

    # Step 0 snapshot: model as-is, before any training.
    if snapshot_fn is not None and 0 in schedule_set:
        model.eval()
        snapshot_fn(0)

    freeze_all(model)
    trainable = unfreeze_mlp_down(model, cfg.attacks.train_layers)
    optim = torch.optim.AdamW(
        trainable, lr=cfg.attacks.sft_lr, weight_decay=cfg.train.weight_decay
    )

    batches = _shuffled_minibatches(recovery_facts, cfg.attacks.sft_batch_size, cfg.seed)
    stats = {"loss": [], "snapshots_taken": []}

    model.train()
    step = 0
    pbar = tqdm(total=max_step, desc="SFT-recovery")
    while step < max_step:
        batch_facts = next(batches)
        batch = tokenize_facts(batch_facts, tokenizer, cfg.train.max_seq_len, device)
        out = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            use_cache=False,
        )
        loss = out.loss
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
        optim.step()
        step += 1
        stats["loss"].append(float(loss.detach()))
        pbar.update(1)
        pbar.set_postfix(loss=f"{float(loss):.3f}")

        if snapshot_fn is not None and step in schedule_set:
            model.eval()
            snapshot_fn(step)
            stats["snapshots_taken"].append(step)
            model.train()

    pbar.close()
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats
