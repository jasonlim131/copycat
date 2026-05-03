"""Frozen 'knowledge-meter' linear probes.

Probes are trained ONCE on the BASE model's hidden states (binary task: 'is this
a forget-set fact?'). Their weights are then frozen and applied to whichever
model is being instrumented during the relearning trajectory.

This is the central interpretability primitive for Phase 2.B: at each SFT-recovery
step, we ask "does the recovering model's residual stream still encode the
forget-set membership signal as the *base* model originally encoded it?"

If probe accuracy on the post-edit model is high while behavioral EM is near 0,
the representation is suppressed-not-erased (H1). If probe accuracy stays low
through recovery while EM climbs, recovery is rebuilding via a different pathway
(H2). If probe accuracy rises *before* EM during recovery, the representation
re-emerges first then behavior follows (H3).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .data_utils import Fact, tokenize_facts
from .model_utils import capture_all_hiddens


@dataclass
class FrozenProbe:
    """A scikit-learn-style frozen linear probe stored as torch tensors."""
    layer: int
    coef: torch.Tensor      # [H]
    intercept: torch.Tensor # scalar
    train_acc: float
    test_acc: float
    test_indices: List[int]


@torch.no_grad()
def _pooled_hiddens(
    model, tokenizer, facts: List[Fact], layers: List[int], device: str, max_seq_len: int,
) -> Dict[int, torch.Tensor]:
    """Mean-pooled hidden state per (fact, layer). Returns {layer: [N, H]}."""
    batch = tokenize_facts(facts, tokenizer, max_seq_len, device)
    hs = capture_all_hiddens(model, batch.input_ids, batch.attention_mask)
    mask = batch.attention_mask.unsqueeze(-1).to(hs[0].dtype)
    out: Dict[int, torch.Tensor] = {}
    for L in layers:
        h = hs[L]                                              # [B, T, H]
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        out[L] = pooled.float().cpu()
    return out


def train_probes(
    base_model,
    tokenizer,
    forget_facts: List[Fact],
    retain_facts: List[Fact],
    layers: List[int],
    device: str,
    max_seq_len: int,
    train_frac: float,
    C: float,
    seed: int,
) -> Dict[int, FrozenProbe]:
    """Train one binary probe per layer on the BASE model. Returns {layer: FrozenProbe}.

    The same train/test split is used at every layer so that test-set probe
    accuracy is comparable across layers (and later, across models).
    """
    from sklearn.linear_model import LogisticRegression

    facts = forget_facts + retain_facts
    labels = np.array([1] * len(forget_facts) + [0] * len(retain_facts), dtype=np.int64)
    pooled = _pooled_hiddens(base_model, tokenizer, facts, layers, device, max_seq_len)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(facts))
    n_train = int(round(len(facts) * train_frac))
    train_idx = perm[:n_train].tolist()
    test_idx = perm[n_train:].tolist()

    probes: Dict[int, FrozenProbe] = {}
    for L in layers:
        X = pooled[L].numpy()
        Xtr, ytr = X[train_idx], labels[train_idx]
        Xte, yte = X[test_idx], labels[test_idx]
        clf = LogisticRegression(max_iter=2000, C=C).fit(Xtr, ytr)
        coef = torch.tensor(clf.coef_[0], dtype=torch.float32)         # [H]
        intercept = torch.tensor(float(clf.intercept_[0]), dtype=torch.float32)
        train_acc = float(clf.score(Xtr, ytr))
        test_acc = float(clf.score(Xte, yte))
        probes[L] = FrozenProbe(
            layer=L, coef=coef, intercept=intercept,
            train_acc=train_acc, test_acc=test_acc, test_indices=test_idx,
        )
    return probes


def save_probes(probes: Dict[int, FrozenProbe], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probes": {
            int(L): {
                "coef": p.coef,
                "intercept": p.intercept,
                "train_acc": p.train_acc,
                "test_acc": p.test_acc,
                "test_indices": p.test_indices,
            }
            for L, p in probes.items()
        },
    }
    torch.save(payload, path)
    return path


def load_probes(path: Path) -> Dict[int, FrozenProbe]:
    payload = torch.load(path, map_location="cpu")
    out: Dict[int, FrozenProbe] = {}
    for L, d in payload["probes"].items():
        out[int(L)] = FrozenProbe(
            layer=int(L),
            coef=d["coef"], intercept=d["intercept"],
            train_acc=float(d["train_acc"]), test_acc=float(d["test_acc"]),
            test_indices=list(d["test_indices"]),
        )
    return out


@torch.no_grad()
def eval_probes_on_model(
    model,
    tokenizer,
    forget_facts: List[Fact],
    retain_facts: List[Fact],
    probes: Dict[int, FrozenProbe],
    device: str,
    max_seq_len: int,
    test_only: bool = True,
) -> Dict[int, float]:
    """Apply the frozen probes (trained on the base model) to `model`'s hidden states.

    Returns {layer: accuracy} where accuracy is computed only on the held-out
    test indices when `test_only=True` (the default — this is the apples-to-apples
    comparison; using train indices would inflate scores).
    """
    layers = sorted(probes.keys())
    facts = forget_facts + retain_facts
    labels = np.array([1] * len(forget_facts) + [0] * len(retain_facts), dtype=np.int64)
    pooled = _pooled_hiddens(model, tokenizer, facts, layers, device, max_seq_len)

    out: Dict[int, float] = {}
    for L in layers:
        probe = probes[L]
        X = pooled[L]                                       # [N, H]
        coef = probe.coef.to(X.dtype)
        logits = X @ coef + probe.intercept                 # [N]
        preds = (logits > 0).numpy().astype(np.int64)
        idx = probe.test_indices if test_only else list(range(len(facts)))
        out[L] = float((preds[idx] == labels[idx]).mean())
    return out
