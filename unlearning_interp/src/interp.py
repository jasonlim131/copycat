"""Interpretability analyses comparing base, RMU-unlearned, and NPO-unlearned models.

All four analyses operate on the *same* fact JSONL splits (forget vs retain) and
produce per-layer numbers ready for plotting.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data_utils import Fact, first_answer_token_id, tokenize_facts
from .model_utils import capture_all_hiddens


def _last_prompt_position(input_ids: torch.Tensor, attn_mask: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
    """Index of the LAST prompt token (the position whose next-token = first answer token)."""
    # answer_mask is 1 on answer tokens. The first answer token's index minus 1.
    # If a row has no answer (paraphrase eval), fall back to last attended position.
    has_ans = answer_mask.sum(dim=1) > 0
    first_ans_idx = answer_mask.argmax(dim=1)              # [B]
    last_attended = attn_mask.sum(dim=1) - 1               # [B]
    return torch.where(has_ans, first_ans_idx - 1, last_attended)


@torch.no_grad()
def logit_lens_acc(
    model,
    tokenizer,
    facts: List[Fact],
    layers: List[int],
    device: str,
    max_seq_len: int = 128,
) -> Dict[int, float]:
    """For each requested layer, the fraction of facts where projecting the hidden
    state at the last-prompt-position through the unembedding gives the gold first
    answer token."""
    model.eval()
    batch = tokenize_facts(facts, tokenizer, max_seq_len, device)
    hiddens = capture_all_hiddens(model, batch.input_ids, batch.attention_mask)
    # hiddens: tuple of length num_layers+1; hiddens[0] is embeddings, hiddens[i] is post-layer-i.
    pos = _last_prompt_position(batch.input_ids, batch.attention_mask, batch.answer_mask)
    pos = pos.to(device)
    arange = torch.arange(batch.input_ids.size(0), device=device)

    gold = torch.tensor(
        [first_answer_token_id(f, tokenizer) for f in facts],
        device=device, dtype=torch.long,
    )
    # Pythia ties output embeddings: model.embed_out is the unembedding head.
    W_U = model.get_output_embeddings().weight  # [V, H]

    out: Dict[int, float] = {}
    for L in layers:
        h = hiddens[L][arange, pos]                # [B, H]
        # Apply final layer norm to give the unembedding well-conditioned input.
        try:
            h_normed = model.gpt_neox.final_layer_norm(h)
        except AttributeError:
            h_normed = h
        logits = h_normed @ W_U.T                  # [B, V]
        pred = logits.argmax(dim=-1)
        out[L] = float((pred == gold).float().mean().cpu())
    return out


@torch.no_grad()
def hidden_cosine(
    base_model,
    other_model,
    tokenizer,
    facts: List[Fact],
    layers: List[int],
    device: str,
    max_seq_len: int = 128,
) -> Dict[int, float]:
    """Mean cosine similarity between base and other hidden states, averaged over
    answer-token positions of forget prompts."""
    batch = tokenize_facts(facts, tokenizer, max_seq_len, device)
    h_base = capture_all_hiddens(base_model, batch.input_ids, batch.attention_mask)
    h_oth = capture_all_hiddens(other_model, batch.input_ids, batch.attention_mask)
    mask = batch.answer_mask.bool()                       # [B, T]

    out: Dict[int, float] = {}
    for L in layers:
        a = h_base[L][mask]                                # [N_tok, H]
        b = h_oth[L][mask]
        if a.numel() == 0:
            out[L] = float("nan")
            continue
        cos = F.cosine_similarity(a, b, dim=-1).mean().float().cpu()
        out[L] = float(cos)
    return out


@torch.no_grad()
def topk_logit_drop(
    base_model,
    other_model,
    tokenizer,
    facts: List[Fact],
    device: str,
    max_seq_len: int = 128,
) -> List[Dict]:
    """Per-fact: base and unlearned final-layer logit on the gold first-answer token,
    plus the difference."""
    batch = tokenize_facts(facts, tokenizer, max_seq_len, device)
    pos = _last_prompt_position(batch.input_ids, batch.attention_mask, batch.answer_mask).to(device)
    arange = torch.arange(batch.input_ids.size(0), device=device)
    gold = torch.tensor(
        [first_answer_token_id(f, tokenizer) for f in facts],
        device=device, dtype=torch.long,
    )

    def _gold_logit(model):
        out = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
        )
        logits = out.logits[arange, pos]                   # [B, V]
        return logits.gather(-1, gold.unsqueeze(-1)).squeeze(-1)

    base_logit = _gold_logit(base_model).cpu().tolist()
    oth_logit = _gold_logit(other_model).cpu().tolist()

    rows = []
    for f, lb, lo in zip(facts, base_logit, oth_logit):
        rows.append({"id": f.id, "subject": getattr(f, "id", ""),
                     "base_logit": lb, "unlearned_logit": lo, "drop": lb - lo})
    return rows


@torch.no_grad()
def linear_probe_layerwise(
    base_model,
    tokenizer,
    forget_facts: List[Fact],
    retain_facts: List[Fact],
    layers: List[int],
    device: str,
    test_frac: float = 0.2,
    max_seq_len: int = 128,
    seed: int = 42,
) -> Dict[int, float]:
    """Train a logistic-regression probe at each layer to predict 'is forget fact'
    from mean-pooled hidden state. Returns per-layer test accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    facts = forget_facts + retain_facts
    labels = np.array([1] * len(forget_facts) + [0] * len(retain_facts))
    batch = tokenize_facts(facts, tokenizer, max_seq_len, device)
    hiddens = capture_all_hiddens(base_model, batch.input_ids, batch.attention_mask)
    mask = batch.attention_mask.unsqueeze(-1)              # [B, T, 1]

    out: Dict[int, float] = {}
    for L in layers:
        h = hiddens[L]                                     # [B, T, H]
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        X = pooled.float().cpu().numpy()
        Xtr, Xte, ytr, yte = train_test_split(
            X, labels, test_size=test_frac, random_state=seed, stratify=labels
        )
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        out[L] = float(clf.score(Xte, yte))
    return out
