"""JSONL loaders and tokenization helpers.

The forget/retain JSONL rows are tokenized into a (prompt + answer) sequence and we
record which positions belong to the answer span. RMU and NPO both need this mask:
RMU pushes hidden states *at the answer positions* toward the steering vector, and
NPO computes the policy/reference log-prob over the answer span only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import torch


@dataclass
class Fact:
    id: str
    prompt: str
    answer: str            # ALWAYS includes a leading space
    paraphrases: List[str]


@dataclass
class Batch:
    input_ids: torch.Tensor       # [B, T]
    attention_mask: torch.Tensor  # [B, T]
    answer_mask: torch.Tensor     # [B, T] — 1 on answer tokens, 0 elsewhere
    labels: torch.Tensor          # [B, T] — answer tokens, -100 elsewhere


def read_jsonl(path: str | Path) -> List[Fact]:
    facts: List[Fact] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            facts.append(
                Fact(
                    id=str(r.get("id", "")),
                    prompt=r["prompt"],
                    answer=r["answer"],
                    paraphrases=list(r.get("paraphrases", [])),
                )
            )
    return facts


def tokenize_facts(
    facts: List[Fact],
    tokenizer,
    max_seq_len: int,
    device: str,
) -> Batch:
    """Tokenize prompt+answer; create attention/answer/labels masks aligned to tokens."""
    input_ids_list, attn_list, ans_mask_list, label_list = [], [], [], []
    for fact in facts:
        prompt_ids = tokenizer(fact.prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(fact.answer, add_special_tokens=False)["input_ids"]
        ids = prompt_ids + answer_ids
        ids = ids[:max_seq_len]
        ans_start = min(len(prompt_ids), len(ids))
        ans_mask = [0] * len(ids)
        labels = [-100] * len(ids)
        for j in range(ans_start, len(ids)):
            ans_mask[j] = 1
            labels[j] = ids[j]
        attn = [1] * len(ids)
        # right-pad
        pad = max_seq_len - len(ids)
        if pad > 0:
            ids += [tokenizer.pad_token_id] * pad
            attn += [0] * pad
            ans_mask += [0] * pad
            labels += [-100] * pad
        input_ids_list.append(ids)
        attn_list.append(attn)
        ans_mask_list.append(ans_mask)
        label_list.append(labels)
    return Batch(
        input_ids=torch.tensor(input_ids_list, dtype=torch.long, device=device),
        attention_mask=torch.tensor(attn_list, dtype=torch.long, device=device),
        answer_mask=torch.tensor(ans_mask_list, dtype=torch.long, device=device),
        labels=torch.tensor(label_list, dtype=torch.long, device=device),
    )


def iter_minibatches(facts: List[Fact], batch_size: int) -> Iterator[List[Fact]]:
    for i in range(0, len(facts), batch_size):
        yield facts[i : i + batch_size]


def first_answer_token_id(fact: Fact, tokenizer) -> int:
    """The token id the model is supposed to emit right after the prompt."""
    ans_ids = tokenizer(fact.answer, add_special_tokens=False)["input_ids"]
    assert len(ans_ids) >= 1, f"answer tokenizes to 0 tokens: {fact.answer!r}"
    return ans_ids[0]
