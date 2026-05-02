"""Evaluation: exact-match accuracy, wikitext-2 perplexity, trivia probe."""
from __future__ import annotations

import math
from typing import Dict, List

import torch
from tqdm import tqdm

from .data_utils import Fact


@torch.no_grad()
def exact_match_acc(
    model,
    tokenizer,
    facts: List[Fact],
    device: str,
    max_new_tokens_pad: int = 2,
    use_paraphrases: bool = False,
) -> Dict:
    """Greedy-decode and check whether the model's continuation contains the gold answer.

    `paraphrases=True` runs each paraphrase as a separate prompt and averages.
    """
    model.eval()
    n_total, n_hit = 0, 0
    per_fact = []
    for fact in tqdm(facts, desc="exact-match"):
        prompts = [fact.prompt] + (fact.paraphrases if use_paraphrases else [])
        gold = fact.answer.strip().lower()
        ans_token_len = len(tokenizer(fact.answer, add_special_tokens=False)["input_ids"])
        max_new = ans_token_len + max_new_tokens_pad
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            cont = tokenizer.decode(
                out[0, ids.input_ids.shape[1]:], skip_special_tokens=True
            )
            hit = gold in cont.lower()
            n_hit += int(hit)
            n_total += 1
            per_fact.append({"id": fact.id, "prompt": prompt, "cont": cont, "hit": hit})
    return {
        "accuracy": n_hit / max(n_total, 1),
        "n": n_total,
        "per_fact": per_fact,
    }


@torch.no_grad()
def wikitext_ppl(model, tokenizer, device: str, split: str, stride: int = 512) -> float:
    """Sliding-window perplexity on a wikitext-2 slice. Lower is better.

    Standard HF recipe: each window is a chunk of `max_len` tokens; only the last
    `stride` tokens (the ones we haven't already scored in a previous window)
    contribute to the NLL. Their next-token targets must be inside the window.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    max_len = getattr(model.config, "max_position_embeddings", 2048)
    seq_len = input_ids.size(1)

    total_nll, total_tokens = 0.0, 0
    prev_end = 0
    for begin in tqdm(range(0, seq_len, stride), desc="wikitext-ppl"):
        end = min(begin + max_len, seq_len)
        trg_len = end - prev_end                       # how many *new* tokens to score
        ids = input_ids[:, begin:end]
        targets = ids.clone()
        targets[:, :-trg_len] = -100                   # ignore already-scored tokens
        out = model(input_ids=ids, labels=targets, use_cache=False)
        n_tok = int((targets != -100).sum().item())
        total_nll += out.loss.float().item() * n_tok
        total_tokens += n_tok
        prev_end = end
        if end == seq_len:
            break
    return math.exp(total_nll / max(total_tokens, 1))


@torch.no_grad()
def trivia_acc(model, tokenizer, facts: List[Fact], device: str, max_new_tokens_pad: int = 2) -> Dict:
    return exact_match_acc(model, tokenizer, facts, device, max_new_tokens_pad, use_paraphrases=False)
