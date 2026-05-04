"""Config loader and global RNG seeding."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml


@dataclass
class TrainCfg:
    batch_size: int
    grad_accum: int
    max_seq_len: int
    lr: float
    weight_decay: float
    grad_clip: float
    update_layers: List[int]


@dataclass
class RMUCfg:
    epochs: int
    layer_idx: int
    c: float
    alpha: float
    retain_coeff: float = 1.0


@dataclass
class NPOCfg:
    epochs: int
    beta: float
    retain_kl_coeff: float


@dataclass
class TARCfg:
    """TAR-1 (first-order tamper-resistant) training, applied on top of RMU.

    Reference: Tamirisa et al. 2024, "Tamper-Resistant Safeguards for Open-Weight LLMs"
    (arxiv:2408.00761). Bilevel: outer minimizes retain anchor + lambda * (-L_forget(theta_K))
    where theta_K is the result of K SFT steps simulating an adversary.
    """
    epochs: int
    inner_steps: int            # K — adversary's simulated SFT length
    inner_lr: float             # adversary's SFT learning rate
    outer_lr: float             # defender's outer-loop lr
    lambda_resist: float        # weight on the tamper-resistance term


@dataclass
class AttacksCfg:
    """Relearning-attack harness (Phase 2.A)."""
    sft_max_steps: int          # total SFT steps the recovery loop runs
    sft_lr: float
    sft_batch_size: int
    snapshot_steps: List[int]   # step indices at which to dump mechanistic state
    train_layers: List[int]     # which layers' MLP-down to fine-tune (mirrors defender scope)


@dataclass
class TrajectoryCfg:
    """Per-step mechanistic instrumentation captured during a relearning attack."""
    layers: List[int]           # logit-lens / cosine / probe layers
    probe_train_frac: float     # split for the frozen 'is forget' probe
    probe_C: float              # logistic-regression regularization
    eval_subset: int            # cap on facts evaluated per snapshot (speed)


@dataclass
class EvalCfg:
    topk: int
    wikitext_split: str
    wikitext_stride: int
    max_new_tokens_pad: int


@dataclass
class InterpCfg:
    layers: List[int]
    probe_test_frac: float


@dataclass
class Paths:
    forget: str
    retain: str
    trivia: str
    checkpoints: str
    metrics: str
    figures: str
    steering_vector: str
    probes: str = "results/checkpoints/probes.pt"
    trajectories: str = "results/metrics/trajectories"


@dataclass
class Config:
    seed: int
    model_name: str
    dtype: str
    device: str
    paths: Paths
    train: TrainCfg
    rmu: RMUCfg
    npo: NPOCfg
    eval: EvalCfg
    interp: InterpCfg
    tar: TARCfg | None = None
    attacks: AttacksCfg | None = None
    trajectory: TrajectoryCfg | None = None
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    def abspath(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def torch_dtype(self):
        return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[self.dtype]


def load_config(yaml_path: str | os.PathLike) -> Config:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        seed=int(raw["seed"]),
        model_name=str(raw["model_name"]),
        dtype=str(raw["dtype"]),
        device=str(raw["device"]),
        paths=Paths(**raw["paths"]),
        train=TrainCfg(**raw["train"]),
        rmu=RMUCfg(**raw["rmu"]),
        npo=NPOCfg(**raw["npo"]),
        eval=EvalCfg(**raw["eval"]),
        interp=InterpCfg(**raw["interp"]),
        tar=TARCfg(**raw["tar"]) if "tar" in raw else None,
        attacks=AttacksCfg(**raw["attacks"]) if "attacks" in raw else None,
        trajectory=TrajectoryCfg(**raw["trajectory"]) if "trajectory" in raw else None,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
