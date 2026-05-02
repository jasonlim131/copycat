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


@dataclass
class NPOCfg:
    epochs: int
    beta: float
    retain_kl_coeff: float


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
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
