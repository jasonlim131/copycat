"""Model/tokenizer loading and parameter-scope helpers for Pythia (GPT-NeoX)."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_tokenizer(model_name: str, dtype: torch.dtype, device: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    return model, tok


def assert_pythia_410m(model) -> None:
    cfg = model.config
    assert cfg.num_hidden_layers == 24, f"expected 24 layers, got {cfg.num_hidden_layers}"
    assert cfg.hidden_size == 1024, f"expected hidden=1024, got {cfg.hidden_size}"


def get_layer_module(model, layer_idx: int) -> nn.Module:
    """Returns the i-th transformer block of a GPT-NeoX model."""
    return model.gpt_neox.layers[layer_idx]


def freeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_mlp_down(model, layer_indices: Iterable[int]) -> List[nn.Parameter]:
    """Unfreeze only `mlp.dense_4h_to_h` of the listed layers; return the trainable params."""
    trainable: List[nn.Parameter] = []
    for i in layer_indices:
        block = get_layer_module(model, i)
        for p in block.mlp.dense_4h_to_h.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def make_frozen_reference(model_name: str, dtype: torch.dtype, device: str):
    """Build a separate, frozen reference copy of the base model on the same device."""
    ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


class LayerCapture:
    """Forward hook that captures the (first element of) a layer's output."""

    def __init__(self, module: nn.Module):
        self.module = module
        self.value: torch.Tensor | None = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, _mod, _inp, out):
        # GPT-NeoX block returns (hidden_states, *aux). Always take [0] when it's a tuple.
        self.value = out[0] if isinstance(out, tuple) else out

    def close(self):
        self._handle.remove()
        self.value = None


def capture_hidden(model, input_ids, attention_mask, layer_idx: int) -> torch.Tensor:
    """Run a forward pass and return the chosen layer's hidden states [B, T, H]."""
    cap = LayerCapture(get_layer_module(model, layer_idx))
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        assert cap.value is not None
        return cap.value
    finally:
        cap.close()


def capture_all_hiddens(model, input_ids, attention_mask) -> Tuple[torch.Tensor, ...]:
    """Returns the tuple of hidden states from all layers (output_hidden_states=True)."""
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return out.hidden_states  # tuple of length num_layers+1
