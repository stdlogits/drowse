"""Shared tiny causal LM + tokenizer for the Jacobian-lens CPU tests.

The toy is a genuine ``transformers`` llama (``TOY_D`` wide, ``TOY_VOCAB``
tokens, three heads over one KV head, eager attention, fp32) with every
weight re-drawn from a seeded generator, so its blocks are what the lens
estimator's :class:`~drowse.model.blocks.BlockSpec` describes and a lens
fit runs on it unmodified.  :class:`CharTokenizer` maps ``a``..``m`` onto
its vocabulary.
"""

from __future__ import annotations

from typing import Any

import torch
import transformers

TOY_D = 6
TOY_VOCAB = 13

__all__ = ["TOY_D", "TOY_VOCAB", "CharTokenizer", "frozen_toy"]


class CharTokenizer:
    """Char-code tokenizer whose encode/decode round-trip for 'a'..'m'."""

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        del return_tensors
        return {"input_ids": torch.tensor([self.encode(text)], dtype=torch.long)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [(ord(c) - 97) % TOY_VOCAB for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + int(i) % 26) for i in ids)


def frozen_toy(n_layers: int = 3, *, seed: int = 99) -> Any:
    """The toy lens model, frozen and in eval mode as ``load_model`` leaves one."""
    config = transformers.LlamaConfig(
        vocab_size=TOY_VOCAB, hidden_size=TOY_D, intermediate_size=12,
        num_hidden_layers=n_layers, num_attention_heads=3, num_key_value_heads=1,
        head_dim=2, max_position_embeddings=256, tie_word_embeddings=False,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, attention_dropout=0.0,
    )
    model = transformers.LlamaForCausalLM(config)
    model.config._attn_implementation = "eager"
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            noise = torch.randn(tensor.shape, generator=gen)
            if "norm" in name:
                tensor.copy_(1.0 + 0.2 * noise)
            elif "embed_tokens" in name or "lm_head" in name:
                tensor.copy_(noise)
            else:
                tensor.copy_(0.5 * noise)
    return model.float().eval().requires_grad_(False)
