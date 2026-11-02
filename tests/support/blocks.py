"""Tiny dense decoders per family, and a whole forward from their specs alone.

:func:`tiny_dense_model` builds a genuine ``transformers`` architecture from a
small config (3 blocks × 32 wide, eager attention, float64) and re-draws every
weight, norm and per-layer scalar at random, so a description that drops a
gain, an offset or a scalar changes the numbers.  Sliding windows are short
enough to bind on an 8-token input.

:func:`default_init_model` builds the same architectures at their own
default initialization in a low precision (bf16), the case a verification
must not mistake rounding for a term in.

:func:`spec_forward` runs the embedding, every block's
:func:`~drowse.model.blocks.block_forward` and the logits path from a
:class:`~drowse.model.blocks.ModelSpec` alone.  Matching the model's own
forward is the proof that the description holds every term the model
computes; it is also the oracle a transport layer map can be checked against.
"""

from __future__ import annotations

from typing import Any

import torch
import transformers

from drowse.model.blocks import ModelSpec, block_forward

__all__ = ["DENSE_FAMILIES", "default_init_model", "spec_forward", "tiny_dense_model"]

_TINY: dict[str, Any] = dict(
    vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    attention_dropout=0.0,
)

#: Per-family config overrides: the dense families transport covers.  Each
#: exercises its family's terms: q/k/v biases (llama with ``attention_bias``,
#: qwen2), a config-level window (mistral), per-layer windows (qwen2, gemma),
#: q/k norms (qwen3, gemma 3/4), sandwich norms and ``1 + w`` weights
#: (gemma 2/3), attention and final softcaps (gemma2, gemma4 here), the value
#: norm, keys-as-values and the per-layer scalar (gemma4).
DENSE_FAMILIES: dict[str, dict[str, Any]] = {
    "llama": {"attention_bias": True},
    "mistral": {"sliding_window": 3},
    "qwen2": {"use_sliding_window": True, "sliding_window": 3, "max_window_layers": 1},
    "qwen3": {},
    "gemma2": {"query_pre_attn_scalar": 8, "sliding_window": 3},
    "gemma3_text": {
        "query_pre_attn_scalar": 8, "sliding_window": 3,
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
    },
    "gemma4_text": {
        "sliding_window": 3, "hidden_size_per_layer_input": 0,
        "attention_k_eq_v": True, "global_head_dim": 16, "num_global_key_value_heads": 1,
        "layer_types": ["sliding_attention", "full_attention", "full_attention"],
        "final_logit_softcapping": 20.0,
    },
    "gemma4_unified_text": {
        "sliding_window": 3,
        "layer_types": ["sliding_attention", "full_attention", "full_attention"],
    },
}


def tiny_dense_model(model_type: str, *, seed: int = 0, **overrides: Any) -> Any:
    """A float64 tiny ``model_type`` with every weight re-drawn at random."""
    settings = {**_TINY, **DENSE_FAMILIES.get(model_type, {}), **overrides}
    config = transformers.AutoConfig.for_model(model_type, **settings)
    model = transformers.AutoModelForCausalLM.from_config(
        config, attn_implementation="eager",
    )
    model = model.double().eval().requires_grad_(False)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
            if not tensor.is_floating_point() or "inv_freq" in name or "embed_scale" in name:
                continue
            noise = torch.randn(tensor.shape, generator=gen, dtype=torch.float64)
            if "norm" in name or "layer_scalar" in name:
                tensor.copy_(0.8 + 0.3 * noise)
            else:
                tensor.copy_(0.3 * noise)
    return model


def default_init_model(model_type: str, *, seed: int, dtype: torch.dtype) -> Any:
    """A tiny ``model_type`` at its own default initialization, cast to ``dtype``.

    The precision a GPU load runs at (bf16), with the small weights and
    near-identity norms of a fresh model: its rounding, not its structure,
    is what separates its forward from the fp32 description.  SDPA where
    the family takes it, eager otherwise (the loader's ladder).
    """
    settings = {**_TINY, **DENSE_FAMILIES.get(model_type, {})}
    config = transformers.AutoConfig.for_model(model_type, **settings)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        try:
            model = transformers.AutoModelForCausalLM.from_config(
                config, attn_implementation="sdpa",
            )
        except ValueError:  # gemma2: no SDPA
            model = transformers.AutoModelForCausalLM.from_config(
                config, attn_implementation="eager",
            )
    return model.to(dtype).eval().requires_grad_(False)


def spec_forward(
    spec: ModelSpec, ids: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """``(mids, outs, logits)`` for one unbatched sequence ``ids`` ``[T]``."""
    h = spec.embed.weight[ids] * spec.embed_scale
    mids: list[torch.Tensor] = []
    outs: list[torch.Tensor] = []
    for block in spec.blocks:
        mid, h = block_forward(spec, block, h)
        mids.append(mid)
        outs.append(h)
    final = spec.final_norm
    r = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + final.eps)
    logits = (h * r * final.gain(dtype=h.dtype)) @ spec.unembed.T.to(h.dtype)
    if spec.final_logit_softcap is not None:
        cap = spec.final_logit_softcap
        logits = cap * torch.tanh(logits / cap)
    return mids, outs, logits
