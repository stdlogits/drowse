"""Dense decoder blocks as data: what transport and unit reads consume.

A :class:`BlockSpec` describes one decoder block in semantic terms (the
attention's input norm, the MLP's down projection, the head layout), so the
lens transport estimator and the unit directions in :mod:`drowse.model.units`
read one description instead of each family's attribute names.  A
:class:`ModelSpec` adds the model-level part: embedding and its scale, final
norm and logit softcap, the rotary table, and the shape.

Every block the description covers computes::

    a     = attn_in_norm(h)
    h_mid = h + attn_out_norm(o(attention(q(a), k(a), v(a))))       # mid
    m     = mlp_in_norm(h_mid)
    h_out = (h_mid + mlp_out_norm(down(act(gate(m)) * up(m)))) * scalar

``attn_out_norm`` and ``mlp_out_norm`` are Gemma's sandwich norms (absent:
identity), ``scalar`` is Gemma 4's per-layer ``layer_scalar`` (absent: 1).
Attention is causal softmax attention with grouped KV heads, optional q/k/v
RMSNorms per head before RoPE, scores ``q·k · scaling`` capped by
``logit_softcap·tanh(s/logit_softcap)`` when set, and an optional sliding
window (position ``i`` sees ``j`` iff ``i − window < j ≤ i``).  ``v`` absent
means the value is the raw key projection (Gemma 4's full layers,
``attention_k_eq_v``).  Every norm is an RMSNorm
``x·rsqrt(mean(x²)+eps)·(offset + w)``.

Discovery is structural: block parts are found by module presence and
config fields, never by ``model_type``.  Norm conventions (``w`` or
``1 + w``) and the activation (SiLU or tanh-GELU) are identified by running
the module on a probe and matching its output, and each gated MLP is checked
to compute exactly ``down(act(gate(x)) * up(x))``.  Anything the description
cannot hold (MoE, fused gate-up or qkv projections, linear-attention layers,
attention sinks, KV sharing, unknown norms or submodules) raises
:class:`~drowse.errors.UnsupportedBlockError` naming the layer and what was
found.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from drowse.errors import UnsupportedBlockError
from drowse.model.arch import get_layers, text_config
from drowse.model.readout import get_final_norm, get_logit_softcap, get_unembedding

ActivationKind = Literal["silu", "gelu_tanh"]

#: Residual-norm roles by attribute.  A block with ``pre_feedforward_layernorm``
#: is a sandwich block (Gemma): its ``post_attention_layernorm`` normalizes the
#: attention output and ``pre_feedforward_layernorm`` feeds the MLP.  Without
#: it, ``post_attention_layernorm`` feeds the MLP (llama, qwen, mistral).
_ATTN_IN = "input_layernorm"
_POST_ATTN = "post_attention_layernorm"
_PRE_FF = "pre_feedforward_layernorm"
_POST_FF = "post_feedforward_layernorm"
_BLOCK_CHILDREN = frozenset({"self_attn", "mlp", _ATTN_IN, _POST_ATTN, _PRE_FF, _POST_FF})
_BLOCK_TENSORS = frozenset({"layer_scalar"})
_ATTN_CHILDREN = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "v_norm"})
_MLP_CHILDREN = frozenset({"gate_proj", "up_proj", "down_proj", "act_fn"})
_MOE_NAMES = frozenset({"experts", "router", "shared_expert", "shared_experts", "gate"})

_GELU_C = math.sqrt(2.0 / math.pi)
#: Probe-match tolerances.  Wide enough for bf16 modules, far below the O(1)
#: gap between the shapes being told apart.
_MATCH_RTOL = 3e-2
_OFFSET_GAP = 0.25


# -- activations -------------------------------------------------------------

def activation(kind: ActivationKind, g: torch.Tensor) -> torch.Tensor:
    """``act(g)``: ``g·sigmoid(g)`` (SiLU) or tanh-approximated GELU."""
    if kind == "silu":
        return g * torch.sigmoid(g)
    return 0.5 * g * (1.0 + torch.tanh(_GELU_C * (g + 0.044715 * g.pow(3))))


def activation_factor(kind: ActivationKind, g: torch.Tensor) -> torch.Tensor:
    """``φ(g) = act(g)/g`` without dividing: the RelP identity rule's factor.

    ``sigmoid(g)`` for SiLU, ``(1 + tanh(√(2/π)(g + 0.044715g³)))/2`` for
    tanh-GELU.
    """
    if kind == "silu":
        return torch.sigmoid(g)
    return 0.5 * (1.0 + torch.tanh(_GELU_C * (g + 0.044715 * g.pow(3))))


def activation_derivative(kind: ActivationKind, g: torch.Tensor) -> torch.Tensor:
    """``act'(g)``, the true derivative the standard (non-RelP) rules use."""
    if kind == "silu":
        s = torch.sigmoid(g)
        return s * (1.0 + g * (1.0 - s))
    inner = _GELU_C * (g + 0.044715 * g.pow(3))
    t = torch.tanh(inner)
    return 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * _GELU_C * (1.0 + 3 * 0.044715 * g * g)


# -- the description -----------------------------------------------------------

@dataclass(frozen=True, eq=False)
class NormSpec:
    """One RMSNorm, ``y = x · rsqrt(mean(x²) + eps) · (offset + w)``.

    ``offset`` is 0 for a plain weight (llama, qwen, Gemma 4) and 1 for the
    zero-centred ``1 + w`` (Gemma 2/3); ``weighted`` is False for a norm with
    no weight (Gemma 4's value norm), which scales by 1.
    """

    module: nn.Module
    dim: int
    eps: float
    offset: float
    weighted: bool

    def gain(
        self, *, device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """The per-channel multiplier ``offset + w``, ``[dim]`` (fp32 by default)."""
        if not self.weighted:
            return torch.ones(self.dim, dtype=dtype, device=device)
        weight = cast(torch.Tensor, getattr(self.module, "weight")).detach()
        return weight.to(device=device, dtype=dtype) + self.offset

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """The norm in fp32 over the last dimension (a reference, not the module)."""
        xf = x.float()
        r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf * r * self.gain(device=xf.device)


@dataclass(frozen=True, eq=False)
class AttentionSpec:
    """A block's causal softmax attention (see the module docstring)."""

    q: nn.Linear
    k: nn.Linear
    #: ``None``: the value is ``k``'s raw output (before ``k_norm`` and RoPE).
    v: nn.Linear | None
    o: nn.Linear
    q_norm: NormSpec | None
    k_norm: NormSpec | None
    v_norm: NormSpec | None
    n_heads: int
    n_kv_heads: int
    head_dim: int
    scaling: float
    sliding_window: int | None
    logit_softcap: float | None
    #: The key the model's rotary table takes for this layer (Gemma 3/4 keep
    #: one table per layer type); ``None`` when the model has one table.
    rope_table: str | None

    @property
    def group_size(self) -> int:
        """Query heads per KV head."""
        return self.n_heads // self.n_kv_heads

    def kv_head(self, head: int) -> int:
        """The KV head query head ``head`` reads."""
        return head // self.group_size

    @property
    def qkv_bias(self) -> bool:
        return any(
            proj is not None and proj.bias is not None
            for proj in (self.q, self.k, self.v)
        )

    @property
    def o_bias(self) -> bool:
        return self.o.bias is not None


@dataclass(frozen=True, eq=False)
class MLPSpec:
    """A gated MLP, ``down(act(gate(x)) * up(x))``."""

    gate: nn.Linear
    up: nn.Linear
    down: nn.Linear
    act: ActivationKind
    d_ff: int

    @property
    def bias(self) -> bool:
        return any(proj.bias is not None for proj in (self.gate, self.up, self.down))


@dataclass(frozen=True, eq=False)
class BlockSpec:
    """One dense decoder block, described for transport and unit reads.

    Holds live module references (weights are read when used, never copied).
    ``attn_out_norm`` / ``mlp_out_norm`` are present only on sandwich blocks;
    ``scalar`` is the block's per-layer residual multiplier when it has one.
    """

    layer: int
    module: nn.Module
    attn_in_norm: NormSpec
    attn_out_norm: NormSpec | None
    mlp_in_norm: NormSpec
    mlp_out_norm: NormSpec | None
    attn: AttentionSpec
    mlp: MLPSpec
    scalar: torch.Tensor | None

    @property
    def d_model(self) -> int:
        return self.attn.q.in_features

    @property
    def d_ff(self) -> int:
        return self.mlp.d_ff

    @property
    def n_heads(self) -> int:
        return self.attn.n_heads

    def scale(self) -> float:
        """The per-layer residual multiplier (1.0 when the block has none)."""
        return 1.0 if self.scalar is None else float(self.scalar.detach().reshape(()))


@dataclass(frozen=True, eq=False)
class ModelSpec:
    """The model-level part around the blocks.

    The residual stream starts at ``embed(ids) · embed_scale`` and the logits
    are ``final_logit_softcap·tanh(W_U·final_norm(h)/final_logit_softcap)``
    (no cap when ``None``).  ``rotary`` is the model's rotary-embedding
    module (called with a layer type when ``rotary_takes_layer_type``).
    ``d_ff`` is the widest block's (every block's on the covered families).
    """

    embed: nn.Embedding
    embed_scale: float
    final_norm: NormSpec
    final_logit_softcap: float | None
    unembed: torch.Tensor
    rotary: nn.Module
    rotary_takes_layer_type: bool
    n_layers: int
    d_model: int
    d_ff: int
    vocab: int
    blocks: tuple[BlockSpec, ...]


# -- discovery -------------------------------------------------------------------

def model_spec(model: PreTrainedModel) -> ModelSpec:
    """Describe ``model``'s dense decoder, or raise :class:`UnsupportedBlockError`."""
    layers = get_layers(model)
    owner = _block_list_owner(model, layers)
    config = text_config(model)
    _refuse_config_quirks(config)
    rotary = getattr(owner, "rotary_emb", None)
    if not isinstance(rotary, nn.Module):
        raise UnsupportedBlockError(
            "no rotary embedding beside the block list ('rotary_emb'); "
            "a dense block here is RoPE attention"
        )
    takes_layer_type = "layer_type" in inspect.signature(rotary.forward).parameters
    blocks = tuple(
        _block_spec(index, block, config, takes_layer_type)
        for index, block in enumerate(layers)
    )
    embed = model.get_input_embeddings()
    if not isinstance(embed, nn.Embedding):
        raise UnsupportedBlockError(
            f"input embedding is {type(embed).__name__}, not an nn.Embedding"
        )
    d_model = int(embed.weight.shape[1])
    embed_scale = _embed_scale(embed)
    unembed = get_unembedding(model)
    if int(unembed.shape[1]) != d_model:
        raise UnsupportedBlockError(
            f"unembedding width {int(unembed.shape[1])} differs from the "
            f"residual width {d_model}"
        )
    final = get_final_norm(model)
    softcap = get_logit_softcap(model)
    return ModelSpec(
        embed=embed,
        embed_scale=embed_scale,
        final_norm=_norm_spec(final, d_model, "final norm", layer=None),
        final_logit_softcap=None if softcap is None else float(softcap),
        unembed=unembed,
        rotary=rotary,
        rotary_takes_layer_type=takes_layer_type,
        n_layers=len(blocks),
        d_model=d_model,
        d_ff=max(block.d_ff for block in blocks),
        vocab=int(unembed.shape[0]),
        blocks=blocks,
    )


def block_specs(model: PreTrainedModel) -> list[BlockSpec]:
    """One :class:`BlockSpec` per decoder block, in order (see :func:`model_spec`)."""
    return list(model_spec(model).blocks)


def _block_list_owner(model: nn.Module, layers: nn.ModuleList) -> nn.Module:
    for module in model.modules():
        if any(child is layers for child in module.children()):
            return module
    raise UnsupportedBlockError("the block list has no owning module")  # pragma: no cover


#: Config fields that scale a forward term (granite's residual, embedding and
#: attention multipliers, its ``logits_scaling``, cohere's ``logit_scale``).
_SCALE_FIELDS = ("logits_scaling", "logit_scale")


def _refuse_config_quirks(config: Any) -> None:
    """Config-level forward terms the description has no field for."""
    for key, value in sorted(config.to_dict().items()):
        if not (key.endswith("_multiplier") or key in _SCALE_FIELDS):
            continue
        if value is not None and value != 1:
            raise UnsupportedBlockError(
                f"config '{key}' = {value!r} scales a forward term the "
                "description has no field for"
            )
    rope = getattr(config, "rope_parameters", None)
    tables = rope.values() if isinstance(rope, Mapping) and all(
        isinstance(v, Mapping) for v in rope.values()
    ) else [rope]
    for table in tables:
        if isinstance(table, Mapping) and table.get("llama_4_scaling_beta") is not None:
            raise UnsupportedBlockError(
                "position-dependent query scaling (rope 'llama_4_scaling_beta') "
                "has no place in a block description"
            )


def _probe(dim: int, *, rows: int = 4, seed: int = 0) -> torch.Tensor:
    """A deterministic fp32 CPU probe whose entries keep |x| ≥ 0.5."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, dim, generator=gen)
    return x.sign() * (0.5 + x.abs())


def _param_like(module: nn.Module, fallback: nn.Module | None = None) -> torch.Tensor | None:
    for source in (module, fallback):
        if source is None:
            continue
        for tensor in source.parameters():
            return tensor
    return None


def _norm_spec(
    module: Any, dim: int, where: str, *, layer: int | None,
    like: nn.Module | None = None,
) -> NormSpec:
    """Identify an RMSNorm's convention by matching its output on a probe."""
    if not isinstance(module, nn.Module):
        raise UnsupportedBlockError(f"{where} is missing", layer=layer)
    kind = type(module).__name__
    eps = getattr(module, "variance_epsilon", getattr(module, "eps", None))
    if isinstance(eps, torch.Tensor) or not isinstance(eps, (float, int)) or isinstance(eps, bool):
        raise UnsupportedBlockError(
            f"{where} ({kind}) exposes no eps; only RMSNorms are described",
            layer=layer,
        )
    weight = getattr(module, "weight", None)
    if weight is not None and (
        not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (dim,)
    ):
        raise UnsupportedBlockError(
            f"{where} ({kind}) weight is not a [{dim}] vector", layer=layer,
        )
    ref = _param_like(module, like)
    device = ref.device if ref is not None else torch.device("cpu")
    dtype = ref.dtype if ref is not None else torch.float32
    x = _probe(dim)
    with torch.no_grad():
        y = module(x.to(device=device, dtype=dtype)).float().cpu()
    x = x.to(dtype=dtype).float()
    if tuple(y.shape) != tuple(x.shape):
        raise UnsupportedBlockError(f"{where} ({kind}) changes its input's shape", layer=layer)
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + float(eps))
    implied = y / (x * r)  # the per-channel multiplier the module applied
    w = None if weight is None else weight.detach().float().cpu()
    candidates = (0.0, 1.0) if w is not None else (0.0,)
    for offset in candidates:
        gain = torch.ones(dim) if w is None else w + offset
        error = (implied - gain).abs()
        if (
            float(error.median()) < _OFFSET_GAP
            and bool((error <= _MATCH_RTOL * gain.abs().clamp_min(1.0)).all())
        ):
            return NormSpec(
                module=module, dim=dim, eps=float(eps), offset=offset,
                weighted=w is not None,
            )
    raise UnsupportedBlockError(
        f"{where} ({kind}) is not an RMSNorm with a plain or 1+w weight",
        layer=layer,
    )


def _activation_kind(act: Any, layer: int) -> ActivationKind:
    if not callable(act):
        raise UnsupportedBlockError("MLP has no activation ('act_fn')", layer=layer)
    g = torch.linspace(-6.0, 6.0, 241)
    with torch.no_grad():
        y = cast(torch.Tensor, act(g)).float()
    for kind in ("silu", "gelu_tanh"):
        kind = cast(ActivationKind, kind)
        if torch.allclose(y, activation(kind, g), rtol=1e-5, atol=1e-6):
            return kind
    raise UnsupportedBlockError(
        f"MLP activation {type(act).__name__} is neither SiLU nor tanh-GELU",
        layer=layer,
    )


def _linear(module: Any, what: str, layer: int) -> nn.Linear:
    if not isinstance(module, nn.Linear):
        raise UnsupportedBlockError(f"{what} is missing or not an nn.Linear", layer=layer)
    return module


def _children(module: nn.Module) -> dict[str, nn.Module]:
    return dict(module.named_children())


def _direct_tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = dict(module.named_parameters(recurse=False))
    out.update(dict(module.named_buffers(recurse=False)))
    return out


def _block_spec(
    index: int, block: nn.Module, config: Any, rotary_takes_layer_type: bool,
) -> BlockSpec:
    children = _children(block)
    if "self_attn" not in children:
        linear = sorted(name for name in children if "linear" in name or "mamba" in name)
        if linear:
            raise UnsupportedBlockError(
                f"linear-attention layer ('{linear[0]}') in a hybrid stack; only "
                "softmax self-attention blocks are described",
                layer=index,
            )
        raise UnsupportedBlockError(
            f"block has no softmax self-attention ('self_attn'; found "
            f"{', '.join(sorted(children)) or 'nothing'})",
            layer=index,
        )
    if "mlp" not in children:
        raise UnsupportedBlockError("block has no 'mlp'", layer=index)
    moe = sorted(_MOE_NAMES & set(children))
    if moe:
        raise UnsupportedBlockError(
            f"mixture-of-experts block ('{moe[0]}'); only dense MLPs are described",
            layer=index,
        )
    unknown = sorted(
        name for name, child in children.items()
        if name not in _BLOCK_CHILDREN and not isinstance(child, nn.Dropout)
    )
    if unknown:
        raise UnsupportedBlockError(
            "unrecognized block submodules " + ", ".join(f"'{n}'" for n in unknown)
            + " (a forward term the description has no field for)",
            layer=index,
        )
    extra = sorted(set(_direct_tensors(block)) - _BLOCK_TENSORS)
    if extra:
        raise UnsupportedBlockError(f"unrecognized block tensor '{extra[0]}'", layer=index)
    scalar = _direct_tensors(block).get("layer_scalar")
    if scalar is not None and scalar.numel() != 1:
        raise UnsupportedBlockError("'layer_scalar' is not a scalar", layer=index)

    attn = _attention_spec(index, children["self_attn"], config, rotary_takes_layer_type)
    d_model = attn.q.in_features
    sandwich = _PRE_FF in children
    if _POST_FF in children and not sandwich:
        raise UnsupportedBlockError(
            f"'{_POST_FF}' without '{_PRE_FF}' is an unrecognized norm placement",
            layer=index,
        )
    attn_in = _norm_spec(children.get(_ATTN_IN), d_model, "attention input norm", layer=index)
    post_attn = _norm_spec(
        children.get(_POST_ATTN), d_model,
        "attention output norm" if sandwich else "MLP input norm", layer=index,
    )
    if sandwich:
        attn_out: NormSpec | None = post_attn
        mlp_in = _norm_spec(children[_PRE_FF], d_model, "MLP input norm", layer=index)
        mlp_out = (
            _norm_spec(children[_POST_FF], d_model, "MLP output norm", layer=index)
            if _POST_FF in children else None
        )
    else:
        attn_out, mlp_in, mlp_out = None, post_attn, None
    return BlockSpec(
        layer=index,
        module=block,
        attn_in_norm=attn_in,
        attn_out_norm=attn_out,
        mlp_in_norm=mlp_in,
        mlp_out_norm=mlp_out,
        attn=attn,
        mlp=_mlp_spec(index, children["mlp"], d_model),
        scalar=scalar,
    )


def _attention_spec(
    index: int, attn: nn.Module, config: Any, rotary_takes_layer_type: bool,
) -> AttentionSpec:
    children = _children(attn)
    if "qkv_proj" in children:
        raise UnsupportedBlockError(
            "fused qkv projection ('qkv_proj'); a dense block keeps separate "
            "q, k and v projections",
            layer=index,
        )
    for name, child in children.items():
        if name not in _ATTN_CHILDREN:
            raise UnsupportedBlockError(
                f"unrecognized attention submodule '{name}' ({type(child).__name__})",
                layer=index,
            )
    params = sorted(_direct_tensors(attn))
    if params:
        raise UnsupportedBlockError(
            f"attention tensor '{params[0]}' (attention sinks or another term "
            "the description has no field for)",
            layer=index,
        )
    if getattr(attn, "is_kv_shared_layer", False):
        raise UnsupportedBlockError(
            "keys and values are shared with an earlier layer", layer=index,
        )
    if getattr(attn, "is_causal", True) is not True:
        raise UnsupportedBlockError("attention is not causal", layer=index)
    if getattr(attn, "use_rope", True) is not True:
        raise UnsupportedBlockError("attention applies no RoPE (a NoPE layer)", layer=index)
    head_dim = getattr(attn, "head_dim", None)
    scaling = getattr(attn, "scaling", None)
    if not isinstance(head_dim, int) or not isinstance(scaling, (float, int)):
        raise UnsupportedBlockError(
            "attention exposes no 'head_dim' and 'scaling'", layer=index,
        )
    q = _linear(children.get("q_proj"), "q projection", index)
    k = _linear(children.get("k_proj"), "k projection", index)
    v = children.get("v_proj")
    v = None if v is None else _linear(v, "v projection", index)
    o = _linear(children.get("o_proj"), "o projection", index)
    n_heads, rem_q = divmod(q.out_features, head_dim)
    n_kv, rem_k = divmod(k.out_features, head_dim)
    if (
        rem_q or rem_k or n_kv == 0 or n_heads % n_kv
        or o.in_features != n_heads * head_dim
        or (v is not None and v.out_features != k.out_features)
        or k.in_features != q.in_features
        or o.out_features != q.in_features
    ):
        raise UnsupportedBlockError(
            f"attention projections do not form a head layout of head_dim {head_dim}",
            layer=index,
        )
    groups = getattr(attn, "num_key_value_groups", None)
    if isinstance(groups, int) and groups != n_heads // n_kv:
        raise UnsupportedBlockError(
            f"num_key_value_groups {groups} disagrees with {n_heads} query and "
            f"{n_kv} key heads",
            layer=index,
        )
    # A per-layer window lives on the attention (qwen2/3, gemma); mistral
    # reads one window for every layer from its config.
    if hasattr(attn, "sliding_window"):
        window = getattr(attn, "sliding_window")
    else:
        window = getattr(config, "sliding_window", None)
    softcap = getattr(attn, "attn_logit_softcapping", None)
    rope_table: str | None = None
    if rotary_takes_layer_type:
        rope_table = getattr(attn, "layer_type", None)
        if not isinstance(rope_table, str):
            raise UnsupportedBlockError(
                "the rotary table is per layer type but the attention names none",
                layer=index,
            )

    def head_norm(name: str) -> NormSpec | None:
        module = children.get(name)
        if module is None:
            return None
        return _norm_spec(module, head_dim, f"attention {name}", layer=index, like=q)

    return AttentionSpec(
        q=q, k=k, v=v, o=o,
        q_norm=head_norm("q_norm"),
        k_norm=head_norm("k_norm"),
        v_norm=head_norm("v_norm"),
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        scaling=float(scaling),
        sliding_window=None if window is None else int(window),
        logit_softcap=None if softcap is None else float(softcap),
        rope_table=rope_table,
    )


def _mlp_spec(index: int, mlp: nn.Module, d_model: int) -> MLPSpec:
    children = _children(mlp)
    moe = sorted(_MOE_NAMES & set(children))
    if moe:
        raise UnsupportedBlockError(
            f"mixture-of-experts MLP ('mlp.{moe[0]}'); only dense MLPs are described",
            layer=index,
        )
    if "gate_up_proj" in children:
        raise UnsupportedBlockError(
            "fused gate-up projection ('mlp.gate_up_proj'); a dense block keeps "
            "separate gate, up and down projections",
            layer=index,
        )
    for name, child in children.items():
        if name not in _MLP_CHILDREN:
            raise UnsupportedBlockError(
                f"unrecognized MLP submodule '{name}' ({type(child).__name__})",
                layer=index,
            )
    if _direct_tensors(mlp):
        raise UnsupportedBlockError("MLP holds tensors outside its projections", layer=index)
    gate = _linear(children.get("gate_proj"), "MLP gate projection", index)
    up = _linear(children.get("up_proj"), "MLP up projection", index)
    down = _linear(children.get("down_proj"), "MLP down projection", index)
    d_ff = gate.out_features
    if (
        gate.in_features != d_model or up.in_features != d_model
        or up.out_features != d_ff or down.in_features != d_ff
        or down.out_features != d_model
    ):
        raise UnsupportedBlockError("MLP projections do not chain d→d_ff→d", layer=index)
    act = _activation_kind(getattr(mlp, "act_fn", None), index)
    # A module can duck-type the gated shape while its forward adds a term.
    weight = gate.weight
    x = _probe(d_model, seed=1).to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        actual = cast(torch.Tensor, mlp(x)).float()
        hidden = activation(act, gate(x).float()).to(weight.dtype) * up(x)
        expected = down(hidden).float()
    scale = float(expected.abs().max().clamp_min(1e-6))
    if float((actual - expected).abs().max()) > _MATCH_RTOL * scale:
        raise UnsupportedBlockError(
            f"MLP {type(mlp).__name__} does not compute down(act(gate(x)) * up(x))",
            layer=index,
        )
    return MLPSpec(gate=gate, up=up, down=down, act=act, d_ff=d_ff)


# -- the reference forward -------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), -1)


def _apply_norm(norm: NormSpec | None, x: torch.Tensor) -> torch.Tensor:
    if norm is None:
        return x
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
    return x * r * norm.gain(device=x.device, dtype=x.dtype)


def _apply_linear(proj: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    bias = None if proj.bias is None else proj.bias.to(x.dtype)
    return torch.nn.functional.linear(x, proj.weight.to(x.dtype), bias)


def rotary_tables(
    spec: ModelSpec, block: BlockSpec, positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)``, ``[T, 1, head_dim]`` fp32, from the model's rotary table."""
    probe = torch.zeros(1, positions.numel(), 1, device=positions.device)
    pos = positions.reshape(1, -1)
    with torch.no_grad():
        if spec.rotary_takes_layer_type:
            cos, sin = spec.rotary(probe, pos, block.attn.rope_table)
        else:
            cos, sin = spec.rotary(probe, pos)
    return cos[0, :, None].float(), sin[0, :, None].float()


def block_forward(
    spec: ModelSpec, block: BlockSpec, h: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mid, out)`` of ``block`` on one unbatched sequence ``h`` ``[T, d]``.

    Computed from the description alone (the formula in the module
    docstring), in fp32 (fp64 inputs stay fp64), at ``positions`` (default
    ``0..T−1``).  This is the reference :func:`verify_model_spec` holds a
    block's own forward to.
    """
    dtype = torch.promote_types(h.dtype, torch.float32)
    h = h.to(dtype)
    T = h.shape[0]
    if positions is None:
        positions = torch.arange(T, device=h.device)
    attn = block.attn
    a = _apply_norm(block.attn_in_norm, h)
    q = _apply_linear(attn.q, a).view(T, attn.n_heads, attn.head_dim)
    k_raw = _apply_linear(attn.k, a).view(T, attn.n_kv_heads, attn.head_dim)
    v = k_raw if attn.v is None else _apply_linear(attn.v, a).view(T, attn.n_kv_heads, attn.head_dim)
    q, k, v = _apply_norm(attn.q_norm, q), _apply_norm(attn.k_norm, k_raw), _apply_norm(attn.v_norm, v)
    cos, sin = rotary_tables(spec, block, positions.to(h.device))
    cos, sin = cos.to(dtype), sin.to(dtype)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    k = k.repeat_interleave(attn.group_size, dim=1)
    v = v.repeat_interleave(attn.group_size, dim=1)
    scores = torch.einsum("ihd,jhd->hij", q, k) * attn.scaling
    if attn.logit_softcap is not None:
        scores = attn.logit_softcap * torch.tanh(scores / attn.logit_softcap)
    i = positions.to(h.device)[:, None]
    j = positions.to(h.device)[None, :]
    visible = j <= i
    if attn.sliding_window is not None:
        visible = visible & (j > i - attn.sliding_window)
    scores = scores.masked_fill(~visible, float("-inf"))
    o = torch.einsum("hij,jhd->ihd", scores.softmax(-1), v).reshape(T, -1)
    mid = h + _apply_norm(block.attn_out_norm, _apply_linear(attn.o, o))
    return mid, mlp_forward(block, mid)


def mlp_forward(block: BlockSpec, mid: torch.Tensor) -> torch.Tensor:
    """The block's output from its ``mid`` residual: the MLP half of :func:`block_forward`."""
    dtype = torch.promote_types(mid.dtype, torch.float32)
    mid = mid.to(dtype)
    mlp = block.mlp
    m = _apply_norm(block.mlp_in_norm, mid)
    z = _apply_linear(mlp.down, activation(mlp.act, _apply_linear(mlp.gate, m)) * _apply_linear(mlp.up, m))
    return (mid + _apply_norm(block.mlp_out_norm, z)) * block.scale()


#: How far a block's own forward may sit from its description, relative to
#: the residual update being checked (bf16 compute lands near 1e-2; a missing
#: term or a different RoPE layout lands near 1).
_VERIFY_RTOL = 5e-2


def verify_model_spec(
    model: PreTrainedModel, spec: ModelSpec | None = None, *, length: int = 12,
) -> ModelSpec:
    """Hold every block's own forward to its description, or refuse.

    Runs the model once on ``length`` probe tokens with hooks on each block
    and checks, per block, that :func:`block_forward` reproduces the
    attention update (``mid − h``) from the block's real input and
    :func:`mlp_forward` the output from its real ``mid``.  A forward term
    the structure does not show (an interleaved RoPE layout, an extra
    multiplier) raises :class:`UnsupportedBlockError` naming the block.

    It runs a forward, so the caller holds the model exclusively and
    unsteered (steering hooks would change the outputs it compares).
    Transport fits call it; static unit reads need only :func:`model_spec`.
    """
    spec = model_spec(model) if spec is None else spec
    device = spec.embed.weight.device
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(3, spec.vocab, (1, int(length)), generator=gen).to(device)
    inputs: dict[int, torch.Tensor] = {}
    mids: dict[int, torch.Tensor] = {}
    outs: dict[int, torch.Tensor] = {}

    def first_tensor(value: Any) -> torch.Tensor:
        return value[0] if isinstance(value, tuple) else value

    handles = []
    for block in spec.blocks:
        index = block.layer

        def take_input(_m: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], index: int = index) -> None:
            inputs[index] = (args[0] if args else kwargs["hidden_states"])[0].detach()

        def take_mid(_m: nn.Module, args: tuple[Any, ...], index: int = index) -> None:
            mids[index] = args[0][0].detach()

        def take_output(_m: nn.Module, _a: Any, output: Any, index: int = index) -> None:
            outs[index] = first_tensor(output)[0].detach()

        handles.append(block.module.register_forward_pre_hook(take_input, with_kwargs=True))
        handles.append(block.mlp_in_norm.module.register_forward_pre_hook(take_mid))
        handles.append(block.module.register_forward_hook(take_output))
    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    def relative(error: torch.Tensor, scale: torch.Tensor) -> float:
        return float(error.float().norm() / scale.float().norm().clamp_min(1e-12))

    for block in spec.blocks:
        h, mid, out = inputs[block.layer], mids[block.layer], outs[block.layer]
        with torch.no_grad():
            mid_ref, _ = block_forward(spec, block, h)
            out_ref = mlp_forward(block, mid)
        mid_err = relative(mid_ref - mid.to(mid_ref.dtype), mid.to(mid_ref.dtype) - h.to(mid_ref.dtype))
        out_err = relative(
            out_ref - out.to(out_ref.dtype),
            out.to(out_ref.dtype) - mid.to(out_ref.dtype) * block.scale(),
        )
        if max(mid_err, out_err) > _VERIFY_RTOL:
            half = "attention" if mid_err > _VERIFY_RTOL else "MLP"
            raise UnsupportedBlockError(
                f"the block's {half} forward differs from its description "
                f"(relative error {max(mid_err, out_err):.2g}); it computes a term "
                "the description has no field for",
                layer=block.layer,
            )
    return spec


def _embed_scale(embed: nn.Embedding) -> float:
    raw = getattr(embed, "embed_scale", None)
    scale = 1.0 if raw is None else float(
        raw.detach().reshape(()) if isinstance(raw, torch.Tensor) else raw
    )
    ids = torch.arange(min(4, embed.num_embeddings), device=embed.weight.device)
    with torch.no_grad():
        actual = embed(ids).float()
    expected = embed.weight[ids].float() * scale
    tol = _MATCH_RTOL * float(expected.abs().max().clamp_min(1e-6))
    if float((actual - expected).abs().max()) > tol:
        raise UnsupportedBlockError(
            f"input embedding {type(embed).__name__} applies a transform other "
            "than a scale"
        )
    return scale


__all__ = [
    "ActivationKind",
    "AttentionSpec",
    "BlockSpec",
    "MLPSpec",
    "ModelSpec",
    "NormSpec",
    "activation",
    "activation_derivative",
    "activation_factor",
    "block_forward",
    "block_specs",
    "mlp_forward",
    "model_spec",
    "rotary_tables",
    "verify_model_spec",
]
