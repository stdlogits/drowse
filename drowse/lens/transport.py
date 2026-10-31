"""Transport: every decoder block's lens map on one document, written out.

The four lens kinds average one Jacobian of the residual stream over a
corpus, summed over offsets, at two sites per block (``out``, the block's
output; ``mid``, the residual after the attention add, the MLP's input):

- output kinds (J, R): ``J_L = mean_t Σ_{t'≥t} ∂h_final(t')/∂h_L(t)``, from
  the site to the last block's output (before the final norm, so the final
  norm and logit softcap are the readout's, never the lens's);
- input kinds (K, S): ``K_L = mean_t Σ_{s≤t} ∂h_L(t)/∂e(s)``, from the
  residual stream's input ``e = embed(ids)·embed_scale`` to the site.

Each block is, on a fixed document, a fixed linear map of its input
residual. :func:`linearize` runs the primal once per document from the
:class:`~drowse.model.blocks.ModelSpec` and keeps what the maps need (norm
scales, attention probabilities and softcap factors, RoPE tables, the MLP's
activation factors); :func:`jvp` pushes a block of tangents up through one
block's map (the input kinds' forward sweep from the embedding) and
:func:`vjp` pulls cotangents down through its transpose (the output kinds'
reverse sweep from the last block).  Each is a few matrix products over
``seeds × positions`` rows, with no autograd.

``rules`` picks the derivative: ``"standard"`` (J, K) takes every derivative
true (an RMSNorm is diagonal plus rank one, ``act'(g)``, the product rule);
``"relp"`` (R, S) takes the RelP rules of :mod:`drowse.lens.relp` (residual
norms hold their scale, the identity factor, the half rule).  Head norms,
softmax, RoPE and the softcap are true under both.

**Positions.** Seeds and reads both sit at the valid positions ``[S, T−1)``
(``S = skip_first``, the attention-sink cut; the last position has no later
one).  Attention is causal, so a tangent seeded there never reaches an
earlier position and an earlier position's cotangent never returns to a
valid one: the maps carry only the ``N`` valid rows, and the primal drops
the last token, which no valid position attends to.  Keys run over every
kept position, padded to a multiple of 8 with masked ones so attention
rows stay 16-byte aligned for cuBLAS.

**Precision.** Matrix products run in the model's dtype; tangents,
cotangents, norm factors and softmax derivatives in ``promote(dtype, fp32)``
(fp32 for a bf16 model, fp64 for an fp64 one).  On CUDA the per-block maps
are compiled with fixed shapes, which fuses the elementwise work between the
products; elsewhere they run eagerly.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from drowse.errors import UnsupportedBlockError
from drowse.lens.relp import HALF_RULE_BETA
from drowse.model.blocks import (
    BlockSpec,
    ModelSpec,
    NormSpec,
    activation,
    activation_derivative,
    activation_factor,
    rotary_tables,
)

log = logging.getLogger(__name__)

Rules = Literal["relp", "standard"]
RULES: tuple[Rules, ...] = ("relp", "standard")

#: Key positions are padded to a multiple of this (masked), so attention
#: rows stay aligned for the fast batched-matmul kernels.
_KEY_ALIGN = 8


def rules_for(kind: str) -> Rules:
    """``"relp"`` for R and S, ``"standard"`` for J and K."""
    if kind in ("R", "S"):
        return "relp"
    if kind in ("J", "K"):
        return "standard"
    raise ValueError(f"lens kind must be J, R, K or S, got {kind!r}")


def valid_rows(n_tokens: int, skip_first: int) -> int:
    """The number of valid positions ``[skip_first, n_tokens − 1)``."""
    return max(0, int(n_tokens) - 1 - int(skip_first))


# -- per-document constants ---------------------------------------------------------

@dataclass(frozen=True, eq=False)
class NormMap:
    """One RMSNorm's map at the valid rows: ``a = gain·rsqrt(mean(x²)+eps)``
    and, for the true derivative, the normalized input ``x̂ = x·rsqrt(…)``.

    JVP ``a⊙(dx − x̂·mean(x̂⊙dx))``; its transpose ``u − x̂·mean(x̂⊙u)`` with
    ``u = a⊙gy``.  ``xhat=None`` is the LN rule's diagonal ``a⊙dx``.
    """

    a: torch.Tensor
    xhat: torch.Tensor | None


@dataclass(frozen=True, eq=False)
class LayerMap:
    """One block's linear map on one document (rows: the ``N`` valid
    positions; keys: ``Tp`` padded positions).

    Projection weights stay the model's own tensors (no copies).  Attention
    tensors group query heads by their KV head: ``qr`` ``[Hk, g·N, hd]``,
    ``kr``/``v`` ``[Hk, Tp, hd]``, ``p`` ``[Hk, g·N, Tp]``.
    """

    layer: int
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor | None
    wo: torch.Tensor
    wg: torch.Tensor
    wu: torch.Tensor
    wd: torch.Tensor
    attn_in: NormMap
    attn_out: NormMap | None
    mlp_in: NormMap
    mlp_out: NormMap | None
    q_norm: NormMap | None
    k_norm: NormMap | None
    v_norm: NormMap | None
    cos: torch.Tensor
    sin: torch.Tensor
    qr: torch.Tensor
    kr: torch.Tensor
    v: torch.Tensor
    p: torch.Tensor
    #: ``scaling`` (a float) or ``scaling·(1 − tanh²(s/cap))`` (``[Hk, g·N, Tp]``).
    score_factor: torch.Tensor | float
    cg: torch.Tensor
    cu: torch.Tensor
    scalar: float
    first: int
    n_heads: int
    n_kv_heads: int
    head_dim: int

    @property
    def n_rows(self) -> int:
        return int(self.attn_in.a.shape[0])


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), -1)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


def _rope_t(y: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE's transpose: ``cos⊙y − rotate_half(sin⊙y)``."""
    return y * cos - _rotate_half(y * sin)


def _norm_primal(
    norm: NormSpec, x: torch.Tensor, work: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(y in x's dtype, x̂, r)``: the norm computed at ``work`` precision."""
    xf = x.to(work)
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.eps)
    xhat = xf * r
    y = xhat * norm.gain(device=x.device, dtype=work)
    return y.to(x.dtype), xhat, r


def _norm_map(
    norm: NormSpec, xhat: torch.Tensor, r: torch.Tensor, *, true: bool,
) -> NormMap:
    a = norm.gain(device=xhat.device, dtype=xhat.dtype) * r
    return NormMap(a=a.contiguous(), xhat=xhat.contiguous() if true else None)


def _norm_fwd(m: NormMap, dx: torch.Tensor) -> torch.Tensor:
    if m.xhat is None:
        return m.a * dx
    return m.a * (dx - m.xhat * (m.xhat * dx).mean(-1, keepdim=True))


def _norm_bwd(m: NormMap, gy: torch.Tensor) -> torch.Tensor:
    u = m.a * gy
    if m.xhat is None:
        return u
    return u - m.xhat * (m.xhat * u).mean(-1, keepdim=True)


def _linear(proj: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, proj.weight, proj.bias)


def _weight(proj: torch.nn.Linear, layer: int) -> torch.Tensor:
    weight = proj.weight
    if not weight.is_floating_point() or (
        tuple(weight.shape) != (proj.out_features, proj.in_features)
    ):
        raise UnsupportedBlockError(
            "projection weights are not a dense floating matrix (a quantized "
            "load); transport fits need the model in a floating dtype",
            layer=layer,
        )
    return weight


@torch.no_grad()
def linearize(
    spec: ModelSpec,
    ids: torch.Tensor,
    *,
    rules: Rules,
    skip_first: int,
    blocks: Sequence[int] | None = None,
) -> dict[int, LayerMap]:
    """Run the primal over one document and keep each block's map.

    ``ids`` is the document's full token row (``[T]``); its last token is
    dropped (no valid position attends to it).  ``blocks`` names the blocks
    whose maps are kept (default all); the primal runs through the highest.
    Raises ``ValueError`` when the document leaves no valid position.
    """
    if rules not in RULES:
        raise ValueError(f"rules must be one of {RULES}, got {rules!r}")
    n_valid = valid_rows(int(ids.numel()), skip_first)
    if n_valid <= 0:
        raise ValueError(
            f"a {int(ids.numel())}-token document leaves no valid position "
            f"after the first {skip_first}"
        )
    wanted = set(range(spec.n_layers)) if blocks is None else {int(b) for b in blocks}
    if not wanted:
        return {}
    top = max(wanted)
    embed = spec.embed
    device = embed.weight.device
    dtype = embed.weight.dtype
    work = torch.promote_types(dtype, torch.float32)
    tokens = ids[:-1].to(device=device, dtype=torch.long)
    T = int(tokens.numel())
    S = int(skip_first)
    Tp = -(-T // _KEY_ALIGN) * _KEY_ALIGN
    positions = torch.arange(T, device=device)
    query = positions[:, None]
    key = torch.arange(Tp, device=device)[None, :]
    causal = (key <= query) & (key < T)
    tables: dict[str | None, tuple[torch.Tensor, torch.Tensor]] = {}
    true = rules == "standard"
    h = embed(tokens)
    maps: dict[int, LayerMap] = {}
    for block in spec.blocks[: top + 1]:
        keep = block.layer in wanted
        rope_key = block.attn.rope_table
        if rope_key not in tables:
            tables[rope_key] = rotary_tables(spec, block, positions)
        cos, sin = tables[rope_key]
        h, layer_map = _block_primal(
            block, h, cos.to(work), sin.to(work), causal,
            S=S, T=T, Tp=Tp, work=work, true=true, rules=rules, keep=keep,
        )
        if layer_map is not None:
            maps[block.layer] = layer_map
    return maps


def _block_primal(
    block: BlockSpec,
    h: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    causal: torch.Tensor,
    *,
    S: int,
    T: int,
    Tp: int,
    work: torch.dtype,
    true: bool,
    rules: Rules,
    keep: bool,
) -> tuple[torch.Tensor, LayerMap | None]:
    """One block's forward from its spec, and (``keep``) its map."""
    attn = block.attn
    dtype = h.dtype
    Hq, Hk, hd = attn.n_heads, attn.n_kv_heads, attn.head_dim
    g = attn.group_size
    N = T - S
    x, xhat1, r1 = _norm_primal(block.attn_in_norm, h, work)
    q_raw = _linear(attn.q, x).view(T, Hq, hd)
    k_raw = _linear(attn.k, x).view(T, Hk, hd)
    v_raw = k_raw if attn.v is None else _linear(attn.v, x).view(T, Hk, hd)

    def head_norm(norm: NormSpec | None, raw: torch.Tensor) -> tuple[torch.Tensor, NormMap | None]:
        if norm is None:
            return raw, None
        y, xhat, r = _norm_primal(norm, raw, work)
        # Head norms keep their true derivative under both rule sets.
        return y, (_norm_map(norm, xhat[S:], r[S:], true=True) if keep else None)

    q, q_map = head_norm(attn.q_norm, q_raw)
    k, k_map = head_norm(attn.k_norm, k_raw)
    v, v_map = head_norm(attn.v_norm, v_raw)
    qr = _rope(q.to(work), cos, sin).to(dtype)
    kr = _rope(k.to(work), cos, sin).to(dtype)
    pad = Tp - T
    # [Hk, g, T, hd] queries grouped by KV head; [Hk, Tp, hd] keys and values.
    q_heads = qr.view(T, Hk, g, hd).permute(1, 2, 0, 3)
    k_heads = F.pad(kr.permute(1, 0, 2), (0, 0, 0, pad))
    v_heads = F.pad(v.permute(1, 0, 2), (0, 0, 0, pad))
    scores = (q_heads @ k_heads[:, None].mT).to(work) * attn.scaling
    capped = None
    if attn.logit_softcap is not None:
        cap = attn.logit_softcap
        scores = cap * torch.tanh(scores / cap)
        capped = scores
    visible = causal
    if attn.sliding_window is not None:
        query = torch.arange(T, device=h.device)[:, None]
        key = torch.arange(Tp, device=h.device)[None, :]
        visible = visible & (key > query - attn.sliding_window)
    scores = scores.masked_fill(~visible, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    p = probs.to(dtype)
    o = (p @ v_heads[:, None]).permute(2, 0, 1, 3).reshape(T, Hq * hd)
    a_in = _linear(attn.o, o)
    if block.attn_out_norm is not None:
        a, xhat2, r2 = _norm_primal(block.attn_out_norm, a_in, work)
    else:
        a, xhat2, r2 = a_in, None, None
    mid = h + a
    m, xhat3, r3 = _norm_primal(block.mlp_in_norm, mid, work)
    mlp = block.mlp
    gate = _linear(mlp.gate, m)
    up = _linear(mlp.up, m)
    hidden = activation(mlp.act, gate.to(work)).to(dtype) * up
    z_in = _linear(mlp.down, hidden)
    if block.mlp_out_norm is not None:
        z, xhat4, r4 = _norm_primal(block.mlp_out_norm, z_in, work)
    else:
        z, xhat4, r4 = z_in, None, None
    scalar = block.scale()
    out = mid + z
    if block.scalar is not None:
        out = out * scalar
    if not keep:
        return out, None

    if capped is None:
        score_factor: torch.Tensor | float = float(attn.scaling)
    else:
        cap = float(attn.logit_softcap or 1.0)
        rows = capped[:, :, S:].reshape(Hk, g * N, Tp)
        score_factor = (attn.scaling * (1.0 - (rows / cap).square())).contiguous()
    gate_v = gate[S:].to(work)
    up_v = up[S:].to(work)
    if rules == "relp":
        cg = HALF_RULE_BETA * activation_factor(mlp.act, gate_v) * up_v
        cu = HALF_RULE_BETA * activation(mlp.act, gate_v)
    else:
        cg = activation_derivative(mlp.act, gate_v) * up_v
        cu = activation(mlp.act, gate_v)
    layer = block.layer
    return out, LayerMap(
        layer=layer,
        wq=_weight(attn.q, layer),
        wk=_weight(attn.k, layer),
        wv=None if attn.v is None else _weight(attn.v, layer),
        wo=_weight(attn.o, layer),
        wg=_weight(mlp.gate, layer),
        wu=_weight(mlp.up, layer),
        wd=_weight(mlp.down, layer),
        attn_in=_norm_map(block.attn_in_norm, xhat1[S:], r1[S:], true=true),
        attn_out=(
            None if block.attn_out_norm is None or xhat2 is None or r2 is None
            else _norm_map(block.attn_out_norm, xhat2[S:], r2[S:], true=true)
        ),
        mlp_in=_norm_map(block.mlp_in_norm, xhat3[S:], r3[S:], true=true),
        mlp_out=(
            None if block.mlp_out_norm is None or xhat4 is None or r4 is None
            else _norm_map(block.mlp_out_norm, xhat4[S:], r4[S:], true=true)
        ),
        q_norm=q_map,
        k_norm=k_map,
        v_norm=v_map,
        cos=cos[S:].contiguous(),
        sin=sin[S:].contiguous(),
        qr=q_heads[:, :, S:].reshape(Hk, g * N, hd).contiguous(),
        kr=k_heads.contiguous(),
        v=v_heads.contiguous(),
        p=p[:, :, S:].reshape(Hk, g * N, Tp).contiguous(),
        score_factor=score_factor,
        cg=cg.to(dtype).contiguous(),
        cu=cu.to(dtype).contiguous(),
        scalar=scalar,
        first=S,
        n_heads=Hq,
        n_kv_heads=Hk,
        head_dim=hd,
    )


# -- the maps -----------------------------------------------------------------------

def jvp(m: LayerMap, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Push tangents ``X`` (``[b, N, d]``) at a block's input through its map.

    Returns ``(out, mid)``: the tangents at the block's output and at its
    ``mid`` residual, at ``X``'s precision.
    """
    dt = m.wq.dtype
    work = X.dtype
    b, N, _ = X.shape
    Hq, Hk, hd = m.n_heads, m.n_kv_heads, m.head_dim
    g = Hq // Hk
    keys = slice(m.first, m.first + N)
    da = _norm_fwd(m.attn_in, X).to(dt)
    dq = F.linear(da, m.wq).view(b, N, Hq, hd).to(work)
    dk = F.linear(da, m.wk).view(b, N, Hk, hd).to(work)
    dv = dk if m.wv is None else F.linear(da, m.wv).view(b, N, Hk, hd).to(work)
    if m.q_norm is not None:
        dq = _norm_fwd(m.q_norm, dq)
    if m.k_norm is not None:
        dk = _norm_fwd(m.k_norm, dk)
    if m.v_norm is not None:
        dv = _norm_fwd(m.v_norm, dv)
    dq = _rope(dq, m.cos, m.sin)
    dk = _rope(dk, m.cos, m.sin)
    dq = dq.to(dt).view(b, N, Hk, g, hd).permute(0, 2, 3, 1, 4).reshape(b, Hk, g * N, hd)
    dk = dk.to(dt).transpose(1, 2)
    dv = dv.to(dt).transpose(1, 2)
    ds = (dq @ m.kr.mT).to(work)
    ds[..., keys] += (m.qr @ dk.mT).to(work)
    ds = ds * m.score_factor
    dp = (m.p * (ds - (m.p * ds).sum(-1, keepdim=True))).to(dt)
    do = dp @ m.v + m.p[..., keys] @ dv
    do = do.view(b, Hk, g, N, hd).permute(0, 3, 1, 2, 4).reshape(b, N, Hq * hd)
    attn = F.linear(do, m.wo).to(work)
    if m.attn_out is not None:
        attn = _norm_fwd(m.attn_out, attn)
    mid = X + attn
    dm = _norm_fwd(m.mlp_in, mid).to(dt)
    hidden = m.cg * F.linear(dm, m.wg) + m.cu * F.linear(dm, m.wu)
    z = F.linear(hidden, m.wd).to(work)
    if m.mlp_out is not None:
        z = _norm_fwd(m.mlp_out, z)
    out = mid + z
    if m.scalar != 1.0:
        out = out * m.scalar
    return out, mid


def vjp(
    m: LayerMap, G: torch.Tensor, mid_seed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull cotangents ``G`` (``[b, N, d]``) at a block's output back through its map.

    ``mid_seed`` joins the cotangents at the block's ``mid`` residual.
    Returns ``(inp, mid)``: the cotangents at the block's input and at its
    ``mid`` residual, at ``G``'s precision.
    """
    dt = m.wq.dtype
    work = G.dtype
    b, N, _ = G.shape
    Hq, Hk, hd = m.n_heads, m.n_kv_heads, m.head_dim
    g = Hq // Hk
    keys = slice(m.first, m.first + N)
    if m.scalar != 1.0:
        G = G * m.scalar
    gz = G if m.mlp_out is None else _norm_bwd(m.mlp_out, G)
    gh = gz.to(dt) @ m.wd
    gm = (m.cg * gh) @ m.wg + (m.cu * gh) @ m.wu
    mid = G + _norm_bwd(m.mlp_in, gm.to(work))
    if mid_seed is not None:
        mid = mid + mid_seed
    ga = mid if m.attn_out is None else _norm_bwd(m.attn_out, mid)
    go = (ga.to(dt) @ m.wo).view(b, N, Hk, g, hd).permute(0, 2, 3, 1, 4).reshape(b, Hk, g * N, hd)
    gp = (go @ m.v.mT).to(work)
    gs = ((m.p * (gp - (m.p * gp).sum(-1, keepdim=True))) * m.score_factor).to(dt)
    gq = (gs @ m.kr).view(b, Hk, g, N, hd).permute(0, 3, 1, 2, 4).reshape(b, N, Hq, hd).to(work)
    gk = (gs[..., keys].mT @ m.qr).transpose(1, 2).to(work)
    gv = (m.p[..., keys].mT @ go).transpose(1, 2).to(work)
    gq = _rope_t(gq, m.cos, m.sin)
    gk = _rope_t(gk, m.cos, m.sin)
    if m.q_norm is not None:
        gq = _norm_bwd(m.q_norm, gq)
    if m.k_norm is not None:
        gk = _norm_bwd(m.k_norm, gk)
    if m.v_norm is not None:
        gv = _norm_bwd(m.v_norm, gv)
    gx = gq.to(dt).reshape(b, N, Hq * hd) @ m.wq
    if m.wv is None:
        # The value is the raw key: both cotangents reach the k projection.
        gx = gx + (gk + gv).to(dt).reshape(b, N, Hk * hd) @ m.wk
    else:
        gx = (
            gx + gk.to(dt).reshape(b, N, Hk * hd) @ m.wk
            + gv.to(dt).reshape(b, N, Hk * hd) @ m.wv
        )
    inp = mid + _norm_bwd(m.attn_in, gx.to(work))
    return inp, mid


@functools.cache
def _compiled(step: Callable[..., tuple[torch.Tensor, torch.Tensor]]) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    return torch.compile(step, dynamic=False)


def step_for(
    step: Callable[..., tuple[torch.Tensor, torch.Tensor]], device: torch.device,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """``step`` as it runs on ``device``: compiled with fixed shapes on CUDA
    (neuron-lens measured 1.6× on an H100: unfused, half the time went
    between the products), eager elsewhere.  Past dynamo's recompile limit,
    rarer document lengths run eagerly."""
    return _compiled(step) if device.type == "cuda" else step


# -- sweeps -------------------------------------------------------------------------

def seed_block(
    b: int, n_rows: int, d: int, start: int, *,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """``[b, N, d]``: basis direction ``start + i`` at every row, in copy ``i``."""
    seeds = torch.zeros(b, n_rows, d, device=device, dtype=dtype)
    index = torch.arange(b, device=device)
    seeds[index, :, start + index] = 1
    return seeds


def reverse_sweep(
    maps: Mapping[int, LayerMap],
    seeds: torch.Tensor,
    *,
    n_layers: int,
    out_layers: Sequence[int],
    mid_layers: Sequence[int],
    emit: Callable[[str, int, torch.Tensor], None],
    step: Callable[..., tuple[torch.Tensor, torch.Tensor]] = vjp,
) -> None:
    """Output kinds: pull ``seeds`` (cotangents at the last block's output)
    down, emitting each requested site's position mean (``[b, d]``: rows of
    ``J_L``).  ``out`` layer ``L`` is read below block ``L + 1``."""
    out_set = {int(layer) for layer in out_layers}
    mid_set = {int(layer) for layer in mid_layers}
    if not (out_set or mid_set):
        return
    lowest = min([layer + 1 for layer in out_set] + list(mid_set))
    G = seeds
    for block in range(n_layers - 1, lowest - 1, -1):
        G, mid = step(maps[block], G)
        if block in mid_set:
            emit("mid", block, mid.mean(1))
        if block - 1 in out_set:
            emit("out", block - 1, G.mean(1))


def forward_sweep(
    maps: Mapping[int, LayerMap],
    seeds: torch.Tensor,
    *,
    out_layers: Sequence[int],
    mid_layers: Sequence[int],
    emit: Callable[[str, int, torch.Tensor], None],
    step: Callable[..., tuple[torch.Tensor, torch.Tensor]] = jvp,
) -> None:
    """Input kinds: push ``seeds`` (tangents at the embedding) up, emitting
    each requested site's position mean (``[b, d]``: columns of ``K_L``)."""
    out_set = {int(layer) for layer in out_layers}
    mid_set = {int(layer) for layer in mid_layers}
    if not (out_set or mid_set):
        return
    highest = max(out_set | mid_set)
    X = seeds
    for block in range(highest + 1):
        X, mid = step(maps[block], X)
        if block in mid_set:
            emit("mid", block, mid.mean(1))
        if block in out_set:
            emit("out", block, X.mean(1))


def blocks_needed(
    direction: str, n_layers: int, layers: Mapping[str, Sequence[int]],
) -> list[int]:
    """The blocks whose maps a sweep over ``layers`` (``{site: layers}``) reads."""
    out = [int(layer) for layer in layers.get("out", ())]
    mid = [int(layer) for layer in layers.get("mid", ())]
    if not (out or mid):
        return []
    if direction == "output":
        lowest = min([layer + 1 for layer in out] + mid)
        return list(range(lowest, n_layers))
    return list(range(max(out + mid) + 1))


def document_estimate(
    spec: ModelSpec,
    ids: torch.Tensor,
    *,
    kind: str,
    layers: Mapping[str, Sequence[int]],
    skip_first: int,
    dim_batch: int = 256,
) -> dict[str, dict[int, torch.Tensor]]:
    """One document's lens estimate at ``layers`` (``{site: layers}``), in
    the maps' working precision, oriented as :class:`JacobianLens` holds
    them: ``J_L`` ``[d_final, d]`` for output kinds, ``K_L`` ``[d, d_emb]``
    for input kinds.  The reference unit of the fit (``fit_lens`` sums these
    over documents); tests compare it to brute-force autograd.
    """
    from drowse.lens.key import LensKey

    key = LensKey(kind)
    rules = rules_for(kind)
    direction = key.side
    maps = linearize(
        spec, ids, rules=rules, skip_first=skip_first,
        blocks=blocks_needed(direction, spec.n_layers, layers),
    )
    if not maps:
        return {}
    any_map = next(iter(maps.values()))
    device = any_map.wq.device
    work = torch.promote_types(any_map.wq.dtype, torch.float32)
    d = spec.d_model
    N = any_map.n_rows
    out: dict[str, dict[int, torch.Tensor]] = {
        site: {int(layer): torch.zeros(d, d, device=device, dtype=work) for layer in wanted}
        for site, wanted in layers.items()
    }
    for start in range(0, d, dim_batch):
        b = min(dim_batch, d - start)
        seeds = seed_block(b, N, d, start, device=device, dtype=work)

        def emit(site: str, layer: int, est: torch.Tensor, start: int = start, b: int = b) -> None:
            target = out.get(site, {}).get(layer)
            if target is None:
                return
            if direction == "output":
                target[start:start + b] = est
            else:
                target[:, start:start + b] = est.T

        if direction == "output":
            reverse_sweep(
                maps, seeds, n_layers=spec.n_layers,
                out_layers=list(layers.get("out", ())), mid_layers=list(layers.get("mid", ())),
                emit=emit,
            )
        else:
            forward_sweep(
                maps, seeds,
                out_layers=list(layers.get("out", ())), mid_layers=list(layers.get("mid", ())),
                emit=emit,
            )
    return out


__all__ = [
    "RULES",
    "LayerMap",
    "NormMap",
    "Rules",
    "blocks_needed",
    "document_estimate",
    "forward_sweep",
    "jvp",
    "linearize",
    "reverse_sweep",
    "rules_for",
    "seed_block",
    "step_for",
    "valid_rows",
    "vjp",
]
