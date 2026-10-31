"""Unit directions: what a neuron or a head reads and writes, from the weights.

Pure functions of a :class:`~drowse.model.blocks.BlockSpec`, in fp32 on CPU,
with the definitions neuron-lens catalogues (its ``design.md`` "The units"):

- **Neuron** ``n`` of block ``L`` is coordinate ``n`` of the down
  projection's input, ``act = act(a·x̂)(b·x̂)`` on the normalized MLP input
  ``x̂``.  Its **write** is what one unit of ``act`` adds to the residual at
  the block's output: ``s · γ_out ⊙ W_down[:, n]``, with ``γ_out`` the MLP
  output norm's gain where the block has one (Gemma's sandwich; its scale at
  a position is held constant, the LN rule) and ``s`` the per-layer scalar.
  Its read rows are ``a = γ_in ⊙ W_gate[n]`` and ``b = γ_in ⊙ W_up[n]``
  (``γ_in`` the MLP input norm's gain).  ``act`` takes both signs, so a
  neuron has two **poles**: ``+`` writes ``+w`` and reads along ``â + b̂``,
  ``−`` writes ``−w`` and reads along ``â − b̂`` (unit rows ``â``, ``b̂``; the
  sum is not renormalized).
- **Head** ``h`` of block ``L`` reads the attention input through its query,
  key and value rows (``W_Q``/``W_K``/``W_V`` rows of its query head and of
  the KV head it shares, times the attention input norm's gain ``γ_attn``;
  a block without ``v`` reads keys as values) and writes
  ``γ_attn_out ⊙ W_O[:, h·hd:(h+1)·hd] z`` into the residual at ``mid``
  (before the MLP, so the per-layer scalar does not apply yet).

The lens rule places them: writes of neurons and reads of heads sit at
``out`` (a head reads ``L − 1``'s output), reads of neurons and writes of
heads at ``mid``.  Indices are checked against the block; whether a
``NeuronRef``/``HeadRef`` names a block of the model is the caller's check.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch

from drowse.errors import RequestError
from drowse.model.blocks import BlockSpec

Pole = Literal["+", "-"]


class UnitIndexError(RequestError):
    """A neuron or head index outside its block (400)."""


def _rows(weight: torch.Tensor, rows: slice) -> torch.Tensor:
    """Rows of a (possibly device, low-precision) weight as fp32 CPU, sliced first."""
    return weight.detach()[rows].to(device="cpu", dtype=torch.float32)


def _check(kind: str, index: int, count: int, spec: BlockSpec) -> int:
    index = int(index)
    if not 0 <= index < count:
        raise UnitIndexError(
            f"{kind} {index} is outside block {spec.layer} "
            f"({count} {kind}s, 0..{count - 1})"
        )
    return index


def _pole_sign(pole: Pole) -> float:
    if pole == "+":
        return 1.0
    if pole == "-":
        return -1.0
    raise ValueError(f"pole must be '+' or '-', got {pole!r}")


# -- neurons ----------------------------------------------------------------------

def _writes(spec: BlockSpec, columns: slice) -> torch.Tensor:
    write = _rows(spec.mlp.down.weight.T, columns).T
    if spec.mlp_out_norm is not None:
        write = spec.mlp_out_norm.gain(device="cpu")[:, None] * write
    return write * spec.scale()


def neuron_writes(spec: BlockSpec) -> torch.Tensor:
    """Every neuron's ``+``-pole write, ``[d_model, d_ff]`` (column ``n``)."""
    return _writes(spec, slice(None))


def neuron_write(spec: BlockSpec, n: int) -> torch.Tensor:
    """Neuron ``n``'s ``+``-pole write ``s · γ_out ⊙ W_down[:, n]``, ``[d_model]``."""
    n = _check("neuron", n, spec.d_ff, spec)
    return _writes(spec, slice(n, n + 1))[:, 0]


def _read_rows(spec: BlockSpec, rows: slice) -> tuple[torch.Tensor, torch.Tensor]:
    gain = spec.mlp_in_norm.gain(device="cpu")
    a = _rows(spec.mlp.gate.weight, rows) * gain
    b = _rows(spec.mlp.up.weight, rows) * gain
    return (
        a / a.norm(dim=1, keepdim=True).clamp_min(1e-12),
        b / b.norm(dim=1, keepdim=True).clamp_min(1e-12),
    )


def neuron_read_rows(spec: BlockSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """``(â, b̂)``: every neuron's unit gate and up rows times ``γ_in``, ``[d_ff, d_model]``."""
    return _read_rows(spec, slice(None))


def neuron_reads(spec: BlockSpec, pole: Pole) -> torch.Tensor:
    """Every neuron's read direction ``â ± b̂`` for ``pole``, ``[d_ff, d_model]``."""
    sign = _pole_sign(pole)
    a, b = neuron_read_rows(spec)
    return a + sign * b


def neuron_read(spec: BlockSpec, n: int, pole: Pole) -> torch.Tensor:
    """Neuron ``n``'s read direction for ``pole``: ``â + b̂`` or ``â − b̂``, ``[d_model]``."""
    n = _check("neuron", n, spec.d_ff, spec)
    sign = _pole_sign(pole)
    a, b = _read_rows(spec, slice(n, n + 1))
    return a[0] + sign * b[0]


# -- heads ------------------------------------------------------------------------

class HeadReads(NamedTuple):
    """One query head's read rows as the residual sees them, ``[head_dim, d_model]``."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


def head_write_matrix(spec: BlockSpec, h: int) -> torch.Tensor:
    """Head ``h``'s write ``γ_attn_out ⊙ W_O[:, h·hd:(h+1)·hd]``, ``[d_model, head_dim]``.

    ``W_O^h z`` for the head's value-weighted output ``z``, as it lands at
    ``mid``.
    """
    attn = spec.attn
    h = _check("head", h, attn.n_heads, spec)
    hd = attn.head_dim
    write = _rows(attn.o.weight.T, slice(h * hd, (h + 1) * hd)).T
    if spec.attn_out_norm is not None:
        write = spec.attn_out_norm.gain(device="cpu")[:, None] * write
    return write.contiguous()


def head_read_rows(spec: BlockSpec, h: int) -> HeadReads:
    """Head ``h``'s query, key and value rows times ``γ_attn`` (its KV head's for k, v)."""
    attn = spec.attn
    h = _check("head", h, attn.n_heads, spec)
    hd = attn.head_dim
    kv = attn.kv_head(h)
    gain = spec.attn_in_norm.gain(device="cpu")
    rows = slice(kv * hd, (kv + 1) * hd)
    key = _rows(attn.k.weight, rows) * gain
    value = key.clone() if attn.v is None else _rows(attn.v.weight, rows) * gain
    return HeadReads(
        query=_rows(attn.q.weight, slice(h * hd, (h + 1) * hd)) * gain,
        key=key,
        value=value,
    )


__all__ = [
    "HeadReads",
    "Pole",
    "UnitIndexError",
    "head_read_rows",
    "head_write_matrix",
    "neuron_read",
    "neuron_read_rows",
    "neuron_reads",
    "neuron_write",
    "neuron_writes",
]
