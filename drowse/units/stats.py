"""Unit statistics from the neutral pass: what a neuron or head usually does.

Two numbers per unit, both measured on the neutral corpus (the same forward
that fills the neutral activation cache, :mod:`drowse.fit.neutral`):

- the **mean** of its activation, the reference a mean-ablation substitutes:
  a neuron's ``act`` (coordinate ``n`` of the down projection's input), a
  head's ``z_h`` (its ``head_dim`` slice of the output projection's input);
- its **scale**, the unit a probe reads in.  neuron-lens's ``act_top``: per
  document, the unit's largest firing; across documents, the median of the
  :data:`UNIT_TOP_DOCS` largest of those.  A neuron's ``act`` takes both signs,
  so each pole has its own (``±act``, clamped at 0); a head's firing is its
  write norm ``‖W_O^h z_h‖`` as it reaches the residual (Gemma's sandwich
  norm included: its gain and its per-position scale, the LN rule).

A reading divides by a **reading scale** (:class:`UnitScales`) from one of
two sources:

- ``"catalog"``: the bound neuron-lens catalog's ``act_top`` per pole (a
  head's ``out_top``), the same statistic over its activation corpora (15.4M
  positions).  A catalogued model reads against it, so 1.0 is the pole's
  typical high firing in text.
- ``"neutral"``: the ``act_top`` measured here, over the neutral statements.
  They are too few for a top-24-of-documents statistic (96 short statements
  against neuron-lens's 60k documents): on Qwen3-8B it sits a median 12x
  below the catalog's, and a sparse pole that never fires on the statements
  gets 0.  A reading over it is relative to the neutral text.

Either way the divisor is a **soft scale**, not the bare ``act_top``:
:func:`soft_scale` blends the unit's own scale with a floor, ``sqrt(s² +
(c · m)²)``, ``c`` = :data:`SCALE_FLOOR_SHARE` and ``m`` the floor's base
(:func:`reading_scales`).  Over the catalog ``m`` is the layer's typical
scale for the unit's kind (:func:`typical_scale`; the catalog's statistics
are robust, so a quiet layer is truly quiet: Qwen3-8B's L1 and L2 medians are
0.009 and 0.014).  Over the neutral statistics ``m`` is the larger of the
layer's and the model's typical scale, a floor that cannot collapse with the
layer: Qwen3-8B's neutral L2 median is 0.0024 with 4,856 of 24,576 poles at
0, and a quarter of it read an ordinary firing in the hundreds (``L2n7950+``
at 849, 0.59x its catalog ``act_top``).  It is soft: no unit is cut, a pole
with no scale at all reads over ``c · m``, and a unit at the median reads 3%
lower than over its own scale.  ``activation_over_scale`` divides by it
everywhere (probes, gates, token readouts, unit activity, the anatomy's unit
rows), so one unit shows one number in every view, and every view names the
source.

Positions follow neuron-lens: position 0 (the attention sink, where
activations are massive) is never counted, and a head's scale starts at
:data:`HEAD_SCALE_FROM` (a head with few keys attends sharply).  Means count
every position from :data:`UNIT_POSITIONS_FROM`.

Everything streams: per chunk the accumulator keeps per-layer sums and the
per-document maxima, and merges them into a running top-``k`` buffer, so
nothing of size ``d_ff × n_tokens`` outlives a forward.  Accumulation is fp32
on the model's device; :meth:`UnitStatsAccumulator.finish` returns CPU fp32.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from drowse.model.blocks import BlockSpec, ModelSpec, NormSpec
from drowse.model.units import head_write_matrix

#: Documents whose maxima the scale takes the median of (neuron-lens's
#: ``TOP_DOCS``).
UNIT_TOP_DOCS = 24
#: First position a unit's mean and a neuron's scale count (0 is the sink).
UNIT_POSITIONS_FROM = 1
#: First position a head's scale counts (neuron-lens: where the lenses start).
HEAD_SCALE_FROM = 16

#: Per-layer tensor names, as stored in the neutral cache shards.
NEURON_MEAN = "neuron_mean"
NEURON_SCALE = "neuron_scale"
HEAD_MEAN = "head_mean"
HEAD_SCALE = "head_scale"
UNIT_TENSOR_NAMES = (NEURON_MEAN, NEURON_SCALE, HEAD_MEAN, HEAD_SCALE)

#: Where a reading scale comes from (:class:`UnitScales`).
ScaleSource = Literal["catalog", "neutral"]

#: The soft floor's share of its base (:func:`soft_scale`).
#: Measured on SmolLM2-360M (96 neutral documents): the 0.1%, 1% and 5%
#: quantiles of a neuron pole's ``act_top`` over its layer's median are 0.20,
#: 0.33 and 0.47 (heads 0.13, 0.24, 0.45); 0.24% of neuron poles and 1.25% of
#: heads sit below a quarter of the median.  A quarter leaves a typical unit
#: within 3% of its own scale (11% at half the median) and holds the far tail
#: (``L31n2394+`` at 0.024 of the median) to about four times the floor's
#: reading instead of forty.
SCALE_FLOOR_SHARE = 0.25


def typical_scale(scale: torch.Tensor) -> torch.Tensor:
    """A layer's typical unit scale: the median of ``scale``'s positive
    entries (0-dim fp32; 0 when none is positive)."""
    flat = scale.reshape(-1).float()
    positive = flat[flat > 0]
    if positive.numel() == 0:
        return torch.zeros((), dtype=torch.float32)
    return positive.median()


def soft_scale(scale: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """``sqrt(s² + (c · m)²)``: a unit's own scale ``s`` blended with the floor
    ``c · m`` (``m`` the floor's base, :func:`reading_scales`; ``c`` =
    :data:`SCALE_FLOOR_SHARE`), the scale every ``activation_over_scale``
    divides by."""
    floor = SCALE_FLOOR_SHARE * base.float()
    return torch.sqrt(scale.float().pow(2) + floor.pow(2))


@dataclass(frozen=True, eq=False)
class LayerScales:
    """One block's reading scales, fp32 on CPU.

    ``neuron_top`` (``[2, d_ff]``: row 0 the ``+`` pole, row 1 the ``−``)
    and ``head_top`` (``[n_heads]``) are the source's own ``act_top``;
    ``neuron`` and ``head`` are what a reading divides by (:func:`soft_scale`
    over the floor's base).
    """

    neuron_top: torch.Tensor
    head_top: torch.Tensor
    neuron: torch.Tensor
    head: torch.Tensor


@dataclass(frozen=True, eq=False)
class UnitScales:
    """Every block's :class:`LayerScales` and where they come from.

    ``source`` is ``"catalog"`` (the bound neuron-lens catalog's ``act_top``)
    or ``"neutral"`` (drowse's own, over the neutral statements, so a reading
    is relative to the neutral text); ``detail`` says which (the catalog's
    repository and commit, or the statement count).
    """

    source: ScaleSource
    detail: str
    layers: Mapping[int, LayerScales]

    def layer(self, layer: int) -> LayerScales:
        return self.layers[int(layer)]

    def describe(self) -> dict[str, str]:
        """``{"source", "detail"}``: the wire form every view names."""
        return {"source": self.source, "detail": self.detail}


def reading_scales(
    neuron_top: Mapping[int, torch.Tensor],
    head_top: Mapping[int, torch.Tensor],
    *,
    source: ScaleSource,
    detail: str,
) -> UnitScales:
    """The reading scales over per-layer ``act_top`` tables (``[2, d_ff]``
    per block for the neurons, ``[n_heads]`` for the heads).

    The floor's base ``m`` per layer and kind: over the catalog, the layer's
    :func:`typical_scale`; over the neutral statistics, the larger of the
    layer's and the model's (every layer's poles, or heads, together), so a
    layer whose neutral statistics collapsed still reads against a floor.
    """
    layers = sorted(int(layer) for layer in neuron_top)
    neurons = {L: neuron_top[L].float() for L in layers}
    heads = {L: head_top[L].float() for L in layers}
    zero = torch.zeros((), dtype=torch.float32)
    model_neuron = model_head = zero
    if source == "neutral":
        model_neuron = typical_scale(torch.cat([t.reshape(-1) for t in neurons.values()]))
        model_head = typical_scale(torch.cat([t.reshape(-1) for t in heads.values()]))
    out: dict[int, LayerScales] = {}
    for L in layers:
        n_base = torch.maximum(typical_scale(neurons[L]), model_neuron)
        h_base = torch.maximum(typical_scale(heads[L]), model_head)
        out[L] = LayerScales(
            neuron_top=neurons[L],
            head_top=heads[L],
            neuron=soft_scale(neurons[L], n_base),
            head=soft_scale(heads[L], h_base),
        )
    return UnitScales(source=source, detail=detail, layers=out)


@dataclass(frozen=True, eq=False)
class LayerUnitStats:
    """One block's unit statistics, fp32 on CPU.

    ``neuron_scale`` row 0 is the ``+`` pole's ``act_top``, row 1 the ``−``
    pole's (both ``≥ 0``).  A reading divides by a :class:`UnitScales`
    (:attr:`UnitStats.reading_scales` over these, or a bound catalog's).
    """

    neuron_mean: torch.Tensor   # [d_ff]
    neuron_scale: torch.Tensor  # [2, d_ff]
    head_mean: torch.Tensor     # [n_heads, head_dim]
    head_scale: torch.Tensor    # [n_heads]

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            NEURON_MEAN: self.neuron_mean,
            NEURON_SCALE: self.neuron_scale,
            HEAD_MEAN: self.head_mean,
            HEAD_SCALE: self.head_scale,
        }

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, torch.Tensor]) -> "LayerUnitStats":
        return cls(
            neuron_mean=tensors[NEURON_MEAN],
            neuron_scale=tensors[NEURON_SCALE],
            head_mean=tensors[HEAD_MEAN],
            head_scale=tensors[HEAD_SCALE],
        )



@dataclass(frozen=True, eq=False)
class UnitStats:
    """Every block's :class:`LayerUnitStats`, and how many documents fed them."""

    layers: Mapping[int, LayerUnitStats]
    n_documents: int

    def layer(self, layer: int) -> LayerUnitStats:
        return self.layers[int(layer)]

    def tensors(self) -> dict[int, dict[str, torch.Tensor]]:
        """``{layer: {name: tensor}}``, the neutral cache's storage form."""
        return {layer: stats.tensors() for layer, stats in self.layers.items()}

    @classmethod
    def from_tensors(
        cls, tensors: Mapping[int, Mapping[str, torch.Tensor]], *, n_documents: int,
    ) -> "UnitStats":
        return cls(
            layers={
                int(layer): LayerUnitStats.from_tensors(row)
                for layer, row in tensors.items()
            },
            n_documents=int(n_documents),
        )

    @functools.cached_property
    def reading_scales(self) -> UnitScales:
        """The ``"neutral"`` reading scales: these ``act_top`` values, floored
        against the larger of the layer's and the model's typical scale."""
        return reading_scales(
            {layer: row.neuron_scale for layer, row in self.layers.items()},
            {layer: row.head_scale for layer, row in self.layers.items()},
            source="neutral",
            detail=f"{self.n_documents} neutral statements",
        )


def top_median(top: torch.Tensor, seen: int) -> torch.Tensor:
    """``act_top`` from a descending top-``k`` buffer ``[k, …]``.

    The median (the lower one, as ``torch.median``) of the ``min(k, seen)``
    entries that came from real documents, clamped at 0; 0 where no document
    was seen.
    """
    k = min(int(top.shape[0]), int(seen))
    if k <= 0:
        return torch.zeros(top.shape[1:], dtype=torch.float32)
    return top[:k].float().median(dim=0).values.clamp_min(0.0)


class _LayerAccumulator:
    """Running sums and top-``k`` buffers for one block, on the model device."""

    def __init__(self, block: BlockSpec, device: torch.device, k: int) -> None:
        attn = block.attn
        self.layer = block.layer
        self.n_heads = attn.n_heads
        self.head_dim = attn.head_dim
        d_ff = block.d_ff
        self.neuron_sum = torch.zeros(d_ff, dtype=torch.float32, device=device)
        self.neuron_top = torch.full(
            (k, 2, d_ff), float("-inf"), dtype=torch.float32, device=device,
        )
        self.head_sum = torch.zeros(
            attn.n_heads, attn.head_dim, dtype=torch.float32, device=device,
        )
        self.head_top = torch.full(
            (k, attn.n_heads), float("-inf"), dtype=torch.float32, device=device,
        )
        # |W_O^h z|² = zᵀ G_h z, the write matrix with the sandwich gain.
        gram = torch.stack([
            (lambda m: m.T @ m)(head_write_matrix(block, h))
            for h in range(attn.n_heads)
        ])
        self.gram = gram.to(device=device, dtype=torch.float32)
        self.out_norm: NormSpec | None = block.attn_out_norm
        # This chunk's contributions, merged by ``commit``.
        self.pending_neuron: tuple[torch.Tensor, torch.Tensor] | None = None
        self.pending_head: tuple[torch.Tensor, torch.Tensor] | None = None


class UnitStatsAccumulator:
    """Streams unit statistics off a batched, right-padded forward.

    Installs one forward pre-hook per down projection (``act`` is its input)
    and one forward hook per output projection (``z`` is its input; the
    sandwich scale reads its output).  Each forward is framed by
    :meth:`begin_chunk` (the rows' token lengths) and :meth:`commit_chunk`
    (merge) or :meth:`discard_chunk` (an out-of-memory retry re-runs the rows,
    so nothing of the failed attempt may count).  ``failure`` is set, and the
    hooks go quiet, when a hook raises anything but an out-of-memory error;
    an out-of-memory error propagates so the capture can halve its batch, and
    at batch size 1 the capture calls :meth:`abandon` and re-runs the rows
    without the hooks: the neutral pass must never fail because of units.
    """

    def __init__(
        self, spec: ModelSpec, *, device: torch.device, k: int = UNIT_TOP_DOCS,
    ) -> None:
        self._k = int(k)
        self._device = device
        self._layers = [_LayerAccumulator(block, device, self._k) for block in spec.blocks]
        self._blocks = list(spec.blocks)
        self._handles: list[Any] = []
        self._mask: torch.Tensor | None = None       # [B, T] mean / neuron-scale positions
        self._head_mask: torch.Tensor | None = None  # [B, T] head-scale positions
        self._chunk_positions = 0
        self._chunk_docs = 0
        self._chunk_head_docs = 0
        self.positions = 0
        self.documents = 0
        self.head_documents = 0
        self.failure: str | None = None

    # ------------------------------------------------------------- hooks

    def install(self) -> None:
        """Register the hook pair on every block (idempotent)."""
        if self._handles:
            return
        try:
            for block, acc in zip(self._blocks, self._layers, strict=True):
                self._handles.append(
                    block.mlp.down.register_forward_pre_hook(self._neuron_hook(acc)),
                )
                self._handles.append(
                    block.attn.o.register_forward_hook(self._head_hook(acc)),
                )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "UnitStatsAccumulator":
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _fail(self, exc: Exception) -> None:
        from drowse.errors import is_out_of_memory_error

        if is_out_of_memory_error(exc):
            raise exc
        if self.failure is None:
            self.failure = f"{type(exc).__name__}: {exc}"

    def _neuron_hook(self, acc: _LayerAccumulator) -> Any:
        def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            if self.failure is not None or self._mask is None:
                return
            try:
                x = args[0]
                mask = self._mask[..., None]
                total = (x * mask.to(x.dtype)).sum(dim=(0, 1), dtype=torch.float32)
                hi = x.masked_fill(~mask, float("-inf")).amax(dim=1).float()
                lo = x.masked_fill(~mask, float("inf")).amin(dim=1).float().neg_()
                acc.pending_neuron = (total, torch.stack([hi, lo], dim=1))
            except Exception as exc:  # noqa: BLE001 — units never fail the pass
                self._fail(exc)
        return hook

    def _head_hook(self, acc: _LayerAccumulator) -> Any:
        def hook(_module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
            if self.failure is not None or self._mask is None:
                return
            try:
                z = args[0]
                b, t = int(z.shape[0]), int(z.shape[1])
                zh = z.float().reshape(b, t, acc.n_heads, acc.head_dim)
                mask = self._mask.float()
                total = torch.einsum("bthi,bt->hi", zh, mask)
                norm = torch.einsum(
                    "bthi,hij,bthj->bth", zh, acc.gram, zh,
                ).clamp_min(0.0).sqrt()
                if acc.out_norm is not None:
                    out = output if isinstance(output, torch.Tensor) else output[0]
                    r = torch.rsqrt(
                        out.float().pow(2).mean(-1) + acc.out_norm.eps,
                    )
                    norm = norm * r[..., None]
                assert self._head_mask is not None
                peak = norm.masked_fill(
                    ~self._head_mask[..., None], float("-inf"),
                ).amax(dim=1)
                acc.pending_head = (total, peak)
            except Exception as exc:  # noqa: BLE001 — units never fail the pass
                self._fail(exc)
        return hook

    # ------------------------------------------------------------ chunks

    def begin_chunk(self, lengths: Sequence[int]) -> None:
        """Frame the next forward: its rows' real token counts (right-padded)."""
        lengths = [int(n) for n in lengths]
        t = max(lengths)
        positions = torch.arange(t)[None, :]
        lens = torch.tensor(lengths)[:, None]
        real = positions < lens
        mask = real & (positions >= UNIT_POSITIONS_FROM)
        head_mask = real & (positions >= HEAD_SCALE_FROM)
        self._mask = mask.to(self._device)
        self._head_mask = head_mask.to(self._device)
        self._chunk_positions = int(mask.sum())
        self._chunk_docs = int(mask.any(dim=1).sum())
        self._chunk_head_docs = int(head_mask.any(dim=1).sum())
        for acc in self._layers:
            acc.pending_neuron = None
            acc.pending_head = None

    def commit_chunk(self) -> None:
        """Merge the framed forward into the running statistics."""
        self._mask = self._head_mask = None
        if self.failure is not None:
            return
        k = self._k
        with torch.no_grad():
            for acc in self._layers:
                if acc.pending_neuron is None or acc.pending_head is None:
                    self.failure = (
                        f"block {acc.layer}'s unit hooks did not fire on a "
                        "neutral forward"
                    )
                    return
                total, peaks = acc.pending_neuron
                acc.neuron_sum = acc.neuron_sum + total
                acc.neuron_top = torch.cat([acc.neuron_top, peaks]).topk(k, dim=0).values
                total, peaks = acc.pending_head
                acc.head_sum = acc.head_sum + total
                acc.head_top = torch.cat([acc.head_top, peaks]).topk(k, dim=0).values
                acc.pending_neuron = acc.pending_head = None
        self.positions += self._chunk_positions
        self.documents += self._chunk_docs
        self.head_documents += self._chunk_head_docs

    def discard_chunk(self) -> None:
        """Drop the framed forward (it failed and will be re-run)."""
        self._mask = self._head_mask = None
        for acc in self._layers:
            acc.pending_neuron = acc.pending_head = None

    def abandon(self, reason: str) -> None:
        """Give up for the rest of the pass: remove the hooks, drop the framed
        forward and record ``reason`` (the capture re-runs the rows without
        the hooks, so units never fail the neutral pass)."""
        self.close()
        self.discard_chunk()
        if self.failure is None:
            self.failure = reason

    # ------------------------------------------------------------ result

    def finish(self) -> UnitStats:
        """The statistics, fp32 on CPU.  Raises when nothing was counted or a
        hook failed (the caller records units as unavailable)."""
        if self.failure is not None:
            raise RuntimeError(self.failure)
        if self.positions <= 0:
            raise RuntimeError("no neutral position reached the unit statistics")
        layers: dict[int, LayerUnitStats] = {}
        n = float(self.positions)
        for acc in self._layers:
            neuron_top = acc.neuron_top.cpu()
            layers[acc.layer] = LayerUnitStats(
                neuron_mean=(acc.neuron_sum / n).cpu(),
                neuron_scale=top_median(neuron_top, self.documents).contiguous(),
                head_mean=(acc.head_sum / n).cpu(),
                head_scale=top_median(acc.head_top.cpu(), self.head_documents),
            )
        return UnitStats(layers=layers, n_documents=self.documents)


__all__ = [
    "HEAD_MEAN",
    "HEAD_SCALE",
    "HEAD_SCALE_FROM",
    "LayerScales",
    "LayerUnitStats",
    "NEURON_MEAN",
    "NEURON_SCALE",
    "SCALE_FLOOR_SHARE",
    "ScaleSource",
    "UNIT_POSITIONS_FROM",
    "UNIT_TENSOR_NAMES",
    "UNIT_TOP_DOCS",
    "UnitScales",
    "UnitStats",
    "UnitStatsAccumulator",
    "reading_scales",
    "soft_scale",
    "top_median",
    "typical_scale",
]
