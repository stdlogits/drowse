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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True, eq=False)
class LayerUnitStats:
    """One block's unit statistics, fp32 on CPU.

    ``neuron_scale`` row 0 is the ``+`` pole's ``act_top``, row 1 the ``−``
    pole's (both ``≥ 0``).
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
    hooks go quiet, when a hook raises anything but an out-of-memory error:
    the neutral pass must never fail because of units.
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
                zh = z.float().view(b, t, acc.n_heads, acc.head_dim)
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
    "LayerUnitStats",
    "NEURON_MEAN",
    "NEURON_SCALE",
    "UNIT_POSITIONS_FROM",
    "UNIT_TENSOR_NAMES",
    "UNIT_TOP_DOCS",
    "UnitStats",
    "UnitStatsAccumulator",
    "top_median",
]
