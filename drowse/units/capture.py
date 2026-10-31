"""The unit tap: the probed units' activations at the last position, per forward.

One forward hook per touched module (a down projection for neurons, an output
projection for heads), installed only while a probe or gate reads it.  Each
hook does device work only: one ``index_select`` of its units' coordinates at
the last position into a preallocated ring row (``out=``, nothing allocated),
plus, for a sandwich block's head, the output row's norm (the per-position
scale of Gemma's post-attention norm).  Forward hooks run after every forward
pre-hook, so a probe reads the activation the model actually used, after any
unit ablation.  The ring is :data:`~drowse.readout.types.AGG_TAIL_DEPTH`
forwards deep; the module that runs last in a forward advances the count.

Values are computed after the forward (:meth:`UnitTap.values_at`): a neuron's
``act`` over its pole's scale, a head's ``‖W_O^h z_h‖`` (times the sandwich
scale) over its scale, as one fp32 vector the caller moves to the host once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from drowse.model.blocks import BlockSpec
from drowse.model.units import head_write_matrix
from drowse.units.runtime import UnitSite

#: Floor under a unit's scale, so a unit silent on the neutral corpus reads
#: large rather than infinite.
SCALE_FLOOR = 1e-6


@dataclass(frozen=True, eq=False)
class TapUnit:
    """One unit the tap reads: its site, block and neutral scale (a neuron's
    ``[+, −]`` pole pair, a head's single scale)."""

    name: str
    site: UnitSite
    block: BlockSpec
    scale: torch.Tensor


@dataclass
class _Module:
    module: nn.Module
    kind: str
    order: tuple[int, int]
    index: torch.Tensor
    start: int
    stop: int
    rows: int
    norm_col: int | None


class UnitTap:
    """Hooks + device ring for one bound run's units (see the module doc)."""

    def __init__(
        self,
        units: Sequence[TapUnit],
        *,
        device: torch.device,
        dtype: torch.dtype,
        depth: int,
    ) -> None:
        if not units:
            raise ValueError("a unit tap needs at least one unit")
        self.names = [unit.name for unit in units]
        self.depth = int(depth)
        self._device = device
        self._handles: list[Any] = []
        self._fwd = 0
        # Group units by module: one hook and one index tensor each.
        groups: dict[int, list[int]] = {}
        for i, unit in enumerate(units):
            groups.setdefault(id(unit.site.module), []).append(i)
        modules: list[_Module] = []
        act_cols = 0
        norm_cols = 0
        # ring columns of each unit's activation (neurons: 1, heads: head_dim)
        unit_cols: dict[int, tuple[int, int]] = {}
        unit_norm: dict[int, int | None] = {}
        for members in groups.values():
            first = units[members[0]]
            site = first.site
            if site.kind == "neuron":
                index = [units[i].site.ref.index for i in members]
                width = 1
            else:
                index = [units[i].site.ref.index for i in members]
                width = site.stop - site.start
            norm_col = None
            if site.kind == "head" and first.block.attn_out_norm is not None:
                norm_col = norm_cols
                norm_cols += 1
            start = act_cols
            for slot, i in enumerate(members):
                unit_cols[i] = (start + slot * width, start + (slot + 1) * width)
                unit_norm[i] = norm_col
            act_cols += width * len(members)
            modules.append(_Module(
                module=site.module,
                kind=site.kind,
                order=(site.layer, 1 if site.kind == "neuron" else 0),
                index=torch.tensor(index, dtype=torch.long, device=device),
                start=start,
                stop=act_cols,
                rows=len(members),
                norm_col=norm_col,
            ))
        self._modules = modules
        self._terminal = max(modules, key=lambda m: m.order).module
        self.acts = torch.zeros(self.depth, act_cols, dtype=dtype, device=device)
        self.norms = torch.zeros(
            self.depth, max(norm_cols, 1), dtype=torch.float32, device=device,
        )
        self._build_readout(units, unit_cols, unit_norm)

    def _build_readout(
        self,
        units: Sequence[TapUnit],
        unit_cols: dict[int, tuple[int, int]],
        unit_norm: dict[int, int | None],
    ) -> None:
        device = self._device
        neurons = [i for i, u in enumerate(units) if u.site.kind == "neuron"]
        heads = [i for i, u in enumerate(units) if u.site.kind == "head"]
        self._neuron_pos = torch.tensor(neurons, dtype=torch.long, device=device)
        self._head_pos = torch.tensor(heads, dtype=torch.long, device=device)
        self._n_units = len(units)
        self._neuron_cols = torch.tensor(
            [unit_cols[i][0] for i in neurons], dtype=torch.long, device=device,
        )
        scales = [units[i].scale.clamp_min(SCALE_FLOOR) for i in neurons]
        pair = torch.stack(scales) if scales else torch.zeros(0, 2)
        self._neuron_plus = pair[:, 0].to(device=device, dtype=torch.float32)
        self._neuron_minus = pair[:, 1].to(device=device, dtype=torch.float32)
        if heads:
            hd = units[heads[0]].site.stop - units[heads[0]].site.start
            same = all(
                units[i].site.stop - units[i].site.start == hd for i in heads
            )
            if not same:
                raise ValueError("heads of different sizes in one tap")
            self._head_cols = torch.stack([
                torch.arange(unit_cols[i][0], unit_cols[i][1]) for i in heads
            ]).to(device)
            grams = []
            for i in heads:
                m = head_write_matrix(units[i].block, units[i].site.ref.index)
                grams.append(m.T @ m)
            self._head_gram = torch.stack(grams).to(device=device, dtype=torch.float32)
            self._head_scale = torch.stack([
                units[i].scale.reshape(()).clamp_min(SCALE_FLOOR) for i in heads
            ]).to(device=device, dtype=torch.float32)
            norm_index: list[int] = []
            norm_dim: list[float] = []
            norm_eps: list[float] = []
            for i in heads:
                col = unit_norm[i]
                out_norm = units[i].block.attn_out_norm
                norm_index.append(-1 if col is None else int(col))
                norm_dim.append(float(units[i].block.d_model))
                norm_eps.append(0.0 if out_norm is None else float(out_norm.eps))
            self._head_norm_col = torch.tensor(norm_index, dtype=torch.long, device=device)
            self._head_norm_dim = torch.tensor(norm_dim, dtype=torch.float32, device=device)
            self._head_norm_eps = torch.tensor(norm_eps, dtype=torch.float32, device=device)
        self._has_heads = bool(heads)

    # ------------------------------------------------------------- hooks

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    @property
    def forwards(self) -> int:
        """Forwards the tap has recorded since it attached."""
        return self._fwd

    def attach(self) -> None:
        """Register one forward hook per touched module (idempotent)."""
        if self._handles:
            return
        self._fwd = 0
        try:
            for entry in self._modules:
                hook = (
                    self._neuron_hook(entry) if entry.kind == "neuron"
                    else self._head_hook(entry)
                )
                self._handles.append(entry.module.register_forward_hook(hook))
        except BaseException:
            self.detach()
            raise

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _advance(self, module: nn.Module) -> None:
        if module is self._terminal:
            self._fwd += 1

    def _neuron_hook(self, entry: _Module) -> Any:
        acts = self.acts
        index = entry.index

        def hook(module: nn.Module, args: tuple[Any, ...], _output: Any) -> None:
            x = args[0]
            last = x[0, -1] if x.dim() == 3 else x[-1]
            if last.dtype != acts.dtype:
                last = last.to(acts.dtype)
            slot = self._fwd % self.depth
            torch.index_select(last, 0, index, out=acts[slot, entry.start:entry.stop])
            self._advance(module)

        return hook

    def _head_hook(self, entry: _Module) -> Any:
        acts = self.acts
        norms = self.norms
        index = entry.index
        width = (entry.stop - entry.start) // entry.rows

        def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            z = args[0]
            last = z[0, -1] if z.dim() == 3 else z[-1]
            if last.dtype != acts.dtype:
                last = last.to(acts.dtype)
            slot = self._fwd % self.depth
            torch.index_select(
                last.reshape(-1, width), 0, index,
                out=acts[slot, entry.start:entry.stop].view(entry.rows, width),
            )
            if entry.norm_col is not None:
                out = output if isinstance(output, torch.Tensor) else output[0]
                row = out[0, -1] if out.dim() == 3 else out[-1]
                torch.linalg.vector_norm(
                    row, dtype=torch.float32, out=norms[slot, entry.norm_col],
                )
            self._advance(module)

        return hook

    # ------------------------------------------------------------- reads

    def has(self, fwd: int) -> bool:
        """Whether forward ``fwd`` (0-based since attach) is still in the ring."""
        return 0 <= fwd < self._fwd and fwd >= self._fwd - self.depth

    def values_at(self, fwd: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(values, raw)`` for forward ``fwd``, fp32 on the device, one entry
        per unit in :attr:`names` order: the probe value and the raw firing
        (a neuron's ``act``, a head's write norm).  ``None`` once the forward
        has left the ring (or before it ran)."""
        if not self.has(fwd):
            return None
        slot = fwd % self.depth
        row = self.acts[slot].float()
        values = torch.zeros(self._n_units, dtype=torch.float32, device=row.device)
        raw = torch.zeros_like(values)
        if self._neuron_pos.numel():
            act = row.index_select(0, self._neuron_cols)
            scale = torch.where(act >= 0, self._neuron_plus, self._neuron_minus)
            values.index_copy_(0, self._neuron_pos, act / scale)
            raw.index_copy_(0, self._neuron_pos, act)
        if self._has_heads:
            z = row[self._head_cols]
            norm = torch.einsum(
                "ki,kij,kj->k", z, self._head_gram, z,
            ).clamp_min(0.0).sqrt()
            out_norm = self.norms[slot].index_select(0, self._head_norm_col.clamp_min(0))
            r = torch.rsqrt(out_norm.square() / self._head_norm_dim + self._head_norm_eps)
            norm = torch.where(self._head_norm_col >= 0, norm * r, norm)
            values.index_copy_(0, self._head_pos, norm / self._head_scale)
            raw.index_copy_(0, self._head_pos, norm)
        return values, raw

    def latest(self) -> int | None:
        """The most recent recorded forward, or ``None`` before the first."""
        return self._fwd - 1 if self._fwd > 0 else None


__all__ = ["SCALE_FLOOR", "TapUnit", "UnitTap"]
