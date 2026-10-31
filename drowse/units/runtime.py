"""The units runtime: one loaded model's neurons and heads as addressable parts.

Resolves a :data:`~drowse.naming.UnitRef` against the model (the block
description of :mod:`drowse.model.blocks`, discovered once inside the
exclusive section) and the neutral unit statistics
(:mod:`drowse.units.stats`), and answers what the rest of drowse needs of a
unit:

- :meth:`UnitsRuntime.check` — the reference exists (``UnitIndexError``, 400,
  naming the valid range otherwise);
- :meth:`UnitsRuntime.site` — where its activation lives (a coordinate span of
  the down or output projection's input) and its neutral mean, for
  mean-ablation and for the probe tap;
- :meth:`UnitsRuntime.write_profile` — a neuron's ``+``-pole write at its
  layer, the one-layer direction a ``neuron/`` push steers along.

A model whose blocks the description refuses answers every call with
:class:`~drowse.errors.UnsupportedBlockError` (422); a described model whose
neutral cache recorded units as unavailable raises
:class:`UnitsUnavailableError` (422).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from drowse.base.exclusive import ExclusiveGuard
from drowse.errors import DrowseError
from drowse.model.blocks import BlockSpec, ModelSpec, model_spec
from drowse.model.units import UnitIndexError, neuron_write
from drowse.naming import HeadRef, NeuronRef, UnitRef, parse_unit_ref
from drowse.units.stats import UnitStats

UnitKind = Literal["neuron", "head"]


class UnitsUnavailableError(RuntimeError, DrowseError):
    """The model's units cannot be read or ablated: its neutral cache holds no
    unit statistics (``reason`` says why).  422."""

    def user_message(self) -> tuple[int, str]:
        return (422, str(self) or self.__class__.__name__)


@dataclass(frozen=True, eq=False)
class UnitSite:
    """Where one unit's activation lives: coordinates ``[start, stop)`` of
    ``module``'s input (a neuron's one coordinate of the down projection's
    input, a head's ``head_dim`` slice of the output projection's input), with
    its neutral mean over those coordinates (fp32, CPU)."""

    ref: UnitRef
    kind: UnitKind
    layer: int
    module: nn.Module
    start: int
    stop: int
    width: int
    mean: torch.Tensor


class UnitsRuntime:
    """One model's units: discovery, validation, sites and write directions.

    ``stats`` returns the resident neutral unit statistics, building the
    neutral baseline when absent, or raises :class:`UnitsUnavailableError`;
    it runs inside this runtime's exclusive section, like the block
    discovery.
    """

    def __init__(
        self,
        *,
        model: Any,
        device: torch.device,
        exclusive: ExclusiveGuard,
        stats: Callable[[], UnitStats],
    ) -> None:
        self._model = getattr(model, "_orig_mod", model)
        self._device = device
        self._exclusive = exclusive
        self._stats = stats
        self._spec: ModelSpec | None = None

    # ------------------------------------------------------------- model

    def spec(self) -> ModelSpec:
        """The model's block description, discovered once.  Discovery runs
        small probe forwards, so the first call takes the exclusive section;
        later calls (validation at scope resolution included) touch nothing.
        A model the description refuses raises
        :class:`~drowse.errors.UnsupportedBlockError` (422) every time."""
        spec = self._spec
        if spec is not None:
            return spec
        with self._exclusive(
            "unit lookup called while another model use is in flight; retry shortly",
        ):
            spec = model_spec(self._model)
        self._spec = spec
        return spec

    @property
    def device(self) -> torch.device:
        return self._device

    def stats(self) -> UnitStats:
        """The neutral unit statistics (raises :class:`UnitsUnavailableError`)."""
        self.spec()
        with self._exclusive(
            "unit statistics requested while another model use is in flight; "
            "retry shortly",
        ):
            return self._stats()

    def block(self, layer: int) -> BlockSpec:
        """Block ``layer``'s description; a layer outside the model is a 400."""
        spec = self.spec()
        layer = int(layer)
        if not 0 <= layer < spec.n_layers:
            raise UnitIndexError(
                f"layer {layer} is outside the model ({spec.n_layers} blocks, "
                f"0..{spec.n_layers - 1})"
            )
        return spec.blocks[layer]

    def check(self, ref: UnitRef) -> UnitRef:
        """``ref`` itself, once its layer and index exist in the model (400)."""
        block = self.block(ref.layer)
        if isinstance(ref, NeuronRef):
            count, kind = block.d_ff, "neuron"
        else:
            count, kind = block.n_heads, "head"
        if not 0 <= ref.index < count:
            raise UnitIndexError(
                f"{kind} {ref.index} is outside block {ref.layer} "
                f"({count} {kind}s, 0..{count - 1})"
            )
        return ref

    def parse(self, selector: str) -> UnitRef:
        """``neuron/L8n344`` / ``head/L20h17`` → its checked reference."""
        namespace, _, name = selector.partition("/")
        return self.check(parse_unit_ref(namespace, name))

    # ------------------------------------------------------------- sites

    def site(self, ref: UnitRef, *, stats: UnitStats | None = None) -> UnitSite:
        """Where ``ref``'s activation lives and its neutral mean."""
        self.check(ref)
        block = self.block(ref.layer)
        layer_stats = (self.stats() if stats is None else stats).layer(ref.layer)
        if isinstance(ref, NeuronRef):
            n = ref.index
            return UnitSite(
                ref=ref, kind="neuron", layer=ref.layer, module=block.mlp.down,
                start=n, stop=n + 1, width=int(block.d_ff),
                mean=layer_stats.neuron_mean[n : n + 1].clone(),
            )
        attn = block.attn
        hd = attn.head_dim
        h = ref.index
        return UnitSite(
            ref=ref, kind="head", layer=ref.layer, module=attn.o,
            start=h * hd, stop=(h + 1) * hd, width=int(attn.n_heads * hd),
            mean=layer_stats.head_mean[h].clone(),
        )

    def write_profile(self, ref: NeuronRef) -> dict[int, torch.Tensor]:
        """A neuron's ``+``-pole write at its layer, fp32 on the session device:
        the one-layer profile a ``neuron/`` push folds and steers along."""
        self.check(ref)
        return {
            ref.layer: neuron_write(self.block(ref.layer), ref.index)
            .to(device=self._device, dtype=torch.float32),
        }


def unit_ref(selector: str) -> UnitRef | None:
    """``selector``'s unit reference when it names one (``neuron/…`` /
    ``head/…``), else ``None``; a malformed unit name raises (400)."""
    namespace, sep, name = selector.partition("/")
    if not sep or namespace not in (NeuronRef.namespace, HeadRef.namespace):
        return None
    return parse_unit_ref(namespace, name)


__all__ = [
    "UnitKind",
    "UnitSite",
    "UnitsRuntime",
    "UnitsUnavailableError",
    "unit_ref",
]
