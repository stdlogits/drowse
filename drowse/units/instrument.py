"""The units instrument: neurons and heads as a read family (``session.units``).

A unit probe reads one model part on the same forward the generation runs:

- ``neuron/L8n344`` — the neuron's activation (coordinate 344 of block 8's
  down projection input) over its pole's neutral scale, signed;
- ``head/L20h17`` — the head's write norm ``‖W_O^h z_h‖`` (Gemma's sandwich
  scale included) over its neutral scale.

The neutral scale is the soft one (:func:`~drowse.units.stats.soft_scale`):
``act_top`` blended with a floor from the layer's typical scale.

Both are :class:`~drowse.readout.types.ScalarReading`\\ s in the
``activation_over_scale`` unit at one layer (depth basis ``single_layer``),
with the raw firing in ``meta``.  Nothing reads the residual capture: a bound
run owns a :class:`~drowse.units.capture.UnitTap` whose hooks exist only while
a probe or gate consumes them (an attached probe with a per-token consumer or
a final aggregate, or a ``@when:neuron/…`` gate).  A gate needs no attached
probe: any unit the steering gates on is read for the gate alone (and appears
in no reading).

One channel, like the lens and SAE: ``<name>`` / ``<name>[0]``.  The family
also resolves ``neuron/`` steering atoms (:meth:`UnitsInstrument.register_direction`).
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from drowse.errors import RequestError
from drowse.naming import NeuronRef, UnitRef, parse_unit_ref
from drowse.readout.replay import TokenReplay
from drowse.readout.types import (
    AGG_TAIL_DEPTH,
    Axis,
    DepthSummary,
    GateRef,
    InstrumentBinding,
    InstrumentFamily,
    InstrumentPlan,
    InstrumentPrep,
    ReadRequest,
    ScalarReading,
    UNIT_ACTIVATION_OVER_SCALE,
    UnitsLiveState,
    next_prep_token,
    parse_gate_ref,
    scalar_gate_keys,
    validate_gate_channels,
)
from drowse.units.capture import TapUnit, UnitTap
from drowse.units.runtime import UnitsRuntime, unit_ref
from drowse.units.stats import UnitStats

if TYPE_CHECKING:
    from drowse.units.reads import FindResult, TopResult, UnitDetail, UnitReader


def _spec(ref: UnitRef) -> dict[str, Any]:
    return {
        "unit": ref.qualified,
        "kind": ref.namespace,
        "layer": int(ref.layer),
        "index": int(ref.index),
    }


def _ref(spec: Mapping[str, Any]) -> UnitRef:
    return parse_unit_ref(str(spec["kind"]), str(spec["unit"]).split("/", 1)[1])


class UnitsRun:
    """Per-generation measurement executor for the units family.

    Owns the bound :class:`UnitTap` (hooks attached at bind, removed at
    close), the frozen probe specs, the gate-only units, and the step-keyed
    stash the token tap reuses from the gate callback.
    """

    def __init__(
        self,
        instrument: "UnitsInstrument",
        binding: InstrumentBinding,
        *,
        tap: UnitTap | None = None,
        live: bool = False,
        bound: bool = False,
    ) -> None:
        self._instrument = instrument
        self.binding = binding
        self.tap = tap
        self.live = live
        self.bound = bound
        self._stash: tuple[int, dict[str, tuple[float, float]]] | None = None

    # ------------------------------------------------------------ reads

    def _values(self, fwd: int | None) -> dict[str, tuple[float, float]]:
        """``{name: (value, raw)}`` for forward ``fwd`` — one host transfer."""
        tap = self.tap
        if tap is None or fwd is None:
            return {}
        out = tap.values_at(fwd)
        if out is None:
            return {}
        values, raw = out
        pairs = torch.stack([values, raw], dim=1).to("cpu").tolist()
        return {
            name: (float(v), float(r))
            for name, (v, r) in zip(tap.names, pairs, strict=True)
        }

    def _step_values(self, step_id: int) -> dict[str, tuple[float, float]]:
        stash = self._stash
        if stash is not None and step_id >= 0 and stash[0] == step_id:
            return stash[1]
        tap = self.tap
        values = self._values(None if tap is None else tap.latest())
        self._stash = (step_id, values)
        return values

    def gate_scalars(
        self,
        step_id: int,
        hidden: dict[int, Any] | None,
        gate_keys: "frozenset[str] | set[str] | None",
    ) -> dict[str, float]:
        """The gate channels' scalars for this forward (stashed by
        ``step_id`` for the token tap).  ``None`` scores every tapped unit."""
        del hidden  # the tap is the units' own capture
        values = self._step_values(step_id)
        if gate_keys is None:
            wanted = set(values)
        else:
            wanted = {key.split("[", 1)[0] for key in gate_keys}
        return scalar_gate_keys({
            name: value for name, (value, _raw) in values.items()
            if name in wanted
        })

    def step_readings(self, step_id: int) -> dict[str, ScalarReading]:
        """The attached probes' readings for this forward (the token tap)."""
        if not self.live or not self.binding.specs:
            return {}
        return self._instrument.readings(
            self.binding.specs, self._step_values(step_id),
        )

    def aggregate_at(self, fwd: int | None) -> dict[str, ScalarReading]:
        """The attached probes' readings at decode forward ``fwd`` (the
        end-of-generation aggregate position); empty once it left the ring."""
        if not self.binding.specs or fwd is None:
            return {}
        return self._instrument.readings(self.binding.specs, self._values(fwd))

    def observe_aggregate(
        self, pooled: dict[int, torch.Tensor],
    ) -> dict[str, ScalarReading]:
        """The protocol's aggregate: the residual slice carries no unit
        activation, so this reads the run's latest forward.  The read plane
        uses :meth:`aggregate_at` with the aggregate's forward index."""
        del pooled
        tap = self.tap
        return self.aggregate_at(None if tap is None else tap.latest())

    def close(self) -> None:
        """Remove the tap's hooks and drop the stash (idempotent)."""
        if self.tap is not None:
            self.tap.detach()
        self._stash = None


class UnitsInstrument:
    """Session-lifetime handle for the units read family."""

    family: InstrumentFamily = "units"

    #: Gate channels a unit probe produces: the one strength axis.
    _GATE_CHANNELS: tuple[type, ...] = (Axis,)

    def __init__(
        self,
        runtime: UnitsRuntime,
        *,
        model_id: str,
        n_layers: int,
        dtype: torch.dtype,
        exclusive: Callable[..., Any],
        replay: TokenReplay,
        put_direction: Callable[[str, dict[int, torch.Tensor]], None],
        has_direction: Callable[[str], bool],
        reader: "UnitReader | None" = None,
    ) -> None:
        #: The model's units: discovery, validation, sites, write directions.
        self.runtime = runtime
        #: Static reads, top, find and the catalog (``None``: not wired).
        self._reader = reader
        self._model_id = model_id
        self._n_layers = int(n_layers)
        self._dtype = dtype
        self._exclusive = exclusive
        self._replay = replay
        self._put_direction = put_direction
        self._has_direction = has_direction
        #: Attached unit probes: name -> {unit, kind, layer, index}.
        self.probes: dict[str, dict[str, Any]] = {}
        # Each probe's neutral scale at attach (its digest's unit identity).
        self._scales: dict[str, tuple[float, ...]] = {}
        self._live = True
        self.state_lock = threading.RLock()
        self.current_run = UnitsRun(self, InstrumentBinding(family=self.family))

    # ------------------------------------------------------------- registry

    def attach(self, selector: str, *, as_name: str | None = None) -> str:
        """Attach ``neuron/L8n344`` or ``head/L20h17`` as a probe.

        The unit is checked against the model (400 with the valid range) and
        needs the neutral unit statistics (422 on a model without them).
        """
        ref = unit_ref(selector)
        if ref is None:
            raise RequestError(
                f"{selector!r} is not a unit (write neuron/L<layer>n<index> or "
                "head/L<layer>h<index>)"
            )
        with self._exclusive(
            "add_probe called while another model operation is in flight; retry shortly"
        ):
            self.runtime.check(ref)
            scale = self._scale(self.runtime.stats(), ref)
            name = as_name if as_name is not None else ref.qualified
            with self.state_lock:
                self.probes[name] = _spec(ref)
                self._scales[name] = tuple(
                    round(float(v), 9) for v in scale.reshape(-1).tolist()
                )
        return name

    def try_detach(self, name: str) -> bool:
        with self.state_lock:
            if name not in self.probes:
                return False
            del self.probes[name]
            self._scales.pop(name, None)
            return True

    def specs(self) -> dict[str, dict[str, Any]]:
        with self.state_lock:
            return {name: dict(spec) for name, spec in self.probes.items()}

    @property
    def names(self) -> list[str]:
        with self.state_lock:
            return list(self.probes)

    def probe_hash(self, name: str) -> str | None:
        """The unit plus the neutral scale that sets its reading's unit (taken
        at attach, so a digest never needs the exclusive section)."""
        with self.state_lock:
            spec = self.probes.get(name)
            if spec is None:
                return None
            unit = str(spec["unit"])
            scale = list(self._scales.get(name, ()))
        return hashlib.sha256(repr((
            "units-readout-v1", self._model_id, unit, scale,
        )).encode("utf-8")).hexdigest()

    def validate_gate(self, ref: GateRef) -> None:
        validate_gate_channels(ref, self._GATE_CHANNELS, family=self.family)

    def gate_unit(self, ref: GateRef) -> bool:
        """Whether a gate reads this family: it names an attached probe, or a
        unit (checked against the model: a bad index is a 400 here, at
        generation preflight)."""
        with self.state_lock:
            if ref.probe in self.probes:
                return True
        unit = unit_ref(ref.probe)
        if unit is None:
            return False
        self.runtime.check(unit)
        return True

    @property
    def active_source(self) -> str | None:
        """Units have no source lifecycle: they are the model's own parts."""
        return None

    # ---------------------------------------------------------------- live

    @property
    def live_state(self) -> UnitsLiveState:
        return UnitsLiveState(enabled=self._live)

    def set_live(self, enabled: bool, **kwargs: Any) -> UnitsLiveState:
        """Toggle per-token unit readings (all-or-nothing; no extras)."""
        if kwargs:
            raise TypeError(f"units live takes no extras, got {sorted(kwargs)}")
        self._live = bool(enabled)
        return self.live_state

    @property
    def is_live(self) -> bool:
        return self._live

    # ------------------------------------------------------------ lifecycle

    def prepare(self, request: ReadRequest) -> InstrumentPrep:
        if self.current_run.bound:
            raise RuntimeError(
                "UnitsInstrument.prepare() on a bound run: close the prior "
                "generation's run (ReadPlane.close_runs) first"
            )
        return InstrumentPrep(
            family=self.family, request=request, token=next_prep_token(),
        )

    def plan(self, prep: InstrumentPrep) -> InstrumentPlan:
        """Demand: the gate keys this family reads and whether a final
        aggregate will.  Units capture no residual layer (the tap is their own
        capture), so ``latest_layers``/``tail_layers`` stay empty."""
        if prep.family != self.family:
            raise TypeError(
                "UnitsInstrument.plan takes the InstrumentPrep its own "
                f"prepare() returned, got family={prep.family!r}"
            )
        request = prep.request
        with self.state_lock:
            has_probes = bool(self.probes)
        return InstrumentPlan(
            family=self.family,
            gate_keys=frozenset(request.gate_keys),
            final_aggregate=bool(has_probes and request.final_aggregate),
            prep_token=prep.token,
        )

    def bind(self, plan: InstrumentPlan, prep: InstrumentPrep) -> UnitsRun:
        """Freeze the roster and attach the tap when anything reads it.

        The tap covers the attached probes when a per-token consumer
        (``request.live``) or the final aggregate reads them, plus every unit
        a gate names.  Hooks attach here (inside the generation's exclusive
        section) and leave at :meth:`close_run`.
        """
        if prep.family != self.family:
            raise TypeError(
                "UnitsInstrument.bind takes the InstrumentPrep its own "
                f"prepare() returned, got family={prep.family!r}"
            )
        if plan.family != self.family:
            raise ValueError(
                f"UnitsInstrument.bind: plan family {plan.family!r} is not "
                f"{self.family!r}"
            )
        if plan.prep_token != prep.token:
            raise ValueError(
                "UnitsInstrument.bind: the plan was not derived from this "
                "prep (prep_token mismatch)"
            )
        request = prep.request
        live = bool(request.live and self._live)
        with self.state_lock:
            specs = {name: dict(spec) for name, spec in self.probes.items()}
        tapped: dict[str, UnitRef] = {}
        if specs and (live or plan.final_aggregate):
            tapped.update({name: _ref(spec) for name, spec in specs.items()})
        for key in plan.gate_keys:
            probe = parse_gate_ref(key).probe
            if probe in tapped:
                continue
            if probe in specs:
                tapped[probe] = _ref(specs[probe])
                continue
            ref = unit_ref(probe)
            if ref is not None:
                tapped[probe] = ref
        tap = self._tap(tapped) if tapped else None
        run = UnitsRun(
            self,
            InstrumentBinding(family=self.family, specs=specs),
            tap=tap, live=live, bound=True,
        )
        self.current_run = run
        if tap is not None:
            tap.attach()
        return run

    def close_run(self) -> None:
        self.current_run.close()
        self.current_run = UnitsRun(self, InstrumentBinding(family=self.family))

    def close(self) -> None:
        """Session teardown: close the run (its hooks leave) and drop the roster."""
        self.close_run()
        with self.state_lock:
            self.probes.clear()
            self._scales.clear()

    def end_capture(self) -> None:
        """Remove the current run's tap hooks; its ring stays readable for the
        finalize aggregate (the decode loop's last forward has run)."""
        tap = self.current_run.tap
        if tap is not None:
            tap.detach()

    @property
    def hooks_attached(self) -> bool:
        """Whether the current run's tap has hooks on the model (they route a
        compiled session to its eager module)."""
        tap = self.current_run.tap
        return tap is not None and tap.attached

    # --------------------------------------------------------------- reading

    def _tap(self, units: Mapping[str, UnitRef], *, depth: int = AGG_TAIL_DEPTH) -> UnitTap:
        runtime = self.runtime
        stats = runtime.stats()
        entries = []
        for name, ref in units.items():
            site = runtime.site(ref, stats=stats)
            entries.append(TapUnit(
                name=name, site=site, block=runtime.block(ref.layer),
                scale=self._scale(stats, ref),
            ))
        return UnitTap(entries, device=runtime.device, dtype=self._dtype, depth=depth)

    @staticmethod
    def _scale(stats: UnitStats, ref: UnitRef) -> torch.Tensor:
        layer = stats.layer(ref.layer)
        if isinstance(ref, NeuronRef):
            return layer.neuron_reading_scale[:, ref.index].clone()
        return layer.head_reading_scale[ref.index].reshape(()).clone()

    def readings(
        self,
        specs: Mapping[str, Mapping[str, Any]],
        values: Mapping[str, tuple[float, float]],
    ) -> dict[str, ScalarReading]:
        """Shape ``{name: (value, raw)}`` into the probes' readings."""
        out: dict[str, ScalarReading] = {}
        depth_den = max(self._n_layers - 1, 1)
        for name, spec in specs.items():
            pair = values.get(name)
            if pair is None:
                continue
            value, raw = pair
            layer = int(spec["layer"])
            meta_key = "activation" if spec["kind"] == NeuronRef.namespace else "write_norm"
            out[name] = ScalarReading(
                value=value,
                unit=UNIT_ACTIVATION_OVER_SCALE,
                per_layer={layer: value},
                depth=DepthSummary(
                    center=(layer / depth_den,), spread=(0.0,), basis="single_layer",
                ),
                meta={"unit": str(spec["unit"]), meta_key: round(raw, 6)},
            )
        return out

    def step_readings(self, step_id: int) -> dict[str, ScalarReading]:
        """This forward's attached-probe readings (the token tap)."""
        return self.current_run.step_readings(step_id)

    def score_aggregate(self, fwd: int | None) -> dict[str, ScalarReading]:
        """The end-of-generation aggregate at decode forward ``fwd``."""
        return self.current_run.aggregate_at(fwd)

    # ------------------------------------------------------------ steering

    def register_direction(self, atom: str) -> str:
        """Register a neuron's ``+``-pole write at its layer as the profile
        ``neuron/<atom>`` (the resolver behind the ``neuron/`` steering atom;
        the push's sign picks the pole).  Idempotent."""
        ref = parse_unit_ref(NeuronRef.namespace, atom)
        name = ref.qualified
        if self._has_direction(name):
            return name
        assert isinstance(ref, NeuronRef)
        self._put_direction(name, self.runtime.write_profile(ref))
        return name

    # ---------------------------------------------------------- static reads

    @property
    def reads(self) -> "UnitReader":
        """The static-read collaborator (:class:`~drowse.units.reads.UnitReader`)."""
        if self._reader is None:
            raise RuntimeError("this units instrument has no reader wired")
        return self._reader

    def read(self, unit: "str | UnitRef", *, k: int = 8, hide_hubs: bool = False) -> "UnitDetail":
        """What a unit is about, from its weights: per neuron pole its writes
        (at ``out``) and what it fires on (at ``mid``), per head its copies,
        suppressions and inputs, each through the depth rule's R/S pair;
        scalars, firing statistics and the catalog label
        (:meth:`~drowse.units.reads.UnitReader.read`)."""
        return self.reads.read(unit, k=k, hide_hubs=hide_hubs)

    def top(
        self, token: str, *, by: str = "writes",
        layers: "list[int] | str | None" = None, k: int = 20,
    ) -> "TopResult":
        """Neuron poles that write (``by="writes"``) or fire on (``"fires"``)
        a token most specifically (:meth:`~drowse.units.reads.UnitReader.top`)."""
        if by not in ("writes", "fires"):
            raise RequestError(f"by must be 'writes' or 'fires', got {by!r}")
        return self.reads.top(token, by="writes" if by == "writes" else "fires", layers=layers, k=k)

    def find(
        self, target: "str | torch.Tensor | Mapping[int, torch.Tensor]", *,
        layer: int | None = None, k: int = 16, layers: "list[int] | str | None" = None,
    ) -> "FindResult":
        """A direction split over each layer's neuron writes
        (:meth:`~drowse.units.reads.UnitReader.find`)."""
        return self.reads.find(target, layer=layer, k=k, layers=layers)

    def stats(self, unit: "str | UnitRef") -> dict[str, Any]:
        """How a unit fires: neutral statistics and the catalog's row."""
        return self.reads.stats(unit)

    def catalog_status(self) -> dict[str, Any]:
        """The model's neuron-lens catalog binding (disk only)."""
        return self.reads.catalog.status()

    def labels(self, units: "Sequence[str | UnitRef]") -> dict[str, Any]:
        """The catalog's labels for ``units``, per neuron pole and per head
        (disk only; :meth:`~drowse.units.reads.UnitReader.labels`)."""
        return self.reads.labels(units)

    def fetch_catalog(
        self, *, force: bool = False, on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Bind the model's neuron-lens catalog (network and disk; no model
        work).  404 for an uncatalogued model."""
        return self.reads.catalog.fetch(force=force, on_progress=on_progress)

    # -------------------------------------------------------------- replay

    def token_readout(
        self,
        node_id: str,
        raw_index: int,
        *,
        top_k: int | None = None,
        layers: "list[int] | str | None" = None,
        apply_steering: bool = True,
        raw: bool = False,
    ) -> dict[str, Any]:
        """Replay the forward that produced a loom token and read every
        attached unit probe there, as the finished ``scope="replay"``
        envelope.  ``top_k`` and ``layers`` are refused (a unit names its own
        layer and there is no ranked readout)."""
        from drowse.readout.measurements import build_measurements

        if top_k is not None:
            raise RequestError("units token-readout takes no top_k (units are read by name)")
        if layers is not None:
            raise RequestError("units token-readout takes no layers (a unit names its layer)")
        specs = self.specs()
        if not specs:
            return {"measurements": None}
        units = {name: _ref(spec) for name, spec in specs.items()}
        holder: dict[str, UnitTap] = {}

        def capture_layers() -> list[int]:
            # Resolved inside the replay's exclusive section: the tap's hooks
            # attach there and the forward runs through the deepest unit.
            tap = self._tap(units, depth=1)
            tap.attach()
            holder["tap"] = tap
            return [max(ref.layer for ref in units.values())]

        def read(_hidden: dict[int, torch.Tensor]) -> dict[str, ScalarReading]:
            tap = holder["tap"]
            out = tap.values_at(0)
            if out is None:
                return {}
            values, raw_values = out
            pairs = torch.stack([values, raw_values], dim=1).to("cpu").tolist()
            return self.readings(specs, {
                name: (float(v), float(r))
                for name, (v, r) in zip(tap.names, pairs, strict=True)
            })

        try:
            token, readings = self._replay(
                node_id, raw_index, layers=capture_layers, read=read,
                apply_steering=apply_steering, raw=raw,
                label="session.units.token_readout",
            )
        finally:
            tap = holder.get("tap")
            if tap is not None:
                tap.detach()
        return {"measurements": build_measurements(
            scope="replay",
            provenance="replayed",
            units_readings=readings,
            steering=token.steering,
        )}


__all__ = ["UnitsInstrument", "UnitsRun"]
