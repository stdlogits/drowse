"""The J-lens instrument: the lens family's residency, lifecycle and reads.

:class:`LensInstrument` is the one public face of the lens family
(``session.lens``).  It owns:

- the resident lens (:class:`~drowse.lens.runtime.LensRuntime`) and the
  residency transactions over it (refresh from disk, adopt, evict, source
  switch), serialized by :attr:`LensInstrument.state_lock`;
- the fit lifecycle (:class:`~drowse.lens.lifecycle.LensFitter`);
- the offline reads (:meth:`LensInstrument.readout`,
  :meth:`LensInstrument.decompose`, :meth:`LensInstrument.register_direction`,
  :meth:`LensInstrument.token_readout`);
- the readout-channel probe registry, the live workspace readout, and the
  per-generation run (:class:`LensRun`).

Dependencies are explicit constructor arguments: the model, tokenizer and
device handles, the capture buffer the per-step reads consume, the
model-exclusive guard, the loom token replay, and a few session callbacks
(registry writes, cache invalidation, profile resolution, the capture
forward).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, NamedTuple, Protocol, cast, overload

import torch

from drowse.readout.replay import TokenReplay
from drowse.readout.types import (
    Axis,
    DepthSummary,
    GateRef,
    InstrumentBinding,
    InstrumentFamily,
    InstrumentPlan,
    InstrumentPrep,
    LensLiveState,
    LensPrep,
    ReadRequest,
    ScalarReading,
    UNIT_MEAN_FOOTPRINT_Z,
    UNIT_MEAN_TOKEN_PROBABILITY,
    next_prep_token,
    parse_gate_ref,
    scalar_gate_keys,
    validate_gate_channels,
)
# The readout half of the lens module is import-light by construction (the
# estimator lives in ``jlens_fit``), so the per-decode-step surfaces below bind
# these at module scope instead of re-importing inside each hot call.
from drowse.lens.jlens import (
    aggregate_readout_tensors_from_probabilities,
    pack_readout_rows_to_host,
    readout_probabilities,
    resolve_token_atom,
    token_readout_stats,
    token_readout_stats_from_probabilities,
    transport_rows,
)
from drowse.base.exclusive import ExclusiveGuard
from drowse.base.lru import BoundedLRU
from drowse.model.layers import LayerSelector
from drowse.lens.depth import LensBand, s_weights
from drowse.lens.footprint import (
    TokenEmbeddings,
    footprint_probe_rows,
    pullback_rows,
    token_mask,
)
from drowse.lens.key import LENS_SITES
from drowse.lens.lifecycle import LensFitRequest, LensFitter, LiveConfig
from drowse.lens.read import LensRead, read_vectors
from drowse.lens.runtime import CaptureHidden, LensRuntime, select_rows
from drowse.readout.protocol import LatestSlices
from drowse.errors import RequestError

_log = logging.getLogger(__name__)

#: A probe spec's ``side``: ``output`` reads the R family (``jlens/``),
#: ``input`` the S family (``slens/``).
_ATOM_SIDES = {"jlens": "output", "slens": "input"}
#: Footprint-norm tables kept for the ``tokens-in`` dictionary (``[vocab]``
#: fp32 per (lens, layer)).
FOOTPRINT_NORM_CACHE_MAX = 64


def _out_layers(lens: Any) -> list[int]:
    """A lens's ``out``-site layers, or none when it holds only ``mid``.

    Probes, steering atoms and the live readout read block outputs, so they
    use the ``out`` site; a mid-only source leaves them without layers rather
    than reading ``mid`` in its place.
    """
    if lens is None:
        return []
    sites = getattr(lens, "sites", None)
    if sites is None:  # a lens-shaped stand-in: its out layers are source_layers
        return [int(layer) for layer in lens.source_layers]
    if "out" not in sites:
        return []
    return [int(layer) for layer in lens.layers("out")]


def _spec_side(spec: Mapping[str, Any]) -> str:
    return str(spec.get("side", "output"))


class _InputBlock(NamedTuple):
    """``slens/`` probe readings still on the device: ``values`` is
    ``[3 + L, P]`` (the ``w_S``-weighted mean z, depth center, depth spread,
    then ``z_L`` per layer) for ``names`` (columns) over ``layers`` (rows
    from 3)."""

    names: tuple[str, ...]
    layers: tuple[int, ...]
    values: torch.Tensor

    def readings(self, host: torch.Tensor) -> dict[str, ScalarReading]:
        """Unpack the host copy of ``values``."""
        out: dict[str, ScalarReading] = {}
        for column, name in enumerate(self.names):
            out[name] = ScalarReading(
                value=float(host[0, column]),
                unit=UNIT_MEAN_FOOTPRINT_Z,
                per_layer={
                    int(layer): float(host[3 + row, column])
                    for row, layer in enumerate(self.layers)
                },
                depth=DepthSummary(
                    center=(float(host[1, column]),),
                    spread=(float(host[2, column]),),
                    basis="footprint_z",
                ),
            )
        return out


class ProfileStore(Protocol):
    """The profile-registry surface the lens writes ``jlens/`` directions
    through (the session's artifact registry).
    """

    @property
    def profiles(self) -> Mapping[str, Any]: ...

    def put_profile(self, name: str, tensors: Mapping[int, torch.Tensor]) -> None: ...

    def evict_prefix(self, prefix: str) -> list[str]: ...


class LensRun:
    """Per-generation measurement executor for the lens family.

    Owns everything generation-scoped (``protocol.py``): the immutable
    :class:`InstrumentBinding` (probe specs frozen at bind), the resident-
    lens disk-identity pin, the live-active flag, the per-forward stash the
    gate callback and the display step share.
    An **idle** run (``bound=False``) backs out-of-generation reads —
    defensive captures, offline readouts — with live-registry passthrough
    semantics, so every read path has a run to consult.
    """

    def __init__(
        self,
        instrument: "LensInstrument",
        binding: InstrumentBinding,
        *,
        lens: Any = None,
        pinned: bool = False,
        active: bool = True,
        live_state: "Mapping[str, Any] | None" = None,
        bound: bool = False,
        input_lens: Any = None,
        input_pinned: bool = False,
        sources: "Mapping[str, str | None] | None" = None,
    ) -> None:
        self._instrument = instrument
        self.binding = binding
        #: The pinned resident lens (or None — unbound / validated-missing).
        self.lens = lens
        #: True when this generation snapshot-pinned the lens identity.
        self.pinned = pinned
        #: The pinned S-family lens ``slens/`` probes read, and its pin flag.
        self.input_lens = input_lens
        self.input_pinned = input_pinned
        #: ``{"R": label, "S": label}``: the source each pinned lens was
        #: selected as, the lens measurement binding (``binding.sources``).
        self.sources: dict[str, str | None] = dict(sources or {})
        #: ``slens/`` readings of one forward, keyed by ``step``, so the
        #: display step reuses what the gate callback computed.
        self.input_stash: dict[str, Any] | None = None
        #: Live-readout activity for this generation.
        self.active = active
        #: The live-readout runtime dict snapshotted at prepare — an
        #: interleaved adoption rebuilds the instrument-level ``live``
        #: against the NEW lens, so a bound run must keep reading the
        #: state that matches its pin (idle runs pass through instead).
        self.live_state = live_state
        #: True for a generation-bound run (idle runs pass through).
        self.bound = bound
        self.step_stash: dict[str, Any] | None = None
        self.last_step_readings: dict[str, ScalarReading] | None = None

    # ------------------------------------------------------------ protocol

    def gate_scalars(
        self,
        step_id: int,
        hidden: dict[int, Any] | None,
        gate_keys: frozenset[str] | set[str] | None,
    ) -> dict[str, float]:
        """The gate channels' scalars for this step.  The per-forward
        matrix-level reuse (band logits stashed for the display step)
        stays inside the instrument's worker — the stash lives on this
        run and is keyed by ``step_id``, so the display reuses rows iff
        they came from the same forward.  ``gate_keys=None`` scores the
        full roster (the session forwarder's bare-call shape)."""
        del hidden  # worker reads the capture's latest slices
        return self._instrument.gate_scalars(
            set(gate_keys) if gate_keys is not None else None,
            step_id=step_id,
        )

    def observe_aggregate(
        self, pooled: dict[int, Any],
    ) -> dict[str, ScalarReading]:
        """End-of-generation aggregate at the pooled last-content slice."""
        return self._instrument.score_probes(pooled)

    def close(self) -> None:
        """Release generation-scoped state (stash, pin)."""
        self.step_stash = None
        self.last_step_readings = None
        self.lens = None
        self.pinned = False
        self.input_lens = None
        self.input_pinned = False
        self.input_stash = None
        self.active = True
        self.live_state = None



class _InputResidency:
    """The S family's slot as a fit drives it (:class:`LensResidency`).

    An S/K fit evicts and adopts here exactly as an R/J fit does on the
    instrument itself; there is no live S readout to carry across.
    """

    def __init__(self, lens: "LensInstrument") -> None:
        self._lens = lens

    @property
    def resident_lens(self) -> Any:
        return self._lens.input_runtime.lens

    @property
    def resident_identity(self) -> str | None:
        return self._lens.input_runtime.identity

    def live_config(self) -> LiveConfig | None:
        return None

    def evict(self) -> None:
        self._lens.evict_input()

    def adopt(
        self,
        lens: Any,
        sidecar: Mapping[str, Any] | None,
        *,
        live: LiveConfig | None = None,
        restore_live: bool = False,
    ) -> Any:
        del live, restore_live
        return self._lens.adopt_input(lens, sidecar)

class LensInstrument:
    """Session-lifetime handle for the J-lens read family."""

    family: InstrumentFamily = "lens"

    #: Gate channels a lens probe can produce: the one strength axis.
    _GATE_CHANNELS: tuple[type, ...] = (Axis,)

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        model_id: str,
        device: torch.device,
        n_layers: int,
        capture: LatestSlices,
        exclusive: ExclusiveGuard,
        capture_hidden: CaptureHidden,
        registry: ProfileStore,
        on_change: Callable[[Iterable[str]], None],
        resolve_profile: Callable[[str], Mapping[int, torch.Tensor]],
        assert_unsteered: Callable[[], None],
        replay: TokenReplay,
        whitener: Callable[[], Any],
    ) -> None:
        self.model_id = model_id
        #: The R family's (J/R) resident lens and readout primitives.
        self.runtime = LensRuntime(
            model=model, tokenizer=tokenizer, model_id=model_id,
            device=device, n_layers=n_layers,
        )
        #: The S family's (K/S) resident lens, residency only: its reads go
        #: through the token embeddings (:mod:`drowse.lens.footprint`).
        self.input_runtime = LensRuntime(
            model=model, tokenizer=tokenizer, model_id=model_id,
            device=device, n_layers=n_layers,
        )
        #: The depth rule's S→R crossover band for :meth:`read`.
        self.band = LensBand()
        # Model-level, built once on the first S-family read: the masked
        # embedding mean and covariance (fp32) every footprint z divides by.
        self._embeddings: TokenEmbeddings | None = None
        # Whitened tokens-in atom norms by (S lens identity, layer), each
        # beside the whitener they were taken in (runtime.cached_norms).
        self._footprint_norms: BoundedLRU[tuple[str | None, int], tuple[Any, torch.Tensor]] = (
            BoundedLRU(FOOTPRINT_NORM_CACHE_MAX)
        )
        # ``slens/`` probe layer weights (``w_S`` normalized over the read
        # layers), ``[L, 1]`` on the device, cached across decode steps.
        self._s_weights: BoundedLRU[tuple[str, tuple[int, ...], LensBand], torch.Tensor] = (
            BoundedLRU(FOOTPRINT_NORM_CACHE_MAX)
        )
        #: The read plane's capture buffer: the per-step reads consume its
        #: latest slices.
        self._capture = capture
        self._exclusive = exclusive
        self._capture_hidden = capture_hidden
        #: The session's profile registry: ``jlens/<word>`` directions land
        #: here and are evicted on every lens change.
        self._registry = registry
        #: Called with the lens probe names whenever registered directions or
        #: the resident lens change (prefix/analytics/probe-hash caches).
        self._on_change = on_change
        self._resolve_profile = resolve_profile
        self._assert_unsteered = assert_unsteered
        #: The session's Mahalanobis whitener (built on first use, ``None``
        #: when it cannot be): the metric every decomposition runs in.
        self._whitener = whitener
        # The loom-anchored token replay (the producing forward's capture);
        # this family supplies the lens read.
        self._replay = replay
        self._fitter = LensFitter(
            model=model, tokenizer=tokenizer, model_id=model_id,
            residency={"output": self, "input": _InputResidency(self)},
        )
        # Pinned J-lens token probes: name -> {word, token_id, layers}.  NOT
        # monitor probes — they read the lens readout channel (per-layer
        # softmax probability), not a whitened subspace coordinate.
        self.probes: dict[str, dict[str, Any]] = {}
        # Live workspace readout (enable_live): device-resident J_l subset +
        # settings, or None when off.  Runtime residency, not user intent —
        # disabling drops this dict but transported stacks stay in the
        # runtime's device cache.
        self.live: dict[str, Any] | None = None
        # THE lens-state boundary: one reentrant lock covering source
        # refresh/adoption/eviction, registry and live-state mutation,
        # and ``prepare``'s complete snapshot, so the snapshot cannot
        # tear mid-``prepare`` under a concurrent un-locked getter
        # read.  A leaf lock: nothing acquires the model-exclusive guard
        # while holding it (callers hold that first).  Reentrant because
        # adoption runs inside the refresh and ``prepare`` refreshes inside
        # its own hold.
        self.state_lock = threading.RLock()
        # The current per-generation run (idle passthrough until bind()).
        # All generation-scoped state — stash, display readings, active
        # flag, disk-identity pin — lives on it.
        self.current_run = LensRun(
            self, InstrumentBinding(family=self.family),
        )

    # ------------------------------------------------------------- run state
    # Generation-scoped state lives on the current run; these name it.

    @property
    def step_stash(self) -> dict[str, Any] | None:
        return self.current_run.step_stash

    @step_stash.setter
    def step_stash(self, value: "dict[str, Any] | None") -> None:
        self.current_run.step_stash = value

    @property
    def last_step_readings(self) -> "dict[str, ScalarReading] | None":
        return self.current_run.last_step_readings

    @last_step_readings.setter
    def last_step_readings(
        self, value: "dict[str, ScalarReading] | None",
    ) -> None:
        self.current_run.last_step_readings = value

    @property
    def active_for_generation(self) -> bool:
        return self.current_run.active

    @active_for_generation.setter
    def active_for_generation(self, value: bool) -> None:
        self.current_run.active = bool(value)

    @property
    def generation_lens(self) -> Any:
        return self.current_run.lens

    @generation_lens.setter
    def generation_lens(self, value: Any) -> None:
        self.current_run.lens = value

    @property
    def generation_lens_active(self) -> bool:
        return self.current_run.pinned

    @generation_lens_active.setter
    def generation_lens_active(self, value: bool) -> None:
        self.current_run.pinned = bool(value)

    # ------------------------------------------------------------ run lifecycle

    def prepare(self, request: ReadRequest) -> LensPrep:
        """The refresh/pin protocol step.

        Refreshes the resident lens from disk (:meth:`resident`) under pin
        demand BEFORE any plan is taken: the adoption path rewrites the
        live probe layer lists when an external replacement lens landed,
        so the capture plan AND the bind-time spec freeze must both see
        the refreshed registry (a plan taken earlier pairs the new lens
        with stale layers and KeyErrors in the transport stack).

        The prep is the **authoritative snapshot**:
        spec ``layers`` are derived from the prepared lens identity
        itself — never left to a later registry reread — and the live
        runtime dict is captured by reference, because an interleaved
        unpinned refresh (:meth:`has_compatible` runs un-locked on the
        session-info route) can adopt a *newer* disk lens inside the
        prepare→bind window and rewrite both.  ``plan``/``bind`` consume
        only this snapshot, so the run's lens, plan layers, and frozen
        specs agree by construction.

        Pin demand is the registry boolean, not plan emptiness — probes
        whose lens vanished still pin the validated-missing state so
        per-token paths never reopen the sidecar.  The refresh
        short-circuits on a bound run's pin flag, which is why prepare
        requires the prior run closed (a stale pin would suppress the
        very refresh this step exists for).
        """
        if self.current_run.bound:
            raise RuntimeError(
                "LensInstrument.prepare() on a bound run: close the prior "
                "generation's run (ReadPlane.close_runs) before preparing "
                "the next — a stale pin suppresses the lens disk refresh"
            )
        # The WHOLE snapshot is one atomic lens-state transaction:
        # items, gate demand, pin demand, the getter refresh, spec
        # derivation, the live-state reference, and the sidecar
        # fingerprint are all read under ``state_lock`` — the same lock
        # the refresh/adoption/eviction, the registry mutations, and the
        # live toggles hold.  A concurrent
        # ``has_compatible`` adoption landing between the refresh
        # and the live-state read would pair lens A's specs with lens
        # B's live device stack; a concurrent detach would split gate
        # demand from the captured roster.
        with self.state_lock:
            all_items = list(self.probes.items())
            items = [(n, s) for n, s in all_items if _spec_side(s) == "output"]
            input_items = [(n, s) for n, s in all_items if _spec_side(s) == "input"]
            names = {name for name, _spec in all_items}
            gate_hit = any(
                parse_gate_ref(key).probe in names
                for key in request.gate_keys
            )
            live_on = bool(request.live and self.live is not None)
            pin_demand = bool(
                live_on or (items and (request.final_aggregate or gate_hit))
            )
            # ``slens/`` probes are read by the final aggregate, by gates, and
            # by the live readout's per-step readings.
            input_pin = bool(
                input_items and (request.final_aggregate or gate_hit or live_on)
            )
            input_lens = self.resident_input() if input_pin else None
            input_specs = {
                name: {
                    **spec,
                    "layers": (
                        _out_layers(input_lens) if input_pin
                        else list(spec["layers"])
                    ),
                }
                for name, spec in input_items
            }
            lens = self.resident() if pin_demand else None
            # Spec layers under a pin come from the prepared identity
            # itself — the atomically consistent source for every probe
            # (the production invariant keeps registry lists identical to
            # the lens's ``out`` layers: attach records the full fitted set
            # and adoption rewrites every probe).  A lens without an ``out``
            # site (mid only) gives its probes no layers, exactly as its
            # adoption did.  A vanished lens (None) pins the
            # validated-missing state: its eviction path already zeroed the
            # registry lists the copies reflect.
            if pin_demand and lens is not None:
                layers = _out_layers(lens)
                specs = {
                    name: {**spec, "layers": list(layers)}
                    for name, spec in items
                }
            else:
                specs = {
                    name: {**spec, "layers": list(spec["layers"])}
                    for name, spec in items
                }
            specs.update(input_specs)
            identity = self.runtime.identity
            sources = {"R": self.runtime.source, "S": self.input_runtime.source}
            return LensPrep(
                family=self.family,
                request=request,
                token=next_prep_token(),
                lens=lens,
                pinned=pin_demand,
                specs=specs,
                live_state=self.live,
                fingerprint=identity,
                input_lens=input_lens,
                input_pinned=input_pin,
                sources=sources,
            )

    def bind(self, plan: InstrumentPlan, prep: InstrumentPrep) -> LensRun:
        """Bind an immutable per-generation run from a declared plan.

        The **prep supplies the pin and the spec snapshot** (``prepare``
        — see its docstring for the ordering + snapshot rationale).
        ``bind`` freezes the prep's specs (never the live registry, which
        an interleaved adoption may have rewritten since prepare) and
        installs the pin and live-state reference it is handed.  The
        frozen spec copies own their ``layers`` list — the live
        registry's lists are replaced in place by the adoption/eviction
        paths.
        """
        if not isinstance(prep, LensPrep) or prep.family != self.family:
            raise TypeError(
                "LensInstrument.bind takes the LensPrep its own prepare() "
                f"returned, got {type(prep).__name__} "
                f"(family={getattr(prep, 'family', None)!r})"
            )
        if plan.family != self.family:
            raise ValueError(
                f"LensInstrument.bind: plan family {plan.family!r} is not "
                f"{self.family!r}"
            )
        if plan.prep_token != prep.token:
            raise ValueError(
                "LensInstrument.bind: the plan was not derived from this "
                "prep (prep_token mismatch) — derive the plan from the same "
                "prepare() call"
            )
        run = LensRun(
            self,
            InstrumentBinding(
                family=self.family,
                # ``source`` stays None: the public source label lives in
                # ``active.json`` and reading it here would put a disk hit
                # on every generation; the prep's sidecar identity — taken
                # in the same atomic snapshot as the pin — is the binding's
                # cheap fingerprint (a bind-time live read could stamp a
                # concurrently adopted replacement's identity onto a run
                # pinned to the older lens).
                fingerprint=prep.fingerprint,
                specs={
                    name: {**spec, "layers": list(spec["layers"])}
                    for name, spec in prep.specs.items()
                },
            ),
            lens=prep.lens,
            pinned=prep.pinned,
            active=prep.request.live,
            live_state=prep.live_state,
            bound=True,
            input_lens=prep.input_lens,
            input_pinned=prep.input_pinned,
            sources=prep.sources,
        )
        self.current_run = run
        return run

    def close_run(self) -> None:
        """Close the current run and restore the idle passthrough run."""
        self.current_run.close()
        self.current_run = LensRun(
            self, InstrumentBinding(family=self.family),
        )

    def _measurement_specs(self) -> "dict[str, Any] | Any":
        """The spec source for measurement: a bound run's frozen binding
        (immune to concurrent spec mutation — e.g. the refresh's eviction
        path zeroing probe layers from another thread), else a copy of the
        live registry taken under the lens-state lock."""
        run = self.current_run
        if run.bound:
            return run.binding.specs
        with self.state_lock:
            return {
                name: {**spec, "layers": list(spec["layers"])}
                for name, spec in self.probes.items()
            }

    def _measurement_live(self) -> "Mapping[str, Any] | None":
        """The live-readout state for measurement: a bound run's
        prepare-time snapshot (an interleaved adoption rebuilds the
        instrument-level ``live`` against the NEW lens — the run must
        keep reading the state that matches its pin), else the live
        config."""
        run = self.current_run
        return run.live_state if run.bound else self.live

    def measurement_sources(self) -> dict[str, str | None]:
        """``{"R": label, "S": label}``: the source labels the lens
        measurement binding names (``binding.sources``), one per family.

        A bound run's prepare-time snapshot (so its envelopes name the
        lenses it pinned, live readout on or off); idle, the residents'.
        No disk read: each label was stamped when its lens became resident.
        """
        run = self.current_run
        if run.bound:
            return dict(run.sources)
        with self.state_lock:
            return {"R": self.runtime.source, "S": self.input_runtime.source}

    def _measurement_input_lens(self) -> Any:
        """The S-family lens ``slens/`` probes read: a bound run's pin, else
        the refreshed resident one."""
        run = self.current_run
        if run.bound and run.input_pinned:
            return run.input_lens
        return self.resident_input()

    def _measurement_state(self) -> "tuple[Any, dict[str, Any] | Any]":
        """The coherent ``(lens, specs)`` pair for measurement.

        Bound: the run's pin + frozen binding — lock-free, immutable by
        construction (an unpinned bound run falls back to the refresh,
        which locks internally; nothing measures on such a run anyway).
        Idle: ONE state-lock hold refreshes the resident lens and copies
        the registry, so an idle-passthrough read (an aggregate on the
        idle run, offline scoring) cannot pair lens A with a
        concurrently adopted replacement's rewritten layers.  Resolving
        the two separately — the registry un-locked, the resident lens
        under its own hold — lets scoring run with A's lens and B's
        layers, which KeyErrors in the transport stack.
        """
        run = self.current_run
        if run.bound:
            lens = run.lens if run.pinned else self.resident()
            return lens, run.binding.specs
        with self.state_lock:
            lens = self.resident()
            specs = {
                name: {**spec, "layers": list(spec["layers"])}
                for name, spec in self.probes.items()
            }
            return lens, specs

    # ------------------------------------------------------------- residency

    @property
    def resident_lens(self) -> Any:
        """The resident lens as last refreshed (no disk read)."""
        return self.runtime.lens

    @property
    def resident_identity(self) -> str | None:
        """The resident lens's artifact identity (no disk read)."""
        return self.runtime.identity

    def resident(self) -> Any:
        """The model's lens from the active source, or ``None`` when absent.

        Refreshes against disk: a removed, replaced or mismatched artifact
        is evicted along with every derived view, and a new one is loaded
        and adopted.  A generation-bound run's pinned lens short-circuits the
        refresh, so a generation reads one lens throughout.  Fit or fetch a
        lens with ``drowse lens fit|fetch``.
        """
        with self.state_lock:
            run = self.current_run
            if run.pinned:
                return run.lens
            self._refresh(load=True)
            return self.runtime.lens

    def resident_input(self) -> Any:
        """The S-family (K/S) lens from its active source, or ``None``.

        Reconciles with disk exactly as :meth:`resident` does for the R
        family; a bound run's input pin short-circuits it.
        """
        with self.state_lock:
            run = self.current_run
            if run.input_pinned:
                return run.input_lens
            self._refresh(load=True, side="input")
            return self.input_runtime.lens

    def has_compatible(self) -> bool:
        """Whether the active R-family source matches the loaded weights.

        Runs un-locked on the session-info route; a lens that is not
        resident yet is only checked, never paged in.
        """
        with self.state_lock:
            run = self.current_run
            if run.pinned:
                return run.lens is not None
            return self._refresh(load=False)

    def _side_runtime(self, side: str) -> LensRuntime:
        return self.input_runtime if side == "input" else self.runtime

    def _refresh(self, *, load: bool, side: str = "output") -> bool:
        """Reconcile one family's resident lens with its active source.

        Returns whether the active source is compatible.  ``load`` pages in
        a compatible lens that is not resident yet.
        """
        from drowse.lens.store import lens_identity, load_lens, load_lens_sidecar

        runtime = self._side_runtime(side)
        evict = self.evict_input if side == "input" else self.evict
        sidecar = load_lens_sidecar(self.model_id, side)
        if not runtime.matches(sidecar):
            if runtime.lens is not None:
                evict()
            if load and sidecar is not None:
                _log.warning(
                    "ignoring stale %s lens for %s: loaded-model "
                    "identity is missing or changed; fetch or re-fit the lens",
                    sidecar.get("kind", "?"), self.model_id,
                )
            return False
        if runtime.lens is not None and runtime.identity == lens_identity(sidecar):
            return True
        if not load and runtime.lens is None:
            return True
        loaded = load_lens(self.model_id, side)
        if loaded is None:
            evict()
            return False
        if not runtime.matches(loaded[1]):
            evict()
            if load:
                _log.warning(
                    "ignoring concurrently replaced %s lens for %s: "
                    "loaded payload does not match the live model",
                    loaded[0].kind, self.model_id,
                )
            return False
        self.adopt(loaded[0], loaded[1])
        return True

    def _require(self) -> Any:
        from drowse.lens.jlens import LensNotFittedError

        lens = self.resident()
        if lens is None:
            raise LensNotFittedError(
                f"no Jacobian lens available for {self.model_id} — run "
                f"`drowse lens fetch {self.model_id}` for a supported official "
                f"lens, or `drowse lens fit {self.model_id}`"
            )
        return lens

    def _no_input_lens(self) -> str:
        return (
            f"no S-lens available for {self.model_id} — run "
            f"`drowse lens fetch {self.model_id} rs-s` where rs-lenses publishes "
            f"one, or select a local S lens with `drowse lens use`"
        )

    def _require_input(self) -> Any:
        from drowse.lens.jlens import LensNotFittedError

        lens = self.resident_input()
        if lens is None:
            raise LensNotFittedError(self._no_input_lens())
        return lens

    def live_config(self) -> LiveConfig | None:
        """The live readout's ``(layers, uses_all_layers)``, or ``None``."""
        with self.state_lock:
            live = self.live
            if live is None:
                return None
            return list(live["layers"]), bool(live.get("uses_all_layers", False))

    def _drop_directions(self) -> None:
        self._registry.evict_prefix("jlens/")

    def _side_probes(self, side: str) -> dict[str, dict[str, Any]]:
        return {
            name: spec for name, spec in self.probes.items()
            if _spec_side(spec) == side
        }

    def evict(self) -> None:
        """Drop the resident R-family lens and every derived view: registered
        ``jlens/`` directions, ``jlens/`` probe layer lists, live state, the
        stash."""
        with self.state_lock:
            self.runtime.clear()
            self.step_stash = None
            self.last_step_readings = None
            self._drop_directions()
            for spec in self._side_probes("output").values():
                spec["layers"] = []
            self.live = None
            self._on_change(list(self.probes))

    def evict_input(self) -> None:
        """Drop the resident S-family lens and its derived views: registered
        ``slens/`` directions, ``slens/`` probe layer lists, the input stash."""
        with self.state_lock:
            self.input_runtime.clear()
            self.current_run.input_stash = None
            self._registry.evict_prefix("slens/")
            for spec in self._side_probes("input").values():
                spec["layers"] = []
            self._on_change(list(self.probes))

    def adopt(
        self,
        lens: Any,
        sidecar: Mapping[str, Any] | None,
        *,
        live: LiveConfig | None = None,
        restore_live: bool = False,
    ) -> Any:
        """Make ``lens`` resident in its family's slot and rebuild every
        derived consumer.

        One lens-state transaction, so a concurrent ``prepare`` snapshot
        sees the old world or the new one, never a mix.  For the R family
        the live readout is rebuilt from its current configuration, or from
        ``live`` when ``restore_live`` (a fit restores the configuration it
        evicted).  An S-family (K/S) lens goes to :meth:`adopt_input`.
        """
        key = getattr(lens, "key", None)
        if key is not None and key.side == "input":
            return self.adopt_input(lens, sidecar)
        with self.state_lock:
            previous = live if restore_live else self.live_config()
            self.runtime.install(lens, sidecar, source=self._active_label("output"))
            self.step_stash = None
            self.last_step_readings = None
            self._drop_directions()
            readout_layers = _out_layers(lens)
            output_probes = self._side_probes("output")
            for spec in output_probes.values():
                spec["layers"] = list(readout_layers)
            if output_probes and readout_layers:
                self.runtime.transport_stack(lens, sorted(readout_layers))
            if previous is not None and readout_layers:
                layers, uses_all_layers = previous
                valid = [layer for layer in layers if layer in readout_layers]
                self.enable_live(
                    layers=None if uses_all_layers else (valid or None),
                )
            else:
                self.live = None
            self._on_change(list(self.probes))
            return lens

    def adopt_input(self, lens: Any, sidecar: Mapping[str, Any] | None) -> Any:
        """Make an S-family lens resident: ``slens/`` directions are dropped
        (they re-resolve against it) and ``slens/`` probes take its ``out``
        layers."""
        with self.state_lock:
            self.input_runtime.install(lens, sidecar, source=self._active_label("input"))
            self.current_run.input_stash = None
            self._registry.evict_prefix("slens/")
            layers = _out_layers(lens)
            for spec in self._side_probes("input").values():
                spec["layers"] = list(layers)
            self._on_change(list(self.probes))
            return lens

    def select_source(self, source: str) -> None:
        """Select a prepared local/external lens source into its family's
        slot and evict that family's resident lens.  Prepare the source first
        (fit or fetch)."""
        from drowse.lens.sources import lens_source_side, use_lens_source

        ref = use_lens_source(self.model_id, source)
        if lens_source_side(ref) == "input":
            self.evict_input()
        else:
            self.evict()

    def close(self) -> None:
        """Session teardown: close the run and drop every resident view."""
        self.close_run()
        with self.state_lock:
            self.probes.clear()
            self.live = None
            self._embeddings = None
            self._footprint_norms.clear()
        self.runtime.close()
        self.input_runtime.close()

    # ------------------------------------------------------------- lifecycle

    def fit(
        self,
        prompts: Sequence[str],
        *,
        kind: str = "J",
        sites: Sequence[str] = ("out", "mid"),
        corpus_spec: str = "custom",
        layers: LayerSelector = None,
        dim_batch: int | None = None,
        seq_len: int | None = None,
        force: bool = False,
        checkpoint_every: int | None = None,
        split_half: bool = True,
        on_progress: Callable[[str], None] | None = None,
        cancel_event: Any | None = None,
    ) -> Any:
        """Fit (or resume) this model's ``kind`` lens at ``sites``.

        One transport estimator fits every kind: ``J`` (standard, the
        library default) and ``R`` (RelP) read what an activation makes the model say,
        ``S`` (RelP) and ``K`` (standard) what put it there; each lands at
        ``local:<kind>``.  ``sites`` is ``out``, ``mid`` or both (default).
        Resume is the default: a matching durable artifact or checkpoint over
        a prefix of this corpus is extended; ``force`` restarts from zero.
        ``layers`` is ``None``/``"all"``, ``"workspace"`` or an index list,
        resolved per site.  Runs under the fit lock and the model-exclusive
        guard, unsteered; a fit holding ``out`` is selected and becomes its
        family's resident lens (the other family's is untouched).  Returns
        the fitted lens restricted to the requested sites and layers.
        """
        from drowse.lens.jlens import (
            DEFAULT_CHECKPOINT_EVERY,
            DEFAULT_DIM_BATCH,
            DEFAULT_SEQ_LEN,
            JacobianLensCancelled,
        )
        from drowse.lens.store import cleanup_lens_artifacts, lens_fit_lock

        request = LensFitRequest(
            kind=kind,
            sites=tuple(sites),
            corpus_spec=corpus_spec,
            layers=layers,
            dim_batch=dim_batch or DEFAULT_DIM_BATCH,
            seq_len=seq_len or DEFAULT_SEQ_LEN,
            force=force,
            checkpoint_every=checkpoint_every or DEFAULT_CHECKPOINT_EVERY,
            split_half=split_half,
        )
        name = request.kind.lower()
        with lens_fit_lock(self.model_id, name):
            cleanup_lens_artifacts(self.model_id, name)
            if cancel_event is not None and cancel_event.is_set():
                raise JacobianLensCancelled("lens fit cancelled before start")
            with self._exclusive(
                "session.lens.fit called while another model use is in flight",
                phase_msg="session.lens.fit called while a generation is in flight",
            ):
                self._assert_unsteered()
                return self._fitter.run(
                    prompts, request,
                    on_progress=on_progress, cancel_event=cancel_event,
                )

    def fit_is_noop(
        self,
        prompt_count: int,
        *,
        kind: str = "J",
        sites: Sequence[str] = ("out", "mid"),
        corpus_spec: str,
        layers: LayerSelector = None,
        seq_len: int | None = None,
    ) -> Mapping[str, Any] | None:
        """Prove :meth:`fit` over the first ``prompt_count`` documents of the
        pinned Hub corpus ``corpus_spec`` would be a no-op, before streaming
        it (:func:`~drowse.lens.lifecycle.prove_lens_fit_noop` against this
        model's exact source).  Returns the proven sidecar, or ``None`` when
        a fit would do work.
        """
        from drowse.lens.lifecycle import prove_lens_fit_noop

        request = LensFitRequest(
            kind=kind, sites=tuple(sites), corpus_spec=corpus_spec, layers=layers,
        )
        return prove_lens_fit_noop(
            self.model_id, request.kind.lower(),
            requested=request.layers,
            sites=request.sites,
            seq_len=seq_len,
            source_fingerprint=self._fitter.source_fingerprint,
            corpus_spec=request.corpus_spec,
            prompt_count=int(prompt_count),
        )

    # ---------------------------------------------------------- offline reads

    @overload
    def readout(
        self,
        prompt: str,
        *,
        layers: LayerSelector = None,
        positions: Sequence[int] | None = None,
        top_k: int = 10,
        aggregate: Literal[False] = False,
    ) -> dict[int, list[list[tuple[str, float]]]]: ...

    @overload
    def readout(
        self,
        prompt: str,
        *,
        layers: LayerSelector = None,
        positions: Sequence[int] | None = None,
        top_k: int = 10,
        aggregate: Literal[True],
    ) -> tuple[
        dict[int, list[list[tuple[str, float]]]],
        list[list[tuple[str, float, float, float]]],
    ]: ...

    def readout(
        self,
        prompt: str,
        *,
        layers: LayerSelector = None,
        positions: Sequence[int] | None = None,
        top_k: int = 10,
        aggregate: bool = False,
    ) -> (
        dict[int, list[list[tuple[str, float]]]]
        | tuple[
            dict[int, list[list[tuple[str, float]]]],
            list[list[tuple[str, float, float, float]]],
        ]
    ):
        """Lens readout on a raw prompt: the top-``top_k`` tokens per
        (layer, position) with their per-layer probabilities ``p_l``.

        ``layers`` defaults to every fitted layer; ``positions`` to the
        final position (explicit indices, negative from the end, read
        more).  Returns ``{layer: [per-position [(token, p_l), ...]]}``.
        ``aggregate=True`` also returns one layer-aggregated chip list
        ``[(token, strength, com, spread), ...]`` per position, from the
        same logits (:func:`drowse.lens.jlens.aggregate_readout`).
        """
        lens = self._require()
        self.runtime.resolve_layers(lens, layers)
        with self._exclusive(
            "session.lens.readout called while another model use is in flight",
            phase_msg="session.lens.readout called while a generation is in flight",
        ):
            return self.runtime.readout(
                lens, prompt,
                capture_hidden=self._capture_hidden,
                layers=layers, positions=positions, top_k=top_k,
                aggregate=aggregate,
            )

    def decompose(
        self,
        selector: str,
        *,
        k: int = 16,
        layers: LayerSelector = None,
        dictionary: str = "tokens-out",
    ) -> dict[int, tuple[float, list[tuple[str, float]]]]:
        """Split a steerable direction against a token dictionary, per layer.

        ``selector`` resolves through the ordinary steering resolver, so any
        steerable vector decomposes.  ``dictionary`` is ``tokens-out``
        (``W_U J_l`` through the R-family lens: the tokens the direction is
        disposed to say) or ``tokens-in`` (``K_l ẽ_v`` through the S-family
        lens's ``out`` site, over the token mask: the tokens whose having
        been read it resembles).  Returns ``{layer: (share, [(token,
        coeff), ...])}``: ``share`` is the fraction of the direction's
        variance carried by its best sparse nonnegative combination of ``k``
        atoms (for ``tokens-out``, how *verbalizable* it is; ~6-15% is
        typical for concept vectors).  Selection, fit and share are all in
        the session whitener's Mahalanobis metric; a whitener that does not
        cover a requested layer raises ``WhitenerError``.
        """
        from drowse.lens.decompose import DICTIONARIES

        if dictionary not in DICTIONARIES:
            raise RequestError(
                f"dictionary must be one of {', '.join(DICTIONARIES)}, got {dictionary!r}"
            )
        if dictionary == "tokens-in":
            return self._decompose_tokens_in(selector, k=k, layers=layers)
        lens = self._require()
        directions = self._resolve_profile(selector)
        return self.runtime.decompose(
            lens, directions, label=selector, whitener=self._whitener(), k=k, layers=layers,
        )

    def _decompose_tokens_in(
        self, selector: str, *, k: int, layers: LayerSelector,
    ) -> dict[int, tuple[float, list[tuple[str, float]]]]:
        from drowse.lens.decompose import TokensIn, sparse_decompose
        from drowse.lens.runtime import cached_norms, require_decompose_whitener
        from drowse.model.layers import resolve_layer_selector

        lens = self._require_input()
        fitted = lens.layers("out")
        directions = self._resolve_profile(selector)
        req = [
            layer
            for layer in resolve_layer_selector(
                layers, fitted, n_layers=self.input_runtime.n_layers,
            )
            if layer in directions and layer in fitted
        ]
        if not req:
            raise RequestError(
                f"no overlap between {selector!r}'s layers and the S lens's "
                f"out layers"
            )
        metric = require_decompose_whitener(self._whitener(), req)
        with self._exclusive(
            "session.lens.decompose called while another model use is in flight",
            phase_msg="session.lens.decompose called while a generation is in flight",
        ), torch.inference_mode():
            embeddings = self._token_embeddings()
            out: dict[int, tuple[float, list[tuple[str, float]]]] = {}
            identity = self.input_runtime.identity
            for layer in req:
                matrix = lens.matrix(layer, "out")
                atoms = TokensIn(
                    matrix, embeddings, metric, layer,
                    atom_norms=cached_norms(self._footprint_norms, (identity, layer), metric),
                )
                dec = sparse_decompose(directions[layer], atoms, layer=layer, k=k)
                self._footprint_norms.put((identity, layer), (metric, atoms.norms()))
                out[layer] = (
                    dec.share,
                    [(self.runtime.decode_id(tok), coeff) for tok, coeff in dec.atoms],
                )
        return out

    def register_direction(self, word: str) -> str:
        """Register ``W_U[v] @ J_l`` for a single-token word (or ``#<id>``)
        as the profile ``jlens/<word>`` and return its name.

        The direction steers and ablates exactly like an extracted vector;
        this is the resolver behind the lazy ``jlens/`` steering atom.  A
        ``jlens/<word>`` *probe* reads the readout channel instead.
        Idempotent; every fitted layer.
        """
        name = f"jlens/{word}"
        if name in self._registry.profiles:
            return name
        lens = self._require()
        self._registry.put_profile(name, self.runtime.token_directions(lens, word))
        self._on_change(())
        return name

    def register_input_direction(self, word: str) -> str:
        """Register the footprint ``K_l ẽ_v`` of a single-token word (or
        ``#<id>``) as the profile ``slens/<word>`` and return its name.

        The S-family lens's ``out`` site, every layer it holds: steering
        injects at block outputs, so the atom pushes the residual toward
        "having read" the token there.  Like ``jlens/`` atoms it is an
        ordinary profile (it pushes, ablates, takes a ``{…}`` layer clause and
        goes through the whitener at composition).  Idempotent.

        Runs in the model-exclusive section: the first S use builds the
        embedding statistics (model work).  The footprints themselves are
        CPU fp32 against the lens's CPU matrices, then land on the session
        device, as ``jlens/`` directions do.  A token the S read's mask drops
        is refused (:meth:`_input_token`).
        """
        name = f"slens/{word}"
        if name in self._registry.profiles:
            return name
        with self._exclusive(
            "an slens/ steering atom was resolved while another model "
            "operation is in flight; retry shortly"
        ):
            lens = self._require_input()
            layers = lens.layers("out")
            embeddings = self._token_embeddings()
            token_id = self._input_token(word, embeddings)
            with torch.inference_mode():
                footprint = embeddings.centred([token_id])[0].cpu()
                directions = {
                    int(layer): lens.footprint(footprint, int(layer), "out").to(
                        self.runtime.device,
                    )
                    for layer in layers
                }
        self._registry.put_profile(name, directions)
        self._on_change(())
        return name

    def _input_token(self, word: str, embeddings: TokenEmbeddings) -> int:
        """The id an ``slens/<word>`` atom or probe names, refused when the S
        read's token mask drops it: such a token never ranks in an S read, so
        a footprint z for it measures nothing the read can show."""
        from drowse.lens.jlens import InvalidTokenIdError

        token_id = self.resolve_token(word, side="input")
        if not bool(embeddings.keep[token_id]):
            raise InvalidTokenIdError(
                f"slens/{word} names token {token_id}, which the S read's token "
                f"mask drops (special, added and byte-fallback pieces and "
                f"embedding padding rows never rank in an S read)"
            )
        return token_id

    def _token_embeddings(self) -> TokenEmbeddings:
        """The masked embedding statistics, built once per model.

        Model work on the first call (the block description is discovered
        to read the embedding scale): call it inside the exclusive section.
        """
        cached = self._embeddings
        if cached is not None:
            return cached
        from drowse.model.blocks import model_spec

        base = getattr(self.runtime.model, "_orig_mod", self.runtime.model)
        spec = model_spec(base)
        weight = spec.embed.weight
        keep = token_mask(self.runtime.tokenizer, int(weight.shape[0]))
        with torch.inference_mode():
            built = TokenEmbeddings.build(weight, spec.embed_scale, keep)
        with self.state_lock:
            if self._embeddings is None:
                self._embeddings = built
            return self._embeddings

    # ------------------------------------------------------------------ reads

    def read(
        self,
        vector: torch.Tensor,
        layer: int,
        site: str = "out",
        *,
        k: int = 8,
    ) -> LensRead | list[LensRead]:
        """Read a vector (``[d]``) or rows (``[n, d]``) at ``(layer, site)``.

        The four-lens rule (:mod:`drowse.lens.read`): the caller picks the
        site from where the vector sits (``out``: a block's output residual,
        a neuron's write, a head's inputs at ``L − 1``; ``mid``: a neuron's
        read direction, a head's write); the result holds the R-family read
        (top-``k`` readout probabilities through the selected J/R lens) and
        the S-family read (top-``k`` footprint z through the selected K/S
        lens), each with its depth weight and lens label.  A family with no
        lens, no lens at ``site`` or no matrix at ``layer`` comes back
        absent with the reason; nothing is read from the other site.

        Returns one :class:`~drowse.lens.read.LensRead` for a ``[d]`` vector
        and a list for ``[n, d]`` rows.  Takes the model-exclusive section
        (the first S read builds the embedding statistics).
        """
        if site not in LENS_SITES:
            raise RequestError(f"site must be one of {', '.join(LENS_SITES)}, got {site!r}")
        n_layers = self.runtime.n_layers
        if not 0 <= int(layer) < n_layers:
            raise RequestError(f"layer {layer} is outside the model's {n_layers} layers")
        if not 1 <= int(k) <= 256:
            raise RequestError("k must be in [1, 256]")
        if vector.ndim not in (1, 2):
            raise RequestError(
                f"read takes a [d] vector or [n, d] rows, got shape {tuple(vector.shape)}"
            )
        width = self._residual_width()
        if int(vector.shape[-1]) != width:
            raise RequestError(
                f"read takes residual-width vectors (d = {width}), got shape "
                f"{tuple(vector.shape)}"
            )
        r_lens = self.resident()
        s_lens = self.resident_input()
        with self._exclusive(
            "session.lens.read called while another model use is in flight",
            phase_msg="session.lens.read called while a generation is in flight",
        ), torch.inference_mode():
            reads = read_vectors(
                vector.detach(), int(layer), site,
                r_lens=r_lens, s_lens=s_lens,
                readout=lambda: self.runtime.readout_modules()[1],
                embeddings=self._token_embeddings,
                decode=self.runtime.decode_id,
                n_layers=n_layers,
                device=self.runtime.device,
                k=int(k),
                band=self.band,
                no_r_lens=(
                    f"no R-family lens is selected for {self.model_id} "
                    f"(`drowse lens fetch {self.model_id}` or `drowse lens fit`)"
                ),
                no_s_lens=self._no_input_lens(),
            )
        return reads[0] if vector.ndim == 1 else reads

    def _residual_width(self) -> int:
        """The model's residual width ``d`` (the text config's hidden size,
        else the input embedding's width)."""
        base = getattr(self.runtime.model, "_orig_mod", self.runtime.model)
        config = getattr(base, "config", None)
        text = getattr(config, "text_config", None) or config
        hidden = getattr(text, "hidden_size", None)
        if isinstance(hidden, int) and not isinstance(hidden, bool):
            return hidden
        return int(base.get_input_embeddings().weight.shape[1])

    def read_prompt(
        self,
        prompt: str,
        *,
        site: str = "out",
        layers: LayerSelector = None,
        positions: Sequence[int] | None = None,
        k: int = 8,
    ) -> dict[int, list[LensRead]]:
        """Read a raw prompt's residuals at ``site`` through the depth rule.

        One forward; ``out`` reads each block's output, ``mid`` the residual
        after its attention (the MLP's input).  ``layers`` defaults to every
        layer either family's lens holds at ``site``; ``positions`` to the
        final position.  Returns ``{position: [LensRead per layer]}``.
        """
        if site not in LENS_SITES:
            raise RequestError(f"site must be one of {', '.join(LENS_SITES)}, got {site!r}")
        from drowse.model.layers import resolve_layer_selector

        held: set[int] = set()
        for lens in (self.resident(), self.resident_input()):
            if lens is not None and site in getattr(lens, "sites", ("out",)):
                held.update(int(layer) for layer in lens.layers(site))
        if not held:
            from drowse.lens.jlens import LensNotFittedError

            raise LensNotFittedError(
                f"no lens of either family holds the {site!r} site for "
                f"{self.model_id} — fetch one (`drowse lens fetch "
                f"{self.model_id} rs-r|rs-s`) or fit one"
            )
        req = resolve_layer_selector(layers, sorted(held), n_layers=self.runtime.n_layers)
        runtime = self.runtime
        ids = runtime.tokenizer(prompt, return_tensors="pt")["input_ids"].to(runtime.device)
        n_pos = int(ids.shape[1])
        pos = [n_pos - 1] if positions is None else [
            p if p >= 0 else n_pos + p for p in positions
        ]
        if any(not 0 <= p < n_pos for p in pos):
            raise RequestError(f"positions must index the prompt's {n_pos} tokens")
        with self._exclusive(
            "session.lens.read called while another model use is in flight",
            phase_msg="session.lens.read called while a generation is in flight",
        ):
            if site == "out":
                hidden = self._capture_hidden(ids, layer_indices=req, pool_index=None)
                rows = {layer: hidden[layer][0, pos, :] for layer in req}
            else:
                rows = self._capture_mid(ids, req, pos)
        out: dict[int, list[LensRead]] = {p: [] for p in pos}
        for layer in req:
            reads = self.read(rows[layer], layer, site, k=k)
            for p, read in zip(pos, cast("list[LensRead]", reads), strict=True):
                out[p].append(read)
        return out

    def _capture_mid(
        self, ids: torch.Tensor, layers: Sequence[int], positions: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        """The residual after each layer's attention (the MLP's input) at
        ``positions``: a pre-hook on the block's MLP input norm, one forward.

        The caller holds the model-exclusive section.
        """
        from drowse.model.blocks import model_spec

        base = getattr(self.runtime.model, "_orig_mod", self.runtime.model)
        spec = model_spec(base)
        wanted = {int(layer) for layer in layers}
        index = torch.tensor(list(positions), device=ids.device, dtype=torch.long)
        rows: dict[int, torch.Tensor] = {}
        handles = []

        def tap(layer: int) -> Callable[..., None]:
            def hook(_module: Any, args: tuple[Any, ...]) -> None:
                rows[layer] = args[0][0].index_select(0, index.to(args[0].device)).detach()
            return hook

        try:
            for block in spec.blocks:
                if block.layer in wanted:
                    handles.append(
                        block.mlp_in_norm.module.register_forward_pre_hook(tap(block.layer))
                    )
            with torch.inference_mode():
                self.runtime.model(input_ids=ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return rows

    # -------------------------------------------------------------- registry

    def attach(self, selector: str, *, as_name: str | None = None) -> str:
        """Attach a ``jlens/<word>`` or ``slens/<word>`` token probe (or
        ``#<id>``).

        Validates the lens artifact + token (the same ``resolve_token_atom``
        contract steering atoms use), records the lens's ``out`` layers, and
        pre-warms the device stack so the first decode step doesn't hitch on
        the transfer.  The probe reads ONE channel: a ``jlens/`` probe the
        mean layer probability ``mean_l p_l(v)``, an ``slens/`` probe the
        mean footprint z ``mean_l z_l(v)`` (S family; its first attach builds
        the embedding statistics).  Cache invalidation (probe-hash cache,
        analytics) is the probe roster's job
        (:class:`~drowse.readout.roster.ProbeRoster`).
        """
        namespace, _, word = selector.partition("/")
        side = _ATOM_SIDES.get(namespace)
        if side is None:
            raise RequestError(f"{selector!r} is not a jlens/ or slens/ probe")
        if not word:
            raise RequestError(f"empty {namespace} probe word")
        name = as_name if as_name is not None else selector
        runtime = self._side_runtime(side)
        with self._exclusive(
            "add_probe called while another model operation is in "
            "flight; retry shortly"
        ):
            # Lens-state boundary inside the exclusive section (consistent
            # outer→leaf lock order): the lens read and the registry write
            # land in one lens-state transaction.
            with self.state_lock:
                if side == "input":
                    lens = self._require_input()
                    readout_layers = [int(layer) for layer in lens.layers("out")]
                    token_id = self._input_token(word, self._token_embeddings())
                else:
                    lens = self._require()
                    token_id = self.resolve_token(word, side=side)
                    readout_layers = [int(l) for l in lens.source_layers]
                runtime.transport_stack(lens, sorted(readout_layers))
                self.probes[name] = {
                    "word": word,
                    "token_id": int(token_id),
                    "layers": readout_layers,
                    "side": side,
                }
        return name

    def resolve_token(self, word: str, *, side: str = "output") -> int:
        """The vocab id a ``jlens/<word>`` (``side="output"``) or
        ``slens/<word>`` (``side="input"``) atom names: a single-token word,
        or ``#<id>`` addressing the id directly (the steering-atom contract),
        bounds-checked against the unembedding or the input embedding."""
        runtime = self.runtime
        if side == "input":
            rows = int(self.runtime.model.get_input_embeddings().weight.shape[0])
        else:
            rows = int(runtime.readout_modules()[0].shape[0])
        return resolve_token_atom(runtime.tokenizer, word, vocab_size=rows)

    def try_detach(self, name: str) -> bool:
        """Atomic membership-check + detach under the lens-state lock — the
        family's only removal surface.

        The session's ``remove_probe`` dispatch routes lens removals here: a
        bare ``name in probes`` check followed by a delete is two
        un-serialized registry touches, so membership and removal have to
        land inside one hold.  Returns False when the name isn't a lens
        probe.
        """
        with self.state_lock:
            if name not in self.probes:
                return False
            del self.probes[name]
            return True

    def specs(self) -> dict[str, dict[str, Any]]:
        """Snapshot of attached probe specifications (one coherent
        lens-state read — a concurrent detach or adoption cannot tear
        the iteration or mix old/new layer lists)."""
        with self.state_lock:
            return {name: dict(spec) for name, spec in self.probes.items()}

    @property
    def names(self) -> list[str]:
        with self.state_lock:
            return list(self.probes)

    def probe_layers(self, names: set[str] | None = None) -> set[int]:
        """Union of the attached probes' fitted layers (coherent read)."""
        with self.state_lock:
            out: set[int] = set()
            probes = self.probes
            if names is not None:
                probes = {
                    name: probes[name] for name in names if name in probes
                }
            for spec in probes.values():
                out.update(spec["layers"])
            return out

    def validate_gate(self, ref: GateRef) -> None:
        validate_gate_channels(ref, self._GATE_CHANNELS, family=self.family)

    # ---------------------------------------------------------------- planning

    def plan(self, prep: InstrumentPrep) -> InstrumentPlan:
        """Declare the lens family's capture demand for one generation.

        Derived **solely from the prep** — the spec snapshot whose layers
        match the prepared lens identity, and the live state captured at
        prepare — never the live registry, which an interleaved adoption
        may have rewritten since: a plan read off the rewritten registry
        pairs the prep's older lens with the new lens's layers and
        KeyErrors in the transport stack.

        ``latest_layers`` — the live workspace readout's layer set (plus the
        ``slens/`` probes' band, which the live step reads) and the
        pinned probes' fitted band (full band when a finalize aggregate
        will pool it; the gated probes' band when gates are the only
        per-step consumer with final readings disabled).  ``tail_layers``
        is the finalize-pooling demand: pinned probes' aggregates pool the
        last content token from the capture tail ring, which must span the
        probe band at ring depth ``AGG_TAIL_DEPTH``.
        """
        if not isinstance(prep, LensPrep) or prep.family != self.family:
            raise TypeError(
                "LensInstrument.plan takes the LensPrep its own prepare() "
                f"returned, got {type(prep).__name__}"
            )
        request = prep.request
        probes: Mapping[str, Mapping[str, Any]] = prep.specs
        live = prep.live_state if request.live else None
        gate_keys = frozenset(
            key for key in request.gate_keys
            if parse_gate_ref(key).probe in probes
        )

        def _band(names: "set[str] | None" = None) -> set[int]:
            return {
                int(layer)
                for name, spec in probes.items()
                if names is None or name in names
                for layer in spec["layers"]
            }

        latest: set[int] = set()
        tail: set[int] = set()
        if live is not None:
            latest.update(int(layer) for layer in live["layers"])
            # The live step also reads every ``slens/`` probe, over its own
            # S lens's layers, not the R live selection.
            latest.update(_band({
                name for name, spec in probes.items() if _spec_side(spec) == "input"
            }))
        if probes and request.final_aggregate:
            band = _band()
            latest.update(band)
            tail.update(band)
        elif gate_keys:
            # Gate-only pinned probes need per-step latest slices, but
            # dormant probes must not keep capture alive when the caller
            # disabled final probe readings.
            gated_names = {parse_gate_ref(key).probe for key in gate_keys}
            latest.update(_band(gated_names))
        return InstrumentPlan(
            family=self.family,
            latest_layers=frozenset(latest),
            tail_layers=frozenset(tail),
            gate_keys=gate_keys,
            final_aggregate=bool(probes and request.final_aggregate),
            prep_token=prep.token,
        )

    def probe_hash(self, name: str) -> str | None:
        """Readout-channel identity digest (no baked tensor exists).

        The channel is the resident lens read through the model's readout
        (output bias, projection, scaling, final soft-capping), so the digest
        folds in the lens artifact's identity: the same word read through a
        different lens — another source, a re-fit — is a different probe.
        """
        with self.state_lock:
            spec = self.probes.get(name)
            if spec is None:
                return None
            if _spec_side(spec) == "input":
                # The footprint channel: the S lens identity and the token
                # mask the z divides by (the tokenizer is the model's).
                head: tuple[Any, ...] = ("slens-footprint-z-v1", self.input_runtime.identity)
            else:
                head = ("jlens-readout-v5", self.runtime.identity)
            return hashlib.sha256(
                repr(
                    (
                        head[0], self.model_id,
                        spec["word"], spec["token_id"],
                        tuple(spec["layers"]),
                        head[1],
                    )
                ).encode("utf-8")
            ).hexdigest()

    # ---------------------------------------------------------------- scoring

    def score_probes(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        only: "set[str] | None" = None,
    ) -> dict[str, ScalarReading]:
        """Score every attached probe from capture hidden slices.

        The capture-slice entry: computes the lens logits itself over the
        probes' fitted layers that ``hidden`` covers.  Callers that already
        hold a calibrated row matrix use :meth:`score_probes_from_rows`.
        Empty when no probe layer is available.
        """
        lens, all_probes = self._measurement_state()
        out = self._input_readings(all_probes, hidden, only=only)
        probes = {n: s for n, s in all_probes.items() if _spec_side(s) == "output"}
        if not probes or lens is None:
            return out
        readout_layers: set[int] = set()
        for spec in probes.values():
            readout_layers.update(spec["layers"])
        layers = sorted(l for l in readout_layers if l in hidden)
        if not layers:
            return out
        logits = self.runtime.logits_rows(
            lens, [(l, hidden[l]) for l in layers],
        )
        out.update(self._readings_from_rows(
            probes, layers, logits=logits, only=only,
        ))
        return out

    def _input_block(
        self,
        probes: "Mapping[str, Mapping[str, Any]]",
        hidden: "Mapping[int, torch.Tensor]",
        *,
        only: "set[str] | None" = None,
    ) -> "_InputBlock | None":
        """The ``slens/`` probes' footprint z from block-output rows, still on
        the device (:meth:`_input_readings` unpacks it; the live step folds it
        into its own host transfer).

        Per layer ``u_L = K_Lᵀ h_L`` (one batched pullback over the S lens's
        ``out`` layers that ``hidden`` covers), then each probe token's
        ``z_L = u_L·ẽ_v / sqrt(u_Lᵀ Σ_E u_L)``: ``O(d²)`` per layer, no pass
        over the vocabulary.  The value is ``Σ_L w_S(L) z_L / Σ_L w_S(L)``,
        ``w_S`` the depth rule's S weight (:func:`~drowse.lens.depth.kind_weights`
        under :attr:`band`), so a probe reads where S is the faithful lens; it
        is 0 when no read layer has S weight.  The depth summary weights each
        layer by ``w_S(L) max(z_L, 0)``; ``per_layer`` keeps every ``z_L``.
        Layers come from the lens read here, so a reading never pairs one lens
        with another's layers.  ``None`` when there is nothing to read.
        """
        names = [
            name for name, spec in probes.items()
            if _spec_side(spec) == "input" and (only is None or name in only)
        ]
        if not names:
            return None
        lens = self._measurement_input_lens()
        embeddings = self._embeddings
        if lens is None or embeddings is None:
            return None
        layers = [layer for layer in _out_layers(lens) if layer in hidden]
        if not layers:
            return None
        runtime = self.input_runtime
        stack = runtime.transport_stack(lens, layers, embeddings.device)
        rows = torch.stack([hidden[layer].reshape(-1) for layer in layers])
        u = pullback_rows(stack, rows)                                  # [L, d_emb]
        token_ids = runtime.long_tensor(
            [int(probes[name]["token_id"]) for name in names], embeddings.device,
        )
        z = embeddings.z_of(u, token_ids)                               # [L, P]
        depth = runtime.depth_tensor(layers, z.device)                   # [L, 1]
        weights = self._s_weight_tensor(layers, z.device)                # [L, 1]
        values = footprint_probe_rows(z, depth, weights).detach()       # [3 + L, P]
        return _InputBlock(tuple(names), tuple(layers), values)

    def _s_weight_tensor(self, layers: "Sequence[int]", device: torch.device) -> torch.Tensor:
        """``[L, 1]`` fp32 ``w_S(L) / Σ w_S`` over ``layers`` (zeros when no
        layer has S weight), built once per layer set, band and device."""
        layer_tuple = tuple(int(layer) for layer in layers)
        band = self.band

        def build() -> torch.Tensor:
            weights = s_weights(layer_tuple, self.input_runtime.n_layers, band)
            return torch.tensor(weights, dtype=torch.float32, device=device).reshape(-1, 1)

        return self._s_weights.get_or_build((str(device), layer_tuple, band), build)

    def _input_readings(
        self,
        probes: "Mapping[str, Mapping[str, Any]]",
        hidden: "Mapping[int, torch.Tensor]",
        *,
        only: "set[str] | None" = None,
    ) -> dict[str, ScalarReading]:
        """The ``slens/`` probes' footprint-z readings (:meth:`_input_block`),
        in one host transfer."""
        block = self._input_block(probes, hidden, only=only)
        return {} if block is None else block.readings(block.values.cpu())

    def _input_stash_split(
        self,
        probes: "Mapping[str, Mapping[str, Any]]",
        only: "set[str] | None",
        step_id: int,
    ) -> tuple[dict[str, ScalarReading], set[str]]:
        """This forward's stashed ``slens/`` readings among the wanted probes,
        and the wanted probes not read on this forward yet.

        The stash records which probes were read (whether or not a reading
        came out), so a display step after a gate that read a subset reads
        only the rest, never drops it."""
        wanted = {
            name for name, spec in probes.items()
            if _spec_side(spec) == "input" and (only is None or name in only)
        }
        stash = self.current_run.input_stash
        if stash is None or step_id < 0 or stash.get("step") != step_id:
            return {}, wanted
        have = {n: r for n, r in stash["readings"].items() if n in wanted}
        return have, wanted - stash["names"]

    def _input_stash_put(
        self, step_id: int, names: "set[str]", readings: "Mapping[str, ScalarReading]",
    ) -> None:
        """Record ``names`` as read on forward ``step_id`` (their readings)."""
        if step_id < 0:
            return
        run = self.current_run
        stash = run.input_stash
        if stash is None or stash.get("step") != step_id:
            stash = {"step": step_id, "names": set(), "readings": {}}
            run.input_stash = stash
        stash["names"].update(names)
        stash["readings"].update(readings)

    def _input_step_readings(
        self,
        probes: "Mapping[str, Mapping[str, Any]]",
        only: "set[str] | None",
        *,
        step_id: int,
    ) -> dict[str, ScalarReading]:
        """``slens/`` readings for this forward, stashed by ``step_id`` so
        the display step reuses them (the same staleness rule as the R
        family's stash); probes the stash has not read yet are read now."""
        have, missing = self._input_stash_split(probes, only, step_id)
        if missing:
            latest = self._capture.latest_per_layer()
            fresh = self._input_readings(probes, latest, only=missing) if latest else {}
            self._input_stash_put(step_id, missing, fresh)
            have.update(fresh)
        return have

    def score_probes_from_rows(
        self,
        *,
        layers: "Sequence[int]",
        logits: torch.Tensor | None = None,
        probabilities: torch.Tensor | None = None,
        only: "set[str] | None" = None,
    ) -> dict[str, ScalarReading]:
        """Score attached probes from precomputed lens rows aligned to
        ``layers`` — exactly one of ``logits`` / ``probabilities``.

        The entry for callers that already calibrated this forward's matrix
        (the gate callback, the live display step, authored prefill): rows are
        restricted to the probes' fitted layer set, never recomputed.
        """
        if (logits is None) == (probabilities is None):
            raise ValueError(
                "score_probes_from_rows takes lens logits OR probabilities"
            )
        _lens, all_probes = self._measurement_state()
        # R-family rows score ``jlens/`` probes only; ``slens/`` probes read
        # the footprint channel (:meth:`_input_readings`).
        probes = {n: s for n, s in all_probes.items() if _spec_side(s) == "output"}
        if not probes:
            return {}
        readout_layers: set[int] = set()
        for spec in probes.values():
            readout_layers.update(spec["layers"])
        # Restrict precomputed rows (e.g. a custom live-lens layer set) to the
        # probes' fitted layer set.
        keep = [i for i, l in enumerate(layers) if l in readout_layers]
        if not keep:
            return {}
        if len(keep) != len(layers):
            if logits is not None:
                logits = logits[keep]
            if probabilities is not None:
                probabilities = probabilities[keep]
        return self._readings_from_rows(
            probes,
            [layers[i] for i in keep],
            logits=logits,
            probabilities=probabilities,
            only=only,
        )

    def _readings_from_rows(
        self,
        probes: "Mapping[str, Mapping[str, Any]]",
        layers: "Sequence[int]",
        *,
        logits: torch.Tensor | None = None,
        probabilities: torch.Tensor | None = None,
        only: "set[str] | None" = None,
    ) -> dict[str, ScalarReading]:
        """Synthesize the readout-channel readings for ``layers``' rows.

        Returns the family-native :class:`ScalarReading`: ONE channel, the
        mean fitted-layer probability ``mean_l p_l(v)``, its explicit
        ``unit``, the per-layer ``p_l`` trace, and a depth summary that
        names its own mass basis.  Not a ``ProbeReading`` with eight
        neutralized geometry fields — a readout probe has no fraction,
        nearest, residual, assignment or membership to report, and shipping
        those as constants is measurement-shaped noise.
        """
        runtime = self.runtime
        names = [
            name for name in probes
            if only is None or name in only
        ]
        if not names:
            return {}
        token_ids = [probes[n]["token_id"] for n in names]
        rows = probabilities if probabilities is not None else logits
        row_device = cast(torch.Tensor, rows).device
        token_ids_tensor = runtime.long_tensor(token_ids, row_device)
        depth_tensor = runtime.depth_tensor(layers, row_device)
        depths = runtime.depths(layers)
        if probabilities is not None:
            stats = token_readout_stats_from_probabilities(
                probabilities, depths, token_ids,
                token_ids_tensor=token_ids_tensor,
                depth_tensor=depth_tensor,
            )
        else:
            stats = token_readout_stats(
                cast(torch.Tensor, logits), depths, token_ids,
                token_ids_tensor=token_ids_tensor,
                depth_tensor=depth_tensor,
            )
        out: dict[str, ScalarReading] = {}
        for name, (strength, com, spread, per_layer) in zip(names, stats):
            out[name] = ScalarReading(
                value=float(strength),
                unit=UNIT_MEAN_TOKEN_PROBABILITY,
                per_layer={
                    int(l): float(p_l) for l, p_l in zip(layers, per_layer)
                },
                depth=DepthSummary(
                    center=(float(com),),
                    spread=(float(spread),),
                    basis="readout_probability_mass",
                ),
            )
        return out

    def gate_scalars(
        self, gate_keys: "set[str] | None" = None, *, step_id: int = -1,
    ) -> dict[str, float]:
        """Per-forward gate scalars from the latest capture slices.

        Called from the gating score callback (once per decode forward,
        before the token tap). Computes the referenced lens logits, stashes
        them for the display step to reuse (``step_stash``, keyed by
        ``step_id`` — the display reuses rows iff ``stash["step"]`` matches
        its own forward index, so staleness is structural and reuse is
        idempotent), and flattens
        the synthesized readings through :meth:`Monitor.flat_scalars` so the
        gate key space is uniform. Gate-only calls score exact
        selected-token softmax columns; live display calls still calibrate
        the full matrix once for downstream card/aggregate reuse.  Empty
        when nothing is capturable yet.

        ``slens/`` gates read first, from the bound specs; the R family's
        state is resolved only when an R-family (``jlens/``) probe is
        scored, so a generation gated only on ``slens/`` probes never
        touches (or refreshes) the R lens per token.
        """
        specs = self._measurement_specs()
        only = None
        if gate_keys is not None:
            # ``None`` is the full-roster sentinel; an explicit empty set
            # means "no gated probes" and scores nothing — the
            # None-vs-empty distinction every family's gate entry keeps.
            only = {
                key.split("[", 1)[0]
                for key in gate_keys
                if key.split("[", 1)[0] in specs
            }
            if not only:
                return {}
        live_display_needs_full_probs = bool(
            self._measurement_live() is not None
            and self.active_for_generation
        )
        input_names = {n for n, s in specs.items() if _spec_side(s) == "input"}
        input_only = input_names if only is None else only & input_names
        input_scalars: dict[str, float] = {}
        if input_only:
            # The live display step reads every ``slens/`` probe on this
            # forward: read them all here once and let it reuse the stash.
            readings = self._input_step_readings(
                specs, None if live_display_needs_full_probs else input_only,
                step_id=step_id,
            )
            input_scalars = scalar_gate_keys({
                name: reading.value
                for name, reading in readings.items() if name in input_only
            })
        if not ((set(specs) if only is None else only) - input_names):
            return input_scalars
        lens, all_probes = self._measurement_state()
        probes = {n: s for n, s in all_probes.items() if _spec_side(s) == "output"}
        if only is not None:
            only = {name for name in only if name in probes}
            if not only:
                return input_scalars
        if not probes or lens is None:
            return input_scalars
        latest = self._capture.latest_per_layer()
        if not latest:
            return input_scalars
        # When the token tap will immediately need every pinned probe reading
        # for the live payload, compute that superset once in the gate
        # callback and let the display reuse it.  Gate-only calls stay on the
        # narrower requested subset.
        probe_read_only = None if live_display_needs_full_probs else only
        readout_layers: set[int] = set()
        for name, spec in probes.items():
            if probe_read_only is None or name in probe_read_only:
                readout_layers.update(spec["layers"])
        layers = sorted(l for l in readout_layers if l in latest)
        if not layers:
            return input_scalars

        logits = self.runtime.logits_rows(
            lens, [(l, latest[l]) for l in layers],
        )
        probabilities = None
        if live_display_needs_full_probs:
            probabilities = readout_probabilities(logits)
            live_stash: dict[str, Any] = {
                "layers": tuple(layers),
                "logits": logits,
                "probabilities": probabilities,
                "step": step_id,
            }
            self.step_stash = live_stash
        else:
            self.step_stash = {
                "layers": tuple(layers),
                "logits": logits,
                "step": step_id,
            }
        if probabilities is not None:
            readings = self.score_probes_from_rows(
                layers=layers,
                probabilities=probabilities,
                only=probe_read_only,
            )
            live_stash = cast("dict[str, Any]", self.step_stash)
            live_stash["readings"] = readings
            live_stash["readings_layers"] = tuple(layers)
            live_stash["readings_step"] = step_id
            self.last_step_readings = readings
        else:
            readings = self.score_probes_from_rows(
                layers=layers, logits=logits, only=probe_read_only,
            )
        return {**input_scalars, **scalar_gate_keys({
            name: reading.value
            for name, reading in readings.items()
            if only is None or name in only
        })}

    def score_aggregate(
        self, generated_ids: list[int], pooled: dict[int, torch.Tensor],
    ) -> dict[str, ScalarReading]:
        """End-of-gen aggregate at ``pooled``, the read plane's slice at the
        last content token (shared with the monitor roster and the SAE
        family, so all three aggregates read the same position)."""
        # Binding-authoritative guard: a probe detached mid-generation
        # stays in this generation's aggregate roster (mutations apply
        # next generation).
        if not self._measurement_specs() or not generated_ids or not pooled:
            return {}
        return self.current_run.observe_aggregate(pooled)

    # ----------------------------------------------------------- live readout

    def enable_live(
        self,
        *,
        layers: "Sequence[int] | None" = None,
    ) -> list[int]:
        """Stream the J-lens readout live during generation.

        The selected layers' ``J_l`` move device-resident here, once;
        ``layers`` defaults to every fitted lens layer.  Attaches no
        forward hooks (the reader consumes the capture's existing
        latest-slice buffers post-forward), so steering fast-path /
        compile eligibility is untouched.  Returns the resolved layer list.
        """
        runtime = self.runtime

        # One lens-state transaction: the lens read and the live-state
        # rebuild must be consistent (adoption calls this while holding
        # the same reentrant lock).
        with self.state_lock:
            lens = self._require()
            uses_all_layers = layers is None
            if layers is None:
                layers = sorted(int(layer) for layer in lens.source_layers)
            else:
                layers = sorted(set(int(l) for l in layers))
                missing = [l for l in layers if l not in lens.jacobians]
                if missing:
                    raise RequestError(
                        f"layers {missing} not in the fitted lens "
                        f"(fitted: {lens.source_layers[0]}.."
                        f"{lens.source_layers[-1]})"
                    )
            device = runtime.device
            layer_list = list(layers)
            if layer_list:
                j_stack = runtime.transport_stack(lens, layer_list, device)
            else:
                sample = next(iter(lens.jacobians.values()))
                j_stack = torch.empty(
                    (0, *sample.shape), device=device, dtype=torch.float32,
                )
            self.live = {
                "layers": layer_list,
                "uses_all_layers": uses_all_layers,
                "J_stack": j_stack,
                "layer_rows": {l: i for i, l in enumerate(layer_list)},
                "readout": runtime.readout_modules()[1],
            }
            return list(layers)

    def disable_live(self) -> None:
        """Stop streaming the live readout and free the device J_l copies."""
        with self.state_lock:
            self.live = None

    @property
    def live_layers(self) -> list[int] | None:
        """The live readout's layer list, or ``None`` when it's off."""
        with self.state_lock:
            if self.live is None:
                return None
            return list(self.live["layers"])

    @property
    def active_source(self) -> str | None:
        """The active R-family lens source in the public source syntax.

        THE resolver — one ``active.json`` read, the same label a lens is
        stamped with when it becomes resident (:meth:`_active_label`), which
        every lens measurement binding carries.  A listing that answered from
        the prepared-sources scan instead would report ``null`` for an active
        pointer whose artifact is gone while persisted rows still carry its
        label.
        """
        return self._active_label("output")

    def _active_label(self, side: str) -> str | None:
        """The active source of ``side``'s slot (``output``: R family,
        ``input``: S family) in the public source syntax."""
        from drowse.lens.sources import LENS_SOURCES

        active = LENS_SOURCES.active(self.model_id, side)
        return None if active is None else LENS_SOURCES.format(active)

    @property
    def live_state(self) -> LensLiveState:
        layers = self.live_layers
        return LensLiveState(
            enabled=layers is not None,
            layers=tuple(layers) if layers is not None else None,
        )

    def set_live(self, enabled: bool, **kwargs: Any) -> LensLiveState:
        """Toggle the live workspace readout.

        ``layers`` (the family extra) selects the live layer set; omitted
        means every fitted layer.  Disabling with an explicit layer list is
        a caller error, not a silent no-op.
        """
        layers = kwargs.pop("layers", None)
        if kwargs:
            raise TypeError(
                f"lens live takes only 'layers', got {sorted(kwargs)}"
            )
        if not enabled:
            if layers is not None:
                raise TypeError("lens live disable takes no 'layers'")
            self.disable_live()
            return LensLiveState(enabled=False, layers=None)
        resolved = self.enable_live(layers=layers)
        return LensLiveState(enabled=True, layers=tuple(resolved))

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
        """The lens readout at the forward that produced a loom token, as the
        finished ``scope="replay"`` measurement envelope.

        The replay rebuilds the producing forward (under the node's recorded
        steering when ``apply_steering``; ``raw`` selects the flat render) and
        reads ``softmax(W_U · norm(J_l h))`` at the producing position for
        each selected layer (default: every fitted layer), plus the
        layer-aggregated chips from the same logits.  The read yields
        per-layer *probabilities*, which is exactly what
        ``build_measurements`` takes, so there is no exp/log round-trip on
        this hop.  Raises :class:`~drowse.lens.jlens.LensNotFittedError` with
        no lens, ``RequestError`` on a bad ``top_k`` or an unfitted layer, and
        ``UnknownNodeError`` / ``InvalidNodeOperationError`` on a bad anchor.
        """
        from drowse.readout.measurements import build_measurements

        width = 8 if top_k is None else int(top_k)
        if not 1 <= width <= 256:
            raise RequestError("top_k must be in [1, 256]")
        runtime = self.runtime
        lens = self._require()
        req = runtime.resolve_layers(lens, layers if layers is not None else "all")

        def read(
            hidden: dict[int, torch.Tensor],
        ) -> tuple[
            dict[int, list[tuple[str, float, int]]],
            list[tuple[str, float, float, float]],
        ]:
            pair_rows = [(layer, hidden[layer]) for layer in req]
            logits = runtime.logits_rows(lens, pair_rows)
            decoded = runtime.topk_rows(lens, pair_rows, top_k=width, logits=logits)
            return (
                dict(zip(req, decoded)),
                runtime.aggregate_rows(logits, req, top_k=width),
            )

        token, (readout, aggregate) = self._replay(
            node_id, raw_index, layers=req, read=read,
            apply_steering=apply_steering, raw=raw,
            label="session.lens.token_readout",
        )
        measurements = build_measurements(
            scope="replay",
            provenance="replayed",
            lens_readout={
                int(layer): [(str(tok), float(p_l)) for tok, p_l, _tid in rows]
                for layer, rows in readout.items()
            },
            lens_token_ids={
                int(layer): [int(tid) for _tok, _p_l, tid in rows]
                for layer, rows in readout.items()
            },
            lens_aggregate=[
                (str(tok), float(strength), float(com), float(spread))
                for tok, strength, com, spread in aggregate
            ],
            lens_sources={"R": runtime.source},
            steering=token.steering,
        )
        return {"measurements": measurements}

    def live_readout_step(
        self, *, top_k: int = 8, step_id: int = -1,
    ) -> (
        tuple[
            dict[int, list[tuple[str, float]]],
            list[tuple[str, float, float, float]],
            dict[int, list[int]],
        ]
        | None
    ):
        """One decode step's lens readout from the capture's latest slices.

        Runs post-forward at the token tap (never inside a hook).  Returns
        ``(per_layer, aggregate, token_ids)`` — top-k tokens per selected
        layer scored by per-layer softmax probability (the one strength
        unit every lens surface reports), the layer-aggregated chip list,
        and the vocabulary ids already selected by ``topk``.  Reuses the
        gate callback's stash rows when they came from THIS forward
        (``stash["step"] == step_id`` — staleness is structural, reuse is
        idempotent, and ``step_id < 0`` never matches) and the layer sets
        overlap.
        """
        runtime = self.runtime
        state = self._measurement_live()
        if state is None or not self.active_for_generation:
            return None
        buckets = self._capture.per_layer_buckets()
        layers_present: list[int] = []
        hidden_rows: list[torch.Tensor] = []
        stack_rows: list[int] = []
        layer_rows: dict[int, int] = state["layer_rows"]
        for layer in state["layers"]:
            bucket = buckets.get(layer)
            if not bucket:
                continue
            layers_present.append(layer)
            # Keep the raw bucket reference until after stash reuse is
            # resolved: an exact gate+live cache hit never needs these hidden
            # rows, so it should not pay a dtype/device conversion just to
            # discard them.
            hidden_rows.append(bucket[-1])
            stack_rows.append(layer_rows[layer])
        if not layers_present:
            return None
        stash = self.step_stash
        logits: torch.Tensor | None = None
        probabilities: torch.Tensor | None = None
        cached_logits: dict[int, torch.Tensor] = {}
        cached_probs: dict[int, torch.Tensor] = {}
        if (
            stash is not None
            and step_id >= 0
            and stash.get("step") == step_id
        ):
            stash_layers = tuple(int(layer) for layer in (stash.get("layers") or ()))
            if stash_layers == tuple(layers_present):
                # The common gate+live path: exact row-set match.  Keep the
                # existing zero-copy reuse of the full matrix rather than
                # restacking full-vocab rows.
                logits = stash["logits"]
                probabilities = stash.get("probabilities")
            else:
                # The gate callback may already have computed this forward's
                # band logits before the token tap.  Reuse any overlapping
                # rows rather than requiring the live display layer set to
                # match exactly.  Softmax calibration is per-layer, so cached
                # probability rows compose exactly with newly computed rows.
                for row, layer in enumerate(stash_layers):
                    if layer in layers_present:
                        cached_logits[int(layer)] = stash["logits"][row]
                        probs = stash.get("probabilities")
                        if probs is not None:
                            cached_probs[int(layer)] = probs[row]
        computed_logits: dict[int, torch.Tensor] = {}
        if logits is None:
            missing = [
                (layer, hidden, transport_row)
                for layer, hidden, transport_row in zip(
                    layers_present, hidden_rows, stack_rows, strict=True,
                )
                if layer not in cached_logits
            ]
            if missing:
                J_stack: torch.Tensor = state["J_stack"]
                J = select_rows(
                    J_stack,
                    [row for _layer, _hidden, row in missing],
                )
                H = torch.stack(
                    [
                        hidden.to(torch.float32)
                        for _layer, hidden, _row in missing
                    ],
                    dim=0,
                )
                computed = state["readout"](transport_rows(J, H))
                if not cached_logits and len(missing) == len(layers_present):
                    logits = computed
                else:
                    computed_logits = {
                        layer: computed[row]
                        for row, (layer, _hidden, _transport) in enumerate(missing)
                    }
            if logits is None:
                logits = torch.stack(
                    [
                        cached_logits[layer]
                        if layer in cached_logits else computed_logits[layer]
                        for layer in layers_present
                    ],
                    dim=0,
                )
        if probabilities is None:
            if not cached_probs and not computed_logits:
                probabilities = readout_probabilities(logits)
            else:
                probability_rows: dict[int, torch.Tensor] = dict(cached_probs)
                uncached_prob_layers = [
                    layer for layer in layers_present
                    if layer not in probability_rows
                ]
                if uncached_prob_layers:
                    uncached_logits = torch.stack(
                        [
                            computed_logits[layer]
                            if layer in computed_logits else cached_logits[layer]
                            for layer in uncached_prob_layers
                        ],
                        dim=0,
                    )
                    uncached_probs = readout_probabilities(uncached_logits)
                    probability_rows.update({
                        layer: uncached_probs[row]
                        for row, layer in enumerate(uncached_prob_layers)
                    })
                probabilities = torch.stack(
                    [probability_rows[layer] for layer in layers_present], dim=0,
                )
        # Pinned lens probes ride the same calibrated matrix — per-step
        # readout-channel readings for the payload merge.  A gate callback
        # may already have computed the same readings from these exact rows;
        # reuse them to avoid a second selected-token host sync on the
        # pinned+gated+live path.
        specs = self._measurement_specs()
        input_have: dict[str, ScalarReading] = {}
        input_missing: set[str] = set()
        input_block: _InputBlock | None = None
        if specs:
            readings_reused = False
            if (
                stash is not None
                and step_id >= 0
                and stash.get("readings_step") == step_id
            ):
                reading_layers = tuple(
                    int(layer) for layer in (stash.get("readings_layers") or ())
                )
                if reading_layers == tuple(layers_present):
                    self.last_step_readings = cast(
                        "dict[str, ScalarReading]",
                        stash.get("readings") or {},
                    )
                    readings_reused = True
            if not readings_reused:
                self.last_step_readings = self.score_probes_from_rows(
                    layers=list(layers_present), probabilities=probabilities,
                )
            # ``slens/`` probes ride the same step: the gate callback's
            # footprint readings when it ran on this forward, and the rest
            # read here and folded into the display's one host transfer.
            input_have, input_missing = self._input_stash_split(specs, None, step_id)
            if input_missing:
                latest = self._capture.latest_per_layer()
                if latest:
                    input_block = self._input_block(specs, latest, only=input_missing)
        else:
            self.last_step_readings = None
        # Display scores are per-layer softmax probabilities — the one
        # strength unit every lens surface reports (softmax is monotone, so
        # the top-k selection is unchanged from the raw-logit ranking).
        k = min(max(int(top_k), 0), int(probabilities.shape[-1]))
        vals, idxs = probabilities.topk(k, dim=-1)
        depth_tensor = runtime.depth_tensor(layers_present, probabilities.device)
        agg_idxs, agg_stats = aggregate_readout_tensors_from_probabilities(
            probabilities,
            runtime.depths(layers_present),
            top_k=k,
            depth_tensor=depth_tensor,
        )
        # Pack the tiny K-wide result (and any ``slens/`` block) through the
        # shared MPS-safe helper and synchronize once.
        n_layers = len(layers_present)
        readout_rows = (vals, idxs, agg_stats, agg_idxs.reshape(1, -1))
        if input_block is None:
            host = pack_readout_rows_to_host(*readout_rows)
        else:
            flat = pack_readout_rows_to_host(
                *(row.reshape(-1) for row in readout_rows),
                input_block.values.to(probabilities.device).reshape(-1),
            )
            split = (2 * n_layers + 4) * k
            host = flat[:split].reshape(2 * n_layers + 4, k)
            input_have.update(input_block.readings(
                flat[split:].reshape(input_block.values.shape),
            ))
        if input_missing:
            self._input_stash_put(step_id, input_missing, input_have)
        if input_have:
            self.last_step_readings = {**(self.last_step_readings or {}), **input_have}
        host_rows = host.tolist()
        all_vals = host_rows[:n_layers]
        all_idxs = host_rows[n_layers:2 * n_layers]
        agg_host = host_rows[2 * n_layers:]
        out: dict[int, list[tuple[str, float]]] = {}
        token_ids: dict[int, list[int]] = {}
        for row, layer in enumerate(layers_present):
            pairs: list[tuple[str, float]] = []
            for v, i in zip(all_vals[row], all_idxs[row]):
                pairs.append((runtime.decode_id(int(i)), float(v)))
            out[layer] = pairs
            token_ids[layer] = [int(i) for i in all_idxs[row]]
        agg = [
            (
                runtime.decode_id(int(agg_host[3][j])),
                float(agg_host[0][j]),
                float(agg_host[1][j]),
                float(agg_host[2][j]),
            )
            for j in range(k)
        ]
        return out, agg, token_ids

    # ------------------------------------------------------- authored prefill

    def authored_capture(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        top_k: int,
    ) -> tuple[
        dict[int, list[tuple[str, float]]],
        list[tuple[str, float, float, float]],
        dict[int, list[int]],
        dict[str, ScalarReading],
    ] | None:
        """Live J-LENS payload for one retained authored producer row.

        Computation only — the token matching, ``j-1`` producer semantics,
        row ordering, and loom persistence stay in the session's
        authored-prefill orchestration.
        """
        runtime = self.runtime
        state = self._measurement_live()
        if state is None or not self.active_for_generation:
            return None
        layers = [int(layer) for layer in state["layers"] if int(layer) in hidden]
        if not layers:
            return None
        lens, specs = self._measurement_state()
        if lens is None:
            return None
        logits = runtime.logits_rows(
            lens, [(layer, hidden[layer]) for layer in layers],
        )
        probabilities = readout_probabilities(logits)
        readings = self.score_probes_from_rows(
            layers=layers, probabilities=probabilities,
        ) if specs else {}
        readings.update(self._input_readings(specs, hidden))
        k = min(max(int(top_k), 0), int(probabilities.shape[-1]))
        values, indices = probabilities.topk(k, dim=-1)
        value_rows = values.detach().to("cpu").tolist()
        id_rows = indices.detach().to("cpu").tolist()
        per_layer: dict[int, list[tuple[str, float]]] = {}
        token_ids: dict[int, list[int]] = {}
        for row, layer in enumerate(layers):
            per_layer[layer] = [
                (runtime.decode_id(int(tid)), float(value))
                for value, tid in zip(value_rows[row], id_rows[row], strict=True)
            ]
            token_ids[layer] = [int(tid) for tid in id_rows[row]]
        aggregate = runtime.aggregate_rows(
            None, layers, top_k=k, probabilities=probabilities,
        )
        return per_layer, aggregate, token_ids, readings


__all__ = ["LensInstrument"]
