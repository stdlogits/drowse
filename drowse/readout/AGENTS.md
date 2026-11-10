# drowse/readout

The read side's lower half: the shared read vocabulary, the instrument contract
all four read families implement, the geometry family (the whitened `Monitor`
and its `GeometryInstrument` adapter), the hidden-state capture buffers, the
pure capture planner, the probe roster, the versioned measurement envelope and
Mahalanobis cosine analytics. How reads fit a generation is the read-plane
section of `ARCHITECTURE.md`; this file is the contract.

Elsewhere: `generation/read_plane.py::ReadPlane` applies capture plans, arms the
step sinks, builds the gate callback and runs finalize reads;
`loom/replay.py::TokenReplayer` implements `TokenReplay`; `lens/instrument.py`,
`sae/instrument.py` and `units/instrument.py` are the other three families;
`geometry/whitener.py` is the metric; `session/session.py` wires it all.

## Module map

| Module | Owns |
|---|---|
| `types.py` | `InstrumentFamily`, gate channels + `GateRef`/`parse_gate_ref`, `ProbeReading`, `ScalarReading`, `DepthSummary`, `LiveState`, `ReadRequest`/`InstrumentPrep`/`LensPrep`/`InstrumentPlan`/`InstrumentBinding`, `AGG_TAIL_DEPTH`, `next_prep_token` |
| `protocol.py` | `Instrument`, `InstrumentRun`, `LatestSlices` (runtime-checkable Protocols) |
| `replay.py` | `TokenReplay` protocol, `ReplayedToken`, `ReplayLayers` |
| `planner.py` | `GateDemand`, `ReadDemand`, `resolve_read_demand`, `CapturePlan`, `plan_capture`, `CaptureMode`, `CaptureState`: pure, no side effects |
| `capture_buffer.py` | `HiddenCapture`, `install_persistent_capture_hooks` |
| `monitor.py` | `Monitor`: every geometry read and gate scalar |
| `monitor_attach.py` | attach-time algebra (`AttachedManifoldProbe`, `_LayerWhiten`, `_FactorStack`, `_CurvedStack`, bandwidths, `NEUTRAL_LABEL`) and the per-token `_curved_geometry` |
| `monitor_flat.py` | `_FlatEngine`: the batched read over every flat probe |
| `geometry.py` | `GeometryInstrument`, `GeometryRun` |
| `roster.py` | `ProbeRoster` and its `GeometryFamily`/`ReadoutFamily` protocols |
| `measurements.py` | `build_measurements`, the persisted projections (`token_row_measurements`, `summary_measurements`, `aggregate_measurements`), `envelope_sources`, the envelope TypedDicts |
| `analytics.py` | `cross_layer_cosine`, `cosine_matrix` (whitened; used by `server/profile_routes.py`) |

Layering: `readout` sits above `steering.{grammar,terms,triggers}` and below
`lens`/`sae`/`units`, so it cannot import them, `steering.composer`/`hooks`,
`generation` or anything above (`TYPE_CHECKING` imports count). Collaborators
from those layers are typed against the Protocols here (`LatestSlices`,
`TokenReplay`, `GeometryFamily`, `ReadoutFamily`) and arrive through
constructors. `monitor_attach.py` does not import `monitor.py`.

## The instrument contract (`protocol.py`)

`Instrument` carries only surface every family answers the same way: `family`,
`state_lock`, `current_run`, `prepare`/`plan`/`bind`/`close_run`, `names`,
`specs`, `try_detach`, `probe_hash`, `validate_gate`, `active_source`,
`live_state`/`set_live`, `token_readout`. Family-native surface (geometry's
`top_n`, the lens's `probe_layers`, the SAE's `probe_values`) and `attach`
(on the roster's `GeometryFamily`/`ReadoutFamily`) stay off it.

**Transaction.** Each capture runs `close_run → prepare → plan → bind` per
family (`ReadPlane.bind_runs`) and reaches `close_runs` in a `finally`.
`prepare` raises on a still-bound run (a stale lens pin would short-circuit the
refresh it exists for), and source refresh precedes planning because lens
adoption may rewrite live probe specs. Each prep draws a token from one
process-wide sequence (`next_prep_token`), echoed as the plan's `prep_token`;
`bind` refuses a plan/prep pair whose family or token disagree. `plan` derives
from the prep only, never the live registry.

`InstrumentRun` is `binding` (frozen source + specs), `bound`,
`observe_aggregate(pooled)` and `close`; an unbound idle passthrough run backs
out-of-generation reads. There is no per-run reading memo: within one forward
the decode loop passes the same `step_id` (`len(generated_ids)` before the
forward) to the capture sink, gate callback and token tap, and reuse is the
sink's row (geometry) or the family's step-keyed stash.

The units family reads no residual capture: its bound run owns a tap
(`units/capture.py::UnitTap`, forward hooks on the down and output projections,
present only while a probe or gate reads them), so its plan declares no layers
and `plan_capture` never sees it; its finalize read takes the aggregate's
forward index (`UnitsInstrument.score_aggregate`).

**Locks.** Each family's `state_lock` is a reentrant leaf guarding roster
mutation and coherent read surfaces (`names`, `specs`, `plan`/`bind` roster
reads, idle run reads). Callers take the exclusive-GPU section first, and the
bound per-token path never takes it. Lens/SAE detaches mid-generation are safe
(the bound run measures its frozen binding); geometry has no frozen roster, so
its attach/detach hold the exclusive section and a detach during a generation
fails fast with `ConcurrentExtractionError` (409).

**`token_readout`** returns the finished `{"measurements": …}` replay
envelope. `top_k=None` means the family default (8 for the ranked lens/SAE
reads). A knob a family cannot honor raises `RequestError` (400): geometry
rejects `top_k` and `layers`, SAE rejects `layers`.

**`validate_gate`** rejects channels a family cannot produce: lens, SAE and
units accept only the strength axis (`validate_gate_channels`), geometry
everything. It runs at preflight in `ReadPlane.resolve_gates` for gates naming
an attached lens/SAE/units probe and for every gate naming a unit (a unit gate
needs no attached probe: `UnitsInstrument.gate_unit` checks the unit against
the model, 400 if unknown).

`session.instruments` is the family enumeration;
`generation/token_payloads.py::_FAMILY_SLOTS` and
`server/instrument_routes.py::CAPABILITIES` are pinned to it by test, so a new
family is a registry entry plus its implementation (checklist: the change map
in `ARCHITECTURE.md`).

## Vocabulary (`types.py`)

Gate scalar keys; `parse_gate_ref` is the one key-shape parser and inverts what
`Monitor.flat_scalars` emits:

| Key | Channel | Producers |
|---|---|---|
| `<probe>` / `<probe>[i]` | `Axis(i)` (`[0]` normalizes to the bare form) | all |
| `<probe>:fraction` | `Fraction` | geometry |
| `<probe>:membership` | `Membership` | geometry |
| `<probe>@<label>` | `Distance` (negated, label-spacing units) | geometry |
| `<probe>~<label>` | `Assignment` (probability) | geometry |

`<probe>` is the attached name verbatim (`jlens/fake`, `default/emotions`,
`pirate:role-x`). Parsing is unambiguous because `NAME_REGEX` forbids `@`/`~`
in names and labels, and `:` splits only on the two exact channel words. The
runtime gate lookup is a plain `dict.get` on the verbatim string; `GateRef` is
for preflight and planning.

Geometry produces `ProbeReading`; lens, SAE and units produce `ScalarReading`
(`value` + explicit `unit`: `mean_token_probability`, `mean_footprint_z`
(`slens/`), `activation_over_max`, `raw_activation`, `activation_over_scale`;
`per_layer`; `DepthSummary`; optional `meta`). A unit reading is a neuron's
activation over its pole's reading scale (signed) or a head's write norm over
its own, at one layer (basis `single_layer`), with the raw firing in `meta` and
the scale's source in `meta.scale` (`"catalog"`: the neuron-lens catalog's
`act_top`; `"neutral"`: the neutral statistics), soft-floored
(`units/stats.py::UnitScales`). `DepthSummary.basis` travels with the numbers
because families weight depth by unrelated masses, so centers are not
comparable across bases. `as_probe_reading` is a compatibility projection for
`TokenEvent.probe_readings` and the OpenAI/Ollama `x-drowse-probe-readings`
key, not the native per-token path. `reading_axis0` is the number the flat
`scores` view and loom rows carry.

`InstrumentPlan` is demand, not mechanics (`latest_layers`, `tail_layers`,
`gate_keys`, `final_aggregate`); physical retention and cross-family upgrades
belong to the planner. `AGG_TAIL_DEPTH = 8` is the finalize ring depth, deep
enough to walk back past trailing special tokens to the last content token.

## Capture planning (`planner.py`)

`resolve_read_demand` folds a generation's consumers into one `ReadDemand` (the
engine masks every per-token geometry consumer when the geometry live toggle is
off):

- `per_token_full_consumer` = (live scores on and loom attached), live token
  scores, persisted per-layer scores, a persisted probe row or persisted
  inspector coords.
- `need_per_token` = a geometry gate is active or `per_token_full_consumer`.
  Lens/SAE gates score on their own paths and never force geometry scoring;
  `units_gating_probe_keys` go to the units tap alone.
- `gating_only_probes` = the gated geometry probes when gates are the only
  per-token consumer.
- `lean_per_token` = per-token need, no `return_hidden`, no geometry gate,
  geometry probes attached, no full-reading consumer: every per-token consumer
  reads axis 0. A lean row is coords + fraction only, so a consumer reading more
  must widen this rule first.

`plan_capture` captures every model layer for `return_hidden` (`widen`);
otherwise the union of geometry `latest_layers` and lens/SAE `latest_layers |
tail_layers` (an empty union attaches nothing). Branch order matters; first
match wins ("monitor" = geometry probes attached and not widened; "ring 8" =
`AGG_TAIL_DEPTH`):

| Mode | When | Retention | Step sink | Curved feet | Finalize (in `ReadPlane`) |
|---|---|---|---|---|---|
| `GATING_SUBSET` | monitor, per-token need, `gating_only_probes` | ring 8 if final aggregate, else 1 | `subset`: planned gate scalars only | warm | pool full roster once from ring |
| `LEAN_INCREMENTAL` | monitor, per-token need, `lean_per_token` | ring 8 | `lean`: `coords_only` rows | warm | coord stream from rows; full aggregate re-scored once from ring |
| `INCREMENTAL` | monitor, per-token need | 1; with a lens/SAE final aggregate or gate, ring 8 (1 if final off) limited to their tail layers when any | `full` rows | warm | aggregate = the row at the pooled position |
| `AGGREGATE_ONLY` | monitor, no per-token need | ring 8 | none | cold | pool once |
| `AGGREGATE_ONLY` | no monitor; lens/SAE declared layers | ring 8 if final, else 1 | none | cold | lens/SAE pool once |
| `FULL` | `return_hidden` | unarmed: clone every step | none | cold | `Monitor.score_per_token` over the stack |

Capture rides the persistent compile-clean buffers iff not widened, no
authored prompt rows (`capture_prompt`) and the caller passed buffers;
otherwise transient hooks. `CaptureState` is what the plan leaves for finalize,
the gate callback and the token tap; its `aggregate_only` is true for
`AGGREGATE_ONLY` and for `GATING_SUBSET` with a final aggregate.

## Capture buffers (`capture_buffer.py`)

Hooks do device work only: detach the last-position `(D,)` slice, then clone it
or `copy_` it into a preallocated buffer. Retention (`set_retention`) is one
policy: unarmed = full retention for `stacked()`; depth 1 = one in-place buffer
per layer; depth > 1 = a ring, restricted to `tail_layers` when set (other
layers stay length-1 and are omitted from `tail_slice_at`). The step sink runs
after the forward returns, so its host read never drains the device
mid-forward. Ring modes count forwards (the highest hooked layer advances the
count; `ingest_persistent` does it on the persistent path); `FULL` does not, so
pooling indexes its stack directly. `install_persistent_capture_hooks` covers
every layer and runs before `torch.compile`; `ingest_persistent` replays the
same retention and sink post-forward. Prompt rows (`set_prompt_positions`)
exist only on transient hooks. `attach_batch_tail` is the `generate_batch`
`[B, D]` ring; `is_transient()` is the compiled-clean routing gate.

## The Monitor

Every probe is a `Manifold`; every read (live token, aggregate, replay) is one
full `ProbeReading` from the same scorer (an aggregate is a live read at the
pooled token, differing only in a curved probe's cold foot start). The wired
whitener must cover every probed layer or attach raises `WhitenerError`.

Reading fields (cross-layer means are share-weighted by the fit's
`mahalanobis_share`, normalized to sum 1):

| Field | Flat (affine) probe | Curved probe |
|---|---|---|
| `coords` | affine map of `M_R⁻¹ B Σ⁻¹(h − mean)` to the domain frame | `invert_parameterization` foot |
| `fraction` | `‖P_M x‖_M / ‖x‖_M` in `[0, 1]` | same |
| `residual` | `0.0` | normalized off-surface distance |
| `membership` | `1.0` | `exp(−res²/2σ(z)²)` at the foot |
| `nearest` | top-`top_n` by raw whitened distance, reported ÷ `label_scale` | same |
| `assignment` | `softmax(−d²/2τ² − R·log τ)`, top-`top_n` | same |

- `M_R⁻¹ B Σ⁻¹(h − mean)` is the Mahalanobis-orthogonal projection's coordinate
  in the write rows `B`. A fitted frame is Mahalanobis-orthonormal, so
  `M_R ≈ I` and this is the frame's `read` (the LDA covector `Σ⁻¹δ` at K = 2);
  the session whitener keeps it exact if the neutral cache changed since the
  fit. A curved fit's reduced node values (`eval_at`, the overlay) read through
  `sub.read`.
- Coordinates are domain-frame: each layer's reduced coord is mapped to the
  domain before averaging (raw per-layer coords are in `‖δ_L‖_M` units).
  Rank-1 reads `1.0` at the positive node. Periodic, sphere, Klein and
  projective domains average in the embedding and map back; a near-zero mean
  falls back to the highest-share layer's coordinate.
- `label_scale` is the median of per-node bandwidths: each node's
  nearest-neighbor whitened distance for a flat (or σ-less) fit, its whitened
  tube thickness σ for a curved fit with a σ-field. Raw whitened distance spans
  about 60× across fits, so `@label` thresholds are portable only after this
  rescale. It is a per-probe constant, so ranking stays by raw distance; a
  single-node fit keeps 1.0.
- The `−R·log τ` log-volume bias is the Gaussian normalization; without it a
  diffuse-τ node swallows probability from far away. τ and the bias are
  precomputed at attach.
- `neutral` (`NEUTRAL_LABEL`) is a virtual candidate at the frame anchor in the
  whitened metric (zero for affine fits, the baked `origin` through `eval_at →
  read → chol` for curved). It rides `node_white_aug` as the last candidate,
  is never a stored node and is suppressed when the manifold has a real node
  named `neutral`.
- `depth_com`/`depth_spread` need `n_layers` at construction; mass is
  `share_L · |coord_L[axis]|` at depth `L/(n_layers−1)`.
- `subspace_coords_per_layer` (the inspector trail) is emitted only while
  `set_subspace_coords(True)` (the engine sets it per generation) and rides
  the live token frame only; a loom row never keeps it. Geometry token replay
  always carries it (the dashboard's Probes room places the token with it).

**Execution.** `_FactorStack` is the one device copy of the whitener's Woodbury
factors per device, grown rather than duplicated and assembled on the host
before one upload. A probe on a device manifold attaches on the host too and
uploads packed (`geometry/packing.py`), keeping the roster at a few device
buffers. A read is one stacked `Σ⁻¹h` apply over every layer; the flat roster
is then one `_FlatEngine` pass (segments bucketed by rank, dim and candidate
count; rank-1 is elementwise) and one host transfer. A curved probe's
per-layer factors and surfaces are stacked at attach (`_CurvedStack`,
`StackedRbf`), and a read runs every layer at once: elementwise work over the
stack, reductions (basis projections, norms) on each layer's own rows at its
own rank, one batched foot solve (`invert_stacked`) per rank, and the
share-weighted means as a left fold in layer order. A reduction batched across
layers moves values by an ulp and an off-surface foot to another basin, so
reductions stay per layer; a reading then does not depend on how many layers
ran together (`tests/readout/test_monitor_curved_stack.py`). A read seeing only
some of a probe's layers uses a cached `_CurvedStack.subset`. Curved feet
warm-start per `(probe, layers)` only while `enable_curved_warm(True)`
(sequential per-token scoring); aggregates, replays and `FULL` stacks read cold
so one-shot reads are reproducible. Curved probes remain the expensive reads: a
SmolLM2 `months` probe costs about 6 ms a token.

**Gates.** `plan_gate_scalars` parses keys once per generation;
`score_planned_gate_scalars` computes only the requested channels (curved
`:fraction`, `@label` and `~label` skip the foot solve). The planned path
scores any candidate label regardless of `top_n`, while `flat_scalars` emits
labels from the top-N lists only, and in `INCREMENTAL` mode the gate callback
fills missing keys from the planned path. A key naming an unattached probe or
an unknown label is dropped: the gate stays inactive, no error.

**Attach.** `add_probe` validates runtime geometry (a raw-space curved fit
needs its σ-field; SAE-space curved fits have none and read `membership` 1.0),
requires finite positive shares and builds every factor up front.
`set_whitener` re-attaches the whole roster atomically (failure keeps the old
whitener and roster; the same instance is a no-op), since bandwidths and
`label_scale` derive from the metric.

## GeometryInstrument

Constructor deps: `monitor`, `exclusive` (`base/exclusive.py::ExclusiveGuard`),
`resolve_manifold`, `ensure_whitener`, `replay`. `attach` enters the exclusive
section, builds the whitener, resolves the manifold, then adds under
`state_lock`. `refresh_swapped` follows the artifact registry's `on_swap`
(re-attach on the replacement with the same `top_n`, detach on eviction).
`plan` returns no layers for an empty or dormant roster (no gate, no per-token
consumer, no final aggregate) and narrows to the gated probes' layers when
gates are the sole consumer and final readings are off. The binding freezes
names only. `is_live` (default on) is the all-or-nothing per-token switch.
`probe_hash` is sha256 over fp32 bytes (the folded direction for a foldable
fit, else per-layer mean/basis/read/node coords), recorded at attach or swap
inside the exclusive section so the lock-free transcript export only looks it
up. Attach likewise snapshots the listing's node coords and the inspector's
geometry payload (`monitor_attach.py::probe_geometry_payload`) from the host
copy before upload, because the probe listing and inspector run without the
GPU lock and a copy off a resident MPS tensor beside another thread's model op
aborts the process. `token_readout` captures the roster's layer union and
scores with a cold foot.

## ProbeRoster

`jlens/…` and `slens/…` route to the lens, `sae/…` to the SAE, `neuron/…` and
`head/…` to the units, anything else to geometry. `detach` asks lens, SAE,
units, then geometry; an unknown name is a no-op (the server 404s first).
Digests are cached per family until `forget`; `changed` drops them and runs
`on_change` (the session invalidates analytics; the prefix cache survives).
Names are not unique across families: a collision shadows (hash and detach
consult lens first; the flat `scores` join keeps the last family written).

## Measurement envelope (`measurements.py`)

The one wire record for token frames (`scope="token"`), final aggregates
(`"aggregate"`) and replays (`"replay"`, `provenance="replayed"`);
`MEASUREMENTS_VERSION = 2`, and the loom reader refuses persisted rows of
another version. `None` means nothing was measured; a family channel exists
only when it has readings or a readout. Lens/SAE channels always carry
`binding`, geometry only when the caller passes one (replay). The SAE binding
is `{source, steering, layer}`; the lens binding is `{sources: {R?, S?},
steering}` (`LensBinding`), naming the source of each lens family that produced
data (`R` for `jlens/` readings or the readout, `S` for `slens/`, told apart by
reading unit via `LENS_FAMILY_BY_UNIT`). `units` is `{binding: {source: null,
steering}, readings}`, present only with readings. `readings` are each
family's native `to_dict()`; `readout` is the native discovery surface;
`scores` is the axis-0 join at 6 significant digits; a reading's trace lives in
the reading (no layer-major `per_layer_scores` copy). The lens readout holds
`p_l`, which becomes the wire's `logprob` here only (floored at `1e-45` so JSON
stays finite).

Three projections persist an envelope, with every float rounded to
`READING_DIGITS` (6) significant digits (`types.round_reading`, `%g`'s form,
`FLT_DIG`; significant digits rather than places because a `jlens/` probability
is often below 1e-4 and read on a log scale):

- `token_row_measurements` is a generated loom token row's slice: the flat
  `scores` (every family's axis 0) and each geometry reading's `fraction` and
  `coords` (`ROW_GEOMETRY_KEYS`), no other channel. Everything else is
  replayed by the token view and hover (geometry, lens and SAE `token_readout`,
  and the token anatomy). About 1.6 kB a SmolLM2 token with the 17 default
  probes, against about 118 kB for the whole envelope.
- `summary_measurements(envelope, per_layer=)` keeps every reading whole but its
  traces: geometry summaries (`GEOMETRY_SUMMARY_KEYS`), scalar readings, the
  lens readout's `aggregate`, SAE features, bindings and `scores`;
  `per_layer=True` adds `GEOMETRY_TRACE_KEYS` and the readout's `layers`. A row
  of a call that set `persist_per_layer_scores` keeps `per_layer=True`; every
  other row keeps `token_row_measurements` and replays the rest (an authored
  row through its node's `prompt_capture`). `subspace_coords_per_layer` is
  never kept.
- `aggregate_measurements` is what a loom node keeps of its reply's aggregate
  (`LoomNode.aggregate_measurements`, stamped at finalize): the summary without
  traces or `scores`. The rack and probe inspector draw the settled reading
  from it after a reload.

`envelope_sources` is the one builder of `lens_sources`/`sae_source`/
`sae_layer`, from the lens instrument's `measurement_sources()` (the labels its
run pinned, stamped on each `LensRuntime` when it became resident, so no disk
read per generation) and the SAE live config; an aggregate binds a family's
source only when it carries that family's readings.

A `replay` envelope may also carry `anatomy` (`TokenAnatomy`, built by
`loom/node_replay.py`; other scopes refuse it): one token's per-layer lens
reads (`reads_as`, `attention`, `mlp`, `steering`, write deltas read as
directions; each a `LensReadDict`), the units that pushed the token
(`AnatomyUnit`: contribution, raw `activation`, `activation_over_scale`, `None`
without unit statistics), the top heads' attention rows (`AnatomyPattern`), the
decoded `context` and a lens `binding`. Its `instruments` hold only the
attached lens probes' readings of the same residual, since the anatomy records
the replay forward, not a probe.

`tests/readout/test_measurements_envelope.py` pins key sets for every scope;
the TypedDicts feed `scripts/generate_webui_types.py` (gated by
`tests/invariants/test_webui_types_parity.py`), and drowse.ai reimplements the
envelope.

Tests: `tests/readout/` (CPU-only; helpers `tests/support/read_plane.py::ReadRig`,
`tests/support/geometry.py::stub_geometry_instrument`).
