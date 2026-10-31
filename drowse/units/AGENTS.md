# drowse/units

MLP neurons and attention heads as objects drowse reads, gates on and steers:
the fourth read family (`units`), its neutral statistics, its probe tap, and
the runtime that resolves a unit reference against the loaded model. The unit
directions themselves (`neuron_write`, `neuron_read`, `head_write_matrix`,
`head_read_rows`) are pure functions of a `BlockSpec` in `model/units.py`; the
block description is `model/blocks.py`. Unit ablation's hook (`UnitOp`) lives
with the other steering hooks in `steering/hooks.py`.

| Module | Owns |
|---|---|
| `stats.py` | `UnitStats` / `LayerUnitStats`, `UnitStatsAccumulator` (the neutral-pass hook pair), `top_median`, the position and top-document constants, the stored tensor names |
| `runtime.py` | `UnitsRuntime` (block description once, `check`, `site`, `write_profile`, `stats`), `UnitSite`, `UnitsUnavailableError` (422), `unit_ref` |
| `capture.py` | `UnitTap` (one forward hook per touched module, a device ring, post-forward values), `TapUnit`, `SCALE_FLOOR` |
| `instrument.py` | `UnitsInstrument` (`session.units`), `UnitsRun` |

Layer 7, beside `lens` and `sae` (`tests/invariants/test_layering.py`): it
imports `model`, `store`, `readout` and below, never `steering.composer`,
`generation` or the session. The composer (layer 8) reaches it through its
`UnitSource` protocol; the session wires everything.

## The units

A unit reference is `naming.NeuronRef` / `HeadRef` (`neuron/L8n344`,
`head/L20h17`; `parse_unit_ref`, canonical uppercase `L`). Definitions follow
neuron-lens (`design.md` "The units"):

- neuron `n` of block `L` is coordinate `n` of the down projection's input,
  `act = act(gate(x̂))·up(x̂)`; it takes both signs, so it has two poles;
- head `h` of block `L` is its `head_dim` slice `z_h` of the output
  projection's input; its write is `W_O^h z_h` (with Gemma's post-attention
  norm: its gain, and its per-position scale held constant, the LN rule).

`UnitsRuntime.check` answers an unknown unit with `model.units.UnitIndexError`
(400) naming the valid range (`neuron 64 is outside block 1 (64 neurons,
0..63)`, `layer 9 is outside the model (4 blocks, 0..3)`). A model whose blocks
`model_spec` refuses raises `UnsupportedBlockError` (422) on every unit call;
the description is discovered once (inside the exclusive section) and kept, so
later checks, including scope resolution, touch no model.

## Neutral statistics (`stats.py`)

Measured on the neutral pass (`fit/neutral.py`), stored in the neutral cache
v5 shards beside each layer's activations (`store/neutral.py`), loaded by
`session/neutral.py::NeutralBaseline` with the means and whitener.

| Tensor | Shape | Definition |
|---|---|---|
| `neuron_mean` | `[d_ff]` | mean `act` over every real position `>= 1` |
| `neuron_scale` | `[2, d_ff]` | neuron-lens `act_top` per pole: per document the largest `+act` (row 0) / `−act` (row 1) over positions `>= 1`; the median of the `UNIT_TOP_DOCS = 24` largest document maxima; clamped at 0 |
| `head_mean` | `[n_heads, head_dim]` | mean `z_h` over positions `>= 1` |
| `head_scale` | `[n_heads]` | `act_top` of the write norm `‖W_O^h z_h‖` (sandwich scale included) over positions `>= HEAD_SCALE_FROM = 16` |

Position 0 is the attention sink (massive activations) and never counts; a
head's scale starts where neuron-lens's does, because a head with few keys
attends sharply. With fewer documents than 24 the median runs over the
documents seen (`top_median`). Everything streams: per chunk the accumulator
keeps sums and per-document maxima on the device in fp32 (bf16 sums accumulate
in fp32; maxima are exact in the activation dtype), merges them into running
top-k buffers on `commit_chunk`, and drops them on `discard_chunk` (the
capture's out-of-memory retry re-runs those rows). One down-projection
pre-hook and one output-projection forward hook per block, on the same forward
the neutral activations come from.

Units never fail the neutral pass: a model `model_spec` refuses, or a hook
that raises anything but an out-of-memory error, publishes a complete cache
whose `units` block reads `{"available": false, "reason": …}`. Reading units
then raises `UnitsUnavailableError` (422) with that reason.

## The read family (`instrument.py`, `capture.py`)

A unit probe is one `ScalarReading` in the `activation_over_scale` unit at one
layer (`per_layer = {L: value}`, depth basis `single_layer`, center
`L / (n_layers − 1)`), with the raw firing in `meta` (`activation` for a
neuron, `write_norm` for a head) and `meta.unit` the canonical reference:

- neuron: `act / scale[pole]`, the pole the sign of `act` picks, so `1.0` is
  the pole's typical high firing on either side (`< 0` on the − pole);
- head: `‖W_O^h z_h‖ / scale`, never negative.

Scales are floored at `SCALE_FLOOR = 1e-6`: a pole silent on the neutral corpus
reads large rather than infinite. One channel (`<name>`, `<name>[0]`);
`validate_gate` refuses the geometry channels like the lens and SAE do (the
grammar already refuses a suffix after a unit probe).

**The tap.** A bound run owns a `UnitTap` only while something reads it: the
attached probes when a per-token consumer (`ReadRequest.live`: the loom row or
a live-scores client, behind the family's live switch) or the final aggregate
reads them, plus every unit a gate names. A gate needs no attached probe:
`@when:neuron/L8n344 > 2` reads that unit for the gate alone (it appears in no
reading), and an unknown unit there is a 400 at generation preflight
(`ReadPlane.resolve_gates`). The tap installs one *forward* hook per touched
module (after every pre-hook, so a probe reads what the model used, ablations
included), each an `index_select` of its units' coordinates at the last
position into a preallocated ring row (`out=`), plus `vector_norm(…, out=)` of
the output row for a sandwich head. No allocation, no host sync; the module
that runs last in a forward advances the count. The ring is `AGG_TAIL_DEPTH`
forwards deep. The hooks come off with the residual capture when decode ends
(`ReadPlane.end` → `end_capture`), so nothing after the last decode forward
moves the ring. `values_at(fwd)` computes the values after the forward, as one
fp32 vector; the run moves it to the host once per step (stashed by `step_id`,
so the gate callback and the token tap share it) and once for the aggregate.

**Reads.** `UnitsRun.gate_scalars` (the gate callback), `step_readings` (the
token tap), `aggregate_at(fwd)` (finalize: `ReadPlane` passes the aggregate
forward index; the residual pooled slice carries no unit activation, so the
protocol's `observe_aggregate(pooled)` reads the latest forward instead). The
units plan declares no residual layers: the family is invisible to
`plan_capture`. `UnitsInstrument.hooks_attached` (via
`ReadPlane.transient_hooks`) routes a compiled session to its eager module.

**Other surfaces.** `attach(selector, as_name=)` (the roster routes `neuron/`
and `head/`), `try_detach`, `specs` (`{unit, kind, layer, index}`), `names`,
`probe_hash` (the unit plus its neutral scale), `live_state`/`set_live`
(all-or-nothing, no extras), `active_source` (`None`: no source lifecycle),
`token_readout` (replays the producing forward with a depth-1 tap attached
inside the replay's exclusive section; `top_k` and `layers` are 400).
`generate_batch` runs serially when unit probes are attached and readings are
wanted (the tap reads one row). Over HTTP the native probe routes attach, list (a
`UnitsProbeInfo` row: `family: "units"`, `unit`, `kind`, `layer`, `index`) and
detach units, and the OpenAI/Ollama `x-drowse-probe-readings` extension
carries their aggregates (`server/runtime.py::attached_probe_names` unions
every family's roster).

## Steering

- **Push.** `α neuron/L8n344` resolves through the registry's `neuron` atom
  resolver to `register_direction`: the `+`-pole write `neuron_write(block, n)`
  at layer `L`, registered as the profile `neuron/L8n344` and then folded,
  normalized and pushed exactly like a one-layer `jlens/` atom (the one
  kernel, the affine fast path). A negative α pushes along the `−` pole's
  write. There is no head push (a parse error).
- **Ablation.** `α !neuron/…` / `α !head/…` (`UnitAblationTerm`) compose via
  `UnitsRuntime.site` into `SteeringPlan.add_unit_ablation` and lower to one
  `UnitOp` per (module, trigger): `x ← x ⊙ keep + bias` on the projection's
  input, `keep = 1 − α`, `bias = α·mean` on the unit's coordinates (α = 1 is
  exact mean-ablation; any finite α blends, overshoots or amplifies). Details
  and routing: `drowse/steering/AGENTS.md`.

## Testing

`tests/units/`: `test_unit_stats.py` (the accumulator against a hand
computation on llama and Gemma 3, the cache, the unavailable path, v4
recompute), `test_unit_probes.py` (the tap by hand, probes through a real
tiny session, gates with and without an attached probe, replay, refusals,
the HTTP family block, the probe routes and the compat extension),
`test_unit_steering.py` (ablation against a hand-patched forward to 1e-5,
partial coefficients, heads, triggers and gates,
the push against the kernel on the write direction, 422s and 400s). Helpers:
`tests/support/units.py`. GPU: `tests/gpu/test_smoke.py::TestAblationPerformance::test_throughput_with_unit_ablation`
(8 neurons and 2 heads ablated, ≥ 80% of vanilla tok/s).
