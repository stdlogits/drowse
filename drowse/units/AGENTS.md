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
| `reads.py` | `UnitReader` (static reads, `top`, `find`, `stats`), the results (`NeuronRead`/`NeuronPoleRead`, `HeadRead`, `PlaceRead`/`UnitHalf`, `Firing`, `TopResult`/`TopHit`, `FindResult`), `ContrastCache`/`ContrastStats`, `own_token_z`, `copy_score`, `logit_moments`, `token_logits` |
| `catalog.py` | the neuron-lens catalog: `CATALOG_REPO`/`CATALOG_REVISION`/`CATALOG_MODELS`, `fetch_catalog`, `catalog_status`, `list_catalogs`, `remove_catalog`, `UnitCatalog` (rows, labels, `url`), `CatalogBinding` (the session's view), `unit_url`, `LABEL_CREDIT`, the errors |

Layer 7, beside `lens` and `sae` (`tests/invariants/test_layering.py`): it
imports `model`, `store`, `readout`, `lens` and below, never
`steering.composer`, `generation` or the session. The composer (layer 8)
reaches it through its `UnitSource` protocol; the session wires everything
(the reader gets the lens instrument, the catalog binding and the weights
fingerprint).

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

Units never fail the neutral pass: a model `model_spec` refuses, a hook that
raises anything but an out-of-memory error, or an out-of-memory error that
persists at capture batch size 1 (the capture then `abandon`s the accumulator
and re-runs the rows without its hooks) publishes a complete cache whose
`units` block reads `{"available": false, "reason": …}`. Reading units
then raises `UnitsUnavailableError` (422) with that reason.

## The read family (`instrument.py`, `capture.py`)

A unit probe is one `ScalarReading` in the `activation_over_scale` unit at one
layer (`per_layer = {L: value}`, depth basis `single_layer`, center
`L / (n_layers − 1)`), with the raw firing in `meta` (`activation` for a
neuron, `write_norm` for a head) and `meta.unit` the canonical reference:

- neuron: `act / scale[pole]`, the pole the sign of `act` picks, so `1.0` is
  the pole's typical high firing on either side (`< 0` on the − pole);
- head: `‖W_O^h z_h‖ / scale`, never negative. The write norm is Euclidean
  in the residual: one of the deliberate exceptions to Mahalanobis-only
  (`drowse/geometry/AGENTS.md`), since it is a scalar firing read against the
  head's own neutral scale, never a geometric comparison.

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
fp32 vector (heads in one batched `zᵀ G z` per `head_dim`, since a model may
mix sizes: Gemma 4's sliding and global layers); the run moves it to the host once per step (stashed by `step_id`,
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
  exact mean-ablation; any finite α blends, overshoots or amplifies). It
  skips position 0, the attention sink: the unit statistics count means from
  position 1, so substituting the mean there would put an out-of-distribution
  value at the sink. Every other position is ablated, a prefill's included.
  The forward's driver names the sink's column (`TriggerContext.sink_column`:
  0 at rest and on a prefill from the start, `None` on an incremental decode
  step); a left-padded batch puts it in a different column per row, so a
  batch with a unit ablation runs on the serial loop. Details and routing:
  `drowse/steering/AGENTS.md`.

## Static reads (`reads.py`)

What a unit is about, from its weights, by the four-lens rule
(`drowse/lens/read.py`): the site comes from where the vector sits, the kind
from the depth, so every place returns an S half and an R half with the depth
rule's weights (`session.lens.band`); the light half is still returned for
the UI to fade. Direction reads are uncentred (only residual reads centre on
the neutral mean). `session.units.read(unit, k=8, hide_hubs=False)`:

- **neuron**, per pole: `writes` reads `±w` (`neuron_write`, Gemma's
  post-feedforward gain and layer scalar included) at `out` of its layer,
  `fires_on` reads `â ± b̂` at `mid`, both through
  `LensInstrument.read`. The last block's `out` is the final residual, where
  the R read is exactly the logit lens (`J = I`); a lens that stops a layer
  short gets it there, labelled `logit`. Scalars as neuron-lens: `w_norm`,
  `cos_ab`, `null_share` (squared-norm share in the bottom 1% of
  right-singular directions of `W_U diag(γ_final)`, the basis built once per
  session), per pole the logit-lens skew (the `−` pole's is negated) and
  kurtosis over the token mask, the neutral mean act.
- **head**: `copies` / `suppresses` (at `mid`) are the top and bottom of each
  token's own-column z in the OV map `M[u, v] = A_u · B_v`, `B_v = W_V^h X_v`
  with `X_v = K^{out}_{L−1} ẽ_v` (the embedding at L0), `A_u` the R
  (`(γ_f ⊙ W_U[u]) J^{mid}_L W_O^h`) or S (`ẽ_u K^{mid⊤}_L W_O^h`) read of
  `u`; `copy_score` is `Σλ/Σ|λ|` of `Bᵀ A` per family. The map is never
  built (the mean and covariance of `A`'s rows). `inputs` (at `L − 1` out)
  ranks, per projection (query, key, value), tokens by the share of their unit
  direction the projection keeps, `‖W^h x̂_v‖` (S: the footprint; R: the
  direction that says `v`), as a z over the mask (`energy_z`). Without an S
  lens the OV read is absent (its input side reads through it).
- **hidden hubs** (`hide_hubs=True`, the UI's "Hide hub tokens"): each half
  ranks by neuron-lens's layer-contrastive `_z`, the token's score z-scored
  across the layer's population. Neurons: R over the lens logits of the `+`
  writes (the `−` pole's z is minus it) and of each pole's read directions
  separately; S over the raw scores of unit covectors (`û · ẽ_v`) of the `+`
  writes and of both poles' read directions together; standard deviations
  are population ones floored as neuron-lens floors them. Heads: each token's
  OV z across the layer's heads (unbiased std). Units read `layer_z`. A head's
  inputs never contrast: the heads of one KV group share their key and value
  rows, so a contrast across heads saturates (every token at one z) and a few
  KV heads are too small a population; they stay the vocabulary z.

`ContrastCache` keeps these per-layer statistics (per token mean and std,
and for R each row's `logsumexp`, which `top` needs for probabilities) under
`models/<id>/units/contrast/<family>-<lens identity>…/L<layer>.safetensors`,
a derived cache: a shard whose record (format, lens, place, token-mask
digest, weights fingerprint) differs is recomputed, never migrated. The
first hidden-hub read or `top` of a layer pays its population pass (the
whole layer's units through both lenses); later reads load the shard.

`top(token, by="writes"|"fires", layers=None, k=20)` scores every pole of
each layer by the token's layer-contrastive z through each present family,
`Σ w_f z_f / Σ w_f` with the depth weights (comparable across families and
layers), and lists the best `k` with each half's z, its raw read (R
probability, S footprint z) and the catalog label. Neurons only; a head's
"writes a token" is a copy of it and waits for the head population stats.
`find(selector | vector, layer=, k=16, layers=)` is `lens.decompose(...,
dictionary="neurons")` (`NeuronWrites`: each layer's writes, signed, in the
whitened metric) with each neuron's pole and label. `stats(unit)` is the
firing block alone.

**Firing.** `Firing.neutral` is drowse's own statistics (a pole's
`act_top`, the mean, the document count; a head's write `act_top` and mean
`‖z_h‖`). The neutral cache has no firing rate, so "fires on X% of tokens"
(`rate`: the catalog's share of tokens with `|act| > act_top / 4` over its
13M-token corpora) comes from a bound catalog, else is absent with the
reason.

## The catalog (`catalog.py`)

`stdlogits/neuron-lens` (the dataset neuron-lens publishes) catalogues every
pole and head of `Qwen/Qwen3-8B` and `google/gemma-4-12B` (the base model; an
instruction-tuned checkpoint has other neurons and gets none). `unit fetch`
binds it at `CATALOG_REVISION`, copying `neurons/labels/L<LL>.parquet` and
`heads/labels.parquet` whole and the `acts` and `legible` tables projected to
their scalar columns (`NEURON_COLUMNS`/`HEAD_COLUMNS`), read over HTTP ranges
so the bulk columns never download; never `reads`, `drivers`, media or the
causal sample. The binding is staged and promoted to
`models/<id>/units/catalog/` with `catalog.json` (format 1: repository,
commit, slug, per file the upstream LFS sha256, the columns kept, the row
count and the local sha256). A binding of another format, repository or
commit is not read (fetch again with `-f`). Labels are data: every label
carries `LABEL_CREDIT` ("Labels by Claude Haiku 5.5") and the unit's page,
`https://neuron-lens.com/<slug>/L<layer>/n<index>` or `/h<index>` (the
explorer's route). An uncatalogued model is `NoUnitCatalogError` (404);
reading or fetching needs pyarrow (`drowse.ai[hf]`), else
`UnitCatalogDependencyError` (503). `CatalogBinding` opens the binding lazily
and reopens after a fetch or removal.

## Testing

`tests/units/test_unit_reads.py` (every read against a hand computation on a
tiny session with hand-built R and S lenses at both sites: writes and fires
on, the scalars, the last layer's logit read, hidden hubs, the contrast cache
following the lens, heads' OV, copy score and inputs, top, find, refusals),
`test_unit_catalog.py` (fetch, projection, provenance, rows, links, the 404,
pyarrow missing, the session's views over a monkeypatched Hub with tiny
parquets), `test_unit_routes.py` (the HTTP routes and the fetch preparation),
`test_unit_stats.py` (the accumulator against a hand
computation on llama and Gemma 3, the cache, the unavailable path, v4
recompute), `test_unit_probes.py` (the tap by hand on llama, Gemma 3 and
Gemma 4's mixed head widths, probes through a real tiny session and a tiny
Gemma 4 one, gates with and without an attached probe, replay, refusals,
the HTTP family block, the probe routes and the compat extension),
`test_unit_steering.py` (ablation against a hand-patched forward to 1e-5,
partial coefficients, heads, triggers and gates,
the push against the kernel on the write direction, 422s and 400s). Helpers:
`tests/support/units.py`. GPU: `tests/gpu/test_smoke.py::TestAblationPerformance::test_throughput_with_unit_ablation`
(8 neurons and 2 heads ablated, ≥ 80% of vanilla tok/s).
