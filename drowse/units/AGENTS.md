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

**The reading scale** (`stats.UnitScales`, `UnitsRuntime.scales()`). A
reading (`activation_over_scale`) divides by one of two sources' `act_top`:

- `"catalog"`: with a neuron-lens catalog bound (`unit fetch`), its
  `act_top` per pole and `out_top` per head (the same statistic over 15.4M
  positions; `UnitCatalog.reading_scales`, read once per opened binding and
  checked against the model's shape, a mismatch a `UnitCatalogError`). 1.0 is
  the pole's typical high firing in text.
- `"neutral"`: otherwise, the neutral statistics' (`UnitStats.reading_scales`).
  96 statements are too few for a top-24-of-documents statistic: on Qwen3-8B
  the catalog's `act_top` is a median 12x the neutral one (by layer: L0 8.5,
  L2 8.6, L8 12.2, L17 9.7, L24 13.1, L33 15.9, L35 22.9), so a reading is
  relative to the neutral text, and every view says so.

Either way it is a soft scale, never the bare `act_top`: `stats.soft_scale`
blends the unit's own `s` with a floor, `sqrt(s² + (c · m)²)`, `c =
SCALE_FLOOR_SHARE = 0.25` (`stats.reading_scales`). Over the catalog `m` is
`typical_scale`, the median of the layer's positive scales (a neuron's over
both poles of every neuron, a head's over the layer's heads); the catalog is
robust, so a quiet layer is truly quiet (Qwen3-8B's L1-L2 catalog medians are
0.009 and 0.014, with 1,559 and 3,149 poles at 0). Over the neutral statistics
`m` is the larger of the layer's and the model's typical scale, a floor that
cannot collapse with the layer: Qwen3-8B's neutral L2 median is 0.0024 with
4,856 of 24,576 poles at 0, and a quarter of it read `L2n7950+` (neutral
`act_top` 0, catalog 0.867) at 849, topping the Strongest list. Why a quarter,
measured on SmolLM2-360M (96 documents): a neuron pole's `act_top` over its
layer's median has its 0.1%, 1% and 5% quantiles at 0.20, 0.33 and 0.47 (heads
0.13, 0.24, 0.45), and 0.24% of neuron poles (1.25% of heads) sit below a
quarter of it. At `c = 0.25` a unit at the median reads 3% lower than over its
own scale and one at half the median 11% lower, while the tail is held
(SmolLM2's `L31n2394+`, 0.055 against its layer's 2.28, topped the activity
list at 169 over its bare scale). It is soft, not a cut: every unit reads and
ranks, and a pole with no scale at all reads over `c · m`. Probes, gates, the
units token readout, unit activity and the anatomy's unit rows all divide by
it, so one unit shows one number everywhere, and every one names the source:
a reading's `meta.scale`, unit activity's `scale` (`{"source", "detail"}`),
the anatomy's `units_scale`, the unit view's `Firing.scale` and the CLI's
firing line. The tap takes its scales when a run binds, so a catalog bound
mid-generation reaches the next one. `capture.SCALE_FLOOR` (1e-6) stays only
as the numeric guard of a layer whose units never fired.

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
then raises `UnitsUnavailableError` (422) with that reason. Only the refusal
lasts: a later load of a cache whose units are unavailable for a model
`model_spec` accepts recomputes it (`drowse/fit/AGENTS.md`).

## The read family (`instrument.py`, `capture.py`)

A unit probe is one `ScalarReading` in the `activation_over_scale` unit at one
layer (`per_layer = {L: value}`, depth basis `single_layer`, center
`L / (n_layers − 1)`), with the raw firing in `meta` (`activation` for a
neuron, `write_norm` for a head), `meta.unit` the canonical reference and
`meta.scale` the reading scale's source (`"catalog"` or `"neutral"`):

- neuron: `act / scale[pole]` (the reading scale, above), the pole the sign
  of `act` picks, so `1.0` is the pole's typical high firing on either side
  (`< 0` on the − pole);
- head: `‖W_O^h z_h‖ / scale`, never negative. The write norm is Euclidean
  in the residual: one of the deliberate exceptions to Mahalanobis-only
  (`drowse/geometry/AGENTS.md`), since it is a scalar firing read against the
  head's own reading scale, never a geometric comparison.

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
`probe_hash` (the unit plus its reading scale and source), `live_state`/`set_live`
(all-or-nothing, no extras), `active_source` (`None`: no source lifecycle),
`token_readout` (replays the producing forward with a depth-1 tap attached
inside the replay's exclusive section; `top_k` and `layers` are 400).
`generate_batch` runs serially when unit probes are attached and readings are
wanted (the tap reads one row). Over HTTP the native probe routes attach, list (a
`UnitsProbeInfo` row: `family: "units"`, `unit`, `kind`, `layer`, `index`) and
detach units, and the OpenAI/Ollama `x-drowse-probe-readings` extension
carries their aggregates (`server/runtime.py::attached_probe_names` unions
every family's roster).

## Over a whole node (`loom/node_replay.py`)

`session.unit_activity(node_id, …)` replays a loom node once and reads every
neuron and head at every token in the same unit (`activation_over_scale`,
each token at its producing forward, as its loom row's live readings and
`token_readout` are), reduced on the device: the top neurons and the top heads
by peak `|value|`, each kind ranked on its own (their values are different
quantities, and one list would hold no head where neurons run hotter), with
their peak token, per-token traces of named units, and one token's whole
population packed as float16 (little-endian, base64). It uses this package's
statistics' reading scales directly (no tap), skips the sink (position 0:
value 0, never a peak), and ranks every unit over its soft scale (above), so
a pole that barely fired on the neutral corpus no longer tops the list on a
near-zero divisor. `session.token_anatomy` ranks units by their direct
contribution to one token through the R lens instead. Details:
`drowse/loom/AGENTS.md`.

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
  the R read is exactly the logit lens (`J = I`), labelled `logit`: the lens
  read's own rule (`lens/read.py::output_transport`), which `_r_matrix` (the
  population statistics, `R-logit` in the contrast cache) shares. Scalars as neuron-lens: `w_norm`,
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
  lens the OV read is absent (its input side reads through it). A block with
  head norms (q/k norms; Gemma 4's weightless value norm) reads each
  projection through its norm's true derivative averaged over the residuals
  entering the block, as neuron-lens does (`norm_derivative`, each position
  weighted by the input norm's `1/ρ`): `B` through the value norm's, each
  input through its own. The residuals are the neutral ones at `L − 1` out
  (the whitener's rows, one pooled position per neutral statement; at L0
  the kept tokens' embeddings); without them the read is a `WhitenerError`.
  Every product takes the `[d, hd]` factors first and streams the vocabulary
  through them in chunks (two passes for the OV z: `A`'s mean and scatter,
  then each token's z), so a read never holds a `[V, d]` tensor.
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

**Firing.** `Firing.scale` is what the unit's readings divide by:
`{"source": "catalog" | "neutral", "detail", "top", "value"}` (`top` the
source's `act_top` of the pole or head, `value` the soft scale over it).
`Firing.neutral` is drowse's own statistics (a pole's `act_top`, the mean,
the document count; a head's write `act_top` and mean `‖z_h‖`). The neutral cache has no firing rate, so "fires on X% of tokens"
(`rate`: the catalog's share of tokens with `|act| > act_top / 4` over its
activation corpora, 15.4M positions) comes from a bound catalog, else is
absent with the reason.

## The catalog (`catalog.py`)

`stdlogits/neuron-lens` (the dataset neuron-lens publishes) catalogues every
pole and head of `Qwen/Qwen3-8B` and `google/gemma-4-12B` (the base model; an
instruction-tuned checkpoint has other neurons and gets none). `unit fetch`
binds it at `CATALOG_REVISION`, copying `neurons/labels/L<LL>.parquet` and
`heads/labels.parquet` whole and the `acts` and `legible` tables projected to
their scalar columns (`NEURON_COLUMNS`/`HEAD_COLUMNS`), read over HTTP ranges
so the bulk columns never download, and `token_counts.pt`'s total counts
(downloaded whole, loaded `weights_only`, stored as
`token_counts.safetensors`: the S read's token mask, `drowse/lens/AGENTS.md`);
never `reads`, `drivers`, media or the causal sample. The binding is staged
and promoted to `models/<id>/units/catalog/` with `catalog.json` (format 2:
repository, commit, slug, per file the upstream LFS sha256, the columns kept,
the row count and the local sha256; a format 1 binding, without the counts,
is not read). A binding of another format, repository or
commit is not read (fetch again with `-f`). Labels are data: every label
carries `LABEL_CREDIT` ("Labels by Claude Haiku 5.5") and the unit's page,
`https://neuron-lens.com/<slug>/L<layer>/n<index>` or `/h<index>` (the
explorer's route). An uncatalogued model is `NoUnitCatalogError` (404);
reading or fetching needs pyarrow (`drowse.ai[hf]`), else
`UnitCatalogDependencyError` (503). `CatalogBinding` opens the binding lazily
and checks `catalog.json`'s on-disk identity (inode, mtime, size; one `stat`)
on every `get`, so a fetch, refetch or removal from another process (`drowse
unit fetch` beside a running server) reaches the session at once, a cached
"no catalog" included. A unit view takes `bound`, `url` and `reason` from one
`view()`, so an unbound block always says why.

## Testing

`tests/units/test_unit_reads.py` (every read against a hand computation on a
tiny session with hand-built R and S lenses at both sites: writes and fires
on, the scalars, the last layer's logit read, hidden hubs, the contrast cache
following the lens, heads' OV, copy score and both halves of the inputs,
Gemma 4's head norms against autograd through its own norms on a tiny Gemma
4 session (`tests/support/units.py::GEMMA4_MIXED_HEADS`), the refusal without
neutral residuals, a head read that never makes a `[V, d]` tensor, top,
find, refusals), `test_unit_catalog.py` (fetch over a fake Hub sibling
listing, so the real path filter runs, projection, provenance, rows, links,
the 404, pyarrow missing, the session's views, a binding made or removed by
another process), `test_unit_routes.py` (the HTTP routes and the fetch preparation),
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
