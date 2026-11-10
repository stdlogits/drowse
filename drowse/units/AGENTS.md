# drowse/units

MLP neurons and attention heads as objects drowse reads, gates on and steers:
the fourth read family (`units`), its neutral statistics, its probe tap, and the
runtime that resolves a unit reference against the loaded model. Unit directions
(`neuron_write`, `neuron_read`, `head_write_matrix`, `head_read_rows`) are pure
functions of a `BlockSpec` in `model/units.py`; the block description is
`model/blocks.py`. Unit ablation's hook (`UnitOp`) lives in `steering/hooks.py`.

| Module | Owns |
|---|---|
| `stats.py` | `UnitStats` / `LayerUnitStats`, `UnitStatsAccumulator` (the neutral-pass hook pair), `top_median`, `soft_scale`, `UnitScales`, the stored tensor names |
| `runtime.py` | `UnitsRuntime` (block description once, `check`, `site`, `write_profile`, `stats`, `scales`), `UnitSite`, `UnitsUnavailableError` (422), `unit_ref` |
| `capture.py` | `UnitTap` (one forward hook per touched module, a device ring, post-forward values), `TapUnit`, `SCALE_FLOOR` |
| `instrument.py` | `UnitsInstrument` (`session.units`), `UnitsRun` |
| `reads.py` | `UnitReader` (static reads, `top`, `find`, `stats`), the result types (`NeuronRead`, `HeadRead`, `Firing`, `TopResult`, `FindResult`, ...), `ContrastCache`/`ContrastStats`, `own_token_z`, `copy_score`, `token_logits` |
| `catalog.py` | the neuron-lens catalog: `fetch_catalog`, `catalog_status`, `list_catalogs`, `remove_catalog`, `UnitCatalog`, `CatalogBinding` (the session's view), `unit_url`, `LABEL_CREDIT` |

Layer 7, beside `lens` and `sae`: imports `model`, `store`, `readout`, `lens` and
below, never `steering.composer`, `generation` or the session. The composer
reaches it through its `UnitSource` protocol; the session wires the reader to the
lens instrument, the catalog binding and the weights fingerprint.
drowse.ai's offline unit packs call `UnitReader.population_artifacts`,
`null_basis` and `head_kept_ids`.

## The units

A unit reference is `naming.NeuronRef` / `HeadRef` (`neuron/L8n344`,
`head/L20h17`; `parse_unit_ref`). Definitions follow neuron-lens (`design.md`
"The units"):

- neuron `n` of block `L` is coordinate `n` of the down projection's input,
  `act = act(gate(x^)) * up(x^)`; it takes both signs, so it has two poles;
- head `h` of block `L` is its `head_dim` slice `z_h` of the output projection's
  input; its write is `W_O^h z_h` (with Gemma's post-attention norm: its gain,
  with the per-position scale held constant, the LN rule).

`UnitsRuntime.check` answers an unknown unit with `UnitIndexError` (400) naming
the valid range. A model whose blocks `model_spec` refuses raises
`UnsupportedBlockError` (422) on every unit call; the description is discovered
once (inside the exclusive section) and kept, so later checks touch no model.

## Neutral statistics (`stats.py`)

Measured on the neutral pass (`fit/neutral.py`), stored in the neutral cache
shards beside each layer's activations (`store/neutral.py`), loaded by
`session/neutral.py::NeutralBaseline`.

| Tensor | Shape | Definition |
|---|---|---|
| `neuron_mean` | `[d_ff]` | mean `act` over every real position `>= 1` |
| `neuron_scale` | `[2, d_ff]` | neuron-lens `act_top` per pole: per document the largest `+act` (row 0) / `-act` (row 1) over positions `>= 1`; the median of the `UNIT_TOP_DOCS = 24` largest document maxima, clamped at 0 |
| `head_mean` | `[n_heads, head_dim]` | mean `z_h` over positions `>= 1` |
| `head_scale` | `[n_heads]` | `act_top` of the write norm `||W_O^h z_h||` (sandwich scale included) over positions `>= HEAD_SCALE_FROM = 16` |

Position 0 is the attention sink (massive activations) and never counts; a head's
scale starts later because a head with few keys attends sharply. With fewer than
24 documents the median runs over those seen (`top_median`). Everything streams:
per chunk the accumulator keeps sums and per-document maxima on device in fp32,
merges them into running top-k buffers on `commit_chunk`, and drops them on
`discard_chunk` (the capture's out-of-memory retry re-runs those rows). One
down-projection pre-hook and one output-projection forward hook per block, on the
same forward the neutral activations come from.

Units never fail the neutral pass. A model `model_spec` refuses, a hook that
raises anything but out-of-memory, or an out-of-memory that persists at capture
batch size 1 (the capture `abandon`s the accumulator and re-runs without its
hooks) publishes a complete cache whose `units` block reads `{"available":
false, "reason": ...}`, and reading units raises `UnitsUnavailableError` (422)
with the reason. Only the refusal is sticky: a later load of a cache unavailable
for a model `model_spec` accepts recomputes it.

### Reading scale

A reading (`activation_over_scale`) divides by one source's `act_top`
(`stats.UnitScales`, `UnitsRuntime.scales()`):

- `"catalog"`: with a neuron-lens catalog bound (`unit fetch`), its `act_top` per
  pole and `out_top` per head (the same statistic over 15.4M positions;
  `UnitCatalog.reading_scales`, checked against the model's shape). 1.0 is the
  pole's typical high firing in text.
- `"neutral"`: otherwise the neutral statistics. 96 statements are too few for a
  top-24-of-documents statistic (on Qwen3-8B the catalog's `act_top` is a median
  12x the neutral one, 8.5x at L0 up to 22.9x at L35), so neutral readings are
  relative to the neutral text, and every view names the source.

Either way the scale is soft: `stats.soft_scale` is `sqrt(s^2 + (c * m)^2)` with
`c = SCALE_FLOOR_SHARE = 0.25` and `m` a typical scale (over the catalog, the
median positive scale of the layer's neurons or heads; over neutral statistics,
the larger of the layer's and the model's, so it cannot collapse with a quiet
layer). Neutral text leaves some poles at `act_top` 0 (Qwen3-8B L2: 4,856 of
24,576), and without the floor such a pole read 849 and topped the Strongest list.
At `c = 0.25` a unit at its layer's median reads 3% lower and one at half the
median 11% lower, while the tail is held (SmolLM2-360M: 0.24% of neuron poles and
1.25% of heads sit below a quarter of their layer's median). It is a floor, not a
cut: every unit reads and ranks.

Probes, gates, the units token readout, unit activity and the anatomy's unit rows
all divide by it, so one unit shows one number everywhere, and each names the
source (`meta.scale`, unit activity's `scale`, the anatomy's `units_scale`,
`Firing.scale`). The tap takes its scales when a run binds, so a catalog bound
mid-generation reaches the next one. `capture.SCALE_FLOOR = 1e-6` is only the
numeric guard for a layer whose units never fired.

## The read family (`instrument.py`, `capture.py`)

A unit probe is one `ScalarReading` in `activation_over_scale` at one layer
(`per_layer = {L: value}`, depth basis `single_layer`, center `L/(n_layers-1)`),
with the raw firing in `meta` (`activation` for a neuron, `write_norm` for a
head), `meta.unit` the canonical reference and `meta.scale` the scale's source:

- neuron: `act / scale[pole]`, the pole the sign of `act` picks, so 1.0 is the
  pole's typical high firing on either side (`< 0` on the `-` pole);
- head: `||W_O^h z_h|| / scale`, never negative. The write norm is Euclidean in
  the residual, one of the deliberate exceptions to Mahalanobis-only
  (`drowse/geometry/AGENTS.md`): a scalar firing read against the head's own
  scale, not a geometric comparison.

One channel (`<name>`, `<name>[0]`); `validate_gate` refuses geometry channels as
the lens and SAE do.

**The tap.** A bound run owns a `UnitTap` only while something reads it: the
attached probes when a per-token consumer (`ReadRequest.live`) or the final
aggregate reads them, plus every unit a gate names. A gate needs no attached
probe: `@when:neuron/L8n344 > 2` reads the unit for the gate alone, and an
unknown unit there is a 400 at generation preflight (`ReadPlane.resolve_gates`).
The tap installs one forward hook per touched module (after every pre-hook, so a
probe reads what the model used, ablations included), each an `index_select` of
its units' coordinates at the last position into a preallocated ring row
(`out=`), plus `vector_norm(..., out=)` of the output row for a sandwich head. No
allocation, no host sync; the module that runs last in a forward advances the
count. The ring is `AGG_TAIL_DEPTH` forwards deep, and the hooks come off with
the residual capture when decode ends (`ReadPlane.end`). `values_at(fwd)`
computes one fp32 vector after the forward (heads in one batched `z^T G z` per
`head_dim`, since a model may mix sizes: Gemma 4's sliding and global layers);
the run moves it to host once per step, stashed by `step_id` so the gate callback
and token tap share it, and once for the aggregate.

**Reads.** `UnitsRun.gate_scalars` (gate callback), `step_readings` (token tap),
`aggregate_at(fwd)` (finalize: the residual pooled slice carries no unit
activation, so `observe_aggregate(pooled)` reads the latest forward instead). The
units plan declares no residual layers, so the family is invisible to
`plan_capture`. `UnitsInstrument.hooks_attached` (via `ReadPlane.transient_hooks`)
routes a compiled session to its eager module.

**Other surfaces.** `attach(selector, as_name=)` (the roster routes `neuron/` and
`head/`), `try_detach`, `specs`, `names`, `probe_hash` (the unit plus its scale
and source; re-taken when a run's tap binds to other scales),
`live_state`/`set_live`, `token_readout` (replays the producing forward with a
depth-1 tap; `top_k` and `layers` are 400). `generate_batch` runs serially when
unit probes are attached and readings are wanted (the tap reads one row). Over
HTTP the probe routes attach, list (`UnitsProbeInfo`: `family: "units"`, `unit`,
`kind`, `layer`, `index`) and detach units, and `x-drowse-probe-readings` carries
their aggregates.

## Over a whole node (`loom/node_replay.py`)

`session.unit_activity(node_id, ...)` replays a loom node once and reads every
neuron and head at every token in `activation_over_scale`, reduced on device: the
top neurons and top heads by peak `|value|` (ranked separately, since their values
are different quantities), peak tokens, per-token traces of named units, and one
token's whole population as float16 base64. It uses the reading scales directly
(no tap), skips the sink (position 0, value 0) and ranks over the soft scale.
`session.token_anatomy` instead ranks units by direct contribution to one token
through the R lens. Details: `drowse/loom/AGENTS.md`.

## Steering

- **Push.** `alpha neuron/L8n344` resolves through the registry's `neuron` atom
  resolver to `register_direction`: the `+` pole's write `neuron_write(block, n)`
  at layer `L`, registered as profile `neuron/L8n344`, then folded, normalized and
  pushed like a one-layer `jlens/` atom (the one kernel, the affine fast path). A
  negative alpha pushes along the `-` pole's write. There is no head push.
- **Ablation.** `alpha !neuron/...` / `alpha !head/...` (`UnitAblationTerm`)
  compose via `UnitsRuntime.site` into `SteeringPlan.add_unit_ablation` and lower
  to one `UnitOp` per (module, trigger): `x <- x (.) keep + bias` on the
  projection's input, `keep = 1 - alpha`, `bias = alpha * mean` on the unit's
  coordinates (alpha = 1 is exact mean-ablation; other finite alpha blends,
  overshoots or amplifies). Position 0 is skipped, since the unit means count
  from position 1 and substituting there would put an out-of-distribution value
  at the sink; every other position is ablated, prefill included. The driver
  names the sink column (`TriggerContext.sink_column`: 0 at rest and on a prefill,
  `None` on an incremental step); a left-padded batch puts it in a different
  column per row, so a batch with a unit ablation runs serially. Routing:
  `drowse/steering/AGENTS.md`.

## Static reads (`reads.py`)

What a unit is about, from its weights, by the four-lens rule
(`drowse/lens/read.py`): site from where the vector sits, kind from the depth, so
every place returns an S half and an R half with the depth rule's weights
(`session.lens.band`); the light half is returned for the UI to fade. Direction
reads are uncentred (only residual reads centre on the neutral mean).
`session.units.read(unit, k=8, hide_hubs=False)`:

- **neuron**, per pole: `writes` reads `+-w` (`neuron_write`, Gemma's
  post-feedforward gain and layer scalar included) at `out` of its layer;
  `fires_on` reads `a^ +- b^` at `mid`; both through `LensInstrument.read`. The
  last block's `out` is the final residual, where the R read is exactly the logit
  lens (`J = I`), labelled `logit` (`lens/read.py::output_transport`, shared with
  `_r_matrix`, the `R-logit` population statistics). Scalars as neuron-lens:
  `w_norm`, `cos_ab`, `null_share` (squared-norm share in the bottom 1% of
  right-singular directions of `W_U diag(gamma_final)`, basis built once per
  session), per pole the logit-lens skew (negated for `-`) and kurtosis over the
  token mask, the neutral mean act.
- **head**: `copies` / `suppresses` (at `mid`) are the top and bottom of each
  token's own-column z in the OV map `M[u, v] = A_u . B_v`, `B_v = W_V^h X_v` with
  `X_v = K^{out}_{L-1} e~_v` (the embedding at L0), `A_u` the R
  (`(gamma_f (.) W_U[u]) J^{mid}_L W_O^h`) or S (`e~_u K^{mid T}_L W_O^h`) read of
  `u`; `copy_score` is `sum(lambda)/sum|lambda|` of `B^T A` per family. The map is
  never built (the mean and covariance of `A`'s rows suffice). `inputs` (at `L-1`
  out) ranks, per projection (query, key, value), tokens by the share of their
  unit direction the projection keeps, `||W^h x^_v||`, as a z over the mask
  (`energy_z`). Without an S lens the OV read is absent. A block with head norms
  (q/k norms; Gemma 4's weightless value norm) reads each projection through its
  norm's true derivative averaged over the residuals entering the block, as
  neuron-lens does (`norm_derivative`, positions weighted by the input norm's
  `1/rho`). Those residuals are the neutral ones at `L-1` out (the whitener's
  rows; at L0 the kept tokens' embeddings); without them the read is a
  `WhitenerError`. Products take the `[d, hd]` factors first and stream the
  vocabulary in chunks, so no `[V, d]` tensor is held.
- **hidden hubs** (`hide_hubs=True`): each half ranks by neuron-lens's
  layer-contrastive `_z`, the token's score z-scored across the layer's
  population. Neurons: R over the lens logits of the `+` writes (the `-` pole is
  minus it) and of each pole's read directions separately; S over raw scores of
  unit covectors (`u^ . e~_v`) of the `+` writes and of both poles' reads
  together. Heads: each token's OV z across the layer's heads (unbiased std). A
  head's inputs never contrast: heads of one KV group share key and value rows, so
  a contrast saturates and a few KV heads are too small a population.

`ContrastCache` keeps the per-layer statistics (per-token mean and std, plus each
row's `logsumexp` for R, which `top` needs for probabilities) under
`models/<id>/units/contrast/<family>-<lens identity>.../L<layer>.safetensors`: a
derived cache whose shard is recomputed when its record (format, lens, place,
token-mask digest, weights fingerprint) differs. The first hidden-hub read or
`top` of a layer pays its population pass (all the layer's units through both
lenses).

`top(token, by="writes"|"fires", layers=None, k=20)` scores every pole of each
layer by the token's layer-contrastive z through each present family,
`sum(w_f z_f) / sum(w_f)` with the depth weights, and lists the best `k` with each
half's z, raw read (R probability, S footprint z) and the catalog label. Neurons
only. `find(selector | vector, layer=, k=16, layers=)` is `lens.decompose(...,
dictionary="neurons")` (`NeuronWrites`: each layer's signed writes in the whitened
metric) with each neuron's pole and label. `stats(unit)` is the firing block
alone.

`population_artifacts(layer)` returns what a reader without a vocabulary pass needs
at one layer, as host fp32 tensors keyed by name (a key absent where its lens half
is): per neuron population the R `rho`, `lse` (`lse_neg` for writes) and contrast
`mean`/`std`, the S `norm`, `footprint_std` and contrast; the heads' `ov` contrast
per family over `head_kept_ids()`; head-norm mean derivatives; the whitened write
norms `find` divides by. `null_basis()` is public beside it.

**Firing.** `Firing.scale` is `{"source": "catalog" | "neutral", "detail", "top",
"value"}` (`top` the source's `act_top`, `value` the soft scale over it).
`Firing.neutral` is drowse's own statistics (a pole's `act_top`, the mean, the
document count; a head's write `act_top` and mean `||z_h||`). The neutral cache
has no firing rate, so `rate` ("fires on X% of tokens": the catalog's share of
tokens with `|act| > act_top/4` over its activation corpora) comes only from a
bound catalog, else is absent with the reason.

## The catalog (`catalog.py`)

`stdlogits/neuron-lens` catalogues every pole and head of `Qwen/Qwen3-8B` and
`google/gemma-4-12B` (base models; an instruction-tuned checkpoint has other
neurons and gets none). `unit fetch` binds it at `CATALOG_REVISION`, copying
`neurons/labels/L<LL>.parquet` and `heads/labels.parquet` whole and the `acts` and
`legible` tables projected to their scalar columns (`NEURON_COLUMNS`/
`HEAD_COLUMNS`) over HTTP ranges, so bulk columns never download, plus
`token_counts.pt`'s total counts (loaded `weights_only`, stored as
`token_counts.safetensors`: the S read's token mask, `drowse/lens/AGENTS.md`).
`reads`, `drivers`, media and the causal sample are not fetched. The binding is
staged and promoted to `models/<id>/units/catalog/` with `catalog.json`; one of
another format, repository or commit is not read (fetch again with `-f`). Every
label carries `LABEL_CREDIT` ("Labels by Claude Haiku 5.5") and the unit's page,
`https://neuron-lens.com/<slug>/L<layer>/n<index>` or `/h<index>`. An
uncatalogued model is `NoUnitCatalogError` (404); reading or fetching needs
pyarrow (`drowse.ai[hf]`), else `UnitCatalogDependencyError` (503).
`CatalogBinding` opens lazily and checks `catalog.json`'s on-disk identity (inode,
mtime, size; one `stat`) on every `get`, so a fetch or removal from another
process (`drowse unit fetch` beside a running server) reaches the session at
once, a cached "no catalog" included. A unit view takes `bound`, `url` and
`reason` from one `view()`, so an unbound block says why.

## Tests

`tests/units/` (CPU; helpers in `tests/support/units.py`).
`tests/gpu/test_smoke.py::TestAblationPerformance::test_throughput_with_unit_ablation`
requires 8 neurons and 2 heads ablated at >= 80% of vanilla tok/s.
