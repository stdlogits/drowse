# lens/

The Jacobian lens family (Gurnee et al., "Verbalizable Representations Form a
Global Workspace in Language Models", Transformer Circuits 2026). It is a
**per-model** artifact. A lens holds one fp32 matrix per layer at each of its
sites: `J_l = E[∂h_final/∂h_l]` for the output kinds, `K_l = E[∂h_l/∂e]` for
the input kinds, averaged over positions and a text corpus. The R family
(J/R) reads what a vector is disposed to *say*: the vocabulary readout
`softmax(W_U · norm(J_l h))`, the `jlens/<word>` steering atom `W_U[v] @ J_l`
and the `tokens-out` decomposition. The S family (K/S) reads what it has
*read*: the footprint z of every token (`footprint.py`), the `slens/<word>`
atom `K_l ẽ_v` and the `tokens-in` decomposition. `read.py` reads any vector
at a place through both, by the four-lens rule. Inside a session the face is
`session.lens` (`LensInstrument`). The package sits beside `sae`, above `readout` and below
`steering.composer`, and imports only `errors`, `base`, `model` and `readout`
(`tests/invariants/test_layering.py`, which counts `TYPE_CHECKING` imports).

| Module | Owns |
|---|---|
| `key.py` | `LensKey` (kind × site), `FITTABLE_KINDS`, `site_layers`, `DEFAULT_FIT_DOCUMENTS`, local-name grammar. Torch-free. |
| `jlens.py` | `JacobianLens` (one kind, both sites), readout math, token-atom resolution, error types, fit defaults. Imports only torch, errors and `key` (decode-step code imports it). |
| `depth.py` | `LensBand`, `kind_weights`: which kind reads a place, by depth. Torch-free. |
| `footprint.py` | `token_mask`, `TokenEmbeddings` (masked embedding mean and `Σ_E`, fp32), footprint z, `pullback_rows`, `depth_moments_positive`. Torch only. |
| `read.py` | `plan_reads` / `PendingReads` (a batch of reads on the device, one host transfer), `read_vectors`, `VectorRead`, `ReadOf`, `LensRead` / `FamilyRead` / `TokenScore`: the read of a vector at `(layer, site)` through both families |
| `decompose.py` | `LensDictionary`, `TokensOut` (`W_U J_l`), `TokensIn` (`K_l ẽ_v`), `NeuronWrites` (the layer's neuron writes, signed), `sparse_decompose`, `DICTIONARIES` |
| `transport.py` | The estimator's per-document block maps: `linearize`, `jvp`, `vjp`, the sweeps, `document_estimate` |
| `fit.py` | `fit_lens` (transport over a corpus), `LensSums`, `LensFit`, `fit_site_layers`. No runtime path imports it. |
| `relp.py` | `RELP_RULES`, `HALF_RULE_BETA`: the RelP rules the R/S kinds are taken under |
| `corpus.py` | Hub corpora (pin and stream in a worker), `read_corpus_file`. `sae train` uses it too. |
| `lifecycle.py` | `plan_lens_fit` (pure), `LensFitter` (executor), `prove_lens_fit_noop` (model-free) |
| `store.py` | Format-v7 local artifacts: save/load/checkpoint/promote/remove, `LensProvenance`, `lens_estimator_policy`, `lens_identity` |
| `sources.py` | `LENS_SOURCES` per-family selection, multi-site provider bindings and fetch, default source order |
| `runtime.py` | `LensRuntime` (resident lens, identity, bounded caches, readout primitives), `matches_loaded_model` |
| `instrument.py` | `LensInstrument` (residency of both families, fit, reads, probes, gates, live, replay) and `LensRun` (per-generation state) |

## Identity: `LensKey`

A lens's **kind** is the Jacobian's direction crossed with its derivative rules:
J (output, standard), R (output, RelP), K/S (input, standard/RelP). A
**site** is `out` (block output) or `mid` (post-attention residual, the MLP's
input). `LensKey(kind, site)` names one kind at one site. One estimator
(transport) fits every kind at either or both sites (`FITTABLE_KINDS` is all
four). Every store, binding and selection path is keyed on `LensKey`. Probes,
gates, steering atoms and the live readout read block outputs, so they use
`out`; `read` and `lens top` take the site.

**One lens, both sites.** A `JacobianLens` holds its kind (`key`, whose site
is `out`; `kind`) and fp32 per-layer matrices at one or both sites:
`jacobians` (positional) is the `out` site and `mid=` the `mid` site. Read a
site explicitly with `matrices(site)`, `layers(site)` or `matrix(layer, site)`;
`sites` lists what it holds and `site_key(site)` gives `R:mid`.
`jacobians`/`source_layers` and the `token_direction`/`atom_norms` readers
default to `out`, so the output-side readout code reads `out` unchanged. A
read at a site the lens lacks raises `LensNotFittedError` (404) naming the
sites it holds; it never falls back to the other site. `select_layers` takes
per-site requests (`{"out": [...], "mid": [...]}`), keeps exactly the sites it
names and asks each only for its own layers, since the sites' layer sets
differ by design (below); `union_layers` works site by site.

**Orientation follows the kind.**

| Kind | Matrix | Shape | Reads |
|---|---|---|---|
| J/R (output) | `J_L = E[Σ_{t'≥t} ∂h_final(t')/∂h_L(t)]` | `[d, d]` | `transport(h, L, site)` = `h @ J_Lᵀ`, into the final basis |
| K/S (input) | `K_L = E[Σ_{s≤t} ∂h_L(t)/∂e(s)]` | `[d, d_emb]` | `pullback(c, L, site)` = `c @ K_L` (a residual covector into the embedding basis); `footprint(ẽ, L, site)` = `K_L ẽ` (an embedding-space vector into the residual) |

K/S follow rs-lenses' `s-lens/{out,mid}.pt` layout. A read of the wrong
orientation raises `JacobianLensError` (422). Construction checks every matrix
against `matrix_shape`; `d_emb` defaults to `d_model` and an output kind has
none of its own. The R-family's `out` site stops below the last layer (its
target); `mid` and the S family cover every layer. At the last block's `out`
the output kinds' transport is the identity (`J = I` exactly, under the
standard and RelP rules alike: no block lies between the site and the
target), so every read there is the logit lens, labelled `logit`:
`read.py::output_transport` is the one place a read meets that layer
(`plan_reads`' R half, `read_prompt`'s default layers, the unit reads'
population statistics, and the anatomy's contribution cotangent, which is the
token's own logit cotangent `u` there). The steering atoms (`jlens/`), probes
and the live readout keep the fitted layers.

**Which kind reads a place** (`depth.py`). Site follows the object's position
(the lens rule: `out` reads residuals at a block's output, neuron writes and a
head's inputs at `L − 1`; `mid` reads neuron read directions and head
writes). Kind follows depth: S is the faithful read early, R late, both
weighted inside a crossover band. `LensBand(lo, hi)` gives the band as
fractions of depth (`layer / (n_layers − 1)`, default 0.15 and 0.35 until
measured per model); `kind_weights(n_layers, band)` returns `(w_S, w_R)` per
layer, `w_S = 1` below `lo`, 0 above `hi`, linear inside, summing to 1.

**The read** (`read.py`, `LensInstrument.read`). `read(vector, layer, site, *,
of, k=8)` takes a `[d]` vector (one `LensRead`) or `[n, d]` rows (a list).
**What is read is part of the request and has no default** (`of`, a
`ReadOf`): a `"residual"` (the stream at a block's output or `mid`) or a
`"direction"` (a steering vector, a unit's write, an attention or MLP write
delta). The S half of a residual reads `h − μ`, `μ` the neutral layer mean at
the read's site (`out`: the monitor's and whitener's centring mean, `X.mean(0)`
of the neutral activations; `mid`: the neutral cache's mid means): the raw
residual's large common component otherwise swamps the position's own tokens
(on SmolLM2, L1 ` Tower` read `ing`, `er`, ` itself` raw and ` towers`, ` Tower`
centred). A direction, and every R half, is read as given. The means come from
the injected `neutral_means(site)` getter (the session's
`NeutralBaseline.site_means`); a residual read with no mean at its site raises
`NeutralMeansError` (422), never reads uncentred. A `LensRead` has `layer`,
`site`, `depth`, `of` and two `FamilyRead` halves, `r` and `s`, each with
`weight` (from `kind_weights`, whether or not the half is present), `unit`
(`probability`: per-layer readout probabilities through the model's final
path; `footprint_z`), `lens` (the label read through, `R:mid`, `J:out`,
`S:out`), `centred` (the S half of a residual), `tokens` (`TokenScore(token_id,
token, value)`, top-k descending) and `absent`, the reason when there is no
lens of the family, no lens at the site, or no matrix at the layer. A vector
whose width is not the model's residual width is a `RequestError`. A half is
never read from the other site or family. The R half at the last block's
`out` reads through the identity (`output_transport`, above), so it is present
whenever the R lens holds the `out` site. The weights stay raw, so a consumer
fades a light half or says that a heavy one is missing. `plan_reads(requests,
k)` (`VectorRead(rows, layer, site, of)` each) computes a batch on the device
and returns `PendingReads`, whose `tensor` leaves in one transfer (alone or
packed into a caller's own) and whose `finish(host)` unpacks it; `read` and
`read_prompt` are its single-transfer uses, token anatomy packs it with its
own results. `read_prompt(prompt, site, layers, positions, k)` reads a raw
prompt's residuals (out: `capture_hidden`; mid: a pre-hook on each block's MLP
input norm through `model_spec`). `neutral_mean(site, layer)` is the mean a
residual read there centres on. `lens.band` (`LensBand()`) is the session's
crossover band. Token anatomy (`loom/node_replay.py`) and the unit views read
through this.

**Footprint z** (`footprint.py`). `u_L = K_Lᵀ h`; `z_L(v) = u_L·ẽ_v /
sqrt(u_Lᵀ Σ_E u_L)`, with `ẽ_v = scale·W_E[v] − mean` (`scale` from
`model_spec`, Gemma's `√d`) and `Σ_E` the covariance of `ẽ` over the token
mask. The mask is the tokenizer's alone: it drops special tokens, every added
token (special or not: Qwen3's `<think>`/`<tool_call>`, Gemma's
`<unusedN>`), byte-fallback pieces (spelled `<0xNN>`, including the ASCII ones
that decode to a control character, or decoding alone to U+FFFD) and
embedding rows past the tokenizer. `z` is the vocabulary z-score of the raw
score, so its top-k is neuron-lens's raw `read_S`/`write_S` wherever both keep
the token. neuron-lens keeps the tokens seen at least five times in its
corpora instead (rare base tokens differ, and added tokens a corpus contains),
scores a unit covector in fp16, and its `_z` reads are layer-contrastive
(across a layer's neurons), which a single read cannot be.
`TokenEmbeddings` is built once per session on the first S use (model work:
inside the exclusive section) and holds `Σ_E` on the model device (`d²` fp32).
A one-token read (`z_of`) costs `O(d²)` per layer, not a vocabulary pass.

- A local directory name carries its kind (`^[jrks](-tag)?$`: `r`, `j`,
  `r-chat`). Every writer refuses a mismatch, so a fit cannot land in or resume
  from another kind's shards. Fits always write the untagged `local:<kind>`
  (`local:r`, `local:j`, `local:s`, `local:k`); a tagged name appears only
  through a direct `save_lens(name=…)`.
- `JacobianLens` defaults to `key=LensKey("J")` and casts every matrix to fp32.
  Code that builds an R-lens by hand must pass `key=LensKey("R")`; a key with
  site `mid` is refused (pass `mid=` matrices).

## On disk (format v7)

```
models/<safe_model_id>/lens/
  active.json            per-side selection {format_version: 2, model_id, output, input}
  bindings/<name>.json   commit-pinned provider binding (format 2)
  local/<kind>[-<tag>]/
    manifest.json        atomic pointer;  checkpoint.json only during a resumable fit
    <site>[.partial].layer-<L>.gen-<id>.safetensors   immutable fp32 [d, d] shards (site out, mid)
```

The manifest schema is exact. It records `kind`, `sites.<site>.{layers,
tensor_files, tensor_sha256}` for each site the lens holds (`out`, `mid`, or
both), `n_prompts`, `d_model`, the `LensProvenance`
fields and `estimator_policy`. The provenance fields are the corpus spec, the
token-id sha256 (`token_ids_v1`), raw corpus digest and counts, `seq_len`,
`dim_batch` (seeds per transport pass), layer count, and the loaded-weight and
source fingerprints. The policy is `{direction, rules, skip_first_positions,
corpus_preprocess_version, doc_chars}`, with `rules = RELP_RULES` for RelP
kinds. There is no `method` field.

Two per-site fields are optional and validated when present.
`sites.<site>.split_half` (`{"<layer>": value}`, every layer of the site,
finite and non-negative) is the fit's split-half difference: local fits
record it, external imports and fits that could not measure it (a corpus
extension of a durable artifact, `split_half=False`) lack it;
`lens_split_half(sidecar, site)` reads it. `sites.<site>.odd`
(`{tensor_files, tensor_sha256}`) appears only in a checkpoint: the odd-half
mean shards (`<site>.odd.partial.layer-L…`) a resumed fit continues its
split-half check from (`load_lens_checkpoint_odd`). Promotion drops it and
generation GC collects the shards.

**v7 reader rule (decided for 6.0; the format freezes with it).** The format
stays v7, and these optional fields are part of v7, not a v8. A v7 reader
must accept `sites.<site>.split_half` on any site of a manifest or a
checkpoint (validating it as above, or ignoring it), and must accept
`sites.<site>.odd` only in `checkpoint.json`: `odd` in a durable
`manifest.json` is malformed and reads as no lens. A reader's exact-key check
is therefore "the required site keys, plus at most these optional ones"
(`store.py::_validate_manifest`; `tests/lens/test_fit_split_half.py` pins both halves).
drowse.ai's browser reader checks exact key sets today, so it must adopt this
rule before it reads a 6.0 local fit (the external-callers note below).

- **Every kind and site.** `save_lens`, `save_lens_at` and `load_local_lens`
  write and read every site the lens holds; local names carry the kind
  (`local:s`, `local:k` for input kinds). Every shard is `[d_model, d_model]`:
  v7 records no embedding width, so K/S shards hold `K_L` with
  `d_emb = d_model` (a text lens reads the token embedding in the residual
  basis), and saving a lens with another `d_emb` raises. Shard shapes are
  checked per kind on save and load. A selective load (`requested_layers`,
  `{site: layers}`) reads only the named sites and layers; a site or layer
  the artifact lacks raises `LensNotFittedError` (404), never the soft
  "re-fit" `None`, because the artifact is sound and the request is not.
  The fit lifecycle asks for its `{site: layers}` and adopts a loaded lens as
  resident only when it holds every site and layer the manifest names. `save_lens_checkpoint` takes
  per-site raw sums (`{site: {layer: sum}}`) and `promote_lens_checkpoint`
  per-site layers (`site_layers`); a checkpoint is subsumed only when the final
  artifact covers it at every site. `load_lens` and `load_lens_sidecar` take a
  `side` (`output` by default, `input` for K/S).
- **The policy is checked by equality on every load.** Editing `RELP_RULES`,
  `LENS_CORPUS_PREPROCESS_VERSION` or `LENS_DOC_CHARS` makes every stored lens
  of the affected kinds load as "no lens". Change them only for a genuinely new
  estimator.
- **Loads fail soft.** A schema, header, digest or finiteness failure logs
  "re-fit with `drowse lens fit`" and returns `None`. The caller chooses
  between `LensNotFittedError` and a re-fit. Shards load one at a time, so a
  selective load reads only the requested layers.
- **Pre-5.4 homes** (`models/<id>/jlens/`, v6, `local/default`/`local/relp`)
  are neither read nor auto-migrated. Run `scripts/migrate_lens_v7.py --home
  <dir>` (`--dry-run` first). It renames shards without touching their bytes.
- **Durability.** New generations are fsynced, then the pointer is swapped
  atomically, then the directory is fsynced before old generations are
  collected. Collection is best-effort and deletes nothing when a pointer is
  unreadable. `lens_fit_lock` is cross-process and spans a whole fit or
  removal, on a `.fit` sibling so pointer reads aren't blocked.
- `save_lens` and checkpoint promotion publish **and select** in the lens's
  family slot when it can serve that family (`key.lens_selectable`): both
  families' probes, gates and steering atoms (and the R family's live
  readout) read `out`, so a lens of either family without it is published
  but never displaces, or becomes, its family's selected source. `lens use`
  selects one explicitly for `read` and `lens top` at `mid`. `select=`
  overrides the rule: the fitter passes whether its request held `out`, so
  a mid-only top-up of an artifact that holds `out` selects nothing, as it
  adopts nothing. `save_lens_at(lens, dir, provenance, split_half=None)`
  writes the same layout anywhere without selecting it; it is the packaging
  path.

## Sources and selection (`sources.py`)

`LENS_SOURCES` is a `base.source_registry.SourceFamily` with one active source
per family: the R family (J/R) in the `output` slot, the S family (K/S) in the
`input` slot (`lens_source_side(ref)`). A source is `local:<name>` or a bare
provider name, and carries one or both sites.

| Source | Kind, sites | Repository |
|---|---|---|
| `rs-r` | R: `out`, `mid` | `stdlogits/rs-lenses` `<model>/r-lens/lens.pt` (`out`, the identity at the target layer) and `mid.pt`; Qwen3-8B, Gemma 4 12B |
| `rs-s` | S: `out`, `mid` | the same repository, `<model>/s-lens/out.pt` and `mid.pt` (`K`/`layers`; text only, a `basis` key is refused) |
| `neuronpedia` | J: `out` | `neuronpedia/jacobian-lens`: the paper lens, matched on its config's `hf_model_name`, config sha256 pinned, `_n1000.pt` preferred |
| `workspace-r` | R: `out` | `camilablank/workspace-lenses` `<model>/r-lens/lens.pt`; embedded provenance must name this model and estimator `relp` |
| `workspace-j` | J: `out` | the same repository, `j-lens` arm, estimator `standard`, same forwards as `workspace-r` |

- `fetch_lens_source` is pure IO. It pins the publisher and base-model
  commits, reads the config at that commit (`trust_remote_code=False`), loads
  each `.pt` with `weights_only=True` (memory-mapped), validates dims, layers
  and finiteness, and writes only the binding. The payload stays in the HF
  cache. `activate=True` selects it in its family's slot.
- A layout provider (`_Provider.files`, site → path) binds every site the
  repository holds for `<model stem>` (the id's last part, casefolded), at
  least one. A `LensBinding` is one kind (`binding.kind`) and one file per
  bound site (`binding.sites`, `binding.file(site)`); its schema allows a
  nonempty subset of the provider's keys. rs-lenses files may embed a fit
  record (Qwen3-8B's do): it must name the model, estimator `relp`, and the
  kind and site the file is bound as, and every file of a binding must agree
  on corpus, `t_max`, `skip_first` and `n_prompts`. Files with no record
  (Gemma 4's) bind to the card's corpus (`RS_LENSES_CORPUS`, 128 tokens, skip
  16) with `provenance.embedded = false`.
- `load_external_lens` is offline and loads every bound site into one
  `JacobianLens`. It rechecks the Neuronpedia config digest, `n_prompts` and
  each site's layer roster. **It does not re-hash the checkpoint** against the
  binding's sha256.
- Default order (`lens_source_preference_key`): the R family first (`rs-r`,
  `workspace-r`, `neuronpedia`, `workspace-j`, `local:r`, `local:j`, tagged
  fits), then the S family (`rs-s`, `local:s`, `local:k`, tagged fits).
  Listings and serve's auto-selection (`cli/verbs/serve.py::enable_live_lens`
  for the R family, `select_s_lens` for the S family, which skips sources
  without `out`) use it. `list_lens_sources` rows carry `lens` (kind),
  `family` (`R` or `S`; `output`/`input` is the internal slot name, never
  shown), `sites`, each site's `layers`, `n_prompts` and, where the fit
  recorded one, `split_half` (`{site: {layer: difference}}`; imports lack
  it). `lens_providers()` lists every fetchable provider by family, in
  picker order. Selecting an S-family source over the server checks that
  it loads for these weights (409 otherwise), as the R family's live enable
  does.
- A fit, a promotion or a fetch replaces its family's selection (a fit or
  promotion only when its request holds `out`; a no-op fit that does
  selects too). Removal unselects only an exact
  match. `select_source` evicts that family's resident lens.
- `matches_loaded_model` decides compatibility. A local fit must match the
  loaded-weight fingerprint exactly. An external lens must match the model id,
  the live `config._commit_hash`, `d_model` and the layer range of every site
  it holds, so a model loaded without a Hub commit never matches one.

## Estimator (`transport.py`, `fit.py`)

One estimator, transport, fits every kind at both sites. It is ported from
neuron-lens's hand-written maps and generalized over
`model.blocks.BlockSpec`, so every family the block description covers
(llama, mistral, qwen2/3, gemma 2/3/4 text) fits without per-family code.

- **Per-document linearization.** `transport.linearize(spec, ids, rules=,
  skip_first=)` runs the primal once from the `ModelSpec` (the embedding and
  its scale, every block) and keeps each block's map: the residual norms'
  `gain·rsqrt` (and the normalized input for the true derivative), the q/k/v
  head norms, RoPE tables, attention probabilities over padded keys, the
  softcap's `1 − tanh²` factor, the MLP's activation factors and the
  per-layer scalar. Projection biases enter only the primal. The final norm
  and logit softcap are the readout's, never the lens's (`h_final` is the
  last block's output).
- **The maps.** `jvp(m, X)` pushes `[b, N, d]` tangents through one block
  and returns them at its output and `mid`; `vjp(m, G, mid_seed=)` is the
  exact transpose (the adjoint identity is tested per block). Products run in
  the model's dtype, elementwise state in `promote(dtype, fp32)`. On CUDA the
  maps are compiled with fixed shapes (falling back to eager if compilation
  fails); elsewhere eager.
- **Rules.** `rules_for(kind)`: `relp` (R, S) holds residual norms' scale
  (LN rule), linearizes `act(g)` as `φ(g)·dg` (identity rule) and halves the
  gated product (β = 0.5); `standard` (J, K) takes every derivative true
  (RMSNorm diagonal plus rank one, `act'(g)`, the product rule). Head norms,
  softmax, RoPE and the softcap keep their true derivative under both.
- **Sweeps and seeds.** Output kinds pull `dim_batch` basis cotangents down
  from the last block's output (`reverse_sweep`); `out` layer `L` is read
  below block `L + 1` and `mid` layer `L` inside block `L`. Input kinds push
  basis tangents up from `e = embed(ids)·embed_scale` (`forward_sweep`). One
  sweep covers every layer and both sites, so a document costs one forward
  plus `ceil(d_model / dim_batch)` sweeps; restricting layers only trims
  them (an output fit stops at its lowest layer, an input fit at its
  highest). `dim_batch` defaults to 256.
- **Positions and weighting.** Seeds and reads sit at the valid positions
  `[16, T − 1)` (`SKIP_FIRST_POSITIONS`): attention is causal, so the maps
  carry only those rows and the primal drops the last token. Each document
  contributes its position mean; documents weigh equally at ragged lengths
  and `n_prompts` counts consumed documents (prompts of 17 tokens or fewer
  are dropped first). A transport J/R equals the autograd J/R on the same
  corpus, so the estimator policy is unchanged and earlier local fits stay
  valid.
- **Split-half.** `fit_lens` keeps the total sums and the odd-indexed
  documents' (`LensSums.total`/`.odd`, fp32 on the CPU: one more lens of
  RAM) and returns `LensFit(lens, split_half)`, per site and layer
  `‖mean_even − mean_odd‖ / ‖mean‖`. Parity follows the corpus index, so a
  resume continues it exactly; `split_half=False` drops it.
- **Verification and refusal.** `fit_lens` verifies the model once
  (`verify_model_spec`, an fp32 replay of every block) unless handed a
  verified `spec`; the lifecycle verifies at the start of `LensFitter.run`,
  before any corpus work, so MoE, fused gate-up or qkv, hybrid linear
  attention and any undescribed term refuse with `UnsupportedBlockError`
  (422). Quantized weights refuse too. External lenses for those models still
  load.
- **Memory and MPS.** A sweep stages its reads on the device and moves them
  to the host in one transfer per pass, which also drains Metal's queue. An
  all-zero row from an accelerator is an asynchronous OOM (Metal zeroes lost
  work): it raises an out-of-memory error, and the OOM ladder halves
  `dim_batch` (kept down for the rest of the fit) and retries the pass.
  `empty_cache` runs at checkpoint cadence; teardown drains, collects and
  flushes the allocator.
- **Cancellation** is checked between passes and documents. A stop between
  documents checkpoints the completed prefix; one inside a document abandons
  its partial sums (the lifecycle reloads from the last checkpoint).
- `initial=LensSums.from_means(lens, odd)` takes ownership of an averaged
  base and scales it to sums in place, so don't reuse the passed lens.
- The autograd estimator lives on as the test oracle
  (`tests/support/lens_oracle.py`, with the RelP rules as derivative patches).

## RelP rules (`relp.py`)

The R- and S-lenses are the J- and K-lenses taken under an LRP-modified
derivative (RelP, arXiv:2508.21258); forwards are unchanged. The R-lens
reads late layers faithfully and early ones better than J, the S-lens is the
faithful read early. The rules: **LN rule** (residual RMSNorms, sandwich
norms included, hold `rsqrt` constant: the diagonal `gain·rsqrt(mean(x²)+eps)`),
**identity rule** (`φ(g) = act(g)/g` replaces `act'(g)`), **half rule**
(`act(gate)·up` passes β = 0.5 per factor). Attention-internal norms keep
their true derivative (`include_qk_norms: false`). `RELP_RULES` records them
in every RelP policy; `transport.py` applies them.

## Corpora (`corpus.py`)

The default is `HuggingFaceFW/fineweb-edu:sample-10BT`, which needs the
optional `datasets` package (`pip install 'drowse.ai[hf]'`). A source is
written `REPO[:CONFIG][@REV][,text_field=F,files=K,seed=S,weight=W]`; several
sources form a weighted mix. Documents are sliced to `LENS_DOC_CHARS = 4000`.
A corpus is **pinned to a commit and named config before it is read**, and its
`spec` is the recorded provenance. The first N documents of a longer read
equal a shorter read's, which is what makes token-prefix resume sound. Pinning
and streaming run in a spawned worker: cancellation can abandon a blocked Hub
call, pyarrow's I/O pool can't hang interpreter exit, and spawn is safe beside
MPS. An empty file corpus raises `EmptyCorpusError` (400).

## Fit lifecycle (`lifecycle.py`)

`plan_lens_fit` decides from metadata alone, site by site: a request is
`{site: layers}` (`resolve_fit_layers(selector, kind=, n_layers=, sites=)`:
each site draws from its own pool, `sample` is refused, an explicit layer no
requested site holds is refused). The identity it compares is the token-id
rows sha256, `seq_len`, the weight fingerprint and the policy.

- **`noop`**: the durable artifact covers the corpus and the requested layers.
  A leftover checkpoint is reaped only if the verified final artifact
  subsumes it.
- **`topup`**: fit only the missing `{site: layers}` and union them in.
  Unchanged shards carry forward after a re-hash, or without one under a
  load proof; the kept sites' split-half values carry forward and the new
  layers are measured.
- **`resume`**: continue from the durable artifact or checkpoint over a
  corpus prefix (`consumed_prefix_sha256`), whichever is farther ahead.
- **`fresh`**: nothing reusable, or `force` (never over a larger or wider
  artifact without it, below).

Extending the corpus of an artifact that holds layers the request omits is
refused, since one artifact has one corpus identity. Pass the full layer set or
`force`. A fit over another corpus publishes over the artifact, so it is
refused too when the artifact (same weights, `seq_len` and policy) holds
more documents than the request, or a site or layer it omits: the default
dropped from 100 documents and `out` only to 64 at both sites, and a re-run
with defaults would otherwise discard an hours-long fit. Ask for at least its
documents of the same corpus (which tops up the missing site), or pass
`force`. Both refusals raise `LensReplaceRefused` (a `JacobianLensError`), so
a caller can offer the forced fit for them alone. When a payload fails to
load, the executor drops it and re-plans.

A fit drives its own family's slot (`LensFitter(residency={"output": …,
"input": …})`; the instrument's `_InputResidency` is the S family's): an R/J
fit evicts and adopts the R resident and its live readout, an S/K fit the S
resident, its `slens/` directions and probe layers, and neither touches the
other family. Selection follows the same request: a fit holding `out`
selects `local:<kind>` before adopting (a no-op included, so the resident is
stamped with its own label and a refresh keeps it), and a fit without the
`out` site selects and adopts nothing, even when it tops up an artifact that
holds `out`: its family's resident lens, live readout and selection stay as
they were.

`prove_lens_fit_noop` proves a no-op without the model. A file corpus is
identified by its raw digest, a pinned Hub corpus by spec plus prompt count
(before streaming). The source fingerprint is computed last and the payload
digests are verified. The CLI preflight and `fit_is_noop` (the server) both
call it. A fit evicts its family's resident lens before the estimator owns
the base and restores it on failure. A complete terminal checkpoint is promoted, not
rewritten.

## Residency and the generation protocol

`LensRuntime` holds the resident lens, its identity (`lens_identity`, the
sha256 of the canonical manifest or binding descriptor) and `BoundedLRU`
caches (two device transport stacks, decoded tokens, depth and selector
tensors). `LensInstrument` takes every dependency explicitly: model, capture
(`LatestSlices`), `ExclusiveGuard`, `capture_hidden`, the registry
(`ProfileStore`), `on_change`, `resolve_profile`, `assert_unsteered`,
`TokenReplay`, the whitener getter and `neutral_means` (site → `{layer:
mean}`, the residual centring means; raises when a site has none).

- **`state_lock` is the lens-state boundary.** It is a reentrant leaf lock
  over refresh, adopt, evict, registry and live mutations, and `prepare`'s
  snapshot. Take the model-exclusive guard first, never while holding it.
- **Two families resident together.** `runtime` holds the R family's lens,
  `input_runtime` the S family's. `resident()` / `resident_input()` reconcile
  each with disk on every call: they evict a removed, replaced or incompatible
  artifact and adopt a new one; a bound run's pin (`run.lens` /
  `run.input_lens`) short-circuits them. `adopt` dispatches on the lens kind
  (an input kind goes to `adopt_input`). `adopt` and `evict` rebuild the
  `jlens/` probes' layers, the live stack and the stash in one transaction and
  evict every `jlens/` profile; `adopt_input` and `evict_input` do the same
  for `slens/` probes, profiles and the input stash. A probe spec carries
  `side` (`output` for `jlens/`, `input` for `slens/`; absent reads as
  `output`). A source without `out` leaves its family's probes and atoms
  without layers rather than reading `mid`.
- **Per generation:** `prepare` → `plan` → `bind` → `close_run`. `prepare`
  refuses while a run is bound, because a stale pin would suppress its
  refresh. Its `LensPrep` is the atomic snapshot (lens, specs, live state,
  fingerprint, and the S family's `input_lens`/`input_pinned`). `plan` and
  `bind` read only the prep, never the registry: a concurrent adoption pairs
  lens A with lens B's layers, which fails with a `KeyError` in the transport
  stack. A run pins the R family when live readout is on or when `jlens/`
  probes feed a final aggregate or a gate, and the S family when `slens/`
  probes feed a final aggregate, a gate or the live readout's per-step
  readings. With live readout on, `plan` also captures the `slens/` probes'
  band (their S lens's `out` layers), not only the R live layers. A source
  without `out` pins with its probes layerless; nothing reads its
  `source_layers`.
- `fit`, `readout` and `attach` run under the exclusive guard. `fit` takes
  `lens_fit_lock` first and calls `assert_unsteered`. `fit(prompts, kind=,
  sites=, layers=, …)` defaults to `kind="J"` and both sites; the CLI and the
  server default to `r`.

## Readout math

`LogitReadout(transport_rows(J, h))` reuses the model's own final path (norm,
OPT projection, logit scale, soft-cap). One unit runs through every surface:
the per-layer softmax probability `p_l(v)`.

- `strength = mean_l p_l(v)` over fitted layers, in [0, 1], with uniform
  weights (softmax already lets a confident layer dominate). `com`/`spread`
  are depth moments weighted by the same `p_l`, where depth = `layer /
  (n_layers − 1)`. Readouts are sharp (median per-layer max ≈ 0.8 on
  gemma-3-4b) and depth changes *which* token leads, so a token's profile over
  depth is its depth signal.
- Aggregate top-k ranks full-vocabulary strength. A union of per-layer top-ks
  would miss a token that is mid-pack everywhere.
- Gate and probe reads take exact columns against the full `logsumexp`, with
  no `[layers, vocab]` probability matrix. Results leave the device in one
  packed fp32 transfer (MPS has no float64). CUDA transport uses scoped TF32.
  Offline readouts chunk positions under `READOUT_CHUNK_BYTES` (256 MiB).

## Atoms, probes, gates

- **Reserved namespaces** (`naming.RESERVED_NAMESPACES`: `jlens`, `slens`,
  `sae`, `neuron`, `head`) name instrument units. Names stay verbatim and
  case-sensitive (`jlens/Paris` ≠ `jlens/paris`), take no `:variant`, skip the
  manifold tiers, and no manifold may be authored or installed under them.
- **Steering atom.** `jlens/<word>` or `jlens/#<id>` resolves lazily through
  the registry's `jlens` atom resolver into `register_direction`. That yields
  `W_U[v] @ J_l` on every fitted layer (`W_U P J_l` under an OPT projection),
  registered as an ordinary profile that pushes, ablates and composes like
  any term. A word must be one token: the leading-space piece is tried first,
  and the match must survive a decode round trip (else `MultiTokenWordError`
  lists the pieces). `#<id>` is bounds-checked against the unembedding
  (`InvalidTokenIdError`). Lens atoms run hot. On gemma-3-4b α ≈ 0.3 was the
  coherent sweet spot and α ≥ 0.5 over-steered into repetition, because the
  atom is one sharp token direction rather than a distributed contrast.
- **`slens/` atom.** `slens/<word>` or `slens/#<id>` resolves through the
  registry's `slens` resolver into `register_input_direction`: the footprint
  `K_l ẽ_v` at the S lens's `out` site on every layer it holds (steering
  injects at block outputs), an ordinary profile that pushes, ablates, takes a
  `{…}` layer clause and goes through the whitener exactly as a `jlens/` atom
  does. `jlens/` steers toward saying a token, `slens/` toward having read it;
  they are different directions. No S lens is `LensNotFittedError` (404, names
  `rs-s`); a mid-only S lens is the same error naming the missing `out` site.
  `#<id>` is bounds-checked against the input embedding, and a token the
  footprint mask drops (special, added, byte-fallback, padding) is
  `InvalidTokenIdError` for atoms and probes alike. Resolution runs in the
  model-exclusive section (the first S use builds the embedding statistics);
  the footprints are CPU fp32 against the lens's CPU matrices.
- **Probe.** `add_probe("jlens/…")` routes through `readout/roster.py` to
  `attach`, never the Monitor: no whitener, no fold. It yields a
  `ScalarReading` (unit `mean_token_probability`) with a per-layer `p_l` trace
  and a depth summary (basis `readout_probability_mass`). `probe_hash` folds
  in the lens identity, so the same word under another source or a re-fit is
  a different probe. The final aggregate pools the same last-content-token
  slice as the Monitor and SAE.
- **`slens/` probe.** `add_probe("slens/…")` routes to the same `attach`
  (`side: input`) and builds the embedding statistics and the neutral `out`
  means (no means: `NeutralMeansError`, nothing attached). Per layer `z_L` of
  the token, read from the residual centred on the neutral block-output mean
  (`u_L = K_Lᵀ (h_L − μ_L)`: one batched pullback over the S lens's `out`
  layers against a cached `[L, d]` mean stack, then `TokenEmbeddings.z_of`, one
  host transfer; `prepare` makes the means resident before the decode loop, so
  a step never builds them); the `ScalarReading` is the
  depth rule's weighted mean `Σ_L w_S(L) z_L / Σ_L w_S(L)` (unit
  `mean_footprint_z`; `w_S` from `depth.kind_weights` under the instrument's
  `band`, so a probe reads where S is the faithful lens and is 0 when no read
  layer has S weight), `per_layer` every `z_L`, and a depth summary weighted
  by `w_S(L) max(z_L, 0)` (basis `footprint_z`). Layers come from the lens
  the reading uses, never the registry. `probe_hash` folds in the S
  lens identity and a digest of the neutral out means at the probe's layers
  (`slens-footprint-z-v2`: the reading is of a centred residual, so other
  means are another probe). It joins the final aggregate, the
  gate path and, with live readout on, the per-step readings.
- **Gate.** `@when:jlens/<word> > x` (also `[0]` or `#<id>`) fires only for an
  attached probe. The strength channel is the only one
  (`_GATE_CHANNELS = (Axis,)`); `:fraction`, `:membership`, `@label` and
  `~label` raise `UnsupportedProbeChannelError` at preflight. Gate scalars are
  computed post-forward and stashed by `step_id`, and the display step reuses
  them only from the same forward. A lens gate forces per-step lens reads
  whatever the live toggle says, but never per-token Monitor scoring.
  `@when:slens/<word> > x` works the same way on the footprint z; its readings
  are stashed by `step_id` on the run (`input_stash`, which records the probes
  read on that forward), so the display step reads only the probes the gate
  did not, and with live readout on the gate reads them all. An `slens/`-only
  gate never resolves the R family. Both read paths build each family's
  probe rows on the device first (`_OutputBlock` from
  `jlens.token_readout_rows`, `_InputBlock`) and transfer once: a gate
  callback reading `jlens/` and `slens/` probes packs both blocks into one
  host copy (`_host_readings`), and the live step folds any `slens/` block it
  reads into its own packed transfer.

## Live readout, replay, decomposition

- `set_live(True, layers=…)` (all fitted layers by default) puts the `J_l`
  stack on the device and attaches **no hooks**. `live_readout_step` reads
  the capture's latest slices post-forward at the token tap, so the steering
  fast path and compile eligibility are untouched. It reuses same-forward gate
  rows and syncs to the host once. The library default is off;
  `authored_capture` does the same for authored-prefill rows.
- `token_readout(node_id, raw_index, top_k=None, …)` replays the producing
  forward through `TokenReplay` (`apply_steering=False` reads the unsteered
  counterfactual). It returns the `scope="replay"` measurement envelope.
  `top_k=None` means 8; valid values are 1 to 256.
- `decompose(selector, k=16, dictionary="tokens-out")` is greedy pursuit
  against a dictionary never materialized, with chunked and cached atom norms:
  `TokensOut` (`W_U J_l`, the R family) or `TokensIn` (`K_l ẽ_v` at the S
  lens's `out` site, masked tokens scoring `-inf`). It is Mahalanobis like
  every activation-space metric: a dictionary is built over the session's
  `LayerWhitener` at its layer and holds its matrix whitened once (`J_l W`,
  `W K_l`, `W = Σ^{-1/2}` from `LayerWhitener.whiten`), `coords` maps the
  target there, so selection, the NNLS fit and `share` are all whitened. A
  whitener that is missing or misses a layer raises `WhitenerError`
  (`runtime.require_decompose_whitener`); the instrument takes the whitener
  getter by injection (`whitener=`, the session's lazy neutral state). The
  whitened norms are cached per lens identity and layer beside the whitener
  they were taken in (`runtime.cached_norms`), so a rebuilt whitener
  recomputes them. Selection is norm-normalized. It re-solves NNLS but takes
  the exact unconstrained solve when that is already nonnegative, and tiny
  solves run on CPU. `share` is the Mahalanobis variance fraction the atoms
  carry. `neurons` (`NeuronWrites`, `decompose_neurons(directions, …)` for raw
  vectors) needs no lens: block `l`'s neuron writes (`model.units.neuron_writes`,
  Gemma's post-feedforward gain included), whitened, signed (selection by
  `|score|`, an unconstrained solve; a negative coefficient is the `−` pole),
  named `neuron/L<l>n<i>`. `unit find` adds poles and catalog labels.
  `token_embeddings()` is the public face of the S read's masked embedding
  statistics (the unit reads use it).

## Errors, seams, tests

- Errors: `JacobianLensError` (422), `LensNotFittedError` (404),
  `JacobianLensCancelled` (409), `LensReplaceRefused` (422, in `lifecycle.py`), `UnsupportedBlockError` (422, in
  `errors.py`, from the block description). `MultiTokenWordError`, `InvalidTokenIdError`,
  `EmptyCorpusError` and `SourceSpecError` are `RequestError`s (400). `HFError`
  (502) means an unresolvable repo commit.
- **External callers** make these names and the v7 layout an interface.
  neuron-lens calls `load_local_lens`, `save_lens`, `LensProvenance`,
  `JacobianLens(key=)` and `RELP_RULES` (its tests also call the removed
  `fit_jacobian_lens`). drowse.ai's hosted fit
  (`browser-runtime/fit_hosted_jlens.py`) still imports the removed
  `fit_jacobian_lens` and calls `save_lens_at`; its browser reader accepts
  v7 packs only, and only J/R with a single `sites.out` and exactly the
  required site keys. **Pending drowse.ai follow-up** (before it advances its
  submodule pin past this fit): move the hosted fit to `fit_lens(kind=,
  sites=("out",))`; make the reader accept the optional `split_half` (and
  refuse `odd` outside a checkpoint) per the v7 reader rule above, and
  validate or skip `mid`. Until then, keep calling `save_lens_at` without
  `split_half`.
  That out-only shape is unchanged from 5.4
  (`test_save_lens_at_keeps_an_out_only_pack_in_the_browser_readers_shape`),
  so v7 is not bumped: its new fields are optional, and a bump would
  invalidate every 5.4 local fit and reject drowse.ai's out-only packs too.
- Tests: CPU tests in `tests/lens/` (one file per module plus
  `test_migrate_lens_v7.py`; `test_lens_sites.py` pins kinds × sites in the
  object and the store, `test_lens_depth.py` the depth rule,
  `test_lens_rs_sources.py` the rs-lenses bindings over a synthetic Hub,
  `test_lens_footprint.py` footprint z, the mask, `read_vectors` and
  `TokensIn`, `test_slens.py` the S family through a tiny session with a
  hand-built S lens) run on the toy model in `tests/support/jlens_toys.py`, a
  genuine tiny llama the transport fits. `test_lens_transport.py` holds every
  dense family × kind × site to the autograd oracle
  (`tests/support/lens_oracle.py`), every block map to the adjoint identity,
  and (when `~/Work/neuron-lens` is checked out) agreement with neuron-lens's
  transport; `test_fit.py` covers `fit_lens`, `test_fit_split_half.py` the
  split-half records and per-site lifecycle. They use `tests/support/lens.py`
  (`LensHarness`, `lens_instrument(**kw)`, `lens_kwargs(**kw)`,
  `FakeCapture`), which builds a real `LensInstrument` with session-shaped
  dependencies. Generation-boundary tests use `tiny_session`.
  `tests/gpu/test_jlens_gpu.py` micro-fits a real model (SmolLM2-360M or
  `DROWSE_TEST_MODEL`).
