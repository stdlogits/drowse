# drowse/lens

The Jacobian lens family (Gurnee et al., "Verbalizable Representations Form a
Global Workspace in Language Models", 2026), a per-model artifact: one fp32
matrix per layer at each site it holds. The **R family** (kinds J, R: output
Jacobians) reads what a vector is disposed to *say*, through `W_U`; the
**S family** (K, S: input Jacobians) reads what it has *read*, against the
input embedding. The session face is `session.lens` (`LensInstrument`), one
resident lens per family. Layer 7 beside `sae` and `units`; runtime imports
are `errors`, `base`, `model` and `readout` (the whitener arrives by
injection; `geometry.whitener` and `units.catalog` are type-only).

| Module | Owns |
|---|---|
| `key.py` | `LensKey` (kind × site), `site_layers`, `lens_selectable`, `FAMILY_DIRECTION`, local names, `DEFAULT_FIT_DOCUMENTS` (64). Torch-free |
| `jlens.py` | `JacobianLens`, readout math, token-atom resolution, errors, fit defaults (`SKIP_FIRST_POSITIONS` 16, seq_len 128, `dim_batch` 256, `CHECKPOINT_SECONDS` 600) |
| `depth.py` | `LensBand`, `kind_weights`, `s_weights`. Torch-free |
| `read.py` | `plan_reads`/`PendingReads`, `read_vectors`, `output_transport`, `LensRead`/`FamilyRead`, `ReadOf` |
| `footprint.py`, `token_counts.py` | the S read: `token_mask`, `TokenEmbeddings`, footprint z; the model's corpus token counts |
| `decompose.py` | `TokensOut`, `TokensIn`, `NeuronWrites`, `sparse_decompose` |
| `relp.py`, `transport.py`, `fit.py` | the estimator: `RELP_RULES`; per-document block maps; `fit_lens`, `LensSums`. No runtime path imports them |
| `corpus.py` | pinned Hub corpora streamed in a spawned worker, `read_corpus_file` (`sae train` uses it too) |
| `lifecycle.py` | `plan_lens_fit` (pure), `LensFitter`, `prove_lens_fit_noop`, `LensReplaceRefused` |
| `store.py` | format-v7 local artifacts, `LensProvenance`, `lens_estimator_policy`, `lens_identity` |
| `sources.py` | `LENS_SOURCES`, provider bindings and fetch, source order |
| `runtime.py` | `LensRuntime` (resident lens, identity, caches, readout primitives), `matches_loaded_model` |
| `instrument.py` | `LensInstrument` (residency, fit, reads, probes, gates, live, replay, decompose), `LensRun` |

## Kinds and sites

| Kind | Family | Rules | Matrix | Read |
|---|---|---|---|---|
| J / R | R | standard / RelP | `J_L = E[Σ_{t'≥t} ∂h_final(t')/∂h_L(t)]`, `[d, d]` | `transport(h) = h @ J_Lᵀ`, into the final basis |
| K / S | S | standard / RelP | `K_L = E[Σ_{s≤t} ∂h_L(t)/∂e(s)]`, `[d, d_emb]` | `pullback(c) = c @ K_L`; `footprint(ẽ) = K_L ẽ` |

A **site** is `out` (block output) or `mid` (post-attention residual, the MLP
input); `LensKey(kind, site)` keys every store, binding and selection path. The
R family's `out` stops below the last block, its target: there `J = I` exactly
under both rule sets, so reads at that layer are the logit lens, labelled
`logit`, and go through `read.py::output_transport` (reads, `read_prompt`'s
default layers, unit population statistics, the anatomy's cotangent). Every
other site covers every block.

A `JacobianLens` is one kind at one or both sites (positional `jacobians` is
`out`, `mid=` is `mid`; read with `matrices(site)`, `layers(site)`,
`matrix(layer, site)`). A read at a missing site raises `LensNotFittedError`
(404) and never falls back to the other site, whose layers differ by design.
The default key is `LensKey("J")`: a hand-built R lens must pass
`key=LensKey("R")`. A wrong-orientation read is `JacobianLensError` (422). v7
stores input kinds with `d_emb = d_model`. Local directory names carry the kind
(`^[jrks](-tag)?$`) and writers refuse a mismatch, so a fit can never land in
or resume from another kind's shards; fits write the untagged `local:<kind>`.

## Reading a place (`depth.py`, `read.py`, `footprint.py`)

Site follows where the vector sits (`out`: residuals at a block output, neuron
writes, a head's inputs at `L − 1`; `mid`: neuron read directions, head
writes). Family follows depth: S is faithful early, R late, weighted inside
`LensBand(lo=0.15, hi=0.35)` (fractions of `layer / (n_layers − 1)`, a prior
until measured per model); `kind_weights` gives `(w_S, w_R)` summing to 1.

`read(vector, layer, site, *, of, k=8)` takes `[d]` or `[n, d]`. `of` has no
default: a `"residual"`'s S half reads `h − μ` (`μ` the neutral mean at that
site: the whitener's centring mean at `out`, the neutral cache's mid means at
`mid`), because the raw residual's common component otherwise swamps the
position's own tokens; a `"direction"` and every R half read as given. No mean
at the site is `NeutralMeansError` (422), never an uncentred read. Each
`LensRead` has `r` and `s` halves (`weight`, `unit`, `lens` label like
`R:mid`, `centred`, top-k `tokens`, or `absent` with the reason); weights stay
raw so a consumer fades a light half and flags a missing heavy one.
`plan_reads` computes a batch on device and leaves in one transfer, packable
into a caller's own (token anatomy does).

**Footprint z.** `u_L = K_Lᵀ h`, `z_L(v) = u_L·ẽ_v / sqrt(u_Lᵀ Σ_E u_L)`, `ẽ_v`
the centred, scaled embedding (Gemma's `√d`), `Σ_E` its covariance over the
token mask: the vocabulary z-score of the raw score (same top-k as
neuron-lens's `read_S`), costing `O(d²)` per layer for one token instead of a
vocabulary pass. The **token mask** drops special and added tokens,
byte-fallback pieces (`<0xNN>`, or decoding alone to U+FFFD), embedding padding
rows, and tokens too rare in text (undertrained embeddings top early S reads).
Counts come from a bound neuron-lens catalog (`unit fetch`, kept at 5), else
the model's corpus counts, else none (tokenizer rules alone); `token_mask_info()`
always names the source. Corpus counts (`drowse lens counts`) live with the
model, never in a lens pack: `models/<id>/tokens/counts.safetensors` plus
`counts.json` (format 1: corpus, documents, tokens, `min_count`, vocabulary
size, `tokenizer` fingerprint), streamed from the S lens's `corpus_spec`
through the tokenizer alone, refused under 1M tokens, thresholded at
neuron-lens's rate (`count_threshold`). Another format, model or vocabulary
size is not read; another tokenizer falls back with a warning.

`TokenEmbeddings` is built on first S use (model work, inside the exclusive
section) and rebuilt when the counts change: `_mask_changed` evicts every
`slens/` profile (they re-resolve on next use), detaches `slens/` probes whose
token the new mask drops, and has the roster forget every `slens/` digest. The
mask and the centring are drowse.ai seam semantics (its S pack carries the
mask, `mean_E` and `Σ_E`).

## On disk (format v7)

```
models/<safe_model_id>/lens/
  active.json            {format_version: 2, model_id, output, input}: one source per family
  bindings/<name>.json   commit-pinned provider binding (format 2)
  local/<kind>[-<tag>]/
    manifest.json        atomic pointer; checkpoint.json only during a resumable fit
    <site>[.odd][.partial].layer-<L>.gen-<id>.safetensors   immutable fp32 [d, d] shards
```

`manifest.json`: `format_version` (7), `kind`, `sites.<site>.{layers,
tensor_files}`, `n_prompts`, `d_model`, `dtype` (`float32`), `checkpoint`,
`consumed_prefix_sha256` (checkpoints only), the `LensProvenance` fields
(`corpus_spec`, `corpus_sha256`, `corpus_hash_kind` = `token_ids_v1`,
`seq_len`, `dim_batch`, `raw_corpus_sha256`, raw/usable prompt counts,
`model_layer_count`, `model_fingerprint`, `model_source_fingerprint`) and
`estimator_policy` (`{direction, rules, skip_first_positions,
corpus_preprocess_version, doc_chars}`, `rules = RELP_RULES` for RelP kinds).
Optional per site: `split_half` (`{"<layer>": ‖mean_even − mean_odd‖/‖mean‖}`,
local fits only) and, read only from `checkpoint.json`, `odd` (the odd-half
shards a resume continues its split-half from).

- **Tolerant reader.** The manifest reader requires the keys it uses and
  ignores the rest (6.1 manifests' per-shard `tensor_sha256` included). Shards
  are checked for header (one `layer_<L>` key, shape, F32) when the pointer is
  read and for finiteness when materialized; nothing is hashed. Corpus digests
  identify the corpus for staleness and prefix resume.
- **`lens_identity`** is the sha256 of the canonical manifest (or a binding's
  descriptor), so any republication changes it. The resident lens, lens probe
  digests, cached atom norms and drowse.ai's packs bind it.
- **The policy is compared by equality on load.** Editing `RELP_RULES`,
  `LENS_CORPUS_PREPROCESS_VERSION` or `LENS_DOC_CHARS` makes every stored lens
  of the affected kinds read as "no lens"; do it only for a new estimator.
- **Loads fail soft**: a bad manifest or shard logs "re-fit" and returns
  `None`, and the caller picks `LensNotFittedError` or a re-fit. A selective
  load naming a site or layer the artifact lacks raises `LensNotFittedError`
  instead (the artifact is sound, the request is not).
- **Durability**: generations fsynced → pointer swapped atomically → directory
  fsynced → unreferenced generations collected (nothing while a pointer is
  unreadable). `lens_fit_lock` is cross-process, spans a whole fit, and locks a
  `.fit` sibling so pointer reads are not blocked.
- **Publishing selects** in the lens's family slot only when it holds `out`
  (`lens_selectable`): probes, gates, atoms and the live readout read `out`, so
  a mid-only lens never displaces a readable source. `save_lens_at(lens, dir,
  provenance, split_half=None)` writes the layout anywhere without selecting.
- Pre-5.4 homes need `scripts/migrate_lens_v7.py`.

## Sources (`sources.py`)

`LENS_SOURCES` (a `base.source_registry.SourceFamily`) holds one active source
per family; `active.json` keys families by Jacobian direction (`output`,
`input`), a spelling frozen there and in `estimator_policy.direction` only.

| Source | Kind: sites | Repository, files under `<stem>/` (model id's last part, casefolded) |
|---|---|---|
| `rs-r` | R: `out`, `mid` | `stdlogits/rs-lenses` `r-lens/{lens,mid}.pt` (Qwen3-8B, Gemma 4 12B) |
| `rs-s` | S: `out`, `mid` | same repo, `s-lens/{out,mid}.pt` |
| `workspace-r` | R: `out` | `camilablank/workspace-lenses` `r-lens/lens.pt`, estimator `relp` |
| `workspace-j` | J: `out` | same repo, `j-lens/lens.pt`, estimator `standard`, same forwards |
| `neuronpedia` | J: `out` | `neuronpedia/jacobian-lens`, matched on its config's `hf_model_name`; `_n1000.pt` preferred |

`fetch_lens_source` is pure IO: it pins publisher and base-model commits,
loads each `.pt` with `weights_only=True, mmap=True`, validates dims, layers
and finiteness, and writes only the binding (payloads stay in the HF cache).
A provider file's key set is checked exactly (optional `provenance` aside): an
extra key such as a `basis` means the matrices are in another basis, so it is
refused rather than read wrong. rs-lenses embedded records must name the
model, estimator `relp`, and the kind and site bound; a binding's files must
agree on corpus, `t_max`, `skip_first` and `n_prompts`; files without a record
(Gemma 4's) bind to `RS_LENSES_CORPUS` (128 tokens, skip 16) with
`embedded = false`.

A binding (`{format_version, kind, provider, name, model_id, model_revision,
repo_id, repo_revision, files: {"R:out": {path, sha256, layers}}, d_model,
n_prompts, corpus, seq_len, skip_first, provenance}`) is read tolerantly. A
file's `sha256` is taken at fetch as identity (it enters `lens_identity`);
`load_external_lens` (offline) does not re-hash payloads but checks
`n_prompts` and each site's layers against the binding, and for Neuronpedia
that the cached config still hashes to `config_sha256` (the config is what
matched the lens to the model).

- Order (`lens_source_preference_key`, serve's auto-selection, listings): R
  family `rs-r`, `workspace-r`, `neuronpedia`, `workspace-j`, `local:r`,
  `local:j`, tagged; then S `rs-s`, `local:s`, `local:k`, tagged.
- A fit holding `out`, a promotion or a fetch replaces its family's selection;
  removal unselects only an exact match; `select_source` evicts that family's
  resident.
- `matches_loaded_model`: a local fit must match the loaded-weight
  fingerprint; an external lens the model id, live `config._commit_hash`,
  `d_model` and every held site's layer range (a model loaded without a Hub
  commit never matches one).

## Estimator (`transport.py`, `fit.py`, `relp.py`)

One estimator, transport (ported from neuron-lens, generalized over
`model.blocks.BlockSpec`), fits every kind at either or both sites with no
per-family code and no autograd (the autograd estimator is the test oracle,
`tests/support/lens_oracle.py`). `linearize` runs the primal once per document
and keeps each block's map; `jvp`/`vjp` move tangents up or cotangents down one
block, returning both sites. The final norm and softcap belong to the readout.

- **RelP rules** (R, S; arXiv:2508.21258; forwards unchanged): residual
  RMSNorms (sandwich norms included) hold `rsqrt` constant; `act(g) = g·φ(g)`
  linearizes as `φ(g)·dg`; `act(gate)·up` passes β = 0.5 per factor. Norms
  inside attention, softmax, RoPE and the softcap stay true.
- **Cost.** Output kinds pull `dim_batch` basis cotangents down from the last
  block, input kinds push tangents up from `embed(ids)·scale`; one sweep covers
  every layer and both sites, so a document is one primal plus
  `ceil(d_model / dim_batch)` passes. Products in the model dtype, elementwise
  state in `promote(dtype, fp32)`; CUDA compiles the maps with fixed shapes.
- **Positions** `[16, T − 1)` (causality lets the maps carry only those rows);
  each document contributes its position mean, so documents weigh equally.
  Prompts of 17 tokens or fewer are dropped up front, so `n_prompts` counts
  consumed documents exactly, which makes prefix resume sound.
- **Split-half**: odd-indexed sums kept beside the total (one more lens of CPU
  RAM); parity follows the corpus index, so a resume continues it.
- **Refusal.** `verify_model_spec` replays every block in fp32 before corpus
  work: MoE, fused gate-up or qkv, hybrid linear attention, quantized weights
  or any undescribed term raise `UnsupportedBlockError` (422). External lenses
  for those models still load.
- **MPS.** Metal reports queue exhaustion asynchronously and zeroes lost work;
  no lens row is exactly zero, so an all-zero row raises out-of-memory and the
  ladder halves `dim_batch` for the rest of the fit.
- **Checkpoints** every `checkpoint_every` (25) documents and at the first
  document boundary 600 s after the last. `cancel_event` is prompt (a stop
  inside a document abandons its partial sums); `stop_event` is graceful (the
  document in flight finishes and is checkpointed). The CLI sets `stop_event`
  on the first signal, `cancel_event` on the second.
- `LensSums.from_means(lens, odd)` scales the base to sums in place; don't
  reuse the lens passed.

**Corpora.** Default `HuggingFaceFW/fineweb-edu:sample-10BT` (needs `[hf]`);
spec `REPO[:CONFIG][@REV][,text_field=F,files=K,seed=S,weight=W]`, several
sources a weighted mix; documents sliced to 4000 characters. A corpus is pinned
to a commit and config before it is read, and the first N documents of a longer
read equal a shorter read's. Pinning and streaming run in a spawned worker (a
blocked Hub call stays cancellable, pyarrow can't hang exit, spawn is safe
beside MPS).

## Fit lifecycle (`lifecycle.py`)

`plan_lens_fit` decides from metadata alone. A request is `{site: layers}`
(`resolve_fit_layers`: each site draws from its own pool; `sample` and a layer
no requested site holds are refused). Identity: token-id rows sha256 (or its
prefix), `seq_len`, weight fingerprint, policy.

| Plan | When |
|---|---|
| `noop` | the durable artifact covers the corpus and every requested layer; a crash-left checkpoint is reaped only if the final artifact subsumes it |
| `topup` | fit only the missing `{site: layers}` and union them in; kept shards carry forward by file name |
| `resume` | continue from the durable artifact or a checkpoint over a corpus prefix (`consumed_prefix_sha256`), whichever is farther ahead |
| `fresh` | nothing reusable, or `force` |

`LensReplaceRefused` (422) refuses a fit that would publish over the artifact
and lose part of it: extending the corpus while omitting layers it holds (one
artifact has one corpus identity), or fitting another corpus when the artifact
holds more documents or a site or layer the request omits. `force` replaces. A
payload that fails to load is dropped and the plan redone.

A fit drives only its own family's residency: it evicts the resident before
the estimator owns the base, restores it on failure, and adopts the result. A
fit holding `out` selects `local:<kind>` before adopting (a no-op too); one
without `out` selects and adopts nothing. A complete terminal checkpoint is
promoted, not rewritten. `prove_lens_fit_noop` proves a no-op without the
model (file corpus by raw digest, Hub corpus by spec plus count, source
fingerprint last) for the CLI preflight and the server's `fit_is_noop`.

## Residency and the generation protocol

`runtime` (R) and `input_runtime` (S) are `LensRuntime`s: resident lens,
identity, `BoundedLRU` caches. `resident()`/`resident_input()` reconcile with
disk on every call (evict a removed, replaced or incompatible artifact; adopt a
new one) unless a bound run pins them. `adopt`/`evict` rebuild that family's
probe layers, live stack and stash in one transaction and evict its atom
profiles. A source without `out` leaves its family's probes and atoms
layerless rather than reading `mid`.

- `state_lock` is a reentrant leaf lock over residency, registry and live
  mutations: take the model-exclusive guard first, never inside it.
- **Per generation** `prepare → plan → bind → close_run`. `prepare` refuses
  while a run is bound (a stale pin would suppress its refresh) and returns the
  atomic snapshot (`LensPrep`); `plan`/`bind` read only it, because a
  concurrent adoption between them would pair lens A with lens B's layers and
  `KeyError` in the transport stack. A run pins R when live readout is on or
  `jlens/` probes feed an aggregate or gate, S when `slens/` probes feed an
  aggregate, a gate or live per-step readings.
- `fit(prompts, kind=, sites=, layers=, …)` runs under the exclusive guard and
  `lens_fit_lock`, after `assert_unsteered` (hooks would bake into the
  artifact). Python defaults to `kind="J"`; CLI and server to `r`.

## Readout math

`LogitReadout(transport_rows(J, h))` reuses the model's final path (norm, OPT
projection, logit scale, soft-cap). The unit everywhere is the per-layer
softmax probability `p_l(v)`; `strength = mean_l p_l(v)` with uniform weights
(softmax already lets a confident layer dominate), `com`/`spread` depth moments
weighted by `p_l`. Aggregate top-k ranks full-vocabulary strength (a union of
per-layer top-ks misses a token mid-pack everywhere). Gate and probe reads take
exact columns against the full `logsumexp`, never a `[layers, vocab]` matrix,
and leave in one packed fp32 transfer. Offline readouts chunk under 256 MiB.

## Atoms, probes, gates

Lens names are verbatim and case-sensitive (`jlens/Paris` ≠ `jlens/paris`) and
take no `:variant`. A word must be one token (the leading-space piece first,
surviving a decode round trip; else `MultiTokenWordError` with the pieces);
`#<id>` is bounds-checked (`InvalidTokenIdError`).

- **`jlens/<word>` atom.** The token's read `u = W_U[v] J_l` (every fitted
  layer) is a covector; the profile is its raised vector `Σu`
  (`LayerWhitener.apply`), the write that moves the reading furthest per unit
  of Mahalanobis displacement. Folded it reads with `Σ⁻¹Σu = u`, so a push
  raises the reading and `!jlens/<word>` zeroes exactly it (LEACE). A whitener
  missing any layer is `WhitenerError`. On real models `Σu` is mostly `u`
  restricted to the neutral span (cosine about 0.15 with `u`). One sharp token
  direction runs hot: start near α ≈ 0.1.
- **`slens/<word>` atom**: the footprint `K_l ẽ_v` at the S lens's `out` site
  (steering injects at block outputs), raised to `Σ K_l ẽ_v` the same way.
  `jlens/` steers toward saying a token, `slens/` toward having read it. No S
  lens holding `out` is `LensNotFittedError`; a masked token is
  `InvalidTokenIdError`.
- **`jlens/` probe** (roster → `attach`, never the Monitor): a
  `ScalarReading` in `mean_token_probability` with a per-layer `p_l` trace.
  `probe_hash` (`jlens-readout-v5`) binds the lens identity, so another source
  or a re-fit is another probe.
- **`slens/` probe**: per layer `z_L` of the residual centred on the neutral
  `out` mean (one batched pullback against an `[L, d]` mean stack made
  resident in `prepare`); the reading is `Σ w_S(L) z_L / Σ w_S(L)`
  (`mean_footprint_z`, 0 when no read layer has S weight). `probe_hash`
  (`slens-footprint-z-v3`) binds the S lens identity, a digest of those means
  and the mask's counts digest.
- **Gates** (`@when:jlens/…`, `@when:slens/…`) need an attached probe and read
  only its strength (`_GATE_CHANNELS = (Axis,)`; other channels are
  `UnsupportedProbeChannelError` at preflight). Scalars are computed
  post-forward and stashed by `step_id`, reused by the display step only from
  the same forward. A lens gate forces per-step lens reads whatever the live
  toggle says; an `slens/`-only gate never resolves R. Both families' rows
  leave in one host copy.

## Live readout, replay, decomposition

- `set_live(True, layers=…)` puts the `J_l` stack on the device and attaches
  **no hooks**: `live_readout_step` reads the capture's latest slices at the
  token tap, so the steering fast path and compile eligibility are untouched.
- `token_readout(node_id, raw_index, top_k=None, layers=None,
  apply_steering=True, raw=False)` replays the producing forward and returns
  the `scope="replay"` envelope with attached `jlens/`/`slens/` probes scored
  from the same capture; `top_k` defaults to 8, valid 1 to 256.
- `decompose(selector, k=16, layers=, dictionary="tokens-out")`: greedy
  pursuit against a never-materialized dictionary, `tokens-out` (`W_U J_l`),
  `tokens-in` (`K_l ẽ_v` at the S lens's `out`, masked tokens `-inf`) or
  `neurons` (block `l`'s neuron writes, Gemma's post-feedforward gain included;
  no lens; signed, so selection is by `|score|` and the sign names the pole).
  All in the session whitener's metric: the target whitens by `Σ^{-1/2}` (a
  vector), token atoms by `Σ^{+1/2}` (`whiten_dual`; they are reads), neuron
  writes by `Σ^{-1/2}`. So a token's score is its reading over that reading's
  neutral standard deviation, and decomposing a raised atom recovers its own
  token exactly. Read atoms share high-variance components, so the `k × k`
  solve runs in fp64 on CPU. `share` is the Mahalanobis variance fraction the
  atoms carry (about 6–15% for concept vectors through `tokens-out`). Whitened
  atom norms are cached per lens identity and layer beside their whitener.

## Errors and external callers

`JacobianLensError` (422), `LensNotFittedError` (404), `JacobianLensCancelled`
(409), `LensReplaceRefused` (422), `NeutralMeansError` (422),
`UnsupportedBlockError` (422); `MultiTokenWordError`, `InvalidTokenIdError`,
`EmptyCorpusError`, `SourceSpecError` are `RequestError`s (400); `HFError`
(502) is an unresolvable repo commit.

- **neuron-lens** (`neuron_lens/lenses.py`) calls `load_local_lens`,
  `save_lens`, `LensProvenance`, `JacobianLens`, `LensKey`.
- **drowse.ai** (`browser-runtime/build_hosted_packs.py`) builds its R and S
  packs from local `fit_lens` artifacts at both sites: it reads manifests
  through `LENS_SOURCES`/`load_local_lens_sidecar`, binds `lens_identity` and
  `token_counts.tokenizer_fingerprint`, and copies shards byte for byte. At its
  current pin it checks each shard against the manifest's `tensor_sha256`,
  which fits no longer write; it hashes shards itself when it moves the pin.

Tests: `tests/lens/` (CPU) on the genuine tiny llama in
`tests/support/jlens_toys.py` via `tests/support/lens.py`.
`test_lens_transport.py` holds every dense family × kind × site to the autograd
oracle and every block map to the adjoint identity (and to neuron-lens's
transport when `~/Work/neuron-lens` is checked out).
