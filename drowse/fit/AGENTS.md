# drowse/fit

Corpora in, fitted manifold tensors out: the one extraction pipeline, corpus
generation, the neutral-activation capture behind the whitener, the offline
bake, cross-model alignment orchestration and the naturalness eval. Layer 10:
above `generation`, below `loom`/`session`, so hosts are Protocols
(`ModelHandle`, `NaturalnessHost`) that `DrowseSession` satisfies. The math is
`geometry/` (Fisher basis, frame, share, discover coords, topology, RBF,
σ-field, fold); folder, tensor and sidecar formats and the locks are `store/`;
namespace resolution, the exclusive section and adopt/evict are
`session/extraction.py`.

| Module | Owns |
|---|---|
| `pipeline.py` | `ManifoldExtractionPipeline`, `prepare_manifold_capture_identity`/`CaptureIdentity`, `prepare_capture_rows`, `offline_fit_identity`, `FitInputs.accepts`, `MANIFOLD_CAPTURE_VERSION = 3`, the DLS notes |
| `publish.py` | `fit_target_lock`, `take_authoring_snapshot`, `load_published_fit`, `publish_fit_if_current`, `ManifoldAuthoringChangedError` (409) |
| `preflight.py` | `preflight_manifold_fit_noop`: weight-free proof that a fit is already exact |
| `capture.py` | pooled residual capture, `_LENGTH_DIRECTIVE`, the prompt/neutral corpus loaders, `compute_neutral_activations` + `ChunkObserver`, `compute_dls_axes` |
| `node_stats.py` | `compute_manifold_node_stats` (centroids + optional row spool), `compute_store_reduced_covariances` (σ-field input) |
| `capture_cache.py` | `CaptureCache`: content-addressed per-layer capture shards |
| `corpus.py` | `CorpusGenerator`, `KIND_TEMPLATES`, `system_for`, `role_for`, `split_composite_source` |
| `neutral.py` | `load_or_compute_neutral_cache` (+ activation-only views), `MidMeans`; IO is `store/neutral.py` |
| `bake.py`, `transfer.py`, `naturalness.py` | `merge_into_manifold`; `load_or_fit_transfer_alignment`; `preflight`/`evaluate` |

A steering vector is the 2-node flat manifold, so `session.extract` and
`manifold fit` both run `ManifoldExtractionPipeline.fit`. `extract` authors a
discover `pca` folder with `{max_dim: 1, var_threshold: 0.7}` (two nodes for a
bipolar `pos.neg`, one for a monopolar concept) and returns the folded
`Profile`; the folder is the only stored artifact. Stages: **inputs** (snapshot,
capture identity, fingerprints, SAE backend; a published fit that `accepts`
returns as is), **capture**, **geometry** (pure tensors), **publish**
(compare-and-swap through `publish.py`, the pipeline's only folder access).

## Corpora

A node corpus is `list[str]` aligned `response[i] ↔ prompt[i % k]` against the
48 baseline prompts (a `$DROWSE_HOME` override replaces them) or a template's
contexts; its length must be a multiple of `k`. `CorpusGenerator` elicits in
character: system `f"{_LENGTH_DIRECTIVE} {system_for(c, kind)}"` under a swapped
assistant label (an explicit `role` beats `role_for`):

| `kind` | System (`{c}` humanized) | Label |
|---|---|---|
| `abstract` (default) | `You are someone {c}. Respond exactly as someone {c} would.` | `someone_<slug>` |
| `concrete` | `You are {art} {c}. Respond exactly as {art} {c} would.` | `<slug>` |
| `custom` | caller template (`{c}`, `{art}`) | none |

`abstract`/`concrete` need a substitutable role header
(`RoleSubstitutionUnsupportedError`, e.g. Mistral). The persona is
generation-only: capture renders `[system: _LENGTH_DIRECTIVE, *history,
assistant: response]`, so nodes pool in standard-assistant space and the signal
rides the text. Only an explicit manifest `role` (the `:role-<x>` fit) is
substituted at capture too, which is why `personas` needs a role-capable family
to regenerate but fits on any.

`_LENGTH_DIRECTIVE` ("Answer in one short paragraph.") is identical at all four
sites (node and neutral generation and capture): common-mode, it cancels in
`node − ν` and keeps answers under the 256-token cap. Generation is HF
`model.generate`, left-padded batches of 16, `temperature=1.0`, `top_p=0.9`. A
row that samples EOS first is retried alone, greedily with `min_new_tokens=8`
(dropping it would misalign later rows). `fill(plan, seed=)` seeds node `i` with
`seed + i` and writes each node as it finishes, so a resumed run equals a full
one.

## Identity and cache keys

`capture_sha256` covers the loaded-model identity (id, class, commit,
`model_type`, loaded fingerprint, `MANIFOLD_CAPTURE_VERSION`), node sizes, and
every rendered row's token ids and pool index. It excludes geometry, layer scope
and folder, so geometry-only refits and identical corpora share a capture. Its
weight-free half is `capture_render_sha256` + `baseline_prompts_sha256`.
`prepare_capture_rows` is the one row flattener for the fit, the preflight and
the registry's steer-time check, so the render must stay deterministic.

`FitInputs.accepts` matches `nodes_sha256` (`store/AGENTS.md`),
`model_fingerprint`, `capture_sha256`, `fit_policy_version` and `sae_release`; a
raw fit also needs `evaluated_layers` (the pre-DLS roster) to equal the request,
an SAE fit its fingerprint, the requested revision, and exact layers for an
explicit scope (else `sae_full_coverage`).

| Constant | A bump invalidates |
|---|---|
| `MANIFOLD_CAPTURE_VERSION = 3` (`pipeline.py`) | every fitted tensor and cached capture |
| `MANIFOLD_FIT_POLICY_VERSION = 2` (`geometry/manifold.py`; 2 typed the frames) | fitted tensors |
| `CAPTURE_CACHE_FORMAT = "5"` (`capture_cache.py`) | capture shards |
| `NEUTRAL_CAPTURE_VERSION = 2` (`store/neutral.py`; 2 added the mid means) | neutral caches |

`dls` is not in the key: toggling `--no-dls` on a fitted target returns the
cached tensor unless forced. Sidecars record no neutral identity, so
regenerating the neutral corpus or cache invalidates no fit.

## Capture

Before any forward, `_require_neutral_baseline` raises `WhitenerError` unless the
whitener `covers_all` fit layers and the neutral means cover them. Rows are
rendered and tokenized once on CPU (thinking off) and pool at
`model.tokens.last_content_index`. `compute_manifold_node_stats` streams every
row across node boundaries, length-sorted, in right-padded batches that start at
64 and halve on OOM; right-padding is exact because the pooled position attends
only to real tokens. `_CaptureComplete` (a `BaseException`, so a model wrapper's
`except Exception` cannot swallow it) aborts after the last requested block, so
the final norm and LM head never run. Centroids reduce per node on device in
fp32; curved raw fits also spool source-dtype rows to an mmap
`ActivationRowStore`. With `sae=`, centroids are reconstructed through the SAE
one layer at a time and a default scope is exactly the release's layers; SAE
fits exclude roles, keep no rows (no σ-field) and stay model-space, so hooks
never touch the SAE.

**Capture cache** (`models/<safe>/manifold_capture/`, one shard per digest,
kind and layer). Each shard carries its identity (format, digest, node sizes) in
its header and is atomically replaced; there is no pointer, so a missing or
foreign shard is a per-layer miss and the worst case is a recapture. `prune()`
evicts whole digests, oldest first and never the current one, down to
`DROWSE_MANIFOLD_CAPTURE_CACHE_GB` (default 8; `≤ 0` off). Rows are kept only
for raw `authored`/`spectral`/`auto` fits; `auto` persists them only if it
resolves curved.

## Geometry

**Monopolar** (1-node `pca`): `concept − ν` per layer folds
(`fold_directions_to_subspace`) into a neutral-anchored ray: no discover
coords, no DLS, no synthetic second node.

**K ≥ 2.** `_whitened_node_geometry` computes `Σ⁻¹X̃` and `G_L = X̃Σ⁻¹X̃ᵀ` of
the node-mean-centered centroids once per layer (and `node_spread = tr(G_L)`)
for discovery, topology, the Fisher basis and the neutral layout. The consensus
Gram `mean_L G_L` is signal-weighted (a layer whose nodes don't separate adds
≈ 0), so there is no reference layer. `_resolve_fit_geometry` is the only mode
branch:

| `fit_mode` | Coords / domain | Per-layer fit | Node floor |
|---|---|---|---|
| `authored` | manifest | curved, exact (node = steering target) | `2k + 1`, poised |
| `pca` | `derive_pca_coords`, `CustomDomain(k)` | flat affine | `k + 1` |
| `spectral` | `derive_spectral_coords` | curved, GCV smoothing | `2k + 1` |
| `auto` | `select_topology`: flat, curved or periodic `BoxDomain` | as resolved | as resolved |

`auto` forwards only `max_dim`, `smoothing`, `persistence_frac` and
`max_subspace_dim` (as the scoring/Fisher width); it accepts and hashes
`var_threshold`/`min_dim`/`k_nn`/`bandwidth` but never uses them.

**Flat**: `fit_affine_subspace` with `n_components` = the layout width and
`orient_to=0`, so a 2-node fit writes `+δ` and reads `Σ⁻¹δ` (PCA@2 ≡ DiM). DLS
prunes axes, then `subspace_share` is baked on the survivors. The display layout
is translated onto ν (`_neutral_centered_layout`) so `% 0,…,0` reads as
neutral; no origin is stored (ν's foot is coordinate 0).

**Curved**: `fit_curved_subspace` over one `RbfFitPlan` (`auto` reuses its
scoring plan and Fisher bases); smoothing `None` (exact) for authored, `"auto"`
(GCV) by default otherwise, `0` exact, a float a fixed λ; width
`max_subspace_dim` (default 64). Raw fits then reduce the retained rows through
each layer's `read` rows for the σ-field, and every layer stores `origin`, ν's
foot, which seeds the solve. No DLS: a curved fit has no polarity.

**DLS** (`compute_dls_axes`; Dang & Ngo 2026, Eq. 9) keeps axis `r` at layer L
iff the node readings `(c_i − ν_L)·p_r` through its read row straddle zero:
same-side axes encode intensity, not polarity, and at K = 2 this is the bipolar
opposite-sign test. A layer with no surviving axis drops; if nothing passes
anywhere, every checkable axis is kept. The fit narrates the outcome
(`dls_fit_note`, that fallback included) and serve's startup repeats it for a
cached probe from the sidecar alone (`dls_load_note`). A pole sitting on ν is
why `deceptive`, `refusing` and `manipulative` ship monopolar.

## Publish and preflight

`fit_target_lock` is per (folder, tensor filename), so distinct targets in one
folder fit concurrently while readers keep the old pair.
`take_authoring_snapshot` runs under the manifest lock: it removes a one-sided
pair (a crashed first fit), merges discover overrides (a mode change resets
hyperparams), re-materializes a drifted templated corpus, mints `artifact_id`,
and hashes `revision = H(artifact_id, target fit_epochs, nodes_sha256,
prompts)`. Compute holds no folder lock; a cache hit (`load_published_fit`, pair
lock only) re-asserts the revision, and `publish_fit_if_current` raises
`ManifoldAuthoringChangedError` (409, retry) on drift. `pack clear` bumps
`fit_epochs` and rm + recreate mints a new `artifact_id`, so a paused fit never
publishes over either.

`preflight_manifold_fit_noop` runs before the CLI opens a session; `None` means
*unproven*, not stale. It declines SAE fits, overrides, `baked`, an unprovable
checkpoint source, a missing neutral cache, an unloadable tokenizer and a
drifted templated corpus, recomputes every key component offline, and bridges
`model_source_fingerprint` to the loaded fingerprint through the neutral cache
sidecar, which a real load wrote.

## Neutral activations

`load_or_compute_neutral_cache` is single-flighted by `neutral_fit_lock` across
recheck, capture and publish. Its identity is the rendered neutral rows (with
`NEUTRAL_CAPTURE_VERSION`), both model fingerprints and the layer set; a stale or
unreadable cache recomputes. `session/neutral.py` derives the means, whitener and
resident `UnitStats` from what it returns.

The same forward feeds two observers framed per chunk (`begin_chunk`,
`commit_chunk`, `discard_chunk` on an OOM retry, `abandon(reason)` at batch size
1, after which the rows re-run without their hooks): `UnitStatsAccumulator`
(`units/AGENTS.md`) and `MidMeans` (one hook per block on the MLP input norm).
Neither fails the pass: a model `model_spec` refuses, a hook that raises, or a
persistent OOM publishes `units`/`mid` unavailable with the reason. Only the
refusal lasts: a hit recording either as unavailable for a model `model_spec`
now accepts is stale and recomputes.

## Bake, transfer, naturalness

**Bake.** `merge_into_manifold` writes `local/<name>` (`fit_mode="baked"`) from
namespace-qualified scalar `+`/`-` terms only; `!`, `%`, `~`/`|`, triggers and
`{…}` raise `MergeError` (projection needs an identity-matched whitener; a bake
is layer-agnostic). Components must fold to one direction, with
`:role-<r>`/`:from-<src>` checked against the sidecar. `linear_sum` unions layers
(`strict`: equal coverage); the sum folds under each model's persisted whitener
(no model load); component fingerprints must agree. CLI `manifold bake` is
offline (no `-m` = every shared model); `session.bake` scopes to the loaded model
and guards its fingerprint.

**Transfer.** `load_or_fit_transfer_alignment` returns the per-layer map, R²
quality, both neutral identities, and the target whitener and means for the
re-bake (`store.manifold_lifecycle.transfer_manifold`). The concurrency is
load-bearing: the directional `alignment_fit_lock` is held throughout; a model
loads only on a neutral miss, under its `neutral_fit_lock` taken before the
load, so concurrent cold transfers share one load; header-only neutral checks
are re-verified after the rows load and a racing writer restarts the
transaction; `force` refits the map, not an exact neutral cache.

**Naturalness.** `preflight` is model-free (not `baked`, ≥ 3 nodes;
`--compare-linear` needs a single `%` term). `evaluate` fits a behavior manifold
over the nodes' unsteered next-token distributions in Hellinger space (`√p`,
Euclidean PCA + exact r³ RBF over the manifold's own layout; output space has no
rogue dimensions, so no whitener) and scores one steered generation (`seed=0`)
per step by Bhattacharyya distance to it.

Tests: `tests/fit/` (CPU; `tests/support/capture.py::install_capture` patches
the batched capture, `StubHandle` is the `ModelHandle`), plus
`tests/store/test_neutral_cache.py`, `tests/units/` and `tests/gpu/test_smoke.py`.
