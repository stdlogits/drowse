# drowse/sae

Sparse autoencoders: provider (SAELens) and Drowse-trained dictionaries, their
per-model source registry, the one resident dictionary a session reads, the
Neuronpedia client, local training, and the `sae` read instrument
(`session.sae`). Layer 7 beside `lens`; imports `errors`, `base`, `model`,
`readout` only, and `DrowseSession.__init__` wires every dependency. The
instrument contract is `drowse/readout/AGENTS.md`.

| Module | Owns |
|---|---|
| `backend.py` | `SaeBackend`, `SaeLensBackend`, `LocalSaeBackend`, `load_sae_backend` (resolution, commit pinning), `resolve_sae_backend`, `select_runtime_layer`, `list_sae_releases`, `sae_binding` |
| `sources.py` | `SAE_SOURCES`, both source grammars, provider bindings, the Neuronpedia feature cache, use/remove/list |
| `store.py` | Drowse-trained artifacts (`sae/local/<name>/`) |
| `runtime.py` | `SaeRuntime`, `SaeResidency`: load, restore, train, feature validation and metadata |
| `neuronpedia.py` | one validated HTTP read per feature |
| `instrument.py` | `SaeInstrument`, `SaeRun`: probes, gates, live discovery, aggregate, replay, `sae/<id>` directions |
| `training.py` | `train_residual_sae`, `SaeTrainingCancelled` |

SAELens (`[sae]`: `sae-lens>=6.0,<7.0`, `nltk>=3.10.3` past its CVE'd
releases) is imported lazily on provider paths only (`load_sae_backend` for a
non-`local:` release, `list_sae_releases`); local SAEs, `sae/<id>` steering,
probes and Neuronpedia work without it. Missing, it raises
`SaeBackendImportError` (400) with the install hint, except that
`list_sae_releases` returns `[]` when local artifacts exist. SAELens 6 shims
live in `backend.py`.

## Sources and disk layout

`SAE_SOURCES` (a `base.source_registry.SourceFamily`) is the one parser of both
grammars: the public source (`local:NAME`, `saelens:RELEASE`; a release need
only be non-blank) and the backend release string (`local:NAME` or the bare
release; `parse_sae_release`/`sae_release` convert).

```
models/<safe_model_id>/sae/
  active.json                      one selection per model (format 1)
  bindings/_z<b32>.json            provider binding (format 3)
  bindings/_z<b32>-features.json   Neuronpedia feature cache (format 4)
  local/<name>/manifest.json       trained artifact (format 1)
  local/<name>/layer-<L>.gen-<hex>.safetensors   fp32 W_enc, W_dec, b_enc, b_dec
```

`_z<b32>` is `base.paths.encode_release_id` (lowercase unpadded Base32,
reversible). Provider weights stay in the SAELens/HF cache, never copied or
written into. Every reader requires the fields it uses and ignores the rest; a
missing or invalid field reads the record as absent.

- **Binding.** `sae_binding(backend, layer, width)` is the one shape (`{layer,
  width, revision, fingerprint, sae_id, repo_id, neuronpedia_id}`) behind the
  binding file (plus `format_version`, `model_id`, `release`),
  `session.sae.info` and the feature cache's key.
- **Feature cache.** `{format_version, model_id, release, source, features}`,
  `source` the seven binding fields, so a cache from another layer, revision
  or dictionary is ignored whole. Rows are `{label, max_act}` under canonical
  decimal ids; one bad row voids the file. The in-memory `checked` flag is not
  persisted.
- **Local artifact.** The manifest records `activation: "relu"`, one `layer`,
  `d_model`/`d_sae`, model and source fingerprints, corpus spec and sha256,
  training hyperparameters, `tensor_file` and `tensor_sha256`. `save_local_sae`
  writes a fresh generation, fsyncs and hashes it, writes the manifest
  atomically under `artifact_lock`, then unlinks older payloads
  (`FileExistsError` without `force`). `tensor_sha256` is the payload's
  identity (the backend's `revision` and `fingerprint`, which fitted-tensor
  reuse compares), not an integrity check: loads check the fields, the
  safetensors header (the four weights' shapes and dtype) and finiteness, and
  never re-hash.
- `use_sae_source` selects only a prepared source; `remove_sae_source` deletes
  a local folder, or a provider's binding and feature cache, and clears the
  selection only if it named that source.

## Backends (`backend.py`)

`SaeBackend`: `release`, `revision`, `fingerprint`, `layers` (0-indexed
blocks), `encode_layer(idx, (N, D)) → (N, F)`, `decode_layer`,
`feature_count`, `feature_direction` (the decoder row).

- **`SaeLensBackend`** resolves the registry eagerly and loads one layer at a
  time (a whole release can outweigh the model). Encode/decode delegate to the
  provider and must not be reimplemented: Gemma Scope 2 JumpReLU encodes
  uncentred (`apply_b_dec_to_input=False`) while local SAEs subtract `b_dec`
  (`tests/sae/test_sae_numerics.py`).
- **`LocalSaeBackend`** loads once on first use (model device and dtype, fp32
  by default), encodes `relu((h − b_dec) W_enc + b_enc)`, covers only its
  layer, and raises if the manifest's `tensor_sha256` changed since resolution.

`load_sae_backend` (provider releases):

- Unknown release: `SaeReleaseNotFoundError` with suggestions; model names
  compare by last path segment, case-insensitive, substring either way, else
  `SaeModelMismatchError`.
- **One SAE per layer**: Neuronpedia-hosted first (it supplies labels and the
  `maxActApprox` strength unit), then narrowest width, then smallest L0
  (`small < medium < big`), then id; warns when it had to choose.
- **Immutable identity.** With a `repo_id` the revision resolves to a Hub
  commit (`timeout=3`; a 40-hex revision skips the round trip, so a pinned
  restore runs offline); if none resolves every layer load raises, so mutable
  weights are never mixed or persisted. `fingerprint` hashes `{release,
  repo_id, resolved_commit, sae_ids_by_layer, sae_lens_version,
  conversion_func, config_overrides, dtype}`. The commit is passed as
  `revision=` when the loader accepts it, else `_commit_pinned_converter`
  clones the converter into isolated globals with every Hub read pinned, else
  the load is refused.
- **Residual-post only.** Capture is the block output; an attention/MLP SAE
  fails on width or, when widths agree, silently reads garbage. The config's
  `hook_layer` must equal the registry mapping and the hook name must end in
  `.hook_resid_post` or `layers.<L>.output`; discovery drops releases named as
  attention, MLP or transcoder families.

`resolve_sae_backend(...) → ResolvedSae(backend, layer, width)` is the one path
behind `sae fetch` and `SaeRuntime.load`: it rejects a local artifact for other
weights (`model_fingerprint`, before loading), picks the layer, reads its width
and checks it against the hidden size. `sae fetch` never instantiates the base
model. `select_runtime_layer`: an explicit layer must be covered
(`SaeCoverageError`); otherwise the covered layer nearest `0.65·(n_layers −
1)`, within the 0.4–0.9 band when any covered layer lies there, lower on ties:
deterministic, so restores and the unqualified `sae/<id>` are stable.
`sae_device_str` maps `mps:0` to `mps`.

## Residency (`runtime.py`)

At most one dictionary at one hook layer: `SaeResidency(backend, layer, width,
feature_meta)`, frozen with `eq=False` so each load's identity is the "still
current?" token. Feature ids mean something only against that release and
layer, which is why `sae/<id>` takes no qualifier. `state_lock` is a reentrant
leaf lock shared with the instrument, never held across a forward or a fetch.

- **`load`** holds the exclusive section across resolution and download (the
  server switches sources only as a background `fetch`; a direct switch is
  409): resolve → feature cache → publish binding and selection → swap and
  `on_swap` under `state_lock`. A failure before the swap leaves the previous
  source resident.
- **`restore_active`** (end of `DrowseSession.__init__`) reloads the binding's
  pinned commit and layer, offline, never upstream HEAD; failure only warns.
- **`train`** runs in the exclusive section under `assert_unsteered()` (an open
  scope would bake into the artifact), on `model._orig_mod` when compiled,
  records `sha256(repr(docs))`, then loads the result.
- **Feature metadata.** `check_feature` (parse and range, `SaeFeatureError`) is
  the steering path's only check. `validate_feature`/`fetch_meta` fetch outside
  every lock (8 workers) and publish only if the residency is still current, so
  a late response never labels a successor SAE. `fetch_meta` is the
  dashboard's between-generation backfill, never called from the decode loop.

**Neuronpedia.** `GET https://www.neuronpedia.org/api/feature/<model>/<source>/<index>`
(`<model>/<source>` is the binding's `neuronpedia_id`; UA `drowse-sae-meta/1`,
10 s). The response must match `modelId`, `layer`, `index`, and
`source.hfRepoId`/`source.saelensSaeId` where known, else nothing is cached.
`max_act` is `maxActApprox` when finite and positive. Local SAEs have no id and
read in the raw unit.

## The instrument (`instrument.py`)

**One channel.** A probe emits `<name>` and `<name>[0]`: `activation /
maxActApprox` (about [0, 1], comparable across features), or raw activation
when `max_act` is unknown (`ScalarReading.unit`: `activation_over_max` /
`raw_activation`). Depth is `layer/(n_layers − 1)`, spread 0, basis
`single_layer`. `validate_gate` admits only the axis
(`UnsupportedProbeChannelError` at preflight otherwise).

- **Probes.** `attach("sae/<id>", as_name=)` validates first (may fetch,
  outside locks), then inserts under the exclusive section and `state_lock`.
  Not Monitor probes: no whitener.
- **Per generation** (`prepare → plan → bind → close_run`). `plan` captures
  the live layer, plus the resident layer when probes feed an aggregate or a
  gate. `bind` freezes specs with `max_act` resolved and snapshots the live
  config, so a backfill or a toggle cannot change what a running generation
  measures.
- **One encode per step.** `gate_scalars` encodes the latest slice and stashes
  it with its `step`; live readout and pinned probes reuse it only on a
  matching `step_id`, so staleness is structural. `probe_values` gathers every
  pinned id with one `index_select` and one host transfer. `score_aggregate`
  reads the pooled last-content-token slice (the monitor's and lens's
  position).
- **Live discovery** rows are `(id, raw activation, label, max_act)`, labels
  from the cache only; the width follows the generation's `return_top_k`, so
  `set_live` takes no extras.
- **Replay.** `token_readout(node_id, raw_index, top_k=None → 8, …)` returns
  the `scope="replay"` envelope with attached probes from the same encode;
  `top_k` in `[1, min(width, 256)]`, and a `layers` argument is a
  `RequestError`.
- **Steering atom.** `sae/<int>` (no `#` ids, no variants) resolves through
  `register_direction` to `W_dec[id]` at the resident layer as the one-layer
  profile `sae/<id>`. A decoder row is a write vector, so unlike a lens atom it
  is not raised. It runs while the composer holds the exclusive section, so it
  range-checks and never fetches.
- **Swap.** `_on_swap` clears probes, live config and stash, evicts every
  `sae/` profile and forgets probe hashes: feature ids never cross a swap.
  `_on_fetched` replaces matching probe specs whole (idle readers may hold
  references) and forgets their hashes.
- **Identity.** `probe_hash` hashes `("sae-readout-v2", model_id, fingerprint,
  release, layer, feature_id, max_act)`; `max_act` sets the unit.
  `active_source` derives from residency, not `active.json`.

## Training (`training.py`) and SAE-space fits

`train_residual_sae` fits a one-layer ReLU SAE on block outputs. Forwards run
under `inference_mode`, `use_cache=False`; a hook raises `_SaeCaptureComplete`
to skip the rest of the model, a `BaseException` because a broad `except
Exception` in modeling code would swallow it and run the full forward.
Decoder rows start as seeded unit vectors with `W_enc = W_decᵀ`, `b_dec` the
first batch's mean; Adam minimizes `MSE + l1·mean|f|`, renormalizing decoder
rows every step (else L1 is evaded by shrinking the encoder and growing the
decoder). Defaults: 1M tokens, seq_len 128, batch 8, expansion 8, lr 3e-4, l1
1e-3, dead threshold 1e-6, seed 0. Fire counts stay on device; one host
transfer per step. `cancel_event` is checked per batch (`SaeTrainingCancelled`,
409).

`--sae REL` fits (`drowse/fit/AGENTS.md`) load their own backend
(`load_sae_backend`, not the resident runtime), round-trip each centroid
through encode/decode per covered layer, and write
`<safe_model>_sae-<rel>.safetensors` in model space, so hooks never touch the
SAE. They refuse role baselines.

## Errors and tests

`SaeReleaseNotFoundError`, `SaeModelMismatchError`, `SaeCoverageError`,
`SaeFeatureError` are `RequestError`s (400); `SaeBackendImportError`
(`ImportError`, 400); `SaeNotLoadedError` (404); `SaeTrainingCancelled` (409).

`tests/sae/` (CPU) with `tests/support/sae.py` (`MockSaeBackend`,
`sae_instrument(...)`, `make_resident(...)`); session tests monkeypatch
`drowse.sae.runtime.resolve_sae_backend`; tests needing SAELens
`importorskip` it.
