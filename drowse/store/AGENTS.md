# drowse/store

On-disk artifacts and their distribution: manifold folders and fitted pairs,
selectors and the bare-atom ladder, templates, bundled materialization, the
`.drowse` archive, install, Hub push/pull/search, GGUF export, and the per-model
neutral and alignment caches. Layer 4: imports only `errors`/`naming`/`events`,
`base`, `geometry`, `model` and itself; it sits below `steering.grammar` because
`parse_expr` resolves bare atoms through `selectors`. No barrel module. The
fit's folder transactions, no-op preflight and capture cache are in `fit/`.

| Module | Owns |
|---|---|
| `manifold_schema.py` | `parse_manifest`/`manifest_payload`, `ManifoldSidecar`, `METHOD_FIT_MODES`, `HYPERPARAMS_BY_MODE`, `MANIFOLD_FORMAT_VERSION = 10` |
| `manifold_folder.py` | `ManifoldFolder` (`nodes_sha256`, lazy sidecars), locks, `replace_manifold_folder`, `pair_is_complete`, `read_fitted_sidecar` |
| `manifold_tensors.py` | `save_manifold`/`load_manifold`, `fitted_tensor_layers` (header only), `ActivationRowStore` |
| `manifold_authoring.py` | `iter_manifold_folders` (the one discovery walk), authors, `publish_baked_manifold`, the streaming discover skeleton |
| `manifold_lifecycle.py` | clear/rm/refresh, `manifold_upstream`, `transfer_manifold`, `manifold_summary` |
| `selectors.py`, `templates.py` | selector grammar, memoized index, `resolve_bare_atom`; `TemplateFolder`, `resolve_template` |
| `bundled.py`, `bundled_probes.py` | `materialize_bundled_artifacts`; `bootstrap_probe_roster` |
| `archive.py`, `install.py`, `hub.py`, `gguf.py` | `.drowse` v1, `install_manifold`, Hub repos, llama.cpp export |
| `neutral.py`, `alignment.py` | neutral-activation cache + `load_neutral_whitener`; cross-model alignment cache |

## `$DROWSE_HOME`

```
config.yaml                          CLI/server defaults (cli/config_file.py)
baseline_prompts.json                replaces drowse/data/baseline_prompts.json outright
neutral_statements.json              replaces drowse/data/neutral_statements.json outright
templates/<ns>/<name>/template.json
manifolds/<ns>/<name>/
  manifold.json
  nodes/NN_<label>.json              one JSON list of strings per node
  <safe>[_sae-<enc>|_from-<enc>].safetensors + .json    raw / SAE-space / transferred fit
{manifolds,templates}/<ns>/.locks/   every lock, outside the removable folder
models/<safe>/
  neutral_activations.json           pointer -> neutral_activations.layer-<L>.gen-<hex>.safetensors
  alignments/<safe_src>.json         pointer -> <safe_src>.layer-<L>.gen-<hex>.safetensors
  manifold_capture/<sha>.{centroids,rows}.layer_<L>.safetensors   (fit/capture_cache.py)
  lens/, sae/, units/{contrast,catalog}/, tokens/counts.safetensors   (their packages' AGENTS.md)
.staging/<uuid>/, .staging/backup/<path>    whole-folder builds and the parked old folder of a swap
```

`<safe>` is `safe_model_id` (`_z` + unpadded urlsafe base64), `<enc>`
`encode_release_id` (`_z` + unpadded lowercase base32) of the SAE release or
transfer source; `base/paths.py` owns both and `parse_tensor_filename`. A role
fit is the canonical `<safe>` tensor (roles live in the sidecar); `role-<x>` is
a selector suffix only.

## Manifold folder

**`manifold.json`** (format 10, exact). Required: `format_version`, `name`,
`fit_mode`, non-empty `nodes`, and `domain` for `authored`/`baked`. Defaulted:
`description`, `source` (`local`), `tags`. Unknown keys and unknown
hyperparameters are ignored on read (authoring raises on the latter). A node is
`{label, role, kind}` plus `coords` when authored. Names match `NAME_REGEX`;
labels match `LABEL_REGEX`, which drops `.` (the bipolar separator; labels are
typed after `%`). `role` (assistant-role substitution) is a fit input; `kind` is
generation provenance; both are in `nodes_sha256`. `artifact_id`/`fit_epochs`
are local transaction state (`fit/publish.py`); with the retired `files` map
they are `LOCAL_STATE_KEYS`, stripped by export, install and the bundled drift
hash. A folder is discovered only when its directory name equals `name` and every
declared corpus file exists.

Fit modes: `authored` (domain + coords, curved RBF, ≥ `min_nodes(n) = 2n + 1`
nodes); discover `pca` (flat), `spectral` (curved), `auto` (`select_topology`
per model); `baked` (corpus-less merge output that never refits).
`template_ref` is discover-only. Corpus row `i` answers prompt `i % k` of the
baseline prompts (48 bundled) or of the template's contexts.

**Fitted tensor** (fp32). Shared `node_coords (K, n)`; per layer
`layer_<L>.mean`, `.basis` (write rows, Mahalanobis-orthonormal) and `.read`
(the dual Σ⁻¹ covectors, `read @ basisᵀ = I`; required since fit policy 2), then
the affine set (`node_coords (K, R)`, optional `affine_map`) or the curved set
(`node_params`, `rbf_weights`, `poly_coeffs`, `coord_offset`, `coord_scale`,
plus the `sigma_rbf_weights`/`sigma_poly_coeffs` pair on raw curved fits). No
`node_params` is the affine marker; a fit is all-affine or all-curved. The
loader ignores unused keys and requires the tensor's layers to equal the
sidecar's `fitted_layers`.

**Sidecar** (`ManifoldSidecar`). The reader requires `format_version`, `name`,
`method`, `domain`, `node_labels`, checks only what math or a cache comparison
uses (domain, labels, sorted `fitted_layers`, share/origin maps keyed by fitted
layers, `node_spread_per_layer` keyed by the pre-DLS roster, role/kind
alignment) and drops unknown keys. The writer is strict: `save_manifold`
rejects metadata keys outside the field list, derives `SIDECAR_DERIVED_FIELDS`
from the `Manifold`, and holds `method` to `METHOD_FIT_MODES`:

| `method` | `fit_mode` | extra provenance |
|---|---|---|
| `manifold_pca`, `manifold_sae` | authored | |
| `manifold_monopolar[_sae]` | pca | 1-node ray `concept − ν` |
| `manifold_discover_{pca,spectral,auto}` | its own | auto: `resolved_fit_mode`, `topology_winner`, `topology_candidates` |
| `manifold_discover_sae` | any discover | |
| `manifold_procrustes_transfer` | any | `source_model_id`, `source_model_fingerprint`, `transfer_quality_estimate` |
| `merge` | baked | `components`, `bake_policy = additive_union_v1` |

The hot path reads only the tensor; the sidecar is the fit's identity.
`model_source_fingerprint`, `capture_version`, `capture_render_sha256` and
`baseline_prompts_sha256` let `fit/preflight.py` prove a no-op without weights;
only the browser writes `context_binding_sha256`.

**Staleness.** `nodes_sha256()` covers labels, corpus bytes, roles, kinds,
domain + coords (authored) or fit_mode + hyperparams (discover), and the
resolved template's `sha256()` (the bare ref when unresolvable, so the fit fails
loudly); a baked folder hashes only `{fit_mode, node_labels}`.
`load_manifold(check_current=True)` takes manifest then pair lock and raises
`StaleFitError` (a `ManifoldFormatError`) when a non-baked fit's
`nodes_sha256` or `fit_policy_version` is not current (the session registry
reads it as a miss and refits); a caller that already accepted the sidecar
passes `False`. A manifest at another format is a hard error; a sidecar at
another format is a cache miss (`tensor_models()` omits it) except in a baked
folder. `sidecar_is_current` reads unreadable bytes as current so the strict
loader reports the real fault.

**Publication.** A pair is published when both halves exist
(`pair_is_complete`); `ManifoldFolder.load` lists only complete pairs and reads
their sidecars lazily, so a malformed fit hides only itself. `save_manifold`
stages and fsyncs both halves, then commits the tensor before its sidecar: a
first-fit crash leaves a bare tensor (not a fit; the next fit of that target
removes it), a refit crash leaves the new tensor under the old sidecar, whose
identity reads as stale. Whole-folder writes build in `.staging/<uuid>` and swap
through `replace_manifold_folder` under `locked_manifold_folder`, which first
restores a crash-parked backup of that target.

**Locks.** Manifest lock (re-entrant), then template lock, then pair locks
(digest-named) sorted by tensor path. All live in `<ns>/.locks/` so `rm` or a
swap cannot unlink a held inode.

## Authoring and lifecycle

Authoring refuses `RESERVED_NAMESPACES` (a folder there would shadow instrument
atoms); every roster-changing call invalidates the selector index.
`publish_baked_manifold` is the only producer of baked folders. The streaming
discover skeleton (`init_…`/`append_…`/`plan_discover_generation`) is the one
in-place writer: undiscoverable until its last corpus file lands, it keeps
finished nodes across a crash.

- `clear_manifold_tensors(variant=raw|sae|from|all, model_scope=)` is the repair
  path: it deletes one-sided halves too and bumps
  `fit_epochs["<safe or *>:<variant>"]` so an in-flight fit cannot publish over
  it. Baked folders refuse clear and scoped refresh (`BakedManifoldError`): the
  tensor is the only copy. After clear/rm, a touched model's capture digest is
  deleted when no surviving sidecar names it; other orphans are left to
  `CaptureCache.prune()`.
- `refresh_manifold` → `scoped` | `bundled` (re-copy manifest + corpus, keep
  fits) | `hf` (the repo's current default revision, not the pinned one) |
  `skipped`.
- `transfer_manifold` checks the source against `preflight_transfer_manifold`'s
  proof (tensor name + header layer roster) and its `model_fingerprint`, maps
  through `geometry/transfer.py` (target whitener mandatory), nulls the
  source-only capture and merge fields, and writes the `_from-` pair.

## Selectors and templates

`selectors.parse`: `all`, `default`, `tag:`, `namespace:`, `model:` (a fitted
tensor of any variant), `[<ns>/]<name>[:variant]`; a bare name in two
namespaces is `AmbiguousSelectorError`. One memoized walk serves `concepts`,
`labels` and `names` (2-node `pca` only), keyed on a stat fingerprint (folder
listings, `manifold.json` inode + mtime, `nodes/` mtime) so another process's
edits show on the next resolve. `resolve_bare_atom` is the ladder the grammar
calls: (1) **label**, for a plain slug with variant `raw`, no typed namespace
and no `.`: a node-label hit becomes `<ns>/<manifold>%<label>` (unfitted
manifolds resolve; every bipolar pole is a label); (2) **name**: a 2-node `pca`
manifold name maps to node 0; (3) **pole**: `canonicalize_atom`. A tier 1–2
collision raises; any other failure propagates rather than falling through.

Templates (`template.json`, format 2): `name`, `slot`, `values`, `contexts`
(`{turns: [{role, content}], assistant}`), `description`, `source`, `tags`.
The slot appears exactly once in each `assistant` and never in history (history
is common-mode across values); the last history turn is `user`; ≥ 2 values,
each slugging to a unique label. `sha256()` covers format, slot, values and
contexts and folds into the referencing manifold's `nodes_sha256`.

## Bundled artifacts

`materialize_bundled_artifacts()` is the one entry point, a no-op after its
first call per process per home (in-process drift cannot be told from a CLI
hyperparameter override). Templates go first, since a bundled manifold may
`template_ref` one (none ships yet). `default/<name>` re-copies on drift
(missing folder, other format, `payload_drift_sha256` of the manifest minus
`LOCAL_STATE_KEYS`, a missing node file); fitted pairs are hard-linked across
and `nodes_sha256` decides whether they stay current. Corpus edits under
`default/` survive until the bundle changes. A `source: "bundled"` folder that no
longer ships is removed with its fits.

Shipped, unfitted: 17 concepts (`pca`, `max_dim 1`, tagged `epistemic`,
`alignment`, `register`, `cultural`): 14 bipolar axes (node 0 the positive pole)
and the one-node `deceptive`, `refusing`, `manipulative`, fitted against ν
because the model's default answer is their other end (a 2-node axis put ν on
its near pole). `personas` (`auto`, 107 nodes), `emotions` (`auto`, 20 PAD
nodes), `months` (authored periodic, because an auto fit prefers flat by GCV).
`bootstrap_probe_roster` fits-or-loads the tagged axes (keyed by bare name) and
attaches multi-node defaults only if already fitted (keyed `default/<name>`).

## Distribution

**`.drowse` v1**: a stored ZIP whose `pack.json` is `{format_version: 1, kind:
"drowse-manifold", producer, primary: "manifolds/<ns>/<name>", template, source:
{uri, repository, revision}}`. The reader needs `format_version`, `kind` and
`primary`; the ZIP CRC covers corruption. Extraction keeps path safety
(canonical relative paths, no case-fold collisions, regular unencrypted entries)
and the `DROWSE_ARCHIVE_*` bounds, extracts only the primary closure and the
declared `template.json`, and ignores other entries. Tensors are validated when
they load. Export carries the manifest (minus `LOCAL_STATE_KEYS`), corpora and
every published fit with a current sidecar, and needs a namespace-qualified
`template_ref`.

**Install** takes a `.drowse`, a local folder or `<owner>/<repo>[@rev]`; each
stages a closure and `install_closure` applies one policy: destination
`local/<name>` unless `as_` (never `default/` or reserved), `source` stamped
`local` or `hf://<repo>@<sha>`, template promoted before manifold. A differing
installed template needs `force` and no other dependent.

**Hub**: a *model* repo tagged `drowse-manifold` with exactly one root
`<name>.drowse` (all install reads), a summary `manifold.json`, and a `README.md`
whose frontmatter `drowse:` block carries the search-row fields (shared with
drowse.ai). Pull pins the commit. **GGUF** folds each raw R = 1 fit to one
llama.cpp control vector per model, never inside `$DROWSE_HOME`.

## Per-model caches

Both are `base/shards.py` generations: immutable per-layer shards behind an
atomic JSON pointer, single-flighted by a `fit_lock`; the next writer collects
superseded generations.

- **Neutral** (format 5, capture version 2): `layer_<L>` `[N, D]` fp32 per
  shard, the one per-model neutral artifact (probe means are `X.mean(0)`, the
  whitener is built from it). When the blocks are described a shard also holds
  the unit statistics `<name>_<L>` (`units/AGENTS.md`) and `mid_mean_<L>` (the
  centring mean of an S read at `mid`); the pointer's `units`/`mid` blocks say
  `available` or give the `reason`. The reader requires exact versions,
  `layers`, both blocks and finite fp32 tensors; anything else recomputes.
  Alignment pointers embed `neutral_cache_identity`, whose `tensor_files` names
  a random generation, so a recapture never shares an identity.
  `load_neutral_whitener` is the offline loader and raises `WhitenerError`
  rather than load a model.
- **Alignment** (format 5): per-layer `left`/`right`/`offset` shards. The
  pointer embeds both neutral identities (stored keys the current identity no
  longer names are ignored); a mismatch is rejected before payloads are read, and
  a damaged layer is skipped and refits.

## Errors and seams

`ManifoldFormatError`/`BakedManifoldError` (`ValueError`, no declared status)
answer 400 via `server/errors.py::_drowse_status`; `TemplateFormatError`,
`AmbiguousTemplateError`, `SelectorError`, `DrowseArchiveError` are
`RequestError`; `TemplateNotFoundError` 404; `HFError` 502.

drowse.ai's `webui/src/hosted/artifacts/drowseArchive.ts` hand-mirrors the
sidecar format, `METHOD_FIT_MODES`, `RESERVED_NAMESPACES` and `pack.json` v1;
`tests/browser_drowse_archive_bridge.py` is the Python end of its interchange
test (keep its path, arguments and stdout).

Tests: `tests/store/` (CPU); fit-side folder transactions in `tests/fit/`.
