# drowse/cli

The `drowse` console script (`drowse.cli.main:main`; `python -m drowse` is the same entry):
nine verb groups (`serve`, `manifold`, `pack`, `config`, `experiment`, `template`, `lens`,
`sae`, `unit`) as thin frontends over `DrowseSession` and the `store`/`fit`/`lens`/`sae`/
`units` packages. Top layer beside `server`, `web`, `notebook`; no session-private reads.
The verb and flag synopsis is in the root `AGENTS.md`; `drowse <verb> -h` lists every flag
and default.

## Module map

| File | Owns |
|---|---|
| `main.py` | `build_parser`, `parse_args`, `main` (the one error boundary) |
| `spec.py` | `Arg`/`arg`, `OneOf`, `Verb`; argument types (`positive_int`, `nonnegative_int`, `bounded_int`, `smoothing`); shared flag specs (the reserved short-flag table); `attach` |
| `context.py` | `CliError`, `exit_code_for`, `user_errors`; `note` / `progress` / `emit_json`; `effective_config`; `open_session`, `resolve_probes`; name resolution; `check_steering`, `parse_layers` |
| `config_file.py` | `ConfigFile` (setup-YAML schema, one validator per key in `_VALIDATORS`), `compose`, `ensure_vectors_installed`, `ConfigFileError` |
| `alpha_grid.py` | `parse_alpha_list` for `experiment fan` grids |
| `verbs/<group>.py` | one `GROUP: Verb` tree per group, its `run_*` runners, local helpers |

## Parser and dispatch

- A group is a `Verb` with `verbs=`, a leaf a `Verb` with `run=` and `args=`. `attach` walks
  the tree once: a leaf gets `set_defaults(func=run, command="<verb path>")`, a group a
  `func` that prints its own help; a leaf without `run` raises at build.
- `Arg` is one `add_argument` call as data; `Arg.but(**kw)` copies a shared spec with keywords
  replaced; `OneOf` is a mutually exclusive group; nested tuples flatten (`MODEL_LOAD`,
  `CONFIG`, `INJECTION`). Reuse `spec.py`'s shared specs instead of redeclaring a flag.
- Bare `drowse` or a bare group prints that level's help and exits 0; an unknown verb is an
  argparse error (2). There is no bare-model fallback: `drowse google/gemma-2-2b-it` is an
  argparse error.
- Building the parser is torch-free: verb modules import only torch-free modules at top level
  (`drowse.cli.*`, `drowse.errors`, `drowse.lens.key`) and runners import the engine in their
  bodies (`test_help_does_not_import_torch`).

## Errors and output

`main` prints one stderr line `drowse <verb path>: <message>` for a handled failure:

| Escaping exception | Exit |
|---|---|
| `CliError` | its `exit_code`: 2 = usage / bad input (default), 1 = runtime miss |
| other `DrowseError` | `exit_code_for(user_message()[0])`: 404 gives 1, other 4xx 2, 5xx 1 |
| `FileNotFoundError` / `FileExistsError` | 1 |
| `KeyboardInterrupt` | 130 |
| anything else | traceback (a bug) |

The server maps the same `user_message()` status, so every 400-status error (`RequestError`
family, `WhitenerError`) exits 2 on both. `user_errors(*types, exit_code=1)` turns a backend's
bare stdlib exception into a `CliError`; `RequestError` subclasses `ValueError`, so wrapping
`ValueError` also re-codes typed request errors (`extract`, `fit`, `merge`, exports exit 1):
type the backend error rather than widening a wrap.

stdout carries only the verb's result. Narration (`note`), progress (`progress`, the
indented `on_progress` sink for long operations) and warnings go to stderr; `-j` prints
exactly one JSON document (`emit_json`).

## Config loading

`-c/--config PATH` (repeatable) and `-s/--strict` ride `serve` and the three `experiment`
verbs. `effective_config(args, *, default_max_tokens=1024, yaml_vectors=False)` composes
`$DROWSE_HOME/config.yaml` (if present) then each `-c` file field by field, later wins
(`vectors` replaced, not concatenated). Precedence: CLI > YAML > runner default.

| Field | Rule |
|---|---|
| `model` | fills only an omitted positional (`serve`, `transcript run`); `transcript run` then falls back to the transcript's `model_id` |
| `max_tokens` | `--max-tokens` defaults to `None`; floors: `serve` 1024 (no flag), `fan` and `transcript run` 256, `naturalness` 128; elsewhere `open_session` uses 1024 |
| `compile`, `cuda_graphs` | YAML `true` switches on; YAML cannot switch off a CLI flag |
| `return_top_k` | fills `--top-k-alts` when unset |
| `temperature`, `top_p`, `thinking`, `system_prompt` | YAML-only session defaults (`system_prompt` via `from_pretrained`, the rest rebind `session.config`) |
| `vectors` | serve only (`yaml_vectors=True`): fills an unset `-S`; research verbs use exactly their own `-S`, so a serve default never shifts a sweep |

`ensure_vectors_installed(steer, strict=args.strict)` walks
`steering.grammar.referenced_selectors`: reserved readout namespaces
(`naming.RESERVED_NAMESPACES`) are skipped, a bare reference counts as resolved when a
manifold label or name tier matches (an ambiguity counts), a missing HF `ns/name` installs, a
missing `local/` or `default/` one cannot. Misses warn; `-s` raises `ConfigFileError` (2).
`ConfigFile.load` rejects unknown keys (a new key needs a dataclass field and a `_VALIDATORS`
entry; an assert keeps them equal) and non-boolean booleans (a quoted `"false"` would flip an
opt-in), and parses `vectors` at load. `config validate FILE` adds
`ensure_vectors_installed(strict=True, dry_run=True)` and exits 2 on any failure.

## Sessions

`open_session(args, *, roster=False)` is the one construction path, narrated on stderr, built
from `device`, `dtype`, `quantize`, `system_prompt`, `max_tokens`, `dls = not --no-dls`,
`compile`, `cuda_graphs`, `return_top_k`, closed on exit.

- `roster=True` only for verbs that read probe readings (`serve`, `experiment fan`,
  `experiment transcript run`). `-p`: unset or `all` gives `None` (tagged concept axes plus
  already-fitted bundled multi-node manifolds), `none` or empty `[]`, categories verbatim.
  Everything else constructs with `probes=[]`, which also skips construction-time neutral
  capture and fits.
- `-d/-q/--dtype` (`MODEL_LOAD`) exist on exactly the verbs that load weights: `serve`,
  `manifold extract|generate|fit`, `template score`, `lens fit|top|decompose`, `sae train`,
  `unit read|top|find|stats`, every `experiment` verb. `manifold transfer` has none but loads
  a model (default device and dtype) when it must capture a missing neutral cache. `--dtype`
  is long-only and reaches the loader and the model-source fingerprint, so fit identity
  records the dtype and the no-op proofs key on it.

## Fail before the load

Everything checkable without weights runs before `open_session`: serve's bind check
(non-loopback `--host` needs `-k/--api-key` or `$DROWSE_API_KEY`) and model presence; every
`-S` parse; fan's grid; `fit.naturalness.preflight` (not `baked`, at least 3 nodes,
`--compare-linear` needs one `%` term); sae train's numeric ranges; extract's arity and
`--kind custom` needing `--system`.

**Exact no-op proofs skip the load.** `manifold extract|fit` run
`fit.preflight.preflight_manifold_fit_noop`; `lens fit` runs `lens.lifecycle.
prove_lens_fit_noop` before streaming its corpus. A proof prints the cache-hit line and exits
0; `-f` skips it. Proofs do not short-circuit on file existence or a model id: a Hub repo is
mutable and fit identity also covers corpus, tokenizer render, role and SAE transform. A
discover override is never proven (it rewrites `manifold.json` inside the fit lock), and
`extract` tries the proof only when every existing node carries the requested `--role`.
`manifold generate` plans first (`plan_discover_generation`): a complete folder costs no
load, a partial one regenerates only missing nodes.

Gap: the per-mode hyperparameter whitelist (`store/manifold_schema.py::HYPERPARAMS_BY_MODE`)
is enforced inside the fit transaction, so `manifold fit X --method pca --k-nn 5` fails only
after the model loads.

## Name resolution

| Helper | Callers | Semantics |
|---|---|---|
| `split_ref` | every reference, first | `NAME` or `NS/NAME`; an empty side, `.`/`..` or another separator exits 2 |
| `resolve_manifold`, `existing_folder` | `merge` sources, `transfer`, `pack show\|push\|rm\|clear\|refresh\|export` | `ns/name` pins verbatim; a bare name walks every namespace; miss exits 1, collision `AmbiguousSelectorError` (2) |
| `folder_or_manifold` | `manifold fit` | an existing directory is a folder (needs `manifold.json`), else a reference |
| `split_new` | `extract`, `generate`, `from-template`, `merge` target, `template create` | authoring: bare goes to `local/` |
| `existing_template` | `template show\|score\|rm`; `from-template` | `store.templates.resolve_template`, cross-namespace |

Resolution first runs `store.bundled.materialize_bundled_artifacts` (verbs resolve folders
before any session exists). `split_new` always hands `session.extract` a namespace, so the CLI
never takes `resolve_extraction_target`'s fit-in-place path: a bare `manifold extract
formal.casual` authors `local/formal.casual` beside `default/formal.casual`, making the bare
name ambiguous; qualify it to fit the installed folder. `manifold bake` always writes
`local/<name>`.

`compare` / `why` resolve through `_fold`: `[ns/]name[:variant]` to that model's fitted
tensor file. Variants use their full form (`:raw`, `:role-<slug>`, `:sae-<release>`,
`:from-<src>`), role and transfer variants are checked against the sidecar, and only a 2-node
affine fit folds.

## `serve` startup

Bind check, `effective_config(yaml_vectors=True)`, model check, parse `-S`,
`open_session(roster=True)`, `enable_live_lens`, `select_s_lens`, `preflight_steering`,
`create_app`, `warm_up` (`--compile` only), `enable_live_sae` (not with `--no-web`),
`uvicorn.run` (with `server/ws_stream.py`'s `ws_max_size` and keepalive: pings every 20 s, no
pong timeout). Lens, SAE and warm-up failures are narrated and never block startup.

- Lens sources precede the preflight because `jlens/<word>` and `slens/<word>` atoms resolve
  against selected sources. `preflight_steering` enters and leaves `session.steering(expr)` once
  (loading every atom as a request would); a missing concept exits 1 with an `extract` hint.
- Live lens: a selected R-family source (`lens use`) is never rewritten (incompatible: live off
  with a hint). With none selected, serve tries prepared R-family sources in
  `lens.sources.lens_source_preference_key` order and keeps the first compatible one selected in
  `active.json`. `select_s_lens` applies the same policy to S-family sources that hold `out`,
  validated by `session.lens.has_compatible_input()` (no paging in); there is no live S readout.
- Live SAE: the active source, then other prepared ones (local first), then `best_sae_release`
  (official `google/` repo, Neuronpedia-labelled, `canonical`, curated base over `-all`, name).
  `--no-web` skips discovery and downloads; an explicit `sae use` still restores at construction.

## Reserved short flags

One letter, one long form, tree-wide (`test_short_flags_keep_one_meaning_tree_wide`):

| Flag | Long | Flag | Long | Flag | Long |
|---|---|---|---|---|---|
| `-m` | `--model` | `-c` | `--config` | `-f` | `--force` |
| `-d` | `--device` | `-s` | `--strict` | `-j` | `--json` |
| `-q` | `--quantize` | `-S` | `--steer` | `-v` | `--verbose` |
| `-p` | `--probes` | `-g` | `--grid` | `-y` | `--yes` |
| `-H` | `--host` | `-a` | `--as` | `-o` | `--output` |
| `-P` | `--port` | `-C` | `--cors` | `-k` | below |

- `-k` is double-booked: `--api-key` on `serve`, `--top-k` on `lens top|decompose` and `unit
  read|top|find`; they never share a parser.
- `-s` is always `--strict`, about something per verb: unresolvable steering references
  (config), identical layer coverage (`manifold bake`), probe-roster drift (`transcript run`).
- `pack push` keeps `--private` and `--dry-run` long-only (it is the one publishing verb, and
  `-p`/`-d` mean `--probes`/`--device` elsewhere).

## Verb notes

Semantics the flag help does not carry.

**manifold**
- `extract CONCEPT [CONCEPT]`: one argument is monopolar or a dotted `pos.neg`, two are poles;
  an `ns/` prefix on the first picks the namespace. `--sae` and `--role` are exclusive;
  `--role` bakes into the corpora and reports a `:role-<slug>` name. `-f` re-authors the
  corpora and refits.
- `generate NAME --concepts C...` (at least 2) writes a `pca` discover folder; `--seed` makes
  each node's corpus reproducible from S and its node index; `--role-per-node` uses each slug
  as its node's role substitution. Fitting is a separate step so a flaky generation leaves
  inspectable corpora.
- `fit TARGET`: `--layers` takes indices and `a-b` ranges (`4-8,20`), `workspace` (the 40-90%
  depth band) or `all`, never `sample`. Overrides apply only to discover folders (authored
  exits 2); only the supplied patch reaches `session.fit`, which merges it under the fit lock
  (the CLI never writes `manifold.json`). `-f` bypasses the tensor cache; it is needed
  because `fit` does not re-author corpora, so an unchanged corpus always cache-hits, even
  after a code-level fit change.
- `bake NAME EXPR` takes only namespace-qualified additive scalar terms; `-m` limits it to
  one model, else one tensor per model every component shares.
- `merge NAME SOURCE...` (at least 2) unions discover corpora into an unfitted folder;
  `--method` is required when the sources' fit modes disagree.
- `transfer` maps a fit through a per-layer Procrustes alignment
  (`fit.transfer.load_or_fit_transfer_alignment`) into the target's `:from-<src>` tensor,
  re-baked in the target's Mahalanobis metric.
- `compare REF... -m` reads cached neutral activations only (missing gives `WhitenerError`,
  exit 2): one reference ranks every other foldable fit, two are pairwise, three or more
  print a matrix; `tag:` / `namespace:` / `all` expand and keep a `:variant` suffix. `why`
  histograms the folded direction's per-layer norm in 16 buckets.

**pack** is pure IO except `install`, `search`, `push` and an `hf` refresh (Hub). `rm` needs
`-y` in every namespace including `default/` (likewise `template rm`, `lens rm`, `sae rm`;
nothing prompts). `push --variant` defaults to `raw` (SAE variants opt-in), `clear --variant`
to `all`. `refresh -m` drops only that model's tensors; otherwise `local` is skipped, bundled
re-materializes, `hf` re-pulls. `export gguf` folds a fitted 2-node `pca` manifold to a
control vector.

**template**: `create --contexts FILE` takes a JSON list of `{turns, assistant}` or the
single-turn `{user, assistant}` sugar (`_canonical_context`). `score` is the only template
verb that loads a model, and has no `-c`.

**lens** (corpus specs, sources, estimators: `drowse/lens/AGENTS.md`)
- `fit MODEL`: `--kind r` (default, RelP R-lens), `j` (standard J-lens), `s` or `k` (the S
  family's S- and K-lenses), each at `local:<kind>`; `--sites out,mid` (default both);
  `--corpus FILE` (one document per line, JSON lines contribute `text`) or repeatable
  `--hf-corpus`, else FineWeb-Edu `sample-10BT` (needs `drowse.ai[hf]`). `--prompts` defaults
  to 64 (`lens/key.py::DEFAULT_FIT_DOCUMENTS`). A Hub corpus is pinned once and the pinned
  spec serves both the no-op proof and the stream. Unset numeric flags take the
  `lens.jlens` defaults (`--dim-batch` 256 seeds per transport pass). `--layers` resolves per
  site (the out site of r/j stops below the last block; an explicit layer no requested site
  holds exits 2); an output fit stops at its lowest layer, an input fit at its highest, so a
  band trims work; `sample` is rejected. `--no-split-half` saves the split-half accumulator,
  otherwise the result prints min/median/max split-half per site.
- Fit resume: `-f` restarts from zero, else a matching partial fit resumes; without `-f` a
  fit that would replace a local lens over more documents, or drop a site or layer, exits 2.
  Checkpoints every `--checkpoint-every` documents (25) and ten minutes after the last
  (`lens.jlens.CHECKPOINT_SECONDS`). The first SIGINT/SIGTERM (`stop_on_signals`) finishes the
  document in flight, checkpoints and exits `128 + signal`; a second stops at once.
- `fetch` never loads the model: the payload stays in the Hugging Face cache and Drowse writes,
  and activates in its family's slot, a commit-pinned binding (`rs-r`/`rs-s` bind every site
  the repository holds). `show` / `rm` default to the active R-family source, else the active
  S-family one; removing a binding never touches the cache. `ls` prints each source's kind and
  sites, `show` its layers per site (an S source also its token mask's source; `-j`
  `token_mask`), `use` the family it selected.
- `counts MODEL` never loads the model: it streams the S lens's recorded corpus (or
  `--corpus`/`--hf-corpus`) through the tokenizer, `--documents` 32768, into the model's corpus
  counts (`lens/token_counts.py`; under 1M tokens exits 2; `--remove` drops it). The S mask uses
  them when no neuron-lens catalog is bound.
- `top MODEL PROMPT [--site out|mid]` is `session.lens.read_prompt` on a raw prompt (no chat
  template): each layer's residual read through both families (S footprint z, R readout
  probability), S line first, with depth weight and lens label, after a line naming the S token
  mask's source; `-j` emits `LensRead.to_dict()` per position and `token_mask`. `decompose
  SELECTOR -m` takes the model as a flag; `--dictionary tokens-out|tokens-in|neurons` (default
  `tokens-out`; `neurons` needs no lens, and `unit find` is it with poles and labels); a
  neuron's coefficient is signed, the sign its pole (`+neuron/L8n344 0.42`; `-j` lists
  `neurons` with `unit`, `pole`, `coeff`).

**unit** (reads and catalog: `drowse/units/AGENTS.md`)
- `read UNIT -m` (`neuron/L8n344`, `head/L20h17`) prints a neuron per pole (writes at `out`,
  fires on at `mid`, each an S line then an R line) with scalars, firing and catalog label; a
  head's copies, suppressions, query/key/value inputs (at `L - 1`) and copy scores.
  `--hide-hubs` ranks by the layer-contrastive z (first use of a layer builds its statistics).
- `top TOKEN -m [--by writes|fires] [--layers L]` lists the poles that write (or fire on) the
  token most specifically across layers (default all; `sample` exits 2); `find SELECTOR -m`
  is `lens decompose --dictionary neurons` with poles and labels; `stats UNIT -m` is the
  firing block alone.
- `fetch MODEL [-f]`, `ls [MODEL]` and `rm MODEL -y` never load a model: the catalog binding
  is the model's label source, handled as `lens` and `sae` sources are (`ls` without a model
  lists every catalogued one). The catalog exists for `Qwen/Qwen3-8B` and
  `google/gemma-4-12B` only; any other model is a 404 (exit 1).

**sae**: `train`'s `--layer` defaults to the layer nearest 65% depth within the 40-90% band
(`sae.backend.select_runtime_layer`); `--width` overrides `--expansion`. `fetch` has no
`-d/-q`: it reads only the model config and the release (`sae.backend.resolve_sae_backend`),
and an uncovered `--layer` exits 2.

**experiment**: `fan` alphas are a comma list, `start:stop:step` or `linspace(a, b, n)`, one
loom sibling per grid row with `-S` composed under each. `transcript run` prints
`Transcript.replay` fail-fast (the library's `isolate=True` is not exposed). `naturalness`
prints `fit.naturalness.evaluate` and runs without a probe roster.

## Not in the CLI

No `vector`, `subspace`, `discover` or `tui` verbs, no top-level `transcript`, no `manifold
template` (authoring is `manifold from-template`), none of the flags the root `AGENTS.md` lists
as absent; `--method` only on discover fitting and merging; a namespace is an `ns/` prefix
(`--namespace` exists only as `pack ls`'s filter); no interactive prompts, no `-j` on `rm` /
`use`, no `-d/-q` on verbs that never load weights.

## Tests

`tests/cli/` is CPU-only and model-free; `test_cli_verbs.py` pins the parser (a new verb or
flag usually needs a case there); runner tests use `tests.support.cli` (`FakeSession`,
`install_fake_session`, `poison_session`).
