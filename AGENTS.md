# AGENTS.md

## What this is

`drowse` is a Python library and dual-protocol HTTP server for activation
steering and trait monitoring on Hugging Face causal LMs. One `drowse serve`
port carries OpenAI `/v1/*`, Ollama `/api/*`, the native `/drowse/v1/*` API
(REST, SSE, one WebSocket) and the Svelte dashboard at `/`. Two frontends share
one engine: `DrowseSession` (programmatic) and `drowse serve` (HTTP plus the
dashboard).

Steering signal comes from representation engineering, unified under one
artifact family, the **manifold**: labeled nodes placed on a domain and fitted
to a per-layer, neutral-anchored subspace in the Mahalanobis metric. A
difference-of-means steering vector is the 2-node flat case and a monopolar
concept the 1-node case; `personas` is a 107-node flat fan, `emotions` a
20-node affect manifold, `months` an authored periodic ring. Every steering
term (vectors, poles, `~`/`|` projections, `!` ablations, `%` manifold
positions, lens, SAE and neuron atoms) lowers at generation time to one
per-layer injection whose semantics are `geometry/inject.py::subspace_inject`;
the one exception is a unit mean-ablation (`!neuron/…`, `!head/…`), which
rewrites a projection's input (`UnitOp`) instead of the residual.
Coefficients are per call; the model is never mutated. Four read families
measure the same forward: geometry probes (the whitened `Monitor`), the
Jacobian lens, SAEs and units (MLP neurons and attention heads).

`ARCHITECTURE.md` is the cross-cutting design; each package has an `AGENTS.md`
reference; `README.md` is the user guide (it ships to PyPI).

## Version and releases

`__version__` in `drowse/__init__.py` is the one version (`pyproject.toml`
reads it). Version bumps are a9's call. A push to `main` that changes it makes
`release.yml` tag `v$VERSION`, build, publish `drowse.ai` to PyPI and cut a
GitHub release; a push without a bump is a no-op. A release is one push.

## Commands

```bash
pip install -e ".[dev]"            # editable + pytest/ruff/pyright/build
# extras: sae (SAELens 6.x), hf (datasets: streamed lens/SAE corpora), gguf,
#         cuda + flash (Linux/NVIDIA), research, notebook

drowse serve [MODEL] [-S EXPR] [-p CATS...] [-H HOST] [-P PORT] [-k KEY] [-C ORIGIN]
             [--no-web] [--no-dls] [--compile] [--cuda-graphs] [--top-k-alts N] [-c PATH] [-s]
drowse manifold extract CONCEPT [CONCEPT] -m MODEL [--kind abstract|concrete|custom] [--system T]
                        [--role SLUG | --sae RELEASE] [--no-dls] [-f]   # ns/ prefix authors outside local/
drowse manifold generate NAME --concepts C... -m MODEL [--kind K] [--system T] [--samples-per-prompt K]
                         [--seed S] [--role-per-node] [--description TEXT] [-f]
drowse manifold from-template TEMPLATE [--name N] [--fit-mode pca|spectral|auto] [--max-dim N]
                              [--var-threshold T] [--description TEXT] [-f]
drowse manifold fit TARGET -m MODEL [--layers L1,L2-L3|workspace|all] [--sae REL] [--method pca|spectral|auto]
                    [--max-dim N] [--min-dim N] [--var-threshold T] [--k-nn K] [--bandwidth S]
                    [--max-subspace-dim R] [--smoothing auto|0|LAMBDA] [--persistence-frac F] [--no-dls] [-f]
drowse manifold bake NAME "0.3 ns/a + 0.5 ns/b" [-m MODEL] [-s] [-f]   # offline, writes local/NAME
drowse manifold merge NAME SRC... [--method M] [--description TEXT] [-f]
drowse manifold transfer NAME --from SRC_MODEL --to TGT_MODEL [-f] [-j]
drowse manifold compare CONCEPT... -m MODEL [-v] [-j]
drowse manifold why CONCEPT -m MODEL [-j]
drowse pack ls [--namespace NS] [-j|-v] | show NAME [-j] | search [QUERY] [-j|-v]
drowse pack install TARGET [-a NS/NAME] [-f]     # ns/name[@rev] on the Hub, a folder, or a .drowse
drowse pack push SEL [-a OWNER/NAME] [-m MODEL] [--variant raw|sae|all] [--private] [--dry-run]
drowse pack rm SEL -y | clear SEL [-m MODEL] [--variant V] | refresh SEL [-m MODEL]
drowse pack export gguf NAME [-m MODEL] [-o PATH] [--model-hint HINT] | export archive NAME [-o FILE]
drowse experiment fan MODEL PROMPT -g CONCEPT=A,B,C [-S EXPR] [--max-tokens N]
drowse experiment transcript run PATH.yaml [MODEL] [--max-tokens N]
drowse experiment naturalness MODEL PROMPT --manifold NAME -S EXPR [--compare-linear]
drowse template create NAME --slot TOKEN --values V... --contexts PATH [--description TEXT] [-f]
drowse template ls | show NAME | rm NAME [-y] | score NAME -m MODEL [-S EXPR] [--by sum|mean]
drowse lens fit MODEL [--kind r|j|s|k] [--sites out,mid] [--corpus FILE | --hf-corpus REPO[:CFG][@REV][,K=V...]...]
                [--prompts N] [--seq-len T] [--dim-batch K] [--checkpoint-every N] [--no-split-half] [--layers L] [-f]
drowse lens fetch MODEL [rs-r|rs-s|neuronpedia|workspace-r|workspace-j] [--repo R] [--revision REV] [-f]
drowse lens counts MODEL [--corpus FILE | --hf-corpus SPEC...] [--documents N] [--remove] [-f]   # S mask counts
drowse lens ls MODEL | show MODEL [SRC] | use MODEL SRC | rm MODEL [SRC] [-y]
drowse lens top MODEL "PROMPT" [-k K] [--layers L|sample|workspace|all] [--position P] [--site out|mid]
drowse lens decompose SELECTOR -m MODEL [-k K] [--dictionary tokens-out|tokens-in|neurons] [--layers L]
drowse sae train MODEL NAME [--corpus FILE] [--layer L] [--tokens N] [...training knobs]
drowse sae fetch MODEL saelens:RELEASE [--layer L] [--revision REV]
drowse sae ls MODEL | show MODEL [SRC] | use MODEL SRC | rm MODEL [SRC] [-y]
drowse config show [-c PATH...] [--no-default] [-m MODEL] | validate FILE
drowse unit read UNIT -m MODEL [-k K] [--hide-hubs] | stats UNIT -m MODEL      # neuron/L8n344, head/L20h17
drowse unit top TOKEN -m MODEL [--by writes|fires] [--layers L] [-k K] | find SELECTOR -m MODEL [-k K] [--layers L]
drowse unit fetch MODEL [-f] | ls [MODEL] | rm MODEL -y        # neuron-lens catalog: Qwen3-8B, Gemma 4 12B

pytest -q -m "not gpu"             # the CI suite (CPU)
pytest -q -n auto -m "not perf"    # everything this machine can run
pytest -m perf -p no:xdist         # the throughput gates, serially
ruff check . && pyright
python scripts/generate_webui_types.py [--check]          # PYTHONPATH=. in a worktree
cd webui && npm ci && npm run check && npm run build      # Node 24.x; commit drowse/web/dist
python scripts/migrate_lens_v7.py --home DIR [--dry-run]  # pre-5.4 lens homes
python scripts/regenerate_bundled.py [TARGETS...]         # bundled corpora + neutral baseline
```

Model-loading verbs also take `-d/--device`, `-q/--quantize 4bit|8bit` and
`--dtype`; most read verbs take `-j/--json`. Precedence is CLI > YAML > runner
default, and `-c` composes `$DROWSE_HOME/config.yaml` first. Details:
`drowse/cli/AGENTS.md`.

## Settled designs

Mahalanobis-only geometry (no Euclidean fallback in activation space); one
extraction method (difference of means, generalized to whitened PCA over
nodes); one injection kernel; coefficients as the only steering knob besides
`--no-dls` (no per-term gain). Re-open these only with new evidence.

## Package map

Packages are layered lowest first; a lower layer that needs a higher object
takes it through a narrow `Protocol` and injection. `drowse/__init__.py` is the
public facade.

| Layer | Package | Owns |
|---|---|---|
| 0 | `errors.py`, `naming.py`, `events.py` | the error family; canonical names, `RESERVED_NAMESPACES` (`jlens`, `slens`, `sae`, `neuron`, `head`), unit references (`L8n344` / `L20h17`); the synchronous `EventBus` |
| 1 | `base/` | torch-free primitives: paths and `$DROWSE_HOME`, atomic writes and artifact locks, shards, staging, Hub access, `SourceFamily`, `ExclusiveGuard`, LRU |
| 2 | `geometry/` | pure tensor math: frames, domains, RBF, the kernel, whitener, topology, fold/`Profile`, transfer |
| 3 | `model/` | loading, fingerprints, layer discovery, logits path, layer selectors, chat traits, roles, scene grammar, compile/StaticCache |
| 4 | `store/` | manifold folders and schema, selectors and the bare-atom ladder, templates, bundled materialization, archive, install, Hub, GGUF export, neutral/alignment caches |
| 5 | `steering.grammar`, `.terms`, `.triggers` | the expression language and its IR |
| 6 | `readout/` | instrument contract, `Monitor`/`GeometryInstrument`, capture planner, roster, measurement envelope |
| 7 | `lens/`, `sae/`, `units/` | the Jacobian lens, SAE and units families |
| 8 | `steering.composer`, `.hooks` | scopes, resolution, composition, lowering, gains, hooks |
| 9 | `generation/` | decode loop and sampler, rendering, prefix cache, compile routing, `ReadPlane`, results, scoring |
| 10 | `fit/` | the extraction pipeline, corpora, capture, publish, preflight, neutral compute, bake, transfer, naturalness |
| 11 | `loom/` | the conversation tree, recipes and `compose_call`, cast, ops, token replay, diff, filter, transcripts |
| 12 | `session/` | `DrowseSession`, the engine, batch runs, registry, neutral baseline, extraction glue |
| 13 | `cli/`, `server/`, `web/`, `notebook/` | frontends |

`store` sits below the grammar because `parse_expr` resolves bare atoms
against installed manifolds; `steering` spans two layers (grammar low, composer
and hooks above `readout` and the instrument families). `drowse/data/` holds
the bundled manifolds, `baseline_prompts.json` and `neutral_statements.json`.
`webui/` is documented by `drowse/web/AGENTS.md`; its build is committed under
`drowse/web/dist/`.

## Invariants

**Mahalanobis only, typed frames.** Every activation-space metric (fit, share,
synthesis, `~`/`|`, monitor reads, compare, transfer, the lens decompositions)
needs a `LayerWhitener` covering every touched layer and raises `WhitenerError`
otherwise: on real LMs a few massive-activation channels dominate the
Euclidean metric. A `LayerSubspace`'s `basis` rows are write vectors
(Mahalanobis-orthonormal) and its `read` rows their dual `Σ⁻¹` covectors, so a
2-node push is DiM, a full `!` is LEACE and the monitor reads the LDA score;
never write along a read row or read with a write row (ARCHITECTURE §5.3).
Lens atoms are reads and steer along their raised vectors `Σu`. Neutral
activations are cached fp32 (fp16 overflows late layers).

**Exclusive GPU.** One reentrant `_gen_lock` serializes all model work (MPS's
command buffer is not thread-safe); a busy section raises
`ConcurrentGenerationError`/`ConcurrentExtractionError` (409) instead of
queueing. Artifact-producing work refuses inside a steering scope
(`SteeringComposer.assert_unsteered`), since the hooks would bake into the
artifact.

**One path per job.** One extraction pipeline, one kernel, one grammar for
Python, YAML, HTTP, CLI and recipes, one sampler (`generation/decode.py`), one
error classifier (`server/errors.py::describe_error`), one layer-selector parser
(`model/layers.py`), one pooling position (`model/tokens.py::last_content_index`),
one measurement envelope. Replace the old path when adding a new one; breaking
compatibility is fine.

**Errors.** Exceptions subclass `DrowseError` (keeping their stdlib base);
`user_message()` returns `(http_status, text)`. Caller mistakes raise
`RequestError` (400) or a subclass where validated; post-parse composition
failures are `SteeringCompositionError` (422); a bare `ValueError` is a bug and
answers a scrubbed 500. The wire `code` is the class name; the CLI maps 404 → 1,
other 4xx → 2, 5xx → 1.

**Persisted formats.** Readers accept the current `format_version`, require the
fields they use and ignore the rest, so dropping a field never strands a file;
derived caches recompute instead of migrating. Cache identity (corpus, capture
and model digests, `MANIFOLD_FIT_POLICY_VERSION`) decides when a fit is stale;
files carry no integrity hashes. Bumping `MANIFOLD_FIT_POLICY_VERSION`
(`geometry/manifold.py`) refits everything. Every `SamplingConfig` field is
persisted in recipes. Pre-5.4 lens homes need `scripts/migrate_lens_v7.py`.

**drowse.ai.** [stdlogits/drowse.ai](https://github.com/stdlogits/drowse.ai) pins
this repo as a submodule, runs `webui/src` with a WebGPU runtime (the
`RuntimeClient` contract in `webui/src/lib/runtime/contracts.ts` and the
`@runtime-*` aliases), reimplements the grammar, sampler, lowering and fit in
the browser, and calls a few Python APIs (`lens.store.save_lens_at`,
`units.reads.UnitReader.population_artifacts`/`null_basis`/`head_kept_ids`,
`tests/browser_drowse_archive_bridge.py`). There are no goldens here: a semantic
change is an interface change drowse.ai catches up to when it moves its pin,
computing parity from the pinned Python.

**Performance.** `tests/gpu/test_smoke.py` gates steered ≥ 85% of vanilla tok/s,
ablation ≥ 80% and the served dashboard ≥ 50% (about 63% for SmolLM2 on MPS;
the live R readout costs about 3 ms of a dashboard token there and is
bandwidth-bound on large models). Hooks write in place, never sync the host and
allocate nothing beyond their own tensors; generation runs under
`torch.inference_mode()`; fit-time math is fp32; one device→host transfer per
decoded token; the always-active affine case is an analytic fast path
(`AffineOp`) and only a static pure push rides the compiled graph; top-p runs
over `torch.topk`; resident artifacts upload packed (`geometry/packing.py`),
because thousands of small device tensors fragment the MPS heap.

**MPS.** Fit-time geometry runs on CPU in fp32; `torch.mps.empty_cache()`
between extraction batches; `torch.mps.synchronize()` at the end of every
decode. Metal reports queue exhaustion asynchronously and silently zeroes work,
so long MPS loops drain the queue and guard against zero rows (`lens/fit.py`).

**Testing.** Most tests run on CPU against a real `DrowseSession` on a tiny
generated checkpoint (`tiny_session`); route tests that need no behavior use
`tests.support.server.mock_session`. The GPU suite is one end-to-end smoke plus
the throughput gates (`DROWSE_TEST_MODEL`, default SmolLM2-360M-Instruct). Add a
regression test when a bug could quietly come back. Tests never touch the real
home. CI runs ruff, pyright, the CPU suite in four parallel shards and the webui
build with a dist-drift check; `main` takes direct pushes.

## State and environment

All state lives under `$DROWSE_HOME` (default `~/.drowse`): `config.yaml`,
prompt/neutral overrides, `templates/`, `manifolds/<ns>/<name>/` and
`models/<safe_model_id>/` (neutral activations, alignments, capture cache,
`lens/`, `sae/`, `units/`: the contrast cache and the catalog binding). The
full tree is in `drowse/store/AGENTS.md`; the lens, SAE and units subtrees in
their packages. No conversation lives there: the dashboard keeps its library
in the browser (IndexedDB, autosaved between generations) and as backup
downloads, and Python saves only to an explicit `LoomTree.save(path)`.

| Variable | Effect |
|---|---|
| `DROWSE_HOME` | relocates all state (there is no `cache_dir=`) |
| `DROWSE_TRUST_REMOTE_CODE` | the only way to let the CLI/server run a model repo's Python |
| `DROWSE_API_KEY` | bearer key when `--api-key` is not given |
| `DROWSE_STRICT_MODEL` | OpenAI/Ollama 404 on a model-name mismatch |
| `DROWSE_MAX_REQUEST_BYTES` | request body cap (default 64 MiB) |
| `DROWSE_MANIFOLD_CAPTURE_CACHE_GB` | fit capture cache bound (default 8) |
| `DROWSE_TEST_MODEL`, `DROWSE_TEST_LIVE_TOKENIZERS` | test knobs (`tests/AGENTS.md`) |
