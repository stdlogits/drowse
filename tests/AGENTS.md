# tests/AGENTS.md

The pytest suite. Most of it runs on CPU against real objects: a real
`DrowseSession` on a tiny generated checkpoint, or one real collaborator built
with the explicit dependencies a session wires. GPU tests load a real model,
the invariant tests scan the source tree, and the browser parity tests pin the
goldens the drowse.ai browser runtime replays.

## Running

| Command | Runs |
|---|---|
| `pytest -q -m "not gpu" -n auto` | The CPU suite, about 50 s on Apple Silicon. CI runs exactly this on Python 3.13. |
| `pytest -q -n auto -m "not perf"` | Everything this machine can run, GPU included (about 80 s on Apple Silicon with MPS). |
| `pytest -m perf -p no:xdist` | The four throughput gates, serially, on an otherwise idle machine. |
| `pytest tests/gpu/test_smoke.py` | Real-model smoke tests only. |

- `--strict-markers` is in `addopts` and `xfail_strict = true`: a new marker
  must be registered in `pyproject.toml`, and an `xfail` that passes fails the run.
- xdist is opt-in (`-n auto`). Inside an xdist worker the collection hook in
  `conftest.py` skips every `perf` test: worker contention makes throughput
  numbers meaningless (a shared-MPS run fails them, a serial rerun passes).
- `tests/` is in pyright's scope with `reportMissingParameterType = "error"`,
  so annotate test and fixture parameters (`tiny_session: Any`). Ruff's `T20`
  (no `print`) applies here too.

| Variable | Effect |
|---|---|
| `DROWSE_TEST_MODEL` | The model GPU tests load (`tests/support/gpu.py::gpu_model_id`); default `HuggingFaceTB/SmolLM2-360M-Instruct`, public. Any compatible chat model works. |
| `DROWSE_TEST_LIVE_TOKENIZERS=1` | Lets `tests/model/test_role_template_drift.py` download the per-family tokenizers (some gated) instead of reading only the local HF cache. Run it before a release. |

Never point a test at the real home: a session-scoped autouse fixture sets a
scratch `DROWSE_HOME` for every worker, so nothing reaches `~/.drowse`.

## Layout

`tests/<pkg>/` mirrors `drowse/<pkg>/`, and a test lives under the package
whose behavior it pins. Every test directory is a package (`__init__.py`). A test
that reads a repo file finds the root as `Path(__file__).resolve().parents[2]`,
so keep test modules exactly one directory deep.

| Directory | Covers |
|---|---|
| `base/` | `drowse/base` (atomic writes, integrity, shards, staging, paths, source registry) plus `events` and `naming` |
| `geometry/` | domains, manifold math, topology, whitener/Mahalanobis, profile, fold, projection, `synthesize_subspace`, frame transport, GGUF I/O |
| `model/` | loading and checkpoint trust, architecture dispatch, chat render cache, role templates, scene grammar, static cache, fingerprint, numerics |
| `store/` | manifold folders and tensors, selectors, templates, bundled materialization, archive, hub, install, alignment, neutral cache |
| `steering/` | expression grammar, triggers, steering scopes, hooks, installation, subspace/manifold steering, ablation |
| `readout/` | monitor, probe roster, read planner, captures, measurements envelope, instrument types, geometry replay, `score_hidden` |
| `lens/`, `sae/` | lens math, the transport estimator (against the autograd oracle), RelP, store, sources, corpus, lifecycle, runtime, atoms, decompose, `scripts/migrate_lens_v7.py`; SAE backend, runtime, artifacts, descriptions |
| `units/` | neutral unit statistics (against a hand computation), the tap, unit probes and gates through a tiny session, replay, unit ablation against hand-patched forwards, neuron pushes, refusals; static unit reads, `top` and `find` against hand computations with hand-built R/S lenses, the neuron-lens catalog over a monkeypatched Hub with tiny parquets, the unit routes |
| `generation/` | decode loop, sampling, logits, prefix cache, render, results, scoring, token payloads |
| `fit/` | extraction pipeline, capture, bake, monopolar fits, preflight, naturalness, transfer alignment |
| `loom/` | tree, ops, diff, filter, transcript, recipe overrides, token rows, token replay, joint logprobs, concurrency |
| `session/` | the session facade and its collaborators: registry, engine, batch, cast, conversation, extraction glue, bake, the tiny-session contract |
| `cli/`, `server/`, `web/`, `notebook/` | verb dispatch, flags, config files; OpenAI/Ollama/native routes, WS, streaming, request errors, HTTP security; dashboard mount; plotly helpers |
| `gpu/` | real-model tests (see Markers) |
| `invariants/` | source-tree scans (see Invariant tests) |
| `browser/` | `test_browser_parity.py`, the Python side of the browser goldens |
| `fixtures/`, `browser_drowse_archive_bridge.py` | seam files at fixed paths (see Browser seam) |
| `support/` | shared helpers; `conftest.py` holds the fixtures |

## Markers

| Marker | Meaning | Where |
|---|---|---|
| `gpu` | Needs CUDA or MPS; may download weights. CI deselects it. | `tests/gpu/test_smoke.py` (one extract → steer → generate smoke and the perf gates); the `mps` params of `tests/model/test_model_numerics.py` and `tests/generation/test_generation_metadata.py`; `tests/generation/test_scoring.py::test_real_forward_score_choices_distribution`. A test that requests `gpu_session` is marked automatically. |
| `perf` | Wall-clock throughput; serial only. | `tests/gpu/test_smoke.py`: `test_throughput_regression` (steered ≥ 85% of vanilla tok/s), `test_throughput_with_ablation` (ablation plus additive ≥ 80%; the extra matmul per ablation layer is why the bar is lower) and `test_throughput_with_unit_ablation` (8 neurons and 2 heads mean-ablated ≥ 80%) and `test_throughput_with_dashboard_instruments` (the served dashboard's decode ≥ 50%: the default roster scored per token with subspace coordinates, the live R readout over a micro-fit lens's every layer, eight alternatives, a loom row and a serialized WebSocket frame per token; about 63% on SmolLM2/MPS). All are also `gpu`. |

GPU modules load with `device="auto"` (cuda > mps > cpu) and skip without CUDA
or MPS. `load_or_skip_inaccessible` skips when loading fails on access (a gated
repo, or a 401/403 auth error anywhere in the cause chain) and re-raises
anything else, so OOM or shape errors still fail. The conftest `gpu_session`
(session scope, `probes=[]`) is available for a GPU test that needs a session.
Real-device behavior is otherwise covered by the tiny sessions on CPU; add a
GPU test only for something a CPU run cannot show (device numerics, MPS
kernels, throughput).

Two MPS parity tests (`tests/geometry/test_frame_transport.py::test_two_d_pairs_on_mps_match_cpu`,
`tests/geometry/test_synthesize_subspace.py::test_provider_fragments_follow_runtime_layer_device`)
skip without MPS but are not marked `gpu`, so `-m "not gpu"` still runs them on Apple Silicon.

## Fixtures (`conftest.py`)

| Fixture | Scope | Gives |
|---|---|---|
| `isolated_home` | function | A fresh `DROWSE_HOME` with selector caches reset. Use it whenever a test writes artifacts; a module opts in wholesale with `pytestmark = pytest.mark.usefixtures("isolated_home")`. |
| `tiny_checkpoint` | session | The llama-family tiny checkpoint, written once per worker. |
| `make_tiny_session` | function | A `TinySessionFactory`; every session and `ScriptedReplies` hook it creates is closed at teardown. |
| `tiny_session` | function | One warm CPU session, `probes=[]`. |
| `app_client` | function | `app_client(session, **create_app_kwargs)` returns a started `TestClient` over the real FastAPI app (lifespan runs); closed at teardown. |
| `gpu_session` | session | The real GPU session on `DROWSE_TEST_MODEL`; skips without a GPU. |

## Tiny sessions

`tests/support/tiny_models.py` and `tests/support/sessions.py`. The default for
any test that runs the session.

- **Model.** A genuine `transformers` architecture from `from_config`: 4 layers
  × 32 hidden, random weights, eager attention, in three chat families
  (`FAMILIES`: `llama`, `qwen3`, `gemma3_text`), each with its real chat wire
  format. The loader, scene grammar, role substitution, capture, hooks and
  decode loop run unmodified; only the weights are meaningless.
- **Tokenizer.** Byte-level BPE with no merges: one token per UTF-8 byte plus
  the family's special tokens. It round-trips any text, and a non-ASCII
  character spans several tokens, which exercises the decode loop's
  partial-UTF-8 buffering.
- **Warmth.** `make_tiny_session()` (`warm=True`) builds the whitener before
  returning, so steering scopes, `~`/`|` projections and probes work at once;
  `warm=False` exercises the cold, lazy paths. Other keyword arguments go to
  `DrowseSession.from_pretrained` (defaults `device="cpu"`, `probes=[]`). The
  neutral activations are computed once per process (about 0.3 s) and copied
  into each new home, so per-test homes hit the cache.
- **Deterministic output.** `make_tiny_session.scripted(replies)` returns
  `(session, script)`. `ScriptedReplies` is a forward hook that rewrites only
  the last position's logits so the model spells the reply after each
  non-system turn header, then end-of-turn. Hidden states stay the model's
  own, so steering, probes and capture are real. `replies` is a string, a
  per-turn sequence (the last entry repeats) or `(label, prompt_text) -> reply`;
  `script.prompts` records every rendered prompt that started from an empty
  cache. It tracks sequences by KV-cache length, so a prefix-cache entry reused
  after unrelated forwards is not tracked.
- **Helpers.** `register_vectors(session, *names)` registers a name-seeded
  random unit direction per layer through `session.registry.put_profile`;
  `forward_logits(session, text)` returns last-position logits of one raw
  forward under whatever hooks are attached (it shows whether a steering scope
  is acting); `baseline_corpus(response)` builds one pole corpus;
  `drowse_home(path)` switches homes for a block. Another family is
  `TinyCheckpoint.write(root, family)` plus `TinySessionFactory(checkpoint, seed_dir)`.

## `tests/support`

Test modules never import from other test modules; a helper two modules share
lives here. Reach for, in order: a tiny session; a rig that builds the one real
collaborator under test with the dependencies a session wires; a fake only at
the model or I/O boundary. Do not build `DrowseSession.__new__` skeletons or
set session privates.

| Module | Provides |
|---|---|
| `composer.py` | `SteeringRig`: a real `SteeringComposer` with its `ArtifactRegistry`, `NeutralBaseline` and `SteeringManager`, no model; `attach_engine()` adds a `GenerationEngine` on the rig's lock. |
| `read_plane.py` | `ReadRig`: a real `ReadPlane`, the four instruments (units over a model-less runtime) and the `ProbeRoster` over a swappable `CaptureSlot` (any instrument can be replaced by a fake); `finalize` runs the finalize reads through `finalize_generation` as the engine does; `gate_refs(expr)`; `fake_readout_family`; `geometry_instrument`. |
| `engine.py` | `engine_over(host, ...)`: a `GenerationEngine` with an inert runtime, for call composition without a model. |
| `lens.py` | `LensHarness` (a `LensInstrument` on the toy model, recording `on_change` and exclusive sections), `lens_instrument`/`lens_kwargs` (constructor arguments with inert defaults), `FakeCapture`. |
| `sae.py` | `MockSaeBackend` (identity encode/decode by default), `sae_instrument` (a real `SaeInstrument` + `SaeRuntime`; registry effects land on `.effects`), `make_resident` (installs a residency without the load path). |
| `units.py` | `inert_units_instrument` (a real `UnitsInstrument` over a model-less runtime, for rigs), `random_unit_stats(spec)`, `patched_forward_logits(model, ids, {module: patch})` (the hand-patched forward unit hooks are checked against), `GEMMA4_MIXED_HEADS` (a tiny Gemma 4 chat family with 8- and 16-wide heads, for `TinyCheckpoint`). |
| `replay.py` | `ReplayRig` (a real `TokenReplayer` over the toy model and a real `LoomTree`, recording renders, scopes, exclusive sections and capture forwards; hand `rig.replay` to the instrument under test), `unwired_replay` (fails loudly if called). |
| `geometry.py` | `stub_geometry_instrument(stub)`: a real `GeometryInstrument` reading a hand-built stub's attributes per call. For the remaining stub suites; new tests use `tiny_session`. |
| `capture.py` | `install_capture` (swaps `fit.capture._encode_and_capture_all_batch` for a synthetic per-row encoder and pins the baseline prompts to one), `StubHandle` (identity layers + a covering synthetic whitener), for model-free `ManifoldExtractionPipeline` tests. |
| `extraction.py` | `make_extraction` → `ExtractionRig`: a real `Extraction` over recording fakes; `folded_vector_manifold`. |
| `whitener.py` | `synthetic_means`, `synthetic_whitener`, `rogue_whitener`, `isotropic_whitener`. Activation-space fits, projections and probe reads require a covering whitener, so model-free tests build one here; the returned means are the dict the whitener centers on. |
| `manifolds.py` | `loop_manifold`, `open_manifold`, `disk_manifold`: curved fits through circle centroids with a 20.0 common-mode offset (without it the kernel's norm cap fires); `stamp_current_geometry`, `euclidean_shares`. |
| `artifacts.py` | `author_manifold` (a two-node discover folder under `local/`, fitted for `MODEL`), `write_fitted_manifold`. |
| `decode.py` | `FakeLogitsModel`/`make_logits_model`, `scripted_model`/`scripted_logits`, `NoCacheModel`, `StopTokenizer`, `CharTokenizer`, for decode-loop, scoring, naturalness and joint-logprob tests. |
| `jlens_toys.py` | `frozen_toy(n_layers)`: a genuine tiny llama (`TOY_D` wide, `TOY_VOCAB` tokens, weights re-drawn) the lens transport fits, with its own `CharTokenizer`. |
| `lens_oracle.py` | The autograd lens oracle: `oracle_estimate`/`oracle_lens` compute every kind and site by autograd through the model's own modules, with the RelP rules as derivative patches (`relp_backward_rules`). The reference the transport is tested against. |
| `browser_units.py` | The hook ABI v5 goldens (`tests/fixtures/browser/units-tiny-models-v1.json` and `<name>-<arch>-v1.json`): the `units-tiny-models-v1` fixture (gemma3_text and qwen3, 4 layers × 64, weights from a portable integer generator that drowse.ai rebuilds bit for bit, every linear q4-exact), `fixture(arch)` (a session in a scratch home with the fixture's neutral corpus, token counts, hand-built R/S lenses and band), one `build_<golden>` per file, the validator (finite values, top-k boundary margins against the call one deeper, decisive greedy picks, peak-row margins, size caps; near-tie runs declared), `golden_problems`/`compare`, and `write_goldens`. Regenerate with `PYTHONPATH=. python -m tests.support.browser_units --write`. |
| `blocks.py` | `tiny_dense_model(model_type)`: a float64 tiny model of each dense family (`DENSE_FAMILIES`) with every weight, norm and scalar re-drawn and windows that bind; `default_init_model(model_type, seed=, dtype=)`: the same architecture at its default initialization in a low precision (bf16), the rounding a verification must not take for a term; `spec_forward(spec, ids)`: mids, block outputs and logits from a `ModelSpec` alone, the oracle for `BlockSpec` and transport layer maps. |
| `server.py` | `mock_session`, `script_generate`, `StubSession` (below). |
| `cli.py` | `FakeSession`, `install_fake_session` (patches `DrowseSession.from_pretrained`, the CLI's one construction seam, so the real narration, YAML defaults and close-on-exit run), `poison_session` (any model load raises). |
| `chat_templates.py` | Per-family templates emitting each family's exact role-header bytes, and `FakeTokenizer` (Jinja render, whitespace tokens). |
| `streams.py`, `tokens.py` | `FakeGenerationStream`; `emit_token`, which drives an `on_token` consumer the way the engine's token tap does. |
| `gpu.py` | `gpu_model_id`, `load_or_skip_inaccessible`. |

**Route tests.** `mock_session(**overrides)` is
`create_autospec(DrowseSession, instance=True)` shaped as an idle session: idle
tree, the four instrument faces idle with no probes, the Monitor, a working
event bus. A name the class does not define raises `AttributeError` and every
method call is signature-checked, so a route that calls a removed or renamed
method fails. Overrides apply last; `script_generate(session, tokens)` makes
`generate` stream tokens and finish. `StubSession` wraps a real `LoomTree`
with scripted generation for the loom routes and the WS request lifecycle. A
route test that needs real behavior uses `app_client(tiny_session)`.

## Browser seam

drowse.ai pins this repository as a submodule at `drowse/` and reads these
files from it, so their paths are part of its interface:

| Path | Consumers |
|---|---|
| `tests/fixtures/browser/*` | `tests/browser/test_browser_parity.py`, which recomputes each golden with the Python engine; drowse.ai's runtime and fitting-wasm tests replay the same files. |
| `tests/fixtures/browser_steering_expression.json` | The parity test (accepted cases parse and match `referenced_selectors`; rejected ones raise `SteeringExprError`) and drowse.ai's TypeScript parser test. |
| `tests/browser_drowse_archive_bridge.py` | drowse.ai's archive round-trip test runs `export\|install FIXTURE HOME ARCHIVE`. It calls `store.manifold_authoring.create_discover_manifold_folder`, `store.archive.export_drowse_archive`, `store.install.install_manifold` and `store.manifold_folder.ManifoldFolder`, so those signatures are seam too. |
| `tests/fixtures/loom_tree_v2.json` | `webui/scripts/conversation-snapshot.test.mjs` (`npm run check`): a `tree_format` 2 tree save. No Python test reads it; keep it in step with the tree format. |
| `tests/fixtures/token_rows.json` | `tests/loom/test_loom.py` and `webui/scripts/conversation-snapshot.test.mjs`: the token rows (`accepted`/`refused`) and authored nodes with their `prompt_capture` (`authored`) both readers accept and refuse; a shape change edits it on both sides at once. |

- Never rename, move or reformat these files. The goldens are byte-frozen: a
  diff under `tests/fixtures/browser/` is a seam change, never a side effect of
  a refactor. A refactor that would need a golden regenerated is not a refactor.
- Python is the source of truth. A deliberate semantic change regenerates the
  affected golden in the same change, and drowse.ai's runtime has to match
  before its submodule pin moves past it. Each golden's file name carries its
  `schemaVersion` (`-v1`, `post-block-rank-one-v2`).
- The hook ABI v5 goldens (units, token anatomy, the `mid` site, the S lens)
  are one file per architecture, each at most 256 KiB and 2 MiB together, all
  built by `tests/support/browser_units.py` and rebuilt by
  `test_units_golden_matches_python` within each file's own `tolerance`
  block. The model file holds integers and hashes only and must regenerate
  byte for byte. A deliberate engine change that moves them regenerates the
  set (`PYTHONPATH=. python -m tests.support.browser_units --write`). The
  validator refuses a top-k boundary without a 1e-3 margin (each ranked
  list against the same call one deeper; probability lists by `ln p`), a
  greedy pick without one, an unmargined peak row a list or trace shows and
  a one-signed traced neuron; a read moves to another `k` (recorded) to
  clear its boundaries. Writing, every margin also clears the
  cross-platform slack (twice the `python` tolerance); the parity test
  rebuilds on the file's recorded choices (`build_golden(..., replay=)`:
  its `k`s, `top`s, knobs and gate thresholds), so no platform re-decides
  near a threshold. Near-ties inside a list are runs in the file's
  `declared_ties` (`{path, key, groups}`): the comparison matches a run's
  entries by identity and compares every field, order kept between runs.
- Add a case to `browser_steering_expression.json` only once drowse.ai's
  TypeScript parser agrees: both sides run the same table.
- `tests/fixtures/steering_ir.json` is the parsed-IR golden, Python-only for
  now (`tests/steering/test_steering_ir.py`): per accepted expression, the
  ordered terms (key, kind, coefficient or along/onto, operands, position,
  layer scope, trigger text and fields, term text) and the canonical
  `format_expr` string; per rejected one, the error class. It carries its own
  manifold registry, installed into a fresh home before parsing. It is the
  target drowse.ai's parser will match; once it does, the browser fixture can
  grow to this scope. After a deliberate grammar change, regenerate it with
  `PYTHONPATH=. python -m tests.steering.test_steering_ir` (inputs stay,
  outputs recompute).
- The measurement envelope wire shape is part of the same seam
  (`measurement-envelope-v2.json`, plus the exact key sets in
  `tests/readout/test_measurements_envelope.py`).
- `steering-term-layers-v1.json` pins `POST .../steering/validate` (without `parts`) over a
  model-free composer: each term's key, fitted layers, mean-1 share and clause selection, the
  refusals composition answers in band, and a curved `%` term's hook share
  (`_manifold_layer_shares`, mean 1 over the clause's layers where the response's share is mean 1
  over the fit) with the lowered plan's curved groups (each layer's `along · share · 2` and
  `min(1, onto · share · 0.5)`, so the gains are pinned too). Its inputs (whiteners, profile directions, a flat and a curved manifold, the
  cases' expressions) stay; after a deliberate change regenerate the outputs with
  `PYTHONPATH=. python -m tests.browser.test_browser_parity`.

## Invariant tests

| Test | Enforces |
|---|---|
| `invariants/test_webui_types_parity.py` | `webui/src/lib/types.gen.ts` equals a fresh render (regenerate with `python scripts/generate_webui_types.py`); every native JSON route declares a response schema; native paths are kebab-case. |

Contract pins elsewhere: `readout/test_measurements_envelope.py` (exact
envelope key sets), `server/test_request_errors.py` (a typed request error
keeps its status, text and class-name code on every HTTP, SSE and polled
surface; a bare `ValueError` is a scrubbed, logged 500), `model/test_architecture_dispatch.py` (zero-download structural
coverage of `GUARANTEED_ARCHS`: layer discovery, residual capture and the
steering write path; append a `model_type` to widen it),
`store/test_bundled_templates.py` (the bootstrap materializes templates before
manifolds), and `session/test_tiny_session.py` (the tiny-session contract the
rest of the suite builds on).

## Writing tests

- Add a regression test when a bug could quietly come back; skip it when the fix
  makes the bug unrepresentable or another test already covers it.
- Write artifacts only under `isolated_home` or a `tmp_path` home.
- A change to a wire record, an error status or a native route shape fails the
  pins above by design; update the pin in the same change.
- Mark wall-clock throughput assertions `perf`; they cannot run under xdist.
