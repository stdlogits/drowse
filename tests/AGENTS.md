# tests/AGENTS.md

The pytest suite. Most of it runs on CPU against real objects: a real
`DrowseSession` on a tiny generated checkpoint, or one real collaborator built
with the dependencies a session wires. The GPU suite is one real-model smoke
plus the throughput gates.

## Running

| Command | Runs |
|---|---|
| `pytest -q -m "not gpu" -n auto` | The CPU suite, about 40 s on Apple Silicon. CI runs it in four shards. |
| `pytest -q -n auto -m "not perf"` | Everything this machine can run, GPU included. |
| `pytest -m perf -p no:xdist` | The four throughput gates, serially, on an idle machine (xdist workers skip `perf`). |

`--strict-markers` and `xfail_strict` are on. Tests are in pyright's scope
(annotate parameters). A session-scoped autouse fixture points `DROWSE_HOME`
at a scratch directory, so nothing reaches `~/.drowse`.

| Variable | Effect |
|---|---|
| `DROWSE_TEST_MODEL` | The GPU suite's model (default `HuggingFaceTB/SmolLM2-360M-Instruct`). |
| `DROWSE_TEST_LIVE_TOKENIZERS=1` | Lets `tests/model/test_role_template_drift.py` download per-family tokenizers instead of reading only the local HF cache. |

## Layout and markers

`tests/<pkg>/` mirrors `drowse/<pkg>/`; shared helpers live in
`tests/support/`, fixtures in `conftest.py`.

| Marker | Meaning |
|---|---|
| `gpu` | Needs CUDA or MPS; CI deselects it. `tests/gpu/test_smoke.py` (extract → steer → generate, and the perf gates), plus the `mps` params in `tests/model/test_model_numerics.py` and `tests/generation/test_generation_metadata.py`. |
| `perf` | Wall-clock throughput, serial only: steered ≥ 85% of vanilla tok/s, ablation ≥ 80%, unit ablation ≥ 80%, the served dashboard ≥ 50% (about 63% on SmolLM2/MPS). |

Add a GPU test only for what a CPU run cannot show (device numerics, MPS
kernels, throughput).

## Fixtures

| Fixture | Gives |
|---|---|
| `isolated_home` | A fresh `DROWSE_HOME` with selector caches reset, for tests that write artifacts. |
| `make_tiny_session` | A `TinySessionFactory`; `make_tiny_session(warm=False)` skips the eager whitener build; `.scripted(replies)` returns `(session, script)` with a `ScriptedReplies` hook that makes the model spell the reply while hidden states stay real. |
| `tiny_session` | One warm CPU session, `probes=[]`. |
| `app_client` | `app_client(session, **create_app_kwargs)`: a started `TestClient` over the real app. |
| `gpu_session` | The real GPU session on `DROWSE_TEST_MODEL`; skips without a GPU. |

Tiny sessions (`tests/support/tiny_models.py`, `sessions.py`) are genuine
`transformers` architectures from config (4 layers × 32 hidden, `llama`,
`qwen3`, `gemma3_text` with their real chat formats) and a byte-level tokenizer,
so the loader, scene grammar, capture, hooks and decode loop run unmodified.
Helpers: `register_vectors(session, *names)`, `forward_logits(session, text)`,
`baseline_corpus(response)`, `drowse_home(path)`.

## `tests/support`

| Module | Provides |
|---|---|
| `composer.py` | `SteeringRig`: a real `SteeringComposer` with its registry, neutral baseline and manager, no model. |
| `read_plane.py` | `ReadRig`: a real `ReadPlane`, the four instruments and the roster over a swappable capture. |
| `engine.py` | `engine_over(host, ...)`: a `GenerationEngine` with an inert runtime. |
| `lens.py` | `LensHarness`, `lens_instrument`/`lens_kwargs`, `FakeCapture`. |
| `sae.py` | `MockSaeBackend`, `sae_instrument`, `make_resident`. |
| `units.py` | `inert_units_instrument`, `random_unit_stats`, `patched_forward_logits`, `GEMMA4_MIXED_HEADS`. |
| `replay.py` | `ReplayRig`, `unwired_replay`. |
| `geometry.py` | `stub_geometry_instrument(stub)` for the remaining stub suites. |
| `capture.py` | `install_capture`, `StubHandle` (model-free pipeline tests). |
| `extraction.py` | `make_extraction`, `folded_vector_manifold`. |
| `whitener.py` | `synthetic_means`, `synthetic_whitener`, `rogue_whitener`, `isotropic_whitener` (exact identity metric). |
| `manifolds.py` | `loop_manifold`, `open_manifold`, `disk_manifold`, `stamp_current_geometry`. |
| `artifacts.py` | `author_manifold`, `write_fitted_manifold`. |
| `decode.py` | Fake logits models and tokenizers for decode, scoring and naturalness tests. |
| `jlens_toys.py`, `lens_oracle.py` | `frozen_toy(n_layers)`; the autograd lens oracle the transport is tested against. |
| `blocks.py` | `tiny_dense_model`, `default_init_model`, `spec_forward` (the `BlockSpec` oracle). |
| `server.py` | `mock_session` (an autospec idle session: a removed or renamed method fails the route test), `script_generate`, `StubSession`. |
| `cli.py`, `chat_templates.py`, `streams.py`, `tokens.py`, `gpu.py` | CLI fakes, per-family templates, stream fakes, `gpu_model_id`/`load_or_skip_inaccessible`. |

## Fixtures other repos read

`tests/browser_drowse_archive_bridge.py` is a script drowse.ai runs for its
archive round trip. `tests/fixtures/loom_tree_v2.json` and `token_rows.json` are
read by `webui/scripts/conversation-snapshot.test.mjs` (`npm run check`) and,
for token rows, `tests/loom/test_loom.py`; change them with the format.

## Writing tests

Pin behavior, not implementation details. Add a regression test when a bug
could quietly come back; skip it when the fix makes the bug unrepresentable or
another test covers it. Write artifacts only under `isolated_home` or a
`tmp_path` home. Mark wall-clock assertions `perf`.
