# drowse/session

`DrowseSession` (`session.py`) is a facade over one loaded model. It owns the
model handle, the exclusive-GPU lock, the loom tree, the event bus and the
generation defaults, builds every collaborator with explicit dependencies, and
delegates each public method to one of them; new behavior goes into a
collaborator, not session state. Layer 12, directly below the frontends.
Nothing outside this package reads a session private; modules inside it may
(the engine calls `_start_loom_assistant`, `_plan_authored_prompt_capture` and
`_steering_composer`). `Extraction` and `SaeRuntime` take the session itself as
a Protocol-typed handle (`fit.pipeline.ModelHandle`: `model`, `tokenizer`,
`model_id`, `device`, `layers`, `whitener`, `layer_means`;
`sae.runtime.SaeModelHandle`), so those properties are a contract.

| Module | Owns |
|---|---|
| `session.py` | wiring, `_model_exclusive`, probe resolution, the public API, `close()` |
| `engine.py` | `GenerationEngine`: the transaction, `CallSpec` resolution, `TokenTap`, decode routing, the prefix-cache slot, the stream worker, `last_result`; `PrefixCacheStateError` (409) |
| `batch.py` | `BatchRunner` (fan / batch / sweep over `generate_one`), `FastBatchPath` (one batched HF `generate`) |
| `registry.py` | `ArtifactRegistry`: in-memory profiles and loaded manifolds, swaps, probe folds, load validation, analytics snapshots |
| `neutral.py` | `NeutralBaseline`: means, whitener, unit statistics and mid means from one neutral load |
| `extraction.py` | `Extraction`: extract / extract_from_corpora / fit / generate_manifold / bake over `fit.pipeline` |
| `authored_capture.py` | measuring authored loom text the first time a prefill re-reads it |

## Construction

`from_pretrained` = `model.load.load_model` → `CompiledRuntime.prepare`
(persistent offset and capture hooks go on before `torch.compile`; capture only
when probes will attach) → `__init__`, which also wraps a pre-loaded model
(`CompiledRuntime.for_model`, no compilation). `__init__` adopts the tokenizer's
EOS set as the model's generation default, always calls
`materialize_bundled_artifacts()`, then wires the collaborators. Several are
wired with lambdas naming attributes built later (`self._roster`,
`self._engine`, `self._lens_instrument`), so model work in `__init__` must come
after the wiring block. `config` is a frozen `GenerationConfig`: rebind it with
`dataclasses.replace` (an in-flight generation keeps its own).

The probe roster comes last, after the neutral baseline. `probes=None` attaches
the bundled concepts tagged in `PROBE_CATEGORIES` (fitting stale ones through
`self.fit`) plus every bundled multi-node manifold *already* fitted (fitting
`personas` would block startup for minutes); a list attaches those categories
only; `probes=[]` attaches nothing and skips the neutral build.
`close()` removes every hook the session attached, the permanent offset and
capture hooks included, so a caller-owned model stays usable.

## The exclusive section

One reentrant `_gen_lock` serializes model work (root `AGENTS.md`).
`_model_exclusive(busy_msg, *, phase_msg=None, exc=ConcurrentExtractionError)`
(`base/exclusive.py::ExclusiveGuard`) is the guard every collaborator receives:
it acquires non-blocking, raises `exc(busy_msg)` when another thread holds the
lock and, with `phase_msg`, also while a generation is in flight.

| Caller | Acquisition |
|---|---|
| a generation (`GenerationEngine.transaction`) | non-blocking; `ConcurrentGenerationError` (409) |
| extract, fit, `generate_manifold`, bake, `evict_artifact`, scoring, token replay, lens fit/readout, SAE train, guarded neutral getters | with `phase_msg`; `ConcurrentExtractionError` (409) |
| probe attach (every family), geometry detach, SAE load/unload | busy check only |
| steering push/pop (`SteeringScope` enter/exit) | blocking: waits out a generation |
| `analytics_profile` | non-blocking; `None` instead of raising |

The lock is reentrant so a generation can push its own steering scope while
holding it; same-thread re-entry from inside a generation (an `on_token`
callback steering or generating) is rejected by the engine's `GenState` phase
(`IDLE → PREAMBLE → RUNNING → FINALIZING`), not by the lock. Open a steering
scope before entering the exclusive section, not inside it. Artifact-producing
work calls `SteeringComposer.assert_unsteered()`, and `NeutralBaseline.ensure`
refuses to compute under an installed plan. The server's `asyncio.Lock` request
serializer is separate.

## Generation (`engine.py`)

`session.steering(value)` returns an un-entered `SteeringScope` over an
expression or a `Steering` (`None` or a dict is a `TypeError`); every artifact
resolves at the call, so an unknown name raises before the `with`. Nesting
flattens; the innermost scope naming a role wins. A generation resolves its
steering before creating the loom node (a bad expression leaves the tree
untouched), enters it after, and pops it before finalize, so finalize reads run
unsteered.

`generate_one` = `resolve_call` (frozen `CallSpec`) → plan consumers and reads →
decode with a `TokenTap` → `finalize` (records `last_result`), inside
`transaction()`. Teardown is exception-hard: pop a leftover scope (swallowing a
teardown error while another failure propagates), release the read plane and
loom reservation, delete a reply node a failure left without a token, then set
`IDLE` and release the lock unconditionally. A failed call therefore leaves the
user turn, never an empty assistant turn.

- **Composition** is `loom/recipe.py::compose_call`; the engine supplies the
  tiers (the cast member for the seat's label, the call's kwargs, and for
  `recipe_override` the anchor recipe plus the modifier derived from it).
  Composed `steering=""` is unsteered.
- **Same-role continuation** (`append_same_role`, default on
  `generate`/`generate_stream`, off for fan/fork/prefill): with no input and a
  selected leaf in the same seat and role label, that node regenerates in place
  with its text as forced prefix, `max_tokens` grown by the prefix, thinking
  off.
- **Thinking**: explicit `thinking` > the steering's > `config.thinking` >
  `chat_traits.supports_thinking`. Readout width: per-call `return_top_k`, else
  the session default, else `DEFAULT_READOUT_TOP_K = 8`.

Decode routing: a **prefix-cache hit** decodes only the suffix and needs no
`return_hidden`, no thinking and no term active in prefill (the cached prefix is
unsteered). **StaticCache** applies when the runtime has it, graph capture is
thread-safe, and steering is absent or entirely the static single-affine fast
path; the total length is bucketed to 256 tokens so most requests share one
allocation and one set of Dynamo guards. **The compiled module** runs only when
the live hook topology matches the traced one (no transient capture or unit-tap
hooks; steering absent, fast path, or lowered to the persistent offsets);
otherwise the eager `_orig_mod` runs on a DynamicCache, avoiding a retrace per
topology.

`TokenTap` is skipped when nothing consumes it (`ConsumerPlan.need_tap`), since
it forces a per-step log-softmax and entropy sync. `on_token` is the public
six-argument callback `(text, is_thinking, token_id, logprob, top_alts,
perplexity)`, which never receives measurements, or an internal `TokenConsumer`
that gets each `TokenEvent` with the measurements it asked for.
`generate_stream` runs `generate_one` on a worker thread; closing the stream
cancels, joins and re-raises the worker's error. `cache_prefix` fills the
**single** prefix slot (an entry-owned cache); it raises
`PrefixCacheStateError` inside a steering scope, a mismatch misses (full
prefill) rather than mis-hits, and a push or pop that leaves prefill steered
clears it.

## Multi-row runs (`batch.py`)

- `generate(n>1)` → `BatchRunner.fan`: seeds from `derive_seed_schedule`; a
  stateful fan re-selects its starting node before each row so rows land as
  siblings (placement only, never the regen anchor).
- `generate_batch` takes `FastBatchPath` (one left-padded HF `generate`) for ≥ 2
  stateless rows unless `fast_batch_blocked` (thinking, logprobs, stop strings,
  logit bias, penalties, seeded sampling, hidden or per-layer persistence, any
  trigger that is not always-active). Serial rows sharing ≥ 16 rendered tokens
  under prefill-inactive steering prefill that prefix once through the single
  slot, replacing any user-cached prefix.
- `generate_sweep`: the Cartesian product of `{concept: [alphas]}` over
  `base_steering`. `stop()` ends a whole series (the runner compares
  `GenerationState.stop_count` across rows).

## Artifacts

**Registry.** `profiles`/`manifolds` are read-only `Mapping` views; every
mutation is a method and drops the analytics snapshots (the unsteered prefix
cache survives). Lazy loads (`ensure_*`) memoize and fire nothing. A replace or
evict (`adopt_fitted`, `evict_artifact`, or `put_profile` over a profile a probe
was folded from) fires `on_swap({id(old): new | None})` → `ProbeRoster.swap`,
which re-attaches each affected geometry probe or detaches it. Folds handed out
by `fold_probe` are tracked weakly per key, so refits reach probes that never
held a loaded manifold. Replacements are built before the first mutation.

`ensure_manifold_loaded(key)` admits a tensor only when its sidecar is current,
a `from-` variant's provenance matches, `role-<r>` was fitted uniformly with
role `r` (and `raw` with none), `model_fingerprint` matches the live weights,
and (except `baked`) `capture_sha256` recomputed against the live tokenizer,
chat template, role framing and baseline prompts matches, and the fit is current
for the live corpus and fit policy. Every miss is `ManifoldNotRegisteredError`
(404, refit hint; a stale fit's `StaleFitError` is chained), which the probe
resolver and the default roster answer by fitting in place; a bare name fitted
in two namespaces is `AmbiguousSelectorError`. Admitted
manifolds are promoted to fp32 on the model device. `concretize_key` pins a bare
`:sae`/`:role`/`:from` to the unique fitted tensor. `ensure_profile_registered`
tries an in-memory profile, then a reserved-namespace atom resolver (`jlens`,
`slens`, `sae`, `neuron`; `head` is reserved but not steerable), then a fitted
manifold folded to one direction. Both lazy loads run their miss path in the
exclusive section (it uploads to the device, and an atom resolver reads the
model).

**Neutral baseline.** One load of the neutral cache yields the means
(`X.mean(0)`), the whitener, the resident `UnitStats` and the mid means (each
with an `…_unavailable` reason when absent). `site_means(site)` is the lens
instrument's centring mean (`out` → means, `mid` → mid means,
`NeutralMeansError` 422 when absent). `ensure()` is unguarded, for callers
already in the section; `session.layer_means`/`session.whitener` are the guarded
getters. An unbuildable whitener stays `None`, so every Mahalanobis consumer
raises `WhitenerError`; a new one re-points the Monitor.

**Extraction.** Without `namespace`, an installed concept is fitted where it
lives, never shadowed by a `local/` copy; `force` on a bundled or hub-installed
folder needs `namespace='local'` (`pack refresh` would restore its corpus);
reusing a folder of more than two nodes as a vector is `ManifoldSpecError`.
`sae` and `role` are exclusive; the returned name carries the variant tail.
Author-or-reuse is decided under the manifest lock while corpus generation runs
outside it. A fit `adopt_fitted`s every in-memory consumer; a failed fit whose
overrides rewrote the manifest, any `generate_manifold` (even part-way) and
`bake` evict the folder's consumers. Frontends call
`session.evict_artifact(ns, name)` after a folder changes on disk.

## Reads

`session.instruments` maps `geometry`/`lens`/`sae`/`units` to the
`readout.protocol.Instrument` implementations; the server's `{family}` routes
iterate it. `ProbeRoster`, `ReadPlane`, `FinalReads` and `TokenTap` name the
families explicitly, so a new family touches them too.

`add_probe(selector, *, as_name=None, top_n=3)` routes through `ProbeRoster`:
`jlens/`/`slens/` → lens, `sae/` → SAE, `neuron/`/`head/` → units, anything
else → geometry. Geometry resolution (`_resolve_probe_manifold`): a registered
profile (folded to a 1-node ray), else a fitted manifold on disk, else an
installed manifold not fitted for this model, which it **fits in place** inside
the attach's section (raw and `:role-<x>` only; `:sae-`/`:from-` is a
`RequestError` naming the producing command), else it **extracts** a bare raw
name that names nothing installed. A typed namespace or variant that names
nothing installed is `ManifoldNotRegisteredError`, never a slugged extract.
Lens/SAE detaches are safe mid-generation (bound runs keep a frozen roster).
Node recipes stamp the geometry probes and their hashes.

**Authored capture.** Authored loom text (`recipe is None`) has no token rows of
its own; the first generation whose prefill renders it measures those positions
from the same forward and writes them with `LoomTree.set_authored_capture`,
anchored by a `PromptCapture` (the call's steering, the render's parameters,
each channel's first position and the digest of the render before it). A node
is captured once, both channels from one render; nothing is tokenized unless the
Monitor or a live lens/SAE consumer will record. Matching retries
whitespace-stripped text (Gemma templates trim content). The first suffix token
after a prefix-cache hit has no producer row and stays unscored. Row and
envelope shapes are the loom token-row contract the browser runtime replays.

## Conversation and API

`tree` is a `LoomTree`; session methods exist only where they add something:
`append_turn` (scene grammar), `fork_from_token`/`prefill_assistant` (planned in
`loom/ops.py`, run through the engine), `set_cast_member` (validates steering
now), `restore_tree`, `joint_logprobs`. `restore_tree` rebinds `session.tree`,
so collaborators read it through a callable (`tree=lambda: self.tree`) and
never cache it; the caller guarantees no generation is in flight.
`joint_logprobs` is uncached; the server fills `session.joint_logprob_cache`,
which the session clears on `LoomMutated` `edit`/`delete`/`reset`/
`finalize_assistant` and on `restore_tree`.

API notes: generation calls return a list-like `RunSet` (`generate_stream` a
`GenerationStream`); `last_result.applied_steering` is the canonical expression
(it round-trips through `parse_expr`); `system_prompt` defaults to the
`INHERIT_SYSTEM_PROMPT` sentinel (explicit `None` clears); `stateless=True`
never touches the tree; `extract`/`bake` return `(name, Profile)`, `fit` the
`Manifold`; `events` is a synchronous `EventBus` (async consumers hop with
`loop.call_soon_threadsafe`).

Tests: `tests/session/` (CPU). Prefer a real session (`tiny_session`,
`.scripted(replies)` for deterministic output) over `DrowseSession.__new__`
skeletons; narrow rigs are `tests.support.engine.engine_over` and
`tests.support.extraction.make_extraction`.
