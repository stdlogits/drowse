# drowse/generation

The decode loop and what one generation reads or produces: sampler, prompt
rendering, prefix KV cache, compile/StaticCache routing, the read plane, token
payloads, results, finalization, completion scoring. Orchestration
(`GenerationEngine`, `TokenTap`, `CallSpec` resolution, decode routing) is
`session/engine.py`; the multi-row runner is `session/batch.py`
(`drowse/session/AGENTS.md`). Layer 9; lower layers' needs are typed against
narrow protocols (`render.ConversationHistory`).

| Module | Owns |
|---|---|
| `decode.py` | `generate_steered`, `GenerationConfig`, `GenerationState`, the sampler (`_sampler_candidates`, `_effective_topk`, `_PenaltyState`), `build_chat_input`, `hf_generation_config` / `hf_unneutralizable_processors` |
| `sampling.py` | `SamplingConfig` (frozen per-call knobs, `merged_with`), `INHERIT_SYSTEM_PROMPT` |
| `types.py` | `GenState`, `CallSpec`, `GenerationTransaction`, `GenerationStream` |
| `render.py` | `PromptRenderer`, `resolve_model_type`, `seat_stop_strings` |
| `prefix_cache.py` | `PrefixCache` (one slot: build / hit / release), `PREFIX_CACHE_MIN_TOKENS = 16` |
| `compile_runtime.py` | `CompiledRuntime`: compile, persistent hooks, StaticCache / CUDA-graph routing |
| `read_plane.py` | `ReadPlane`: capture apply, instrument transaction, step sinks, gate callback, pooled slice, finalize reads, `score_hidden` |
| `token_callback.py` | `TokenCallback` (6 positional args), `StepTokenCallback` (+ `step_id`), `TokenConsumer` / `TokenConsumerOptions` |
| `token_payloads.py` | `TokenProbePayload`: per-family reading slots, rendered as the `scope="token"` envelope |
| `finalize.py` | `FinalReads`, `finalize_generation`, `MIN_ELAPSED_FOR_RATE = 0.1` |
| `results.py` | `GenerationResult`, `RunSet`, `TokenEvent`, `TokenAlt`, `ProbeReadings`, `ResultCollector` |
| `scoring.py` | `ChoiceScorer`, `ChoiceScores`, `shared_prefix_len` |

## Decode loop

`generate_steered` runs on a worker thread under `torch.inference_mode()`. Each
iteration: stop check, `step_id = len(generated_ids)`, write `TriggerContext`
(`is_prefill`, `thinking`, `gen_step`), forward, `step_callback(step_id)`,
`score_callback(step_id)` into `trigger_ctx.probe_scores`, sample,
thinking/stop bookkeeping, `on_token(..., step_id)`.

- Sink, gate callback and tap share one `step_id`, and instrument runs key
  per-forward stashes on it, so gate and display reads of one forward pair
  without rescoring. A buffered UTF-8 flush carries the flushing forward's id.
- Gates lag one forward: scalars scored after forward `k` steer `k+1`; prefill
  sees none.
- Scoring runs after the forward, not in the capture hook: a host read at the
  deepest probe layer would drain the device queue before the remaining layers
  and LM head are enqueued.
- One device-to-host transfer per token. Entropy, chosen logprob and top-K ride
  the same packed transfer (int64 ids plus fp32 bits viewed as int32); with none
  requested it is one `.item()`.

### Sampler

Order: (1) `nan_to_num_` (nan 0, +-100), only under steering; finite logits are
never clipped, since clipping a large common offset flattens valid
distributions (Pythia). (2) presence/frequency penalties on raw logits, before
temperature; `_PenaltyState` counts completion tokens, thinking included.
(3) `logit_bias`. (4) `temperature <= 0`: argmax as a one-token pool. (5)
`logits / T`, then `topk(k)`, `k = _effective_topk`: the user's `top_k`, else
1024 (nucleus is insensitive past it), capped at vocab; top-k precedes top-p
(llama.cpp/Ollama order) and there is no full-vocab sort. (6) softmax, zero
entries whose preceding cumulative mass is `>= top_p`, floor top-1,
renormalize, `multinomial`.

The pool has fixed width with the excluded tail zeroed in place; a mask select
would make the shape data-dependent and add a second sync. Zero-probability
entries mean excluded. Logprobs, `TokenAlt`s and perplexity (`exp` of entropy)
describe this sampler distribution; the completion scorer reads the raw one.

Three other places reproduce the sampler and change with it.
`hf_generation_config` is the explicit HF `GenerationConfig` of the
`session/batch.py` fast path (processors neutral, `top_k = _effective_topk`,
greedy as `do_sample=False`, `remove_invalid_values` only under steering). HF
fills `None` fields from the checkpoint, so `hf_unneutralizable_processors`
names checkpoint processors that cannot be neutralized (`min_p`,
`suppress_tokens`, ...); any present sends the batch to the serial loop.
`loom/joint_logprobs.py` reuses `_sampler_candidates`, `_effective_topk` and
`_PenaltyState` but re-implements the penalty/bias step (`_LogitAdjust`). The
drowse.ai browser runtime mirrors the sampler.

### Forks, thinking, stops

- `forced_prefix` overrides the fed-back token for the first `N` steps while the
  multinomial draw still runs, so re-seeding keeps the RNG stream bit-identical
  through a fork. It takes exact raw ids, delimiters included; a forced id
  outside the pool reports logprob `None`. `seed` seeds the global torch RNG;
  MPS is not bitwise deterministic.
- The thinking machine (`IDLE`/`PREAMBLE`/`THINKING`/`RESPONSE_PREAMBLE`) runs
  off `ChatTraits.think` even with `thinking=False`, since channel formats
  (gpt-oss) emit their analysis channel regardless. `thinking=False` turns off
  `starts_in_thinking` only for templates with an `enable_thinking` switch; for
  channel formats EOS inside thinking is a separator. Any exit mid-thought sets
  `thinking_end_idx = len(generated_ids)`, so the response decodes empty.
- Stop sequences match non-thinking text against a tail of `max(len(s)) - 1`
  chars plus the current emit. A hit emits the pre-stop part and sets
  `state.response_text` and `response_aggregate_index`. Matching runs only when
  `on_token` is set; the engine forces a tap for any stop list.
- A token whose lone decode contains U+FFFD is buffered and flushed as a group
  (`token_id = -1`, no logprob). Text comes from the lazy `ChatTraits.token_text`.
- `results.FinishReason`: `"stop"` (EOS), `"length"`, `"stop_sequence"`,
  `"cancelled"` (client Stop, closed stream, disconnect; the batch fast path
  marks rows its stop criterion cut).

### KV paths

DynamicCache omits `cache_position` (HF derives it; an explicit `None` breaks
that). StaticCache is caller-supplied: `cache_position` is an arange on prefill,
then one 1-element buffer advanced by `fill_` from a Python counter (stable
address for graph replay, no `.item()`). A prefix hit sets
`cache_position_offset` to the prefix length and feeds the suffix. A model
returning no `past_key_values` on prefill drops to O(N^2) re-feeding, with one
warning. `logits_to_keep=1` avoids `[1, T, V]` prefill logits (a `TypeError`
retries once without it). Decode ends with `torch.mps.synchronize()`, so a rapid
regenerate cannot recommit an in-flight Metal command buffer.

`GenerationState` holds the stop event (thread-local `cancellation_scope`),
`emit_map` (behind `TokenEvent.raw_index`), `thinking_end_idx`, `finish_reason`,
`response_text`, `response_aggregate_index`; `reset()` keeps `stop_count`, which
the batch runner compares across rows. `GenerationConfig` (frozen;
`max_new_tokens=1024`, `temperature=1.0`, `top_p=0.9`, `top_k=None`) is the
session default; per-call `SamplingConfig` composes into a local copy and
`merged_with` lets the other side's non-default fields win.

## Rendering

`PromptRenderer.prepare_input` turns a conversation into `input_ids`. `input` is
a new user turn (`str`, labelled `user_role`), `None`, or a message list; else
`GenerationRequestError` (400). A stateless list renders alone; a stateful one is
recorded first (`loom/ops.py::split_message_list`). `raw=True` renders
`tree.flat_text(...) + input` with no template. `INHERIT_SYSTEM_PROMPT` reads the
live session default; `None` means none. On the assistant seat a role-augmented
steering scope's role wins (steer baseline = extract baseline), with
`RoleBaselineMismatchWarning` when the send's `assistant_role` differs. The scene
grammar resolves lazily once (`None` = unavailable); routing among stitcher,
`apply_chat_template`, per-turn splice and `render_scene_raw` is in
`drowse/model/AGENTS.md`. The render cache keys on messages, labels, thinking,
seat, scene, model type, the live template, special tokens and added-token count.
`seat_stop_strings` adds a seat's close as a stop string only when it differs
from the assistant's (gpt-oss `<|end|>` vs `<|return|>`), preserving the
no-stop-list fast path otherwise.

## Prefix cache

One opt-in prompt-prefix KV slot (reuse policy: `drowse/session/AGENTS.md`).
`build` prefills outside any steering scope with capture suspended, so the KV is a
pure function of the tokens. A hit needs an exact token prefix and a non-empty
suffix (the first sample reads the suffix's last logit); a static entry also needs
a StaticCache-eligible generation fitting `max_cache_len`, and a failed static
build falls back to a DynamicCache entry. `release(hit)` follows every reuse and a
failed rollback drops the entry. Rollback by layer kind: full attention
`crop(-n)`; sliding-window and conv states via upstream `activate_past_recording`
armed after the prefill (recording a long prefill is the memory spike), then
`crop`; recurrent states snapshotted and restored; StaticCache by resetting each
layer's `cumulative_length` (a wrapped sliding layer cannot rewind). Static
entries record no past states, which would reassign buffers a compiled graph
captured. No transformers class is patched.

## Compile routing

`CompiledRuntime.prepare` compiles only on CUDA/MPS, installing the persistent
offset hooks and (unless `probes=[]`) persistent last-token capture hooks before
`torch.compile`, so they trace into the graph (removed if `compile_with_probe`
falls back to eager). `compile_mode=None` picks `"reduce-overhead"` only when CUDA
graphs are supported, else `"default"`. `static_cache_active` holds iff (compiled
or CUDA graphs) and the architecture's StaticCache constructs.
`graph_thread_safe()`: CUDA graph trees are thread-local and warmed on the
constructing thread, so other threads run eager. `acquire_static_cache` returns
one identity-stable cache, reset in place and regrown on demand (Dynamo guards key
on the K/V tensors). `take_offset_hooks()` hands the offset buffers to
`SteeringManager`. Which generation takes the static or compiled path is
`GenerationEngine.decode`; curved, gated and phased steering never compiles.

## Read plane

`ReadPlane` applies one generation's read side over explicit dependencies
(capture, Monitor, the three instruments, layers, tokenizer, composer
`gate_refs`); the instrument transaction, gate-channel validation and capture-mode
table are in `drowse/readout/AGENTS.md`.

- `begin` binds runs, plans, attaches capture, arms the ring and step sink, and
  resets or disables curved warm-start. Sinks append one row per forward (empty
  rows included), so row `k` is forward `k`.
- `gate_score_callback` scores only gate keys naming attached probes (subset row,
  incremental row with backfill, or latest slices; lens/SAE through their bound
  runs). An empty capture scores nothing, so gates read inactive, never stale.
  `gate_run()` is the same capture for a forced replay (joint logprobs, loom
  token and node replays of a probe-gated reply), capturing only what open gates
  read.
- `aggregate_index` / `pooled_slice` is the one finalize pooling for all three
  families: the loop's visible endpoint when set, `None` for a stop-sequence
  finish without one, else `last_content_index` over `generated_ids` (thinking
  tokens count). The geometry aggregate uses a cold curved foot solve, since the
  pooled position need not be the last forward a warm foot followed.
- `begin_batch` / `batch_row_reads` serve the HF fast batch from a `[B, D]` ring;
  `score_hidden` is the offline read.

## Tokens, payloads, results

`TokenConsumer` receives whole `TokenEvent`s and declares what it reads in
`TokenConsumerOptions` (`live_scores`, `lens_readout`, `sae_readout`,
`perplexity`, `probe_readings`) so the engine skips unconsumed work;
`live_scores` means the full reading, and whether the loom row keeps per-layer
detail is the call's `persist_per_layer_scores`. `TokenProbePayload` keeps one
slot per `InstrumentFamily` and renders `{"measurements": envelope}`
(`readout/measurements.py::build_measurements`); the compat `all_readings`
projection (feeding `x-drowse-probe-readings`) runs only for a consumer that set
`probe_readings`. `TokenEvent.raw_index` is the `generated_ids` index behind an
emission (the fork join key); the engine leaves `TokenEvent.finish_reason` unset.

Logprob width is `max(return_top_k, logprobs)` when the alternative width (per
call, else session default) is positive, else `SamplingConfig.logprobs`; `None`
captures nothing. `finalize_generation` decodes `generated_ids[thinking_end_idx:]`
(or the exact `response_text` after a stop), builds the `scope="aggregate"`
envelope and compat dict, calls `record` (`last_result`), emits `ProbeScored`,
then finalizes the loom node. `thinking_text` is stamped only when the scene
grammar has `think_open`, since a stamped block elsewhere makes later renders
raise. `GenerationResult.applied_steering` round-trips through `parse_expr`;
`RunSet` is a `list[GenerationResult]` with `node_ids`, `grid`, `kind`,
`metrics`.

## Completion scoring

`ChoiceScorer` (`session.score_choices` / `score_template`) gives each
candidate's joint `log P(candidate | context)`: the logit counterpart of a
template-derived manifold fit (`drowse/store/AGENTS.md`). `steering=` shows
whether steering shifted the distribution, not just the argmax.

- Raw model distribution: fp32 `log_softmax`, temperature 1, no top-k/top-p.
- Reports `sum_logprob` and length-normalized `mean_logprob`, each with its own
  softmax (`prob_sum`, `prob_mean`). A degenerate candidate (no distinct token,
  `sum_logprob` 0.0) is masked to `-inf`; an all-degenerate set returns zeros.
- `shared_prefix_len` recovers each span across the boundary-token merge
  (`"Monday"` vs `"Tuesday"`); `loom/joint_logprobs.py` reuses it.
- One right-padded, `use_cache=False` teacher-forced forward per `_SCORE_BATCH =
  16` rows, `logits_to_keep` trimmed to the completion tail. Rendering is plain
  `build_chat_input`; `system_prompt=None` means none. Text after a template's
  slot is not scored.
- The steering scope opens outside the `ExclusiveGuard` section: composer
  push/pop take the same lock blocking, so the other nesting self-deadlocks.

## Tests

`tests/generation/` (CPU), `tests/readout/test_read_planner.py` (rig
`tests/support/read_plane.py::ReadRig`), `tests/session/test_engine.py`.
`tests/gpu/test_smoke.py::test_throughput_regression` gates steered decode at
>= 85% of vanilla tok/s; loop, capture or routing changes want that run.
