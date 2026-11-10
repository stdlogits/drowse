# drowse/loom

The conversation layer: the branching tree a session generates into
(`LoomTree`), the receipt on each generated turn (`Recipe`), the one rule that
composes a generation's effective recipe (`compose_call`), plus the cast
roster, conversation ops, token and node replay, diff, joint logprobs, the
filter grammar and YAML transcripts. Nothing here owns a model. Layer 11 (above
`fit`, below `session`); lower layers never import it (`generation/render.py`
types the tree as `ConversationHistory`). Collaborators take the tree as a
getter because `restore_tree` replaces `session.tree`. Neighbours:
`session/engine.py` (tiers; `TokenTap` writes rows), `session/session.py`
(stamps recipes, the verbs), `session/authored_capture.py`, `model/scene.py`,
`server/tree_routes.py`.

| Module | Owns |
|---|---|
| `tree.py` | `LoomNode`, `LoomTree` (mutators, reservation, cast, persistence), `PromptCapture`/`prompt_digest`, `TREE_FORMAT_VERSION` (2), `TOKEN_SIDECAR_FORMAT_VERSION` (3) |
| `recipe.py` | `Recipe`, `CastMember`, `compose_call`, `CALL_ONLY_SAMPLING_FIELDS`, regen modifiers, `derive_seed_schedule` |
| `ops.py` | `append_turn`, seating and thinking checks, `resolve_token_anchor`/`resolve_prompt_anchor`, `plan_fork`/`plan_prefill`, `split_message_list`, `begin_generation_turn` |
| `replay.py` | `TokenReplayer` (implements `readout/replay.py::TokenReplay`), `ReplayForward`, `replay_segments`, `drive_segments`, `capture_pooled` |
| `node_replay.py` | `NodeReplayer` (`session.token_anatomy`, `session.unit_activity`) |
| `diff.py`, `filter.py` | node/text/readings/token diffs, `steering_delta`; the filter grammar |
| `transcript.py`, `joint_logprobs.py` | YAML transcripts; `JointLogprobReplayer` |
| `errors.py` | `LoomTreeError` (400), `UnknownNodeError` (`KeyError`, 404), `InvalidNodeOperationError` (400), `MutationDuringGenerationError` (409), `RecipeOverrideError` (400); `require_fields` |

## Tree data model

- **Root**: a synthetic, empty, parentless `system` node; a system prompt is
  never a node.
- **Provenance is recipe presence.** A node with a `Recipe` is generated, in
  either seat (the model can generate into the user seat); `recipe is None` is
  authored. Rewind, anchoring, authored capture, replay, transcript import and
  joint logprobs key on this, never on `role`.
- **`role_label`**: the cast label the turn was sent under (`None` = family
  default), immutable, unrelated to `SteeringComposer.active_role`.
- **`thinking_text`** is stamped at finalize only when the scene grammar has
  `think_open` (a block the family can't re-render would break every later
  render).
- **Token rows.** `tokens`/`thinking_tokens` hold `TokenTap` row dicts, never
  mutated (a shallow list copy is a stable snapshot). `raw_token_ids` is the
  exact decode sequence, delimiters and unmerged partial UTF-8 included: the
  forceable prefix for forks and replays, joined to rows by `raw_index`.
- **Prompt capture.** An authored turn's rows arrive once with their replay
  anchor `PromptCapture`: the capturing call's steering, the render's system
  prompt, thinking flag and flat mode, each channel's first render position
  (`start`, `thinking_start`) and `digest` (`prompt_digest`, SHA-256 of the
  comma-joined ids of the render before the later channel). Captured rows have
  no `raw_index`; a token's address is its place among the rows, thinking
  first. `set_authored_capture` refuses a captured node, so addresses never
  shift; `append_authored` clears rows and anchor.
- **Readings.** `aggregate_readings` holds axis-0 aggregates from all four read
  families; `aggregate_measurements` the aggregate envelope (summaries only, no
  traces, no `scores`; `None` for authored, imported or unmeasured turns).
- **Edits.** `edit` rewrites `text` only; recipe, rows and `raw_token_ids` keep
  the original decode, so forks, replay and joint logprobs read the original.
  An earlier turn edited since an authored capture fails the anchor's digest,
  and that replay refuses (400).

## Mutation rules

Public methods take the tree's `RLock`. Each mutation bumps `rev` and emits one
`LoomMutated` (`drowse/events.py`); surfaces refetch on a `rev` gap. No-ops
emit nothing; hot-loop `append_token` neither bumps `rev` nor emits (deltas ride
the token stream).

**Reservation.** `begin_generation_turn` reserves the parent of the streaming
node until `session._release_generation`'s `finally`. Meanwhile `edit`,
`append_authored` and `delete_subtree` refuse the reserved node, its ancestors
and descendants, and `reset` refuses outright (409); new turns, `branch`,
`graft`, `navigate`, star, note and cast stay free. `replacing` doesn't check
it, so `restore_tree`'s caller excludes generation (the server's session
lock).

| Op | Worth knowing |
|---|---|
| `add_user_turn` | reuses a user child with identical text, `role_label` and `thinking_text` (regen and prefill rely on it) |
| `add_assistant_turn` | authored through begin/finalize: `tokens=None`, `raw_token_ids` kept (forkable); `dedup_existing` reuses an identical child |
| `begin_continuation` | resets a node in place; the caller replays its old text as the forced prefix |
| `branch` | authored sibling; copies `role_label`, and `thinking_text` only for unchanged text; refuses the root |
| `graft` | validates every turn first, then chain + new cast labels + note in one `rev` |
| `reset` | new root; the cast survives and `rev` keeps counting |
| `rewind` | up one submission: two nodes from a generated node, one from an authored one |

## Persistence

`tree_format` 2 and `token_sidecar_format` 3 exactly, no migration. Tree, node,
cast-member and recipe readers (`errors.require_fields`) require the fields
they use and ignore any other key. `from_dict` also checks structure
(`children_of` keys equal node ids, one agreeing parent, all reachable from a
parentless `system` root, no cycles, the four token fields `tokens`,
`thinking_tokens`, `raw_token_ids`, `prompt_capture` all present or none) and
skips `structural`/`observed` cast entries, so the server's effective-roster
JSON round-trips.

**Token rows and `PromptCapture` stay strict on purpose**: the dashboard's
validator (`webui/src/lib/tokenRows.ts`) refuses exactly what this reader
refuses, and both run `tests/fixtures/token_rows.json`. A row's keys are a
subset of `token_id`, `text`, `logprob`, `sampler_entropy` (the browser
runtime's), `perplexity`, `raw_index`, `top_alts`, `measurements`; its envelope
is `MEASUREMENTS_VERSION` (2) in one of two shapes:

- **generated** (`readout/measurements.py::token_row_measurements`): `scores`
  and each geometry reading's exactly `fraction` and `coords`, no other channel
  (the token view replays the rest);
- **kept whole** (`summary_measurements(per_layer=True)`: authored capture, or
  `persist_per_layer_scores`): each geometry reading's summary and all three
  traces.

Mixing the two, pre-6.0 shapes (row-level `probes`/`per_layer_scores`,
`subspace_coords_per_layer`, a summary without its traces) and a lens readout
entry without its vocabulary `id` (readers key the readout by id) are refused.
An authored node has rows exactly where its `prompt_capture` places a channel,
each with `token_id` and no `raw_index`. A node's `aggregate_measurements` is
null or an `aggregate`-scope envelope without `scores` whose geometry readings
carry exactly their summary. `finish_reason` is null or one of
`FINISH_REASONS` (`cancelled`: a Stop or disconnect cut the reply short).

`to_dict()` omits token payloads; `GET /tree` sends `include_tokens=True` and
`PUT /tree` (the dashboard's restore) reads them back inline. `save` writes a
gzip `<stem>.tokens.json.gz` sidecar, then the JSON atomically; `load` caps the
JSON at 64 MiB and the decompressed sidecar at 256 MiB (`max_bytes=`
overrides). `LoomTree.replacing` refuses another model's tree (null `model_id`
passes), keeps the event bus and `session_id`, and sets `rev = max(prev, saved)
+ 1`; the owner installs it and calls `announce_replacement`. There is no
autosave.

## Recipes and composition (`recipe.py`)

A `Recipe` carries `steering` (canonical; `None` unsteered), `sampling` (the
composed config, seat labels and observation knobs included), `thinking`,
`system_prompt` (resolved; absent means inherit, while `None` and `""` are
captured values) and `probes`/`probe_hashes` (geometry probes). The seed lives
in `sampling.seed`; a top-level `seed` mirrors it on disk and wins on read.
Every `SamplingConfig` field is persisted and required on read, so a new
sampling field changes saved trees, transcripts and the drowse.ai seam. A
recipe carrying `position_spans` (the browser runtime's position targeting) is
refused: ignoring it would replay the steering at every position.

`compose_call` is the single composition rule. Tiers, weakest first: `cast`,
the call's kwargs, `anchor`, `modifier`. A stronger tier sets
`steering`/`thinking` when not `None` (`""` is explicit unsteered and blocks
weaker tiers) and `system_prompt` when not inherited; `sampling` merges
field-wise (`SamplingConfig.merged_with`: a default field is unset, so no tier
can reset a field to its default). Exceptions:

- sampling ranks the call above the anchor (cast < anchor < call < modifier),
  so a regenerate sending no sampling reproduces the anchor and a variation
  runs at the call's `max_tokens`;
- `CALL_ONLY_SAMPLING_FIELDS` (`logprobs`, `return_hidden`, `return_top_k`,
  both `persist_*`, `return_probe_readings`) always come from the call: a
  recipe never decides what a consumer captures;
- the seed resolves modifier > call > anchor > cast, so fanned per-sibling
  seeds beat the anchor while an unseeded one-shot regen reproduces it.

`GenerationEngine.compose_call` supplies the tiers: the cast member is
`tree.cast[label]` (the seat's role label, else the seat name; raw mode skips
the roster); anchor and modifier exist only with `recipe_override`, the anchor
being `anchor_recipe(parent)` (the nearest generated ancestor's recipe), never
the regenerated node or a sibling. Modifiers (`Recipe.compose_modifier`):
`unsteered`, `inverted` (coefficients negated; `along` on a `%` term, `onto`
kept, and a curved `%` then refuses at the steering plan, 422), `reseed`,
`cool`/`hot` (0.3/1.2), a `Recipe`, or a partial `"field=value, …"` split only
at `, <ident>=` (steering may hold commas). A modifier seed beats the call, so
`seed=42` on a fan pins every sibling. `derive_seed_schedule(base, n)`: `n>1`
derives `blake2b(base, i)` seeds.

**Cast.** `tree.cast` maps label → `CastMember` (optional standing `Recipe` +
`notes`). `cast_roster()` is structural `user`/`assistant`, every label
observed anywhere in the tree (so speakers don't flicker on navigation), then
configured members. A fieldless member contributes nothing; steering `""` pins
a label unsteered.

## Conversation operations (`ops.py`)

- `append_turn` is the one authored-append rule: same seat and label as the
  parent concatenates (`append_authored`); otherwise a new turn. Seating is
  free in scene and flat mode; a legacy chat template renders only
  alternation, so there seats must alternate.
- `begin_generation_turn` opens and reserves. A message list is a whole
  conversation: `split_message_list` validates it all before recording
  anything, and its turns land from the root reusing identical children, so a
  re-sent list shares its history and its replies are siblings.
- `resolve_token_anchor` is the one `(node_id, raw_index)` validation.
- `plan_fork` replays `raw_ids[:raw_index]`, forces `alt_token_id` or
  `replacement_text` (exactly one), then samples under the node's recipe. The
  original seed re-applies by default and the sampler still draws on every
  forced step, so the RNG stays aligned; `seed=` asks for a new ending.

Gotchas: a fork passes the recipe's steering verbatim, so a node stamped
unsteered (`None`) picks up its label's cast steering if a member now exists
(transcript replay passes `""` for this reason). `plan_prefill` carries neither
the node's `role_label` (unless `sampling.user_role` matches) nor its
`thinking_text`, so such a user node gets a duplicate sibling.

## Replay (`replay.py`, `node_replay.py`)

`TokenReplayer` backs every family's `token_readout`: it rebuilds the node's
continue-mode render to its parent (stamped labels, recipe thinking and system
prompt, seat), appends `raw_ids[:raw_index]`, and hands the activations pooled
at the producing position to the family's `read`. The steering scope opens
**outside** the exclusive section (composer push/pop take the generation lock
blocking; nesting deadlocks); the render (which uploads ids) and the read run
inside it (an upload outside once raced a lock-free read for MPS's one command
buffer and aborted the server). `forward(node_id, raw_index, run=…)` hands a
`ReplayForward` (`ids`, `prompt_length`; raw token `i` is produced at
`prompt_length + i − 1`) to a caller-owned forward.

An authored token routes through `resolve_prompt_anchor`: the path render with
the capture's parameters, checked against its digest (mismatch: 400), then the
channel's earlier tokens from the rows, as one prefill under the capture's
steering (a prefill fires a term iff it has a prompt slot and no probe gate).

**Triggers.** A trigger decides per forward, so `replay_segments` splits a
replay as generation ran it: one cacheless forward when every open term is
`Trigger.BOTH`; otherwise the prefill, then decode positions in runs where
every trigger fires alike, on one KV cache with `is_prefill`, `gen_step` and
`sink_column` set as the decode loop sets them. A probe-gated node binds a
gate-only capture (`ReadPlane.gate_run`) and steps each decode position alone,
feeding `ctx.probe_scores` from the step before. A thinking-phase window on a
family that thinks follows a delimiter machine the node does not record, so a
steered replay under one raises `SteeringCompositionError` (422), as does a
gate in an ambient scope that is not the node's. Token readout, token anatomy
and unit activity share this split, so they read one number for one token
(`tests/loom/test_replay_parity.py`).

**Node replay.** `NodeReplayer` runs one forward over a node's tokens with
forward hooks where parts meet the residual (block input, `mid`, output before
and after the steering hooks, both projections' inputs and outputs), stopping
after the last block; hooks see what each module used (after a unit
ablation's pre-hook). Everything leaves in one packed fp32 transfer, lens reads
included via `plan_reads`.

- **`token_anatomy(node_id, raw_index, k=8, units=8, patterns=4)`**: per layer
  the residual at `out` read as a residual, and as directions the attention
  write `mid − input` (at `mid`), the MLP write `out_pre − mid` and the steering
  injection `out − out_pre` (`input + attention + mlp + steering == out`,
  pinned); the top neurons and heads by direct contribution to the token's
  logit through the R lens (`u = r·γ ⊙ W_U[v]`, the final norm's scale held and
  the softcap's slope folded in, `g_L = J_Lᵀ u` at `out` for neurons and `mid`
  for heads; pinned against autograd), each with `activation_over_scale`; and
  the attention rows of the top `patterns` heads (eager attention only then).
- **`unit_activity(node_id, top=32, units=(), raw_index=None, layers=None)`**:
  anchored at the last token so one forward covers every producing position;
  reduces on device (`drowse/units/AGENTS.md`). A raw first turn's row 0 has no
  producing position: it reads 0 like the sink.

## Diff, filter, transcripts, joint logprobs

`text_diff` is word-level `SequenceMatcher(autojunk=False)`; `per_token_diff`
aligns only where cumulative UTF-8 byte offsets and texts both match;
`steering_delta` subtracts by canonical `Steering.alphas` key.

```
filter := clause ("," clause)*                              # AND
clause := [("agg" | "any" | "last") ":"] probe op number    # default agg
op     := ">" | ">=" | "<" | "<="
probe  := [ns "/"] NAME ["." NAME] | "sae/" INT
```

This filters what the loom persisted (separate from `@when:`). `agg:` reads
`aggregate_readings`; `any:` (max for `>`, min for `<`) and `last:` read the
rows' `measurements.scores`. A missing probe makes the clause false. Keys go
through `parse_gate_ref`: only axis 0 is persisted, so any other channel is
`FilterParseError` (400); names are ASCII.

**Transcripts.** YAML, `drowse_transcript: 3`. Header `model_id`,
`system_prompt`, optional `cast`, `probes` (`name` + `sha256`); turns `role`,
`text`, optional `speaker` (`role_label`), `thinking`, `recipe`, `readings`,
`finish_reason`. The reader requires `role`/`text` and ignores other keys; a
non-slug label, a non-finite reading, or a `finish_reason` outside
`FINISH_REASONS` or on a recipe-less turn is `TranscriptFormatError`.
`import_into` is one `graft`; recipe turns re-attach as generated nodes;
imported nodes have no rows (not forkable). Modes `default` (off the root),
`here`, `merge` (under the deepest matching active-path user turn); guards
warn on model, system-prompt and probe mismatch (`strict=True` raises on probe
drift). `replay` regenerates each generated turn as a sibling with its recorded
steering (`""` when unsteered), sampling, seed, thinking, system prompt and
seat, and diffs recorded → replayed.

**Joint logprobs** (`session.joint_logprobs(a, b)`) force-replay both nodes
under their recipes and report per aligned position both chosen-token
logprobs, the cross values, `rank_changed` and `approx_kl` (KL(P_A‖P_B) over
the union of each side's top 32). Rows are sampler support pools after the
decode loop's logit edits, so off-pool is `-inf`, exact for the sampler, with
no dense vocabulary table. Position-independent steering without penalties
replays teacher-forced in 128-token chunks; otherwise one forced token per
forward. The server caches results (sorted-pair key, LRU of 8); the session
clears the cache on `edit`, `delete`, `reset`, `finalize_assistant`, restore
and close.

## drowse.ai seam

drowse.ai ports the tree ops and events, `compose_call`, the recipe, node and
tree JSON, the filter grammar, diff and transcripts, and computes parity from
its pinned copy of this repo; there are no goldens here, so a change to any of
them is an interface change it catches up to. The dashboard shares
`tests/fixtures/token_rows.json` and `loom_tree_v2.json` with these readers.

Tests: `tests/loom/` (CPU); `tests/session/test_cast.py`,
`test_conversation.py` on `tiny_session`; routes in
`tests/server/test_server_loom.py`.
