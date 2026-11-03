# drowse/loom — AGENTS.md

The conversation layer: the branching tree a session generates into
(`LoomTree`), the receipt on each generated turn (`Recipe`), and the one rule
that composes a generation's effective recipe (`compose_call`). It also holds
the cast roster, conversation ops, token replay, diff, joint logprobs, the
filter grammar and YAML transcripts. Nothing here owns a model. Neighbours:
`session/engine.py` (`GenerationEngine.compose_call` gathers the tiers,
`TokenTap` writes rows), `session/session.py` (`_start_loom_assistant` stamps
recipes; the verbs, `restore_tree`, the joint-logprob cache),
`session/authored_capture.py`, `generation/{finalize,render}.py`,
`model/scene.py` (seats, labels, think delimiters), `server/tree_routes.py`.

## Module map

| Module | Owns |
|---|---|
| `tree.py` | `LoomNode`, `LoomTree` (mutators, reservation, cast roster, persistence), `TokenRow`, `PromptCapture`/`prompt_digest`, `TREE_FORMAT_VERSION`, `TOKEN_SIDECAR_FORMAT_VERSION` |
| `recipe.py` | `Recipe`, `CastMember`, `compose_call`, `CALL_ONLY_SAMPLING_FIELDS`, regen modifiers, `derive_seed_schedule` |
| `ops.py` | `append_turn`, seating/thinking checks, `resolve_token_anchor`/`resolve_prompt_anchor`, `plan_fork`/`plan_prefill` (→ `ForcedTurn`), `begin_generation_turn` |
| `replay.py` | `TokenReplayer` (implements `readout/replay.py::TokenReplay`; `forward` hands the producing ids to a caller's own forward; `segments`), `ReplayForward`, `replay_segments`/`ReplaySegment`, `drive_segments`, `capture_pooled` |
| `node_replay.py` | `NodeReplayer` (`session.token_anatomy`, `session.unit_activity`), `tap_token`/`TokenTaps`, the contribution math, the anatomy record builders, `UnitActivityJSON`, the population encoding |
| `diff.py` | `text_diff`, `readings_diff`, `per_token_diff`, `node_diff`, `steering_delta` |
| `filter.py` | `parse_filter`, `filter_tree`, `node_token_series`, `FilterParseError` |
| `transcript.py` | `Transcript`, `Turn`, `ProbeRef`, replay results, `Transcript*Error` |
| `joint_logprobs.py` | `JointLogprobReplayer`, `JointLogprobs`, `SamplerRows`, `reorient_for_request` |
| `errors.py` | `LoomTreeError` 400, `UnknownNodeError` (KeyError) 404, `InvalidNodeOperationError` (ValueError) 400, `MutationDuringGenerationError` (RuntimeError) 409, `RecipeOverrideError` (RequestError) 400; strict-reader helpers |

## Layering

`loom` sits above `fit` and below `session`. `TYPE_CHECKING`
imports count too. Lower layers never import
loom: `generation/render.py` types the tree as `ConversationHistory`, and
`generation/finalize.py` takes `tree: Any`.

Collaborators take the tree as a getter (`tree: Callable[[], LoomTree]`),
because `restore_tree` replaces `session.tree`. Only `transcript.py` takes a
duck-typed `session`: it uses `tree`, `model_id`, `config.system_prompt`,
`probe_hashes()` and `generate`.

## Tree data model

- **Root.** The root is a synthetic, empty, parentless `system` node.
  `messages_for` and `flat_text` skip it; a system prompt is never a node.
- **Provenance is recipe presence.** A node with a `Recipe` is generated in
  either seat, since the model can generate into the user seat.
  `recipe is None` is authored. `rewind`, `anchor_recipe`,
  `set_authored_capture`, token replay, transcript import and joint logprobs
  key on this. Never infer provenance from `role`.
- **`role_label`** is the cast label the turn was sent under (`None` = family
  default). It is stamped at creation and immutable, and is unrelated to the
  transient `SteeringComposer.active_role`.
- **`thinking_text`** is committed by the author or stamped at finalize, and
  finalize stamps it only when the scene grammar has `think_open`. A block the
  family can't re-render would make every later render raise.
- **Token rows.** `tokens`/`thinking_tokens` hold opaque row dicts from
  `TokenTap` (browser-seam shape), never mutated, so a shallow list copy is a
  stable snapshot. `raw_token_ids` is the exact decode sequence, delimiters
  and unmerged partial UTF-8 included: the forceable prefix for forks and
  replays, joined to rows by `raw_index`. Authored and imported turns have
  `tokens=None` until authored capture adds rows.
- **Prompt capture.** An authored turn's rows come once, with their replay
  anchor `prompt_capture` (`PromptCapture`): the capturing call's steering,
  the render's resolved system prompt, thinking flag and flat mode, each
  channel's first render position (`start`, `thinking_start`, `None` for a
  channel the capture did not find) and `digest`, the `prompt_digest` (SHA-256
  of the comma-joined decimal ids) of the render before the later channel. The
  rows carry no `raw_index`: a token's address is its place among the node's
  captured rows, thinking rows first. `set_authored_capture` sets both
  channels and the anchor in one mutation and refuses a node already captured,
  so the address never shifts; `append_authored` clears all three.
- **Readings.** `aggregate_readings` holds axis-0 finalize aggregates from all
  four read families; `aggregate_measurements` the same aggregate envelope as
  a node keeps it (`readout/measurements.py::aggregate_measurements`: each
  reading's summary, no traces, no `scores`), which the rack and the probe
  inspector draw after a reload, `None` for an authored or imported turn and
  a reply that measured nothing; `mean_logprob`/`mean_surprise` cover the
  response span.
- **Edits.** `edit` rewrites `text` only. The recipe, rows and
  `raw_token_ids` keep the original decode, so forks, replay and joint
  logprobs read the original; `edit_count`/`edited_at` are the only signal.
  An authored node's prompt replay takes its own tokens from its rows, so an
  edit of its text reads the original too; an earlier turn edited since the
  capture fails the anchor's digest and the replay refuses (400).
  `append_authored` instead clears the receipt, making the node authored.

## Mutation rules

Public methods take the tree's `RLock`. Each mutation bumps `rev` and emits
one `LoomMutated` (`drowse/events.py`); surfaces refetch on a `rev` gap.
No-op calls emit nothing, and hot-loop `append_token` neither bumps `rev` nor
emits (deltas ride the token stream). Ops: `add_user`, `navigate` (also
`rewind` and a dedup hit), `begin_assistant` (also `begin_continuation`, as
`updated`), `finalize_assistant`, `capture_authored`, `edit` (also
`append_authored`), `branch`, `delete`, `star`, `note`, `reset`, `import`
(`graft`), `cast` (no node ids), `restore` (`announce_replacement`).

**Reservation.** `begin_generation_turn` reserves the parent of the node a
generation streams into, and `session._release_generation` clears it in a
`finally`. While set, `edit`, `append_authored` and `delete_subtree` refuse
the reserved node, its ancestors and its descendants (siblings of the
streaming node included), and `reset` refuses outright, all with
`MutationDuringGenerationError` (409). New turns, `branch`, `graft`,
`navigate`, star, note and cast ops stay free. `replacing` doesn't check it,
so `session.restore_tree`'s caller must exclude generation (the server holds
its session lock).

| Op | Behavior worth knowing |
|---|---|
| `add_user_turn` | Reuses a user child with identical text, `role_label` and `thinking_text` (regen and prefill rely on this) |
| `add_assistant_turn` | Authored turn through the same begin/finalize pair: `tokens=None`, `raw_token_ids` kept, so it stays forkable; `dedup_existing` reuses an identical assistant child outside the reservation (a recorded message list relies on this) |
| `begin_continuation` | Resets a node in place; the caller replays its old text as the forced prefix |
| `branch` | Authored sibling. Copies `role_label`; copies `thinking_text` only for unchanged text; `role=` overrides the seat; refuses the root |
| `graft` | Validates all turns (user/assistant seats, role slugs) first, then chain + absent cast labels + note in one `rev` |
| `delete_subtree` | Re-seats an active node inside the subtree on the surviving parent |
| `reset` | New root (old id in `removed`); the cast survives and `rev` keeps counting |
| `rewind` | Up one submission: two nodes from a generated node, one from an authored one. No-op only at the root |

## Persistence

`tree_format` 2 and `token_sidecar_format` 3 are required exactly. Readers are
strict (`errors.require_fields`: every written field required, unknown keys
rejected) and never migrate. `from_dict` also checks structure (`children_of`
matches node ids, one agreeing parent, all reachable from a parentless
`system` root, no cycles, all four token fields (`tokens`,
`thinking_tokens`, `raw_token_ids`, `prompt_capture`) or none) and refuses a token
row whose `measurements` envelope is not `MEASUREMENTS_VERSION` (2): rows
arrive through the sidecar and inline through `PUT /tree` (the dashboard's
restore), and a 5.x row's lens binding has another shape. It also refuses a
row whose lens readout entry (aggregate or per-layer) lacks its vocabulary
`id`: version 2 was never released without one, readers key the readout by
id, and an unreleased pre-6.0 row or an outdated producer would collide.
A row's keys are exactly `token_id`, `text`, `logprob`, `sampler_entropy`
(the browser runtime's), `perplexity`, `raw_index`, `top_alts` and
`measurements` (each optional). A generated row's envelope is
`readout/measurements.py::token_row_measurements` of the token's envelope:
`scores` and each geometry reading's `fraction` and `coords`, no other
channel (the token view replays the rest); an authored (prompt) row is the
same projection. A row kept whole (a call that set
`persist_per_layer_scores`, generated or authored) is
`summary_measurements(per_layer=True)`: each geometry reading carries its
summary and all three traces. An authored node has rows exactly where its
`prompt_capture` places a channel (an empty list is no rows), each with its
`token_id` and no `raw_index`, thinking before visible; a generated node has
no capture. So authored rows saved before the anchor (kept whole, with none)
are refused; `tests/fixtures/token_rows.json`'s `authored` cases pin the rule
for both readers. The reader refuses the earlier unreleased
sidecar-3 shapes by what they carry: a row-level `probes` or
`per_layer_scores` copy, an envelope `per_layer_scores` map, a geometry
reading with `subspace_coords_per_layer` or with its summary but not all of
its traces, and a row mixing generated readings with whole ones. A node's
`aggregate_measurements` is null or an `aggregate`-scope envelope of this
version, without `scores`, whose geometry readings carry exactly their
summary.

A node's `finish_reason` is `null` (authored, imported without one, or still
streaming) or one of `generation.results.FINISH_REASONS`; the reader refuses
any other string. `"cancelled"` is a reply a client Stop or a disconnect cut
short: the engine reports it and `finalize_assistant` records it, and the
dashboard marks the reply.

`to_dict()` omits token payloads; `GET /tree` sends `include_tokens=True` and
`PUT /tree` reads it back. `save` writes the JSON atomically after a gzip
`<stem>.tokens.json.gz` sidecar when any node has rows or `raw_token_ids`
(else removes a stale one). `load` caps the JSON at 64 MiB and the
decompressed sidecar at 256 MiB (`max_bytes=` overrides both). The reader
skips `structural`/`observed` cast entries, so the server's effective-roster
JSON round-trips. `LoomTree.replacing` refuses another model's tree (null
`model_id` passes), keeps the event bus and `session_id`, and sets
`rev = max(prev, saved) + 1`. The owner installs it, then calls
`announce_replacement` (WS subscribers serialize `session.tree` on that
event). There is no autosave.

## Recipes and composition (`recipe.py`)

A stamped `Recipe` carries `steering` (canonical; `None` when unsteered),
`sampling` (the composed config, including the seat labels and observation
knobs), `thinking`, `system_prompt` (resolved; `None` in raw mode), and
`probes`/`probe_hashes` (geometry probes only; a transcript header uses
`session.probe_hashes()`, all families). The seed lives only in
`sampling.seed`: on disk a top-level `seed` mirrors it and wins on read, and a
seed without a sampling block is `seed_only` (writes `"sampling": null`; build
with `Recipe.from_parts`). Every `SamplingConfig` field is persisted and
required on read, so a new sampling field changes saved trees, transcripts
and the browser seam. An absent `system_prompt` means inherit at use; `None`
and `""` are captured values.

`compose_call` is the single composition rule (the browser edition mirrors
it; golden `tests/fixtures/browser/recipe-compose-v1.json`). Tiers, weakest
first: `cast`, the call's kwargs, `anchor`, `modifier`. A stronger tier sets
`steering`/`thinking` when not `None` (`""` is explicit unsteered and blocks
weaker tiers) and `system_prompt` when not inherited, and merges `sampling`
field-wise via `SamplingConfig.merged_with` (a default field is unset, so no
tier can reset a field to its default). Sampling ranks the call above the
anchor (cast < anchor < call < modifier): a field the call sets wins over the
turn it anchors on, so a regenerate that sends no sampling reproduces the
anchor exactly and a variation runs at the call's `max_tokens`; the
dashboard's comparisons send no generation sampling for that reason. Two
exceptions:
`CALL_ONLY_SAMPLING_FIELDS` (`logprobs`, `return_hidden`, `return_top_k`, both
`persist_*` flags, `return_probe_readings`) always come from the call, so a
recipe never decides what a consumer captures; and the seed resolves in the
order modifier > call > anchor > cast, so fanned per-sibling seeds beat the
anchor (N distinct samples) while an unseeded one-shot regen reproduces it.

`GenerationEngine.compose_call` supplies the tiers. The cast member is
`tree.cast[label]`, the label being the call's seat role label or else the
seat name (a member named `assistant` covers every unlabeled assistant turn);
raw mode skips the roster. Anchor and modifier exist only with
`recipe_override`; the anchor is `anchor_recipe(parent_node_id)`, the parent's
recipe if generated, else the nearest generated ancestor's. A regen passes the
regenerated node's parent, so the anchor is never that node or a sibling. A
composed `""` becomes no steering before parsing.

Modifiers (`Recipe.compose_modifier`): `unsteered` (steering `""`), `inverted`
(every coefficient negated; `along` on a `%` term, `onto` kept), `reseed`
(fresh 31-bit seed), `cool`/`hot` (temperature 0.3/1.2), a `Recipe`
passthrough, or a partial `"field=value, …"` over `steering`, `thinking`
(`true`/`false`), `seed`, `temperature` ≥0, `top_p` [0,1], `top_k` ≥0,
`max_tokens` ≥1, `presence_penalty`/`frequency_penalty` [-2,2]. It splits only
at `, <ident>=` (steering may contain commas) and parse-validates steering;
errors are `RecipeOverrideError`.

Gotchas: a modifier seed beats the call, so `seed=42` on a fan pins every
sibling and `reseed` gives each a fresh seed instead of the schedule.
`inverted` on a curved `%` term composes, but the steering plan refuses the
negated `along` (`SteeringCompositionError`, 422); curvature is per-model, so
the refusal is the composer's.

`derive_seed_schedule(base, n)`: `n=1` returns the base (31-bit), `n>1`
derives cross-machine-deterministic `blake2b(base, i)` seeds, `None` draws
entropy. Each sibling stamps its own seed (`session/batch.py`, the WS fan).

## Cast roster

`tree.cast` maps label → `CastMember` (optional standing `Recipe` + `notes`).
`cast_roster()` is the effective view: structural `user`/`assistant`, every
label observed anywhere in the tree (not just the active path, so speakers
don't flicker on navigation), then configured members on top. Roster ops are
decoration-tier and validate the label as a role slug.
`LoomTree.set_cast_member` checks only the label; `session.set_cast_member`
builds through `CastMember.from_parts`, which parse-validates steering at
authoring time. A fieldless member contributes nothing; steering `""` pins a
label unsteered.

## Conversation operations (`ops.py`)

- `append_turn` is the one authored-append rule. Same seat and label as the
  parent concatenates via `append_authored` (`raw_token_ids` = tokenization
  of the full text); otherwise a new user turn (deduping) or authored
  assistant turn lands. Seating is free in scene mode (`grammar` set) and flat
  mode. A legacy chat template renders only alternation, so there a user turn
  may not hang under a user node and an assistant turn must.
- `check_thinking_commit` raises `SceneThinkingUnsupportedError` (400) at
  commit time when the grammar has no `think_open`.
- `begin_generation_turn` opens and reserves. A `str` input commits the user
  turn first (label from `sampling.user_role`); an assistant-seat send refuses
  a user parent even in scene mode (commit with `append_turn`, then generate
  from `None`). Flat mode lands a non-empty span as its own node and lets an
  empty one extend the parent. `continuation_node_id` reuses a message. A
  message list is a whole conversation: `split_message_list` checks the whole
  list before anything is recorded (a leading system message becomes the
  call's system prompt; it refuses a `parent_node_id`, raw mode, other roles,
  empty content, and non-alternating turns on a legacy template) and returns
  `MessageTurn`s. They land from the root, each message its own turn, reusing
  an identical child at each step (`add_user_turn` / `add_assistant_turn`
  with `dedup_existing`), so a re-sent list shares its whole history and its
  replies are siblings; no existing node is edited. The reply hangs under the
  last turn and the engine renders from those turns, so it replays the prompt
  it answered.
- `resolve_token_anchor` is the one `(node_id, raw_index)` validation, shared
  by forks and replay; `TokenAnchor` snapshots `raw_ids`.
- `plan_fork` replays `raw_ids[:raw_index]`, forces `alt_token_id` or
  `replacement_text`'s tokens (exactly one), then samples under the node's
  recipe steering, sampling, thinking and system prompt. The original seed
  re-applies by default and the sampler still draws every forced step
  (`generation/decode.py`), so the RNG stays aligned through the fork; `seed=`
  asks for a new ending. `max_tokens` grows by the forced span. The sibling
  keeps the node's parent, seat and `role_label`.
- `plan_prefill` opens an assistant reply to a user node with the text,
  thinking off, re-sending the user text so dedup reuses the node. It is
  Python-only (`session.prefill_assistant`); forks reach the WS via `generate`.

Gotchas: a fork passes the recipe's steering verbatim, so a node stamped
unsteered (`None`) picks up its label's cast steering if a member now exists
(transcript replay passes `""` for exactly this reason). Prefill carries
neither the node's `role_label` (unless `sampling.user_role` matches) nor its
`thinking_text`, so a labelled or thinking-carrying user node gets a duplicate
sibling.

## Token replay (`replay.py`)

`TokenReplayer` backs every family's `token_readout`. It resolves the anchor,
rebuilds the node's continue-mode render to its parent (stamped labels, recipe
thinking and system prompt, node seat), appends `raw_ids[:raw_index]`, runs
the capture forward (`capture_pooled` through `drive_segments`, segmented as
below), and hands `{layer: (hidden,) fp32}` pooled at the producing position
to the family's `read`. The steering scope opens
**outside** the exclusive-GPU section (`base/exclusive.py::ExclusiveGuard`):
composer push/pop take the generation lock blocking, so nesting would
deadlock. The render (which uploads the ids to the device), `layers` and
`read` all run inside it: a device upload outside the section once raced a
lock-free device read for MPS's one command buffer and aborted the server. `apply_steering=False` reads
the unsteered counterfactual.

An authored node routes through `resolve_prompt_anchor` instead (`raw_index`
indexes its captured rows, thinking first): the render of the path to the
node with the capture's system prompt, thinking flag and flat mode and no
generation prompt, cut at the token's channel and checked against the
anchor's digest (a mismatch refuses, 400), then the channel's tokens before
it from the rows, as one prefill under the capture's steering. A prefill
fires a term iff it has a prompt slot and no probe gate (`Trigger.active`),
so the whole expression replays the capture's prompt-phase steering and
`replay_segments` refuses no gate or phase window when there is no decode
position. `raw` does not apply (the anchor records the render). The node replay's
unit activity reads an authored turn's every row in one prefill: it anchors
the row latest in the render, and `ReplayForward.spans()` places each
channel's rows at their producing positions (a gap between the thinking and
visible channels included). A raw first turn's row 0 sits at render
position 0, which no position produces: it reads 0 like the sink's rows,
and only a population asked there refuses (400). The forward fires the open terms as generation
fired them (`TokenReplayer.segments`, the **Triggers** rule below, shared with
the node replay), so a family's token readout, the token anatomy and the unit
activity read one number for one token; a probe-gated term replays one decode
position at a time, and a thinking-phase window on a family that thinks
refuses (422) in every steered replay.
`raw=True` picks the flat render, which the node does not record.

`TokenReplayer.forward(node_id, raw_index, run=…)` is the same rebuild
without the capture: `run` receives a `ReplayForward` (`ids`, `prompt_length`,
the node's `raw_ids`, the anchored `raw_index`; raw token `i` is produced at
`prompt_length + i − 1`) inside the exclusive section under the scope, and
owns its forward. `__call__` is its capture use. An authored token's
`ReplayForward` is all prompt (`prompt_length == len(ids)`), with the node's
captured ids as `raw_ids`; the token anatomy reads it unchanged.

## Node replay (`node_replay.py`)

`NodeReplayer` (session-wired over the token replayer, the model, the lens
instrument, the units runtime, the tokenizer, the tree getter, the chat
traits, the steering manager's `TriggerContext` and the open stack's
triggers) runs one forward over a node's tokens with taps where the model's
parts meet the residual, stopping after the last block (no vocabulary head
over every position): each block's input, its `mid` (the MLP input norm's
input), its output before the steering hooks (a `prepend` hook) and after
them, the down projection's input and output, and the output projection's
input and output. Every tap is a forward hook, so it sees what the module
used (after a unit ablation's pre-hook). Eager attention (the attention
config's `_attn_implementation`, restored after) runs only when a pattern can
be returned: `patterns > 0` and the R lens holds a `mid` layer to rank heads by.

**Triggers.** Generation prefills the prompt in one forward and decodes one
token per forward, and a trigger decides per forward. `replay_segments` (in
`replay.py`, through `TokenReplayer.segments`, so the token replay splits the
same way) splits the replay the same way: one cacheless forward when every open term is
`Trigger.BOTH` (exact, the common case); otherwise the prefill, then the
decode positions in runs where every trigger fires alike (`@first:N`,
`@after:N` boundaries, judged by `Trigger.active` itself), run on one KV cache
with `is_prefill`, `gen_step` and `sink_column` set as the decode loop sets
them and the rest state restored after. A probe gate fires on live probe
scores the node does not record: when the node's own steering has one,
`TokenReplayer.forward` binds a gate-only capture (`gate_reads`, the session's
`ReadPlane.gate_run`, the same binding joint logprobs uses) and
`replay_segments(gate=...)` steps every decode position as its own forward,
`drive_segments` setting `ctx.probe_scores` from the scores after the one
before, as the decode loop does. On a family that thinks, a thinking-phase
window (`@after`, `@thinking`) follows the decode loop's delimiter machine,
which the node does not record, so a steered replay under one raises
`SteeringCompositionError` (422), as does a gate in an ambient scope that is
not the node's; `apply_steering=False` replays without the node's own terms.
Pinned against the live unit readings of `@prompt`, `@response`, `@first:2`,
`@after:2` and `@when:` generations.

- **Token anatomy** (`token_anatomy(node_id, raw_index, k=8, units=8,
  patterns=4)`): per layer the residual at `out` read as a residual
  (`reads_as`, S centred), the attention write `mid − input` read at `mid`, the
  MLP write `out_pre − mid` and the steering injection `out − out_pre` read at
  `out` as directions (`input + attention + mlp + steering == out`, pinned),
  `steering` `None` where nothing was injected; the top `units` neurons and
  heads by direct contribution to the token's logit through the selected R
  lens: `u = r·γ ⊙ W_U[v]` (the final norm linearized at the final residual,
  its scale held; the softcap's slope folded in), `g_L = J_Lᵀ u`, neuron `act_n
  · (s · γ_out ⊙ W_down[:, n] · r_mlp) · g_out`, head `Σ_i (γ_attn ⊙ g_mid ·
  W_O)[h·hd+i] z_i · r_attn` (`r` a sandwich norm's per-position scale, held;
  1 without one), `g` at `out` for neurons (the identity at the last layer,
  the R target) and `mid` for heads, pinned against an autograd Jacobian;
  across layers the top `units`; and the attention rows of the top `patterns`
  heads from the token's producing position. Each layer's unit row carries its
  raw activation and `activation_over_scale` (over the unit's reading scale,
  the bound catalog's `act_top` or the neutral statistics', soft-floored:
  `units/stats.py::UnitScales`; `units_scale` names which), so a unit shows the
  number its probe, unit activity and the unit view show. No R lens:
  `units_absent`.
  It returns the `scope="replay"` envelope's `anatomy`, beside a lens channel
  with the attached `jlens/`/`slens/` probes' readings of the same residual
  (a loom row keeps only their values).
- **Unit activity** (`unit_activity(node_id, top=32, units=(), raw_index=None,
  layers=None)`): anchored at the node's last token, so one forward covers
  every token's producing position; per-layer hooks reduce on the device
  (`drowse/units/AGENTS.md`). Each peak's `label` is `None` here; the session
  fills it from a bound catalog (the label of the pole that peaked, a head's
  label), so the Units room can list and filter labelled units.

Everything a result carries is packed into one fp32 vector and moved to the
host once (`_Pack`); the lens reads ride the same transfer through
`LensInstrument.plan_reads`, and so do an anatomy's context ids (exact in fp32
below 2²⁴). Unit activity derives its sink row on the host from
`prompt_length`. Only decoded tokens (a turn's raw record) are
addressable, as with `token_readout`; prompt render positions appear only as
pattern context. On SmolLM2 (MPS, warm) an anatomy takes about 0.3 s, a
24-token activity about 0.15 s.

## Diff (`diff.py`)

`text_diff` is word-level `SequenceMatcher(autojunk=False)` on
whitespace-split tokens (replace = delete then insert; whitespace not
preserved). `readings_diff` is per-probe `b − a` sorted by `|Δ|`, an absent
side reading 0.0 with originals kept. `per_token_diff` aligns only where
cumulative UTF-8 byte offsets and texts both match; tails are unaligned at
index `-1`. `node_diff` sets `parent_id` only for siblings. `steering_delta`
subtracts by canonical `Steering.alphas` key (bare atoms resolve against
installed manifolds) and returns `""` for no label, never a sentinel.

## Filter grammar (`filter.py`)

```
filter := clause ("," clause)*                              # AND
clause := [("agg" | "any" | "last") ":"] probe op number    # default agg
op     := ">" | ">=" | "<" | "<="
probe  := [ns "/"] NAME ["." NAME] | "sae/" INT
```

Deliberately separate from `@when:`, which gates live per-step readings; this
filters what the loom persisted. `agg:` reads `aggregate_readings`. `any:`
(max for `>`/`>=`, min for `<`/`<=`) and `last:` read the node's own rows,
thinking first, from `measurements.scores` (at the row's 6 significant digits),
skipping rows without the probe. A missing probe makes the clause false. No
OR, negation or grouping; compose with `LoomTree.filter(pred)`. Keys go
through `readout/types.py::parse_gate_ref`: `name[0]` normalizes to `name`,
and any other channel is a `FilterParseError` (400), since only axis 0 is
persisted. Names must be ASCII `[ns/]name[.pole]`, `jlens/#<id>` /
`slens/#<id>` or `sae/<id>`, so a non-ASCII lens word or a variant-suffixed
name can't be filtered. The browser ports this grammar; change both together.

## Transcripts (`transcript.py`)

YAML, `drowse_transcript: 3` exactly. Header: `model_id`, `system_prompt`
(the session default; per-turn prompts ride the recipes), optional `cast`,
`probes` (`name` + `sha256`). Turns: `role` (seat), `text`, optional
`speaker` (`role_label`), `thinking`, `recipe`, `readings`, `finish_reason`
(generated turns only; one of `FINISH_REASONS`, so `cancelled` marks a reply
a Stop or disconnect cut short). The reader is strict
(`TranscriptFormatError`).

`import_into` is one `graft`, so everything lands or nothing does. Turns with
a recipe re-attach as generated nodes in their seat; system turns are
skipped; imported nodes have no rows or `raw_token_ids` (not forkable or
replayable). New cast labels land, and a label the roster holds with a
different member keeps the session's and is flagged. Modes: `default`
(branch off the root), `here` (under the active node), `merge` (under the
deepest active-path user turn matching the transcript's user turn at the same
position, else the root; the matched prefix is skipped). Guards warn and note
the first grafted node: model mismatch (raises `TranscriptModelMismatch`
under `merge`), system-prompt mismatch, missing probes, probe drift
(`strict=True` raises `TranscriptProbeDriftError`).

`replay` imports, then regenerates each generated turn as a sibling under its
own parent (full history in context) with the recorded steering (`""` when
unsteered), sampling, seed, thinking, system prompt and seat, and diffs
recorded → replayed (`node_diff`, `a` = recorded). Fail-fast; `isolate=True`
records `DrowseError` failures and continues. CLI: `drowse experiment
transcript run`.

## Joint logprobs (`joint_logprobs.py`)

`JointLogprobReplayer.compute(a, b)` (`session.joint_logprobs`) force-replays
both nodes under their stamped recipes and reports, per aligned position,
both chosen-token logprobs, `lp_a_in_b`/`lp_b_in_a`, `rank_changed` and
`approx_kl`.

- Prompts rebuild as the node was opened (labels, seat, system prompt;
  thinking from recipe, else steering, else model default). A rowless node
  recovers tokens from the closed render, split at the shared prefix. Rows
  with `token_id < 0` (buffered partial UTF-8) are skipped; thinking is forced
  but its rows dropped.
- Rows are sampler support pools (`SamplerRows`): effective top-k with
  renormalized logprobs after the decode loop's logit edits (NaN/±inf clamp,
  penalties, `logit_bias`). Off-pool is `-inf`, exact for the sampler, with no
  dense `[rows, vocab]` table.
- Position-independent steering (`all_fast_path()` or `static_steerable()`)
  with no penalties replays teacher-forced in `_REPLAY_CHUNK = 128` chunks;
  otherwise one forced token per forward drives the trigger context, with a
  gate-only read-plane binding (`ReadPlane.gate_run`) when probe gates need it
  (runs close in a `finally`).
- Alignment is `shared_prefix_len` then `per_token_diff`; indices are relative
  to the divergence point and cross values exist only on aligned rows.
  `approx_kl` is KL(P_A‖P_B) over the union of each side's top
  `_KL_TOP_K = 32`; infinite KL reports `None`.
- The caller serializes against generation (server session lock). Unknown ids
  raise a bare `KeyError`; the route pre-checks.
- Caching is the server's: `session.joint_logprob_cache`, keyed by
  `_cache_key` (sorted pair), LRU of 8 (`server/tree_routes.py`). The session
  clears it on `edit`, `delete`, `reset`, `finalize_assistant`, restore and
  close. `reorient_for_request` swaps columns but keeps `approx_kl`, which
  stays in the first-computed orientation.

## Browser seam

The browser edition (`stdlogits/drowse.ai`) ports the tree ops,
`compose_call`, the filter grammar, diff and transcripts.
`tests/fixtures/browser/loom-operation-transcript-v1.json` (checked by
`tests/browser/test_browser_parity.py`) pins a shared script's events (`op`,
`rev`, ids, `active_node_id`) and final tree; token rows carry the generated
row's projection and nodes their `aggregate_measurements`, both pinned by
`measurement-envelope-v2.json` (`expectedRow`, `expectedNodeAggregate`).
Changing an event, the recipe, node or tree JSON, the composition rule or the
filter grammar is an interface change there.

## Testing

All CPU. `tests/loom/` covers the tree, reservation
(`test_loom_concurrency.py`), ops, composition (`test_recipe_override.py`),
diff, filter, transcripts, rows, replay and joint logprobs.
`tests/session/test_cast.py` and `test_conversation.py` run on `tiny_session`
(a real tiny model). Routes: `tests/server/test_server_loom.py`; browser
golden: `tests/browser/test_browser_parity.py`.
