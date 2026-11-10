# drowse/server

One `DrowseSession` served over OpenAI `/v1/*`, Ollama `/api/*` and the native
`/drowse/v1/*` tree (REST, SSE, one WebSocket), plus the dashboard at `/` when `web=True`
(`drowse/web/AGENTS.md`). Handlers validate, take the lock, call public session APIs and
serialize. Top layer: no session-private reads and no imports of a sibling's `_private`
helper (shared helpers live in `runtime.py`, `errors.py`, `operations.py`,
`request_helpers.py`).

## Module map

| Module | Owns |
|---|---|
| `app.py` | `create_app(session, default_steering, cors_origins, api_key, *, web, max_request_bytes)`, the only export. Registers OpenAI, Ollama, operation routes (installs `app.state.operations`), native routers, the WebSocket, then the dashboard (its catch-all must come last) |
| `runtime.py` | `session_lock(app)`, `refuse_if_busy`, `exclusive_or_busy`, `attached_probe_names` (every family's roster), `run_in_thread` / `finish_worker` / `stream_events`, `ClosingStreamingResponse`, `NativeRequest`, `session_router()` |
| `errors.py` | `describe_error` (the one classifier), `protocol_error` (the three envelopes), `validation_message` |
| `security.py` | `HttpSecurityMiddleware`, `ws_auth_ok` |
| `operations.py` | `run_operation` (SSE-or-JSON jobs), `OperationLedger` (receipts), `Preparation` / `PreparationGroup` (polled jobs) |
| `request_helpers.py` | `build_sampling_config` (the one `SamplingConfig` construction site), `compose_request_steering`, `flatten_content`, `validate_logit_bias` |
| `compat/` | `CompatGeneration` + shared driver; `openai.py`, `ollama.py`; `model_names.py` (Ollama aliases: `_HF_TO_OLLAMA_ALIASES`, else `<family>:<size>`) |
| `*_routes.py` + `*_models.py` | native REST: sessions, profiles, probes, tree, manifolds, templates, instruments, units |
| `ws_stream.py`, `ws_models.py`, `ws_events.py` | `WS /sessions/{id}/stream`, frame schemas, `GET /sessions/{id}/requests/{id}` |
| `response_models.py` | response `TypedDict`s for every non-streaming native route |

Native request bodies subclass `NativeRequest` (`extra="forbid"`); Ollama bodies are
`extra="allow"` (clients send fields drowse ignores; explicit `null` reads as unset). A
non-streaming native route annotates its return with a `response_models.py` `TypedDict`
named like the dashboard's TS interface; `scripts/generate_webui_types.py` renders
`webui/src/lib/types.gen.ts` from it (`drowse/web/AGENTS.md`). `_Open` (`extra="allow"`)
marks payloads from engine serializers, which a closed model would truncate. Only `PATCH
/sessions/{id}` mutates `session.config`; per-request sampling and steering ride the call.

## Locks and workers

| Guard | On contention |
|---|---|
| Session lock `runtime.session_lock(app)`: one FIFO `asyncio.Lock`, shared by all protocols | waits `SESSION_LOCK_TIMEOUT_SECONDS = 300`, then `SessionLockTimeout` (503) |
| Exclusive-GPU section `session.gen_lock`, taken non-blocking by the engine and every model op | `ConcurrentGenerationError` / `ConcurrentExtractionError` (409) |
| `refuse_if_busy(session)`, non-blocking probe of `gen_lock` | `EngineBusyError` (409) |
| Loom reservation (`LoomTree.reserve`) | `MutationDuringGenerationError` (409) |
| `PreparationGroup`: one lens or SAE preparation at a time | 409 |

- Forward passes and hook state run under the session lock: compat generations, a WS turn
  (the whole `n`-way fan), `steering/validate`, tree restore/reset/transcript load,
  `joint-logprobs`, instrument `live` / `source` / `token-readout` / SAE load, every
  `run_operation` job, manifold merge, delete and forced from-template.
- Lens fit and SAE train hold the exclusive-GPU section but not the session lock, so a
  generation arriving mid-fit gets an immediate 409 instead of queueing. For the same reason a
  route that replaces or removes a manifold folder calls `refuse_if_busy` even under the
  session lock, then `session.evict_artifact(ns, name)` after writing.
- Blocking work uses `run_in_thread` / `finish_worker`, not bare `asyncio.to_thread`: a
  cancelled request stays parked, shielded from AnyIO and repeated cancellation, until its
  worker exits, so the lock is never released while a worker still writes.
  `ClosingStreamingResponse` and `contextlib.aclosing` around the compat driver make a dropped
  stream join the engine worker first.

## Errors

`errors.describe_error(exc, *, failure=...) -> ErrorInfo(status, message, code)` is used by the
FastAPI handlers, `run_operation`, polled preparations and WS error frames. `code` is the
exception class name (clients key on `PoisednessError`, `RecipeOverrideError`, ...). It logs
what it scrubs.

| Exception | Status | Client text |
|---|---|---|
| `DrowseError` | `user_message()` status; a class still at base 500 falls back on its stdlib parent (`KeyError` 404, `ValueError` 400) | `user_message()` text |
| `HTTPException` raised `from` a `DrowseError` or `OSError` | the `HTTPException`'s | the cause's safe text |
| other `HTTPException` | its own | `detail` flattened to one string (sent verbatim) |
| pydantic `ValidationError` / `RequestValidationError` | 400 (not 422) | every error as `field.path: msg`, joined by `; ` |
| `OSError` | `FileNotFoundError` 404, `FileExistsError` 409, else 500 | fixed text (OS messages carry paths) |
| `ImportError` naming a third-party module | 503 | `optional dependency not installed: <module>` |
| anything else (bare `ValueError`, failed `drowse.*` import) | 500 | `<failure>. Check the server log for details.` |

Envelopes by prefix: `/api/` `{"error": msg}`; `/v1/` `{"error": {message, type, param,
code}}` (`code` the HTTP status; `type` `conflict` for 409, `invalid_request_error` for other
4xx, else `server_error`; `param` the first invalid field); else `{"detail": msg}`. After
headers are sent, errors go in band: OpenAI SSE `data: {"error": {...}}` then `data: [DONE]`;
Ollama NDJSON `{..., "error"}` then a `done_reason: "error"` frame; native SSE `event: error
{message, code, status}`; WS an `error` frame.

## Security

`drowse serve` is a trusted-local/lab server; TLS, rate limits and multi-tenancy are
deployment concerns.

- `HttpSecurityMiddleware` is pure ASGI and authenticates before body parsing on `/v1/`,
  `/api/`, `/drowse/v1/`. With a key (`api_key` or `DROWSE_API_KEY`): `Authorization: Bearer`
  via `hmac.compare_digest` (401). Without one: the TCP peer must be loopback and `Host`
  loopback or the server address, a DNS-rebinding defense (403). The dashboard shell, `HEAD /`
  and `/docs`, `/redoc`, `/openapi.json` are public and carry no session data.
- A present `Origin` on private paths (except `OPTIONS`) or any non-`GET/HEAD/OPTIONS`
  request must equal the request's own or be in `cors_origins` (403); a wildcard entry grants
  nothing; no `Origin` passes.
- `Content-Length` must be one decimal value (400); bodies over `max_request_bytes` (64 MiB,
  `DROWSE_MAX_REQUEST_BYTES`) are 413. Every response carries `nosniff`, `X-Frame-Options:
  DENY`, `no-referrer`, a closed `Permissions-Policy` and the dashboard CSP; private paths add
  `Cache-Control: no-store`.
- `ws_auth_ok` runs before `accept()` (failure closes 1008): it scrubs a `token` query
  parameter from the scope (so the access log never sees one), applies the host and origin
  rules, and takes the key as a bearer header or one `drowse.auth.<base64url-unpadded-key>`
  subprotocol (browsers cannot set WebSocket headers). Only `drowse.v1` is echoed.
- The CLI binds `127.0.0.1` and refuses a non-loopback bind without a key.

## OpenAI and Ollama (`compat/`)

Both lower to a `CompatGeneration` run by one driver (`run_generation` / `stream_generation`:
`StreamStarted`, `TokenEvent`s, then `StreamEnd` or a final `ErrorInfo`). Generations are
always `stateless=True` (the request's messages are the whole context; the loom is untouched),
streams pass `live_scores=False, live_readouts=False`, and probe data arrives only as the
end-of-generation aggregate. A client disconnect stops the engine at the next token.

- Steering (`compose_request_steering`): `None` inherits the server default
  (`create_app(default_steering=)`, CLI `-S`); an expression overrides per key; an empty string
  clears it and reaches the session as an empty `Steering` (not `None`, which the cast roster
  refills).
- Attached probes' aggregates (Monitor, lens and SAE rosters unioned) ride
  `x-drowse-probe-readings`: on each OpenAI choice, top level on the Ollama final frame.
- `DROWSE_STRICT_MODEL` (`1/true/yes/on`) 404s a `model` outside the HF id and its aliases.
- OpenAI: `/v1/models[/{id}]`, `/v1/chat/completions`, `/v1/completions` (raw prompt).
  Extensions: `steering` (expression string), `thinking` (`None` = auto). Accepted and ignored:
  `user`, `n`, `response_format` `text`, empty `tools`, `tool_choice` `none|auto`; other values
  and non-text content parts are `UnsupportedContentError` (400). Streams send a role delta
  (chat), thinking as `reasoning_content`, a final chunk with `finish_reason` and readings, an
  optional usage chunk, then always `[DONE]`.
- Ollama: version, tags, ps, show, chat, generate, pull (no-op for a hosted name, else 404),
  `HEAD /`; 501 for push, create, copy, delete, embeddings, embed. `/api/show` returns the real
  HF Jinja template. NDJSON streaming is the default (`null` does not stream). `repeat_penalty`
  p maps to `presence_penalty = ln p` when p > 1 and none is set (Ollama divides positive
  logits by p, which subtracts a per-seen-token constant). Unknown options are dropped. `steer`
  (in `options` or top level) must be a string (`OllamaBadRequest` before headers flush);
  `think` takes a bool or `low|medium|high` (any level means on). `/api/generate` with
  `system` renders through the chat template and never returns `context`.

## Native REST

Paths carry `{session_id}` but only `"default"` resolves; `S` below is
`/drowse/v1/sessions/default`. Manifolds and templates live under
`/drowse/v1/{manifolds,templates}`.

- **Sessions.** `session_info` carries `instruments`, one `{family, live, source, probes,
  capabilities}` block per family (the read plane's only listing); `jlens_fitted` is
  `session.lens.has_compatible()`, a metadata check that never pages in. `PATCH S`: explicit
  `null` clears `top_k` or `system_prompt`, `null` elsewhere preserves. `POST
  S/steering/validate` enters `session.steering(parsed)` in a worker thread holding the model
  lock non-blocking (`exclusive_or_busy`: 409 while the model is busy), so a green result
  composes as a generation would; rejection is in band (`{valid: false, error}`). A valid
  non-empty expression also returns `n_layers`, `terms` (`SteeringScope.term_layers`: key,
  fitted layers, share, selected layers) and `parts` (`grammar.term_parts`: per term in parse
  order its `format_expr` text, op, target, coefficient(s), position, clause, trigger (also as
  `phase`, `first_n`, `after_n`, `gate`) and projection), which the dashboard draws from
  without a parser of its own.
- **Profiles.** `pairwise` and `correlation` read `session.analytics_profile`, CPU snapshots
  built inside the exclusive-GPU section so polling never copies off the device during a model
  op. Mahalanobis-only: 409 without a whitener; a snapshot not built yet or a whitener missing
  a layer is 409 on `pairwise` (retryable) and `null` in `correlation`.
- **Probes.** One collection over the Monitor and the lens/SAE readout registries; each row
  has a `family` and only that family's fields. The listing and `GET
  S/probes/{name}/geometry` (whitened read frame) are lock-free and answer from host copies
  taken at attach (`AttachedManifoldProbe.node_coords_host`, `.geometry`): copying resident MPS
  tensors raced the model op holding the GPU and aborted the server. Readout probes have no
  subspace (404).
- **Tree.** Conflicts come from the tree's reservation; navigate, star, note, branch and cast
  edits never conflict. `PUT S/tree` restores atomically and refuses another model's tree.
  `cast_json` gives each label an `origin` (`configured`, `structural` for `user`/`assistant`,
  `observed`) and rides the full-tree GET and every `tree_mutated` frame. `joint-logprobs`
  replays both branches under the lock, cached in `session.joint_logprob_cache` (8 entries,
  LRU, sorted-pair key; cleared on edit, delete, reset, finalize, restore).
- **Manifolds.** Rows are `store.manifold_lifecycle.manifold_summary` (the `pack show -j`
  shape) plus `fitted_for_session`, `stale` (sidecar weight fingerprint or `nodes_sha256`
  mismatch), `resolved_fit_mode` (`pca`/`spectral` once fitted, `null` for an unfitted `auto`)
  and a fitted discover folder's geometry. The list stays cheap (`_fitted_geometry` reads only
  the safetensors header and the `node_coords` entry; an unreadable artifact reports
  unresolved geometry instead of failing). Authored creation returns soft poisedness /
  flat-axis `advisories`. `install`, `generate` (resumes; `force` starts clean) and
  `{ns}/{name}/fit` are progress operations. Steering a fitted manifold needs no route: a `%`
  term loads it on scope entry.
- **Templates.** CRUD is IO; `score` is a progress operation over `session.score_template`.
- **Instruments** (`S/instruments/{family}`); `CAPABILITIES` declares what each supports:

  | Family | `sources` | `preparations` | `token-readout` | `PUT source` |
  |---|---|---|---|---|
  | geometry | 404 | 404 | yes | 404 |
  | lens | yes | `fetch`, `fit` | yes | yes |
  | sae | yes | `fetch`, `train` | yes | 409 (loading weights is the `fetch` preparation) |
  | units | 404 | `fetch` (neuron-lens catalog) | yes | 404 |

  - Lens `sources` also returns what the dashboard setup draws: fetchable `providers`, the
    crossover `band` (`lo`, `hi`, `[w_S, w_R]` per layer), `fit_documents`, and per active row
    `compatible`. A lens fit status names its target (`source`, `local:<kind>`) and sets
    `replaceable` only after a `LensReplaceRefused`; the same request with `force` is the way on.
  - `POST live` forwards extras to `instrument.set_live` (its `TypeError` is the one rejection
    rule, 400); `top_k` is refused everywhere (readout width is the generation's
    `return_top_k`). `PUT source` (lens) is refused while a lens preparation runs, then turns
    live off, selects, and turns it back on. `token-readout` runs `instrument.token_readout`,
    which builds the whole `scope="replay"` envelope and rejects knobs it cannot honor (400).
    `sae/features/metadata` is network and disk cache only, no lock.
  - Node replay (`loom/node_replay.py`, under the session lock): `GET token-anatomy`
    (`node_id`, `raw_index`, `k`, `units`, `patterns`, `steered`, `raw`) returns the replay
    envelope with `anatomy`; `GET units/activity` (`node_id`, `top`, `units`, `raw_index`,
    `layers`, `steered`, `raw`) returns `UnitActivityJSON`. Both answer 422
    (`SteeringCompositionError`) for steered thinking-phase terms the replay cannot reproduce
    (a probe gate replays one position a forward). On an authored node (no recipe) `raw_index`
    addresses its captured prompt rows (thinking rows, then visible) and replays through
    `prompt_capture` (`raw` does not apply); a prompt whose earlier turns changed since capture
    is 400.
- **Units** (`unit_routes.py`, `S/instruments/units/`, literal segments beside `{family}`): `GET
  read?unit=&k=&hide_hubs=`, `GET top?token=&by=writes|fires&layers=&k=`, `POST find
  {selector | vector + layer, k, layers}` (exactly one target, else 400) run under the session
  lock. `GET catalog` and `GET labels?units=a,b` (at most 128 units) are disk only. The catalog
  fetch is the units `fetch` preparation (no session lock; an uncatalogued model is 404 at
  POST; `DELETE` is 409, a fetch is not cancellable).

## Long-running operations (`operations.py`)

**Progress operations** (`/extract`, manifold `install` / `generate` / `fit`, template
`score`) go through `run_operation`: a server-owned task holds the session lock for the job's
real lifetime and the response is a view of it.

- `Accept: text/event-stream` waits for the lock unbounded, then streams `event: progress
  {message}` and one terminal `event: done <payload>` or `event: error {message, code,
  status}`, with a `: heartbeat` comment after 15 s idle. Progress is coalesced into a 256-line
  queue (oldest dropped); the terminal frame never waits on a consumer that left. Otherwise one
  JSON body under the bounded lock wait. Without a receipt, an SSE client that leaves while the
  job is queued withdraws it; a started job runs to completion.
- **Receipts.** `X-Drowse-Request-Id` (1-128 of `[A-Za-z0-9._:-]`) admits the operation into
  `OperationLedger` under a fingerprint of `[kind, path, model_id, body]`; a reused id is 409,
  128 active receipts 503. `GET /drowse/v1/operations/{id}` returns `state`
  (`queued|running|completed|failed|interrupted|cancelled`), the last 64 progress lines,
  `result`, `error`; 404 means unknown or evicted. The ledger keeps 256 records and 16 MiB of
  payloads (4 MiB each; past budget the oldest finished payloads drop and are marked
  `result_truncated`). Shutdown cancels queued jobs (`interrupted`) and joins started ones.

**Preparations** (lens `fetch` / `fit`, SAE `fetch` / `train`) are polled: `POST
S/instruments/{family}/preparations {operation, ...fields}` (202), `GET` for running-or-latest
status, `DELETE` to cancel. Status: `{state: idle|running|done|error, operation, progress:
{current, total, unit} | null, message, error, started_at, finished_at, cancellable,
...extras}` (fit counts `prompts`, train `tokens`, parsed from the job's progress lines). A
failure reports `message: "<label> failed"`; a cooperative cancel settles to `state: "done",
message: "cancelled"`. `DELETE` cancels a running lens fit (else 409; a fetch is not
cancellable) or SAE train (else unloads the resident SAE).

- Lens `fetch`: a provider from `LENS_PROVIDERS` (400 at POST), downloaded into the HF cache
  without the lock, then selected and live.
- Lens `fit {kind: r|j|s|k = r, sites: [out, mid], prompts = 64, ...}` writes `local:<kind>`;
  `layers=sample` is 400. The corpus is pinned in an isolated spawned worker (cancellable,
  never resolved in the server process). Unless `force`, a proven no-op
  (`session.lens.fit_is_noop`) only selects `local:<kind>`. A fit or no-op including the `out`
  site selects through `select_source` in its own family's slot (evicting only that family's
  resident); an R-family one turns live readout on afterwards, an S-family one does not; a
  `mid`-only fit selects nothing (`key.lens_selectable`).
- SAE `fetch` takes `local:<name>` or `saelens:<release>` (a bare release is 400) and loads
  under the lock. SAE `train` streams `ceil(tokens / seq_len)` default-corpus documents, then
  trains.
- Live readout after a fit or train is best-effort: failure is logged, not a failed job.

## WebSocket: `WS S/stream`

Inbound (`ws_models.py`):

- `submit` is the authored-turn contract: `text` + `authored_role` appends a span;
  `generated_role` then generates from it in that seat, and alone continues from the selected
  leaf. Append-only uses `session.append_turn` (no decode); append+generate commits once inside
  the generation worker. The turn's cast label rides `sampling.user_role` / `assistant_role`.
- `generate` is the specialist frame: continue (`input: null`), explicit message lists, seats
  (`generate_seat`), `recipe_override` (via `loom/recipe.py::compose_call`) and the token fork.
  `stateless` defaults `true`; `system_prompt` omitted inherits the default, `null` sends none.
  A fork needs `fork_node_id`, `fork_raw_index` and exactly one of `fork_alt_token_id` /
  non-empty `fork_replacement_text`, and takes steering and sampling from the source node's
  recipe. A message's `label` is the per-turn cast label the scene stitcher renders.
- `stop {request_id?}` stops the running turn (only a matching one if `request_id` is set).

Outbound:

- `started {generation_id, node_id: null, sibling_index, sibling_count}` per sibling.
- `token {text, thinking, token_id, node_id, raw_index?, logprob?, perplexity?, top_alts?,
  measurements?}`: `measurements` is the token-scope envelope the tap built, forwarded whole
  (`ws_events.build_token_event`); the loom row keeps a projection of it
  (`readout/measurements.py::token_row_measurements`; full traces only under
  `persist_per_layer_scores`).
- `done {result: {text, tokens, finish_reason, applied_steering, usage, tok_per_sec, elapsed,
  measurements?, mean_logprob, mean_surprise}, node_id, sibling_index, sibling_count}` with the
  aggregate-scope envelope and the engine's own decode rate. A client `stop` or disconnect
  ends it `finish_reason: "cancelled"`; a stop racing the reply's end keeps its natural reason
  while `request_complete` still says `cancelled`. A commit-only `submit` sends one `started`
  and one `done` (`{role, text, node_id, ...}`) and no tokens.
- `tree_mutated {op, rev, added, removed, updated, active_node_id, cast}` forwards every
  `LoomMutated`, from any source. All frames share one ordered buffer, so a reply's
  `finalize_assistant` delta precedes its `done`.
- `resync {dropped}`: the client read slower than the connection produced and `dropped` token /
  `tree_mutated` frames were never sent. Later frames are live; the client rebuilds from `GET
  /tree`. It precedes any `done` it affects.
- `error {message, code, status, node_id, sibling_index}` on a connection that stays open
  (classified with `describe_error`; schema failures, bad JSON and unknown types carry code
  `ValidationError` and list every error).
- `request_complete {request_id, state, completed_siblings}` ends every correlated turn.

A `request_id` admits a turn into the same `OperationLedger` (kind `generate`) and rides every
frame of the turn. Resubmitting a finished id replays its recorded frames without generating;
a live id or different parameters gets `error` coded `REQUEST_RECONCILE_REQUIRED` /
`REQUEST_ID_CONFLICT` (frame has only `type`, `request_id`, `code`, `message`); a full ledger
`REQUEST_QUEUE_FULL` (503). `GET S/requests/{id}` reports `state: "unknown"` when absent.

Concurrency and limits:

- One reader task owns `receive_text()` (the `websockets` receive flag makes overlapping
  receives a `RuntimeError`) and feeds a bounded FIFO: 1 MiB per frame (close 1009; the CLI
  passes the same `ws_max_size`) and 16 pending frames / 4 MiB (close 1013); `stop` is a
  coalesced control outside that budget. Turns run one at a time per connection.
- One sender task writes the ordered outbound buffer (`_OutboundQueue`); producers never wait
  on it. `token` and `tree_mutated` frames bound at 256 queued (`MAX_WS_OUTBOUND_EVENTS`); past
  it they are dropped until the sender catches up, then one `resync` is sent. Control frames
  (`started`, `done`, `error`, `request_complete`) are never dropped. A slow reader never slows
  or cancels generation, and the session lock releases when the last sibling's worker ends. A
  send the peer accepts nothing of for 30 s closes 1013 (the one dead-peer rule), so `drowse
  serve` pings every 20 s with no pong timeout (`WS_PING_INTERVAL_SECONDS`,
  `WS_PING_TIMEOUT_SECONDS = None`): a slow reader answers a ping only after the backlog ahead
  of it, and uvicorn's default 20 s timeout closed such pages mid-reply (1011, recorded
  `cancelled`).
- A turn runs in a worker thread under `session.generation_state.cancellation_scope`. `n > 1`
  fans siblings serially under seeds from `loom.recipe.derive_seed_schedule`, holding the
  session lock for the fan; a stop cancels the current sibling and skips the rest, an error
  aborts the rest. Disconnect cancels the turn and joins only this connection's worker; an
  unexpected exception sends an error frame and closes 1011.

## Not in the server

A multi-session server (`{session_id}` is URL shape only), `/packs` routes or HTTP push (HF
upload is CLI `pack push`), per-family `/lens/*` or `/sae/*` groups, one-shot text-scoring
endpoints, a traits SSE stream or per-token aliases on `token` frames, tools / structured
output / embeddings on the compat protocols.

## Tests

CPU-only under `tests/server/`. `tests.support.server.mock_session(**overrides)` is an autospec
of `DrowseSession` for routes needing no behavior; otherwise `tiny_session` + `app_client`
(`tests/conftest.py`); `StubSession` and `script_generate` script WS generations.
