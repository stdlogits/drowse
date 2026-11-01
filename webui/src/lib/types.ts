// Shared types for the drowse webui.  Every panel/drawer/store imports
// from here so renames stay one-shot — this module is THE import surface,
// including for the generated half.
//
// Two halves:
//
//   * `./types.gen` — the native REST response shapes, GENERATED from the
//     FastAPI app's OpenAPI schema by `scripts/generate_webui_types.py`.
//     Re-exported below, so `import type { SessionInfo } from "./types"`
//     keeps resolving.  Never edit that file; edit the server's
//     `response_models.py` and regenerate.
//   * this file — the shapes OpenAPI cannot describe: the WebSocket frame
//     vocabulary, request bodies, and the dashboard's own client-local
//     state types.  It also composes a few unions over generated members
//     (`ProbeInfo`, `InstrumentLiveState`), which is why the generated
//     names are imported as well as re-exported.

export * from "./types.gen";

import type {
  CastMemberJSON,
  GeometryProbeInfo,
  GeometryLiveState,
  LensAggregateTokenJSON,
  LensLiveState,
  LensProbeInfo,
  LensReadoutLayerJSON,
  LoomNodeJSON,
  MeasurementsEnvelopeJSON,
  ProbeReadingJSON,
  SaeFeatureJSON,
  SaeLiveState,
  SaeProbeInfo,
  ScalarReadingJSON,
  UnitsProbeInfo,
} from "./types.gen";

/** Why a generation ended, the server's ``generation.results.FinishReason``.
 *  ``cancelled`` is a reply a Stop or a disconnect cut short; the loom node
 *  records it and the chat marks the reply. */
export type FinishReason = NonNullable<LoomNodeJSON["finish_reason"]>;
export const FINISH_REASONS: readonly FinishReason[] = ["stop", "length", "stop_sequence", "cancelled"];

// ---------------------------------------------------------- triggers --

/** Per-term trigger window from the steering-expression grammar.
 *
 * The five presets mirror the canonical names of ``_TRIGGER_CANONICAL`` in
 * steering/grammar.py (``both``/``before``/``after``/``thinking``/``response``);
 * the grammar's ``prompt``/``generated`` aliases are not UI values. The two
 * counted windows are ``@first:N`` (``FIRST_N``) and ``@after:N`` (``AFTER_N``),
 * their count on the term's ``tokens``. A term's probe gate (``&when:…``) is
 * its own field, ``gate``. */
export type Trigger =
  | "BOTH"
  | "BEFORE" // prompt prefill only
  | "AFTER" // reply after thinking
  | "THINKING"
  | "RESPONSE" // every generated token
  | "FIRST_N" // the first ``tokens`` generated tokens
  | "AFTER_N"; // every generated token after the first ``tokens``

/** A probe gate's comparison. */
export type GateOp = ">" | ">=" | "<" | "<=";

/** A term's probe gate (``@when:<probe><op><threshold>``, or ``@<phase>&when:…``): the term acts
 *  only on generated tokens where the probe's last reading passes. ``probe`` is the scalar key
 *  verbatim: an attached probe's name (its main reading) or one of its channels. A gate never
 *  reads the prompt (there is no reading yet), so a gated term never acts there. */
export interface SteerGate {
  probe: string;
  op: GateOp;
  threshold: number;
}

/** When a term acts: its window, the count a counted window takes, and its probe gate. Every
 *  rack entry carries it. */
export interface SteerTiming {
  trigger: Trigger;
  /** The count of a ``FIRST_N`` or ``AFTER_N`` window (a whole number); absent otherwise. */
  tokens?: number;
  /** Its probe gate; absent or ``null`` for none. */
  gate?: SteerGate | null;
}

/** Tensor variant suffix from the shared steering grammar. */
export type Variant =
  | "raw"
  | "sae"
  | `sae-${string}`
  | "role"
  | `role-${string}`
  | "from"
  | `from-${string}`;

// ----------------------------------------------------- session info --

// -------------------------------------------------- jacobian lens --

/** The J-lens readout at one decode step of a loom node (the forward that
 *  produced the clicked token) — the drilldown's render shape.  Built from
 *  the ``GET .../instruments/lens/token-readout`` measurements-replay
 *  envelope (``instruments.lens.readout`` + ``binding``). */
export interface LensTokenReadoutJSON {
  node_id: string;
  raw_index: number;
  /** The clicked token — for highlighting its appearances in the matrix. */
  token_id: number;
  token_text: string;
  /** The steering expression the replay ran under, or ``null`` for an
   *  unsteered read (no recipe steering, or ``steered=false``). */
  steering: string | null;
  /** Layer-aggregated view of the same logits across all requested layers,
   *  strength-descending.  Empty from a pre-aggregate server. */
  aggregate?: LensAggregateTokenJSON[];
  layers: LensReadoutLayerJSON[];
}

/** The instrument-preparation operation a POST launches / a GET/DELETE
 *  reports. */
export type PreparationOp = "fetch" | "fit" | "train";

/** The read families. ``units`` (MLP neurons and attention heads) has probes, a node replay and
 *  the catalog fetch, but no live switch of its own beside the probes'. */
export type InstrumentFamily = "geometry" | "lens" | "sae" | "units";

/** Live-readout state, discriminated by family: geometry is an
 *  all-or-nothing switch, the lens resolves a layer list, the SAE reports
 *  its resident layer + source.  The three members are generated; the union
 *  over them is composed here (OpenAPI inlines an `anyOf` at the property
 *  rather than naming it). */
export type InstrumentLiveState =
  | GeometryLiveState
  | LensLiveState
  | SaeLiveState;

// ------------------------------------------------ sparse autoencoder --

export interface SaeTokenReadoutJSON {
  node_id: string;
  raw_index: number;
  token_id: number;
  token_text: string;
  steering: string | null;
  layer: number;
  features: SaeFeatureJSON[];
}

// --------------------------------- 5.x measurement envelope (token wire) --
//
// The single JSON-safe read-side record on the WS ``token`` frame (whole),
// the loom token row (``measurements`` key: its ``token_row_measurements``
// projection, each reading's summary at 6 significant digits and no per-layer traces
// unless the call persisted them), and the token-readout replay endpoint
// (wrapped in ``{measurements}``); see drowse.readout.measurements.

/** Either family's reading shape, as it arrives inside the envelope. */
export type AnyReadingJSON = ProbeReadingJSON | ScalarReadingJSON;

/** True for the single-axis families' native reading. */
export function isScalarReading(
  reading: AnyReadingJSON,
): reading is ScalarReadingJSON {
  return "value" in reading;
}

// ----------------------------------------------------- manifolds --

/** One axis of a Box manifold domain.  ``periodic`` axes wrap (the
 *  authoring coordinate is taken mod ``period``); open axes clamp to
 *  ``[lo, hi]``. */
export interface AxisSpec {
  name: string;
  periodic: boolean;
  /** Period of a periodic axis — server canonicalizes; for an open
   *  axis this is typically ``hi - lo``. */
  period: number;
  lo: number;
  hi: number;
}

/** Geometry of a steering manifold's authoring domain.  ``box`` carries
 *  per-axis specs (1D/2D/3D in the webui builder); ``sphere`` is S^dim
 *  with a chordal metric; ``custom`` is the JSON-authored escape hatch
 *  the webui shows read-only.
 *
 *  Hand-written rather than generated: it is a ``type``-tagged union, and
 *  the server-side declaration is a plain open mapping (a TypedDict cannot
 *  express the tag).  The generated response types reference this one
 *  through the generator's property-override table. */
export type ManifoldDomain =
  | { type: "box"; axes: AxisSpec[] }
  | { type: "sphere"; dim: number }
  | { type: "klein" }
  | { type: "projective"; dim: 2 }
  | { type: "custom"; [key: string]: unknown };

/** One node of a manifold — a label, its authoring coordinates (one
 *  per intrinsic dimension), and the statement corpus pooled into its
 *  centroid. */
export interface ManifoldNodeSpec {
  label: string;
  coords: number[];
  statements: string[];
  /** Optional per-node assistant-role substitution.  When set, this
   *  node's centroid is pooled with the chat-template's assistant-role
   *  label replaced by this slug — the persona-manifold building block
   *  that lets one fitted manifold span multiple personas in
   *  role-baselined activation space.  Slug must match
   *  ``[a-z0-9._-]+``.  Omit / leave empty for the standard assistant
   *  baseline (the legacy default).  Family-unsupported (Mistral-3 /
   *  talkie) raises at fit time. */
  role?: string | null;
}

/** PCA discover-fit diagnostics block surfaced in the inspector.
 *  Wire-shape mirror of ``drowse.geometry.manifold.PcaDiagnostics``.  Tensor
 *  fields are flattened to plain number[]'s server-side; everything
 *  else is a primitive. */
export interface ManifoldPcaDiagnostics {
  per_component_variance: number[];
  cumulative_variance: number[];
  picked_k: number;
  threshold: number;
}

/** Spectral (Laplacian-eigenmaps) discover-fit diagnostics block.
 *  Wire-shape mirror of ``drowse.geometry.manifold.SpectralDiagnostics``. */
export interface ManifoldSpectralDiagnostics {
  eigenvalues: number[];
  picked_k: number;
  gap_index: number;
  gap_magnitude: number;
  bandwidth: number;
  k_nn: number;
  component_count: number;
}

/** Body for POST /drowse/v1/manifolds/install. */
export interface InstallManifoldRequest {
  /** HF coord (``owner/repo[@revision]``) or local folder path. */
  target: string;
  /** Override the install destination (``<ns>/<name>``).  Wire field is
   *  ``as_`` since ``as`` is a Python keyword — matches the route
   *  body model. */
  as_?: string;
  force?: boolean;
}

/** One source folder in a manifold merge — fully qualified ``ns/name``. */
export interface MergeManifoldSource {
  namespace: string;
  name: string;
}

/** Body for POST /drowse/v1/manifolds/merge.
 *
 *  Restricted to discover-mode (autofitted) sources by design — the
 *  server unions their node corpora into one heap and writes a fresh
 *  unfitted discover folder.  Run ``apiManifoldFitStream`` against the
 *  merged folder to derive coords from the combined heap.
 */
export interface MergeManifoldRequest {
  /** Destination namespace (defaults to ``"local"`` server-side). */
  namespace?: string;
  /** Destination manifold name. */
  name: string;
  description?: string;
  /** ≥ 2 discover-mode source folders. */
  sources: MergeManifoldSource[];
  /** Override the merged folder's fit_mode.  Required when sources
   *  disagree; defaults to the shared mode otherwise. */
  fit_mode?: "pca" | "spectral" | "auto";
  hyperparams?: Record<string, unknown>;
  force?: boolean;
}

/** Body for POST /drowse/v1/manifolds. */
export interface CreateManifoldRequest {
  namespace?: string;
  name: string;
  description: string;
  domain: ManifoldDomain;
  nodes: ManifoldNodeSpec[];
}

/** One node of a discover-mode manifold — label + statements only.
 *  Coords are derived per-model at fit time, so the authoring shape
 *  carries no ``coords`` field. */
export interface DiscoverManifoldNodeSpec {
  label: string;
  statements: string[];
  /** Optional per-node assistant-role substitution; see
   *  :class:`ManifoldNodeSpec` for semantics. */
  role?: string | null;
}

/** Body for POST /drowse/v1/manifolds/discover.
 *
 *  The user supplies labeled statement corpora; the matching ``fit``
 *  call derives node coordinates per-model via PCA, spectral embedding,
 *  or ``auto`` (let ``select_topology`` pick the geometry per-model). */
export interface CreateDiscoverManifoldRequest {
  namespace?: string;
  name: string;
  description?: string;
  fit_mode: "pca" | "spectral" | "auto";
  nodes: DiscoverManifoldNodeSpec[];
  hyperparams?: Record<string, number | string>;
}

/** Body for POST /drowse/v1/manifolds/from-template. */
export interface CreateManifoldFromTemplateRequest {
  namespace?: string;
  name: string;
  description?: string;
  fit_mode: "pca" | "spectral" | "auto";
  template_ref: string;
  hyperparams?: Record<string, number | string>;
  force?: boolean;
}

// ---- standalone templated-completion artifact (/drowse/v1/templates) ----

/** One turn in a template context's multi-turn history. */
export interface TemplateTurn {
  role: "system" | "user" | "assistant";
  content: string;
}

/** A multi-turn context: history turns + the slotted final assistant turn.
 *  The slot appears exactly once in ``assistant`` and never in a history turn. */
export interface TemplateContextSpec {
  turns: TemplateTurn[];
  assistant: string;
}

/** Body for POST /drowse/v1/templates — author a standalone template. */
export interface CreateTemplateRequest {
  namespace?: string;
  name: string;
  slot: string;
  values: string[];
  contexts: TemplateContextSpec[];
  description?: string;
  tags?: string[];
  force?: boolean;
}

/** Body for POST /drowse/v1/manifolds/generate.
 *
 *  LLM-author a discover-mode manifold from a flat concept list: the
 *  server plans the folder and runs ``DrowseSession.generate_manifold``
 *  (each concept answers the shared baseline prompts in character, one
 *  corpus per node), leaving a fresh discover folder ready for
 *  ``POST .../fit``. */
export interface GenerateManifoldRequest {
  namespace?: string;
  name: string;
  description?: string;
  concepts: string[];
  /** Per-concept system-prompt framing: ``abstract`` → "someone {c}",
   *  ``concrete`` → "{article} {c}", ``custom`` → ``custom_system``.
   *  Default abstract. */
  kind?: "abstract" | "concrete" | "custom";
  /** Required for ``kind: "custom"``; ``{c}`` is replaced by each concept. */
  custom_system?: string;
  /** In-character responses generated per shared baseline prompt. */
  samples_per_prompt?: number;
  /** Seeds the sampler: each node's corpus is reproducible from the seed
   *  and its node index.  Omitted or null: unseeded. */
  seed?: number | null;
  fit_mode?: "pca" | "spectral" | "auto";
  hyperparams?: Record<string, number | string>;
  force?: boolean;
  /** Persona-manifold opt-in: each ``concepts[i]`` slug doubles as
   *  that node's assistant-role substitution at fit time, producing a
   *  role-paired manifold.  Steering through it implies the nearest
   *  node's role at decode time (the manifold lives in
   *  role-baselined activation space). */
  role_per_node?: boolean;
}

/** Body for POST /drowse/v1/manifolds/{ns}/{name}/fit.
 *
 *  Authored folders consume ``sae`` plus layer/force controls; discover
 *  folders additionally accept ``fit_mode`` / ``hyperparams`` overrides
 *  that get persisted into the folder before the fit runs so the cache
 *  key reflects the actual inputs. */
export interface FitManifoldRequest {
  sae?: string | null;
  layers?: number[] | "workspace" | "all" | null;
  force?: boolean;
  fit_mode?: "pca" | "spectral" | "auto" | null;
  hyperparams?: Record<string, number | string> | null;
}

// ----------------------------------------------------- vectors --

export interface ExtractRequest {
  /** Concept represented by the positive node (or sole monopolar node). */
  concept: string;
  baseline?: string | null;
  /** Elicitation framing for both poles — the same knob
   *  ``POST /manifolds/generate`` carries. ``abstract`` -> "someone {c}",
   *  ``concrete`` -> "{art} {c}", ``custom`` -> ``custom_system`` (no role
   *  swap, works on every model family). */
  kind?: "abstract" | "concrete" | "custom";
  /** System template for ``kind: "custom"`` ({c} = the concept). Required
   *  when ``kind`` is ``custom``; rejected-with-400 otherwise. */
  custom_system?: string | null;
  sae?: string | null;
  /** Role-augmented extraction: replace the assistant-role label in
   * the chat template with this slug at extract time (e.g. "pirate").
   * The same substitution rides at steer time so the extract baseline
   * matches the steer baseline.  The canonical tensor records the uniform
   * role and is addressed via the matching ``:role-<slug>`` alias; the
   * reserved ``_role-*`` filename is not written. Slug must match
   * ``[a-z0-9._-]+``;
   * mutually exclusive with ``sae``. */
  role?: string | null;
  /** Destination namespace for the extracted 1/2-node manifold folder.
   *  ``null`` / unset lands under
   *  ``~/.drowse/manifolds/local/<canonical>/``; another value selects
   *  ``manifolds/<namespace>/<canonical>/``. */
  namespace?: string | null;
  /** Regenerate/re-author the manifold corpus and refit even when a valid
   *  fitted tensor exists. Default false keeps the exact cache hit. */
  force?: boolean;
}

// ----------------------------------------------------- probes --

/** One attached probe row, discriminated by ``family`` — the unified
 *  read-side collection under ``/probes`` (the pre-4.0 split of vector vs
 *  manifold probes collapsed onto one route). */
export type ProbeInfo = GeometryProbeInfo | LensProbeInfo | SaeProbeInfo | UnitsProbeInfo;

/** Body for ``POST /drowse/v1/sessions/{id}/probes`` — attach any probe
 *  shape by selector (the same ``[ns/]name[:variant]`` the ``%`` steering
 *  term consumes). */
export interface ProbeRequest {
  selector: string;
  name?: string;
  top_n?: number;
}

// ------------------------------------------------- probe readings --

// ----------------------------------------------------- probe geometry --

// ----------------------------------------------------- correlation --

// --------------------------------------------------- pairwise compare --

// ----------------------------------------------------- WS protocol --

export interface WSSampling {
  temperature?: number | null;
  top_p?: number | null;
  top_k?: number | null;
  max_tokens?: number | null;
  seed?: number | null;
  stop?: string[] | null;
  logit_bias?: Record<string, number> | null;
  presence_penalty?: number;
  frequency_penalty?: number;
  /** Logit-pass: opt in to top-K alternatives + chosen-token logprob on
   *  the WS ``token`` event.  Server-side clamped to ``[0, 256]``.  Zero
   *  (or absent) means logprob-only — chosen-token logprob still flows
   *  when any on_token consumer is live, just no top alternatives.
   *  Default 0 keeps the wire shape unchanged for opt-out users. */
  return_top_k?: number | null;
  /** Skip final aggregate probe scoring when only gate control is needed. */
  return_probe_readings?: boolean | null;
  /** Native dashboard requests this so streamed token rows can rehydrate
   *  the token-drilldown layer heatmap after a refresh. */
  persist_per_layer_scores?: boolean | null;
  /** Native dashboard requests per-layer whitened subspace coords on each
   *  token's probe reading (the probe-inspector live point + fading trail).
   *  Set true only while that inspector is open; forces per-token scoring. */
  persist_subspace_coords?: boolean | null;
  /** Per-message role-substitution labels (roleplay scaffold).  Ride each
   *  generate / commit like ``seed``; stamped onto the produced loom nodes
   *  and rendered per-turn.  null/empty = standard role label. */
  user_role?: string | null;
  assistant_role?: string | null;
}

/** One message of an explicit conversation replay on the ``generate``
 *  frame's ``input`` list. */
export interface WSInputMessage {
  role: ChatRole | "system";
  content: string;
  label?: string | null;
}

export interface WSGenerateRequest {
  type: "generate";
  request_id?: string;
  system_prompt?: string | null;
  /** False preserves the selected turn and creates a new child, even for n=1. */
  append_same_role?: boolean;
  /** ``null`` is a continue — no committed turn, the model speaks next
   *  from ``parent_node_id`` (or the active leaf). A string is the
   *  compatibility prompt form, and a messages list is the explicit-
   *  conversation replay the unsteered shadow uses. */
  input?: string | WSInputMessage[] | null;
  steering?: string | null;
  /** Browser runtime only: absolute input-token ranges [start, end).
   * Includes template/BOS tokens; mutually exclusive with position_indices.
   * Nonnegative integers, no overlap; adjacent ranges are merged. At most
   * 4096 ranges (not 4096 tokens). [] disables injection; null clears targeting.
   * Applied to input residuals: position p influences prediction p+1 and can
   * causally affect later tokens. Intersects the expression's phase/gates.
   * Saved recipes and token forks retain normalized spans. Python requests
   * and tree imports reject this browser-only extension. */
  position_spans?: [number, number][] | null;
  position_indices?: number[] | null;
  sampling?: WSSampling | null;
  thinking?: boolean | null;
  /** Mirrors the native generate frame: omitted means stateless. Stateful
   *  loom continuations and authored string prompts pass ``false``. */
  stateless?: boolean;
  raw?: boolean;
  /** Loom: attach result as a child of this node.  ``null``/absent =
   *  active node.  Lets phase-3 regen target a specific user-parent. */
  parent_node_id?: string | null;
  /** Loom: spawn ``n`` sibling assistant nodes (deterministic seed schedule
   *  per Decision 20).  Default 1 server-side. */
  n?: number;
  /** Loom: partial Recipe overlaid on the parent's — phase-5 fan-out /
   *  auto-regen.  Accepted as a mode string (``"unsteered"`` etc) or a
   *  partial-recipe expression string.  Engine resolves the overlay. */
  recipe_override?: string | Record<string, unknown> | null;
  /** Token fork: regenerate an existing generated node as a sibling from
   *  one exact raw-token boundary. The replacement is either a captured
   *  alternative id or locally tokenized authored text. */
  fork_node_id?: string | null;
  fork_raw_index?: number | null;
  fork_alt_token_id?: number | null;
  fork_replacement_text?: string | null;
  fork_seed?: number | null;
  /** Cast model: which seat the generated turn occupies.  ``"user"``
   *  renders the generation prompt as a user-seat header (labeled by
   *  ``sampling.user_role``) and lands the node with ``role="user"`` +
   *  a stamped recipe.  Absent/null = assistant.  Needs scene mode. */
  generate_seat?: "user" | "assistant" | null;
}

export type ChatRole = "user" | "assistant";

/** Native composer submission. The authored and generated structural roles
 * are independent; omit ``generated_role`` for an append-only action. */
export interface WSSubmitRequest {
  type: "submit";
  request_id?: string;
  system_prompt?: string | null;
  text?: string | null;
  authored_role?: ChatRole | null;
  generated_role?: ChatRole | null;
  steering?: string | null;
  /** Browser runtime only; same position contract as WSGenerateRequest. */
  position_spans?: [number, number][] | null;
  position_indices?: number[] | null;
  sampling?: WSSampling | null;
  thinking?: boolean | null;
  authored_thinking?: string | null;
  raw?: boolean;
  parent_node_id?: string | null;
  n?: number;
  recipe_override?: string | null;
}

export interface WSStopRequest {
  type: "stop";
  request_id?: string;
}

export type WSClientMessage = WSGenerateRequest | WSSubmitRequest | WSStopRequest;

export interface WSStartedEvent {
  type: "started";
  request_id?: string;
  generation_id: string;
  node_id: string | null;
  sibling_index: number;
  sibling_count: number;
}

export interface WSGenerationProgressEvent {
  type: "generation_progress";
  request_id?: string;
  node_id: string;
  completed: number;
  total: number;
}

/** Logit-pass (v2.3): one alternative the model considered at this
 *  position.  Wire-shape mirror of ``drowse.generation.results.TokenAlt``.
 *  ``logprob`` is the post-sampler natural-log probability under the
 *  post-temperature / post-top-p / post-top-k distribution sampling
 *  actually drew from. */
export interface TokenAltJSON {
  id: number;
  text: string;
  logprob: number;
}

export interface WSTokenEvent {
  type: "token";
  request_id?: string;
  text: string;
  thinking: boolean;
  token_id: number | null;
  /** Logit-pass: chosen-token logprob under the post-sampler distribution.
   *  Populated whenever the engine's log_softmax ran (any ``on_token``
   *  consumer or an explicit ``logprobs``/``return_top_k`` request).
   *  Absent on current uncaptured events. */
  logprob?: number | null;
  /** Shannon entropy in nats of the exact post-filter sampler distribution. */
  sampler_entropy?: number | null;
  /** Per-token perplexity under the sampled distribution.  The native WS
   *  explicitly opts into this channel so the workbench status and exported
   *  turn provenance are backed by the engine rather than reconstructed. */
  perplexity?: number | null;
  /** Logit-pass: top-K alternatives sorted by descending logprob.  Length
   *  matches ``SamplingConfig.return_top_k`` when populated, else absent.
   *  The chosen token may or may not appear in this list depending on
   *  K. */
  top_alts?: TokenAltJSON[] | null;
  /** Logit-pass: raw decode-step index — the join key a logit fork slices
   *  ``raw_token_ids`` on.  Rides the ``token`` event directly; absent on
   *  current uncaptured events. */
  raw_index?: number | null;
  /** Loom: node id this token belongs to.  Routes the token to the right
   * sibling render during n-way regen.  Optional. */
  node_id: string | null;
  /** The 5.x measurement envelope — the single read-side record carrying the
   *  per-family ``instruments`` (``readings`` + ``readout``) plus the flat
   *  ``scores`` view, whole: every per-layer trace.  The loom token row keeps
   *  its projection.  Omitted when nothing was measured, so clients read it
   *  defensively. */
  measurements?: MeasurementsEnvelopeJSON;
}

export interface WSDoneResult {
  text: string;
  tokens: number;
  finish_reason: FinishReason;
  applied_steering?: string | null;
  terminal_reason?: "eos" | "stop_sequence" | "external_stop" | "length";
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  /** Logit-pass: per-turn mean chosen-token logprob over the assistant
   *  response span (thinking tokens excluded by construction).  Null when
   *  logprob capture wasn't live (replay / no on_token consumer). */
  mean_logprob?: number | null;
  mean_surprise?: number | null;
  /** The runtime's own decode rate and elapsed seconds for this reply. A page that read the
   *  stream slower than it was written (a resync) shows these, not its frame arrival rate.
   *  Optional: a runtime that omits them leaves the page's own measure. */
  tok_per_sec?: number;
  elapsed?: number;
  /** End-of-generation measurement envelope, ``scope: "aggregate"`` — the
   *  same shape the ``token`` frame carries, pooled at the last-content
   *  token.  The per-attached-probe readings live under
   *  ``instruments.{geometry,lens,sae}.readings`` and merge by name exactly
   *  as the token path merges them; there is no flat ``probe_readings``
   *  alias on this frame.  Omitted when no probe is attached — read
   *  defensively. */
  measurements?: MeasurementsEnvelopeJSON;
}

export interface WSDoneEvent {
  type: "done";
  request_id?: string;
  result: WSDoneResult;
  /** Loom: node id this gen finalised. */
  node_id: string | null;
  sibling_index: number;
  sibling_count: number;
}

export interface WSErrorEvent {
  type: "error";
  request_id?: string;
  message: string;
  code?: string;
  node_id?: string;
}

// ----------------------------------------------------- loom (v2.3) --

/** Wire-shape mirror of drowse.loom.tree.LoomNode.  Optional fields are
 * absent on the wire when null/empty server-side to keep payloads slim. */
/** One token-row inside a node's ``tokens`` / ``thinking_tokens`` array:
 *  exactly these keys (the loom reader refuses any other, and the earlier
 *  row-level ``probes`` / ``per_layer_scores`` copies; ``lib/tokenRows.ts``
 *  is the client's check).  All optional because the engine legitimately
 *  omits some (``top_alts`` only when ``return_top_k > 0``; ``raw_index`` is
 *  absent for transcript-imported nodes). */
export interface LoomTokenRowJSON {
  token_id?: number | null;
  text: string;
  logprob: number | null;
  sampler_entropy?: number | null;
  perplexity: number | null;
  top_alts?: { id: number; text: string; logprob: number }[];
  raw_index?: number | null;
  /** A generated row keeps what the thread's tint, the rack's trace and the tree filter read
   *  (``token_row_measurements``: the flat ``scores`` and each geometry reading's ``fraction``
   *  and ``coords``, at 6 significant digits); the token view and the hover replay the rest. A
   *  row kept whole (authored capture, or a call that set ``persist_per_layer_scores``) carries
   *  every reading with its traces (``summary_measurements``). */
  measurements?: TokenRowEnvelopeJSON;
}

/** A generated row's geometry reading (``readout/measurements.py::RowProbeReadingDict``): the
 *  share on the probe's subspace and its domain coordinates. */
export interface RowProbeReadingJSON {
  fraction: number;
  coords: number[];
}

/** A generated loom token row's ``measurements`` (``TokenRowMeasurements``): the envelope
 *  header, the flat ``scores`` and each geometry probe's share and coordinates; no other
 *  channel. */
export interface TokenRowMeasurementsJSON {
  version: number;
  scope: MeasurementsEnvelopeJSON["scope"];
  provenance: MeasurementsEnvelopeJSON["provenance"];
  instruments: { geometry?: { readings: Record<string, RowProbeReadingJSON> } };
  scores?: Record<string, number>;
}

/** What a token's ``measurements`` holds: a live frame's whole envelope or a row kept whole,
 *  or a generated row's projection. ``wholeEnvelope`` tells them apart. */
export type TokenRowEnvelopeJSON = MeasurementsEnvelopeJSON | TokenRowMeasurementsJSON;

/** Per-op delta sent on every tree mutation.  Clients apply in-place
 * keyed by ``rev`` continuity; full re-fetch on gap.
 *
 * Note: phase-2 server sends ``updated`` as full LoomNodeJSON entries
 * (the plan's "partial fields" shape simplifies to "send the node again"
 * because LoomMutated doesn't track which fields changed).  Clients merge
 * by replacing the node entry wholesale. */
export interface WSTreeMutatedEvent {
  request_id?: string;
  type: "tree_mutated";
  op:
    | "edit"
    | "branch"
    | "navigate"
    | "delete"
    | "star"
    | "note"
    | "reset"
    | "regenerate"
    | "begin_assistant"
    | "add_user"
    | "finalize_assistant"
    | "capture_authored"
    | "restore"
    | "cast"
    | string;
  added?: LoomNodeJSON[];
  removed?: string[];
  updated?: LoomNodeJSON[];
  active_node_id?: string | null;
  rev: number;
  /** Full effective roster inlined on every mutation so observed role labels
   * and configured cast members reconcile without a refetch. */
  cast?: Record<string, CastMemberJSON>;
}

/** Fired at the start of each branch in an n-way generate so the client
 * can allocate render slots before token events arrive. */
export type WSServerMessage =
  | WSStartedEvent
  | WSGenerationProgressEvent
  | WSTokenEvent
  | WSDoneEvent
  | WSErrorEvent
  | WSRequestCompleteEvent
  | WSTreeMutatedEvent
  | WSResyncEvent;

/** The client read slower than the runtime produced, and ``dropped`` token and
 *  ``tree_mutated`` frames were never sent (``ws_stream._OutboundQueue``).
 *  Generation went on into the tree; frames after this one are live.  The
 *  client rebuilds what it missed from the authoritative tree: the streaming
 *  node's rows past its last ``raw_index``, and any tree delta.  It always
 *  precedes a ``done`` whose reply it affects. */
export interface WSResyncEvent {
  type: "resync";
  dropped: number;
}

export interface WSRequestCompleteEvent {
  type: "request_complete";
  request_id: string;
  state: "completed" | "cancelled" | "failed";
  completed_siblings: number;
}

// ----------------------------------------------------- chat / UI --

/** Per-token score row for chat highlighting — the text + whichever
 * probe scores are known for the token, filled at render time. */
export interface TokenScore {
  text: string;
  thinking: boolean;
  /** Whichever score we know for the currently-selected highlight probe.
   * Filled at render time, not persisted. */
  score?: number;
  /** Full per-probe scores once available. */
  probes?: Record<string, number>;
  /** Full per-axis domain-frame coordinates per probe, captured live from the
   *  ``probe_readings`` wire channel.  Backs per-PC token highlighting (the
   *  ``personas[3]`` axis targets) — axis 0 already lives in ``probes``, so
   *  this is populated only for multi-axis (rank-R) probes.  It survives
   *  in-session navigation by reference in ``tokenScoreCache`` but is absent
   *  after a transcript / localStorage reload. */
  coordsByProbe?: Record<string, number[]>;
  /** Token-id from the WS event when available — useful for debugging. */
  tokenId?: number | null;
  /** Logit-pass: chosen-token post-sampler logprob. Absent on imported
   *  turns and when no consumer requested log-softmax capture. Drives the inline ``surprise`` highlight
   *  mode and the token drilldown's logits tab. */
  logprob?: number | null;
  /** Shannon entropy in nats of the post-filter sampling distribution. */
  samplerEntropy?: number | null;
  /** Per-token perplexity emitted by the runtime. */
  perplexity?: number | null;
  /** Logit-pass: top-K alternatives captured at this position (descending
   *  by logprob).  Absent when ``return_top_k == 0`` or replayed. */
  topAlts?: TokenAltJSON[] | null;
  /** Raw decode-step index of this token in the backing node's
   *  ``raw_token_ids`` — the join key a logit fork slices on.  Absent on
   *  transcript-imported nodes (engine pre-dates raw-id capture),
   *  in which case the token can't be forked. */
  rawIndex?: number | null;
  /** The measurements from the original decode step: the live frame's whole envelope while the
   *  reply is in this page, else its loom row's (a generated row keeps its scores and each
   *  geometry share and coordinates; ``wholeEnvelope`` says which). */
  measurements?: TokenRowEnvelopeJSON;
}

export interface ChatTurn {
  role: "user" | "assistant" | "system";
  text: string;
  /** Per-turn role-substitution label (roleplay scaffold) carried from the
   *  backing loom node — drives the bubble heading.  null/undefined =
   *  standard role label. */
  roleLabel?: string | null;
  /** Loom node backing this turn, when the server tree is active. */
  nodeId?: string | null;
  /** Whether a model run authored this turn. Used for generation artifacts
   *  and analysis only; never to name, style, or gate rerolling the role. */
  generated?: boolean;
  /** True iff any thinking content was emitted. */
  thinking?: boolean;
  /** Visible response tokens with score data. */
  tokens?: TokenScore[];
  /** Thinking-only tokens with score data (rendered inside the
   * <Collapsible> equivalent). */
  thinkingTokens?: TokenScore[];
  /** A/B-mode pair: the unsteered same-seat shadow turn. */
  abPair?: ChatTurn;
  /** Steering expression applied: the server's ``format_expr`` text. Only the server parses it (``steering/validate`` answers its parts); the dashboard has no grammar of its own. */
  appliedSteering?: string | null;
  /** Aggregate probe readings for the turn (mean per probe). */
  aggregateReadings?: Record<string, number>;
  /** Why the reply ended, set at done and from the loom node. */
  finishReason?: FinishReason;
  tokensSoFar?: number;
  perplexity?: number;
  /** Logit-pass: per-turn mean chosen-token logprob (response span only,
   *  thinking excluded).  Populated from the WS ``done`` event; absent for
   *  current uncaptured turns. */
  meanLogprob?: number | null;
}

// ----------------------------------------------------- steer rack --
//
// One unified steering term, addressed as a position on a fitted geometry —
// a steering vector is the K=2 flat case of a manifold, so there is no longer
// a separate "vector" shape.  ``mode`` is the geometry family, the
// discriminator the card branches on and the serializer reads:
//
//   ``subspace`` — a flat affine fit (a 2-node bipolar axis through the
//                  rank-8 ``personas`` fan).  Every subspace term shares one
//                  rack-level ``subspaceAlong`` master (the merged affine
//                  subspace has a single slide), so the card carries NO
//                  per-card along knob — only a position (snap-to-node /
//                  XYPad).  Serializes ``<subspaceAlong> name[:variant]%pos``.
//   ``manifold`` — a curved fit (e.g. ``emotions``).  Each curved term is its own
//                  injection, so it keeps a per-card ``along`` + ``onto``.
//                  Serializes ``<along[,onto]> name[:variant]%pos``.
//
// ``mode`` is set at add time (``RackDrawer`` picks the adder off the
// catalog's ``fit_mode``: pca/baked → subspace, spectral/authored →
// manifold). Rank-one subspaces expose push/ablate directly; higher-rank
// projection remains an advanced expression operation. ``:variant`` survives
// — it rides the atom (``name:sae%pos``).

/** Subspace (flat affine) steering term — a position on a flat fit.  The
 *  magnitude is the rack-level ``subspaceAlong`` master (shared across every
 *  subspace term — the merged affine subspace slides once), so this entry
 *  carries no per-card coefficient; relative weight between subspace terms is
 *  expressed by how far each position sits from neutral. */
export interface SubspaceSteerEntry extends SteerTiming {
  mode: "subspace";
  /** Mean-ablate a rank-one concept instead of steering toward a position. */
  ablate: boolean;
  /** Authoring coordinates, one per intrinsic dimension.  Rank-1 (a 2-node
   *  concept) is a single signed coord on the bipolar axis. */
  coords: number[];
  /** Node-label form (``name%label``); ``null`` = free coords (drag).  A
   *  fresh 2-node concept defaults to its positive pole's label. */
  label: string | null;
  /** Tensor variant — rides the atom (``name:sae%pos``).  Not authorable via
   *  the card today (kept for round-trip of pasted/legacy expressions). */
  variant: Variant;
  trigger: Trigger;
  /** The term's layer clause (absent = every fitted layer). */
  layers?: LayerScope;
  /** When false, the term is excluded from serialization (visual but
   * not active). */
  enabled: boolean;
}

/** Manifold (curved) steering term — a placement on a curved fit with its own
 *  per-card ``along`` + ``onto`` (each curved term is its own injection). */
export interface ManifoldSteerEntry extends SteerTiming {
  mode: "manifold";
  /** ``along`` blend fraction in [0, 1] — how far to slide the in-subspace
   *  foot toward the position.  Serializes as the first value of the ``%``
   *  coefficient slot. */
  blend: number;
  /** ``onto`` collapse fraction in [0, 1] — pulls the off-surface in-subspace
   *  residual onto the surface.  ``0`` = off.  Serializes as the second value
   *  of the coefficient slot (``along,onto``) only when > 0. */
  onto: number;
  /** Authoring coordinates, one per intrinsic dimension. */
  coords: number[];
  /** Node-label form (``name%label``); ``null`` = free coords (drag). */
  label: string | null;
  /** Tensor variant — rides the atom (``name:sae%pos``). */
  variant: Variant;
  trigger: Trigger;
  /** The term's layer clause (absent = every fitted layer). */
  layers?: LayerScope;
  enabled: boolean;
}

/** J-lens token steering term — pushes along the lens direction
 *  ``W_U[v] @ J_l`` over all fitted layers (``α jlens/<word>``). The rack
 *  key is the full ``jlens/<word>`` atom.  Per-chip ``alpha`` (not the
 *  shared ``subspaceAlong`` master): lens atoms run hotter than concept
 *  vectors — α≈0.3 is the coherent sweet spot, α≥0.5 over-steers into
 *  repetition — so each token needs its own dial. */
export interface JLensSteerEntry extends SteerTiming {
  mode: "jlens";
  /** Mean-ablate the token direction instead of pushing along it. */
  ablate: boolean;
  /** Push coefficient (the plain-atom α slot). */
  alpha: number;
  trigger: Trigger;
  /** The term's layer clause (absent = every fitted layer). */
  layers?: LayerScope;
  enabled: boolean;
}

/** S-lens token steering term (``α slens/<word>``): pushes along the S lens's direction
 *  for the token, toward having read it (``jlens/`` pushes toward saying it), over the S lens's
 *  fitted layers.  The same shape and dial as a ``jlens/`` term; the rack key is the full atom. */
export interface SLensSteerEntry extends SteerTiming {
  mode: "slens";
  /** Mean-ablate the token direction instead of pushing along it. */
  ablate: boolean;
  alpha: number;
  trigger: Trigger;
  /** The term's layer clause (absent = every fitted layer). */
  layers?: LayerScope;
  enabled: boolean;
}

/** Resident SAE decoder-row steering term (``α sae/<id>``). */
export interface SaeSteerEntry extends SteerTiming {
  mode: "sae";
  /** Mean-ablate the decoder-row direction instead of pushing along it. */
  ablate: boolean;
  alpha: number;
  trigger: Trigger;
  /** The term's layer clause (absent = every fitted layer). */
  layers?: LayerScope;
  enabled: boolean;
}

/** One MLP neuron or attention head (``neuron/L31n4410``, ``head/L28h12``: the rack key).  A
 *  unit carries its own layer, so it takes no layer clause.  A neuron pushes along its write (the
 *  sign of ``alpha`` picks the pole) or is silenced; a head has no single write direction and is
 *  only ever silenced (``ablate`` is always true).  Silencing replaces the unit's activation by
 *  its neutral mean, blended by ``alpha`` (1 = fully). */
export interface UnitSteerEntry extends SteerTiming {
  mode: "unit";
  ablate: boolean;
  /** Signed push strength (a neuron's − pole below zero); an ablation's blend, never negative
   *  (1 is exact mean-ablation; lib/unitTerms.ts holds the rule). */
  alpha: number;
  trigger: Trigger;
  enabled: boolean;
}

/** A projection (``α base~onto``, ``α base|onto``): a flat concept pushed by only the part of it
 *  that lies along a second concept (``~``), or by all of it but that part (``|``). The rack key is
 *  the selector ``base~onto``; its own signed ``alpha``, since a projection is a plain push and not
 *  a position on the shared flat slide. */
export interface ProjectionSteerEntry extends SteerTiming {
  mode: "projection";
  /** The concept pushed (a flat concept's name, ``ns/name``). */
  base: string;
  /** ``~`` keeps the part along ``onto``; ``|`` removes it. */
  operator: "~" | "|";
  /** The second concept. */
  onto: string;
  /** Signed push strength (below zero, toward the base's second pole). */
  alpha: number;
  /** The term's layer clause (absent = every fitted layer). */
  layers?: LayerScope;
  enabled: boolean;
}

/** The single-direction *atom* families — a lens token direction (toward
 *  saying it, ``jlens/``, or toward having read it, ``slens/``) and an SAE
 *  decoder row.  All rack as ``α <prefix><id>`` with one coefficient and
 *  no geometry, so they share a mutator set; only the key prefix differs.  A unit (``UnitSteerEntry``) is its own mode: it
 *  carries its layer and follows lib/unitTerms.ts. */
export type AtomMode = "jlens" | "slens" | "sae";

export type AtomSteerEntry = JLensSteerEntry | SLensSteerEntry | SaeSteerEntry;

/** A racked steering term — subspace (flat), manifold (curved), or a
 *  single-direction atom. */
export type SteerEntry =
  | SubspaceSteerEntry
  | ManifoldSteerEntry
  | ProjectionSteerEntry
  | JLensSteerEntry
  | SLensSteerEntry
  | SaeSteerEntry
  | UnitSteerEntry;

/** A term's ``{…}`` layer clause: ``null`` (or absent) acts at every layer the term is fitted at;
 *  ``"workspace"`` at the workspace band; a sorted index list at exactly those layers, and the
 *  empty list at none (the term is left out of the expression until it has a layer again). */
export type LayerScope = null | "workspace" | number[];

// ----------------------------------------------------- probe rack --

export type ProbeSortMode = "name" | "value" | "change";

export interface ProbeRackEntry {
  /** Exact attach request that produced this registered row.  ``ProbeInfo``
   * is display metadata and may canonicalize a geometry selector, so it
   * cannot by itself preserve an alias or tensor variant across save/load. */
  request: ProbeRequest;
  /** Server-side row — metadata, domain, node layout, and the ``is_affine``
   *  flat-vs-curved flag that selects the subspace vs manifold card. */
  info: ProbeInfo;
  /** Last N values of the primary scalar for the sparkline — ring-buffer-ish,
   *  capped client-side.  Primary scalar is the signed axis-0 ``coords[0]``
   *  for a subspace (flat) probe, the ``fraction`` for a manifold (curved). */
  sparkline: number[];
  current: number;
  previous: number;
  /** Most recent token's per-layer readings for *this* probe.  Layer-key
   * strings keep the wire shape; the card sorts numerically.  For a subspace
   * probe this is axis-0 ``coords_per_layer``; for a manifold, ``fraction_per_layer``. */
  perLayer: Record<string, number>;
  /** Latest full per-token reading (coords / fraction / nearest / residual +
   *  per-layer traces).  Null until the first ``token`` event lands. */
  reading: AnyReadingJSON | null;
  /** End-of-gen aggregate the ``done`` event lands — the settled reading.
   *  Null between gens; set on ``done``, cleared on the next ``started``. */
  aggregate: AnyReadingJSON | null;
  /** Scalar aggregate restored from the selected saved Loom node.  The tree
   * keeps this portable summary but not the full per-layer reading; cards use
   * it instead of presenting a false zero after reload/navigation. */
  savedAggregate: number | null;
  savedCoordinates?: number[];
  savedFraction?: number | null;
  /** The selected saved node's reading of this probe on its last measured row (the token its
   *  aggregate reads), as the row keeps it: a geometry summary without per-layer traces, a scalar
   *  reading whole. The rack, the Lens panel and the inspector read it after a reload where the
   *  live ``aggregate`` is gone, so they show what the live view showed (but the trail). */
  savedReading?: AnyReadingJSON | null;
  /** Most-recent per-token nearest list (ascending distance).  Drives the
   *  inline nearest readout + mini-map hover; empty until the first token. */
  nearest: [string, number][];
  /** Inferred per-token coord trajectory for 2-D box mini-map rendering —
   *  each token's ``nearest[0]`` looked up in ``info.node_coords``.  Empty
   *  for non-2-D / sphere / custom probes and unfitted-discover (no coords). */
  trajectory: number[][];
  /** Ring buffer (last ~64 tokens) of per-layer whitened subspace coords for
   *  the probe-inspector geometry plot's live point + fading trail.  Each entry
   *  is one token's ``subspace_coords_per_layer`` (layer-key -> (R,) coords), so
   *  the inspector can reproject for any scrubbed layer at render time.  Only
   *  populated while the inspector is open (the ``persist_subspace_coords``
   *  generate flag); cleared on each generation ``started``. */
  subspaceTrail: SubspaceTrailSample[];
}

/** One token's per-layer whitened subspace coords for the inspector trail. */
export interface SubspaceTrailSample {
  perLayer: Record<string, number[]>;
}

// ----------------------------------------------------- gen status --

export interface PerplexityAccumulator {
  /** Sum of ln(ppl) across scored steps — geometric mean assembled
   * lazily via ``geometricMeanPpl``. */
  logSum: number;
  count: number;
  mean: number | null;
}

export interface GenStatus {
  active: boolean;
  replay?: { completed: number; total: number } | null;
  tokensSoFar: number;
  maxTokens: number;
  /** Wall-clock start (``performance.now()`` ms). */
  startedAt: number | null;
  /** Completion (``performance.now()`` ms), retained so the final elapsed time stays frozen: the
   *  runtime's ``elapsed`` after ``startedAt`` when the done frame carries it, else wall clock. */
  finishedAt: number | null;
  tokPerSec: number;
  ppl: PerplexityAccumulator;
  finishReason: FinishReason | null;
}

// ----------------------------------------------------- pending actions --

/** Actions queued during in-flight generation.
 *
 * The queue drains one item per WS ``done`` event in arrival order —
 * each ``apply`` either kicks off another gen (``awaitsGen=true``,
 * the next drain waits for that gen's own ``done``) or completes
 * instantly (``awaitsGen=false``, the next drain fires immediately).
 *
 * ``text`` is the user-facing string for the chat-side pending
 * bubble and the ↑-pull-and-edit re-issue path; ``rebuild`` is a
 * factory the input recall path calls to re-encode a pulled-and-
 * edited item with the same kind/role/target unchanged.  Both are
 * ``null`` for non-editable items (instant mutations like
 * ``clearChat`` / ``regen``) — those render as ghosted action chips
 * and can't be pulled, only cancelled with the ``×``.
 */
export interface PendingAction {
  id: string;
  label: string;
  text: string | null;
  apply: () => void | Promise<void>;
  awaitsGen: boolean;
  rebuild: ((newText: string) => PendingAction) | null;
  createdAt: number;
  /** Coalesce tag for fold-into-tail batching.  When a fresh action
   *  carries the same ``coalesceKey`` as the *current queue tail*, its
   *  ``apply`` is chained onto that tail item instead of appending a
   *  new slot — so a slider drag (dozens of intermediate steering
   *  values) collapses to a single queued bubble carrying the net
   *  effect.  Only set on instant rack/steering mutations; ``undefined``
   *  for sends, commits, and one-shot mutations, which never coalesce. */
  coalesceKey?: string;
  onCancel?: () => void;
}

// ----------------------------------------------------- drawers --

export type DrawerName =
  /** Shared rack browser, subspace (flat) family — every flat affine
   *  fit (``fit_mode`` pca / baked): 2-node concept axes plus higher-rank
   *  flats like ``personas``.  White ``--accent``.  Split Fitted /
   *  Unfitted, per-row steer / probe / re-fit / delete, with a
   *  "+ build manifold" launcher (flat authoring folds into the manifold
   *  builder's pca path).  ``RackDrawer`` with ``family: "subspace"``.
   *  Opened from both rack "+ add" buttons. */
  | "subspace"
  /** Shared rack browser, manifold (curved) family — curved fits only
   *  (``fit_mode`` spectral / authored), e.g. ``emotions``, in the depth
   *  tint.  Same layout as the subspace half, with a
   *  "+ build manifold" launcher.  ``RackDrawer`` with
   *  ``family: "manifold"``. */
  | "manifolds"
  /** Manifold authoring form — domain step + node editor.  Reached
   *  from the "+ build manifold" button inside ``manifolds``. */
  | "manifold_builder"
  /** Discover-mode node-union merge.  Unions the node corpora of two or
   *  more discover-mode manifolds into a fresh discover folder; restricted
   *  to discover sources by design. Reached from the command palette. */
  | "manifold_merge"
  /** Local manifold catalog plus HF search/install for
   *  ``drowse-manifold``-tagged repositories. */
  | "manifold_pack"
  | "save_conversation"
  | "download_chat"
  | "load_conversation"
  | "compare"
  | "system_prompt"
  | "token_drilldown"
  | "correlation"
  /** Per-probe inspector — subsumes the layer-norms view for probes and
   *  adds a rank-aware whitened geometry plot (line / 2D scatter / 3D PCA
   *  scatter) with a layer scrubber and a fading live trajectory trail.
   *  Opened from a probe card's ⓘ button.  ``params: { name }``. */
  | "probe_inspector"
  | "advanced_sampling"
  | "health"
  | "session_admin"
  | "local_runtime"
  | "help"
  | "feedback"
  /** Cross-branch diff drawer — phase 5.  ``params`` carries the
   * selected node ids (1 user node → compare its children, 2+
   * assistant nodes → compare those). */
  | "node_compare"
  /** Transcript export/import drawer — phase 5. */
  | "transcript"
  /** Templated-completion lab — author standalone templates (slot + values
   *  + multi-turn contexts) and score the restricted-choice value
   *  distribution (steering-aware before/after). Reached from the workspace
   *  rail's "manifolds → templates…" entry. */
  | "template_lab"
  /** Cast manager (phase 3) — the tree's roster of named labels with
   *  standing steering recipes.  Reached from the composer cast row's
   *  "cast…" launcher.  A steering surface, not a chat feature. */
  | "cast";

export interface DrawerState {
  open: DrawerName | null;
  /** Per-drawer params — typed loosely because each drawer owns its own
   * shape (e.g. token drilldown carries the click-target token row). */
  params: unknown;
}
