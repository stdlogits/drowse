// A loom token row, as the client reads it back.
//
// A generated row keeps what the thread's tint, the rack's trace and the tree filter read
// (``drowse.readout.measurements.token_row_measurements``): the flat ``scores`` and each geometry
// reading's ``fraction`` and ``coords``, at 6 significant digits, and no other channel. The rest of
// the token (a geometry reading's nearest nodes, assignment, residual, membership and depth; the
// lens and SAE readings, chips and features; the per-layer traces) replays while the model is
// loaded: the token view and the hover ask the family's token readout or the anatomy. An authored
// (prompt) row is the same projection: its node's ``prompt_capture`` anchors the replay, addressed
// by the row's place among the node's captured rows (``promptIndex``). A row of a call that set
// ``persist_per_layer_scores`` (which the dashboard does not) keeps every reading whole with its
// traces (``summary_measurements``). The reply's settled reading rides the node
// (``aggregate_measurements``), which the rack and the probe inspector draw after a reload.
import type {
  AnyReadingJSON,
  MeasurementsEnvelopeJSON,
  ProbeReadingJSON,
  RowProbeReadingJSON,
  TokenRowEnvelopeJSON,
  TokenScore,
} from "./types";

/** The keys a token row may carry (the loom reader refuses any other). */
export const TOKEN_ROW_KEYS: ReadonlySet<string> = new Set([
  "token_id", "text", "logprob", "sampler_entropy", "perplexity", "raw_index", "top_alts", "measurements",
]);

/** The envelope version a row's ``measurements`` carries (``readout/measurements.py::MEASUREMENTS_VERSION``). */
export const ROW_MEASUREMENTS_VERSION = 2;

/** A generated row's geometry reading carries exactly these (``ROW_GEOMETRY_KEYS``). */
const ROW_GEOMETRY_KEYS = ["fraction", "coords"] as const;
/** A reading kept whole carries the summary (``GEOMETRY_SUMMARY_KEYS``) and all of
 *  ``GEOMETRY_TRACE_KEYS``; a node's aggregate carries the summary alone. */
const GEOMETRY_SUMMARY_KEYS = [
  "fraction", "nearest", "coords", "residual", "assignment", "membership", "depth_com", "depth_spread",
] as const;
const GEOMETRY_TRACE_KEYS = ["fraction_per_layer", "coords_per_layer", "residual_per_layer"] as const;

const RETIRED = "saved by an earlier drowse (its probe readings are in a retired shape)";

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** Whether ``reading`` carries exactly ``keys``. */
const hasExactly = (reading: Record<string, unknown>, keys: readonly string[]): boolean =>
  Object.keys(reading).length === keys.length && keys.every((key) => key in reading);

/** The entries of a list field as the loom reader walks it (``x or ()``): a list's items, nothing
 *  for an absent or empty value, and one unreadable entry for anything else. */
const entriesOf = (value: unknown): unknown[] => {
  if (Array.isArray(value)) return value;
  if (value === undefined || value === null || value === false || value === 0 || value === "") return [];
  if (isObject(value) && Object.keys(value).length === 0) return [];
  return [null];
};

/** Why ``readout`` (a lens channel's) lacks a vocabulary id on an entry, or null. */
function lensReadoutProblem(readout: unknown): string | null {
  if (readout === undefined || readout === null) return null;
  if (!isObject(readout)) return "its lens readout is not an object";
  const entries = [...entriesOf(readout.aggregate)];
  for (const layer of entriesOf(readout.layers)) {
    entries.push(...(isObject(layer) ? entriesOf(layer.tokens) : [null]));
  }
  for (const entry of entries) {
    const id = isObject(entry) ? entry.id : undefined;
    if (typeof id !== "number" || !Number.isInteger(id)) return "a lens readout entry has no vocabulary id";
  }
  return null;
}

/** Why ``row`` is not a token row in the 6.0 shape (a fragment that reads after "token row: "), or
 *  null. It refuses exactly what the loom reader refuses (``loom/tree.py::_check_row_measurements``;
 *  both sides run ``tests/fixtures/token_rows.json``): an unknown row field, an envelope of another
 *  version, a geometry reading that is neither a generated row's (``fraction``, ``coords``) nor kept
 *  whole (the summary and all three traces), a generated reading beside a whole one or another
 *  family's channel, and a lens readout entry without its vocabulary id. A row saved before 6.0 kept
 *  a row-level ``probes`` / ``per_layer_scores`` copy, a layer-major ``per_layer_scores`` map in its
 *  envelope, the inspector's subspace coordinates on every reading, or a reading's summary without
 *  its traces; those read as retired. */
export function tokenRowProblem(row: Record<string, unknown>): string | null {
  if ("probes" in row || "per_layer_scores" in row) return RETIRED;
  for (const key of Object.keys(row)) {
    if (!TOKEN_ROW_KEYS.has(key)) return `unknown field ${key}`;
  }
  const envelope = row.measurements;
  if (envelope === undefined || envelope === null) return null;
  if (!isObject(envelope)) return "its measurements are not an object";
  if (envelope.version !== ROW_MEASUREMENTS_VERSION) {
    return `its measurements are version ${JSON.stringify(envelope.version ?? null)} (this build reads ${ROW_MEASUREMENTS_VERSION})`;
  }
  if ("per_layer_scores" in envelope) return RETIRED;
  const instruments = isObject(envelope.instruments) ? envelope.instruments : null;
  const geometry = instruments && isObject(instruments.geometry) ? instruments.geometry : null;
  const readings = geometry && isObject(geometry.readings) ? geometry.readings : {};
  const whole = [...GEOMETRY_SUMMARY_KEYS, ...GEOMETRY_TRACE_KEYS];
  let generated = false;
  let kept = false;
  for (const reading of Object.values(readings)) {
    if (!isObject(reading)) return "a probe reading is not an object";
    if ("subspace_coords_per_layer" in reading) return RETIRED;
    if (hasExactly(reading, ROW_GEOMETRY_KEYS)) generated = true;
    else if (hasExactly(reading, whole)) kept = true;
    else if (GEOMETRY_SUMMARY_KEYS.every((key) => key in reading)) return RETIRED;
    else return "a probe reading is neither a generated row's nor kept whole";
  }
  const otherChannel = instruments !== null && Object.keys(instruments).some((key) => key !== "geometry");
  if (generated && (kept || otherChannel)) return "it mixes a generated row's readings with readings kept whole";
  const lens = instruments && isObject(instruments.lens) ? instruments.lens : null;
  return lensReadoutProblem(lens ? lens.readout : undefined);
}

/** Why ``value`` is not a node's ``aggregate_measurements`` (the reply's aggregate envelope as the
 *  node keeps it: ``readout/measurements.py::aggregate_measurements``), or null. Null, or an
 *  ``aggregate`` envelope of this version without ``scores`` whose geometry readings carry exactly
 *  their summary; the loom reader refuses the same (``loom/tree.py::_check_aggregate_measurements``). */
export function aggregateMeasurementsProblem(value: unknown): string | null {
  if (value === null) return null;
  if (!isObject(value)) return "it is not an object";
  if (value.version !== ROW_MEASUREMENTS_VERSION) {
    return `it is version ${JSON.stringify(value.version ?? null)} (this build reads ${ROW_MEASUREMENTS_VERSION})`;
  }
  if (value.scope !== "aggregate" || "scores" in value) return "it is not an aggregate envelope without scores";
  if (!isObject(value.instruments)) return "it carries no instruments";
  const geometry = isObject(value.instruments.geometry) ? value.instruments.geometry : null;
  const readings = geometry && isObject(geometry.readings) ? geometry.readings : {};
  for (const reading of Object.values(readings)) {
    if (!isObject(reading) || !hasExactly(reading, GEOMETRY_SUMMARY_KEYS)) return "a probe reading is not its summary";
  }
  const lens = isObject(value.instruments.lens) ? value.instruments.lens : null;
  return lensReadoutProblem(lens ? lens.readout : undefined);
}

/** The keys of an authored node's ``prompt_capture`` (``loom/tree.py::PromptCapture``). */
const PROMPT_CAPTURE_KEYS = ["steering", "system_prompt", "thinking", "raw", "start", "thinking_start", "digest"] as const;

const isPosition = (value: unknown): boolean =>
  typeof value === "number" && Number.isInteger(value) && value >= 0;

/** The token-side fields of a loom node the capture rule reads. */
export interface CapturedNodeFields {
  recipe: unknown;
  tokens?: unknown;
  thinking_tokens?: unknown;
  prompt_capture?: unknown;
}

/** Why ``node``'s rows and ``prompt_capture`` disagree (a fragment that reads after "prompt
 *  capture: "), or null. The loom reader's rule (``loom/tree.py::PromptCapture.from_dict`` and
 *  ``_check_prompt_capture``; both sides run ``tests/fixtures/token_rows.json``'s ``authored``
 *  cases): a generated node has none; an authored node has rows exactly where its capture places a
 *  channel (an empty list is no rows), each with its ``token_id`` and no ``raw_index``, thinking
 *  before visible; the capture has exactly its seven fields, a SHA-256 hex ``digest`` and at
 *  least one start. Authored rows saved before the anchor (kept whole, with none) are refused. */
export function promptCaptureProblem(node: CapturedNodeFields): string | null {
  const capture = node.prompt_capture ?? null;
  if (capture !== null) {
    if (!isObject(capture)) return "it is not an object";
    if (!hasExactly(capture, PROMPT_CAPTURE_KEYS)) return "it does not carry exactly its fields";
    for (const key of ["steering", "system_prompt"] as const) {
      if (capture[key] !== null && typeof capture[key] !== "string") return `its ${key} is not a string`;
    }
    for (const key of ["thinking", "raw"] as const) {
      if (typeof capture[key] !== "boolean") return `its ${key} flag is not a boolean`;
    }
    for (const key of ["start", "thinking_start"] as const) {
      if (capture[key] !== null && !isPosition(capture[key])) return `its ${key} is not a position`;
    }
    if (capture.start === null && capture.thinking_start === null) return "it places no channel";
    if (typeof capture.digest !== "string" || !/^[0-9a-f]{64}$/.test(capture.digest)) return "its digest is not SHA-256 hex";
  }
  // Provenance is recipe presence: only a null recipe is authored (a node without the field is
  // refused by the loom reader for that).
  if (node.recipe !== null) return capture === null ? null : "a generated node has none";
  const placed = isObject(capture) ? capture : {};
  for (const [field, start] of [["tokens", placed.start], ["thinking_tokens", placed.thinking_start]] as const) {
    const rows = node[field];
    const present = Array.isArray(rows) && rows.length > 0;
    if (present !== (start !== undefined && start !== null)) {
      return `an authored node's ${field} are not where it places a channel (rows saved before their replay anchor are not read)`;
    }
    for (const row of present ? (rows as unknown[]) : []) {
      if (!isObject(row)) continue;
      if ("raw_index" in row) return "an authored row carries a raw index";
      if (typeof row.token_id !== "number" || !Number.isInteger(row.token_id)) return "an authored row has no token id";
    }
  }
  const visible = placed.start, thinking = placed.thinking_start;
  if (typeof visible === "number" && typeof thinking === "number"
    && thinking + (node.thinking_tokens as unknown[]).length > visible) {
    return "its thinking channel does not precede the visible one";
  }
  return null;
}

/** A captured authored row's replay address: its place among the node's captured rows, thinking
 *  rows first (``loom/ops.py::resolve_prompt_anchor``), or null for a generated or uncaptured node. */
export function authoredRowIndex(
  node: CapturedNodeFields & { thinking_tokens?: unknown[] | null },
  thinking: boolean,
  index: number,
): number | null {
  if (node.recipe !== null || node.prompt_capture === null || node.prompt_capture === undefined) return null;
  return thinking ? index : (node.thinking_tokens?.length ?? 0) + index;
}

/** The index a token replays at: a generated token's raw decode index, an authored token's place
 *  in its node's capture. Null when the token has neither (a live prompt token, an imported turn). */
export function replayIndex(token: Pick<TokenScore, "rawIndex" | "promptIndex"> | null | undefined): number | null {
  return token?.rawIndex ?? token?.promptIndex ?? null;
}

/** ``m`` when it is a whole envelope (a live token frame's, or a row kept whole), else null (a
 *  generated row's projection, which keeps only the scores and each geometry share and
 *  coordinates). A generated row carries no channel but geometry, and its readings only
 *  ``fraction`` and ``coords``. */
export function wholeEnvelope(m: TokenRowEnvelopeJSON | null | undefined): MeasurementsEnvelopeJSON | null {
  if (!m) return null;
  const instruments = m.instruments as Record<string, unknown>;
  if (Object.keys(instruments).some((key) => key !== "geometry")) return m as MeasurementsEnvelopeJSON;
  const readings = Object.values(m.instruments.geometry?.readings ?? {});
  return readings.some((reading) => "nearest" in reading) ? (m as MeasurementsEnvelopeJSON) : null;
}

/** Each geometry reading's share and coordinates, from either shape (what the tint and the trace read). */
export function rowGeometry(m: TokenRowEnvelopeJSON | null | undefined): Record<string, RowProbeReadingJSON> {
  return (m?.instruments.geometry?.readings ?? {}) as Record<string, RowProbeReadingJSON>;
}

/** ``readings`` (a steered replay of the token) with the values its generated row kept: each
 *  geometry reading's ``fraction`` and ``coords`` as captured, the rest (nearest, assignment,
 *  residual, depth, traces) as replayed. A replay's forward rounds a little differently from the
 *  decode's, so the headline numbers stay the ones the thread and the rack read. */
export function withRowValues(
  readings: Record<string, ProbeReadingJSON>,
  row: Record<string, RowProbeReadingJSON>,
): Record<string, ProbeReadingJSON> {
  const out: Record<string, ProbeReadingJSON> = {};
  for (const [name, reading] of Object.entries(readings)) {
    const kept = row[name];
    out[name] = kept && kept.coords.length === reading.coords.length
      ? { ...reading, fraction: kept.fraction, coords: kept.coords }
      : reading;
  }
  return out;
}

/** Whether every geometry reading in ``readings`` kept its per-layer traces (a live token frame,
 *  or a row kept whole). A node's aggregate summary has none: those layers replay. */
export function geometryTracesKept(readings: Record<string, ProbeReadingJSON> | null | undefined): boolean {
  return Object.values(readings ?? {}).every((reading) => reading.coords_per_layer !== undefined);
}

/** A reading's primary value at each layer: a scalar reading's ``per_layer``, a curved geometry
 *  probe's share (``share``), else a geometry probe's axis-0 coordinate. Null when the reading kept
 *  no trace of it (a saved geometry summary): the caller replays the token. */
export function readingLayers(reading: AnyReadingJSON | null | undefined, share = false): Record<string, number> | null {
  if (!reading) return null;
  const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
  const out: Record<string, number> = {};
  if ("value" in reading) {
    if (!reading.per_layer) return null;
    for (const [layer, v] of Object.entries(reading.per_layer)) if (finite(v)) out[layer] = v;
    return out;
  }
  if (share) {
    if (!reading.fraction_per_layer) return null;
    for (const [layer, v] of Object.entries(reading.fraction_per_layer)) if (finite(v)) out[layer] = v;
    return out;
  }
  if (!reading.coords_per_layer) return null;
  for (const [layer, c] of Object.entries(reading.coords_per_layer)) if (finite(c[0])) out[layer] = c[0];
  return out;
}
