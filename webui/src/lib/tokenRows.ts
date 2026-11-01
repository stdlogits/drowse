// A loom token row, as the client reads it back.
//
// A row keeps its measurement envelope's projection (``drowse.readout.measurements.
// token_row_measurements``): the flat ``scores``, each geometry reading's summary, each scalar
// reading whole and the lens aggregate chips, at 6 significant digits. The geometry readings' per-layer traces
// and the lens readout's layer rows ride the live token frame; a finished reply's per-layer views
// replay the token while the model is loaded. A row keeps them only when its call set
// ``persist_per_layer_scores``, which the dashboard does not.
import type { AnyReadingJSON, ProbeReadingJSON } from "./types";

/** The keys a token row may carry (the loom reader refuses any other). */
export const TOKEN_ROW_KEYS: ReadonlySet<string> = new Set([
  "token_id", "text", "logprob", "sampler_entropy", "perplexity", "raw_index", "top_alts", "measurements",
]);

/** The envelope version a row's ``measurements`` carries (``readout/measurements.py::MEASUREMENTS_VERSION``). */
export const ROW_MEASUREMENTS_VERSION = 2;

/** A row's geometry reading carries exactly these (``GEOMETRY_SUMMARY_KEYS``), plus all of
 *  ``GEOMETRY_TRACE_KEYS`` or none. */
const GEOMETRY_SUMMARY_KEYS = [
  "fraction", "nearest", "coords", "residual", "assignment", "membership", "depth_com", "depth_spread",
] as const;
const GEOMETRY_TRACE_KEYS = ["fraction_per_layer", "coords_per_layer", "residual_per_layer"] as const;

const RETIRED = "saved by an earlier drowse (its probe readings are in a retired shape)";

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** The entries of a list field as the loom reader walks it (``x or ()``): a list's items, nothing
 *  for an absent or empty value, and one unreadable entry for anything else. */
const entriesOf = (value: unknown): unknown[] => {
  if (Array.isArray(value)) return value;
  if (value === undefined || value === null || value === false || value === 0 || value === "") return [];
  if (isObject(value) && Object.keys(value).length === 0) return [];
  return [null];
};

/** Why ``row`` is not a token row in the 6.0 shape (a fragment that reads after "token row: "), or
 *  null. It refuses exactly what the loom reader refuses (``loom/tree.py::_check_row_measurements``;
 *  both sides run ``tests/fixtures/token_rows.json``): an unknown row field, an envelope of another
 *  version, a geometry reading whose keys are not the summary (plus all three traces or none), and a
 *  lens readout entry without its vocabulary id. A row saved before 6.0 kept a row-level ``probes`` /
 *  ``per_layer_scores`` copy, a layer-major ``per_layer_scores`` map in its envelope and the
 *  inspector's subspace coordinates on every reading; those read as retired. */
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
  for (const reading of Object.values(readings)) {
    if (!isObject(reading)) return "a probe reading is not an object";
    if ("subspace_coords_per_layer" in reading) return RETIRED;
    const traces = GEOMETRY_TRACE_KEYS.filter((key) => key in reading).length;
    if (traces !== 0 && traces !== GEOMETRY_TRACE_KEYS.length) return "a probe reading keeps part of its layer traces";
    const summary = GEOMETRY_SUMMARY_KEYS.every((key) => key in reading);
    if (!summary || Object.keys(reading).length !== GEOMETRY_SUMMARY_KEYS.length + traces) {
      return "a probe reading is not a row summary";
    }
  }
  const lens = instruments && isObject(instruments.lens) ? instruments.lens : null;
  const readout = lens ? lens.readout : undefined;
  if (readout !== undefined && readout !== null) {
    if (!isObject(readout)) return "its lens readout is not an object";
    const entries = [...entriesOf(readout.aggregate)];
    for (const layer of entriesOf(readout.layers)) {
      entries.push(...(isObject(layer) ? entriesOf(layer.tokens) : [null]));
    }
    for (const entry of entries) {
      const id = isObject(entry) ? entry.id : undefined;
      if (typeof id !== "number" || !Number.isInteger(id)) return "a lens readout entry has no vocabulary id";
    }
  }
  return null;
}

/** Whether every geometry reading in ``readings`` kept its per-layer traces (a live token frame,
 *  or a row whose call persisted them). A saved row's summary has none. */
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
