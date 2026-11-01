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

const GEOMETRY_TRACE_KEYS = ["fraction_per_layer", "coords_per_layer", "residual_per_layer"] as const;

const RETIRED = "saved by an earlier drowse (its probe readings are in a retired shape)";

/** Why ``row`` is not a token row in the 6.0 shape (a fragment that reads after "token row: "), or
 *  null. A row saved before 6.0 kept a row-level ``probes`` / ``per_layer_scores`` copy, a
 *  layer-major ``per_layer_scores`` map in its envelope and the inspector's subspace coordinates on
 *  every reading; those are refused, as the loom reader refuses them, rather than read half. */
export function tokenRowProblem(row: Record<string, unknown>): string | null {
  if ("probes" in row || "per_layer_scores" in row) return RETIRED;
  for (const key of Object.keys(row)) {
    if (!TOKEN_ROW_KEYS.has(key)) return `unknown field ${key}`;
  }
  const envelope = row.measurements;
  if (envelope === undefined || envelope === null) return null;
  if (typeof envelope !== "object" || Array.isArray(envelope)) return "its measurements are not an object";
  const m = envelope as Record<string, unknown>;
  if ("per_layer_scores" in m) return RETIRED;
  const instruments = m.instruments as Record<string, unknown> | undefined;
  const geometry = instruments?.geometry as { readings?: Record<string, unknown> } | undefined;
  for (const reading of Object.values(geometry?.readings ?? {})) {
    if (!reading || typeof reading !== "object") return "a probe reading is not an object";
    const keys = reading as Record<string, unknown>;
    if ("subspace_coords_per_layer" in keys) return RETIRED;
    const traces = GEOMETRY_TRACE_KEYS.filter((key) => key in keys).length;
    if (traces !== 0 && traces !== GEOMETRY_TRACE_KEYS.length) return "a probe reading keeps part of its layer traces";
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
