import type { AnyReadingJSON, MeasurementsEnvelopeJSON, ProbeInfo, ProbeReadingJSON } from "./types";

/** The reading a saved reply settled on for ``name``: its node's aggregate
 *  (``aggregate_measurements``), the reading the live rack showed when the reply finished. A
 *  geometry reading is its summary (share, coordinates, nearest, assignment, residual, depth; no
 *  per-layer traces), a scalar reading whole (a lens probe's band and depth). Null when the node
 *  kept none (an authored or imported turn, a reply that measured nothing) or the probe was not
 *  read. */
export function savedProbeReading(
  info: ProbeInfo,
  name: string,
  node: { aggregate_measurements: MeasurementsEnvelopeJSON | null } | undefined,
): AnyReadingJSON | null {
  const instruments = node?.aggregate_measurements?.instruments as
    | Partial<Record<string, { readings?: Record<string, AnyReadingJSON> }>>
    | undefined;
  return instruments?.[info.family]?.readings?.[name] ?? null;
}

/** What the rack shows for a probe on a saved reply: the node's flat aggregate scores (its axis-0
 *  value, exact), and, from ``reading`` (``savedProbeReading``), what the scores do not keep: a
 *  curved probe's share and its other coordinates. */
export function savedProbeSummary(info: ProbeInfo, scores: Record<string, number>, reading: AnyReadingJSON | null = null) {
  const scalar = (key: string): number | null => {
    const value = scores[key];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  };
  const geometry = info.family === "geometry" && reading && !("value" in reading) ? (reading as ProbeReadingJSON) : null;
  const finite = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);
  const coordinates: number[] = [];
  const fraction = info.family === "geometry" ? scalar(`${info.name}:fraction`) ?? finite(geometry?.fraction) : null;
  if (info.family === "geometry") {
    for (let axis = 0; axis < info.intrinsic_dim; axis += 1) {
      const value = (axis === 0 ? scalar(info.name) : scalar(`${info.name}[${axis}]`)) ?? finite(geometry?.coords?.[axis]);
      if (value === null) break;
      coordinates.push(value);
    }
  }
  const scalarValue = reading && "value" in reading ? finite(reading.value) : null;
  return {
    value: info.family === "geometry" && !info.is_affine ? fraction : scalar(info.name) ?? (info.family === "geometry" ? coordinates[0] ?? null : scalarValue),
    coordinates,
    fraction,
  };
}
