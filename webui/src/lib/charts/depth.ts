// The depth ramp and the trace mark's arithmetic, kept free of stores and components so the
// node suites can load it directly.
//
// Colour is depth: a layer's place on the ramp from --deep (layer 0) to --late (the last layer),
// mixed in oklch (Ledger's accent mix). A reading drawn at a depth sets --tint to this mix and
// fills with it; it never colours text.

/** The ramp's colour at a depth in [0, 1] (0 = the first layer, 1 = the last). Out-of-range and
 *  non-finite depths clamp; a missing depth is null, so the caller keeps the inherited --tint. */
export function depthTint(depth: number | null | undefined): string | null {
  if (depth == null || !Number.isFinite(depth)) return null;
  const d = Math.max(0, Math.min(1, depth));
  return `color-mix(in oklch, var(--deep) ${Math.round((1 - d) * 100)}%, var(--late))`;
}

/** A layer's depth among the layers a reading covers, by index over the deepest one seen (the
 *  client does not know the model's depth, so the deepest fitted layer stands in for the last). */
export function layerDepth(layer: number, deepest: number): number | null {
  if (!Number.isFinite(layer) || !Number.isFinite(deepest) || deepest <= 0) return null;
  return Math.max(0, Math.min(1, layer / deepest));
}

/** The layer whose reading is largest in magnitude (where a probe peaks), or null with none. */
export function peakLayer(perLayer: Record<string, number> | null | undefined): number | null {
  let best: number | null = null;
  let bestAbs = -1;
  for (const [key, value] of Object.entries(perLayer ?? {})) {
    const layer = Number(key);
    if (!Number.isFinite(layer) || typeof value !== "number" || !Number.isFinite(value)) continue;
    const a = Math.abs(value);
    if (a > bestAbs) {
      bestAbs = a;
      best = layer;
    }
  }
  return best;
}

export interface TraceBar {
  /** Height as a percent of the mark, 0 for a gap (a missing reading). */
  height: number;
  /** A reading below zero: drawn in faint ink, never in the accent. */
  negative: boolean;
  /** No reading here (null or non-finite). */
  missing: boolean;
}

/** One bar per reading, the most recent ``limit`` of them, scaled to ``scale`` (the largest
 *  magnitude shown when no scale is given). Signed readings draw their magnitude; a nonzero reading
 *  keeps a sliver of height so it never reads as missing. */
export function traceBars(
  values: readonly (number | null | undefined)[],
  { scale, limit = 48 }: { scale?: number; limit?: number } = {},
): TraceBar[] {
  const shown = limit > 0 && values.length > limit ? values.slice(values.length - limit) : values;
  let top = scale !== undefined && Number.isFinite(scale) && scale > 0 ? scale : 0;
  if (top === 0) {
    for (const v of shown) if (typeof v === "number" && Number.isFinite(v)) top = Math.max(top, Math.abs(v));
  }
  return shown.map((v) => {
    if (typeof v !== "number" || !Number.isFinite(v)) return { height: 0, negative: false, missing: true };
    const share = top > 0 ? Math.min(1, Math.abs(v) / top) : 0;
    return { height: v === 0 ? 0 : Math.max(8, Math.round(share * 1000) / 10), negative: v < 0, missing: false };
  });
}

/** At most ``max`` readings: past that, consecutive readings share a bar that keeps the largest
 *  magnitude among them (a long reply still shows its peaks). ``size`` is readings per bar. */
export function bucketed(
  values: readonly (number | null | undefined)[],
  max: number,
): { values: (number | null)[]; size: number } {
  const clean = (v: number | null | undefined) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  if (max <= 0 || values.length <= max) return { values: values.map(clean), size: 1 };
  const size = Math.ceil(values.length / max);
  const out: (number | null)[] = [];
  for (let start = 0; start < values.length; start += size) {
    let pick: number | null = null;
    for (const v of values.slice(start, start + size)) {
      const c = clean(v);
      if (c !== null && (pick === null || Math.abs(c) > Math.abs(pick))) pick = c;
    }
    out.push(pick);
  }
  return { values: out, size };
}

/** A reading shown beside its row: two decimals, a true minus sign, and σ-free. */
export function readingText(value: number | null | undefined, { signed = false, percent = false } = {}): string {
  if (value == null || !Number.isFinite(value)) return "–";
  if (percent) {
    const p = value * 100;
    return `${p !== 0 && Math.abs(p) < 0.1 ? "<0.1" : p.toFixed(p >= 10 ? 0 : 1)}%`;
  }
  const text = Math.abs(value).toFixed(Math.abs(value) >= 100 ? 0 : 2);
  if (value < 0 && text !== "0.00") return `−${text}`;
  return signed && value > 0 ? `+${text}` : text;
}
