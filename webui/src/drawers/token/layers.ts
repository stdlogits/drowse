// The token view's layer math: where a layer sits on the depth ramp, how the chosen token's
// probability builds over depth in the lens readout, the layer it leads from, and which layers
// the folded table shows. Pure functions over the readout, so the view only draws.

export interface LensCell { id: number; token: string; logprob: number }
export interface LensLayer { layer: number; tokens: LensCell[] }

/** A layer's colour on the depth ramp: --deep at the first layer, --late at the last. */
export function depthTint(layer: number, last: number): string {
  const share = last > 0 ? Math.min(1, Math.max(0, layer / last)) : 1;
  return `color-mix(in oklch, var(--deep) ${Math.round(100 - share * 100)}%, var(--late))`;
}

/** The deepest layer a readout names (the ramp's end when the model's depth is not known). */
export function lastLayer(layers: readonly LensLayer[]): number {
  return layers.reduce((most, row) => Math.max(most, row.layer), 0);
}

/** The chosen token's probability at each layer; null where it fell below the layer's top-k
 *  (unmeasured, which is not zero). Matched by id, so a whitespace variant never stands in. */
export function chosenByDepth(layers: readonly LensLayer[], tokenId: number | null): { layer: number; p: number | null }[] {
  return layers.map(row => {
    const hit = tokenId == null ? undefined : row.tokens.find(cell => cell.id === tokenId);
    return { layer: row.layer, p: hit ? Math.exp(hit.logprob) : null };
  });
}

/** The first layer from which the chosen token is every later layer's first reading; null when the
 *  last layer reads something else (or the token is unknown). */
export function leadLayer(layers: readonly LensLayer[], tokenId: number | null): number | null {
  if (tokenId == null || layers.length === 0) return null;
  let lead: number | null = null;
  for (let i = layers.length - 1; i >= 0; i -= 1) {
    if (layers[i].tokens[0]?.id !== tokenId) break;
    lead = layers[i].layer;
  }
  return lead;
}

/** The layers the folded table shows: every third, the last, and the lead. All of them when the
 *  readout is short enough not to fold. */
export function foldedLayers(layers: readonly number[], lead: number | null, step = 3): Set<number> {
  if (layers.length <= step * 4) return new Set(layers);
  const last = layers[layers.length - 1];
  return new Set(layers.filter((layer, i) => i % step === 0 || layer === last || layer === lead));
}

/** A probability as the view prints it: two places, three below 0.01, then an exponent. */
export function formatP(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return "–";
  if (p >= 0.01) return p.toFixed(2);
  if (p >= 0.001) return p.toFixed(3);
  return p.toExponential(1);
}

/** A pinned probe's value in the Readings list: a probability as ``formatP`` writes it, anything
 *  else (an activation, a unit's value over its scale, a footprint z) signed to two places, as the
 *  Inspector's rows are. */
export function readingText(value: number, unit: string | null | undefined): string {
  return unit === "mean_token_probability" || unit === "token_probability" ? formatP(value) : formatSigned(value, 2);
}

/** A signed reading: a true minus and a plus, two places. */
export function formatSigned(value: number, places = 2): string {
  if (!Number.isFinite(value)) return "–";
  const text = Math.abs(value).toFixed(places);
  if (Number(text) === 0) return text;
  return `${value < 0 ? "−" : "+"}${text}`;
}
