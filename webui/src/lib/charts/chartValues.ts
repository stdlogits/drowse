/** A layer's weight in a probe's reading, as a percent: its fitted per-layer weight (the
 *  manifold's ``mahalanobis_share``, a raw whitened spread, not a 0-to-1 share) over the sum of
 *  every layer's, which is how the monitor weighs layers into one reading. */
export function readingWeight(weight: number, weights: readonly number[]): string {
  const total = weights.reduce((sum, value) => sum + (Number.isFinite(value) ? Math.abs(value) : 0), 0);
  return chartValue(total > 0 && Number.isFinite(weight) ? Math.abs(weight) / total : 0, true);
}

export function chartValue(value: number, percentage = false): string {
  if (!Number.isFinite(value)) return "Not available";
  if (value !== 0 && Math.abs(value) < 0.0001) {
    return percentage ? (value < 0 ? "−<0.01%" : "<0.01%") : value.toExponential(2);
  }
  if (percentage) return `${(value * 100).toLocaleString("en-US", { maximumFractionDigits: 2 })}%`;
  return value.toLocaleString("en-US", { maximumFractionDigits: 4 });
}

export function barTooltip(value: number, max: number, percentage = false): string {
  if (!Number.isFinite(value)) return "Not available";
  if (percentage) return chartValue(value, true);
  if (!Number.isFinite(max) || max <= 0) return `${chartValue(value)} · scale unavailable`;
  return `${chartValue(value)} · ${chartValue(Math.abs(value) / max, true)} of scale (${chartValue(max)})`;
}

export function barExtent(value: number, max: number): number {
  return Number.isFinite(value) && Number.isFinite(max) && max > 0
    ? Math.min(1, Math.abs(value) / max) : 0;
}
