// A steering term's layers: the ``{…}`` clause it writes and the depth strip's arithmetic. Pure
// functions over plain arrays, kept free of stores and components so the node suites load them.
//
// The clause text mirrors the grammar's formatter (drowse/model/layers.py::format_layer_indices):
// indices sorted, each maximal run of consecutive layers as ``a-b`` and a lone layer as ``a``,
// joined by ``,`` with no spaces; ``workspace`` lower-case; every fitted layer writes no clause.
// The server's parser is the only parser, so nothing here reads a clause back.

import type { LayerScope } from "../types";

/** Sorted, distinct, non-negative integer layers. */
export function sortedLayers(layers: Iterable<number>): number[] {
  return [...new Set([...layers].filter(L => Number.isInteger(L) && L >= 0))].sort((a, b) => a - b);
}

/** The maximal runs of consecutive layers, as ``[first, last]`` pairs. */
export function layerRuns(layers: Iterable<number>): [number, number][] {
  const runs: [number, number][] = [];
  for (const L of sortedLayers(layers)) {
    const last = runs.at(-1);
    if (last && L === last[1] + 1) last[1] = L;
    else runs.push([L, L]);
  }
  return runs;
}

/** The canonical index text the grammar writes: ``4-8,20``. */
export function formatLayerIndices(layers: Iterable<number>): string {
  return layerRuns(layers).map(([a, b]) => (a === b ? `${a}` : `${a}-${b}`)).join(",");
}

/** The runs for reading, with an en dash and a space after each comma: ``4–8, 20``. */
export function layerRunsText(layers: Iterable<number>): string {
  return layerRuns(layers).map(([a, b]) => (a === b ? `${a}` : `${a}–${b}`)).join(", ");
}

/** The clause a term writes after its selector: nothing for every fitted layer, ``{workspace}``,
 *  or ``{4-8,20}``. The empty selection writes nothing too: such a term is left out of the
 *  expression (``termIsInert``), so its clause is never sent. */
export function layerClause(scope: LayerScope | undefined): string {
  if (scope == null) return "";
  if (scope === "workspace") return "{workspace}";
  const text = formatLayerIndices(scope);
  return text ? `{${text}}` : "";
}

/** A term scoped to no layer does nothing, so the expression leaves it out. */
export function termIsInert(scope: LayerScope | undefined): boolean {
  return Array.isArray(scope) && scope.length === 0;
}

/** The scope a selection writes: ``null`` when it is exactly every fitted layer. */
export function scopeForSelection(selected: Iterable<number>, fitted: Iterable<number>): LayerScope {
  const pick = sortedLayers(selected);
  const all = sortedLayers(fitted);
  if (pick.length === all.length && pick.every((L, i) => L === all[i])) return null;
  return pick;
}

/** The selection after a paint stroke reaches ``layer``: added when ``on``, removed otherwise. */
export function paintLayer(selected: readonly number[], layer: number, on: boolean): number[] {
  const has = selected.includes(layer);
  if (has === on) return [...selected];
  return on ? sortedLayers([...selected, layer]) : selected.filter(L => L !== layer);
}

/** The depth ramp's colour at ``layer`` of a model ``nLayers`` deep (Ledger's accent mix). */
export function layerTint(layer: number, nLayers: number): string {
  const last = Math.max(1, nLayers - 1);
  const depth = Math.max(0, Math.min(1, layer / last));
  return `color-mix(in oklch, var(--deep) ${Math.round((1 - depth) * 100)}%, var(--late))`;
}

/** Where a term acts over the model's depth, for a chip's span: the stretch from its first to its
 *  last layer as a percent of the depth, filled from the first layer's colour to the last's. */
export function depthSpan(
  layers: readonly number[],
  nLayers: number,
): { left: number; width: number; background: string } | null {
  const pick = sortedLayers(layers).filter(L => L < nLayers);
  if (pick.length === 0 || nLayers <= 0) return null;
  const first = pick[0];
  const last = pick[pick.length - 1];
  const round = (value: number) => Math.round(value * 10) / 10;
  const background = first === last
    ? layerTint(first, nLayers)
    : `linear-gradient(90deg, ${layerTint(first, nLayers)}, ${layerTint(last, nLayers)})`;
  return { left: round((first / nLayers) * 100), width: round(((last - first + 1) / nLayers) * 100), background };
}

/** The layer a term's share peaks at among ``layers`` (its colour on the editor), or null. */
export function peakLayer(fitted: readonly number[], share: readonly number[], layers?: readonly number[]): number | null {
  let best: number | null = null;
  let top = -Infinity;
  fitted.forEach((L, i) => {
    if (layers && !layers.includes(L)) return;
    const value = share[i] ?? 0;
    if (value > top) {
      top = value;
      best = L;
    }
  });
  return best;
}

/** Each layer's push as composition weighs it under this selection: its share over the mean share
 *  of the selected layers (the mean-1 normalization runs over the selection). Null outside it. */
export function selectedGain(
  fitted: readonly number[],
  share: readonly number[],
  selected: readonly number[],
): Map<number, number> {
  const picked = fitted.map((L, i) => [L, share[i] ?? 0] as const).filter(([L]) => selected.includes(L));
  const mean = picked.reduce((sum, [, s]) => sum + s, 0) / (picked.length || 1);
  return new Map(picked.map(([L, s]) => [L, mean > 0 ? s / mean : 1]));
}
