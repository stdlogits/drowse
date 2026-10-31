// Merge + sort for the two instrument pillars' PROBE sections.
//
// The J-LENS and SAE tabs both render one visible roster of pinned probe
// cards and unpinned live-discovery cards.  Pinning changes a card's
// persistence and its actions — never its position outside the selected
// sort — so the two groups merge before sorting, not after.
import { tokenParts } from "@stdlogits/ledger";

export type InstrumentProbeSort = "strength" | "name" | "depth";

/** A probe's ID as its row writes it: a feature as ``L{layer} f{id}`` (the resident SAE's hook
 *  layer, when known), a lens word as the word (``lead`` marks a word start, drawn as ·), and a
 *  geometry probe by its name without the namespace. */
export function probeTitle(
  info: { family: string; word?: string; feature_id?: number } | null | undefined,
  name: string,
  saeLayer: number | null = null,
): { lead: boolean; text: string } {
  if (info?.family === "sae" && typeof info.feature_id === "number") {
    return { lead: false, text: `${saeLayer === null ? "" : `L${saeLayer} `}f${info.feature_id}` };
  }
  if (info?.family === "lens" && typeof info.word === "string" && info.word.length > 0) {
    const parts = tokenParts(info.word);
    return { lead: parts.lead, text: parts.empty ? JSON.stringify(info.word) : parts.text };
  }
  return { lead: false, text: name.split("/").pop() || name };
}

/** The sort keys every instrument probe row carries.  Panels extend this
 *  with whatever their card needs; the helper only reads these. */
export interface InstrumentProbeRow {
  /** Keyed-each identity. */
  key: string;
  /** What the `name` sort compares — the word, label, or feature id. */
  sortName: string;
  /** What the `strength` sort compares, in the pillar's own unit. */
  strength: number;
  /** Depth centre of mass (J-lens only).  Rows without it sort last
   *  under `depth`, which the SAE pillar never offers. */
  com?: number | null;
}

/** Merge the pinned and discovered rows into one roster in `sortMode`
 *  order.  `numericNames` picks natural-number-aware name comparison (the
 *  SAE pillar's feature ids), so `sae/9` precedes `sae/10`. */
export function mergeInstrumentProbeRows<T extends InstrumentProbeRow>(
  pinned: readonly T[],
  discovered: readonly T[],
  sortMode: InstrumentProbeSort,
  { numericNames = false }: { numericNames?: boolean } = {},
): T[] {
  const collate = numericNames
    ? (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true })
    : (a: string, b: string) => a.localeCompare(b);
  const rows = [...pinned, ...discovered];
  if (sortMode === "name") {
    rows.sort((a, b) => collate(a.sortName, b.sortName) || b.strength - a.strength);
  } else if (sortMode === "depth") {
    rows.sort(
      (a, b) =>
        (a.com ?? Infinity) - (b.com ?? Infinity) || b.strength - a.strength,
    );
  } else {
    rows.sort((a, b) => b.strength - a.strength || collate(a.sortName, b.sortName));
  }
  return rows;
}
