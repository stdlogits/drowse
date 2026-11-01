// The Lens room's arithmetic, store-free so the node suites load it directly: which family reads
// a layer (the server's crossover band), each source's coverage and split-half difference per
// layer, the per-family source list (prepared rows and the providers still to fetch), and the
// local fit's document ladder around the server's default.
import type { InstrumentSourceJSON, LensBandJSON, LensProviderJSON, PreparationStatusJSON } from "./types";

export type LensFamily = "R" | "S";
export type LensSite = "out" | "mid";
export const LENS_SITES: readonly LensSite[] = ["out", "mid"];

/** The kind a family fits by default (RelP), and its standard-rules alternative. */
export const FAMILY_KINDS: Record<LensFamily, readonly ["r" | "s", "j" | "k"]> = {
  R: ["r", "j"],
  S: ["s", "k"],
};

/** Layers in the model: the band's (one weight pair per layer), else one past the deepest layer
 *  any source holds, else 0. */
export function layerCount(band: LensBandJSON | null | undefined, sources: readonly InstrumentSourceJSON[]): number {
  if (band && band.weights.length > 0) return band.weights.length;
  let deepest = -1;
  for (const row of sources) {
    for (const layers of Object.values(row.layers ?? {})) {
      for (const layer of layers) deepest = Math.max(deepest, layer);
    }
  }
  return deepest + 1;
}

/** A family's weight at every layer (``w_S`` or ``w_R``), or null without a band. */
export function familyWeights(band: LensBandJSON | null | undefined, family: LensFamily): number[] | null {
  if (!band || band.weights.length === 0) return null;
  const at = family === "S" ? 0 : 1;
  return band.weights.map(pair => {
    const w = pair[at];
    return typeof w === "number" && Number.isFinite(w) ? Math.max(0, Math.min(1, w)) : 0;
  });
}

export interface Span { first: number; last: number }

/** The first and last layer where ``test`` holds, or null when it holds nowhere. */
function spanWhere(values: readonly number[], test: (value: number) => boolean): Span | null {
  let first = -1;
  let last = -1;
  values.forEach((value, layer) => {
    if (!test(value)) return;
    if (first < 0) first = layer;
    last = layer;
  });
  return first < 0 ? null : { first, last };
}

/** The layers a family reads at all (weight above zero). */
export function familySpan(band: LensBandJSON | null | undefined, family: LensFamily): Span | null {
  const weights = familyWeights(band, family);
  return weights ? spanWhere(weights, w => w > 0) : null;
}

/** The crossover: the layers both families read, each weighted (S strictly between 0 and 1). */
export function crossover(band: LensBandJSON | null | undefined): Span | null {
  const weights = familyWeights(band, "S");
  return weights ? spanWhere(weights, w => w > 0 && w < 1) : null;
}

/** ``L5–L12``, or ``L5`` for one layer. */
export function spanText(span: Span | null): string {
  if (!span) return "";
  return span.first === span.last ? `L${span.first}` : `L${span.first}–L${span.last}`;
}

export interface SiteSeries {
  /** Whether the source holds a matrix at each layer. */
  covered: boolean[];
  /** The split-half difference at each layer, null where none was recorded. */
  split: (number | null)[];
  /** Whether any layer of the site recorded one (imports do not). */
  measured: boolean;
}

/** One source's site over ``n`` layers: which layers it holds and its split-half difference. */
export function siteSeries(row: InstrumentSourceJSON | null | undefined, site: LensSite, n: number): SiteSeries {
  const covered: boolean[] = Array.from({ length: n }, () => false);
  const split: (number | null)[] = Array.from({ length: n }, () => null);
  for (const layer of row?.layers?.[site] ?? []) if (layer >= 0 && layer < n) covered[layer] = true;
  let measured = false;
  for (const [key, value] of Object.entries(row?.split_half?.[site] ?? {})) {
    const layer = Number(key);
    if (!Number.isInteger(layer) || layer < 0 || layer >= n || typeof value !== "number" || !Number.isFinite(value)) continue;
    split[layer] = value;
    measured = true;
  }
  return { covered, split, measured };
}

/** The largest split-half difference among ``rows`` at both sites: the one scale every card
 *  shares, so their bars compare. Null when none recorded one. */
export function sharedScale(rows: readonly (InstrumentSourceJSON | null | undefined)[]): number | null {
  let top = 0;
  for (const row of rows) {
    for (const values of Object.values(row?.split_half ?? {})) {
      for (const value of Object.values(values)) if (Number.isFinite(value)) top = Math.max(top, Math.abs(value));
    }
  }
  return top > 0 ? top : null;
}

/** ``0.01–0.06`` over the recorded values (one value alone when they agree), or "". */
export function rangeText(values: readonly (number | null)[]): string {
  const shown = values.filter((v): v is number => v !== null && Number.isFinite(v));
  if (shown.length === 0) return "";
  const lo = Math.min(...shown).toFixed(2);
  const hi = Math.max(...shown).toFixed(2);
  return lo === hi ? lo : `${lo}–${hi}`;
}

/** The covered layers as a span (``L0–L30``), or "" when none. */
export function coveredText(series: SiteSeries): string {
  return spanText(spanWhere(series.covered.map(c => (c ? 1 : 0)), c => c > 0));
}

const KIND_NAMES: Record<string, string> = { J: "J-lens", R: "R-lens", K: "K-lens", S: "S-lens" };

/** A kind's name when it is not its family's own (a J-lens in the R family), else "". */
export function foreignKind(lens: string | undefined, family: LensFamily): string {
  return lens && lens !== family ? (KIND_NAMES[lens] ?? lens) : "";
}

/** ``1,000 documents``. */
export function documentsText(n: number | undefined): string {
  if (!n || !Number.isFinite(n)) return "";
  return `${n.toLocaleString("en-US")} ${n === 1 ? "document" : "documents"}`;
}

/** ``out and mid``, ``out only``, ``mid only``. */
export function sitesText(sites: readonly string[] | undefined): string {
  const held = LENS_SITES.filter(site => sites?.includes(site));
  if (held.length === 2) return "out and mid";
  return held.length === 1 ? `${held[0]} only` : "";
}

/** Where a source comes from: its repository, or "local" for a fit made here. */
export function originText(row: InstrumentSourceJSON | null | undefined, provider: LensProviderJSON | null | undefined): string {
  if (row?.kind === "local") return "local";
  return row?.repo_id ?? provider?.repo_id ?? "";
}

export interface FamilyItem {
  source: string;
  /** The prepared row, null for a provider still to fetch. */
  row: InstrumentSourceJSON | null;
  provider: LensProviderJSON | null;
}

/** A family's sources in picker order: every provider (prepared or not, in the server's order),
 *  then the prepared rows no provider names (local fits), in the server's preference order. */
export function familyItems(
  sources: readonly InstrumentSourceJSON[],
  providers: readonly LensProviderJSON[],
  family: LensFamily,
): FamilyItem[] {
  const prepared = new Map(sources.filter(row => row.family === family).map(row => [row.source, row]));
  const items: FamilyItem[] = providers
    .filter(provider => provider.family === family)
    .map(provider => ({ source: provider.source, row: prepared.get(provider.source) ?? null, provider }));
  const named = new Set(items.map(item => item.source));
  for (const row of prepared.values()) {
    if (!named.has(row.source)) items.push({ source: row.source, row, provider: null });
  }
  return items;
}

/** The family a source string selects into (``local:r`` → R), from its row or provider, else
 *  from a local name's kind letter. Null when nothing says. */
export function sourceFamily(
  source: string | null | undefined,
  sources: readonly InstrumentSourceJSON[],
  providers: readonly LensProviderJSON[],
): LensFamily | null {
  if (!source) return null;
  const known = sources.find(row => row.source === source)?.family ?? providers.find(row => row.source === source)?.family;
  if (known) return known;
  const local = /^local:([jrks])/.exec(source);
  if (!local) return null;
  return local[1] === "j" || local[1] === "r" ? "R" : "S";
}

/** The layer under a pointer in a chart whose columns carry ``data-layer``, or -1 off them. */
export function layerAt(event: Event): number {
  const target = event.target as Element | null;
  const column = target?.closest?.("[data-layer]") as HTMLElement | null;
  const layer = Number(column?.dataset.layer);
  return Number.isInteger(layer) ? layer : -1;
}

/** Whether a family's selected source fails to load for these weights: the server says so on the
 *  active row (``compatible``); a runtime that does not say counts as loaded. */
export function unloaded(row: InstrumentSourceJSON | null | undefined): boolean {
  return row?.active === true && row.compatible === false;
}

/** The body a local fit sent, so a refused one can be sent again with ``force``. */
export type FitBody = { prompts: number; kind: string; sites: string[] };

/** The artifact a failed fit may replace (``local:r``): only when the server marked the failure as
 *  the refusal to replace it (``replaceable``) and the fit is the one this view sent. Any other
 *  failure (an unreachable Hub, a cancelled fit) offers nothing. */
export function replaceTarget(
  status: Pick<PreparationStatusJSON, "state" | "source" | "replaceable"> | null | undefined,
  sent: FitBody | null,
): string | null {
  if (!status || !sent || status.state !== "error" || status.replaceable !== true) return null;
  const target = `local:${sent.kind}`;
  return status.source === target ? target : null;
}

/** The room's own words for a refused replacement (the server's names the CLI flag and the Python
 *  keyword): what ``target`` holds and what the fit would put in its place. */
export function replaceMessage(target: string, held: number | undefined, prompts: number): string {
  const holds = documentsText(held);
  return `${target} already holds a lens${holds ? ` over ${holds}` : ""}. This fit, over ${documentsText(prompts)}, would replace it.`;
}

/** A running fit's line: its stage until the first document is fitted ("Streaming 64 documents
 *  from …"), then the documents done. */
export function fitProgressText(current: number, total: number, message: string | null | undefined): string {
  const stage = (message ?? "").trim();
  if (current <= 0 && stage) return stage.charAt(0).toUpperCase() + stage.slice(1);
  return `${current} of ${total} documents`;
}

/** The documents a local fit can average over: the server's default doubled and halved, from an
 *  eighth of it to sixteen times it, whole and at least one. */
export function documentLadder(defaultDocuments: number): number[] {
  const base = Number.isFinite(defaultDocuments) && defaultDocuments >= 1 ? Math.round(defaultDocuments) : 1;
  const steps = [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16].map(f => Math.max(1, Math.round(base * f)));
  return [...new Set(steps)];
}

/** The next rung of ``ladder`` from ``current`` in direction ``dir`` (clamped at the ends). */
export function stepDocuments(ladder: readonly number[], current: number, dir: -1 | 1): number {
  if (ladder.length === 0) return current;
  if (dir > 0) return ladder.find(n => n > current) ?? ladder[ladder.length - 1];
  for (let i = ladder.length - 1; i >= 0; i -= 1) if (ladder[i] < current) return ladder[i];
  return ladder[0];
}
