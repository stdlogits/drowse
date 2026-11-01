// Units (MLP neurons and attention heads) as the dashboard shows them: their IDs, the reads of a
// unit view, the activity over a conversation and the atlas's arithmetic. Store-free, so the node
// suites load it directly; lib/stores/units.svelte.ts fetches, the views draw.
//
// A unit's reference is the engine's (``neuron/L31n4410``, ``head/L28h12``) and its ID as the
// views write it is ``L31 n4410`` / ``L28 h12``.
import { stem } from "@stdlogits/ledger";
import type {
  ActivityPeak,
  HeadReadJSON,
  NeuronPoleJSON,
  NeuronReadJSON,
  UnitHalfJSON,
  UnitLabelJSON,
  UnitPlaceJSON,
  UnitTokenJSON,
} from "./types";

export type UnitKind = "neuron" | "head";

export interface UnitRef {
  kind: UnitKind;
  layer: number;
  index: number;
  /** The engine's reference: ``neuron/L31n4410``. */
  id: string;
}

/** A unit's reference from what a person types or a route names: ``neuron/L31n4410``, ``L31 n4410``,
 *  ``l31n4410``, ``L28 h12``, ``head/L28h12``. Null when it names no unit. */
export function parseUnit(text: string): UnitRef | null {
  const t = text.trim().replace(/^(neuron|head)\//i, (_, kind: string) => `${kind.toLowerCase()[0]}:`);
  const m = /^(?:([nh]):)?l\s*(\d{1,3})\s*[\s/]?\s*([nh])\s*(\d{1,6})$/i.exec(t);
  if (!m) return null;
  const letter = m[3].toLowerCase();
  if (m[1] && m[1] !== letter) return null;
  const kind: UnitKind = letter === "n" ? "neuron" : "head";
  const layer = Number(m[2]);
  const index = Number(m[4]);
  return { kind, layer, index, id: `${kind}/L${layer}${letter}${index}` };
}

/** The ID as the views write it: ``L31 n4410``. A reference that does not parse comes back as is. */
export function unitTitle(id: string): string {
  const ref = parseUnit(id);
  return ref ? `L${ref.layer} ${ref.kind === "neuron" ? "n" : "h"}${ref.index}` : id;
}

// ---------------------------------------------------------------- reads

/** A half's lens as its group's meta names it: ``R:out`` → ``R · out``; ``logit`` and
 *  ``embedding`` as they are. */
export function lensText(half: Pick<UnitHalfJSON, "lens" | "family">): string {
  if (!half.lens) return half.family;
  const [family, site] = half.lens.split(":");
  return site ? `${family} · ${site}` : family;
}

/** A place's two halves in the depth rule's order: the one its depth calls for first (the larger
 *  weight; R on a tie), the other, faded, second. An absent primary yields to a present other. */
export function orderedHalves(place: Pick<UnitPlaceJSON, "r" | "s">): { primary: UnitHalfJSON; other: UnitHalfJSON } {
  let primary = place.s.weight > place.r.weight ? place.s : place.r;
  let other = primary === place.r ? place.s : place.r;
  if (primary.absent && !other.absent) [primary, other] = [other, primary];
  return { primary, other };
}

export interface ReadToken {
  /** The token as the tokenizer writes it (a leading space is a word start). */
  token: string;
  /** Variants merged into it (release, released, Release…). */
  count: number;
  /** Another group names it too. */
  agree: boolean;
  value: number;
}

/** A read's tokens with their variants merged (first spelling kept, counted), in rank order. */
export function mergedTokens(tokens: readonly UnitTokenJSON[]): { token: string; count: number; value: number; key: string }[] {
  const out: { token: string; count: number; value: number; key: string }[] = [];
  const at = new Map<string, number>();
  for (const t of tokens) {
    const key = stem(t.token);
    const i = at.get(key);
    if (i === undefined) {
      at.set(key, out.length);
      out.push({ token: t.token, count: 1, value: t.value, key });
    } else {
      out[i] = { ...out[i], count: out[i].count + 1 };
    }
  }
  return out;
}

/** Groups of tokens with ``agree`` set on a token whose variant key another group also names. */
export function withAgreement(groups: readonly (readonly UnitTokenJSON[])[]): ReadToken[][] {
  const merged = groups.map(mergedTokens);
  const seen = new Map<string, number>();
  for (const group of merged) for (const key of new Set(group.map(t => t.key))) seen.set(key, (seen.get(key) ?? 0) + 1);
  return merged.map(group => group.map(({ token, count, value, key }) => ({ token, count, value, agree: (seen.get(key) ?? 0) > 1 })));
}

/** A neuron's pole (``+`` or ``−``) from a read. */
export function poleOf(read: NeuronReadJSON, pole: "+" | "-"): NeuronPoleJSON | null {
  return read.poles.find(p => p.pole === pole) ?? null;
}

/** The label line a unit view shows: what it fires on (or attends to), else what it writes. */
export function labelText(label: UnitLabelJSON | null | undefined): string | null {
  if (!label) return null;
  const text = label.fires ?? label.attends ?? label.writes ?? label.note ?? null;
  return typeof text === "string" && text.trim() ? text.trim() : null;
}

export interface UnitFacts {
  /** A pole's typical high firing (neutral ``act_top``), or a head's typical write norm. */
  typical: number | null;
  /** Share of tokens it fires on (the catalog's rate), 0–1. */
  rate: number | null;
  /** The neuron's write norm. */
  writeNorm: number | null;
  /** A head's copy score through the lens its depth calls for. */
  copy: number | null;
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** The facts line of a unit view: for a neuron, the shown pole's. */
export function unitFacts(read: NeuronReadJSON | HeadReadJSON, pole: "+" | "-" = "+"): UnitFacts {
  if (read.kind === "neuron") {
    const p = poleOf(read, pole);
    return {
      typical: num(p?.firing.neutral?.act_top),
      rate: num(p?.firing.rate),
      writeNorm: num(read.scalars.w_norm),
      copy: null,
    };
  }
  const { primary } = orderedHalves(read.copies);
  return {
    typical: num(read.firing.neutral?.write_top),
    rate: num(read.firing.rate),
    writeNorm: null,
    copy: num(read.copy_score[primary.family]),
  };
}

// ---------------------------------------------------------------- activity

/** A firing over a pole's typical high firing counts from a quarter of it (the catalog's rule:
 *  ``|act| > act_top / 4``), so a trace in ``activation_over_scale`` fires at 0.25. */
export const FIRES_AT = 0.25;

/** How many of a trace's tokens the unit fires on (either pole). */
export function firesOn(trace: readonly (number | null)[], at = FIRES_AT): number {
  let n = 0;
  for (const v of trace) if (typeof v === "number" && Number.isFinite(v) && Math.abs(v) >= at) n += 1;
  return n;
}

/** The token a trace peaks on (largest magnitude), or -1 for an empty or silent trace. */
export function peakIndex(trace: readonly (number | null)[]): number {
  let best = -1;
  let top = 0;
  trace.forEach((v, i) => {
    if (typeof v === "number" && Number.isFinite(v) && Math.abs(v) > top) {
      top = Math.abs(v);
      best = i;
    }
  });
  return best;
}

/** The side of a neuron that fired here: the sign of its trace's peak (``+`` for a silent trace).
 *  The unit view opens on it and the room's summary reads it, so the two show one pole. */
export function peakPole(trace: readonly (number | null)[]): "+" | "-" {
  const at = peakIndex(trace);
  return at >= 0 && (trace[at] ?? 0) < 0 ? "-" : "+";
}

/** A float16 little-endian base64 array (``float16-le-base64``) as numbers. */
export function decodeFloat16(text: string): Float32Array {
  const bytes = Uint8Array.from(atob(text), c => c.charCodeAt(0));
  const out = new Float32Array(bytes.length >> 1);
  for (let i = 0; i < out.length; i++) {
    const h = bytes[2 * i] | (bytes[2 * i + 1] << 8);
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    out[i] = exp === 0
      ? sign * frac * 2 ** -24
      : exp === 0x1f
        ? (frac ? NaN : sign * Infinity)
        : sign * (1 + frac / 1024) * 2 ** (exp - 15);
  }
  return out;
}

/** The atlas's ten shades (alpha), faintest first: a unit's shade is its decile. */
export const SHADES = [0.1, 0.16, 0.24, 0.32, 0.42, 0.52, 0.62, 0.74, 0.86, 1] as const;

/** Nine cut points splitting ``values``' magnitudes into ten equal shares (from a strided sample of
 *  at most ``sample`` values, so a model's whole population costs one sort of the sample). */
export function decileCuts(values: ArrayLike<number>, sample = 20000): number[] {
  const n = values.length;
  if (n === 0) return new Array(9).fill(Infinity);
  const step = Math.max(1, Math.floor(n / sample));
  const picked: number[] = [];
  for (let i = 0; i < n; i += step) {
    const v = Math.abs(values[i]);
    if (Number.isFinite(v)) picked.push(v);
  }
  picked.sort((a, b) => a - b);
  if (picked.length === 0) return new Array(9).fill(Infinity);
  return [1, 2, 3, 4, 5, 6, 7, 8, 9].map(d => picked[Math.min(picked.length - 1, Math.floor((d / 10) * picked.length))]);
}

/** A magnitude's shade (0–9) against ``decileCuts``. */
export function shadeOf(value: number, cuts: readonly number[]): number {
  const a = Math.abs(value);
  if (!Number.isFinite(a)) return 9;
  let k = 0;
  while (k < 9 && a >= cuts[k]) k += 1;
  return k;
}

export interface AtlasBar {
  id: string;
  index: number;
  value: number;
  /** 0–9 by rank among every bar shown. */
  shade: number;
}

/** The strongest units of each layer (``perLayer`` at most), each column sorted by magnitude and
 *  every bar given one of ten shades by its rank among all the bars shown. */
export function strongestColumns(
  peaks: readonly Pick<ActivityPeak, "unit" | "layer" | "index" | "peak">[],
  layers: number,
  perLayer = 40,
): AtlasBar[][] {
  const cols: { id: string; index: number; value: number }[][] = Array.from({ length: layers }, () => []);
  for (const p of peaks) {
    if (p.layer < 0 || p.layer >= layers) continue;
    cols[p.layer].push({ id: p.unit, index: p.index, value: p.peak });
  }
  for (const col of cols) {
    col.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    col.splice(perLayer);
  }
  const all = cols.flatMap((col, L) => col.map((bar, i) => ({ L, i, a: Math.abs(bar.value) })));
  all.sort((a, b) => a.a - b.a);
  const shade = cols.map(col => col.map(() => 0));
  all.forEach(({ L, i }, r) => { shade[L][i] = Math.min(9, Math.floor((r / all.length) * 10)); });
  return cols.map((col, L) => col.map((bar, i) => ({ ...bar, shade: shade[L][i] })));
}

/** One column's units, every one of them, laid out ``g`` a row in the bars' share of the column
 *  (``fw`` × ``fh`` px), each cell as square as the column allows. */
export function fieldGeometry(fw: number, fh: number, count: number): { g: number; cw: number; ch: number; rows: number } {
  if (fw <= 0 || fh <= 0 || count <= 0) return { g: 1, cw: 0, ch: 0, rows: 0 };
  const side = Math.sqrt((fw * fh) / count);
  const g = Math.max(1, Math.floor(fw / side));
  const rows = Math.ceil(count / g);
  return { g, cw: fw / g, ch: fh / rows, rows };
}

// ---------------------------------------------------------------- the search

export type UnitQuery =
  | { mode: "active" }
  | { mode: "id"; ref: UnitRef | null; text: string }
  | { mode: "word"; word: string };

/** What the room's search names: nothing (the most active units), a unit's ID, or a word (the units
 *  that write it). A word keeps its leading space when typed with one or a ·, so ``·stars`` asks
 *  for the word start. */
export function parseQuery(raw: string): UnitQuery {
  const text = raw.trim();
  if (!text) return { mode: "active" };
  const ref = parseUnit(text);
  if (ref) return { mode: "id", ref, text };
  if (/^(neuron|head)\//i.test(text) || /^l\s*\d+\s*[nh]\s*\d*$/i.test(text)) return { mode: "id", ref: null, text };
  const word = text.replace(/^·/, " ");
  return { mode: "word", word: raw.startsWith(" ") || text.startsWith("·") ? ` ${word.trim()}` : word };
}
