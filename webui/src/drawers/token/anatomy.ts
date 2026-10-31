// The token anatomy's math: which read a cell shows (S early, R late, both in the crossover band),
// the chosen token over depth, the units that pushed it, and where the top heads read from. Pure
// functions over the replay envelope's `anatomy`, so the view only draws.

import type {
  AnatomyLayer,
  AnatomyPattern,
  AnatomyUnit,
  FamilyReadDict,
  LensReadDict,
  TokenScoreDict,
  UnitLabelJSON,
} from "../../lib/types";
import { formatP } from "./layers";

export type AnatomyColumn = "reads_as" | "attention" | "mlp" | "steering";

/** One token a cell shows, from one family's half. */
export interface CellToken {
  id: number;
  token: string;
  family: "R" | "S";
  /** The lighter half's token in the band, drawn faint. */
  faint: boolean;
  tip: string;
}

export interface AnatomyCell {
  /** The heavier half's top token, then the lighter half's when the band shows both and they differ. */
  tokens: CellToken[];
  /** Both halves read the same top token (the 14% agreement tint). */
  agree: boolean;
  /** Why nothing is shown (both halves absent, or the only weighted one): the engine's reason. */
  absent: string | null;
  /** The family whose absence the cell shows (the heavier half's first), for `absentTip`. */
  absentFamily: "R" | "S" | null;
}

function present(half: FamilyReadDict): boolean {
  return half.absent === null && half.tokens.length > 0;
}

/** A half's value as the tip prints it: R a probability, S a footprint z. */
export function halfValue(half: FamilyReadDict, score: TokenScoreDict): string {
  return half.unit === "probability" ? `p ${formatP(score.value)}` : `z ${score.value.toFixed(2)}`;
}

function lensName(half: FamilyReadDict): string {
  return half.lens ?? `${half.family}-lens`;
}

/** The halves heaviest first; on a tie S first (the depth rule's shallow-faithful order). */
function byWeight(read: LensReadDict): [FamilyReadDict, FamilyReadDict] {
  return read.r.weight > read.s.weight ? [read.r, read.s] : [read.s, read.r];
}

/** The depth-picked read of one cell: the heavier half's top token; in the band (both weighted)
 *  the lighter half's top token too, faint, unless both read the same one. A heavy half that came
 *  back absent gives way to a weighted light one; never to a half of weight 0. */
export function pickCell(read: LensReadDict): AnatomyCell {
  const [heavy, light] = byWeight(read);
  const lightWeighted = light.weight > 0;
  const tokenOf = (half: FamilyReadDict, faint: boolean): CellToken => {
    const top = half.tokens[0];
    return { id: top.token_id, token: top.token, family: half.family, faint, tip: `${lensName(half)} · ${halfValue(half, top)}` };
  };
  const shown = { agree: false, absent: null, absentFamily: null };
  if (present(heavy)) {
    const first = tokenOf(heavy, false);
    if (!lightWeighted || !present(light)) return { ...shown, tokens: [first] };
    const second = tokenOf(light, true);
    if (second.id === first.id) {
      return { ...shown, tokens: [{ ...first, tip: `${first.tip} · ${second.tip}` }], agree: true };
    }
    return { ...shown, tokens: [first, second] };
  }
  if (lightWeighted && present(light)) return { ...shown, tokens: [tokenOf(light, true)] };
  const why = heavy.absent !== null ? heavy : light.absent !== null ? light : null;
  return { tokens: [], agree: false, absent: why?.absent ?? "No read here", absentFamily: why?.family ?? null };
}

/** The layers at which each column's family came back (not absent), keyed `column:family`. */
export function lensCoverage(layers: readonly AnatomyLayer[]): Map<string, number[]> {
  const out = new Map<string, number[]>();
  for (const column of ["reads_as", "attention", "mlp", "steering"] as const) {
    for (const family of ["R", "S"] as const) {
      out.set(`${column}:${family}`, layers.filter(row => {
        const read = columnRead(row, column);
        return read !== null && (family === "R" ? read.r : read.s).absent === null;
      }).map(row => row.layer));
    }
  }
  return out;
}

/** A short reason for a family's absent half at a layer, from where that family does read (the
 *  engine's reason names the fitted layer list): the lens stops early, starts late, skips the
 *  layer, does not read the column's site, or is not selected. */
export function absentTip(coverage: ReadonlyMap<string, number[]>, column: AnatomyColumn, family: "R" | "S", layer: number): string {
  const here = coverage.get(`${column}:${family}`) ?? [];
  if (here.length === 0) {
    const anywhere = [...coverage].some(([key, read]) => key.endsWith(`:${family}`) && read.length > 0);
    if (!anywhere) return `No ${family} lens is selected`;
    return column === "attention" ? `The ${family} lens does not read attention` : `The ${family} lens does not read here`;
  }
  const first = here[0];
  const last = here[here.length - 1];
  if (layer > last) return `The ${family} lens stops at L${last}`;
  if (layer < first) return `The ${family} lens starts at L${first}`;
  return `The ${family} lens skips L${layer}`;
}

/** The row's lens mark: S or R outside the band, both (heavier first) inside it. */
export function lensTag(read: LensReadDict): string {
  const s = read.s.weight;
  if (s >= 1) return "S";
  if (s <= 0) return "R";
  return s >= 0.5 ? "S R" : "R S";
}

/** The row's weights, for its tip. */
export function weightTip(read: LensReadDict): string {
  const s = read.s.weight;
  if (s >= 1) return "S-lens";
  if (s <= 0) return "R-lens";
  return `S-lens ${s.toFixed(2)} · R-lens ${read.r.weight.toFixed(2)}`;
}

/** A column's read at a layer (the steering read is null where nothing was injected). */
export function columnRead(layer: AnatomyLayer, column: AnatomyColumn): LensReadDict | null {
  return column === "steering" ? layer.steering : layer[column];
}

/** The chosen token's R probability over depth, from what each layer's residual reads as; null
 *  where it fell below the read's top k. A last layer the R lens does not reach reads as the
 *  model's own output, so it takes the token's probability (`final`). */
export function anatomyByDepth(
  layers: readonly AnatomyLayer[],
  tokenId: number,
  probability: number,
): { layer: number; p: number | null; final: boolean }[] {
  const last = layers.length - 1;
  return layers.map((row, i) => {
    const r = row.reads_as.r;
    if (!present(r) && i === last) return { layer: row.layer, p: probability, final: true };
    const hit = r.tokens.find(score => score.token_id === tokenId);
    return { layer: row.layer, p: hit ? hit.value : null, final: false };
  });
}

/** The first layer from which the chosen token is every later layer's top R read. Layers whose R
 *  read is absent are passed over; null when the deepest read layer reads something else. */
export function anatomyLead(layers: readonly AnatomyLayer[], tokenId: number): number | null {
  let lead: number | null = null;
  for (let i = layers.length - 1; i >= 0; i -= 1) {
    const r = layers[i].reads_as.r;
    if (!present(r)) continue;
    if (r.tokens[0].token_id !== tokenId) break;
    lead = layers[i].layer;
  }
  return lead;
}

/** A unit as the data writes it: `neuron/L31n4410` → `L31 n4410`, `head/L28h12` → `L28 h12`. */
export function unitShort(unit: string): string {
  const match = /^(?:neuron|head)\/L(\d+)([nh])(\d+)$/.exec(unit);
  return match ? `L${match[1]} ${match[2]}${match[3]}` : unit;
}

export interface PushRow {
  unit: string;
  short: string;
  kind: "neuron" | "head";
  layer: number;
  contribution: number;
  /** |contribution| over the largest |contribution| in the list. */
  share: number;
  /** A neuron's pole, from its activation's sign (null for a head, or when unknown). */
  pole: "+" | "-" | null;
}

/** The units that pushed the token, as signed rows. The cross-layer list carries no activation,
 *  so a neuron's pole comes from its layer's row (a unit in the cross-layer top k is always in its
 *  layer's top k, both ranked by |contribution|). */
export function pushRows(units: readonly AnatomyUnit[], layers: readonly AnatomyLayer[]): PushRow[] {
  const peak = Math.max(1e-12, ...units.map(u => Math.abs(u.contribution)));
  const byLayer = new Map(layers.map(row => [row.layer, row]));
  return units.map(u => {
    let activation = u.activation;
    if (activation === null && u.kind === "neuron") {
      activation = byLayer.get(u.layer)?.neurons.find(n => n.unit === u.unit)?.activation ?? null;
    }
    return {
      unit: u.unit,
      short: unitShort(u.unit),
      kind: u.kind,
      layer: u.layer,
      contribution: u.contribution,
      share: Math.abs(u.contribution) / peak,
      pole: u.kind === "neuron" && activation !== null && activation !== 0 ? (activation > 0 ? "+" : "-") : null,
    };
  });
}

/** A layer's own top units, neurons and heads together by |contribution|. */
export function layerUnits(layer: AnatomyLayer, n: number): AnatomyUnit[] {
  return [...layer.neurons, ...layer.heads]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, n);
}

/** A contribution with a true minus, to two or three figures. */
export function formatContribution(value: number): string {
  if (!Number.isFinite(value)) return "–";
  const size = Math.abs(value);
  const text = size >= 10 ? size.toFixed(0) : size >= 1 ? size.toFixed(1) : size.toFixed(2);
  if (Number(text) === 0) return text;
  return `${value < 0 ? "−" : "+"}${text}`;
}

/** A token a head attended to, at its strongest: Ledger's 9% (22% marks the token a unit fired on,
 *  14% a token two reads agree on). */
export const HEAT_CAP = 9;

export interface HeatPiece {
  /** The context position, or -1 for an elision. */
  pos: number;
  text: string;
  /** The tint, 0 to `HEAT_CAP` percent: the weight over the row's largest past the sink. */
  tint: number;
  weight: number;
  /** The producing position (the query the head reads from). */
  here: boolean;
}

/** Where a head reads from, compacted to one quiet line: its strongest positions past the sink
 *  (at most `keep`, each at least `floor` of the strongest) with a neighbour either side, the
 *  query position at the end, gaps elided. Position 0 (the sink) only counts toward `sink`. */
export function heatRow(
  weights: readonly number[],
  context: readonly string[],
  { keep = 5, floor = 0.2 }: { keep?: number; floor?: number } = {},
): { pieces: HeatPiece[]; sink: number } {
  const n = Math.min(weights.length, context.length);
  if (n === 0) return { pieces: [], sink: 0 };
  const query = n - 1;
  const sink = n > 1 ? weights[0] : 0;
  const first = n > 1 ? 1 : 0;
  let peak = 0;
  for (let j = first; j < n; j += 1) peak = Math.max(peak, weights[j]);
  const strong = Array.from({ length: n - first }, (_, i) => i + first)
    .filter(j => peak > 0 && weights[j] >= floor * peak)
    .sort((a, b) => weights[b] - weights[a])
    .slice(0, keep);
  const kept = new Set<number>([query]);
  if (query - 1 >= first) kept.add(query - 1);
  for (const j of strong) {
    for (const k of [j - 1, j, j + 1]) if (k >= first && k <= query) kept.add(k);
  }
  const order = [...kept].sort((a, b) => a - b);
  const pieces: HeatPiece[] = [];
  let prev = first - 1;
  for (const j of order) {
    if (j > prev + 1) pieces.push({ pos: -1, text: "…", tint: 0, weight: 0, here: false });
    const share = peak > 0 ? weights[j] / peak : 0;
    pieces.push({ pos: j, text: context[j], tint: Math.round(HEAT_CAP * share * 10) / 10, weight: weights[j], here: j === query });
    prev = j;
  }
  return { pieces, sink };
}

/** The pattern rows in the order the server ranked them (by |contribution|). */
export function patternRows(
  patterns: readonly AnatomyPattern[] | null,
  context: readonly string[] | null,
): { unit: string; short: string; layer: number; contribution: number; pieces: HeatPiece[]; sink: number }[] {
  if (!patterns || !context) return [];
  return patterns.map(p => ({ unit: p.unit, short: unitShort(p.unit), layer: p.layer, contribution: p.contribution, ...heatRow(p.weights, context) }));
}

/** A unit's label as a row shows it: what it writes, else what it fires on or attends to, else
 *  the note; the tip holds every part. */
export function labelText(label: UnitLabelJSON | null | undefined, kind: "neuron" | "head"): { text: string; tip: string } | null {
  if (!label) return null;
  const text = label.writes ?? (kind === "head" ? label.attends : label.fires) ?? label.fires ?? label.attends ?? label.note ?? null;
  if (!text) return null;
  const parts = [
    label.fires ? `fires on ${label.fires}` : "",
    label.attends ? `attends to ${label.attends}` : "",
    label.writes ? `writes ${label.writes}` : "",
    label.note ?? "",
  ].filter(Boolean);
  return { text, tip: parts.join(" · ") };
}
