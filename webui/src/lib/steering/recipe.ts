// A recipe from elsewhere (an agent's expression, a tree's, the command line's) as chips. The
// server's parser is the only parser: its validation answers each term's fields in order
// (``parts``), and this maps them onto rack entries. A recipe becomes chips only when every term
// does, exactly: each chip must write its term's own text back (``formatTerm`` against the
// server's ``text``) and sit inside its dial; otherwise the recipe stays one read-only chip, with
// the reason. Store-free, so the node suites load it.

import type { ManifoldInfo, SteerEntry, SteerTiming, SteeringPartJSON, Trigger, Variant } from "../types";
import { formatTerm } from "../expression";
import { fitGeometry } from "../manifolds/fitKind";
import { unitTermProblem } from "../unitTerms";
import { canRemove, strengthDial } from "./terms";

/** The rack a recipe becomes: its entries and the shared flat strength. */
export interface RecipeChips {
  entries: Map<string, SteerEntry>;
  subspaceAlong: number;
}

export type RecipeAsChips = { chips: RecipeChips } | { problem: string };

const PHASES: Record<string, Trigger> = {
  both: "BOTH",
  before: "BEFORE",
  response: "RESPONSE",
  thinking: "THINKING",
  after: "AFTER",
};

/** A term's timing from its fields (the server's: ``phase``, ``first_n``/``after_n``, ``gate``),
 *  or why a chip cannot hold it. ``tokens`` and ``gate`` are set only when the term has them. */
function timingOf(part: SteeringPartJSON): SteerTiming | string {
  const gate = part.gate === null ? {} : { gate: { probe: part.gate.probe, op: part.gate.op, threshold: part.gate.threshold } };
  if (part.first_n !== null) return { trigger: "FIRST_N", tokens: part.first_n, ...gate };
  if (part.after_n !== null) return { trigger: "AFTER_N", tokens: part.after_n, ...gate };
  const trigger = part.phase === null ? undefined : PHASES[part.phase];
  if (trigger === undefined) return `it runs @${part.trigger}, which a chip cannot hold`;
  return { trigger, ...gate };
}

const ATOMS = { "jlens/": "jlens", "slens/": "slens", "sae/": "sae" } as const;

function atomMode(target: string): "jlens" | "slens" | "sae" | null {
  for (const [prefix, mode] of Object.entries(ATOMS)) if (target.startsWith(prefix)) return mode;
  return null;
}

function isUnit(target: string): boolean {
  return target.startsWith("neuron/") || target.startsWith("head/");
}

/** A catalog row by the name a term writes (``ns/name`` or a bare name). */
function catalogRow(catalog: readonly ManifoldInfo[], name: string): ManifoldInfo | null {
  return catalog.find((m) => `${m.namespace}/${m.name}` === name)
    ?? catalog.find((m) => m.name === name)
    ?? null;
}

/** ``name:variant`` split (reserved namespaces take none, so only concept names reach here). */
function splitVariant(target: string): [string, Variant] {
  const at = target.indexOf(":");
  return at < 0 ? [target, "raw"] : [target.slice(0, at), target.slice(at + 1) as Variant];
}

/** One term as a rack entry, or why it cannot be one. */
function entryOf(part: SteeringPartJSON, catalog: readonly ManifoldInfo[]): { name: string; entry: SteerEntry } | string {
  const timing = timingOf(part);
  if (typeof timing === "string") return timing;
  const scope = part.layers === null ? {} : { layers: part.layers };
  const common = { ...timing, enabled: true };
  if (part.op === "project") {
    if (part.projection === null) return `${part.target} is a projection without its parts`;
    const { base, operator, onto } = part.projection;
    return { name: part.target, entry: { mode: "projection", base, operator, onto, alpha: part.coeff, ...scope, ...common } };
  }
  const atom = atomMode(part.target);
  if (atom !== null && (part.op === "push" || part.op === "ablate")) {
    return { name: part.target, entry: { mode: atom, ablate: part.op === "ablate", alpha: part.coeff, ...scope, ...common } as SteerEntry };
  }
  if (isUnit(part.target) && (part.op === "push" || part.op === "ablate")) {
    const entry: SteerEntry = { mode: "unit", ablate: part.op === "ablate", alpha: part.coeff, ...common };
    return unitTermProblem(part.target, entry) ?? { name: part.target, entry };
  }
  if (part.op === "push") return `${part.target} is a plain vector, which a chip cannot hold`;
  const [name, variant] = splitVariant(part.target);
  const info = catalogRow(catalog, name);
  if (!info) return `${name} is not among the installed manifolds`;
  const geometry = fitGeometry(info);
  if (part.op === "ablate") {
    const entry: SteerEntry = { mode: "subspace", ablate: true, coords: [], label: null, variant, ...scope, ...common };
    return geometry === "flat" && canRemove(name, entry, info) ? { name, entry } : `${name} cannot be removed as a chip`;
  }
  // A position: its label (with the node's coordinates, as the picker sets them) or coordinates.
  const label = typeof part.position === "string" ? part.position : null;
  const index = label === null ? -1 : info.node_labels.indexOf(label);
  const coords = Array.isArray(part.position)
    ? [...part.position]
    : index >= 0 && info.node_coords[index] ? [...info.node_coords[index]] : [];
  if (geometry === "curved") {
    return { name, entry: { mode: "manifold", blend: part.coeff, onto: part.onto, coords, label, variant, ...scope, ...common } };
  }
  if (geometry !== "flat") return `${name} is not fitted for this model`;
  if (part.onto !== 0) return `${name} is flat, so it has no onto`;
  return { name, entry: { mode: "subspace", ablate: false, coords, label, variant, ...scope, ...common } };
}

/** The recipe ``parts`` as chips, or why it stays one read-only chip. ``along`` is the shared flat
 *  strength when the recipe has no flat term. */
export function recipeAsChips(parts: readonly SteeringPartJSON[], catalog: readonly ManifoldInfo[], along: number): RecipeAsChips {
  const built: { name: string; entry: SteerEntry; part: SteeringPartJSON }[] = [];
  for (const part of parts) {
    const made = entryOf(part, catalog);
    if (typeof made === "string") return { problem: made };
    if (built.some((row) => row.name === made.name)) return { problem: `${made.name} appears twice, which chips cannot hold` };
    built.push({ ...made, part });
  }
  // Every flat term slides by one shared strength (the composer merges them into one subspace).
  const flat = new Set(built.filter((row) => row.entry.mode === "subspace").map((row) => row.part.coeff));
  if (flat.size > 1) return { problem: "its flat terms have different strengths, and chips share one" };
  const subspaceAlong = flat.size === 1 ? [...flat][0] : along;
  for (const { name, entry, part } of built) {
    // Each chip writes its term back exactly, and its strength sits on its dial.
    if (formatTerm(name, entry, subspaceAlong) !== part.text) return { problem: `${part.text} does not come back the same from a chip` };
    const dial = strengthDial(name, entry, subspaceAlong);
    if (dial.value < dial.min || dial.value > dial.max) return { problem: `${part.text} is outside its chip's range` };
  }
  return { chips: { entries: new Map(built.map((row) => [row.name, row.entry])), subspaceAlong } };
}
