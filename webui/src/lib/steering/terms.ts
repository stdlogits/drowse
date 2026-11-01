// What a racked term reads as: its name, where it comes from, its strength as a chip writes it, its
// trigger in words, and the strength dial it gets. Pure functions over the entry and the catalog
// row, so the node suites load them; the chips and the term editor draw from them.

import type { GateOp, ManifoldInfo, ProbeInfo, SteerEntry, SteerGate, SteerTiming, Trigger } from "../types";
import { formatCoeff } from "../expression";
import { parseUnit, unitTitle } from "../units";
import { UNIT_PUSH_ALPHA, unitTermProblem } from "../unitTerms";

/** The When choices: the preset windows and ``TOKENS``, a counted window (``FIRST_N`` or
 *  ``AFTER_N``, its count on the term). */
export type WhenChoice = Exclude<Trigger, "FIRST_N" | "AFTER_N"> | "TOKENS";

/** The When choices, in phase order, with the grammar's exact window as each one's tip. The
 *  thinking pair shows only on a model that thinks (or on a term already set to one); Always and
 *  Prompt leave a gated term's choices (a gate never reads the prompt) unless it is set to one. */
export const WHEN: readonly { value: WhenChoice; label: string; hint: string; thinking?: true; prompt?: true }[] = [
  { value: "BOTH", label: "Always", hint: "The prompt and every generated token", prompt: true },
  { value: "BEFORE", label: "Prompt", hint: "The prompt only; generated tokens are left alone", prompt: true },
  { value: "RESPONSE", label: "Response", hint: "Every generated token, thinking included, not the prompt" },
  { value: "THINKING", label: "Thinking", hint: "Thinking tokens only", thinking: true },
  { value: "AFTER", label: "After thinking", hint: "The reply once thinking has closed", thinking: true },
  { value: "TOKENS", label: "Tokens", hint: "The first so many generated tokens, or every one after them (thinking counts), not the prompt" },
];

/** The When choice a trigger shows as. */
export function whenChoice(trigger: Trigger): WhenChoice {
  return trigger === "FIRST_N" || trigger === "AFTER_N" ? "TOKENS" : trigger;
}

/** A fresh counted window's count. */
export const DEFAULT_WINDOW_TOKENS = 8;

/** The gate comparisons, as the editor offers and the chip says them. */
export const GATE_OPS: readonly { value: GateOp; label: string; hint: string }[] = [
  { value: ">", label: ">", hint: "Above" },
  { value: ">=", label: "≥", hint: "At or above" },
  { value: "<", label: "<", hint: "Below" },
  { value: "<=", label: "≤", hint: "At or below" },
];

/** The scalar key a gate on an attached probe reads, the number its card shows: a curved
 *  geometry probe's share (``name:fraction``; a dotted name takes only the axis forms, so it reads
 *  its first axis), every other probe's main reading (its bare name: a flat probe's first axis, a
 *  lens, SAE or unit probe's value). */
export function gateProbeKey(name: string, info: ProbeInfo): string {
  const curved = info.family === "geometry" && !info.is_affine;
  return curved && !bareName(name).includes(".") ? `${name}:fraction` : name;
}

/** A gate as words: ``while honest.deceptive > 1`` (its probe's last segment, its comparison as a
 *  sign, its threshold as the expression writes it, with a true minus). */
export function gateWord(gate: SteerGate): string {
  const op = GATE_OPS.find(row => row.value === gate.op)?.label ?? gate.op;
  return `while ${bareName(gate.probe)} ${op} ${formatCoeff(gate.threshold).replace(/^-/u, "−")}`;
}

/** A term's timing as a chip's meta says it (nothing for the default): its window, then its gate. */
export function triggerWord({ trigger, tokens, gate }: SteerTiming): string {
  const window = (() => {
    switch (trigger) {
      case "BEFORE": return "prompt";
      case "RESPONSE": return gate ? "" : "response";
      case "THINKING": return "thinking";
      case "AFTER": return "after thinking";
      case "FIRST_N": return `first ${tokens ?? 0} tokens`;
      case "AFTER_N": return `after ${tokens ?? 0} tokens`;
      default: return "";
    }
  })();
  return [window, gate ? gateWord(gate) : ""].filter(Boolean).join(" ");
}

/** A coefficient as text: at most two decimals, trailing zeros dropped, a true minus sign. */
export function coefficientText(value: number, { fixed = false }: { fixed?: boolean } = {}): string {
  if (!Number.isFinite(value)) return "0";
  const rounded = Math.round(value * 100) / 100 || 0;
  const body = fixed ? Math.abs(rounded).toFixed(2) : String(Math.abs(rounded));
  return rounded < 0 ? `−${body}` : body;
}

/** A unit's layer, from its key (``neuron/L31n4410`` → 31). */
export function unitLayer(name: string): number | null {
  return parseUnit(name)?.layer ?? null;
}

/** The selector's last segment: ``default/emotions`` → ``emotions``. */
export function bareName(name: string): string {
  const selector = name.split("%")[0];
  return selector.split("/").at(-1) ?? selector;
}

/** A projection's relation in words: ``along calm.anxious`` (``~``, only the part along it) or
 *  ``not along calm.anxious`` (``|``, all but that part). */
export function relationWord(operator: "~" | "|", onto: string): string {
  return `${operator === "~" ? "along" : "not along"} ${bareName(onto)}`;
}

/** Whether a term removes what it names rather than pushing toward it. */
export function removes(name: string, entry: SteerEntry): boolean {
  if (entry.mode === "unit") return entry.ablate;
  return (entry.mode === "subspace" || entry.mode === "jlens" || entry.mode === "slens" || entry.mode === "sae") && entry.ablate;
}

/** The term's name as its chip and its editor show it. A manifold position off its nodes names
 *  its coordinates by their axes (``month 6.00``, with the manifold's ``info``), else as a point,
 *  so a bare number never reads as a quantity of the manifold's name ("6 months"). */
export function termTitle(name: string, entry: SteerEntry, info?: ManifoldInfo | null, tokens?: ReadonlyMap<number, string>): string {
  switch (entry.mode) {
    case "subspace": return entry.ablate ? bareName(name) : entry.label ?? bareName(name);
    case "projection": return bareName(entry.base);
    case "manifold": {
      if (entry.label) return entry.label;
      const axes = info?.domain.type === "box" ? info.domain.axes : [];
      return entry.coords.length === 1 && axes.length === 1
        ? `${axes[0].name} ${coefficientText(entry.coords[0], { fixed: true })}`
        : `point ${entry.coords.map(c => coefficientText(c, { fixed: true })).join(", ")}`;
    }
    case "jlens":
    case "slens": {
      const word = name.slice(`${entry.mode}/`.length);
      const id = lensTokenId(name);
      return (id !== null ? tokens?.get(id)?.trim() : "") || word;
    }
    case "sae": return `feature ${name.slice("sae/".length)}`;
    case "unit": return unitTitle(name);
  }
}

/** A lens atom's vocabulary id when it names one (``jlens/#282``), else null. */
export function lensTokenId(name: string): number | null {
  const match = /^[js]lens\/#([0-9]+)$/u.exec(name);
  return match ? Number(match[1]) : null;
}

/** Where the term comes from, in a word or two: the manifold it sits on, or the kind of atom. */
export function termSource(name: string, entry: SteerEntry): string {
  switch (entry.mode) {
    case "subspace": return !entry.ablate && entry.label !== null && entry.label !== bareName(name) ? bareName(name) : "";
    case "projection": return relationWord(entry.operator, entry.onto);
    case "manifold": return bareName(name);
    case "jlens": return lensTokenId(name) === null ? "lens token, as said" : `lens token #${lensTokenId(name)}, as said`;
    case "slens": return lensTokenId(name) === null ? "lens token, as read" : `lens token #${lensTokenId(name)}, as read`;
    case "sae": return "SAE feature";
    case "unit": return name.startsWith("head/") ? "attention head" : "MLP neuron";
  }
}

/** What the term is, for the editor's facts: ``a 20-node manifold``, ``a concept``. */
export function termKind(entry: SteerEntry, info: ManifoldInfo | null): string {
  if (entry.mode === "projection") return entry.operator === "~" ? "the part along another concept" : "all but the part along another concept";
  if (entry.mode === "subspace" || entry.mode === "manifold") {
    if (!info) return "";
    const nodes = info.node_count;
    if (entry.mode === "subspace" && nodes <= 2) return "a concept";
    return `a ${nodes}-node ${entry.mode === "subspace" ? "flat fit" : "manifold"}`;
  }
  return "";
}

/** The strength a chip shows: for a removal, how much of the component it takes (``without`` for
 *  all of it, ``without 50%``, ``without 150%``) or, below zero, adds (``+50%``); else the
 *  coefficient. */
export function chipStrength(name: string, entry: SteerEntry, subspaceAlong: number): string {
  if (removes(name, entry)) {
    const value = entry.mode === "subspace" ? subspaceAlong : "alpha" in entry ? entry.alpha : 0;
    const percent = Math.round(Math.abs(value) * 100);
    if (percent === 0) return "off";
    if (value < 0) return `+${percent}%`;
    return percent === 100 ? "without" : `without ${percent}%`;
  }
  switch (entry.mode) {
    case "subspace": return coefficientText(subspaceAlong);
    case "manifold": return coefficientText(entry.blend);
    default: return coefficientText(entry.alpha);
  }
}

/** A strength dial: its range, its step, and whether every flat term shares it. */
export interface Dial {
  value: number;
  min: number;
  max: number;
  step: number;
  shared: boolean;
}

/** The dial a term's strength takes. Every dial is signed but a curved term's. Flat terms share
 *  one slide (the composer merges them into one subspace), from −2 to 2, whether a term pushes
 *  (below zero, away from its position) or removes (below zero, it adds to the component); a
 *  projection is a plain push of its own, over the same range; a curved term slides toward its
 *  position, from 0 to 1 (a curved position has no opposite, so composition refuses a negative
 *  along); an atom pushes either way, from −1 to 1, and its removal runs from −1 (twice the
 *  component) to 1 (none of it); a unit's removal runs from none of it to all of it (the one unit
 *  rule refuses a negative one). */
export function strengthDial(name: string, entry: SteerEntry, subspaceAlong: number): Dial {
  if (entry.mode === "subspace") return { value: subspaceAlong, min: -2, max: 2, step: 0.05, shared: true };
  if (entry.mode === "projection") return { value: entry.alpha, min: -2, max: 2, step: 0.05, shared: false };
  if (entry.mode === "manifold") return { value: entry.blend, min: 0, max: 1, step: 0.05, shared: false };
  if (entry.mode === "unit" && entry.ablate) return { value: entry.alpha, min: 0, max: 1, step: 0.05, shared: false };
  return { value: entry.alpha, min: -1, max: 1, step: 0.05, shared: false };
}

/** What the strength does, in a few words. A removal says how much of the component it takes: all
 *  of it at 1 (down to the neutral mean), and on the shared flat dial past 1, more than all of it;
 *  below zero it adds to the component instead (``50% more of it``). */
export function strengthSays(name: string, entry: SteerEntry, value: number, info?: ManifoldInfo | null): string {
  const title = termTitle(name, entry, info);
  // A projection of a two-node concept pushes toward one pole or the other.
  if (entry.mode === "projection" && info?.node_labels.length === 2) {
    if (value === 0) return "Off";
    return `Toward ${info.node_labels[value > 0 ? 0 : 1]}`;
  }
  if (removes(name, entry)) {
    if (value === 0) return "Off";
    const percent = Math.round(Math.abs(value) * 100);
    if (value < 0) return `${percent}% more of it`;
    return percent === 100 ? "All of it" : `${percent}% of it`;
  }
  if (value === 0) return "Off";
  return value > 0 ? `Toward ${title}` : `Away from ${title}`;
}

/** Where the expression line may wrap: after a namespace's ``/`` and a list's comma, and before a
 *  ``%`` position, a ``{…}`` clause and an ``@`` trigger, so a break never splits a name or orphans
 *  a brace. The pieces join back to ``text``. */
export function wrapPieces(text: string): string[] {
  return text.split(/(?<=[/,])|(?=[%{@])/u).filter(Boolean);
}

/** Whether the term can be pushed or removed (a rank-one direction): a two-node concept, a lens
 *  token, an SAE feature, a neuron. A curved term and a head cannot be both. */
export function canRemove(name: string, entry: SteerEntry, info: ManifoldInfo | null): boolean {
  if (entry.mode === "jlens" || entry.mode === "slens" || entry.mode === "sae") return true;
  // A unit switches both ways when the one rule lets it push (a neuron, not a head).
  if (entry.mode === "unit") return unitTermProblem(name, { ablate: false, alpha: UNIT_PUSH_ALPHA }) === null;
  if (entry.mode === "subspace") return info?.node_count === 1 || info?.node_count === 2;
  return false;
}
