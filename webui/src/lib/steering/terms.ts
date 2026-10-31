// What a racked term reads as: its name, where it comes from, its strength as a chip writes it, its
// trigger in words, and the strength dial it gets. Pure functions over the entry and the catalog
// row, so the node suites load them; the chips and the term editor draw from them.

import type { ManifoldInfo, SteerEntry, Trigger } from "../types";

/** The trigger choices, in phase order, with the grammar's exact window as each one's tip. The
 *  thinking pair shows only on a model that thinks (or on a term already set to one). */
export const WHEN: readonly { value: Trigger; label: string; hint: string; thinking?: true }[] = [
  { value: "BOTH", label: "Always", hint: "The prompt and every generated token" },
  { value: "BEFORE", label: "Prompt", hint: "The prompt only; generated tokens are left alone" },
  { value: "RESPONSE", label: "Response", hint: "Every generated token, thinking included, not the prompt" },
  { value: "THINKING", label: "Thinking", hint: "Thinking tokens only", thinking: true },
  { value: "AFTER", label: "After thinking", hint: "The reply once thinking has closed", thinking: true },
];

/** A trigger as a chip's meta says it (nothing for the default). */
export function triggerWord(trigger: Trigger): string {
  switch (trigger) {
    case "BEFORE": return "prompt";
    case "RESPONSE": return "response";
    case "THINKING": return "thinking";
    case "AFTER": return "after thinking";
    default: return "";
  }
}

/** A coefficient as text: at most two decimals, trailing zeros dropped, a true minus sign. */
export function coefficientText(value: number, { fixed = false }: { fixed?: boolean } = {}): string {
  if (!Number.isFinite(value)) return "0";
  const rounded = Math.round(value * 100) / 100 || 0;
  const body = fixed ? Math.abs(rounded).toFixed(2) : String(Math.abs(rounded));
  return rounded < 0 ? `−${body}` : body;
}

/** A unit's ID as the data writes it: ``neuron/L31n4410`` → ``L31 n4410``. */
export function unitLabel(name: string): string {
  const match = /^(?:neuron|head)\/L(\d+)([nh])(\d+)$/u.exec(name);
  return match ? `L${match[1]} ${match[2]}${match[3]}` : name;
}

/** A unit's layer, from its key. */
export function unitLayer(name: string): number | null {
  const match = /^(?:neuron|head)\/L(\d+)[nh]\d+$/u.exec(name);
  return match ? Number(match[1]) : null;
}

/** The selector's last segment: ``default/emotions`` → ``emotions``. */
export function bareName(name: string): string {
  const selector = name.split("%")[0];
  return selector.split("/").at(-1) ?? selector;
}

/** Whether a term removes what it names rather than pushing toward it. */
export function removes(name: string, entry: SteerEntry): boolean {
  if (entry.mode === "unit") return entry.ablate || name.startsWith("head/");
  return (entry.mode === "subspace" || entry.mode === "jlens" || entry.mode === "sae") && entry.ablate;
}

/** The term's name as its chip and its editor show it. */
export function termTitle(name: string, entry: SteerEntry): string {
  switch (entry.mode) {
    case "subspace": return entry.ablate ? bareName(name) : entry.label ?? bareName(name);
    case "manifold": return entry.label ?? entry.coords.map(c => coefficientText(c)).join(", ");
    case "jlens": return name.slice("jlens/".length);
    case "sae": return `feature ${name.slice("sae/".length)}`;
    case "unit": return unitLabel(name);
  }
}

/** Where the term comes from, in a word or two: the manifold it sits on, or the kind of atom. */
export function termSource(name: string, entry: SteerEntry): string {
  switch (entry.mode) {
    case "subspace": return !entry.ablate && entry.label !== null && entry.label !== bareName(name) ? bareName(name) : "";
    case "manifold": return bareName(name);
    case "jlens": return "lens token";
    case "sae": return "SAE feature";
    case "unit": return name.startsWith("head/") ? "attention head" : "MLP neuron";
  }
}

/** What the term is, for the editor's facts: ``a 20-node manifold``, ``a concept``. */
export function termKind(entry: SteerEntry, info: ManifoldInfo | null): string {
  if (entry.mode === "subspace" || entry.mode === "manifold") {
    if (!info) return "";
    const nodes = info.node_count;
    if (entry.mode === "subspace" && nodes <= 2) return "a concept";
    return `a ${nodes}-node ${entry.mode === "subspace" ? "flat fit" : "manifold"}`;
  }
  return "";
}

/** The strength a chip shows: ``without`` for a removal, else the coefficient. */
export function chipStrength(name: string, entry: SteerEntry, subspaceAlong: number): string {
  if (removes(name, entry)) return "without";
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

/** The dial a term's strength takes. Flat terms share one slide (the composer merges them into
 *  one subspace), from 0 to 2; a curved term and an atom push either way, from −1 to 1; a removal
 *  runs from none of it to all of it. */
export function strengthDial(name: string, entry: SteerEntry, subspaceAlong: number): Dial {
  if (entry.mode === "subspace") return { value: subspaceAlong, min: 0, max: 2, step: 0.05, shared: true };
  if (entry.mode === "manifold") return { value: entry.blend, min: -1, max: 1, step: 0.05, shared: false };
  if (removes(name, entry)) return { value: entry.alpha, min: 0, max: 1, step: 0.05, shared: false };
  return { value: entry.alpha, min: -1, max: 1, step: 0.05, shared: false };
}

/** What the strength does, in a few words. */
export function strengthSays(name: string, entry: SteerEntry, value: number): string {
  const title = termTitle(name, entry);
  if (removes(name, entry)) {
    if (value <= 0) return "Off";
    return value >= 1 ? "All of it" : `${Math.round(value * 100)}% of it`;
  }
  if (value === 0) return "Off";
  return value > 0 ? `Toward ${title}` : `Away from ${title}`;
}

/** Whether the term can be pushed or removed (a rank-one direction): a two-node concept, a lens
 *  token, an SAE feature, a neuron. A curved term and a head cannot be both. */
export function canRemove(name: string, entry: SteerEntry, info: ManifoldInfo | null): boolean {
  if (entry.mode === "jlens" || entry.mode === "sae") return true;
  if (entry.mode === "unit") return name.startsWith("neuron/");
  if (entry.mode === "subspace") return info?.node_count === 1 || info?.node_count === 2;
  return false;
}
