// A unit's terms in the recipe: a neuron pushed along its write (either pole) or a neuron or head
// mean-ablated. Import-free, so the snapshot validator (lib/conversationSnapshot.ts, which loads
// outside the Svelte build too) shares the one rule with WebMCP, the rack's mutators
// (stores/steering.svelte.ts: setUnitTerm, patchSteerEntry) and the term editor. The term's text is
// lib/expression.ts::formatUnitTerm's, like every other term's.

export type UnitTerm = "plus" | "minus" | "silence";

/** The default push strength. A neuron pushes at its one layer, where a lens atom or a concept
 *  pushes at each of its layers with the same per-layer budget (the composer's mean-1 shares), so a
 *  neuron needs a larger α to show. On SmolLM2-360M (temperature 0) a top writer's word first
 *  appears near 3 from layers 26–27, while the last layer's writers already loop by 2 and early
 *  layers' only shift the theme even at 8. */
export const UNIT_PUSH_ALPHA = 3;

/** The push dial's reach either way. */
export const UNIT_PUSH_MAX = 8;

/** An ablation's strength: exact mean-ablation, the unit held at its neutral mean. */
export const UNIT_ABLATE_ALPHA = 1;

/** What is wrong with a unit term (a rack row keyed by its reference), or null. A neuron is pushed
 *  (signed α, the − pole's below zero) or ablated; a head is only ablated. An ablation blends the
 *  unit toward its neutral mean (``x ← (1 − α)·x + α·mean``), so a negative α would amplify the
 *  unit's deviation from it instead: the snapshot validator and WebMCP refuse one. */
export function unitTermProblem(name: string, entry: { ablate: unknown; alpha: unknown }): string | null {
  if (typeof entry.alpha !== "number" || !Number.isFinite(entry.alpha)) return `${name} needs a finite alpha`;
  if (typeof entry.ablate !== "boolean") return `${name} needs ablate true or false`;
  const neuron = /^neuron\/L(?:0|[1-9]\d*)n(?:0|[1-9]\d*)$/u.test(name);
  const head = /^head\/L(?:0|[1-9]\d*)h(?:0|[1-9]\d*)$/u.test(name);
  if (!neuron && !head) return `${name} names no unit`;
  if (head && !entry.ablate) return `${name}: a head has no push, only ablation`;
  if (entry.ablate && entry.alpha < 0) return `${name}: an ablation's alpha must not be negative (1 holds the unit at its neutral mean)`;
  return null;
}
