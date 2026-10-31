// A unit's terms in the recipe: a neuron pushed along its write (either pole) or a neuron or head
// mean-ablated. Import-free, so the snapshot validator (lib/conversationSnapshot.ts, which loads
// outside the Svelte build too) shares the one rule with WebMCP and the rack.

export type UnitTerm = "plus" | "minus" | "silence";

/** The default push strength: a neuron's write runs as hot as a lens atom's direction. */
export const UNIT_PUSH_ALPHA = 0.2;

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

/** The steering expression a unit term writes: ``0.2 neuron/L31n4410``, ``-0.2 neuron/…``,
 *  ``!neuron/…`` (exact mean-ablation). A head has no push. */
export function unitTermExpression(id: string, term: UnitTerm, alpha = UNIT_PUSH_ALPHA): string {
  if (term === "silence") return `!${id}`;
  if (id.startsWith("head/")) throw new Error("a head has no push");
  const a = Math.abs(alpha);
  return `${term === "minus" ? "-" : ""}${a} ${id}`;
}
