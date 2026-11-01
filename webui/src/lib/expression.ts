// Serialization of the unified steering-expression rack.
//
// Mirrors drowse/steering/grammar.py (serialize direction only):
//
//   expr     := term (("+" | "-") term)*
//   term     := [coeff "*"?] ["!"] selector ["{" layers "}"] ["@" trigger]
//   layers   := indices and a-b ranges ("4-8,20") | "workspace"   (lib/steering/layers.ts)
//   selector := atom (("~" | "|") atom | "%" coord_list)?
//   coord_list := signed_num ("," signed_num)*
//   atom     := [ns "/"] NAME ["." NAME] [":" variant]
//             | ("neuron" | "head") "/" UNIT                     (no clause: a unit has its layer)
//   trigger  := before|after|both|thinking|response|prompt|generated
//   variant  := raw | sae | sae-<release> | role | role-<name>
//             | from | from-<safe-source-model>
//
// A ``%`` selector is the manifold operator — ``<manifold> % a,b,...``
// places generation at a point of a fitted steering manifold.  The
// coefficient is the blend fraction.
//
// One unified steer rack is the source of truth — a ``Map<string,
// SteerEntry>`` where each entry is ``mode: "subspace"`` (flat: coords,
// label, variant, trigger, enabled — magnitude is the rack-level
// ``subspaceAlong`` master) or ``mode: "manifold"`` (curved: blend, onto,
// coords, label, variant, trigger, enabled).  Both serialize as ``%`` terms;
// the magnitude slot is the shared ``subspaceAlong`` for subspace terms and
// the per-card ``along[,onto]`` for manifold terms.

import type {
  JLensSteerEntry,
  SLensSteerEntry,
  SaeSteerEntry,
  ManifoldSteerEntry,
  SteerEntry,
  SubspaceSteerEntry,
  Trigger,
  UnitSteerEntry,
  Variant,
} from "./types";
import { layerClause, termIsInert } from "./steering/layers";

// ----------------------------------------------- triggers ----------------

/** Map UI Trigger values to the keyword the formatter emits.  BOTH is the
 * default and is omitted from the output. */
const TRIGGER_TO_KEYWORD: Record<Trigger, string | null> = {
  BOTH: null,
  BEFORE: "before",
  AFTER: "after",
  THINKING: "thinking",
  RESPONSE: "response",
};

// ----------------------------------------------- formatter ---------------

/** Render the rack as a canonical expression string.  Disabled entries, and
 * entries scoped to no layer, are skipped; subspace (flat) terms emit first,
 * then lens and SAE atoms, units, and manifold (curved) terms.
 * Every term is a ``%`` position; the coefficient slot is the shared
 * ``subspaceAlong`` master for subspace terms and the per-card
 * ``along[,onto]`` for manifold terms.  Empty/all-disabled racks return "". */
export function serializeExpression(
  rack: Map<string, SteerEntry>,
  subspaceAlong = 1,
): string {
  const parts: string[] = [];
  for (const mode of TERM_ORDER) {
    for (const [name, entry] of rack) {
      if (entry.mode !== mode || !entry.enabled || termIsInert(entry.mode === "unit" ? null : entry.layers)) continue;
      parts.push(formatTerm(name, entry, subspaceAlong));
    }
  }
  return joinTerms(parts);
}

/** The order families emit in: flat terms first (they share one slide), then the atoms, units
 *  and curved terms. */
const TERM_ORDER: readonly SteerEntry["mode"][] = ["subspace", "jlens", "slens", "sae", "unit", "manifold"];

/** Render one rack entry as its own term (whatever ``enabled`` says). */
export function formatTerm(name: string, entry: SteerEntry, subspaceAlong: number): string {
  switch (entry.mode) {
    case "subspace": return formatSubspaceTerm(name, entry, subspaceAlong);
    case "jlens":
    case "slens": return formatJLensTerm(name, entry);
    case "sae": return formatSaeTerm(name, entry);
    case "unit": return formatUnitTerm(name, entry);
    case "manifold": return formatManifoldTerm(name, entry);
  }
}

/** Join rendered terms: the first keeps its leading sign, the rest join with `` + `` or `` - ``. */
function joinTerms(parts: readonly string[]): string {
  if (parts.length === 0) return "";
  // First term keeps its leading sign verbatim; subsequent terms join
  // with " + " or " - " depending on coefficient sign.
  let out = parts[0];
  for (let i = 1; i < parts.length; i++) {
    const p = parts[i];
    if (p.startsWith("-")) out += " - " + p.slice(1).trimStart();
    else out += " + " + p;
  }
  return out;
}

/** Render one subspace (flat) term — ``<subspaceAlong> name[:variant]%<pos>``
 *  plus an optional ``@trigger``.  All subspace terms share the one
 *  ``subspaceAlong`` magnitude (the merged affine subspace slides once);
 *  relative weight between them lives in how far each position sits from
 *  neutral.  ``<pos>`` is the label form (``personas%hacker``) when
 *  ``entry.label`` is set, else the comma-joined coord list. */
export function formatSubspaceTerm(
  name: string,
  entry: SubspaceSteerEntry,
  subspaceAlong: number,
): string {
  const clause = layerClause(entry.layers);
  if (entry.ablate) {
    return `${formatBang(`${nameWithVariant(name, entry.variant)}${clause}`, subspaceAlong)}${formatTriggerSuffix(entry.trigger)}`;
  }
  const position = entry.label ?? entry.coords.map((c) => formatCoeff(c)).join(",");
  const selector = `${nameWithVariant(name, entry.variant)}%${position}`;
  return `${formatCoeff(subspaceAlong)} ${selector}${clause}${formatTriggerSuffix(entry.trigger)}`;
}

function nameWithVariant(name: string, variant: Variant): string {
  return variant === "raw" ? name : `${name}:${variant}`;
}

/** A number as the server's ``format_expr`` writes it: Python's ``%g`` (six significant digits,
 *  trailing zeros dropped, an exponent below 1e-4 or from 1e6 on: ``1e-05``, ``1.5e+06``), so the
 *  chips' expression is the canonical one. (An exact binary tie at the seventh digit, an integer
 *  such as 1234565, rounds up here and to even in Python; no dial reaches one.) The leading sign
 *  rides the term; the joiner above splits +/-. */
export function formatCoeff(coeff: number): string {
  if (Number.isNaN(coeff) || !Number.isFinite(coeff)) return "0";
  if (coeff === 0) return Object.is(coeff, -0) ? "-0" : "0";
  const [mantissa, power] = coeff.toExponential(5).split("e");
  const exponent = Number(power);
  const trim = (digits: string) => digits.includes(".") ? digits.replace(/0+$/u, "").replace(/\.$/u, "") : digits;
  if (exponent < -4 || exponent >= 6) {
    return `${trim(mantissa)}e${exponent < 0 ? "-" : "+"}${String(Math.abs(exponent)).padStart(2, "0")}`;
  }
  return trim(coeff.toFixed(5 - exponent));
}

/** An ablation as the server writes it: ``!x`` at full strength, ``-!x`` at −1, ``0.5 !x``
 *  otherwise. ``target`` carries the clause (``jlens/storm{20-30}``); the trigger follows. */
function formatBang(target: string, coeff: number): string {
  if (coeff === 1) return `!${target}`;
  if (coeff === -1) return `-!${target}`;
  return `${formatCoeff(coeff)} !${target}`;
}

function formatTriggerSuffix(trigger: Trigger): string {
  const kw = TRIGGER_TO_KEYWORD[trigger];
  return kw ? `@${kw}` : "";
}

/** Render one lens token term — the plain atom ``<alpha> jlens/<word>``
 *  (or ``slens/<word>``) plus its clause and an optional ``@trigger``.  The
 *  rack key is the full atom (the engine resolves it through the lens's
 *  ``register_direction`` / ``register_input_direction``); per-chip
 *  ``alpha`` because lens atoms run hotter than concept vectors. */
export function formatJLensTerm(name: string, entry: JLensSteerEntry | SLensSteerEntry): string {
  return formatAtomTerm(name, entry);
}

export function formatSaeTerm(name: string, entry: SaeSteerEntry): string {
  return formatAtomTerm(name, entry);
}

/** One single-direction atom: ``<alpha> <atom>{clause}`` pushed, or its ablation. */
function formatAtomTerm(name: string, entry: JLensSteerEntry | SLensSteerEntry | SaeSteerEntry): string {
  const target = `${name}${layerClause(entry.layers)}`;
  return `${entry.ablate ? formatBang(target, entry.alpha) : `${formatCoeff(entry.alpha)} ${target}`}${formatTriggerSuffix(entry.trigger)}`;
}

/** Render one unit term — ``<alpha> neuron/L31n4410`` (a push; a negative α takes the other pole)
 *  or ``!neuron/…`` / ``!head/…`` (mean-ablated: α 1, written bare like every full ablation;
 *  ``0.5 !head/…`` blends halfway).  A unit carries its own layer, so it never writes a clause
 *  (the grammar refuses one); which terms a unit takes is lib/unitTerms.ts's rule, held by the
 *  rack. */
export function formatUnitTerm(name: string, entry: UnitSteerEntry): string {
  return `${entry.ablate ? formatBang(name, entry.alpha) : `${formatCoeff(entry.alpha)} ${name}`}${formatTriggerSuffix(entry.trigger)}`;
}

/** Render one manifold (curved) rack entry — ``<along[,onto]>
 *  <name>[:variant]%<position>`` plus an optional ``@trigger`` suffix.
 *  Coefficient is the per-card blend fraction (``-`` rides the joiner).
 *  ``<position>`` is the label form (``emotions%happy``) when ``entry.label`` is
 *  set; otherwise the comma-joined coord list (``emotions%0.3,0.8``). */
export function formatManifoldTerm(name: string, entry: ManifoldSteerEntry): string {
  const position = entry.label
    ? entry.label
    : entry.coords.map((c) => formatCoeff(c)).join(",");
  const selector = `${nameWithVariant(name, entry.variant)}%${position}`;
  // Coefficient slot: ``along`` alone, or ``along,onto`` when onto is set — the curved-manifold
  // residual-collapse fraction. The term's sign applies to the whole run, so ``onto`` is written
  // relative to ``along``'s sign, as format_expr writes it.
  const onto = entry.onto ?? 0;
  const coeff =
    onto !== 0
      ? `${formatCoeff(entry.blend)},${formatCoeff(entry.blend < 0 ? -onto : onto)}`
      : formatCoeff(entry.blend);
  return `${coeff} ${selector}${layerClause(entry.layers)}${formatTriggerSuffix(entry.trigger)}`;
}
