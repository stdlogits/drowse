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
//   trigger  := phase ["&" gate] | gate
//   phase    := before|after|both|thinking|response|prompt|generated | (first|after) ":" INT
//   gate     := "when:" probe (">" | ">=" | "<" | "<=") signed_num
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
  ProjectionSteerEntry,
  SteerEntry,
  SteerTiming,
  SubspaceSteerEntry,
  UnitSteerEntry,
  Variant,
} from "./types";
import { layerClause, termIsInert } from "./steering/layers";

// ----------------------------------------------- triggers ----------------

/** The preset windows' canonical keywords (grammar ``_TRIGGER_CANONICAL``). */
const PRESET_KEYWORD = {
  BOTH: "both",
  BEFORE: "before",
  AFTER: "after",
  THINKING: "thinking",
  RESPONSE: "response",
} as const;

/** A term's ``@`` clause as ``format_expr`` writes it (grammar ``_trigger_name``): nothing for
 *  ``BOTH``; a preset's keyword; ``first:N`` / ``after:N`` for a counted window; with a gate,
 *  ``@when:G`` on the ``RESPONSE`` window (the gate's own) and ``@<window>&when:G`` on any other
 *  (``@both&when:G`` included), the threshold written as ``%g``. */
export function triggerClause({ trigger, tokens, gate }: SteerTiming): string {
  const window = trigger === "FIRST_N"
    ? `first:${tokens ?? 0}`
    : trigger === "AFTER_N" ? `after:${tokens ?? 0}` : PRESET_KEYWORD[trigger];
  if (!gate) return trigger === "BOTH" ? "" : `@${window}`;
  const when = `when:${gate.probe}${gate.op}${formatCoeff(gate.threshold)}`;
  return trigger === "RESPONSE" ? `@${when}` : `@${window}&${when}`;
}

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

/** The order families emit in: flat terms first (they share one slide), projections, then the
 *  atoms, units and curved terms. */
const TERM_ORDER: readonly SteerEntry["mode"][] = ["subspace", "projection", "jlens", "slens", "sae", "unit", "manifold"];

/** Render one rack entry as its own term (whatever ``enabled`` says). */
export function formatTerm(name: string, entry: SteerEntry, subspaceAlong: number): string {
  switch (entry.mode) {
    case "subspace": return formatSubspaceTerm(name, entry, subspaceAlong);
    case "projection": return formatProjectionTerm(entry);
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
    return `${formatBang(`${nameWithVariant(name, entry.variant)}${clause}`, subspaceAlong)}${triggerClause(entry)}`;
  }
  const position = entry.label ?? entry.coords.map((c) => formatCoeff(c)).join(",");
  const selector = `${nameWithVariant(name, entry.variant)}%${position}`;
  return `${formatCoeff(subspaceAlong)} ${selector}${clause}${triggerClause(entry)}`;
}

function nameWithVariant(name: string, variant: Variant): string {
  return variant === "raw" ? name : `${name}:${variant}`;
}

/** A number as the server's ``format_expr`` writes it: Python's ``%g`` (six significant digits,
 *  trailing zeros dropped, an exponent below 1e-4 or from 1e6 on: ``1e-05``, ``1.5e+06``), so the
 *  chips' expression is the canonical one. The leading sign rides the term; the joiner above
 *  splits +/-. */
export function formatCoeff(coeff: number): string {
  if (Number.isNaN(coeff) || !Number.isFinite(coeff)) return "0";
  return formatG(coeff, 6);
}

/** ``x`` as C's (and Python's) ``%.{precision}g``: the exact binary value rounded to ``precision``
 *  significant digits, an exact tie to even (``toExponential`` and ``toPrecision`` round a tie
 *  up, so 1.015625 would write 1.01563 where Python writes 1.01562); fixed notation while the
 *  decimal exponent is from −4 to below ``precision``, else ``d.ddde±XX``; trailing zeros dropped.
 *  ``-0`` writes ``-0``. Finite input only. */
export function formatG(x: number, precision: number): string {
  if (x === 0) return Object.is(x, -0) ? "-0" : "0";
  const sign = x < 0 ? "-" : "";
  // The exact value: an integer numerator over a power of two.
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, Math.abs(x));
  const hi = view.getUint32(0);
  const lo = view.getUint32(4);
  const biased = (hi >>> 20) & 0x7ff;
  let mantissa = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let shift = biased - 1075;
  if (biased === 0) shift = -1074;
  else mantissa |= 1n << 52n;
  let num = shift >= 0 ? mantissa << BigInt(shift) : mantissa;
  let den = shift >= 0 ? 1n : 1n << BigInt(-shift);
  // The decimal exponent: 10^e <= value < 10^(e+1), estimated in floating point, then made exact.
  let exponent = Math.floor(Math.log10(Math.abs(x)));
  const pow10 = (n: number) => 10n ** BigInt(n);
  const atLeast = (e: number) => (e >= 0 ? num >= den * pow10(e) : num * pow10(-e) >= den);
  while (!atLeast(exponent)) exponent -= 1;
  while (atLeast(exponent + 1)) exponent += 1;
  // value · 10^(precision−1−exponent), rounded half to even.
  const scale = precision - 1 - exponent;
  if (scale >= 0) num *= pow10(scale);
  else den *= pow10(-scale);
  let digits = num / den;
  const twice = 2n * (num % den);
  if (twice > den || (twice === den && digits % 2n === 1n)) digits += 1n;
  if (digits === pow10(precision)) {
    digits = pow10(precision - 1);
    exponent += 1;
  }
  const text = digits.toString();
  const trim = (s: string) => (s.includes(".") ? s.replace(/0+$/u, "").replace(/\.$/u, "") : s);
  if (exponent < -4 || exponent >= precision) {
    const mantissaText = trim(`${text[0]}.${text.slice(1)}`);
    return `${sign}${mantissaText}e${exponent < 0 ? "-" : "+"}${String(Math.abs(exponent)).padStart(2, "0")}`;
  }
  const decimals = precision - 1 - exponent;
  if (decimals === 0) return `${sign}${text}`;
  const padded = text.padStart(decimals + 1, "0");
  return `${sign}${trim(`${padded.slice(0, -decimals)}.${padded.slice(-decimals)}`)}`;
}

/** An ablation as the server writes it: ``!x`` at full strength, ``-!x`` at −1, ``0.5 !x``
 *  otherwise. ``target`` carries the clause (``jlens/storm{20-30}``); the trigger follows. */
function formatBang(target: string, coeff: number): string {
  if (coeff === 1) return `!${target}`;
  if (coeff === -1) return `-!${target}`;
  return `${formatCoeff(coeff)} !${target}`;
}

/** Render one projection — ``<alpha> base~onto`` or ``<alpha> base|onto``, its clause and its
 *  ``@`` clause (grammar ``_fmt_projected``). */
export function formatProjectionTerm(entry: ProjectionSteerEntry): string {
  return `${formatCoeff(entry.alpha)} ${entry.base}${entry.operator}${entry.onto}${layerClause(entry.layers)}${triggerClause(entry)}`;
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
  return `${entry.ablate ? formatBang(target, entry.alpha) : `${formatCoeff(entry.alpha)} ${target}`}${triggerClause(entry)}`;
}

/** Render one unit term — ``<alpha> neuron/L31n4410`` (a push; a negative α takes the other pole)
 *  or ``!neuron/…`` / ``!head/…`` (mean-ablated: α 1, written bare like every full ablation;
 *  ``0.5 !head/…`` blends halfway).  A unit carries its own layer, so it never writes a clause
 *  (the grammar refuses one); which terms a unit takes is lib/unitTerms.ts's rule, held by the
 *  rack. */
export function formatUnitTerm(name: string, entry: UnitSteerEntry): string {
  return `${entry.ablate ? formatBang(name, entry.alpha) : `${formatCoeff(entry.alpha)} ${name}`}${triggerClause(entry)}`;
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
  // relative to ``along``'s sign, as format_expr writes it: the sign bit, so a −0 along (an agent
  // may set one) writes ``-0,-0.3``, which parses back to along −0, onto 0.3.
  const onto = entry.onto ?? 0;
  const negative = entry.blend < 0 || Object.is(entry.blend, -0);
  const coeff =
    onto !== 0
      ? `${formatCoeff(entry.blend)},${formatCoeff(negative ? -onto : onto)}`
      : formatCoeff(entry.blend);
  return `${coeff} ${selector}${layerClause(entry.layers)}${triggerClause(entry)}`;
}
