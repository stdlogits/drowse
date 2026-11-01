// Pure helpers behind the loom dialogs: the fan-out's concepts, its strength list and the
// recipe each variation runs, and the go-to-turn id lookup.

import { formatTerm } from "../../lib/expression";
import { fitGeometry } from "../../lib/manifolds/fitKind";
import type { ManifoldInfo } from "../../lib/types";

/** The concepts a fan-out sweeps: the fitted flat fits (the rack's "A concept" list) with nodes to
 *  steer toward. A curved fit has no signed strength, so it is not offered. */
export function fanoutConcepts(catalog: readonly ManifoldInfo[]): ManifoldInfo[] {
  return catalog.filter((row) => row.fitted_for_session && fitGeometry(row) === "flat" && row.node_labels.length > 0);
}

/** The composer's sampling a fan-out carries into each variation's partial recipe. */
export interface FanoutSampling {
  temperature?: number | null;
  top_p?: number | null;
  top_k?: number | null;
  max_tokens?: number | null;
}

const FANOUT_SAMPLING_FIELDS = ["temperature", "top_p", "top_k", "max_tokens"] as const;

/** One variation's partial recipe: steered by ``info`` toward its node ``toward`` alone, at
 *  ``alpha``, written as the chips write it (``steering=-0.5 default/honest.deceptive%honest``),
 *  then the composer's ``sampling`` (``temperature``, ``top_p``, ``top_k``, ``max_tokens``). A
 *  variation anchors on a committed turn, whose regen anchor is the nearest generated ancestor's
 *  recipe, and that anchor outranks the call's own sampling; the partial recipe outranks the
 *  anchor, so the variations run at what the composer shows. ``null`` when ``toward`` is not one
 *  of its nodes. Picked, never typed: the dashboard takes no recipe text. */
export function fanoutRecipe(
  info: ManifoldInfo, toward: string, alpha: number, sampling: FanoutSampling | null = null,
): string | null {
  const index = info.node_labels.indexOf(toward);
  if (index < 0) return null;
  const term = formatTerm(`${info.namespace}/${info.name}`, {
    mode: "subspace", ablate: false, coords: [...(info.node_coords[index] ?? [])], label: toward,
    variant: "raw", trigger: "BOTH", enabled: true,
  }, alpha);
  const fields = FANOUT_SAMPLING_FIELDS.flatMap((key) => {
    const value = sampling?.[key];
    return typeof value === "number" && Number.isFinite(value) ? [`${key}=${value}`] : [];
  });
  return [`steering=${term}`, ...fields].join(", ");
}

/** Parse a fan-out alpha string (comma list / ``linspace(a, b, n)`` /
 *  ``start:stop:step``) into ``number[]``. */
export function parseStrengthList(raw: string): number[] {
  const trimmed = raw.trim();
  if (!trimmed) return [];

  const linspaceMatch = trimmed.match(
    /^linspace\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,)]+?)\s*\)\s*$/i,
  );
  if (linspaceMatch) {
    const start = Number(linspaceMatch[1]);
    const stop = Number(linspaceMatch[2]);
    const count = Number(linspaceMatch[3]);
    if (!Number.isFinite(start) || !Number.isFinite(stop)) return [];
    if (!Number.isInteger(count) || count < 1) return [];
    if (count === 1) return [start];
    const step = (stop - start) / (count - 1);
    const out: number[] = [];
    for (let i = 0; i < count; i++) out.push(start + step * i);
    return out;
  }

  if (trimmed.includes(":")) {
    const parts = trimmed.split(":").map((p) => p.trim());
    if (parts.length !== 3) return [];
    const start = Number(parts[0]);
    const stop = Number(parts[1]);
    const step = Number(parts[2]);
    if (![start, stop, step].every(Number.isFinite)) return [];
    if (step === 0) return [];
    if ((stop - start) * step < 0) return [];
    const out: number[] = [];
    const eps = Math.abs(step) * 1e-9;
    const ascending = step > 0;
    let v = start;
    let guard = 0;
    while (
      (ascending ? v <= stop + eps : v >= stop - eps) &&
      guard++ < 10000
    ) {
      out.push(Number.parseFloat(v.toPrecision(12)));
      v += step;
    }
    if (guard >= 10000) return [];
    return out;
  }

  const out: number[] = [];
  for (const part of trimmed.split(",")) {
    const t = part.trim();
    if (!t) continue;
    const v = Number(t);
    if (!Number.isFinite(v)) return [];
    out.push(v);
  }
  return out;
}

export interface PrefixResolution {
  /** Single unambiguous match — caller can navigate immediately. */
  id: string | null;
  /** Every node id whose ulid starts with the prefix.  ``length > 1``
   *  signals ambiguity; ``length == 0`` signals no match. */
  matches: string[];
}

/** Resolve a node by full id or id prefix (``root`` names the root). */
export function resolveNodeIdPrefix(prefix: string, ids: Iterable<string>, rootId: string | null): PrefixResolution {
  const p = prefix.trim();
  if (!p) return { id: null, matches: [] };
  if (p === "root") {
    return rootId ? { id: rootId, matches: [rootId] } : { id: null, matches: [] };
  }
  const matches: string[] = [];
  for (const id of ids) {
    if (id === p) return { id, matches: [id] };  // exact match wins
    if (id.startsWith(p)) matches.push(id);
  }
  if (matches.length === 1) return { id: matches[0], matches };
  return { id: null, matches };
}
