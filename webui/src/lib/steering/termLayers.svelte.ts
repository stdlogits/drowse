// Where each racked term acts, as the server says: the layers it is fitted at, its share at each
// and the layers its clause selects, read off ``POST .../steering/validate`` (its ``terms`` and
// ``n_layers``). The term editor draws its depth strip from it and the chips their spans. A
// runtime that does not report them answers ``unavailable``, and the strip stays hidden.
//
// One request per probe text, cached (the text names the term, its position and its clause, not
// its strength or trigger, so scrubbing a coefficient asks nothing). A probe steers ``@response``,
// which leaves the prompt prefill unsteered, so a validation never drops the session's cached
// prefix.

import { SvelteMap } from "svelte/reactivity";
import { apiSessions } from "../runtime/services";
import { describeError } from "../runtime/errors";
import { formatTerm } from "../expression";
import type { SteerEntry } from "../types";

export interface TermWhere {
  /** The layers the term can act on, ascending. */
  fitted: number[];
  /** Its weight at each fitted layer (aligned with ``fitted``, mean 1 over them). */
  share: number[];
  /** The layers its clause selects (``fitted`` without one). */
  layers: number[];
  /** The model's depth. */
  nLayers: number;
}

export type TermLayersState =
  | { status: "loading" }
  | { status: "ready"; where: TermWhere }
  | { status: "invalid"; error: string }
  | { status: "unavailable" };

/** The cache's bound: a recipe holds a handful of terms, each with a few clauses in flight. */
const LIMIT = 64;

const cache = new SvelteMap<string, TermLayersState>();

/** The text a term's layers are asked with: the term at strength 1, ``@response``, with its clause
 *  when ``scoped`` and without one otherwise (every fitted layer, the strip's extent). */
export function layersProbe(name: string, entry: SteerEntry, scoped: boolean): string {
  const trigger = "RESPONSE" as const;
  switch (entry.mode) {
    case "subspace":
      return formatTerm(name, { ...entry, trigger, layers: scoped ? entry.layers : null }, 1);
    case "manifold":
      return formatTerm(name, { ...entry, trigger, blend: 1, onto: 0, layers: scoped ? entry.layers : null }, 1);
    case "jlens":
    case "sae":
      return formatTerm(name, { ...entry, trigger, alpha: 1, layers: scoped ? entry.layers : null }, 1);
    case "unit":
      return formatTerm(name, { ...entry, trigger, alpha: 1 }, 1);
  }
}

/** What the server said about ``probe`` (asking it the first time). Keyed by the model too, so a
 *  model change never reads another model's layers. */
export function termLayers(probe: string, model: string | null | undefined): TermLayersState {
  const key = `${model ?? ""}\n${probe}`;
  const known = cache.get(key);
  if (known) return known;
  const loading: TermLayersState = { status: "loading" };
  // Asked inside a reactive read: the write lands after it, never during it.
  queueMicrotask(() => {
    if (cache.has(key)) return;
    cache.set(key, loading);
    while (cache.size > LIMIT) cache.delete(cache.keys().next().value as string);
    void ask(key, probe);
  });
  return loading;
}

async function ask(key: string, probe: string): Promise<void> {
  try {
    const result = await apiSessions.validateSteering(probe);
    if (!result.valid) {
      cache.set(key, { status: "invalid", error: result.error ?? "This term does not resolve." });
      return;
    }
    const row = result.terms?.length === 1 ? result.terms[0] : undefined;
    if (!row || typeof result.n_layers !== "number") {
      cache.set(key, { status: "unavailable" });
      return;
    }
    cache.set(key, {
      status: "ready",
      where: { fitted: [...row.fitted], share: [...row.share], layers: [...row.layers], nLayers: result.n_layers },
    });
  } catch (error) {
    cache.set(key, { status: "invalid", error: describeError(error) });
  }
}

/** A racked term's place: where it can act (``where``, every fitted layer) and the layers its
 *  clause selects now (``selected``), or why that is not known. */
export interface TermPlace {
  status: TermLayersState["status"];
  where: TermWhere | null;
  selected: number[];
  error: string | null;
}

/** Where ``entry`` acts, asked of the server for ``model``: its extent from the unscoped term, and
 *  its selection from the clause (``workspace``, and any clause the server refuses, are read from
 *  the scoped term, so a refusal shows as ``error``). */
export function placeOf(name: string, entry: SteerEntry, model: string | null | undefined): TermPlace {
  const all = termLayers(layersProbe(name, entry, false), model);
  if (all.status !== "ready") {
    return { status: all.status, where: null, selected: [], error: all.status === "invalid" ? all.error : null };
  }
  const scope = entry.mode === "unit" ? null : entry.layers ?? null;
  if (scope === null) return { status: "ready", where: all.where, selected: all.where.fitted, error: null };
  if (Array.isArray(scope) && scope.length === 0) return { status: "ready", where: all.where, selected: [], error: null };
  const scoped = termLayers(layersProbe(name, entry, true), model);
  const fallback = Array.isArray(scope) ? scope.filter(L => all.where.fitted.includes(L)) : [];
  if (scoped.status === "ready") return { status: "ready", where: all.where, selected: scoped.where.layers, error: null };
  return {
    status: "ready",
    where: all.where,
    selected: fallback,
    error: scoped.status === "invalid" ? scoped.error : null,
  };
}

/** Forget every answer: a fit, an install or a deletion can change where a term acts. */
export function forgetTermLayers(): void {
  cache.clear();
}
