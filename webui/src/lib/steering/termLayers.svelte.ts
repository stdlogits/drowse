// Where each racked term acts, as the server says: the layers it is fitted at, its share at each
// and the layers its clause selects, read off ``POST .../steering/validate`` (its ``terms`` and
// ``n_layers``). The term editor draws its depth strip from it and the chips their spans. A
// runtime that does not report them answers ``unavailable``, and the strip stays hidden.
//
// One request per probe text, cached (the text names the term, its position and its clause, not
// its strength or trigger, so scrubbing a coefficient asks nothing). A probe steers ``@response``,
// which leaves the prompt prefill unsteered, so a validation never drops the session's cached
// prefix. Moving a term's position changes its probes: a term's first probe goes at once, and a
// later one waits until the position has rested ``SETTLE_MS``, so a drag asks once, not once a
// frame. Until the new answer comes, a term keeps the place it last showed.

import { SvelteMap } from "svelte/reactivity";
import { apiSessions } from "../runtime/services";
import { describeError } from "../runtime/errors";
import { formatTerm } from "../expression";
import { layerClause } from "./layers";
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

/** The bound on each map here: a recipe holds a handful of terms, each with a few clauses in flight. */
const LIMIT = 64;

/** How long a moved term's position rests before its new probe is asked. */
export const SETTLE_MS = 150;

const LOADING: TermLayersState = { status: "loading" };

const cache = new SvelteMap<string, TermLayersState>();

/** Per probe slot (the model, the term, and its extent or one clause): present once the slot has
 *  asked, holding the probe that waits for the position to rest. Plain, not reactive: the answer
 *  lands in ``cache``, which is. */
const slots = new Map<string, { key: string; timer: ReturnType<typeof setTimeout> } | null>();

/** The place each term (per model) last showed, drawn while a new probe is asked. */
const shown = new Map<string, { where: TermWhere; selected: number[] }>();

/** Set ``key`` as the newest entry of ``map``, dropping the oldest past ``LIMIT``. */
function remember<V>(map: Map<string, V>, key: string, value: V): void {
  map.delete(key);
  map.set(key, value);
  while (map.size > LIMIT) map.delete(map.keys().next().value as string);
}

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
 *  model change never reads another model's layers. A probe that names a ``slot`` replaces that
 *  slot's earlier one: past the slot's first, it is asked once it has rested ``SETTLE_MS``. */
export function termLayers(probe: string, model: string | null | undefined, slot?: string): TermLayersState {
  const key = `${model ?? ""}\n${probe}`;
  const known = cache.get(key);
  if (known) return known;
  // Asked inside a reactive read: the write lands after it, never during it.
  queueMicrotask(() => request(key, probe, slot === undefined ? null : `${model ?? ""}\n${slot}`));
  return LOADING;
}

function request(key: string, probe: string, slot: string | null): void {
  if (cache.has(key)) return;
  const pending = slot === null ? undefined : slots.get(slot);
  if (pending?.key === key) return;
  if (pending) clearTimeout(pending.timer);
  if (slot === null || pending === undefined) {
    if (slot !== null) remember(slots, slot, null);
    start(key, probe);
    return;
  }
  const timer = setTimeout(() => {
    if (slots.get(slot)?.key === key) slots.set(slot, null);
    if (!cache.has(key)) start(key, probe);
  }, SETTLE_MS);
  remember(slots, slot, { key, timer });
}

function start(key: string, probe: string): void {
  remember(cache, key, LOADING);
  void ask(key, probe);
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
 *  clause selects now (``selected``), or why that is not known. While a moved term's new probe is
 *  asked, ``status`` is ``loading`` and ``where`` and ``selected`` are the ones it last showed. */
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
  const term = `${model ?? ""}\n${name}`;
  const last = shown.get(term);
  const all = termLayers(layersProbe(name, entry, false), model, `${name}\n`);
  const scope = entry.mode === "unit" ? null : entry.layers ?? null;
  // A clause is asked beside the extent, not after it.
  const scoped = scope === null || (Array.isArray(scope) && scope.length === 0)
    ? null
    : termLayers(layersProbe(name, entry, true), model, `${name}\n${layerClause(scope)}`);
  let where: TermWhere;
  if (all.status === "ready") where = all.where;
  else if (all.status === "loading" && last) where = last.where;
  else {
    if (all.status !== "loading") shown.delete(term);
    return { status: all.status, where: null, selected: [], error: all.status === "invalid" ? all.error : null };
  }
  let selected: number[];
  let error: string | null = null;
  if (scope === null) selected = where.fitted;
  else if (scoped === null) selected = [];
  else if (scoped.status === "ready") selected = scoped.where.layers;
  else {
    // The clause's own layers, or (``workspace`` before its answer) the layers last shown.
    selected = Array.isArray(scope) ? scope.filter(L => where.fitted.includes(L)) : scoped.status === "loading" ? last?.selected ?? [] : [];
    if (scoped.status === "invalid") error = scoped.error;
  }
  remember(shown, term, { where, selected });
  return { status: all.status, where, selected, error };
}

/** Forget every answer: a fit, an install or a deletion can change where a term acts. A term keeps
 *  the place it last showed until its new answer comes, and asks it at once. */
export function forgetTermLayers(): void {
  for (const pending of slots.values()) if (pending) clearTimeout(pending.timer);
  slots.clear();
  cache.clear();
}
