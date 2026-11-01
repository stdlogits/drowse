// Units (MLP neurons and attention heads): what the unit view and the Units room read. A unit's
// static read (its writes and what it fires on, from its weights), its activity over a
// conversation (one replay of a loom node: the top units, a unit's trace, one token's whole
// population), the units that write a word, and the model's neuron-lens catalog binding.
//
// Every read is keyed and kept (a small LRU per kind) and deduplicated while in flight. Views read
// with the plain getters (``unitRead``, ``nodeActivity``, ``unitsForWord``) inside ``$derived`` and
// ask with the matching ``request*`` from an ``$effect``; nothing is asked while a generation runs
// (the model is busy), so a view asks again once it ends. A runtime without unit reads
// (``unitReadsAvailable``) has none of this, and the views that need it hide.
import { SvelteMap } from "svelte/reactivity";
import { runtimeClient } from "../runtime/client";
import { unitReadsAvailable } from "../runtime/contracts";
import { userFacingError } from "../runtime/userFacingError";
import type {
  HeadReadJSON,
  NeuronReadJSON,
  TokenScore,
  UnitActivityJSON,
  UnitCatalogJSON,
  UnitTopJSON,
} from "../types";
import { chatLog, genStatus } from "./chat.svelte";
import { sessionState } from "./session.svelte";
import { attachProbe, detachProbe, probeRack } from "./probes.svelte";
import { patchSteerEntry, setUnitTerm, steerRack } from "./steering.svelte";
import { pushToast } from "./toasts.svelte";
import { labelText } from "../units";
import { type UnitTerm, UNIT_ABLATE_ALPHA, UNIT_PUSH_ALPHA } from "../unitTerms";

export type UnitRead = NeuronReadJSON | HeadReadJSON;

export interface Loaded<T> {
  status: "loading" | "ready" | "error";
  value: T | null;
  error: string | null;
}

/** Whether this runtime reads units (the unit view, the Units room). */
export function unitsAvailable(): boolean {
  return unitReadsAvailable(runtimeClient);
}

/** Whether this runtime replays a node for unit activity ("On this conversation", the atlas). */
export function unitActivityAvailable(): boolean {
  return typeof runtimeClient.instruments.unitActivity === "function";
}

export const unitsState: { catalog: UnitCatalogJSON | null; catalogRead: boolean; fetching: boolean } = $state({
  catalog: null,
  catalogRead: false,
  fetching: false,
});

const LIMIT = 48;
const reads = new SvelteMap<string, Loaded<UnitRead>>();
const activity = new SvelteMap<string, Loaded<UnitActivityJSON>>();
const tops = new SvelteMap<string, Loaded<UnitTopJSON>>();
/** Labels learned from reads and top hits, by unit and pole (a head's under ``+``). */
const labels = new SvelteMap<string, string>();
const labelKey = (id: string, pole: "+" | "-") => `${id}|${pole}`;

/** Forget every read (another model's units are other units). */
export function forgetUnits(): void {
  reads.clear();
  activity.clear();
  tops.clear();
  labels.clear();
  unitsState.catalog = null;
  unitsState.catalogRead = false;
}

let model: string | null | undefined;

/** One keyed load into ``map``, unless the entry is there (loading, ready, or failed: a failure
 *  waits for ``retry``) or a generation holds the model. A new model forgets the old one's reads. */
function load<T>(map: SvelteMap<string, Loaded<T>>, key: string, run: () => Promise<T>, onReady?: (value: T) => void): void {
  const current = sessionState.info?.model_id ?? null;
  if (model !== undefined && model !== current) forgetUnits();
  model = current;
  if (map.has(key) || genStatus.active) return;
  map.set(key, { status: "loading", value: null, error: null });
  while (map.size > LIMIT) {
    const first = map.keys().next().value;
    if (first === undefined || first === key) break;
    map.delete(first);
  }
  run().then(
    (value) => {
      map.set(key, { status: "ready", value, error: null });
      onReady?.(value);
    },
    (error) => {
      map.set(key, { status: "error", value: null, error: userFacingError(error, "This could not be read. Try again.") });
    },
  );
}

/** Forget a failed read, so the view's next ask goes to the server again. */
export function retryUnits(): void {
  for (const map of [reads, activity, tops] as SvelteMap<string, Loaded<unknown>>[]) {
    for (const [key, entry] of [...map.entries()]) if (entry.status === "error") map.delete(key);
  }
}

// ---------------------------------------------------------------- static reads

function readKey(id: string, hideHubs: boolean): string {
  return `${id}|${hideHubs ? "z" : "raw"}`;
}

/** A unit's read from its weights, as far as it has come (``hideHubs``: ranked by the
 *  layer-contrastive z, the hub tokens out). */
export function unitRead(id: string, hideHubs: boolean): Loaded<UnitRead> | null {
  return reads.get(readKey(id, hideHubs)) ?? null;
}

export function requestUnitRead(id: string, hideHubs: boolean): void {
  const read = runtimeClient.instruments.unitRead;
  if (!read) return;
  load(reads, readKey(id, hideHubs), () => read(id, { k: 10, hideHubs }), (value) => {
    if (value.kind === "head") {
      const label = labelText(value.label);
      if (label) labels.set(labelKey(id, "+"), label);
      return;
    }
    for (const p of value.poles) {
      const label = labelText(p.label);
      if (label) labels.set(labelKey(id, p.pole === "-" ? "-" : "+"), label);
    }
  });
}

/** The catalog label of a unit's ``pole`` (a neuron's; a head has one), once a read or a hit carried
 *  it. A neuron's ID wears its + pole's; a row or summary on the side that fired asks for that side. */
export function unitLabelFor(id: string, pole: "+" | "-" = "+"): string | null {
  return labels.get(labelKey(id, pole)) ?? null;
}

/** The catalog label of the pole a racked unit term pushes (a head's, and an ablation's, under
 *  ``+``), once one is known: what the recipe's chip and the term editor show beside the unit. */
export function unitTermLabel(id: string, entry: { ablate: boolean; alpha: number }): string | null {
  return unitLabelFor(id, !entry.ablate && entry.alpha < 0 ? "-" : "+");
}

// ---------------------------------------------------------------- activity

export interface ActivityRequest {
  units?: readonly string[];
  rawIndex?: number | null;
  top?: number;
}

function activityKey(nodeId: string, req: ActivityRequest): string {
  return `${nodeId}|${req.top ?? 32}|${[...(req.units ?? [])].sort().join(",")}|${req.rawIndex ?? ""}`;
}

/** A loom node's unit activity, as far as it has come: its top units, the traces of ``units`` and,
 *  with ``rawIndex``, that token's population. */
export function nodeActivity(nodeId: string, req: ActivityRequest = {}): Loaded<UnitActivityJSON> | null {
  return activity.get(activityKey(nodeId, req)) ?? null;
}

export function requestNodeActivity(nodeId: string, req: ActivityRequest = {}): void {
  const replay = runtimeClient.instruments.unitActivity;
  if (!replay) return;
  load(activity, activityKey(nodeId, req), () => replay(nodeId, {
    top: req.top ?? 32,
    units: req.units && req.units.length > 0 ? [...req.units] : undefined,
    rawIndex: req.rawIndex ?? undefined,
  }));
}

export interface Reply {
  nodeId: string;
  tokens: TokenScore[];
}

/** The latest generated reply that has a loom node and tokens: what "this conversation" reads. */
export function latestReply(): Reply | null {
  for (let i = chatLog.turns.length - 1; i >= 0; i -= 1) {
    const turn = chatLog.turns[i];
    if (turn.generated && turn.nodeId && turn.tokens?.length) return { nodeId: turn.nodeId, tokens: turn.tokens };
  }
  return null;
}

// ---------------------------------------------------------------- the units that write a word

function topKey(word: string, by: "writes" | "fires", k: number): string {
  return `${by}|${k}|${word}`;
}

/** The neuron poles that write (or fire on) a single-token word, best first, as far as it has come. */
export function unitsForWord(word: string, by: "writes" | "fires" = "writes", k = 48): Loaded<UnitTopJSON> | null {
  return tops.get(topKey(word, by, k)) ?? null;
}

export function requestUnitsForWord(word: string, by: "writes" | "fires" = "writes", k = 48): void {
  const top = runtimeClient.instruments.unitTop;
  if (!top || !word) return;
  load(tops, topKey(word, by, k), () => top(word, { by, k }), (value) => {
    for (const hit of value.hits) {
      const label = labelText(hit.label);
      if (label) labels.set(labelKey(hit.unit, hit.pole === "-" ? "-" : "+"), label);
    }
  });
}

// ---------------------------------------------------------------- the catalog

/** The model's neuron-lens catalog binding (disk only on the server). */
export async function loadCatalog(): Promise<void> {
  const read = runtimeClient.instruments.unitCatalog;
  if (!read) return;
  try {
    unitsState.catalog = await read();
  } catch {
    unitsState.catalog = null;
  } finally {
    unitsState.catalogRead = true;
  }
}

/** Fetch a catalogued model's labels (the units ``fetch`` preparation), then read the binding and
 *  the units again. */
export async function fetchCatalog(): Promise<void> {
  if (unitsState.fetching) return;
  unitsState.fetching = true;
  try {
    let status = await runtimeClient.instruments.startPreparation("units", { operation: "fetch" });
    while (status.state === "queued" || status.state === "running") {
      await new Promise(resolve => setTimeout(resolve, 1000));
      status = await runtimeClient.instruments.preparationStatus("units");
    }
    if (status.error) throw new Error(status.error);
    forgetUnits();
    await loadCatalog();
  } catch (error) {
    pushToast(userFacingError(error, "The labels could not be fetched. Try again."), { kind: "error" });
  } finally {
    unitsState.fetching = false;
  }
}

// ---------------------------------------------------------------- the recipe and the rack

/** The unit's one term in the recipe, if it has one. */
export function unitTerm(id: string): UnitTerm | null {
  const entry = steerRack.entries.get(id);
  if (entry?.mode !== "unit" || steerRack.readOnlyRecipe !== null) return null;
  if (entry.ablate) return "silence";
  return entry.alpha < 0 ? "minus" : "plus";
}

/** Put ``term`` into the recipe for the unit, or take the unit's term out when it is the one there.
 *  A push keeps the strength the unit was pushed at. */
export function toggleUnitTerm(id: string, term: UnitTerm): void {
  if (unitTerm(id) === term) {
    setUnitTerm(id, null);
    return;
  }
  const held = steerRack.entries.get(id);
  const strength = held?.mode === "unit" && !held.ablate ? Math.abs(held.alpha) : UNIT_PUSH_ALPHA;
  setUnitTerm(id, term === "silence" ? { ablate: true, alpha: UNIT_ABLATE_ALPHA } : { ablate: false, alpha: term === "minus" ? -strength : strength });
}

/** Switch a racked unit term between push and ablation (the term card's two buttons). Each takes
 *  its own strength: an ablation α 1, the unit held at its neutral mean (a push's α kept would
 *  blend away from the mean instead, and a negative one amplify), a push the default strength. A
 *  head has no push. */
export function setUnitOperation(id: string, ablate: boolean): void {
  const entry = steerRack.entries.get(id);
  if (entry?.mode !== "unit" || entry.ablate === ablate) return;
  // patchSteerEntry holds the one rule (lib/unitTerms.ts): a head's push changes nothing.
  patchSteerEntry(id, { ablate, alpha: ablate ? UNIT_ABLATE_ALPHA : UNIT_PUSH_ALPHA }, ["unit"]);
}

/** Whether the unit is an attached probe (its reading rides every generation). */
export function unitProbed(id: string): boolean {
  return probeRack.active.includes(id);
}

const probing = new SvelteMap<string, boolean>();

export function unitProbeBusy(id: string): boolean {
  return probing.get(id) ?? false;
}

/** Attach the unit as a probe, or detach it. */
export async function toggleUnitProbe(id: string): Promise<void> {
  if (probing.get(id)) return;
  const attached = unitProbed(id);
  probing.set(id, true);
  try {
    if (attached) await detachProbe(id);
    else await attachProbe(id);
  } catch (error) {
    pushToast(userFacingError(error, `${id} could not be ${attached ? "detached" : "attached"}. Try again.`), { kind: "error" });
  } finally {
    probing.delete(id);
  }
}
