// One unified steer rack.
//
// A steering vector is the K=2 flat case of a manifold, so every term is a
// position on a fitted geometry — one ``entries`` map of tagged
// ``SteerEntry`` (``mode: "subspace" | "manifold" | "jlens" | "slens" | "sae" | "unit"``), one
// card, and one serializer.  Subspace (flat) terms share the rack-level
// ``subspaceAlong`` master (the merged affine subspace slides once);
// manifold (curved) terms keep their per-card along/onto; the two
// single-direction atom families share one mutator set.
//
// The sidecars live here too: ``profiles`` + ``correlation`` (vector
// metadata) and ``catalog`` + ``loading`` / ``error`` (the manifold HTTP
// surface).

import { SvelteMap } from "svelte/reactivity";
import { apiManifolds, apiProfiles, apiSessions } from "../runtime/services";
import { userFacingError } from "../runtime/userFacingError";
import type { CorrelationData, ManifoldInfo, VectorInfo } from "../types";
import type {
  AtomMode,
  AtomSteerEntry,
  LayerScope,
  ManifoldSteerEntry,
  ProjectionSteerEntry,
  SteerEntry,
  SteerTiming,
  SteeringPartJSON,
  SubspaceSteerEntry,
  UnitSteerEntry,
  Variant,
} from "../types";
import { formatLayerIndices } from "../steering/layers";
import { recipeAsChips } from "../steering/recipe";
import { forgetTermLayers } from "../steering/termLayers.svelte";
import { serializeExpression } from "../expression";
import { parseUnit } from "../units";
import { unitTermProblem } from "../unitTerms";
import { enqueueOrApply } from "./pending.svelte";

/** Default shared subspace-along master — the ~0.5 coherent sweet spot
 *  (matches the engine's ``_SUBSPACE_GAIN`` calibration so a freshly
 *  racked concept at its pole lands at a usable strength). */
const DEFAULT_SUBSPACE_ALONG = 0.5;

export interface SteerRack {
  /** Rack key = atom display form (``honest``, ``ns/foo``, ``happy.sad``,
   *  ``personas``).  One entry per name; ``mode`` discriminates subspace
   *  (flat) vs manifold (curved).  Variant lives on the entry, not the key —
   *  matching the drowse parser's Steering.alphas semantics. */
  entries: Map<string, SteerEntry>;
  /** A recipe from elsewhere (an agent's expression, a saved conversation's, the command line's
   *  default) that the chips cannot hold: shown as one read-only chip and sent as written. ``null``
   *  when the chips are the recipe. People never type one; the chips are how they write recipes
   *  (``applyRecipe`` turns a recipe into chips whenever every term fits one). */
  readOnlyRecipe: string | null;
  /** Why ``readOnlyRecipe`` is not chips (``null`` while that is being checked), and whether
   *  checking again could change it (the check could not run). */
  recipeNote: { text: string; retry: boolean } | null;
  /** Shared "subspace along" master — the single slide magnitude every
   *  subspace (flat) term serializes with (the merged affine subspace has one
   *  slide).  Unclamped (a high-share layer is meant to overshoot; the engine
   *  bounds it via ``norm_cap``).  Defaults to the ~0.5 coherent sweet spot. */
  subspaceAlong: number;
  /** Per-profile metadata fetched from GET /profiles/{name}.
   * Populated lazily; absent until the user opens a strip's expander. */
  profiles: Map<string, VectorInfo>;
  /** Cosine matrix from GET /correlation.  Fetched only by the open
   *  correlation drawer and the explicit health refresh, never per
   *  generation.  ``invalidateCorrelation`` clears it when the probe or
   *  vector roster changes; ``markCorrelationStale`` (after a generation)
   *  keeps it but has the drawer refetch while it is open
   *  (``correlationCache``). */
  correlation: CorrelationData | null;
  /** Server-side catalog of available manifolds. */
  catalog: ManifoldInfo[];
  loading: boolean;
  error: string | null;
}

// SvelteMap from svelte/reactivity — plain Map mutations don't trigger
// Svelte 5 rune reactivity, so any rack add/remove or profile cache
// update wouldn't re-render the strips list.  SvelteMap.set/.delete is
// rune-tracked.  Inner-object property writes still aren't tracked, so
// callers that mutate an entry must reassign via .set(name, {...e, …}).
export const steerRack: SteerRack = $state({
  entries: new SvelteMap(),
  readOnlyRecipe: null,
  recipeNote: null,
  subspaceAlong: DEFAULT_SUBSPACE_ALONG,
  profiles: new SvelteMap(),
  correlation: null,
  catalog: [],
  loading: false,
  error: null,
});

/** Server-derived list of registered vectors — names only.  Mirrors
 * sessionState.info?.profiles but kept as its own slice so panels that
 * only care about the list don't re-render when other session fields
 * change. */
export const vectorsState: { names: string[] } = $state({ names: [] });

/** Freshness of ``steerRack.correlation``.  ``revision`` bumps whenever the
 * server's matrix may have changed; ``fetched`` is the revision the cached
 * matrix was fetched at.  The correlation drawer refetches while it is open
 * and the two differ, so nothing fetches while it is closed. */
export const correlationCache: { revision: number; fetched: number } = $state({
  revision: 0,
  fetched: -1,
});

/** Drop the cached matrix and prevent an older in-flight request from
 * repopulating it after the direction/probe roster changes. */
export function invalidateCorrelation(): void {
  correlationCache.revision += 1;
  steerRack.correlation = null;
}

/** Keep the cached matrix on screen but refetch it the next time it is
 * shown.  A generation can register the steering vectors it used (they join
 * the server's pool) and frees the GPU for the snapshots a busy server left
 * as null cells. */
export function markCorrelationStale(): void {
  correlationCache.revision += 1;
}

export async function refreshVectorList(): Promise<void> {
  const r = await apiProfiles.list();
  const names = r.profiles.map((v) => v.name);
  // Registered vectors are part of the correlation pool.
  if (names.join("\n") !== vectorsState.names.join("\n")) invalidateCorrelation();
  vectorsState.names = names;
  // Cache profile metadata — cheap, server already serialized.
  steerRack.profiles.clear();
  for (const v of r.profiles) {
    steerRack.profiles.set(v.name, v);
  }
}

export async function refreshCorrelation(
  names?: string[] | null,
  strict = false,
): Promise<void> {
  const requestedAt = correlationCache.revision;
  try {
    const data = await apiProfiles.correlation(names);
    if (requestedAt === correlationCache.revision) {
      steerRack.correlation = data;
      correlationCache.fetched = requestedAt;
    }
  } catch (error) {
    if (requestedAt === correlationCache.revision) steerRack.correlation = null;
    if (strict) throw error;
  }
}

// -------------------------------------------- subspace-mode mutators -

function defaultSubspaceEntry(
  coords: number[] = [],
  label: string | null = null,
  variant: Variant = "raw",
): SubspaceSteerEntry {
  return {
    mode: "subspace",
    ablate: false,
    coords,
    label,
    variant,
    trigger: "BOTH",
    enabled: true,
  };
}

// SvelteMap tracks .set/.delete, not mutations of stored objects, so every
// rack edit reassigns its entry with a fresh spread through ``patchSteerEntry``.

type EntryOf<M extends SteerEntry["mode"]> = Extract<SteerEntry, { mode: M }>;
type SteerPatch = { [K in Exclude<keyof SteerEntry, keyof SteerTiming>]?: SteerEntry[K] }
  & Partial<Pick<SubspaceSteerEntry, "ablate" | "coords" | "label" | "layers">>
  & Partial<Pick<ManifoldSteerEntry, "blend" | "onto">>
  & Partial<Pick<AtomSteerEntry | UnitSteerEntry | ProjectionSteerEntry, "alpha">>;

/** Describe a rack edit for the pending-queue bubble. */
function describePatch(name: string, patch: SteerPatch): string {
  if (patch.enabled !== undefined) return `${patch.enabled ? "enable" : "disable"} ${name}`;
  if (patch.ablate !== undefined) return `${patch.ablate ? "ablate" : "push"} ${name}`;
  if (patch.layers !== undefined) {
    const where = patch.layers === null ? "every layer" : patch.layers === "workspace" ? "the workspace" : `layers ${formatLayerIndices(patch.layers) || "none"}`;
    return `${name} at ${where}`;
  }
  const [field, value] = Object.entries(patch)[0] ?? ["edit", ""];
  return `${field} ${name}${typeof value === "number" ? ` ${value.toFixed(3)}` : typeof value === "string" ? ` ${value}` : ""}`;
}

/** Apply ``patch`` to one rack entry (queued behind a running generation).
 *  ``modes`` restricts it to entries of those modes; an absent entry or one
 *  of another mode is left alone. */
export function patchSteerEntry<M extends SteerEntry["mode"]>(
  name: string,
  patch: Omit<Partial<EntryOf<M>>, keyof SteerTiming> & SteerPatch,
  modes?: readonly M[],
): void {
  enqueueOrApply(describePatch(name, patch), () => {
    const entry = steerRack.entries.get(name);
    if (!entry || (modes && !modes.includes(entry.mode as M))) return;
    const next = { ...entry, ...patch } as SteerEntry;
    // A unit's edit keeps to the one rule (a head is not pushed, an ablation not negative).
    if (next.mode === "unit" && unitTermProblem(name, next) !== null) return;
    steerRack.entries.set(name, next);
  });
}

/** A term's timing alone (its window, its count and its gate), each key only when it is set. */
export function timingOf(entry: SteerTiming): SteerTiming {
  const counted = entry.trigger === "FIRST_N" || entry.trigger === "AFTER_N";
  return {
    trigger: entry.trigger,
    ...(counted ? { tokens: entry.tokens ?? 0 } : {}),
    ...(entry.gate ? { gate: { ...entry.gate } } : {}),
  };
}

/** Set when a term acts: its window (a preset, or a counted one with its count) and its probe
 *  gate, replacing the ones it had (queued behind a running generation like every rack edit). */
export function setSteerTiming(name: string, timing: SteerTiming): void {
  const next = timingOf(timing);
  enqueueOrApply(`when ${name}`, () => {
    const entry = steerRack.entries.get(name);
    if (!entry) return;
    const { tokens: _tokens, gate: _gate, ...rest } = entry;
    steerRack.entries.set(name, { ...rest, ...next } as SteerEntry);
  });
}

/** A flat concept's relation to a second one: push only the part of it along ``onto``
 *  (``operator`` ``~``), or all but that part (``|``); ``null`` pushes all of it again. The term's
 *  key is its selector, so a relation re-keys it (``base~onto``): ``onselect`` gets the new key
 *  once the change has applied (queued behind a running generation like every rack edit). A
 *  projection keeps its own signed strength, from the shared flat one with the sign of the pole
 *  it pushed toward; all of it again rejoins the shared strength at the pole its sign points to.
 *  Nothing changes when the new key is already a chip. */
export function relateSteerEntry(
  name: string,
  relation: { operator: "~" | "|"; onto: string } | null,
  onselect?: (key: string) => void,
): void {
  enqueueOrApply(relation === null ? `all of ${name}` : `${name} ${relation.operator === "~" ? "along" : "not along"} ${relation.onto}`, () => {
    const entry = steerRack.entries.get(name);
    if (!entry || (entry.mode !== "subspace" && entry.mode !== "projection")) return;
    const layers = entry.layers === undefined ? {} : { layers: entry.layers };
    const common = { ...timingOf(entry), ...layers, enabled: entry.enabled };
    let key: string;
    let next: SteerEntry;
    if (relation === null) {
      if (entry.mode !== "projection") return;
      const info = manifoldByName(entry.base);
      const pole = entry.alpha < 0 && (info?.node_labels.length ?? 0) > 1 ? 1 : 0;
      const label = info?.node_labels[pole] ?? null;
      const coords = info?.node_coords[pole] ? [...info.node_coords[pole]] : [];
      key = entry.base;
      next = { mode: "subspace", ablate: false, coords, label, variant: "raw", ...common };
    } else {
      const base = entry.mode === "projection" ? entry.base : name;
      let alpha: number;
      if (entry.mode === "projection") alpha = entry.alpha;
      else {
        const info = manifoldByName(name);
        alpha = entry.label !== null && info?.node_labels.indexOf(entry.label) === 1 ? -steerRack.subspaceAlong : steerRack.subspaceAlong;
      }
      key = `${base}${relation.operator}${relation.onto}`;
      next = { mode: "projection", base, operator: relation.operator, onto: relation.onto, alpha: alpha || 0, ...common };
    }
    if (key !== name && steerRack.entries.has(key)) return;
    steerRack.entries.delete(name);
    steerRack.entries.set(key, next);
    onselect?.(key);
  });
}

/** Move a geometric (subspace or manifold) term to free authoring coords —
 *  an XYPad or slider drag — dropping any node-label binding so it
 *  serializes as a coord list. */
export function setSteerCoords(name: string, coords: number[]): void {
  patchSteerEntry(name, { coords: [...coords], label: null }, ["subspace", "manifold"]);
}

/** Bind a geometric term to a node label (``<name>%<label>``), mirroring the
 *  node's coords so the XYPad still renders the position; ``null`` reverts
 *  to coord form.  Without catalog metadata the label is kept as-is and
 *  resolves server-side. */
export function setSteerLabel(name: string, label: string | null): void {
  const info = label === null ? null : manifoldByName(name);
  const index = info ? info.node_labels.indexOf(label!) : -1;
  const coords = index >= 0 && info!.node_coords[index] ? [...info!.node_coords[index]] : undefined;
  patchSteerEntry(name, coords ? { label, coords } : { label }, ["subspace", "manifold"]);
}

/** The shared "subspace along" master — one slide magnitude for every
 *  subspace (flat) term (the merged affine subspace slides once).  Adjusting
 *  it scales every flat term uniformly. */
export function setSubspaceAlong(along: number): void {
  enqueueOrApply(`subspace along ${along.toFixed(3)}`, () => {
    steerRack.subspaceAlong = along;
  });
}

/** Add a flat (subspace) term.  A 2-node concept defaults to its positive
 *  pole (label form); a higher-rank flat (personas) to the domain centroid;
 *  an uncatalogued typed name to its positive pole label.  Magnitude is the
 *  shared ``subspaceAlong`` master, not per-card. */
export function addSubspaceToRack(name: string, variant: Variant = "raw"): void {
  if (steerRack.entries.has(name)) return;
  dropRecipe();
  const info = manifoldByName(name);
  let coords: number[] = [];
  let label: string | null = null;
  if (info && info.node_count === 2 && info.node_labels.length > 0) {
    label = info.node_labels[0];
    coords = info.node_coords?.[0] ? [...info.node_coords[0]] : [];
  } else if (info) {
    coords = manifoldCentroid(info);
  } else {
    const bare = name.includes("/") ? name.slice(name.indexOf("/") + 1) : name;
    label = bare.split(".")[0];
  }
  steerRack.entries.set(name, defaultSubspaceEntry(coords, label, variant));
}

/** Scope a term to layers (its ``{…}`` clause): ``null`` for every fitted layer, ``"workspace"``,
 *  or exact indices (the empty list leaves the term out until it has a layer again).  Units carry
 *  their own layer and take no clause. */
export function setSteerLayers(name: string, layers: LayerScope): void {
  const scope = Array.isArray(layers) ? [...layers] : layers;
  patchSteerEntry(name, { layers: scope }, ["subspace", "manifold", "projection", "jlens", "slens", "sae"]);
}

/** Take any term off the rack.  Never deletes the server-side artifact. */
export function removeSteerEntry(name: string): void {
  steerRack.entries.delete(name);
}

/** The canonical expression string the rack would send to the server.
 * Recomputed on demand; cheap.  Subspace terms first (at the shared
 * ``subspaceAlong`` master), then manifold (curved) terms. */
export function currentSteeringExpression(): string {
  return steerRack.readOnlyRecipe
    ?? serializeExpression(steerRack.entries, steerRack.subspaceAlong);
}

// ------------------------------------------------ recipes from elsewhere --
//
// An agent's expression (WebMCP ``drowse_set_steering``), a saved conversation's read-only recipe
// and the command line's default (``drowse serve -S``) arrive as text. Each becomes chips when the
// server's validation describes every term and every term fits a chip exactly
// (lib/steering/recipe.ts); otherwise it stays one read-only chip that can be cleared. Mixing a
// read-only recipe with chips would misstate what generation uses, so a chip added afterwards
// replaces it.

/** Bumped by every change of the recipe, so a check that answers late never overwrites a newer
 *  rack. */
let recipeTicket = 0;

/** Make ``expression`` the recipe the next reply uses (queued behind a running generation like
 *  every rack edit): as chips when they can hold it, else read-only. ``parts`` is the caller's
 *  validation of it when it has one; otherwise the server is asked. Empty means unsteered. */
export function applyRecipe(expression: string, parts?: readonly SteeringPartJSON[]): void {
  enqueueOrApply("use a recipe", () => {
    steerRack.entries.clear();
    holdRecipe(expression, parts);
  });
}

/** Restore a saved rack: its chips, its shared flat strength and the read-only recipe it held,
 *  checked again (the chips may hold it now). */
export function restoreRack(entries: Iterable<[string, SteerEntry]>, subspaceAlong: number, readOnlyRecipe: string | null): void {
  steerRack.entries.clear();
  for (const [name, entry] of entries) steerRack.entries.set(name, entry);
  steerRack.subspaceAlong = subspaceAlong;
  holdRecipe(readOnlyRecipe ?? "");
}

/** Check a read-only recipe again (its check could not run before). */
export function retryRecipe(): void {
  if (steerRack.readOnlyRecipe !== null) holdRecipe(steerRack.readOnlyRecipe);
}

function holdRecipe(expression: string, parts?: readonly SteeringPartJSON[]): void {
  const text = expression.trim();
  const ticket = ++recipeTicket;
  steerRack.readOnlyRecipe = text || null;
  steerRack.recipeNote = null;
  if (!text) return;
  if (parts) settleRecipe(text, parts, ticket);
  else void checkRecipe(text, ticket);
}

async function checkRecipe(text: string, ticket: number): Promise<void> {
  try {
    const result = await apiSessions.validateSteering(text);
    if (ticket !== recipeTicket) return;
    if (!result.valid) steerRack.recipeNote = { text: result.error ?? "It does not resolve on this model.", retry: false };
    else if (!result.parts) steerRack.recipeNote = { text: "This runtime does not describe its terms.", retry: false };
    else settleRecipe(text, result.parts, ticket);
  } catch (error) {
    if (ticket === recipeTicket) steerRack.recipeNote = { text: userFacingError(error, "It could not be checked now."), retry: true };
  }
}

function settleRecipe(text: string, parts: readonly SteeringPartJSON[], ticket: number): void {
  if (ticket !== recipeTicket || steerRack.readOnlyRecipe !== text) return;
  const made = recipeAsChips(parts, steerRack.catalog, steerRack.subspaceAlong);
  if ("problem" in made) {
    steerRack.recipeNote = { text: `Chips cannot hold it, so only an agent can write it: ${made.problem}.`, retry: false };
    return;
  }
  steerRack.entries.clear();
  for (const [name, entry] of made.chips.entries) steerRack.entries.set(name, entry);
  steerRack.subspaceAlong = made.chips.subspaceAlong;
  steerRack.readOnlyRecipe = null;
}

/** A chip added while a read-only recipe stands replaces it (and any check still running). */
function dropRecipe(): void {
  recipeTicket += 1;
  steerRack.readOnlyRecipe = null;
  steerRack.recipeNote = null;
}

// ------------------------------------------------ manifold catalog --

/** Fetch the manifold catalog. */
export async function refreshManifoldList(): Promise<void> {
  steerRack.loading = true;
  try {
    const r = await apiManifolds.list();
    steerRack.catalog = r.manifolds;
    // A fit, an install or a deletion can move where a term acts.
    forgetTermLayers();
    steerRack.error = null;
  } catch (e) {
    steerRack.catalog = [];
    steerRack.error = userFacingError(e, "Saved directions could not be refreshed.");
  } finally {
    steerRack.loading = false;
  }
}

/** Look up a catalog row by display name (``ns/name`` or bare name). */
export function manifoldByName(name: string): ManifoldInfo | null {
  for (const m of steerRack.catalog) {
    if (`${m.namespace}/${m.name}` === name || m.name === name) return m;
  }
  return null;
}

/** Domain-centroid coordinates for a manifold — the default rack
 *  position.  Box: midpoint of each axis.  Sphere: the north pole
 *  ``[0,…,0,1]`` in R^(dim+1) embedding (here we just author with
 *  ``dim`` intrinsic coords, all zero, which the domain maps to a valid
 *  point). */
export function manifoldCentroid(m: ManifoldInfo): number[] {
  if (m.domain.type === "box") {
    return m.domain.axes.map((a) => (a.lo + a.hi) / 2);
  }
  // Sphere / custom — intrinsic_dim zeros is a safe authoring default.
  return new Array(m.intrinsic_dim).fill(0);
}

// -------------------------------------------- manifold-mode mutators -

/** Add a curved manifold to the rack at its domain centroid, along 0.5. */
export function addManifoldToRack(name: string, variant: Variant = "raw"): void {
  if (steerRack.entries.has(name)) return;
  dropRecipe();
  const info = manifoldByName(name);
  const coords = info ? manifoldCentroid(info) : [];
  steerRack.entries.set(name, {
    mode: "manifold",
    blend: 0.5,
    onto: 0,
    coords,
    label: null,
    variant,
    trigger: "BOTH",
    enabled: true,
  });
}

// ------------------------------------------------------ atom mutators
//
// The two single-direction atom families (``jlens/<word>``, ``sae/<id>``)
// rack identically — one α, one trigger, one enable, no geometry — and are
// edited through ``patchSteerEntry`` like every other term.  Only the two
// adds stay family-specific: they differ in key construction and validation.

/** Default α for a fresh atom chip.  Atoms run hotter than concept vectors
 *  (a single sharp direction, not a distributed contrast): ≈0.3 is the
 *  coherent sweet spot, ≥0.5 over-steers into repetition. */
export const ATOM_DEFAULT_ALPHA = 0.3;

/** Rack-key prefix per family — the atom's namespace segment in the
 *  steering grammar. */
export const ATOM_PREFIX: Record<AtomMode, string> = {
  jlens: "jlens/",
  slens: "slens/",
  sae: "sae/",
};

function addAtomToRack(mode: AtomMode, id: string): void {
  const name = `${ATOM_PREFIX[mode]}${id}`;
  if (steerRack.entries.has(name)) return;
  dropRecipe();
  steerRack.entries.set(name, {
    mode,
    ablate: false,
    alpha: ATOM_DEFAULT_ALPHA,
    trigger: "BOTH",
    enabled: true,
  } as AtomSteerEntry);
}

/** A lens atom named by vocabulary id (``jlens/#282``, from the token view's Steer toward it) reads
 *  on its chip as the token's text: the decoded token each id was added for, this page's memory
 *  only (a chip from elsewhere, or after a reload, shows its id). */
export const lensTokenText = new SvelteMap<number, string>();

/** Add a J-lens token steering chip (``α jlens/<word>``).  Accepts a bare
 *  word or a full ``jlens/…`` atom; the rack key is the full atom.
 *  Dashboard callers validate through ``apiInstruments.validateLensToken``
 *  before this local mutation; the engine revalidates when it resolves the
 *  atom. */
export function addJLensToRack(word: string): void {
  const bare = word.trim().replace(/^jlens\//, "");
  if (!bare) return;
  addAtomToRack("jlens", bare);
}

/** Add an S-lens token steering chip (``α slens/<word>``): toward having read the word, where
 *  ``jlens/`` pushes toward saying it.  Accepts a bare word or a full ``slens/…`` atom. */
export function addSLensToRack(word: string): void {
  const bare = word.trim().replace(/^slens\//, "");
  if (!bare) return;
  addAtomToRack("slens", bare);
}

/** Add a resident-SAE decoder-row steering chip (``α sae/<id>``). */
export function addSaeToRack(featureId: number): void {
  addAtomToRack("sae", String(featureId));
}

// ------------------------------------------------------ unit terms

/** Put a unit's one term into the recipe, or take it out (``null``). ``id`` names the unit in any
 *  form ``parseUnit`` reads (``neuron/L31n4410``, ``L28 h12``); the rack key is the engine's
 *  reference. A push runs along the neuron's write (``alpha`` signed, the − pole's below zero), an
 *  ablation blends the unit toward its neutral mean (``alpha`` 1 is exact mean-ablation). A unit
 *  holds one term at a time, so a push replaces an ablation and back, keeping its trigger. The one
 *  rule (lib/unitTerms.ts) refuses a head push and a negative ablation: nothing changes and the
 *  answer is null. Otherwise the answer is the rack key; the edit queues behind a running
 *  generation like every rack edit. */
export function setUnitTerm(id: string, term: { ablate: boolean; alpha: number } | null): string | null {
  const name = parseUnit(id)?.id ?? null;
  if (name === null) return null;
  if (term !== null && unitTermProblem(name, term) !== null) return null;
  const what = term === null ? `remove ${name}` : term.ablate ? `ablate ${name}` : `push ${name} ${term.alpha}`;
  enqueueOrApply(what, () => {
    if (term === null) {
      steerRack.entries.delete(name);
      return;
    }
    const prev = steerRack.entries.get(name);
    dropRecipe();
    steerRack.entries.set(name, {
      mode: "unit",
      ablate: term.ablate,
      alpha: term.alpha,
      ...(prev?.mode === "unit" ? timingOf(prev) : { trigger: "BOTH" as const }),
      enabled: prev?.mode === "unit" ? prev.enabled : true,
    });
  });
  return name;
}
