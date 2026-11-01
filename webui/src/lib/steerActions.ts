// The one way into the recipe. Every steering term is made here and lands as a chip in the
// composer's strip, selected, so its editor opens under the chips for its strength: "+ Steer"'s
// pickers (panels/rack/SteerPicker.svelte) and the one-click Push and Silence on the instruments'
// rows (the lens's words, the SAE's features, a unit's view, the token view's units and heads, the
// Units room). Nothing else edits a strength.
//
// A unit's term queues behind a running reply like every rack edit; its chip opens once it lands
// (RecipeBar holds a selected term while a rack edit is queued).
import { select } from "./stores/selection.svelte";
import {
  ATOM_DEFAULT_ALPHA,
  addJLensToRack,
  addManifoldToRack,
  addSaeToRack,
  addSLensToRack,
  addSubspaceToRack,
  lensTokenText,
  patchSteerEntry,
  setSteerLabel,
  setUnitTerm,
  steerRack,
} from "./stores/steering.svelte";
import type { LensNamespace } from "./stores/instruments.svelte";
import type { Variant } from "./types";
import { parseUnit } from "./units";
import { UNIT_ABLATE_ALPHA, UNIT_PUSH_ALPHA } from "./unitTerms";

/** Select a racked term: its chip opens for its strength. */
export function openTerm(name: string): void {
  select({ kind: "term", name });
}

/** Push a neuron along its write, on ``pole`` (``-``: its − pole, a negative strength). A neuron
 *  already pushed keeps its strength and takes the pole; one silenced is pushed at the default.
 *  A head has no push: nothing changes and the answer is null. Else the answer is the rack key. */
export function pushUnit(id: string, pole: "+" | "-" = "+"): string | null {
  const ref = parseUnit(id);
  if (ref?.kind !== "neuron") return null;
  const held = steerRack.entries.get(ref.id);
  const strength = held?.mode === "unit" && !held.ablate && held.alpha !== 0 ? Math.abs(held.alpha) : UNIT_PUSH_ALPHA;
  const alpha = pole === "-" ? -strength : strength;
  const same = held?.mode === "unit" && !held.ablate && held.alpha === alpha && steerRack.readOnlyRecipe === null;
  const name = same ? ref.id : setUnitTerm(ref.id, { ablate: false, alpha });
  if (name !== null) openTerm(name);
  return name;
}

/** Silence a neuron or a head: hold it at its neutral mean (exact mean-ablation). One already
 *  silenced keeps its strength. */
export function silenceUnit(id: string): string | null {
  const ref = parseUnit(id);
  if (!ref) return null;
  const held = steerRack.entries.get(ref.id);
  const same = held?.mode === "unit" && held.ablate && steerRack.readOnlyRecipe === null;
  const name = same ? ref.id : setUnitTerm(ref.id, { ablate: true, alpha: UNIT_ABLATE_ALPHA });
  if (name !== null) openTerm(name);
  return name;
}

/** A lens token: a word (``stars``), or a vocabulary id with the text it decodes to (the token
 *  views name a token by id, since two ids can decode to one text). */
export type LensToken = { word: string } | { id: number; text: string };

/** Push toward a lens token: as said (``jlens/``, toward saying it) or as read (``slens/``, toward
 *  having read it). Callers check the word resolves (``validateLensToken``); the engine checks it
 *  again when it composes. A term already there that removes the token turns into a push. */
export function pushLensToken(token: LensToken, namespace: LensNamespace): string | null {
  let bare: string;
  if ("id" in token) {
    lensTokenText.set(token.id, token.text);
    bare = `#${token.id}`;
  } else {
    bare = token.word.trim().replace(/^[js]lens\//u, "");
  }
  if (!bare) return null;
  const name = `${namespace}/${bare}`;
  if (namespace === "jlens") addJLensToRack(bare);
  else addSLensToRack(bare);
  return pushAtom(name, namespace);
}

/** Push along an SAE feature's decoder row (``sae/<id>``). */
export function pushFeature(featureId: number): string | null {
  if (!Number.isInteger(featureId) || featureId < 0) return null;
  addSaeToRack(featureId);
  return pushAtom(`sae/${featureId}`, "sae");
}

function pushAtom(name: string, mode: "jlens" | "slens" | "sae"): string | null {
  const entry = steerRack.entries.get(name);
  if (!entry) return null;
  if (entry.mode === mode && entry.ablate) patchSteerEntry(name, { ablate: false, alpha: ATOM_DEFAULT_ALPHA }, [mode]);
  openTerm(name);
  return name;
}

/** Push toward a flat fit (a concept, or a flat fan like personas) by its rack key
 *  (``namespace/name``), at a node when ``label`` names one (a concept's pole). Flat terms share
 *  one strength. A term already there moves to the node. */
export function pushConcept(key: string, variant: Variant, label: string | null = null): string {
  addSubspaceToRack(key, variant);
  if (label !== null) setSteerLabel(key, label);
  openTerm(key);
  return key;
}

/** Steer to a position on a curved manifold: a node (``label``), or its first one. A term already
 *  there moves to the node. */
export function pushPosition(key: string, variant: Variant, label: string | null = null): string {
  addManifoldToRack(key, variant);
  if (label !== null) setSteerLabel(key, label);
  openTerm(key);
  return key;
}
