// Dashboard bootstrap — call once on App mount.
//
// The only slice that depends on every other one, and nothing depends on
// it: the leaf of the store graph.

import { loadGenUiMode } from "./chat.svelte";
import { refreshLoomTree } from "./loom.svelte";
import { loadPersistedPreferences } from "./persistence.svelte";
import { refreshProbeList } from "./probes.svelte";
import { refreshSession, sessionState } from "./session.svelte";
import { applyRecipe, refreshManifoldList, refreshVectorList, steerRack } from "./steering.svelte";
import { ensureRuntimeChannel } from "./ws.svelte";

/** Bootstrap the dashboard — call once on App mount.  Resolves only once
 * every parallel fetch settles so the UI's first paint has a real session
 * shape. */
export async function bootstrap(): Promise<void> {
  // Session info has to land before the localStorage key is known
  // (it's scoped by model_id), so we serialize that step.  The other
  // refreshes parallelize as before.
  await refreshSession();
  // The command line's default recipe (``drowse serve -S``), applied once the manifold catalog is
  // here so it can become chips.
  const defaultRecipe = steerRack.readOnlyRecipe === null && steerRack.entries.size === 0
    ? sessionState.info?.default_steering ?? null
    : null;
  // Restore presentation preferences before attaching the persist effect so
  // we do not immediately overwrite them.  The tree always comes from the
  // authoritative server fetch below.
  loadPersistedPreferences();
  // Per-model render-mode override (base vs chat) — also model-scoped.
  loadGenUiMode();
  try {
    // Opening the channel rehydrates: the authoritative tree, then the profiles, the probe roster
    // and the manifold catalog. Doing it here (rather than fetching them and then opening the
    // channel, which fetched them all again) reads each once per load.
    await ensureRuntimeChannel({ sessionFresh: true });
  } catch {
    // The channel could not open (its failure is already shown); read everything once so the page
    // still opens. The send paths reopen the channel on demand.
    await Promise.allSettled([
      refreshVectorList(),
      refreshProbeList(),
      refreshManifoldList(),
      refreshLoomTree(),
    ]);
  }
  if (defaultRecipe && steerRack.readOnlyRecipe === null && steerRack.entries.size === 0) applyRecipe(defaultRecipe);
}
