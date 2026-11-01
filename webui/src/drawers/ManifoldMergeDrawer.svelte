<script lang="ts">
  import { onMount as onInterfaceMount } from "svelte";
  import { registerInterfaceController } from "../lib/workspaceController";
  import { mergeDraftSchema } from "../lib/interfaceSchemas";
  import { ToolError } from "../lib/webmcp/types";

  import Select from "../lib/Select.svelte";
  import { fadeIn, Icon, IconButton, MorphText } from "@stdlogits/ledger";
  import "./manifold/form.css";
  // ManifoldMergeDrawer — discover-mode node-union merge.
  //
  // Unions the *node corpora* of two or more discover-mode manifolds and
  // lets the next autofit derive coords from the combined heap.  Restricted
  // to discover-mode sources by design — authored manifolds carry
  // user-declared geometry that isn't mergeable without a shared coordinate
  // system.
  //
  // Reached from the command palette's steering-manifold tools.

  import { onMount } from "svelte";
  import { SvelteSet } from "svelte/reactivity";
  import { apiManifolds, describeError } from "../lib/runtime/services";
  import {
    closeDrawer,
    openDrawer,
    steerRack,
    refreshManifoldList,
  } from "../lib/stores.svelte";
  import { dismissToast, pushToast } from "../lib/stores/toasts.svelte";
  import type { ManifoldInfo } from "../lib/types";
  import { isDiscoverFit } from "../lib/manifolds/fitKind";

  let { params }: { params?: unknown } = $props();
  const embedded = $derived((params as { embedded?: unknown } | null)?.embedded === true);

  function rowKey(m: ManifoldInfo): string {
    return `${m.namespace}/${m.name}`;
  }

  // ----- form state ----------------------------------------------------
  const selected = new SvelteSet<string>();
  let targetName: string = $state("");
  let fitMode: "" | "pca" | "spectral" = $state("");
  let merging: boolean = $state(false);

  /** Discover-only catalog — merge sources must be autofitted. */
  const discoverManifolds = $derived(
    steerRack.catalog.filter((m) => isDiscoverFit(m)),
  );

  function toggleSource(key: string): void {
    if (selected.has(key)) selected.delete(key);
    else selected.add(key);
  }

  /** Reconcile preview — when all selected sources share a fit_mode the
   *  merge inherits it; otherwise an explicit override is required. */
  const sourceModes = $derived.by<string[]>(() => {
    const modes = new Set<string>();
    for (const key of selected) {
      const m = steerRack.catalog.find((x) => rowKey(x) === key);
      if (m && m.fit_mode) modes.add(m.fit_mode);
    }
    return [...modes].sort();
  });

  const canSubmit = $derived(
    !merging
    && selected.size >= 2
    && targetName.trim().length > 0
    && (sourceModes.length <= 1 || fitMode !== "")
  );

  async function onSubmit(ev: SubmitEvent): Promise<void> {
    ev.preventDefault();
    const target = targetName.trim();
    if (selected.size < 2) {
      pushToast("pick >= 2 discover manifolds to merge", { kind: "error" });
      return;
    }
    if (!target) {
      pushToast("target name required", { kind: "error" });
      return;
    }
    const sources = [...selected].map((k) => {
      const [namespace, name] = k.split("/");
      return { namespace, name };
    });
    merging = true;
    const toastId = pushToast(
      `merging ${sources.length} manifolds into '${target}'…`,
      { kind: "info", ttlMs: null },
    );
    try {
      await apiManifolds.merge({
        name: target,
        sources,
        fit_mode: fitMode || undefined,
      });
      await refreshManifoldList();
      dismissToast(toastId);
      pushToast(
        `Merged into local/${target}. Fit the merged manifold next.`,
        { kind: "info" },
      );
      closeDrawer();
    } catch (e) {
      dismissToast(toastId);
      pushToast(`Couldn't merge the manifolds: ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      merging = false;
    }
  }

  onMount(() => {
    void refreshManifoldList();
  });

  onInterfaceMount(() => registerInterfaceController("manifold_merge", {
    schema: mergeDraftSchema,
    read: () => ({ busy: merging, values: { sources: [...selected], target_name: targetName, fit_mode: fitMode }, available_sources: discoverManifolds.map(rowKey), can_submit: canSubmit }),
    update: async (change) => {
      if (merging) throw new ToolError("BUSY", "Wait for this interface operation to finish.");
      if (change.sources && (change.sources as string[]).some(key => !discoverManifolds.some(row => rowKey(row) === key))) throw new ToolError("NOT_FOUND", "Choose merge sources from available_sources.");
      if (change.sources !== undefined) { selected.clear(); for (const key of change.sources as string[]) selected.add(key); }
      if (change.target_name !== undefined) targetName = change.target_name as string;
      if (change.fit_mode !== undefined) fitMode = change.fit_mode as typeof fitMode;
    },
  }));
</script>

<section class="mf-tool" class:dialog={!embedded} aria-label="Combine response controls">
  {#if !embedded}
    <header class="mf-head">
      <h2 class="dw-title">Combine</h2>
      <IconButton icon="close" label="Close" onclick={closeDrawer} />
    </header>
  {/if}

  {#if discoverManifolds.length < 2}
    <div class="dw-empty">
      <p>Combining needs two or more learned manifolds.</p>
      <button type="button" class="dw-quiet" onclick={() => openDrawer("manifold_builder")}>Build one</button>
    </div>
  {:else}
    <form class="mb-form" onsubmit={onSubmit}>
      <section class="step" aria-labelledby="merge-sources-title">
        <div class="mf-sec-head">
          <h3 id="merge-sources-title" class="step-title"><span class="n">1</span>Sources</h3>
          <span class="meta"><MorphText text={`${selected.size} of ${discoverManifolds.length}`} /></span>
        </div>
        <ul class="mf-rows" role="list">
          {#each discoverManifolds as m, i (rowKey(m))}
            {@const key = rowKey(m)}
            <li class="rise" style:--i={i}>
              <label class="mf-row">
                <input
                  class="sr-only"
                  type="checkbox"
                  checked={selected.has(key)}
                  onchange={() => toggleSource(key)}
                  disabled={merging}
                />
                <span class="mf-id">
                  <span class="mf-name">{key}</span>
                  <span class="meta">{m.domain_label} · {m.node_count} {m.node_count === 1 ? "node" : "nodes"} · {m.fit_mode}</span>
                </span>
                <span class="tick" aria-hidden="true">{#if selected.has(key)}<span class="glyph" in:fadeIn><Icon name="check" /></span>{/if}</span>
              </label>
            </li>
          {/each}
        </ul>
      </section>

      <section class="step" aria-label="Result">
        <h3 class="step-title"><span class="n">2</span>Result</h3>
        <div class="grid2">
          <label class="field">
            <span class="label">Name</span>
            <input
              type="text"
              class="input"
              placeholder="combined"
              aria-label="New combined response control name"
              bind:value={targetName}
              disabled={merging}
              autocomplete="off"
              spellcheck="false"
            />
          </label>
          <label class="field">
            <span class="label">Layout</span>
            <Select
              ariaLabel="Combined response control layout method"
              bind:value={fitMode}
              disabled={merging || selected.size === 0}
              options={[
                ...(sourceModes.length <= 1 ? [{ value: "", label: `Match the sources (${sourceModes[0] ?? "automatic"})` }] : []),
                { value: "pca", label: "PCA" }, { value: "spectral", label: "Spectral" },
              ]}
            />
          </label>
        </div>
        {#if sourceModes.length > 1}
          <p class="dim-note">The sources use different layouts ({sourceModes.join(", ")}): choose one.</p>
        {/if}
      </section>

      <div class="submit">
        <button
          type="submit"
          class="dw-button"
          disabled={!canSubmit}
          {...{ "aria-description": (selected.size < 2
              ? "Choose at least two response controls"
            : !targetName.trim()
              ? "Enter a name for the combined control"
              : "Combine controls") }}
        >
          <MorphText text={merging ? "Combining…" : `Combine ${selected.size}`} />
        </button>
        {#if !embedded}<button type="button" class="dw-text" onclick={closeDrawer}>Cancel</button>{/if}
      </div>
    </form>
  {/if}
</section>

<style>
  .tick { display: grid; place-items: center; width: 24px; height: 24px; color: var(--ink); }
  .glyph { display: grid; }
</style>
