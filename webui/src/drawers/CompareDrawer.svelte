<script lang="ts">
  import { onMount as onInterfaceMount } from "svelte";
  import { registerInterfaceController } from "../lib/workspaceController";
  import { profileCompareViewSchema } from "../lib/interfaceSchemas";
  import { ToolError } from "../lib/webmcp/types";

  import { MorphText } from "@stdlogits/ledger";
  import { IconButton } from "@stdlogits/ledger";
  // Pairwise compare drawer — cross-layer cosine matrix between two
  // named steering profiles / probes. Two dropdowns pick from registered
  // profiles and active probes; the body renders an L_A × L_B heatmap akin to
  // the correlation matrix, but indexed by layer rather than by name.
  //
  // Data: GET /drowse/v1/sessions/{id}/profiles/pairwise?a=&b= — the server
  // falls back to monitor profiles when a name isn't a registered steering
  // profile, so probe names resolve cleanly without a new endpoint.

  import { apiProfiles, ApiError } from "../lib/runtime/services";
  import { getRuntimeClient } from "../lib/runtime/registry";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import { comparisonProfileNames } from "../lib/profileAnalytics";
  import {
    closeDrawer,
    probeRack,
    vectorsState,
    refreshVectorList,
  } from "../lib/stores.svelte";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import Select from "../lib/Select.svelte";
  import type { PairwiseCompareResponse } from "../lib/types";

  const inferHttpProbeProfiles = getRuntimeClient().mode === "http";

  // Hosted profile lists already include the exact attached aliases that can
  // be folded to one direction.  Do not advertise lens, SAE, curved, or
  // multidimensional probes: pairwise analytics cannot compare them.
  const names = $derived(comparisonProfileNames(
    vectorsState.names,
    probeRack.active,
    probeRack.entries,
    inferHttpProbeProfiles,
  ));

  let conceptA = $state<string>("");
  let conceptB = $state<string>("");

  /** Options for the A / B pickers — same list both sides; the
   *  "(empty)" fallback only renders when the catalog is empty. */
  const nameOptions = $derived(
    names.length === 0
      ? [{ value: "", label: "(empty)" }]
      : names.map((n) => ({ value: n, label: n })),
  );
  let data = $state<PairwiseCompareResponse | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);

  // Refresh the rack on mount so newly extracted vectors show up in the
  // picker without requiring a full drawer reopen.  Cheap idempotent.
  $effect(() => {
    void refreshVectorList().catch(() => {/* non-fatal */});
  });

  // Auto-pick: first two distinct names when nothing is selected yet
  // (or when the prior selections drop out of the pool).  A drives B's
  // default to "next available != A" so the matrix renders on open
  // instead of waiting for a second click.
  $effect(() => {
    if (names.length === 0) {
      conceptA = "";
      conceptB = "";
      return;
    }
    if (!conceptA || !names.includes(conceptA)) {
      conceptA = names[0];
    }
    if (!conceptB || !names.includes(conceptB)) {
      conceptB = names.find((n) => n !== conceptA) ?? conceptA;
    }
  });

  async function load(a: string, b: string): Promise<void> {
    if (!a || !b) {
      data = null;
      return;
    }
    loading = true;
    error = null;
    try {
      data = await apiProfiles.pairwise(a, b);
    } catch (e) {
      if (e instanceof ApiError) {
        const detail =
          e.body && typeof e.body === "object" && "detail" in (e.body as object)
            ? String((e.body as { detail: unknown }).detail)
            : e.message;
        error = userFacingError(e, `Unable to compare these directions. ${detail}`);
      } else {
        error = userFacingError(e, "Unable to compare these directions. Try again.");
      }
      data = null;
    } finally {
      loading = false;
    }
  }

  // Re-fetch when either selection changes.  Idempotent server-side; no
  // need to dedupe identical (a, b) pairs.
  $effect(() => {
    void load(conceptA, conceptB);
  });

  function cellTitle(la: number, lb: number, v: number | null): string {
    return `${conceptA} L${la} × ${conceptB} L${lb}: ${v == null ? "-" : v.toFixed(3)}`;
  }

  function onClose(): void {
    closeDrawer();
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      onClose();
    }
  }

  /** Cell pixel size — matches the correlation matrix.  Typical model
   * is ~30 layers so the matrix lands ~900px square; the scroll
   * container handles larger models. */
  let focusedCell = $state("Select a cell to inspect its value");
  const CELL_SIZE = 26;

  const matrix = $derived(data?.matrix ?? null);
  const layersA = $derived<number[]>(data?.layers_a ?? []);
  const layersB = $derived<number[]>(data?.layers_b ?? []);

  onInterfaceMount(() => registerInterfaceController("profile_compare", {
    schema: profileCompareViewSchema,
    read: () => ({ busy: false, values: { profile_a: conceptA, profile_b: conceptB }, available_profiles: names, loading, error }),
    update: async (change) => {
      if (false) throw new ToolError("BUSY", "Wait for this interface operation to finish.");
      for (const key of ["profile_a", "profile_b"]) if (change[key] !== undefined && !names.includes(change[key] as string)) throw new ToolError("NOT_FOUND", "Choose a profile from available_profiles.");
      if (change.profile_a !== undefined) conceptA = change.profile_a as string;
      if (change.profile_b !== undefined) conceptB = change.profile_b as string;
    },
  }));
</script>

<svelte:window onkeydown={onKeydown} />

<aside class="drawer" aria-label="Pairwise compare">
  <header class="drawer-header">
    <div class="title">
      <h2 class="dw-title">Compare by layer</h2>
      <div class="dw-facts">
        {#if data}
          <span><b>{conceptA}</b> × <b>{conceptB}</b></span>
          <span><b>{layersA.length}</b> × <b>{layersB.length}</b> layers</span>
          <span>{data.model ?? "-"}</span>
        {:else}
          <span>{names.length < 2 ? "Needs two controls" : "Choose two controls"}</span>
        {/if}
      </div>
    </div>
    <IconButton icon="close" label="Close" onclick={onClose} />
  </header>

  <div class="picker-row">
    <label class="picker">
      <span class="meta">Rows</span>
      <Select
        bind:value={conceptA}
        options={nameOptions}
        disabled={names.length === 0}
        ariaLabel="Concept A"
      />
    </label>
    <label class="picker">
      <span class="meta">Columns</span>
      <Select
        bind:value={conceptB}
        options={nameOptions}
        disabled={names.length === 0}
        ariaLabel="Concept B"
      />
    </label>
  </div>

  <div class="body" aria-busy={loading}>
    {#if error}
      <p class="empty" role="alert">Comparison failed: {error}</p>
    {:else if loading && !matrix}
      <div class="dw-blocks" role="status" aria-label="Loading layer comparison"><i></i><i></i></div>
    {:else if !matrix || layersA.length === 0 || layersB.length === 0}
      <p class="empty">Nothing to compare yet</p>
    {:else}
      <p class="focused-cell" role="status"><MorphText text={focusedCell} numbers={false} /></p>
      <div class="grid-scroll">
        <table class="grid" style="--cell: {CELL_SIZE}px;">
          <thead>
            <tr>
              <th class="corner" scope="col">
                <span class="sr-only">{conceptA} by {conceptB}</span>
              </th>
              {#each layersB as lb (lb)}
                <th class="col-label" scope="col" {...{ "aria-description": (conceptB) + " L" + (lb) }}>
                  <span>L{lb}</span>
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each layersA as la, i (la)}
              <tr>
                <th class="row-label" scope="row" {...{ "aria-description": (conceptA) + " L" + (la) }}>L{la}</th>
                {#each layersB as lb, j (lb)}
                  {@const v = matrix[i]?.[j] ?? null}
                  <td class="cell-td">
                    <button class="matrix-pick" type="button" onpointerenter={() => focusedCell = cellTitle(la, lb, v)} onfocus={() => focusedCell = cellTitle(la, lb, v)} onclick={() => focusedCell = cellTitle(la, lb, v)} aria-label={cellTitle(la, lb, v)}>
                    <HeatmapCell
                      value={v}
                      size={CELL_SIZE}
                      title={cellTitle(la, lb, v)}
                    />
                    </button>
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>

</aside>

<style>
  /* Two controls compared layer by layer: rows are the first, columns the second. */
  .drawer { display: flex; flex-direction: column; gap: var(--s4); height: 100%; min-height: 0; padding: var(--s5) var(--s6) 0; color: var(--ink2); }
  .drawer-header { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--s5); }
  .title { display: grid; gap: var(--s1); min-width: 0; }
  .picker-row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s4); }
  .picker { display: grid; gap: var(--s1); min-width: 0; }
  .body { flex: 1 1 auto; min-height: 0; overflow: auto; padding-bottom: var(--s6); }
  .empty { color: var(--ink3); font: var(--type-small); }
  .focused-cell { min-height: 20px; margin: 0 0 var(--s3); color: var(--ink); font: var(--type-small); overflow-wrap: anywhere; }
  .matrix-pick { display: grid; place-items: center; min-width: 44px; min-height: 44px; padding: 0; border: 0; background: transparent; cursor: crosshair; }
  .grid-scroll { max-height: 100%; overflow: auto; }
  .grid { border-collapse: separate; border-spacing: 1px; font-variant-numeric: tabular-nums; }
  .grid th, .grid td { margin: 0; padding: 0; background: var(--paper); }
  .grid thead th { position: sticky; top: 0; z-index: 2; }
  .grid .row-label { position: sticky; inset-inline-start: 0; z-index: 1; padding: 0 var(--s3) 0 var(--s2); color: var(--ink3); font: var(--type-meta); text-align: end; white-space: nowrap; }
  .grid .corner { position: sticky; top: 0; inset-inline-start: 0; z-index: 3; }
  .grid .col-label { width: var(--cell); min-width: var(--cell); max-width: var(--cell); padding: 0 0 var(--s2); color: var(--ink3); font: var(--type-meta); text-align: center; vertical-align: bottom; }
  .grid .cell-td { line-height: 0; }
  @media (max-width: 720px) {
    .drawer { padding: var(--s4) var(--s4) 0; }
    .picker-row { grid-template-columns: minmax(0, 1fr); }
  }
</style>
