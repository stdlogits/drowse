<script lang="ts">
  import { MorphText } from "@stdlogits/ledger";
  import { IconButton } from "@stdlogits/ledger";
  // Correlation overlay — N×N magnitude-weighted cosine matrix across the
  // profiles the runtime can represent as one residual direction. Readout,
  // curved, and multidimensional probes are named explicitly as omissions.
  //
  // Data: GET /drowse/v1/sessions/{id}/correlation with no ``names=`` filter —
  // the runtime returns its supported direction pool. It deliberately omits
  // probe families/shapes for which a direction cosine is not defined.
  //
  // Layout mirrors TokenDrilldownDrawer: header (title + ✕) · sticky-
  // header table body · footer hint.  Cells reuse <HeatmapCell showValue>
  // for the printed cosine — same color mapping as the click-token grid
  // so reading a row across both surfaces stays consistent.

  import {
    closeDrawer,
    correlationCache,
    probeRack,
    refreshCorrelation,
    steerRack,
  } from "../lib/stores.svelte";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import { omittedAnalyticsProbes } from "../lib/profileAnalytics";
  import { userFacingError } from "../lib/runtime/userFacingError";

  // Fetch while open whenever the cached matrix is missing or stale
  // (``correlationCache``); a fresh cache makes a reopen land instantly, and
  // a stale one stays on screen while it refetches.
  let loading = $state(false);
  let error = $state<string | null>(null);

  async function reload(): Promise<void> {
    loading = true;
    error = null;
    try {
      // No names filter asks the runtime for every supported direction.
      await refreshCorrelation(null);
    } catch (e) {
      error = userFacingError(e, "Unable to compare the active directions. Try again.");
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    if (!steerRack.correlation || correlationCache.fetched !== correlationCache.revision) void reload();
  });

  const data = $derived(steerRack.correlation);
  const names = $derived<string[]>(data?.names ?? []);
  const omissions = $derived(data
    ? omittedAnalyticsProbes(probeRack.active, probeRack.entries, data.names)
    : []);

  function cellTitle(a: string, b: string, v: number | null): string {
    return `${a} vs ${b}: ${v == null ? "-" : v.toFixed(3)}`;
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

  /** Cell pixel size — wider than the click-drilldown's grid because
   * we want to read the printed cosine value inside each cell, and
   * narrow column count (typical N=20-40) leaves room. */
  let focusedCell = $state("Select a cell to inspect its value");
  const CELL_SIZE = 26;
</script>

<svelte:window onkeydown={onKeydown} />

<aside class="drawer" aria-label="Correlation matrix">
  <header class="drawer-header">
    <div class="title">
      <h2 class="dw-title">Correlations</h2>
      <div class="dw-facts">
        <span><b>{names.length}</b> {names.length === 1 ? "direction" : "directions"}</span>
        {#if omissions.length > 0}<span><b>{omissions.length}</b> omitted</span>{/if}
      </div>
    </div>
    <div class="actions">
      <button type="button" class="dw-text" onclick={() => void reload()} disabled={loading}>{loading ? "Refreshing…" : "Refresh"}</button>
      <IconButton icon="close" label="Close" onclick={onClose} />
    </div>
  </header>

  {#if data && omissions.length > 0}
    <p class="omissions meta" role="note">
      <span>Omitted:</span>
      {#each omissions as omission, index (omission.name)}
        {#if index > 0}<span aria-hidden="true">, </span>{/if}
        <span class="omitted" {...{ "aria-description": (omission.reason) }}>{omission.name}</span>
        <span> ({omission.reason})</span>
      {/each}
    </p>
  {/if}

  <div class="body" aria-busy={loading}>
    {#if error}
      <p class="empty" role="alert">Correlation failed: {error}</p>
    {:else if loading && !data}
      <div class="dw-blocks" role="status" aria-label="Loading correlations"><i></i><i></i></div>
    {:else if !data || names.length === 0}
      <p class="empty">No single-direction profiles to correlate</p>
    {:else}
      <p class="focused-cell" role="status"><MorphText text={focusedCell} numbers={false} /></p>
      <div class="grid-scroll">
        <table class="grid" style="--cell: {CELL_SIZE}px;">
          <thead>
            <tr>
              <th class="corner" scope="col"><span class="sr-only">name</span></th>
              {#each names as col (col)}
                <th class="col-label" scope="col" {...{ "aria-description": (col) }}>
                  <span>{col}</span>
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each names as a (a)}
              <tr>
                <th class="row-label" scope="row" {...{ "aria-description": (a) }}>{a}</th>
                {#each names as b (b)}
                  {@const v = data.matrix[a]?.[b] ?? null}
                  <td class="cell-td">
                    <button class="matrix-pick" type="button" onpointerenter={() => focusedCell = cellTitle(a, b, v)} onfocus={() => focusedCell = cellTitle(a, b, v)} onclick={() => focusedCell = cellTitle(a, b, v)} aria-label={cellTitle(a, b, v)}>
                    <HeatmapCell
                      value={v}
                      size={CELL_SIZE}
                      title={cellTitle(a, b, v)}
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
  /* A square of cosines between saved directions: labels in the text face, the cell in view named
   * above the grid. */
  .drawer { display: flex; flex-direction: column; gap: var(--s4); height: 100%; min-height: 0; padding: var(--s5) var(--s6) 0; color: var(--ink2); }
  .drawer-header { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--s5); }
  .title { display: grid; gap: var(--s1); min-width: 0; }
  .actions { display: flex; flex: none; align-items: center; gap: var(--s4); }
  .omissions { margin: 0; }
  .omitted { color: var(--ink2); }
  .body { flex: 1 1 auto; min-height: 0; overflow: auto; padding-bottom: var(--s6); }
  .empty { color: var(--ink3); font: var(--type-small); }
  .focused-cell { min-height: 20px; margin: 0 0 var(--s3); color: var(--ink); font: var(--type-small); overflow-wrap: anywhere; }
  .matrix-pick { display: grid; place-items: center; min-width: 44px; min-height: 44px; padding: 0; border: 0; background: transparent; cursor: crosshair; }
  .grid-scroll { max-height: 100%; overflow: auto; }
  .grid { border-collapse: separate; border-spacing: 1px; font-variant-numeric: tabular-nums; }
  .grid th, .grid td { margin: 0; padding: 0; background: var(--paper); }
  .grid thead th { position: sticky; top: 0; z-index: 2; }
  .grid .row-label { position: sticky; inset-inline-start: 0; z-index: 1; padding: 0 var(--s3) 0 var(--s2); color: var(--ink2); font: var(--type-meta); text-align: end; white-space: nowrap; }
  .grid .corner { position: sticky; top: 0; inset-inline-start: 0; z-index: 3; }
  .grid .col-label { width: var(--cell); min-width: var(--cell); max-width: var(--cell); height: 112px; padding: 0; color: var(--ink2); font: var(--type-meta); vertical-align: bottom; }
  .grid .col-label > span { display: inline-block; padding-bottom: var(--s2); white-space: nowrap; transform: rotate(-60deg); transform-origin: left bottom; }
  .grid .cell-td { line-height: 0; }
  @media (max-width: 720px) { .drawer { padding: var(--s4) var(--s4) 0; } }
  @media (pointer: coarse) { .dw-text { min-height: 44px; } }
</style>
