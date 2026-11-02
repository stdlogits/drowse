<script lang="ts">
  // The Lens room's four lenses (the Lens canvas's 2x2): the S family beside the R family, each a
  // column with its selected source at the out and mid sites beside the row labels, then its sources.
  // Both runtimes' source sections render it (the @runtime-jlens-source alias): drowse's own passes
  // `column` (LensFit, a family with the local fit) and `check` (the fit's status check); drowse.ai's
  // hosted one renders the plain LensFamily columns with no props, since its runtime fits nothing.
  // Neither this module nor LensFamily imports the fit job (lib/stores/instrumentAuthoring) or
  // carries the fit's markup.
  import { onMount, type Component } from "svelte";
  import LensFamily from "./LensFamily.svelte";
  import { layerCount, sharedScale, type LensFamily as Family } from "../../../lib/lensRoom";
  import { instrumentFamily, lensFetch, lensSourceState, refreshLensSources } from "../../../lib/stores.svelte";

  type Column = Component<{ family: Family; n: number; scale: number | null }>;
  /** `check` is the caller's own mount-time status check, run first: the fit's status, then the
   *  fetch's, then the sources. */
  let { column: FamilyColumn = LensFamily, check }: { column?: Column; check?: () => void } = $props();

  const n = $derived(layerCount(lensSourceState.band, lensSourceState.sources));
  /** One scale for every card: the largest split-half difference among the selected sources. */
  const scale = $derived(sharedScale(lensSourceState.sources.filter(row => row.active === true)));

  onMount(() => {
    check?.();
    if (instrumentFamily("lens")?.capabilities.preparations.includes("fetch") === true) void lensFetch.check();
    void refreshLensSources();
  });
</script>

<section class="quad" aria-label="The four lenses">
  <div class="rowlabs" aria-hidden="true">
    <span></span>
    <span class="rowlab" data-tip="neuron outputs, attention inputs">Out</span>
    <span class="rowlab" data-tip="neuron inputs, attention outputs">Mid</span>
    <span></span>
  </div>
  <FamilyColumn family="S" {n} {scale} />
  <FamilyColumn family="R" {n} {scale} />
</section>
{#if lensSourceState.error}<p class="meta" role="alert">{lensSourceState.error}</p>{/if}

<style>
  .quad { display: grid; grid-template-columns: 64px repeat(2, minmax(0, 1fr)); grid-template-rows: repeat(4, auto); gap: var(--s3) var(--s5); min-width: 0; }
  .rowlabs { grid-row: 1 / span 4; display: grid; grid-template-rows: subgrid; min-width: 0; }
  .rowlab { padding-top: var(--s4); color: var(--ink); font: var(--type-control); cursor: default; }
  @media (max-width: 720px) {
    .quad { grid-template-columns: minmax(0, 1fr); grid-template-rows: none; gap: var(--s7); }
    .rowlabs { display: none; }
  }
</style>
