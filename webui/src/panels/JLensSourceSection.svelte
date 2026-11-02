<script lang="ts">
  // The Lens room's four lenses (the Lens canvas's 2x2): the S family beside the R family, each
  // with its selected source at the out and mid sites, its sources and a local fit. drowse.ai swaps
  // this module for its hosted source section through the @runtime-jlens-source alias, so the fit
  // and fetch jobs (instrumentAuthoring) are imported only here.
  import { onMount } from "svelte";
  import LensFamily from "./rooms/lens/LensFamily.svelte";
  import { layerCount, sharedScale } from "../lib/lensRoom";
  import { runtimeOperationAvailability } from "../lib/runtime/ui-capabilities";
  import { instrumentFamily, lensFetch, lensSourceState, refreshLensSources } from "../lib/stores.svelte";
  import { lensFit } from "../lib/stores/instrumentAuthoring.svelte";

  const n = $derived(layerCount(lensSourceState.band, lensSourceState.sources));
  /** One scale for every card: the largest split-half difference among the selected sources. */
  const scale = $derived(sharedScale(lensSourceState.sources.filter(row => row.active === true)));

  onMount(() => {
    if (runtimeOperationAvailability("jlens_fitting").available) void lensFit.check();
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
  <LensFamily family="S" {n} {scale} />
  <LensFamily family="R" {n} {scale} />
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
