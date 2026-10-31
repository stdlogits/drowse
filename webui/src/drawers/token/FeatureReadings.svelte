<script lang="ts">
  // The SAE features that fired at this token, as rows: the feature as the data writes it (its layer
  // and index), its label, a meter on its layer's tint, and its activation (relative to the
  // feature's reference maximum when the pack has one, else raw on this readout's own scale). A row
  // opens to the full label, the activation and maximum, and the published record. Labels missing
  // from the pack load from the published Neuronpedia records on request; only feature ids leave.
  import { Meter, fadeIn } from "@stdlogits/ledger";
  import { onMount, untrack } from "svelte";
  import { refreshSaeSources, saeSourceState } from "../../lib/stores.svelte";
  import { goto } from "../../lib/stores/rooms.svelte";
  import { loadSaeDescription, type SaeDescription, type SaeDescriptionSource } from "../../lib/saeDescriptions";
  import type { SaeFeatureJSON, SaeTokenReadoutJSON } from "../../lib/types";
  import type { ReplayReadout } from "./readout.svelte";
  import type { CatalogInstrumentAvailability } from "../../lib/runtime/instrumentAvailability";
  import ReadingWait from "./ReadingWait.svelte";
  import { depthTint } from "./layers";

  let { readout, saeLoaded, availability, hasReplayContext, replayAvailable, lastLayer }: {
    readout: ReplayReadout<SaeTokenReadoutJSON>;
    saeLoaded: boolean;
    availability: CatalogInstrumentAvailability;
    hasReplayContext: boolean;
    replayAvailable: boolean;
    /** The model's deepest layer when known, for the SAE layer's place on the ramp. */
    lastLayer: number | null;
  } = $props();

  onMount(() => { void refreshSaeSources(); });
  const data = $derived(readout.data);
  const sourceInfo = $derived(saeSourceState.sources.find(source => source.source === readout.source && source.layer === data?.layer));
  const descriptionSource = $derived(sourceInfo?.description_source as SaeDescriptionSource | null | undefined);
  const descriptionIdentity = $derived(JSON.stringify(descriptionSource ?? null));
  const readoutIdentity = $derived(`${data?.node_id}/${data?.raw_index}/${readout.source}/${data?.layer}/${data?.features.map(feature => feature.id).join(",")}`);
  let query = $state("");
  let byId = $state(false);
  let open = $state<number | null>(null);
  let descriptions = $state<Record<number, SaeDescription>>({});
  let descriptionsLoading = $state(false);
  let descriptionsError = $state("");
  let metadataController: AbortController | null = null;
  const missingDescriptions = $derived((data?.features ?? []).some(feature => !feature.label?.trim() && !descriptions[feature.id]));
  const published = $derived(Object.keys(descriptions).length > 0);

  $effect(() => {
    void readoutIdentity;
    void descriptionIdentity;
    descriptions = {};
    descriptionsError = "";
    descriptionsLoading = false;
    query = "";
    open = null;
    untrack(() => { void loadDescriptions(); });
    return () => { metadataController?.abort(); metadataController = null; };
  });

  async function loadDescriptions(): Promise<void> {
    const binding = descriptionSource;
    if (!binding || !data || descriptionsLoading) return;
    const controller = new AbortController();
    metadataController = controller;
    descriptionsLoading = true;
    descriptionsError = "";
    const timeout = setTimeout(() => controller.abort(), 20_000);
    const features = data.features.filter(feature => !feature.label?.trim() && !descriptions[feature.id]);
    try {
      for (let start = 0; start < features.length; start += 4) {
        const results = await Promise.allSettled(features.slice(start, start + 4).map(async feature =>
          [feature.id, await loadSaeDescription(binding, feature.id, controller.signal)] as const));
        if (metadataController !== controller) return;
        for (const result of results) {
          if (result.status === "fulfilled") descriptions = { ...descriptions, [result.value[0]]: result.value[1] };
          else descriptionsError = "Some labels did not load";
        }
        if (controller.signal.aborted) break;
      }
    } catch {
      if (metadataController === controller) descriptionsError = "Some labels did not load";
    } finally {
      clearTimeout(timeout);
      if (metadataController === controller) descriptionsLoading = false;
    }
  }

  /** Raw activations without a reference maximum share this readout's largest as their unit. */
  const rawScale = $derived(Math.max(1, ...(data?.features ?? []).filter(f => !(f.max_act != null && f.max_act > 0)).map(f => f.activation)));
  function relative(f: SaeFeatureJSON): number | null {
    return f.max_act != null && f.max_act > 0 ? f.activation / f.max_act : null;
  }
  function labelOf(f: SaeFeatureJSON): string {
    return f.label?.trim() || descriptions[f.id]?.label || "";
  }

  const visible = $derived.by(() => {
    const text = query.trim().toLowerCase();
    return [...(data?.features ?? [])]
      .filter(f => !text || `f${f.id} sae/${f.id} ${labelOf(f)}`.toLowerCase().includes(text))
      .sort((a, b) => byId ? a.id - b.id : b.activation - a.activation);
  });
  const many = $derived((data?.features.length ?? 0) > 8);
  const tint = $derived(data && data.layer >= 0 ? depthTint(data.layer, lastLayer ?? data.layer) : undefined);
</script>

<section class="dw-sec" aria-label="Features" style:--tint={tint}>
  <div class="dw-sec-head">
    <span class="head">Features</span>
    <span class="meta">
      {#if readout.origin === "replayed"}replayed{/if}
      {#if published}{readout.origin === "replayed" ? " · " : ""}labels from Neuronpedia{/if}
      {#if !data && saeLoaded && !replayAvailable}<span data-tip="Features recorded during generation still show">not replayable here</span>{/if}
    </span>
  </div>

  {#if readout.loading}
    <ReadingWait progress={readout.progress} label="Feature readout" blocks={[44, 44, 44]} />
  {:else if readout.error}
    <p class="meta" role="alert">{readout.error}</p>
  {:else if data}
    {#if many}
      <div class="tools">
        <label class="sr-only" for="token-feature-find">Find a feature</label>
        <input id="token-feature-find" type="search" placeholder="Find a feature" bind:value={query} autocomplete="off" />
        <button type="button" class="dw-text" aria-pressed={byId} onclick={() => (byId = !byId)}>{byId ? "By index" : "By activation"}</button>
      </div>
    {/if}
    {#if data.features.length === 0}
      <p class="meta">None fired here</p>
    {:else}
      <div class="rows" role="list">
        {#each visible as feature, i (feature.id)}
          {@const share = relative(feature)}
          {@const label = labelOf(feature)}
          {@const record = descriptions[feature.id]}
          <div role="listitem" class="item rise" style:--i={i}>
            <button type="button" class="row" aria-expanded={open === feature.id} onclick={() => (open = open === feature.id ? null : feature.id)}>
              <span class="uid">L{data.layer} f{feature.id}</span>
              {#key label}<span class="label" class:none={!label} in:fadeIn>{label || (descriptionsLoading ? "…" : "")}</span>{/key}
              <Meter value={share ?? Math.max(0, feature.activation) / rawScale} label={share != null ? `relative ${share.toFixed(2)}` : `activation ${feature.activation.toFixed(2)}`} width={56} />
              <span class="val">{share != null ? share.toFixed(2) : feature.activation.toFixed(1)}</span>
            </button>
            {#if open === feature.id}
              <div class="detail">
                {#if label}<p class="full">{label}</p>{/if}
                <div class="facts meta">
                  <span>activation <b>{feature.activation.toFixed(3)}</b></span>
                  {#if share != null}<span data-tip="The reference maximum this pack gives the feature">of <b>{feature.max_act?.toFixed(3)}</b></span>{/if}
                  {#if record}<a class="prose-link" href={record.url} target="_blank" rel="noreferrer">Neuronpedia{record.explanationModel ? ` · ${record.explanationModel}` : ""}</a>{/if}
                </div>
              </div>
            {/if}
          </div>
        {/each}
      </div>
      {#if visible.length === 0}<button type="button" class="dw-text" onclick={() => (query = "")}>Clear “{query}”</button>{/if}
    {/if}
    {#if descriptionSource && missingDescriptions && !descriptionsLoading}
      <button type="button" class="dw-text more" data-tip="Looks up published labels by feature id; your text and activations stay here" onclick={loadDescriptions}>{descriptionsError ? "Retry labels" : "Load labels"}</button>
    {:else if !descriptionSource && saeSourceState.error}
      <button type="button" class="dw-text more" onclick={() => void refreshSaeSources()}>Retry the label source</button>
    {/if}
    {#if descriptionsError}<p class="meta" role="alert">{descriptionsError}</p>{/if}
  {:else if !saeLoaded}
    {#if availability === "unavailable"}
      <p class="meta">None for this model</p>
    {:else if availability === "context-unavailable"}
      <p class="meta">None for this length</p>
    {:else}
      <div class="dw-row"><button type="button" class="dw-quiet" onclick={() => goto("lens")}>Choose an SAE</button></div>
    {/if}
  {:else if replayAvailable && !hasReplayContext}
    <p class="meta">No replay record for this token</p>
  {/if}
</section>

<style>
  .tools { display: flex; align-items: center; gap: var(--s3); min-width: 0; }
  .tools input { flex: 1 1 auto; min-width: 0; background: var(--token-fill); }
  .rows { display: grid; gap: 2px; }
  .item { display: grid; min-width: 0; }
  .row {
    display: grid;
    grid-template-columns: 96px minmax(0, 1fr) 56px 40px;
    gap: var(--s3);
    align-items: center;
    width: 100%;
    min-height: 40px;
    padding: var(--s2) var(--s3);
    border: 0;
    border-radius: var(--r-box);
    background: transparent;
    color: inherit;
    font: inherit;
    text-align: left;
    cursor: pointer;
    transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease);
  }
  @media (hover: hover) { .row:hover { background: var(--token-fill); } }
  .row[aria-expanded="true"] { background: var(--token-fill); }
  .row:active { scale: 0.99; }
  .uid { color: var(--ink); font: 600 14px/20px var(--display); letter-spacing: -0.01em; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .label { overflow: hidden; color: var(--ink2); font: var(--type-small); white-space: nowrap; text-overflow: ellipsis; }
  .val { color: var(--ink2); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; }
  .detail { display: grid; gap: var(--s2); padding: var(--s2) var(--s3) var(--s4); }
  .full { color: var(--ink); font: var(--type-small); }
  .facts { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s1) var(--s4); }
  .facts b { color: var(--ink); font-weight: 500; }
  .more { justify-self: start; }
  @media (max-width: 720px) {
    .row { grid-template-columns: minmax(0, 1fr) 48px 40px; grid-template-areas: "id meter val" "label label label"; min-height: 48px; row-gap: 2px; }
    .uid { grid-area: id; }
    .row > :global(.meter) { grid-area: meter; width: 100% !important; }
    .val { grid-area: val; }
    .label { grid-area: label; }
    .tools input { min-height: 44px; }
    .tools .dw-text, .more { min-height: 44px; }
  }
</style>
