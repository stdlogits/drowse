<script lang="ts">
  // One lens family as a column of the Lens canvas's 2x2 (S early, R late): its selected source at
  // each site, then its sources (choose one; fetch a provider not yet here). The column's rows sit in
  // the parent's grid (LensQuad) so the two families' out and mid cards line up beside the row labels.
  // A runtime with a local fit passes it as `fit` (drowse's LensFit), which the column places in
  // three parts: a running fit among the sources, a failed fit's error under them, the fit's form
  // after them. This module imports none of the fit, so drowse.ai's hosted build carries none of it.
  import type { Snippet } from "svelte";
  import { MorphText } from "@stdlogits/ledger";
  import LensCard from "./LensCard.svelte";
  import {
    documentsText,
    familyItems,
    familySpan,
    familyWeights,
    foreignKind,
    rangeText,
    siteSeries,
    sitesText,
    sourceFamily,
    unloaded,
    type FitPart,
    type LensFamily,
  } from "../../../lib/lensRoom";
  import { instrumentFamily, lensFetch, lensSourceState, useLensSource } from "../../../lib/stores.svelte";

  let { family, n, scale, fitRunning = false, fit }: {
    family: LensFamily;
    n: number;
    scale: number | null;
    /** A local fit is running, for either family. */
    fitRunning?: boolean;
    /** The local fit's parts, where the runtime has one. */
    fit?: Snippet<[FitPart]>;
  } = $props();

  const allowFetch = $derived(instrumentFamily("lens")?.capabilities.preparations.includes("fetch") === true);
  const allowSwitch = $derived(instrumentFamily("lens")?.capabilities.source_switch === true);

  const weights = $derived(familyWeights(lensSourceState.band, family));
  const span = $derived(familySpan(lensSourceState.band, family));
  const items = $derived(familyItems(lensSourceState.sources, lensSourceState.providers, family)
    .filter(item => allowSwitch || item.row?.active === true || (item.row === null && allowFetch)));
  const active = $derived(lensSourceState.sources.find(row => row.family === family && row.active === true) ?? null);
  const activeProvider = $derived(lensSourceState.providers.find(row => row.source === active?.source) ?? null);

  // One lens preparation runs at a time (the server's lens group), whichever family it is for.
  const busy = $derived(lensSourceState.busy || lensFetch.state.running || fitRunning);
  const ofFamily = (source: string | null | undefined) =>
    sourceFamily(source, lensSourceState.sources, lensSourceState.providers) === family;
  const fetching = $derived(lensFetch.state.running ? (lensFetch.state.latest?.source ?? null) : null);
  const fetchError = $derived(ofFamily(lensFetch.state.latest?.source) ? lensFetch.state.error : null);

  function itemMeta(item: (typeof items)[number]): string {
    if (unloaded(item.row)) return "not loaded for this model";
    if (item.row) return [foreignKind(item.row.lens, family), documentsText(item.row.n_prompts), sitesText(item.row.sites)].filter(Boolean).join(" · ");
    const provider = item.provider;
    return provider ? [foreignKind(provider.lens, family), sitesText(provider.sites), provider.repo_id].filter(Boolean).join(" · ") : "";
  }
</script>

<div class="fam">
  <div class="colhead"><h2 class="h">{family}-lens</h2><span class="meta">{family === "S" ? "early layers" : "late layers"}</span></div>

  <LensCard {family} site="out" row={active} provider={activeProvider} {n} {weights} {span} {scale} />
  <LensCard {family} site="mid" row={active} provider={activeProvider} {n} {weights} {span} {scale} />

  <div class="foot">
    <div class="dw-sec tight">
      <span class="head">Sources</span>
      <div class="items">
        {#each items as item, i (item.source)}
          {#if item.row && !unloaded(item.row)}
            <button type="button" class="item rise" style:--i={i} aria-pressed={item.row.active === true}
              disabled={busy && item.row.active !== true} onclick={() => { if (!item.row?.active) void useLensSource(item.source); }}>
              <span class="l1">{item.source}</span>
              <span class="meta im">{itemMeta(item)}</span>
              <span class="meta">{rangeText(siteSeries(item.row, "out", n).split)}</span>
            </button>
          {:else}
            <!-- A provider still to fetch, or the selected source when it does not load for these
                 weights (a provider's offers to fetch it again; a local fit's is fitted again below). -->
            <div class="item rise" class:sel={item.row !== null} style:--i={i} aria-current={item.row ? "true" : undefined}>
              <span class="l1" class:todo={!item.row}>{item.source}</span>
              <span class="meta im">{#if fetching === item.source}<MorphText text={lensFetch.state.message ?? "Fetching"} />{:else}{itemMeta(item)}{/if}</span>
              {#if allowFetch && item.provider}
                <button type="button" class="dw-text act" disabled={busy} onclick={() => void lensFetch.start({ source: item.source })}>
                  {fetching === item.source ? "Fetching" : item.row ? "Fetch again" : "Fetch"}
                </button>
              {/if}
            </div>
          {/if}
        {/each}
        {@render fit?.("status")}
      </div>
      {@render fit?.("error")}
      {#if fetchError}<p class="meta err" role="alert">{fetchError}</p>{/if}
    </div>

    {@render fit?.("form")}
  </div>
</div>

<style>
  /* The column's four rows (head, out, mid, foot) are the parent quad's, so cards align across. */
  .fam { grid-row: 1 / span 4; display: grid; grid-template-rows: subgrid; gap: var(--s3); min-width: 0; }
  .colhead { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s1) var(--s3); }
  .h { margin: 0; color: var(--ink); font: var(--type-mark); letter-spacing: -0.01em; }

  .foot { display: grid; gap: var(--s4); align-content: start; min-width: 0; padding-top: var(--s2); }
  .tight { gap: var(--s2); }
  .items { display: grid; gap: 2px; }
  .item {
    display: grid;
    grid-template-columns: 104px minmax(0, 1fr) auto;
    gap: var(--s3);
    align-items: center;
    min-height: 40px;
    padding: var(--s1) var(--s3);
    border: 0;
    border-radius: var(--r-box);
    background: transparent;
    color: inherit;
    font: inherit;
    text-align: left;
    transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease);
  }
  button.item { cursor: pointer; }
  button.item:hover:not(:disabled) { background: var(--wash); }
  .item[aria-pressed="true"], .item[aria-pressed="true"]:hover, .item.sel { background: var(--wash2); cursor: default; }
  button.item:active:not(:disabled) { scale: 0.99; }
  button.item:disabled { cursor: default; }
  button.item:disabled:not([aria-pressed="true"]) { opacity: 0.5; }
  .l1 { overflow: hidden; color: var(--ink); font: var(--type-control); white-space: nowrap; text-overflow: ellipsis; }
  .l1.todo { color: var(--ink2); }
  .im { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  /* A morphing fetch message ellipsizes inside the morph, as plain metadata does. */
  .im :global(.morph-text) { display: block; overflow: hidden; text-overflow: ellipsis; }
  .act { font: var(--type-meta); }
  .err { color: var(--ink2); }

  /* Phone sizes (the Phone canvas): every target 44 px. */
  @media (max-width: 720px) {
    .fam { grid-row: auto; grid-template-rows: none; }
    .item { grid-template-columns: 88px minmax(0, 1fr) auto; min-height: 44px; }
    .act { min-width: 44px; min-height: 44px; }
  }
</style>
