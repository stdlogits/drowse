<script lang="ts">
  // The Artifacts room's Units section (#/artifacts/units): the model's neurons and heads and the
  // neuron-lens catalog that labels them (the labels the Units room, the unit view and the token
  // anatomy show). A catalogued model fetches its labels here, the units ``fetch`` preparation at
  // the catalog's pinned commit; the Units room then reads them.
  import { MorphText } from "@stdlogits/ledger";
  import { goto } from "../../../lib/stores/rooms.svelte";
  import { runtimeClient } from "../../../lib/runtime/client";
  import { sessionState } from "../../../lib/stores/session.svelte";
  import { fetchCatalog, loadCatalog, unitsState } from "../../../lib/stores/units.svelte";

  const readable = typeof runtimeClient.instruments.unitCatalog === "function";
  $effect(() => {
    if (readable && !unitsState.catalogRead) void loadCatalog();
  });
  const catalog = $derived(unitsState.catalog);
  const bound = $derived(catalog?.bound === true);
  const canFetch = $derived(catalog?.available === true && !bound);
  const binding = $derived(bound ? catalog?.binding ?? null : null);
  const loading = $derived(readable && !unitsState.catalogRead);
  const shape = $derived(sessionState.info?.architecture ?? null);
  const count = (value: number) => value.toLocaleString("en-US");
  const status = $derived(loading ? "" : bound ? "labelled" : canFetch ? "labels to fetch" : "no labels");
  const fetched = $derived(binding?.fetched_at
    ? new Date(binding.fetched_at).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })
    : null);
  const revision = $derived((binding?.revision ?? catalog?.revision ?? "").slice(0, 7));
  const detail = $derived([
    revision ? `at ${revision}` : "",
    binding ? `${count(binding.files)} files` : "",
    fetched ? `fetched ${fetched}` : "",
  ].filter(Boolean).join(" · "));
</script>

<section class="dw-sec panel" aria-labelledby="artifact-units-head">
  <div class="dw-sec-head"><span class="head" id="artifact-units-head">Unit catalog</span>{#if status}<span class="meta">{status}</span>{/if}</div>
  {#if shape}
    <div class="dw-facts">
      {#if shape.neurons_per_layer !== null}<span><b>{count(shape.neurons_per_layer * shape.layers)}</b> neurons</span>{/if}
      {#if shape.heads_per_layer !== null}<span><b>{count(shape.heads_per_layer * shape.layers)}</b> heads</span>{/if}
      <span><b>{count(shape.layers)}</b> layers</span>
    </div>
  {/if}
  {#if catalog && (bound || canFetch)}
    <div class="row">
      <span class="top"><span class="uid">{catalog.slug ?? catalog.repo}</span><span class="what">{catalog.repo}</span></span>
      {#if detail}<span class="meta">{detail}</span>{/if}
      {#if bound && catalog.credit}<span class="meta">{catalog.credit}</span>{/if}
    </div>
  {:else if catalog?.reason}
    <p class="meta">{catalog.reason}</p>
  {:else if loading}
    <div class="ghost" aria-hidden="true"></div>
  {/if}
  <div class="dw-row">
    {#if canFetch}
      <button type="button" class="dw-button" disabled={unitsState.fetching} onclick={() => void fetchCatalog()}>
        <MorphText text={unitsState.fetching ? "Fetching labels…" : "Fetch labels"} />
      </button>
    {/if}
    <button type="button" class="dw-quiet" onclick={() => goto("units")}>Read units</button>
  </div>
</section>

<style>
  .panel { max-width: 760px; }
  .row { display: grid; gap: 2px; min-width: 0; padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .top { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s1) var(--s2); min-width: 0; }
  .uid { color: var(--ink); font: var(--type-control); overflow-wrap: anywhere; }
  .what { color: var(--ink2); font: var(--type-small); overflow-wrap: anywhere; }
  .row .meta { overflow-wrap: anywhere; }
  .ghost { height: 56px; border-radius: var(--r-box); background: var(--wash); }
</style>
