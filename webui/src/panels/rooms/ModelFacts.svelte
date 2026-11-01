<script lang="ts">
  // The Model room's own section (the Phone canvas's Model view), under the room's facts line (its
  // layers, neurons and heads a layer): the depth ramp, the lenses prepared for the model and a way
  // to fit one. The shape is the server's (session info's ``architecture``, read from the config at
  // load); a runtime that does not describe its model leaves the ramp out. The lenses are the Lens
  // room's sources.
  import { onMount } from "svelte";
  import { depthTint } from "../../lib/charts/depth";
  import { documentsText, lensKindName, originText, rangeText, siteSeries, unloaded } from "../../lib/lensRoom";
  import { goto } from "../../lib/stores/rooms.svelte";
  import { instrumentFamily, lensSourceState, refreshLensSources } from "../../lib/stores.svelte";
  import { sessionState } from "../../lib/stores/session.svelte";

  const info = $derived(sessionState.info);
  const shape = $derived(info?.architecture ?? null);
  const n = $derived(shape?.layers ?? 0);
  const count = (value: number) => value.toLocaleString("en-US");
  const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
  const lenses = $derived(lensSourceState.sources.filter(row => row.family === "R" || row.family === "S"));
  const canFit = $derived(instrumentFamily("lens")?.capabilities.preparations.includes("fit") === true);

  function lensMeta(row: (typeof lenses)[number]): string {
    const provider = lensSourceState.providers.find(p => p.source === row.source) ?? null;
    const split = rangeText([...siteSeries(row, "out", n).split, ...siteSeries(row, "mid", n).split]);
    return [
      unloaded(row) ? "not loaded for this model" : row.active ? "in use" : "",
      originText(row, provider),
      documentsText(row.n_prompts),
      split ? `split-half ${split}` : "",
    ].filter(Boolean).join(" · ");
  }

  onMount(() => {
    if (lensSourceState.sources.length === 0) void refreshLensSources();
  });
</script>

{#if shape}
  <section class="dw-sec" aria-labelledby="model-depth-head">
    <div class="dw-sec-head"><span class="head" id="model-depth-head">Depth</span><span class="meta">{count(shape.hidden_size)} wide · {compact.format(shape.parameters)} parameters</span></div>
    <div class="depth" role="img" aria-label={`${shape.layers} layers, from L0 to L${shape.layers - 1}`}>
      <div class="ramp" style:--n={n}>
        {#each { length: n } as _, layer (layer)}<i style:background={depthTint(n > 1 ? layer / (n - 1) : 0)}></i>{/each}
      </div>
      <div class="ends meta"><span>L0</span><span>L{n - 1}</span></div>
    </div>
  </section>
{/if}

<section class="dw-sec" aria-labelledby="model-lens-head">
  <div class="dw-sec-head"><span class="head" id="model-lens-head">Lenses</span>{#if lenses.length > 0}<span class="meta">{lenses.length === 1 ? "one prepared" : `${lenses.length} prepared`}</span>{/if}</div>
  {#if lenses.length > 0}
    <div class="lenses">
      {#each lenses as row (row.source)}
        <div class="lens">
          <span class="top"><span class="uid">{row.source}</span><span class="what">{lensKindName(row.lens ?? row.family)}, {row.family === "S" ? "input side" : "output side"}</span></span>
          <span class="meta">{lensMeta(row)}</span>
        </div>
      {/each}
    </div>
  {:else}
    <p class="meta">{lensSourceState.loading ? "Looking for lenses" : "No lens is prepared for this model yet."}</p>
  {/if}
  {#if canFit}
    <div class="dw-row">
      <button type="button" class="dw-quiet" onclick={() => goto("lens")}>Fit a lens here</button>
      {#if lensSourceState.fitDocuments}<span class="meta">{documentsText(lensSourceState.fitDocuments)}</span>{/if}
    </div>
  {/if}
</section>

<style>
  .depth { display: grid; gap: var(--s1); max-width: 720px; }
  .ramp { display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: 2px; height: 24px; }
  .ramp > i { display: block; border-radius: 2px; }
  .ends { display: flex; justify-content: space-between; }
  .lenses { display: grid; gap: 2px; max-width: 720px; }
  .lens { display: grid; gap: 2px; min-width: 0; padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .top { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s1) var(--s2); min-width: 0; }
  .uid { color: var(--ink); font: var(--type-control); overflow-wrap: anywhere; }
  .what { color: var(--ink2); font: var(--type-small); }
  .lens .meta { overflow-wrap: anywhere; }
</style>
