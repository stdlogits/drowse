<script lang="ts">
  // One lens at one site (a cell of the Lens canvas's 2x2): the family's selected source, the
  // layers it holds and its split-half difference at each, drawn directly (0 is perfect agreement,
  // so weak layers stand tall) on the depth ramp and faded by how much the family is read there,
  // over a strip of that weight. Imports record no split-half, so their held layers draw as a
  // baseline. Hovering a column names its layer, difference and weight.
  import { MorphText } from "@stdlogits/ledger";
  import { depthTint } from "../../../lib/charts/depth";
  import {
    coveredText,
    documentsText,
    foreignKind,
    layerAt,
    originText,
    rangeText,
    siteSeries,
    spanText,
    unloaded,
    type LensFamily,
    type LensSite,
    type Span,
  } from "../../../lib/lensRoom";
  import type { InstrumentSourceJSON, LensProviderJSON } from "../../../lib/types";

  let { family, site, row, provider, n, weights, span, scale }: {
    family: LensFamily;
    site: LensSite;
    row: InstrumentSourceJSON | null;
    provider: LensProviderJSON | null;
    n: number;
    weights: number[] | null;
    span: Span | null;
    scale: number | null;
  } = $props();

  let hover = $state(-1);
  const series = $derived(siteSeries(row, site, n));
  const held = $derived(series.covered.some(Boolean));
  // A local fit's name already says so ("local:r"); a provider's names its repository.
  // A selected source that does not load for these weights says so first: it reads nothing.
  const meta = $derived([unloaded(row) ? "not loaded for this model" : "", row?.kind === "local" ? "" : originText(row, provider), foreignKind(row?.lens, family), documentsText(row?.n_prompts)].filter(Boolean).join(" · "));
  const where = $derived(span ? `${site} · used ${spanText(span)}` : site);

  const cols = $derived(series.covered.map((covered, layer) => {
    const w = weights?.[layer] ?? 1;
    const value = series.split[layer];
    const tint = depthTint(n > 1 ? layer / (n - 1) : 0);
    return {
      c: covered ? tint : "var(--wash2)",
      h: covered && value !== null && scale ? `${Math.min(100, (Math.abs(value) / scale) * 100).toFixed(1)}%` : "0%",
      o: covered ? (0.25 + 0.75 * w).toFixed(2) : "1",
      use: weights ? `color-mix(in oklch, ${tint} ${Math.round(w * 100)}%, var(--wash2))` : null,
    };
  }));

  const chartMeta = $derived.by(() => {
    if (!row) return "";
    if (!held) return `no ${site} site`;
    if (hover >= 0 && hover < n) {
      const value = series.split[hover];
      return [
        `L${hover}`,
        !series.covered[hover] ? "not held" : value !== null ? `difference ${value.toFixed(2)}` : "",
        weights ? `weight ${weights[hover].toFixed(2)}` : "",
      ].filter(Boolean).join(" · ");
    }
    return series.measured ? `split-half difference ${rangeText(series.split)}` : `${coveredText(series)} · no split-half`;
  });
  const label = $derived(!row ? `No ${family}-lens` : !held ? `No ${site} site`
    : series.measured ? `Split-half difference by layer, ${rangeText(series.split).replace("–", " to ")}` : `Holds ${coveredText(series)}`);
</script>

<div class="card">
  <div class="card-top">
    <div class="id"><span class="name" class:none={!row}>{row?.source ?? "None"}</span>{#if meta}<span class="meta">{meta}</span>{/if}</div>
    <span class="meta">{where}</span>
  </div>
  <div class="chart">
    <div class="bars" role="img" aria-label={label} onpointerover={(e) => (hover = layerAt(e))} onpointerleave={() => (hover = -1)}>
      {#each cols as col, layer (layer)}
        <span class="col" style:--lens-fill={col.c} style:--h={col.h} style:--o={col.o} data-layer={layer}><i></i></span>
      {/each}
    </div>
    {#if weights}
      <div class="use" aria-hidden="true">{#each cols as col, layer (layer)}<i style:background={col.use}></i>{/each}</div>
    {/if}
    <div class="ends"><span>L0</span><span class="mid"><MorphText text={chartMeta} /></span><span>L{Math.max(0, n - 1)}</span></div>
  </div>
</div>

<style>
  .card { display: grid; gap: var(--s3); align-content: start; min-width: 0; padding: var(--s4); border-radius: var(--r-row); background: var(--wash); }
  .card-top { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--s1) var(--s3); }
  .id { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s1) var(--s3); min-width: 0; }
  .name { color: var(--ink); font: 600 16px/24px var(--display); letter-spacing: -0.01em; overflow-wrap: anywhere; }
  .name.none { color: var(--ink3); }

  .chart { display: grid; gap: var(--s1); }
  .bars { display: flex; align-items: stretch; height: 44px; }
  .col { flex: 1 1 0; display: flex; align-items: flex-end; height: 100%; padding: 0 1px; }
  .col > i { display: block; width: 100%; min-height: 2px; height: var(--h); border-radius: 2px; background: var(--lens-fill); opacity: var(--o); transition: opacity var(--fast) var(--ease), height var(--glide) var(--ease); }
  .bars:hover .col > i { opacity: calc(var(--o) * 0.4); }
  .bars .col:hover > i { opacity: 1; }
  .use { display: flex; height: 4px; }
  .use > i { flex: 1 1 0; margin: 0 1px; border-radius: 1px; }
  .ends { display: flex; justify-content: space-between; gap: var(--s2); color: var(--ink3); font: var(--type-meta); font-variant-numeric: tabular-nums; }
  .ends .mid { min-width: 0; overflow: hidden; text-align: center; white-space: nowrap; text-overflow: ellipsis; }
</style>
