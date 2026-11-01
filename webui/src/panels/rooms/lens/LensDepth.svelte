<script lang="ts">
  // Which lens reads each depth (the Lens canvas's first section): one column per layer, the S
  // family's weight over the R family's, drawn on the depth ramp, with the crossover band where
  // both read on a wash behind them. The weights are the server's (lib/lens/depth.py); hovering a
  // column names its layer and the two weights.
  import { MorphText } from "@stdlogits/ledger";
  import { depthTint } from "../../../lib/charts/depth";
  import { crossover, familySpan, familyWeights, layerAt, spanText } from "../../../lib/lensRoom";
  import type { LensBandJSON } from "../../../lib/types";

  let { band, model }: { band: LensBandJSON; model: string } = $props();

  const n = $derived(band.weights.length);
  const s = $derived(familyWeights(band, "S") ?? []);
  const r = $derived(familyWeights(band, "R") ?? []);
  const cross = $derived(crossover(band));
  const sSpan = $derived(familySpan(band, "S"));
  const rSpan = $derived(familySpan(band, "R"));
  let hover = $state(-1);

  const meta = $derived(hover >= 0 && hover < n
    ? `L${hover} · S ${s[hover].toFixed(2)} · R ${r[hover].toFixed(2)}`
    : `${model} · ${n} layers`);
  const label = $derived([
    sSpan ? `S-lens reads ${spanText(sSpan)}` : "",
    rSpan ? `R-lens ${spanText(rSpan)}` : "",
    cross ? `both, weighted, ${spanText(cross)}` : "",
  ].filter(Boolean).join(", "));
  const tint = (layer: number) => depthTint(n > 1 ? layer / (n - 1) : 0);
</script>

<section class="dw-sec" aria-labelledby="lens-depth-head">
  <div class="dw-sec-head"><span class="head" id="lens-depth-head">Which lens, by depth</span><span class="meta"><MorphText text={meta} /></span></div>
  <div class="axis model" style:--n={n} role="img" aria-label={label} onpointerover={(e) => (hover = layerAt(e))} onpointerleave={() => (hover = -1)}>
    {#if cross}<span class="band" style:grid-column={`${cross.first + 2} / span ${cross.last - cross.first + 1}`}></span>{/if}
    <span class="rl" style:grid-row="1">S-lens</span>
    <span class="rl" style:grid-row="2">R-lens</span>
    {#each s as ws, layer (layer)}
      <span class="mcol" style:grid-column={`${layer + 2} / span 1`} style:--lens-fill={tint(layer)} data-layer={layer}>
        <span><i class:off={ws <= 0} style:height={ws > 0 ? `${(ws * 100).toFixed(1)}%` : null}></i></span>
        <span><i class:off={r[layer] <= 0} style:height={r[layer] > 0 ? `${(r[layer] * 100).toFixed(1)}%` : null}></i></span>
      </span>
    {/each}
  </div>
  <div class="axis ends" style:--n={n}>
    <span class="end-first">L0</span>
    {#if cross}<span class="end-band" style:grid-column={`${cross.first + 2} / span ${cross.last - cross.first + 1}`}>{spanText(cross)} both, weighted</span>{/if}
    <span class="end-last">L{n - 1}</span>
  </div>
</section>

<style>
  .axis { display: grid; grid-template-columns: 64px repeat(var(--n), minmax(0, 1fr)); min-width: 0; }
  .model { grid-template-rows: 32px 32px; row-gap: 6px; }
  .band { grid-row: 1 / span 2; margin: -6px 0; border-radius: var(--r-col); background: var(--wash); }
  .rl { grid-column: 1; align-self: center; color: var(--ink2); font: var(--type-small); white-space: nowrap; }
  .mcol { grid-row: 1 / span 2; display: grid; grid-template-rows: 32px 32px; row-gap: 6px; padding: 0 1px; transition: opacity var(--fast) var(--ease); }
  .mcol > span { display: flex; align-items: flex-end; }
  .mcol i { display: block; width: 100%; border-radius: 2px; background: var(--lens-fill); transition: height var(--glide) var(--ease); }
  .mcol i.off { height: 2px; background: var(--wash2); }
  .model:hover .mcol { opacity: 0.4; }
  .model .mcol:hover { opacity: 1; }

  .ends { color: var(--ink3); font: var(--type-meta); font-variant-numeric: tabular-nums; }
  .end-first { grid-column: 2 / span 4; }
  .end-band { grid-row: 1; text-align: center; white-space: nowrap; }
  .end-last { grid-column: -6 / -1; grid-row: 1; text-align: right; }

  @media (max-width: 720px) {
    .axis { grid-template-columns: 48px repeat(var(--n), minmax(0, 1fr)); }
    /* Too narrow to sit the band's label under the band: the three ends spread along the row. */
    .ends { display: flex; justify-content: space-between; gap: var(--s2); padding-left: 48px; }
  }
</style>
