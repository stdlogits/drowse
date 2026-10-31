<script lang="ts">
  // The Inspector with nothing selected: the instruments, as Ledger sections (the Instruments
  // canvas). Probes (flat and curved geometry readings), the live lens readout, SAE features,
  // sampling, and which sources the lens and SAE read through (picked in the Lens room).
  import ProbeRack from "../ProbeRack.svelte";
  import JLensPanel from "../JLensPanel.svelte";
  import SaePanel from "../SaePanel.svelte";
  import SamplingStrip from "../SamplingStrip.svelte";
  import { sessionState } from "../../lib/stores/session.svelte";
  import { goto } from "../../lib/stores/rooms.svelte";

  const sources = $derived.by(() => {
    const blocks = sessionState.info?.instruments ?? [];
    const of = (family: string) => blocks.find(block => block.family === family)?.source ?? null;
    return [
      { key: "Lens", value: of("lens"), meta: sessionState.info?.jlens_fitted ? "fitted" : "" },
      { key: "SAE", value: of("sae"), meta: "" },
    ];
  });
</script>

<div class="instruments">
  <section class="dw-sec" id="instrument-probes" aria-labelledby="instrument-probes-head">
    <div class="dw-sec-head"><span class="head" id="instrument-probes-head">Probes</span></div>
    <div class="racks">
      <div id="instrument-subspace"><ProbeRack family="subspace" /></div>
      <div id="instrument-manifold"><ProbeRack family="manifold" /></div>
    </div>
  </section>

  <section class="dw-sec" id="instrument-lens" aria-labelledby="instrument-lens-head">
    <div class="dw-sec-head"><span class="head" id="instrument-lens-head">Live</span><span class="meta">lens readout</span></div>
    <JLensPanel sources={false} />
  </section>

  <section class="dw-sec" id="instrument-sae" aria-labelledby="instrument-sae-head">
    <div class="dw-sec-head"><span class="head" id="instrument-sae-head">Features</span></div>
    <SaePanel />
  </section>

  <section class="dw-sec" id="instrument-sampling" aria-labelledby="instrument-sampling-head">
    <div class="dw-sec-head"><span class="head" id="instrument-sampling-head">Sampling</span></div>
    <SamplingStrip />
  </section>

  <section class="dw-sec" aria-labelledby="instrument-sources-head">
    <span class="head" id="instrument-sources-head">Sources</span>
    <div class="srcs">
      {#each sources as row (row.key)}
        <div class="src"><span class="sk">{row.key}</span><span class="sv" class:none={!row.value}>{row.value ?? "none"}</span><span class="meta">{row.meta}</span></div>
      {/each}
    </div>
    <button type="button" class="dw-text room-link" onclick={() => goto("lens")}>Lens room</button>
  </section>
</div>

<style>
  .instruments { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s6); min-width: 0; }
  /* The rack panels were drawn for a fixed-height column with scrolling middles; in the inspector
   * they flow with the page. (They go when the instruments lane redraws them.) */
  .instruments :global(:is(.rack, .jlens, .sae, .rack .strips, .jlens .section, .sae .section, .steer-cards, .jlens .scroll, .sae .scroll)) {
    display: flex;
    flex: none;
    height: auto;
    max-height: none;
    min-height: 0;
    overflow: visible;
  }
  .instruments :global(:is(.rack, .jlens, .sae, .jlens .section, .sae .section)) { padding-inline: 0; }
  .instruments :global(:is(.jlens, .sae)) { flex-direction: column; }
  .racks { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s5); min-width: 0; }
  .srcs { display: grid; gap: var(--s2); }
  .src { display: grid; grid-template-columns: 96px auto minmax(0, 1fr); gap: var(--s4); align-items: baseline; }
  .sk { color: var(--ink2); font: var(--type-small); }
  .sv { color: var(--ink); font: 600 14px/20px var(--display); letter-spacing: -0.01em; overflow-wrap: anywhere; }
  .sv.none { color: var(--ink3); font: var(--type-small); }
  .room-link { justify-self: start; }
</style>
