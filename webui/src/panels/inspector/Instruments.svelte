<script lang="ts">
  // The Inspector with nothing selected: the instruments (the Instruments canvas). Probes as rows,
  // the lens live, the SAE's features, sampling, and the sources the lens and SAE read through,
  // one line each (they are chosen in the Lens room).
  import { tokenParts } from "@stdlogits/ledger";
  import ProbeRack from "../ProbeRack.svelte";
  import JLensPanel from "../JLensPanel.svelte";
  import SaePanel from "../SaePanel.svelte";
  import SamplingStrip from "../SamplingStrip.svelte";
  import { saeLoaded, sessionState } from "../../lib/stores/session.svelte";
  import { tokenHoverState } from "../../lib/stores.svelte";
  import { goto } from "../../lib/stores/rooms.svelte";

  const saeOn = $derived(saeLoaded());
  const hovered = $derived(tokenHoverState.active ? tokenParts(tokenHoverState.tokenText) : null);

  const sources = $derived.by(() => {
    const blocks = sessionState.info?.instruments ?? [];
    const of = (family: string) => blocks.find(block => block.family === family)?.source ?? null;
    return [
      { key: "Lens", value: of("lens"), meta: sessionState.info?.jlens_fitted ? "fitted" : "" },
      { key: "SAE", value: of("sae"), meta: "" },
    ];
  });
</script>

{#snippet at()}
  {#if hovered}<span class="meta">at <span class="dw-ws">{hovered.lead ? "·" : ""}</span>{hovered.text}</span>{/if}
{/snippet}

<div class="instruments">
  <section class="dw-sec" id="instrument-probes" aria-labelledby="instrument-probes-head">
    <div class="dw-sec-head"><span class="head" id="instrument-probes-head">Probes</span>{@render at()}</div>
    <ProbeRack />
  </section>

  <section class="dw-sec" id="instrument-lens" aria-labelledby="instrument-lens-head">
    <div class="dw-sec-head"><span class="head" id="instrument-lens-head">Live</span>{@render at()}</div>
    <JLensPanel sources={false} />
  </section>

  <section class="dw-sec" id="instrument-sae" aria-labelledby="instrument-sae-head">
    <div class="dw-sec-head"><span class="head" id="instrument-sae-head">Features</span>{#if !saeOn}<span class="meta">Off</span>{:else}{@render at()}{/if}</div>
    {#if saeOn}
      <SaePanel sources={false} />
    {:else}
      <div class="dw-row"><button type="button" class="dw-quiet" onclick={() => goto("lens")}>Choose an SAE</button></div>
    {/if}
  </section>

  <section class="dw-sec" id="instrument-sampling" aria-labelledby="instrument-sampling-head">
    <div class="dw-sec-head"><span class="head" id="instrument-sampling-head">Sampling</span></div>
    <SamplingStrip />
  </section>

  <section class="dw-sec" aria-labelledby="instrument-sources-head">
    <span class="head" id="instrument-sources-head">Sources</span>
    <div class="srcs">
      {#each sources as row (row.key)}
        <div class="src">
          <span class="sk">{row.key}</span>
          <span class="sv" class:none={!row.value}>{row.value ?? "none"}</span>
          <span class="meta">{row.meta}</span>
          <button type="button" class="dw-text change" onclick={() => goto("lens")} aria-label={`Change the ${row.key} source`}>Change</button>
        </div>
      {/each}
    </div>
  </section>
</div>

<style>
  .instruments { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s6); min-width: 0; }
  .srcs { display: grid; gap: var(--s2); }
  .src { display: grid; grid-template-columns: 96px auto minmax(0, 1fr) auto; gap: var(--s4); align-items: baseline; }
  .sk { color: var(--ink2); font: var(--type-small); }
  .sv { color: var(--ink); font: 600 14px/20px var(--display); letter-spacing: -0.01em; overflow-wrap: anywhere; }
  .sv.none { color: var(--ink3); font: var(--type-small); }
  .change { font: var(--type-meta); }
  @media (max-width: 720px) {
    .src { grid-template-columns: 64px auto minmax(0, 1fr) auto; gap: var(--s3); }
  }
</style>
