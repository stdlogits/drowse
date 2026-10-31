<script lang="ts">
  // A unit's activation on each token of a reply (the Unit canvas's "On this conversation"): one bar
  // per token, up in the unit's depth colour for a firing, down in faint ink below zero. The peak,
  // or the bar under the pointer, is named with its token and value. A firing under a quarter of
  // the pole's typical high firing stays faint.
  import { tokenParts } from "@stdlogits/ledger";
  import { FIRES_AT, peakIndex } from "../../lib/units";

  let { values, tokens, label }: { values: readonly (number | null)[]; tokens: readonly string[]; label: string } = $props();

  let hot = $state(-1);
  const peak = $derived(peakIndex(values));
  const top = $derived(Math.max(1e-9, ...values.map(v => (typeof v === "number" && Number.isFinite(v) ? Math.abs(v) : 0))));
  const named = $derived(hot >= 0 ? hot : peak);
  const UP = 60;
  const DOWN = 14;

  function text(v: number): string {
    const a = Math.abs(v);
    const t = a >= 100 ? a.toFixed(0) : a >= 10 ? a.toFixed(1) : a.toFixed(2);
    return v < 0 ? `−${t}` : t;
  }
</script>

<div class="trace">
  <div class="plot" style:--n={values.length} role="img" aria-label={label} onpointerleave={() => (hot = -1)}>
    {#each values as v, i (i)}
      {@const value = typeof v === "number" && Number.isFinite(v) ? v : 0}
      {@const share = Math.abs(value) / top}
      <span class="c" role="presentation" onpointerenter={() => (hot = i)}>
        {#if value >= 0}
          <i class="up" class:lit={value >= FIRES_AT || hot === i} style:height="{Math.max(2, share * UP)}px"></i>
        {:else}
          <i class="down" class:lit={hot === i} style:height="{Math.max(2, share * DOWN)}px"></i>
        {/if}
        {#if i === named && tokens[i] !== undefined}
          {@const parts = tokenParts(tokens[i])}
          <span class="lab" class:start={i < values.length * 0.15} class:end={i > values.length * 0.85}
            style:bottom="{20 + (value >= 0 ? Math.max(2, share * UP) : 2) + 4}px">
            {#if parts.lead}<span class="dw-ws">·</span>{/if}{parts.empty ? "∅" : parts.text} <b>{text(value)}</b>
          </span>
        {/if}
      </span>
    {/each}
  </div>
  <div class="dw-ends meta"><span>token 1</span><span>token {values.length}</span></div>
</div>

<style>
  .trace { display: grid; gap: var(--s1); min-width: 0; }
  .plot { display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); height: 112px; }
  .c { position: relative; min-width: 0; }
  .c > i { position: absolute; left: 19%; width: 62%; border-radius: 2px; opacity: 0.4; transition: opacity var(--fast) var(--ease), height var(--glide) var(--ease); }
  .c > i.lit { opacity: 1; }
  .up { bottom: 20px; background: var(--tint); }
  .down { top: 94px; background: var(--ink3); }
  .down.lit { opacity: 0.6; }
  .lab { position: absolute; left: 50%; translate: -50% 0; color: var(--ink2); font: var(--type-meta); white-space: nowrap; pointer-events: none; animation: rise var(--in) var(--ease) backwards; }
  .lab.start { left: 0; translate: 0 0; }
  .lab.end { left: auto; right: 0; translate: 0 0; }
  .lab b { color: var(--ink); font-weight: 600; }
  .dw-ends { display: flex; justify-content: space-between; }
</style>
