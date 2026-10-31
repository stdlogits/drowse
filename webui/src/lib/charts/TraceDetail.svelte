<script lang="ts">
  // A reading token by token, large: the bars, the reading under a cursor, and a scrubber. Moving
  // the scrubber pauses the trace where it is; Follow latest lets it run again.
  import Slider from "../Slider.svelte";
  import { MorphText } from "@stdlogits/ledger";
  import Trace from "./Trace.svelte";
  import { chartValue } from "./chartValues";
  let { points, percentage = false, height = 56, onpick }: {
    points: (number | null)[];
    percentage?: boolean;
    height?: number;
    /** The token the scrubber rests on, or null while it follows the latest. */
    onpick?: (index: number | null) => void;
  } = $props();
  let index = $state<number | null>(null);
  let paused = $state<(number | null)[] | null>(null);
  const displayPoints = $derived(paused ?? points);
  const selected = $derived(Math.min(index ?? displayPoints.length - 1, displayPoints.length - 1));
  const value = $derived(displayPoints[selected]);
</script>

{#if displayPoints.length > 1}
  <div class="trace-detail">
    <Trace values={displayPoints} limit={0} {height} cursor={paused ? selected : null}
      label={`Reading by token, ${displayPoints.length} tokens`} />
    <Slider value={selected} min={0} max={displayPoints.length - 1} step={1} ariaLabel="Trace sample" displayValue={`Sample ${selected + 1}`} oninput={v => { paused ??= points.slice(); index = v; onpick?.(v); }} />
    <div class="readout">
      <span><MorphText text={`token ${selected + 1} of ${displayPoints.length}`} /></span>
      <b><MorphText text={value == null ? "–" : chartValue(value, percentage)} /></b>
      {#if paused}<button type="button" class="dw-text follow" onclick={() => { index = null; paused = null; onpick?.(null); }}>Follow latest</button>{/if}
    </div>
  </div>
{/if}

<style>
  .trace-detail { display: grid; gap: var(--s2); min-width: 0; }
  .readout { display: flex; align-items: baseline; gap: var(--s3); min-height: 20px; color: var(--ink3); font: var(--type-meta); font-variant-numeric: tabular-nums; }
  .readout b { color: var(--ink); font-weight: 500; }
  .follow { margin-left: auto; font: var(--type-meta); }
</style>
