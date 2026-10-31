<script lang="ts">
  // A reading over time or over depth, as flat bars (Ledger's chart mark): one bar per token, or
  // one per layer in that layer's own place on the ramp. Bars fill with --tint (or the colour the
  // caller gives each one); a negative reading is faint ink, a missing one a 2 px wash.
  import { traceBars } from "./depth";

  let {
    values,
    scale,
    limit = 48,
    height = 24,
    colors,
    cursor = null,
    label,
  }: {
    values: readonly (number | null | undefined)[];
    /** The magnitude a full bar stands for; the largest shown when omitted. */
    scale?: number;
    /** The most recent readings shown (0 shows them all). */
    limit?: number;
    height?: number;
    /** A fill per bar (a layer's depth colour), aligned with the shown bars. */
    colors?: readonly (string | null)[];
    /** One bar to bring forward (the others recede). */
    cursor?: number | null;
    label: string;
  } = $props();

  const bars = $derived(traceBars(values, { scale, limit }));
</script>

<span class="trace" class:has-cursor={cursor !== null} style:height="{height}px" style:gap={bars.length > 40 ? "1px" : "2px"} role="img" aria-label={label}>
  {#each bars as bar, i (i)}
    <i class:neg={bar.negative} class:missing={bar.missing} class:on={cursor === i}
      style:height={bar.missing ? undefined : `${bar.height}%`}
      style:background={bar.missing || bar.negative ? undefined : colors?.[i] ?? undefined}></i>
  {/each}
</span>

<style>
  .trace { display: flex; align-items: flex-end; gap: 2px; min-width: 0; }
  i { flex: 1 1 0; min-width: 1px; min-height: 2px; border-radius: 1px; background: var(--tint); transition: height var(--glide) var(--ease), opacity var(--fast) var(--ease); }
  i.neg { background: var(--ink3); opacity: 0.5; }
  i.missing { height: 2px; background: var(--wash2); }
  .has-cursor i:not(.on) { opacity: 0.4; }
  .has-cursor i.neg:not(.on) { opacity: 0.25; }
</style>
