<script lang="ts">
  // A reading over time or over depth, as flat bars (Ledger's chart mark): one bar per token, or
  // one per layer in that layer's own place on the ramp. Bars fill with --tint (or the colour the
  // caller gives each one); a negative reading is faint ink, a missing one a 2 px wash. A long
  // trace folds neighbouring readings into one bar (its largest), so it never outgrows its width.
  import { bucketed, traceBars } from "./depth";

  let {
    values,
    scale,
    limit = 48,
    maxBars = 160,
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
    /** The most bars drawn; more readings fold together. */
    maxBars?: number;
    height?: number;
    /** A fill per reading (a layer's depth colour), aligned with ``values``. */
    colors?: readonly (string | null)[];
    /** One reading to bring forward (the others recede), by its index in ``values``. */
    cursor?: number | null;
    label: string;
  } = $props();

  const shown = $derived(limit > 0 && values.length > limit ? values.slice(values.length - limit) : values);
  const offset = $derived(values.length - shown.length);
  const folded = $derived(bucketed(shown, maxBars));
  const bars = $derived(traceBars(folded.values, { scale, limit: 0 }));
  const barCursor = $derived(cursor === null || cursor < offset ? null : Math.floor((cursor - offset) / folded.size));
  const fill = (i: number) => (folded.size === 1 ? colors?.[offset + i] ?? undefined : undefined);
</script>

<span class="trace" class:has-cursor={barCursor !== null} style:height="{height}px" style:gap={bars.length > 80 ? "0px" : bars.length > 40 ? "1px" : "2px"} role="img" aria-label={label}>
  {#each bars as bar, i (i)}
    <i class:neg={bar.negative} class:missing={bar.missing} class:on={barCursor === i}
      style:height={bar.missing ? undefined : `${bar.height}%`}
      style:background={bar.missing || bar.negative ? undefined : fill(i)}></i>
  {/each}
</span>

<style>
  .trace { display: flex; align-items: flex-end; gap: 2px; min-width: 0; }
  i { flex: 1 1 0; min-width: 0; min-height: 2px; border-radius: 1px; background: var(--tint); transition: height var(--glide) var(--ease), opacity var(--fast) var(--ease); }
  i.neg { background: var(--ink3); opacity: 0.5; }
  i.missing { height: 2px; background: var(--wash2); }
  .has-cursor i:not(.on) { opacity: 0.4; }
  .has-cursor i.neg:not(.on) { opacity: 0.25; }
</style>
