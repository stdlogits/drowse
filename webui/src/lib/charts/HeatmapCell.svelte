<script lang="ts">
  // One heatmap cell: a square whose fill is a score read against ``scale`` (the value at which
  // the ramp is full). Used by the correlation and compare matrices and the per-layer strips. A
  // positive score is the depth accent mixed over the wash2 base in oklch; a negative one is ink
  // mixed over it in oklab (a grey's mix), so weak and zero readings stay legible as marks.

  import { HIGHLIGHT_SAT } from "../tokens";

  interface Props {
    /** Heatmap value.  Read against ``scale`` (full color at ``±scale``);
     * ``null`` and non-finite values render as the empty-cell placeholder. */
    value: number | null | undefined;
    /** Saturation scale — the value at which the ramp reaches full color.
     * Defaults to ``HIGHLIGHT_SAT`` (the [-1, 1] cosine-grid regime). */
    scale?: number;
    /** Cell width in pixels. */
    size?: number;
    /** Optional accessible label override.  When omitted, the value is
     * formatted to 3 decimals. */
    title?: string;
    /** The fills a positive and a negative score mix toward (the depth accent and ink by
     * default; a caller may pass another depth's colour). */
    positiveColor?: string;
    negativeColor?: string;
    /** Keyboard-selected state supplied by a LayerStrip. */
    active?: boolean;
    /** When true, render the numeric value inside the cell.  Useful
     * for the correlation matrix where the magnitude carries info; off
     * for fine-grained per-token grids where the cells are too small
     * to read text. */
    showValue?: boolean;
  }

  let {
    value,
    scale = HIGHLIGHT_SAT,
    size = 14,
    title,
    positiveColor = "var(--tint)",
    negativeColor = "var(--ink3)",
    active = false,
    showValue = false,
  }: Props = $props();

  const isNull = $derived(
    value === null || value === undefined || !Number.isFinite(value),
  );
  const normalized = $derived.by(() => {
    if (isNull) return 0;
    const safeScale = Number.isFinite(scale) && scale > 1e-6 ? scale : HIGHLIGHT_SAT;
    return Math.max(-1, Math.min(1, (value as number) / safeScale));
  });
  const bg = $derived.by(() => {
    if (isNull) return "var(--wash)";
    const t = normalized;
    if (Math.abs(t) < 1e-9) return "var(--wash2)";
    // Magnitude moves lightness as well as hue, so the cells read on low-contrast screens and
    // without colour vision.
    const mix = Math.round(15 + Math.abs(t) * 85);
    return t > 0
      ? `color-mix(in oklch, ${positiveColor} ${mix}%, var(--wash2))`
      : `color-mix(in oklab, ${negativeColor} ${mix}%, var(--wash2))`;
  });
  const textColor = $derived(
    !isNull && Math.abs(normalized) >= 0.55 ? "var(--paper)" : "var(--ink)",
  );
  const tip = $derived(
    title ?? (isNull ? "-" : (value as number).toFixed(3)),
  );
  const text = $derived(showValue && !isNull ? (value as number).toFixed(2) : "");
</script>

<div
  class="cell"
  class:active
  style="width: {size}px; height: {size}px; background: {bg}; color: {textColor};"

  role="img"
  aria-label={tip}
>
  {#if text}
    <span class="t">{text}</span>
  {/if}
</div>

<style>
  .cell {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: default;
    font: 500 10px/1 var(--text);
    font-variant-numeric: tabular-nums;
    border-radius: 2px;
    transition: background-color var(--fast) var(--ease);
  }
  .cell.active {
    box-shadow: 0 0 0 2px var(--ink);
  }
</style>
