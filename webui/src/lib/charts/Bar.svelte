<script lang="ts">
  import { barExtent, barTooltip } from "./chartValues";

  interface Props {
    value: number;
    max: number;
    width?: number;
    height?: number;
    color?: string;
    bipolar?: boolean;
    percentage?: boolean;
    title?: string;
  }

  let {
    value, max, width = 144, height = 8,
    color, bipolar = false, percentage = false, title,
  }: Props = $props();

  const filled = $derived(barExtent(value, max) * (bipolar ? 50 : 100));
  const fillX = $derived(bipolar ? (value < 0 ? 50 - filled : 50) : 0);
  // A positive reading fills with the depth accent, a negative one with faint ink.
  const fill = $derived(color ?? (value < 0 ? "var(--ink3)" : "var(--tint)"));
  const faint = $derived(color === undefined && value < 0);
  const tip = $derived(title ?? barTooltip(value, max, percentage));
</script>

<span
  class="bar"
  style:--bar-width={`${width}px`}
  style:height={`${height}px`}

  role="img"
  aria-label={tip}
>
  <span class="fill" class:faint style:left={`${fillX}%`} style:width={`${filled}%`} style:background={fill}></span>
  {#if bipolar}<span class="zero"></span>{/if}
</span>

<style>
  /* A flat meter: a wash2 track, the reading as a fill. A bipolar bar grows either way from a
   * paper notch at zero. */
  .bar {
    position: relative;
    display: inline-block;
    vertical-align: middle;
    width: var(--bar-width);
    max-width: 100%;
    border-radius: 2px;
    background: var(--wash2);
    overflow: hidden;
  }
  .fill {
    position: absolute;
    top: 0;
    bottom: 0;
    border-radius: inherit;
    transition: width var(--glide) var(--ease), left var(--glide) var(--ease);
  }
  .fill.faint { opacity: 0.6; }
  .zero {
    position: absolute;
    inset-block: 0;
    left: calc(50% - 1px);
    width: 2px;
    background: var(--paper);
  }
</style>
