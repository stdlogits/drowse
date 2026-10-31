<script lang="ts">
  import type { Snippet } from "svelte";

  let {
    left,
    bar,
    middle,
    right,
    ariaLabel,
  }: {
    left: Snippet;
    bar: Snippet;
    middle: Snippet;
    right: Snippet;
    ariaLabel?: string;
  } = $props();
</script>

<div class="reading" role="group" aria-label={ariaLabel}>
  <div class="label">{@render left()}</div>
  <div class="bar">{@render bar()}</div>
  <div class="middle">{@render middle()}</div>
  <div class="value">{@render right()}</div>
</div>

<style>
  .reading {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) max-content;
    align-items: center;
    gap: var(--s1) var(--s3);
    padding-block: var(--s1);
    min-width: 0;
    color: var(--ink2);
    font: var(--type-small);
    font-variant-numeric: tabular-nums;
  }
  .label,
  .middle {
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .middle { text-align: end; }
  .value {
    color: var(--ink);
    text-align: end;
    overflow-wrap: normal;
    white-space: nowrap;
  }
  .bar {
    min-width: 0;
    grid-column: 1 / -1;
    grid-row: 2;
  }
  .label :global(*), .middle :global(*) {
    white-space: normal;
  }
  .value :global(*) {
    white-space: nowrap;
  }
  .bar :global(.bar) {
    width: 100%;
    height: 4px;
    display: block;
  }
</style>
