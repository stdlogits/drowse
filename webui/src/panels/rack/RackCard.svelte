<script lang="ts">
  // The one shared card for a steering term or a reading: a statline on top (marker, name, status,
  // actions on one row) and its controls or meters stacked below, on a wash. Cards differ only by
  // ``accent`` (exposed to the content as ``--card-accent``, a depth colour that only fills) and by
  // the marker their statline draws.

  import type { Snippet } from "svelte";

  interface Props {
    /** The name of the custom property the card's accent reads (``"--tint"`` for a depth). */
    accent?: string;
    /** A term switched off: the card steps back to a fainter wash. */
    disabled?: boolean;
    /** Chosen (the reading the reply is tinted by, a gate that just fired): wash2. */
    active?: boolean;
    /** The identity line: marker, name, status, actions. */
    statline: Snippet;
    /** Control or meter rows, top to bottom. */
    body: Snippet;
  }

  let {
    accent = "--tint",
    disabled = false,
    active = false,
    statline,
    body,
  }: Props = $props();
</script>

<div
  class="card"
  class:disabled
  class:active
  style:--card-accent="var({accent})"
  role="group"
>
  <div class="statline">{@render statline()}</div>
  <div class="body">{@render body()}</div>
</div>

<style>
  .card {
    display: flex;
    flex-direction: column;
    min-width: 0;
    max-width: 100%;
    gap: var(--s3);
    padding: var(--s3) var(--s4) var(--s4);
    border-radius: var(--r-box);
    background: var(--wash);
    font: var(--type-small);
    transition: background-color var(--fast) var(--ease), opacity var(--fast) var(--ease);
  }
  .card.active { background: var(--wash2); }
  .card.disabled { background: color-mix(in oklab, var(--wash) 50%, transparent); }

  .statline {
    display: flex;
    align-items: center;
    gap: var(--s2);
    min-height: 36px;
    min-width: 0;
  }

  .body {
    display: flex;
    flex-direction: column;
    gap: var(--s2);
    min-width: 0;
  }
</style>
