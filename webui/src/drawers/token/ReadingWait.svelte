<script lang="ts">
  // A reading on its way: wash blocks where it will land, and the replay's progress as a meter and
  // a line once the runtime reports one. Never a spinner.
  import { Meter } from "@stdlogits/ledger";
  import type { ReadoutProgress } from "./readout.svelte";

  let { progress, label, blocks = [56] }: {
    progress: ReadoutProgress | null;
    /** What is being read, for the progress bar's name. */
    label: string;
    /** The heights of the wash blocks, in px. */
    blocks?: number[];
  } = $props();

  const share = $derived(progress && progress.progress > 0 ? progress.progress : null);
</script>

<div class="wait" role="status" aria-live="polite" aria-label={label}>
  {#each blocks as height, i (i)}<i style:height="{height}px"></i>{/each}
  {#if progress}
    <span class="line">
      {#if share !== null}<span role="progressbar" aria-label={label} aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(share * 100)}><Meter value={share} label={`${Math.round(share * 100)}%`} width={48} /></span>{/if}
      <span class="meta">{progress.message}</span>
    </span>
  {/if}
</div>

<style>
  .wait { display: grid; gap: var(--s2); min-width: 0; }
  .wait > i { display: block; border-radius: var(--r-box); background: var(--wash); }
  .line { display: flex; align-items: center; gap: var(--s3); min-width: 0; }
  .line > span[role="progressbar"] { display: inline-flex; }
  .line .meta { overflow-wrap: anywhere; }
</style>
