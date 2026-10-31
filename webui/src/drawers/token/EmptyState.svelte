<script lang="ts">
  // An absence in the token view: one quiet line for what is missing, an optional second, and the
  // action that fixes it when the caller has one. drowse.ai's lens-missing state renders this too.
  import type { Snippet } from "svelte";

  let { title, detail = null, children }: {
    title: string;
    detail?: string | null;
    /** The fix, when there is one (a quiet button). */
    children?: Snippet;
  } = $props();
</script>

<div class="empty">
  <p class="line">{title}</p>
  {#if detail}<p class="meta">{detail}</p>{/if}
  {#if children}<div class="dw-row">{@render children()}</div>{/if}
</div>

<style>
  .empty { display: grid; gap: var(--s2); justify-items: start; min-width: 0; }
  .line { color: var(--ink2); font: var(--type-small); }
  .empty :global(code) { color: var(--ink); }
</style>
