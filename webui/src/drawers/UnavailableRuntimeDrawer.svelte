<script lang="ts">
  // Stands in for a tool this runtime does not have (here, the local runtime page; in drowse.ai,
  // API access), through the @runtime-local-drawer and @runtime-session-drawer aliases.
  import { IconButton } from "@stdlogits/ledger";
  import { closeDrawer } from "../lib/stores.svelte";
</script>

<section class="unavailable" aria-label="Unavailable runtime drawer">
  <header class="dlg-head">
    <h2>Not available here</h2>
    <IconButton icon="close" label="Close" onclick={closeDrawer} />
  </header>
  <div class="body">
    <div class="dw-empty">
      <p>This runtime does not have this tool.</p>
      <button type="button" class="dw-quiet" onclick={closeDrawer}>Close</button>
    </div>
  </div>
</section>

<style>
  .unavailable { display: flex; flex-direction: column; min-height: 0; }
  .dlg-head { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); padding: var(--s5) var(--s5) 0; }
  .dlg-head h2 { margin: 0; color: var(--ink); font: var(--type-mark); }
  .body { padding: var(--s5); }
  @media (max-width: 720px) {
    .dlg-head { padding: var(--s4) var(--s4) 0; }
    .body { padding: var(--s4); }
  }
  @media (max-width: 720px), (pointer: coarse) {
    .dlg-head :global(.icon) { width: 44px; height: 44px; }
  }
</style>
