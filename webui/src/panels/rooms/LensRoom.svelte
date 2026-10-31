<script lang="ts">
  // The Lens room: which lens reads each depth and where its sources come from. Until the engine's
  // four lenses land, it hosts the lens source section (fetch, fit, choose), which drowse.ai swaps
  // for its hosted one through the @runtime-jlens-source alias.
  import RuntimeJLensSourceSection from "@runtime-jlens-source";
  import { goto } from "../../lib/stores/rooms.svelte";
  import { sessionState } from "../../lib/stores/session.svelte";

  const source = $derived(sessionState.info?.instruments.find(block => block.family === "lens")?.source ?? null);
</script>

<div class="dw-room">
  <div class="head-block">
    <div class="dw-crumbs">
      <button type="button" onclick={() => goto("workbench")}>Workbench</button><span aria-hidden="true">›</span>
      <span class="dw-here">Lens</span>
    </div>
    <h2 class="dw-title">Lens</h2>
    <div class="dw-facts">
      <span>source <b>{source ?? "none"}</b></span>
      <span>{sessionState.info?.jlens_fitted ? "fitted for this model" : "not fitted for this model"}</span>
    </div>
  </div>
  <section class="dw-sec sources" aria-label="Lens sources">
    <RuntimeJLensSourceSection />
  </section>
</div>

<style>
  .head-block { display: grid; gap: var(--s1); }
  .sources { max-width: 760px; }
</style>
