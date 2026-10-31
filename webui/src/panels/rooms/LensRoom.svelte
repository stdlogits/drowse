<script lang="ts">
  // The Lens room: which lens reads each depth and where its sources come from. Until the engine's
  // four lenses land, it hosts the lens source section (fetch, fit, choose) and the SAE source
  // section (fetch, train, choose), which drowse.ai swaps for its hosted ones through the
  // @runtime-jlens-source and @runtime-sae-source aliases. The inspector's Sources and its
  // Features ("Choose an SAE") lead here.
  import RuntimeJLensSourceSection from "@runtime-jlens-source";
  import RuntimeSaeSourceSection from "@runtime-sae-source";
  import { goto } from "../../lib/stores/rooms.svelte";
  import { sessionState } from "../../lib/stores/session.svelte";

  const source = $derived(sessionState.info?.instruments.find(block => block.family === "lens")?.source ?? null);
  const sae = $derived(sessionState.info?.instruments.find(block => block.family === "sae")?.source ?? null);
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
  <section class="dw-sec sources" id="lens-sae" aria-labelledby="lens-sae-head">
    <div class="dw-sec-head"><span class="head" id="lens-sae-head">SAE</span><span class="meta">{sae ?? "none"}</span></div>
    <RuntimeSaeSourceSection />
  </section>
</div>

<style>
  .head-block { display: grid; gap: var(--s1); }
  .sources { max-width: 760px; }
</style>
