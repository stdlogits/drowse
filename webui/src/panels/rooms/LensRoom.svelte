<script lang="ts">
  // The Lens room (#/lens, the Lens canvas): which lens reads each depth, then the four lenses (the
  // S and R families at the out and mid sites) with their sources and local fits, then the SAE's
  // source. The lens and SAE source sections are runtime-specific: drowse.ai swaps them for its
  // hosted ones through the @runtime-jlens-source and @runtime-sae-source aliases. The depth axis
  // draws the server's crossover band and shows only where the runtime sends one. The inspector's
  // Sources and its Features ("Choose an SAE") lead here.
  import RuntimeJLensSourceSection from "@runtime-jlens-source";
  import RuntimeSaeSourceSection from "@runtime-sae-source";
  import LensDepth from "./lens/LensDepth.svelte";
  import { lensSourceState } from "../../lib/stores/instruments.svelte";
  import { sessionState } from "../../lib/stores/session.svelte";

  const sae = $derived(sessionState.info?.instruments.find(block => block.family === "sae")?.source ?? null);
  const model = $derived(sessionState.info?.model_id.split("/").at(-1) ?? "");
</script>

<div class="dw-room">
  {#if lensSourceState.band && lensSourceState.band.weights.length > 0}
    <LensDepth band={lensSourceState.band} {model} />
  {/if}
  <RuntimeJLensSourceSection />
  <section class="dw-sec sae" id="lens-sae" aria-labelledby="lens-sae-head">
    <div class="dw-sec-head"><span class="head" id="lens-sae-head">SAE</span><span class="meta">{sae ?? "none"}</span></div>
    <RuntimeSaeSourceSection />
  </section>
</div>

<style>
  .sae { max-width: 760px; }
</style>
