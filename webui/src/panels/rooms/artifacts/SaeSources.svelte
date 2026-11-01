<script lang="ts">
  // The Artifacts room's SAEs section (#/artifacts/saes): the SAE sources, fetched and trained. The
  // sources, a release to fetch and a local training are the runtime's source section
  // (@runtime-sae-source: drowse.ai swaps in its hosted one); the head names the SAE in use, and the
  // workbench's Features read with it.
  import RuntimeSaeSourceSection from "@runtime-sae-source";
  import { showInstrument } from "../../../lib/commands";
  import { sessionState } from "../../../lib/stores/session.svelte";

  const sae = $derived(sessionState.info?.instruments.find(block => block.family === "sae")?.source ?? null);
</script>

<section class="dw-sec panel" aria-labelledby="artifact-saes-head">
  <div class="dw-sec-head"><span class="head" id="artifact-saes-head">SAE</span><span class="meta">{sae ?? "none"}</span></div>
  <RuntimeSaeSourceSection />
  {#if sae}
    <div class="dw-row"><button type="button" class="dw-quiet" onclick={() => showInstrument("sae")}>Read features</button></div>
  {/if}
</section>

<style>
  .panel { max-width: 760px; }
</style>
