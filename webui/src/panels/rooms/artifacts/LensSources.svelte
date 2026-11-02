<script lang="ts">
  // The Artifacts room's Lenses section (#/artifacts/lenses): the lens sources, fetched and fitted.
  // The S and R families with their selected source at the out and mid sites, their sources and a
  // local fit are the runtime's source section (@runtime-jlens-source: drowse.ai swaps in its hosted
  // one); the head counts what is prepared, and the Lens room reads with it.
  import RuntimeJLensSourceSection from "@runtime-jlens-source";
  import { goto } from "../../../lib/stores/rooms.svelte";
  import { lensSourceState } from "../../../lib/stores/instruments.svelte";

  const prepared = $derived(lensSourceState.sources.filter(row => row.family === "R" || row.family === "S"));
  const inUse = $derived(prepared.filter(row => row.active === true).map(row => row.family));
  const status = $derived(prepared.length === 0
    ? (lensSourceState.loading ? "" : "none prepared")
    : [`${prepared.length} prepared`, inUse.length > 0 ? `${inUse.join(" and ")} in use` : ""].filter(Boolean).join(" · "));
</script>

<section class="dw-sec" aria-labelledby="artifact-lenses-head">
  <div class="dw-sec-head"><span class="head" id="artifact-lenses-head">Lenses</span>{#if status}<span class="meta">{status}</span>{/if}</div>
  <RuntimeJLensSourceSection />
  {#if inUse.length > 0}
    <div class="dw-row"><button type="button" class="dw-quiet" onclick={() => goto("lens")}>Read the lens</button></div>
  {/if}
</section>
