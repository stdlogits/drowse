<script lang="ts">
  // A steering term, selected from the recipe's chips: its card (strength, position, trigger,
  // without, remove) and, for a flat concept, the shared strength every flat term slides with.
  // The full term editor (strength, when, where by layer) arrives with per-term layer scoping.
  import { MorphText } from "@stdlogits/ledger";
  import SteerCard from "../rack/SteerCard.svelte";
  import AtomSteerCard from "../rack/AtomSteerCard.svelte";
  import Slider from "../../lib/Slider.svelte";
  import { setSubspaceAlong, steerRack } from "../../lib/stores/steering.svelte";
  import { clearSelection } from "../../lib/stores/selection.svelte";

  let { name }: { name: string } = $props();

  const entry = $derived(steerRack.entries.get(name));
  $effect(() => {
    // A removed term leaves nothing to show.
    if (!entry && steerRack.customExpression === null) clearSelection();
  });
</script>

{#if entry}
  <div class="term">
    {#if entry.mode === "subspace"}
      <div class="dw-sec">
        <div class="dw-sec-head"><span class="head">Shared strength</span><span class="meta">every flat concept</span></div>
        <div class="along">
          <Slider value={steerRack.subspaceAlong} min={0} max={2} step={0.05}
            oninput={(value) => { if (Number.isFinite(value)) setSubspaceAlong(value); }} ariaLabel="Shared strength of the flat concepts" />
          <span class="value"><MorphText text={steerRack.subspaceAlong.toFixed(2)} /></span>
        </div>
      </div>
    {/if}
    {#if entry.mode === "jlens" || entry.mode === "sae" || entry.mode === "unit"}
      <AtomSteerCard mode={entry.mode} {name} {entry} />
    {:else}
      <SteerCard {name} {entry} />
    {/if}
  </div>
{:else}
  <p class="meta">This term is no longer in the recipe.</p>
{/if}

<style>
  .term { display: grid; gap: var(--s5); min-width: 0; }
  .along { display: grid; grid-template-columns: minmax(0, 1fr) 48px; gap: var(--s4); align-items: center; }
  .value { color: var(--ink2); font: var(--type-meta); text-align: right; }
</style>
