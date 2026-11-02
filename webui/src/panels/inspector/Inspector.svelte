<script lang="ts">
  // The Inspector: whatever is selected (lib/stores/selection.svelte.ts). Nothing shows the
  // instruments; a token its drilldown; a probe its trace and geometry; a unit (a neuron or a head)
  // its view (UnitView). A steering term opens in place under the recipe's chips (TermEditor), apart
  // from the selection, so a Push from a token's or a unit's view leaves that view here. The crumbs
  // lead back to the instruments.
  import { IconButton, Swap } from "@stdlogits/ledger";
  import DrawerOutlet from "../../drawers/DrawerOutlet.svelte";
  import Instruments from "./Instruments.svelte";
  import UnitView from "./UnitView.svelte";
  import { unitTitle } from "../../lib/units";
  import { clearSelection, selection } from "../../lib/stores/selection.svelte";
  import { workbenchUi } from "../../lib/stores/workbench.svelte";

  let { sheet = false, onclose }: { sheet?: boolean; onclose?: () => void } = $props();

  const current = $derived(selection.current);
  // A selected term shows the instruments, as nothing selected does.
  const shown = $derived(current.kind === "unit" ? `unit:${current.id}` : current.kind);
  const here = $derived.by(() => {
    switch (current.kind) {
      case "token": return "Token";
      case "probe": return "Probe";
      case "unit": return unitTitle(current.id);
      default: return "Instruments";
    }
  });

  function close(): void {
    if (onclose) onclose(); else clearSelection();
  }

  /** Back to the instruments. In the phone's sheet the sheet stays up to show them. */
  function toInstruments(): void {
    if (sheet) workbenchUi.sheet = true;
    clearSelection();
  }
</script>

<div class="inspector" class:sheet>
  <div class="dw-crumbs">
    {#if shown === "none"}
      <span class="dw-here">Instruments</span>
    {:else}
      <button type="button" onclick={toInstruments}>Instruments</button><span aria-hidden="true">›</span><span class="dw-here">{here}</span>
    {/if}
    {#if shown !== "none" || sheet}
      <span class="dw-crumbs-end"><IconButton icon="close" label="Close" onclick={close} /></span>
    {/if}
  </div>

  <Swap key={shown}>
    {#if current.kind === "token"}
      <DrawerOutlet name="token_drilldown" docked active mobile={sheet} params={current.params} onclose={clearSelection} />
    {:else if current.kind === "probe"}
      <DrawerOutlet name="probe_inspector" params={current.params} onclose={clearSelection} />
    {:else if current.kind === "unit"}
      <UnitView id={current.id} />
    {:else}
      <Instruments />
    {/if}
  </Swap>
</div>

<style>
  .inspector { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s4); min-width: 0; align-content: start; }
  /* The tools shown here sit flush with the crumbs (their own gutters are for a dialog). */
  .inspector { --drawer-gutter-inline: 0px; --drawer-gutter-block: 0px; }
</style>
