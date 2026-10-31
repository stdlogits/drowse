<script lang="ts">
  // The Inspector: whatever is selected (lib/stores/selection.svelte.ts). Nothing shows the
  // instruments; a token its drilldown; a probe its trace and geometry; a steering term its card; a
  // unit its view, once the engine has units. The crumbs lead back to the instruments.
  import { IconButton, Swap } from "@stdlogits/ledger";
  import DrawerOutlet from "../../drawers/DrawerOutlet.svelte";
  import Instruments from "./Instruments.svelte";
  import TermView from "./TermView.svelte";
  import { clearSelection, selection } from "../../lib/stores/selection.svelte";
  import { steerRack } from "../../lib/stores/steering.svelte";
  import { workbenchUi } from "../../lib/stores/workbench.svelte";

  let { sheet = false, onclose }: { sheet?: boolean; onclose?: () => void } = $props();

  const current = $derived(selection.current);
  const here = $derived.by(() => {
    switch (current.kind) {
      case "token": return "Token";
      case "probe": return "Probe";
      case "term": return steerRack.entries.has(current.name) ? "Steering" : "Term";
      case "unit": return current.id;
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
    {#if current.kind === "none"}
      <span class="dw-here">Instruments</span>
    {:else}
      <button type="button" onclick={toInstruments}>Instruments</button><span aria-hidden="true">›</span><span class="dw-here">{here}</span>
    {/if}
    {#if current.kind !== "none" || sheet}
      <span class="dw-crumbs-end"><IconButton icon="close" label="Close" onclick={close} /></span>
    {/if}
  </div>

  <Swap key={current.kind === "term" ? `term:${current.name}` : current.kind}>
    {#if current.kind === "token"}
      <DrawerOutlet name="token_drilldown" docked active mobile={sheet} params={current.params} onclose={clearSelection} />
    {:else if current.kind === "probe"}
      <DrawerOutlet name="probe_inspector" params={current.params} onclose={clearSelection} />
    {:else if current.kind === "term"}
      <TermView name={current.name} />
    {:else if current.kind === "unit"}
      <div class="dw-empty"><span class="head">{current.id}</span><p>Units arrive with the engine.</p></div>
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
