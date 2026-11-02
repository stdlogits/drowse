<script lang="ts">
  // A room of tools: Artifacts (the unit catalog, lenses, build, packs, combine, templates, roles,
  // SAEs) and Model (its shape, depth and what is prepared for it, then health or the local runtime,
  // API access). Its sections are tabs. A drawer section is a tool that used to open as a drawer,
  // shown in place (``openDrawer`` of one of these lands here); a panel section (``drawer: null``) is
  // the component the room passes in ``panels`` under the section's value
  // (lib/stores/rooms.svelte.ts). The way back is the header's tabs (on a phone, the shell's crumb).
  import type { Component } from "svelte";
  import { Swap, Tabs } from "@stdlogits/ledger";
  import DrawerOutlet from "../../drawers/DrawerOutlet.svelte";
  import ModelFacts from "./ModelFacts.svelte";
  import { drawerParams } from "../../drawers";
  import { ROOMS, goto, route, shownSections, type Room } from "../../lib/stores/rooms.svelte";
  import { sessionState } from "../../lib/stores/session.svelte";
  import { loomTree } from "../../lib/stores/loom.svelte";
  import { DROWSE_UI_VERSION } from "../../lib/version";

  let { room, panels = {} }: { room: Room; panels?: Readonly<Record<string, Component>> } = $props();

  const sections = $derived(shownSections(room).filter(row => row.drawer !== null || row.value in panels));
  const section = $derived(sections.find(row => row.value === route.section) ?? sections[0]);
  const title = $derived(room === "model"
    ? (sessionState.info?.model_id.split("/").at(-1) ?? "Model")
    : ROOMS.find(row => row.value === room)?.label ?? room);
  const params = $derived(section?.drawer ? drawerParams(section.drawer, { ...(route.params as Record<string, unknown> | null ?? {}), embedded: true }) : null);
  const Panel = $derived(section && section.drawer === null ? panels[section.value] : undefined);
</script>

<div class="dw-room">
  <div class="head-block">
    <h2 class="dw-title">{title}</h2>
    {#if room === "model" && sessionState.info}
      {@const shape = sessionState.info.architecture}
      <div class="dw-facts">
        {#if shape}
          <span><b>{shape.layers.toLocaleString("en-US")}</b> layers</span>
          {#if shape.neurons_per_layer !== null}<span><b>{shape.neurons_per_layer.toLocaleString("en-US")}</b> neurons a layer</span>{/if}
          {#if shape.heads_per_layer !== null}<span><b>{shape.heads_per_layer.toLocaleString("en-US")}</b> heads a layer</span>{/if}
        {/if}
        <span>on <b>{sessionState.info.device.split(":")[0].toUpperCase()}</b></span>
        <span><b>{sessionState.info.dtype}</b></span>
        {#if sessionState.info.is_base_model}<span>base model</span>{/if}
        {#if sessionState.info.supports_thinking}<span>thinks</span>{/if}
        {#if loomTree.drowse_version}<span>drowse <b>{loomTree.drowse_version}</b></span>{/if}
        <span>dashboard <b>{DROWSE_UI_VERSION}</b></span>
      </div>
    {/if}
  </div>

  {#if room === "model"}
    <ModelFacts />
  {/if}

  {#if sections.length > 1}
    <Tabs label={`${title} sections`} value={section?.value ?? ""} onchange={(value) => goto(room, value)}
      options={sections.map(row => ({ value: row.value, label: row.label }))} />
  {/if}

  {#if section?.drawer}
    <Swap key={section.value}>
      <div class="tool">
        <DrawerOutlet name={section.drawer} {params} />
      </div>
    </Swap>
  {:else if section && Panel}
    <Swap key={section.value}>
      <Panel />
    </Swap>
  {:else}
    <div class="dw-empty">
      <p>Nothing here in this runtime</p>
      <button type="button" class="dw-quiet" onclick={() => goto("workbench")}>Workbench</button>
    </div>
  {/if}
</div>

<style>
  .head-block { display: grid; gap: var(--s1); }
  .tool { display: flex; flex-direction: column; min-width: 0; max-width: 1040px; border-radius: var(--r-row); background: var(--paper); }
</style>
