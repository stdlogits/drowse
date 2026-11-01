<script lang="ts">
  // The dashboard shell, drawn in Ledger: the header (the drowse mark, the rooms as tabs, the model
  // pill, search and the theme), then the room you are in. The workbench is home (Thread | Page |
  // Inspector); Manifolds, Units, Lens and Model are rooms beside it, as routes (#/lens). Tools that
  // are neither a room nor a selection open as a dialog over the page.
  import { onMount, tick } from "svelte";
  import { IconButton, Tabs, ThemeButton, fadeIn, fadeOut, rise, tips } from "@stdlogits/ledger";
  import { theme } from "./lib/theme";
  import { registerWorkspaceController } from "./lib/workspaceController";
  import { loadWebMcp } from "./lib/webmcp/loader";
  import { sessionState } from "./lib/stores/session.svelte";
  import { attachPersistence } from "./lib/stores/persistence.svelte";
  import { manifoldJobs } from "./lib/stores/manifoldJobs.svelte";
  import { ROOMS, goto, route, shownRooms, startRouting, type Room } from "./lib/stores/rooms.svelte";
  import { clearSelection, selection } from "./lib/stores/selection.svelte";
  import { escapeAction } from "./lib/escapeKey";
  import { workbenchUi } from "./lib/stores/workbench.svelte";
  import { paletteState, closePalette, openPalette } from "./lib/stores/palette.svelte";
  import { runtimeClient } from "./lib/runtime/client";
  import { userFacingError } from "./lib/runtime/userFacingError";
  import { restoreConversationSnapshot } from "./lib/conversationWorkspace";
  import { resetSettings } from "./lib/stores/settingsReset.svelte";
  import { conversationLibrary, savedConversationState } from "./lib/stores/savedConversations.svelte";
  import { clearConversationOpen, peekConversationOpen } from "./lib/conversationOpen";
  import { containTab, tabStops } from "./lib/focusTrap";
  import {
    bootstrap,
    ensureRuntimeChannel,
    drawerState,
    closeDrawer,
    genStatus,
    sendStop,
    loomTree,
    currentLoomTreeSnapshot,
    loomUiState,
    loomRegenerateActive,
    requestLoomModal,
    openDrawer,
    pushToast,
  } from "./lib/stores.svelte";
  import { DRAWERS, drawerParams, loadedDrawer } from "./drawers";
  import type { DrawerName } from "./lib/types";
  import DrawerOutlet from "./drawers/DrawerOutlet.svelte";
  import Workbench from "./panels/workbench/Workbench.svelte";
  import RoomSections from "./panels/rooms/RoomSections.svelte";
  import LensRoom from "./panels/rooms/LensRoom.svelte";
  import UnitsRoom from "./panels/rooms/UnitsRoom.svelte";
  import CommandPalette from "./panels/CommandPalette.svelte";
  import Toaster from "./lib/Toaster.svelte";
  import ConversationAutosave from "./lib/ui/ConversationAutosave.svelte";
  import ManifoldJobProgress from "./lib/ui/ManifoldJobProgress.svelte";
  import HeaderMenu, { type MenuGroup } from "./lib/ui/HeaderMenu.svelte";

  let { onhome, onmodels }: { onhome?: () => Promise<void>; onmodels?: () => Promise<void> } = $props();

  type BootStatus = "loading" | "ready" | "failed";
  let bootStatus: BootStatus = $state("loading");
  let bootError: string | null = $state(null);
  let returningHome = $state(false);
  let phone = $state(false);
  let topHeight = $state(68);
  let dialogEl: HTMLElement | null = $state(null);
  let focusWhenReady: DrawerName | null = null;

  const modelId = $derived(sessionState.info?.model_id ?? "");
  const modelName = $derived(modelId.split("/").at(-1) || "No model");
  const device = $derived((sessionState.info?.device ?? "").split(":")[0].toUpperCase());
  const dialogOpen = $derived(drawerState.open !== null);
  const hosted = $derived(runtimeClient.mode !== "http");

  // ---- WebMCP and persistence, once the workbench is ready ----------------------------------

  const webMcpModelId = $derived(sessionState.info?.model_id);
  const webMcpInstruments = $derived(JSON.stringify(sessionState.info?.instruments));
  $effect(() => {
    if (bootStatus !== "ready" || !webMcpModelId) return;
    const webMcp = loadWebMcp();
    if (!webMcp) return;
    let unmount: (() => void) | null = null;
    let cancelled = false;
    void webMcp.then(module => { if (!cancelled) unmount = module.mountWorkspaceWebMcp(); });
    return () => { cancelled = true; unmount?.(); };
  });
  $effect(() => {
    void webMcpInstruments;
    void loadWebMcp()?.then(module => module.refreshWebMcp());
  });
  $effect(() => {
    if (bootStatus === "ready") return attachPersistence();
  });

  onMount(() => {
    const stop = startRouting();
    const query = window.matchMedia("(max-width: 720px)");
    const update = () => { phone = query.matches; };
    update();
    query.addEventListener("change", update);
    return () => { stop(); query.removeEventListener("change", update); };
  });

  onMount(() => registerWorkspaceController({
    read: () => ({ room: route.room, section: route.section, page: workbenchUi.page }),
    navigate: async (change) => {
      if (change.room !== undefined || change.section !== undefined) goto(change.room ?? route.room, change.section ?? null);
      if (change.page !== undefined) {
        if (route.room !== "workbench") goto("workbench");
        workbenchUi.page = change.page;
      }
      await tick();
    },
  }));

  // ---- boot ----------------------------------------------------------------------------------

  function creationActive(): boolean {
    return manifoldJobs.current !== null && ["running", "waiting", "cancelling"].includes(manifoldJobs.current.status);
  }

  function beforeLeave(event: BeforeUnloadEvent): void {
    if (!creationActive()) return;
    event.preventDefault();
    event.returnValue = "";
  }

  async function runBootstrap(): Promise<void> {
    bootStatus = "loading";
    bootError = null;
    try {
      await bootstrap();
      clearSelection();
      savedConversationState.activeId = null;
      savedConversationState.avatarSeed = null;
      savedConversationState.accent = "purple";
      savedConversationState.status = "idle";
      const pendingConversation = peekConversationOpen();
      if (pendingConversation) {
        const record = await conversationLibrary.get(pendingConversation.id);
        if (record.modelId !== pendingConversation.modelId) {
          throw new Error("The saved chat changed before it could be opened. Return to Chats and try again.");
        }
        await restoreConversationSnapshot(record.snapshot);
        savedConversationState.activeId = record.id;
        savedConversationState.avatarSeed = record.avatarSeed;
        savedConversationState.accent = record.accent ?? "purple";
        clearConversationOpen();
      } else if (runtimeClient.mode === "browser" && sessionState.info) {
        const tree = currentLoomTreeSnapshot();
        const record = tree && await conversationLibrary.findForTree(sessionState.info.model_id, tree.root_id);
        if (record) {
          await restoreConversationSnapshot({ ...record.snapshot, tree });
          savedConversationState.activeId = record.id;
          savedConversationState.avatarSeed = record.avatarSeed;
          savedConversationState.accent = record.accent ?? "purple";
        } else if (tree?.nodes.length === 1 && tree.active_node_id === tree.root_id) {
          await resetSettings(true);
        }
      }
      // Open the WebSocket now so the first generation doesn't pay connect latency; the send
      // paths reopen it on demand if this fails.
      try { await ensureRuntimeChannel(); } catch { /* reopened on send */ }
      bootStatus = "ready";
    } catch (e) {
      bootError = userFacingError(e, "Unable to open the workbench. Reopen the model and try again.");
      bootStatus = "failed";
    }
  }

  onMount(() => {
    void runBootstrap();
    const onModelsRequest = () => void returnHome("models");
    window.addEventListener("drowse:models", onModelsRequest);
    return () => window.removeEventListener("drowse:models", onModelsRequest);
  });

  // drowse.ai's hosted shell passes these: its chats and models live outside this app.
  async function returnHome(destination: "chats" | "models" = "chats"): Promise<void> {
    if (returningHome) return;
    if (creationActive()) {
      pushToast("Finish or cancel creation before leaving this model. Your progress stays visible above the workspace.", { kind: "warning" });
      return;
    }
    if (!onhome) {
      openDrawer("load_conversation");
      return;
    }
    if (genStatus.active && !window.confirm(`Stop the current reply and ${destination === "models" ? "open Models" : "return to Your chats"}? Your conversation will be saved on this device.`)) return;
    returningHome = true;
    try { await (destination === "models" && onmodels ? onmodels() : onhome()); }
    catch (error) {
      pushToast(userFacingError(error, `${destination === "models" ? "Models" : "Your chats"} could not open. Stay here and try again.`), { kind: "error" });
    } finally { returningHome = false; }
  }

  // ---- the dialog: tools that are neither a room nor a selection ------------------------------

  const DIALOG_LABELS: Partial<Record<DrawerName, string>> = {
    advanced_sampling: "Sampling settings",
    compare: "Compare controls by layer",
    correlation: "Compare saved readings",
    help: "Help and shortcuts",
    feedback: "Feedback",
    load_conversation: "Saved chats",
    manifolds: "Add a manifold",
    node_compare: "Compare branches",
    save_conversation: "Save chat",
    download_chat: "Download chat",
    subspace: "Add a concept",
    system_prompt: "System prompt",
    transcript: "Transcript",
  };

  let previousDialog: DrawerName | null = null;
  $effect(() => {
    const open = drawerState.open;
    if (open !== null && open !== previousDialog) {
      if (!loadedDrawer(open)) focusWhenReady = open;
      void tick().then(focusDialog);
    } else if (open === null) {
      focusWhenReady = null;
    }
    previousDialog = open;
  });

  function focusDialog(): void {
    if (!dialogEl) return;
    (tabStops(dialogEl)[0] ?? dialogEl).focus({ preventScroll: true });
  }

  function dialogReady(name: DrawerName): void {
    if (focusWhenReady !== name) return;
    focusWhenReady = null;
    if (document.activeElement === dialogEl) focusDialog();
  }

  // ---- keys: ⌘K searches, Escape closes the nearest thing, ⌘R/E/B/N/D act on the loom -------

  async function onWindowKey(ev: KeyboardEvent): Promise<void> {
    if (ev.defaultPrevented) return;
    const mod = ev.ctrlKey || ev.metaKey;
    if (mod && !ev.shiftKey && !ev.altKey && ev.key.toLowerCase() === "k") {
      ev.preventDefault();
      if (paletteState.open) closePalette(); else openPalette();
      return;
    }
    if (ev.key === "Escape") {
      const action = escapeAction({
        paletteOpen: paletteState.open,
        modalOpen: loomUiState.modalRequest.kind !== null,
        drawerOpen: drawerState.open !== null,
        selectionOpen: route.room === "workbench" && selection.current.kind !== "none",
        generating: genStatus.active,
      });
      if (action === "palette") closePalette();
      else if (action === "drawer") closeDrawer();
      else if (action === "selection") clearSelection();
      else if (action === "stop") sendStop();
      if (action !== "modal" && action !== "none") ev.preventDefault();
      return;
    }
    if (!mod || ev.shiftKey || route.room !== "workbench") return;
    const k = ev.key.toLowerCase();
    const active = loomTree.active_node_id;
    const node = active ? loomTree.nodes.get(active) : undefined;
    if (k === "r") {
      ev.preventDefault();
      if (!active) return;
      // A node with a generation receipt regenerates; a committed one opens the modal to author a
      // fresh continuation instead.
      if (node?.recipe) await loomRegenerateActive(1);
      else requestLoomModal("regenerate", { nodeId: active, n: 1 });
    } else if (k === "e") {
      ev.preventDefault();
      if (active) requestLoomModal("edit", { nodeId: active, text: node?.text ?? "" });
    } else if (k === "b") {
      ev.preventDefault();
      if (active) requestLoomModal("branch", { nodeId: active, text: node?.text ?? "" });
    } else if (k === "n") {
      ev.preventDefault();
      requestLoomModal("navpicker", { nodeId: active });
    } else if (k === "d") {
      ev.preventDefault();
      if (active) requestLoomModal("delete", { nodeId: active });
    }
  }

  // ---- the menu ------------------------------------------------------------------------------

  const menuGroups = $derived<MenuGroup[]>([
    ...(phone ? [{ label: "Rooms", items: shownRooms().map(room => ({ label: room.label, current: route.room === room.value, run: () => goto(room.value) })) }] : []),
    {
      label: "Conversation",
      items: [
        { label: "Save chat", run: () => openDrawer("save_conversation") },
        { label: "Saved chats", run: () => openDrawer("load_conversation") },
        { label: "Download chat", disabled: loomTree.nodes.size <= 1 || genStatus.active, run: () => openDrawer("download_chat") },
        { label: "Transcript", run: () => openDrawer("transcript") },
      ],
    },
    {
      label: "Settings",
      items: [
        { label: "System prompt", run: () => openDrawer("system_prompt") },
        { label: "Sampling", run: () => openDrawer("advanced_sampling") },
        ...(hosted && onhome ? [{ label: "Your chats", disabled: returningHome, run: () => void returnHome() }] : []),
        ...(hosted && onmodels ? [{ label: "Models", disabled: returningHome, run: () => void returnHome("models") }] : []),
      ],
    },
    {
      items: [
        { label: "Search everything", run: openPalette },
        { label: "Help and shortcuts", run: () => openDrawer("help") },
        { label: "Feedback", run: () => openDrawer("feedback") },
      ],
    },
  ]);

  function pickRoom(room: Room): void {
    goto(room);
  }
</script>

<svelte:window onkeydown={onWindowKey} onbeforeunload={beforeLeave} />

{#if bootStatus === "failed"}
  <main class="boot" role="alert">
    <span class="dw-mark">drowse</span>
    <h1 class="dw-title">{runtimeClient.mode === "http" ? "The server isn't answering" : "The runtime isn't available"}</h1>
    <p>{bootError}</p>
    <p class="meta">{runtimeClient.mode === "http" ? "Start it with drowse serve, then try again." : "Retry the local model, or return to device setup."}</p>
    <div class="dw-row">
      <button type="button" class="dw-button" onclick={runBootstrap}>Try again</button>
      {#if runtimeClient.mode !== "http"}
        <button type="button" class="dw-quiet" onclick={() => window.location.assign("/app")}>Model setup</button>
      {/if}
    </div>
  </main>
{:else}
  <div class="shell" class:phone use:tips style:--top-h={`${topHeight}px`}>
    <a class="skip" href="#main">Skip to the workbench</a>
    <div class="top" bind:clientHeight={topHeight}>
    <header class="dw-top" inert={dialogOpen || paletteState.open || returningHome}>
      <button type="button" class="dw-mark mark" onclick={() => pickRoom("workbench")}>drowse</button>
      {#if !phone}
        <nav class="rooms" aria-label="Rooms">
          <Tabs label="Rooms" options={shownRooms().map(room => ({ value: room.value, label: room.label }))} value={route.room} onchange={pickRoom} />
        </nav>
      {/if}
      <div class="dw-top-end">
        <button type="button" class="dw-model" aria-current={route.room === "model" ? "page" : undefined}
          aria-label={`${modelName} on ${device || "this device"}: model room`} onclick={() => pickRoom("model")}>
          <span class="dw-model-name">{modelName}</span>{#if device}<span class="meta">{device}</span>{/if}
        </button>
        <IconButton icon="search" label="Search everything (⌘K)" onclick={openPalette} />
        <HeaderMenu groups={menuGroups}>
          {#snippet extra()}{#if phone}<span>Theme</span><ThemeButton {theme} />{/if}{/snippet}
        </HeaderMenu>
        {#if !phone}<ThemeButton {theme} />{/if}
      </div>
    </header>

    <div class="notices" inert={dialogOpen || paletteState.open}>
      {#if bootStatus === "ready"}<ConversationAutosave />{/if}
      {#if !dialogOpen}<ManifoldJobProgress compact />{/if}
    </div>
    </div>

    <main id="main" class="rooms-host" class:waiting={bootStatus !== "ready"} inert={dialogOpen || paletteState.open} aria-busy={bootStatus === "loading"}>
      <h1 class="sr-only">{ROOMS.find(room => room.value === route.room)?.label}</h1>
      <div class="room" hidden={route.room !== "workbench"}>
        <Workbench {phone} active={route.room === "workbench"} />
      </div>
      {#if route.room === "manifolds" || route.room === "model"}
        {#key route.room}<div class="room" in:rise><RoomSections room={route.room} /></div>{/key}
      {:else if route.room === "lens"}
        <div class="room" in:rise><LensRoom /></div>
      {:else if route.room === "units"}
        <div class="room" in:rise><UnitsRoom /></div>
      {/if}
    </main>

    {#if bootStatus === "loading"}
      <div class="booting" role="status" aria-live="polite" in:fadeIn out:fadeOut>
        <span class="sr-only">Opening the workbench</span>
        <div class="dw-blocks" aria-hidden="true"><i></i><i></i><i></i></div>
      </div>
    {/if}

    {#if drawerState.open !== null}
      {@const name = drawerState.open}
      <div class="scrim" aria-hidden="true" onclick={closeDrawer} in:fadeIn out:fadeOut></div>
      <div bind:this={dialogEl} class="dialog" class:narrow={DRAWERS[name].narrow} role="dialog" aria-modal="true"
        aria-label={DIALOG_LABELS[name] ?? name.replaceAll("_", " ")} tabindex="-1"
        onkeydown={event => { if (dialogEl) containTab(event, dialogEl); }} in:rise out:fadeOut>
        <ManifoldJobProgress compact />
        <DrawerOutlet {name} params={drawerParams(name, drawerState.params)} mobile={phone} onclose={closeDrawer} onready={dialogReady} />
      </div>
    {/if}

    <Toaster />
    <CommandPalette />
  </div>
{/if}

<style>
  .shell { min-height: 100dvh; background: var(--paper); color: var(--ink2); font-variant-numeric: tabular-nums; }
  .skip { position: absolute; left: var(--s4); top: -64px; z-index: 60; padding: var(--s2) var(--s4); border-radius: var(--r-pill); background: var(--ink); color: var(--paper); font: var(--type-control); }
  .skip:focus { top: var(--s2); }
  .mark { padding: 0; border: 0; background: transparent; cursor: pointer; }
  .rooms { min-width: 0; }
  .notices { max-height: 25dvh; overflow-y: auto; overscroll-behavior: contain; }
  .rooms-host { min-width: 0; }
  /* Not inert while booting: a piece that measures itself on mount (a gliding thumb) must see it. */
  .rooms-host.waiting { pointer-events: none; }
  .room[hidden] { display: none; }
  .booting { position: fixed; inset: 96px var(--s6) auto; z-index: 30; max-width: 720px; margin: 0 auto; pointer-events: none; }
  .scrim { position: fixed; inset: 0; z-index: 50; background: color-mix(in oklab, var(--wash2) 88%, transparent); -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px); }
  .dialog {
    position: fixed;
    inset: var(--s5) var(--s5) var(--s5) auto;
    z-index: 51;
    display: flex;
    flex-direction: column;
    width: min(960px, calc(100vw - var(--s5) * 2));
    overflow: hidden;
    border-radius: var(--r-row);
    background: var(--paper);
  }
  .dialog.narrow { width: min(520px, calc(100vw - var(--s5) * 2)); }
  .dialog:focus { outline: none; }
  .boot { display: grid; gap: var(--s4); justify-items: start; max-width: 640px; margin: 0 auto; padding: var(--s9) var(--s6); }
  .boot p { color: var(--ink2); overflow-wrap: anywhere; }
  @media (prefers-reduced-transparency: reduce) { .scrim { background: var(--paper); opacity: 0.9; } }
  @media (max-width: 720px) {
    .dw-top { flex-wrap: nowrap; gap: var(--s2); }
    .mark { margin-right: auto; }
    .dw-top-end { margin-left: 0; min-width: 0; }
    .dw-model { min-width: 0; max-width: 42vw; }
    .dialog, .dialog.narrow { inset: var(--s4) 0 0; width: 100%; border-radius: var(--r-row) var(--r-row) 0 0; }
    .booting { inset: 80px var(--s4) auto; }
  }
</style>
