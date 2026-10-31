<script lang="ts">
  // The workbench: three regions on paper, Thread | Page | Inspector, as the ratified canvas draws
  // them (Main, Instruments). They wrap at narrower widths; on a wide screen each scrolls on its
  // own. On a phone it is one column (Phone): the thread folds into a row over the conversation,
  // the inspector rises as a sheet between the conversation and the composer, and the composer
  // folds away while the sheet is pulled up.
  import { Icon, IconButton } from "@stdlogits/ledger";
  import Chat from "../Chat.svelte";
  import LoomSidebar from "../loom/LoomSidebar.svelte";
  import Thread from "./Thread.svelte";
  import Inspector from "../inspector/Inspector.svelte";
  import { workbenchUi, showConversation } from "../../lib/stores/workbench.svelte";
  import { clearSelection, selection } from "../../lib/stores/selection.svelte";
  import { loomUiState } from "../../lib/stores/loomUi.svelte";

  let { phone = false, active = true }: { phone?: boolean; active?: boolean } = $props();

  let threadOpen = $state(false);
  const loomPage = $derived(workbenchUi.page === "loom");
  const sheetShown = $derived(selection.current.kind !== "none" || workbenchUi.sheet);
  const LOOM_VIEWS: Record<string, string> = { weave: "Weave", map: "Map", path: "Path", options: "Next options", saved: "Starred" };

  $effect(() => {
    // A fresh selection on the phone rises at the peek height.
    if (selection.current.kind === "none") workbenchUi.sheetOpen = false;
  });

  function closeSheet(): void {
    clearSelection();
    workbenchUi.sheet = false;
    workbenchUi.sheetOpen = false;
  }
</script>

<div class="workbench" class:phone class:sheet-up={phone && sheetShown && workbenchUi.sheetOpen}>
  <div class="dw-main">
    {#if phone}
      <div class="thread-fold">
        <button type="button" class="fold" aria-expanded={threadOpen} aria-controls="workbench-thread" onclick={() => (threadOpen = !threadOpen)}>
          <span>Thread</span><span class="chev" class:open={threadOpen}><Icon name="down" /></span>
        </button>
      </div>
    {/if}
    <aside class="dw-thread" id="workbench-thread" aria-label="Thread" hidden={phone && !threadOpen}>
      <Thread />
    </aside>

    <section class="dw-page" aria-label={loomPage ? "Loom" : "Conversation"}>
      {#if loomPage}
        <div class="dw-crumbs loom-crumbs">
          <button type="button" onclick={showConversation}>Conversation</button><span aria-hidden="true">›</span>
          <span class="dw-here">{LOOM_VIEWS[loomUiState.view] ?? "Loom"}</span>
          <span class="dw-crumbs-end"><IconButton icon="close" label="Back to the conversation" onclick={showConversation} /></span>
        </div>
      {/if}
      <div class="loom" hidden={!loomPage}>
        <LoomSidebar active={active && loomPage} />
      </div>
      <div class="conversation" hidden={loomPage}>
        <Chat {phone} />
      </div>
    </section>

    {#if !phone}
      <aside class="dw-insp" aria-label="Inspector">
        <Inspector />
      </aside>
    {:else if sheetShown}
      <section class="sheet" class:open={workbenchUi.sheetOpen} aria-label="Inspector">
        <button type="button" class="grab" aria-label={workbenchUi.sheetOpen ? "Collapse the inspector" : "Expand the inspector"}
          aria-expanded={workbenchUi.sheetOpen} onclick={() => (workbenchUi.sheetOpen = !workbenchUi.sheetOpen)}><i></i></button>
        <div class="sheet-body">
          <Inspector sheet onclose={closeSheet} />
        </div>
      </section>
    {/if}
  </div>
</div>

<style>
  /* Positioned, so absolutely placed copies (screen-reader text) inside a scrolling region clip with it. */
  .workbench { position: relative; min-width: 0; }
  .dw-thread, .dw-page, .dw-insp { position: relative; }
  .dw-thread[hidden], .loom[hidden], .conversation[hidden] { display: none; }
  .dw-page { align-content: start; height: calc(100dvh - var(--top-h, 68px) - var(--s4)); min-height: 420px; }
  .dw-page { display: flex; flex-direction: column; gap: var(--s4); }
  .conversation, .loom { flex: 1 1 0; min-height: 0; display: flex; flex-direction: column; }
  .loom-crumbs { flex: none; }

  /* Wide: the three regions side by side, each scrolling on its own under the header. */
  @media (min-width: 1312px) {
    .workbench:not(.phone) { height: calc(100dvh - var(--top-h, 68px)); overflow: hidden; }
    .workbench:not(.phone) .dw-main { height: 100%; flex-wrap: nowrap; align-items: stretch; padding-bottom: 0; }
    .workbench:not(.phone) .dw-thread,
    .workbench:not(.phone) .dw-insp { overflow-y: auto; overscroll-behavior: contain; padding-bottom: var(--s8); align-content: start; scrollbar-width: thin; }
    .workbench:not(.phone) .dw-page { height: auto; padding-bottom: var(--s5); }
  }

  /* The phone: one column under the header; the conversation, then the sheet, then the composer. */
  .phone { height: calc(100dvh - var(--top-h, 60px)); overflow: hidden; }
  .phone .dw-main { display: flex; flex-direction: column; flex-wrap: nowrap; align-items: stretch; gap: 0; height: 100%; padding: 0; }
  .phone .thread-fold { flex: none; padding: 0 var(--s4); }
  .phone .dw-thread { flex: none; max-height: 40dvh; overflow-y: auto; padding: 0 var(--s4) var(--s3); }
  .phone .dw-page, .phone .conversation { display: contents; }
  .phone .loom { flex: 1 1 0; min-height: 0; overflow: auto; padding: 0 var(--s4); }
  .phone .loom-crumbs { padding: 0 var(--s4); }
  .phone :global(.chat-panel), .phone :global(.chat) { display: contents; }
  .phone :global(.chat .log), .phone :global(.chat .raw-host) { order: 1; flex: 1 1 0; min-height: 0; }
  .phone .sheet { order: 2; }
  .phone :global(.chat .dock) { order: 3; }
  .phone.sheet-up :global(.chat .dock) { display: none; }
  .fold { display: inline-flex; align-items: center; gap: var(--s2); height: 44px; padding: 0; border: 0; background: transparent; color: var(--ink3); font: var(--type-control); cursor: pointer; transition: color var(--fast) var(--ease); }
  .fold:hover, .fold[aria-expanded="true"] { color: var(--ink); }
  .chev { display: grid; transition: rotate var(--glide) var(--ease); }
  .chev.open { rotate: 180deg; }

  .sheet {
    position: relative;
    z-index: 2;
    flex: none;
    display: flex;
    flex-direction: column;
    height: min(380px, 46dvh);
    border-radius: var(--r-row) var(--r-row) 0 0;
    background: var(--wash);
    transition: height var(--glide) var(--ease);
  }
  .sheet.open { height: calc(100dvh - var(--top-h, 60px) - var(--s5)); }
  .grab { position: relative; flex: none; align-self: center; display: grid; place-items: center; width: 88px; height: 24px; padding: 0; border: 0; background: transparent; cursor: pointer; transition: scale var(--fast) var(--ease); }
  .grab::before { content: ""; position: absolute; inset: -16px 0 0; }
  .grab:active { scale: 0.97; }
  .grab > i { width: 36px; height: 4px; border-radius: 2px; background: var(--wash2); }
  .sheet-body { flex: 1 1 0; min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding: 0 var(--s4) var(--s5); }
  /* On the sheet's wash, the pieces that sit on a wash step up to wash2. */
  .sheet-body :global(.dw-quiet), .sheet-body :global(.dw-term), .sheet-body :global(.tok), .sheet-body :global(.icon) { background: var(--wash2); }
</style>
