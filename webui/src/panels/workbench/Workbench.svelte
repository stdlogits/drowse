<script lang="ts">
  // The workbench: three regions on paper, Thread | Page | Inspector, as the ratified canvas draws
  // them (Main, Instruments). Wide (1240 px and up) they sit side by side and each scrolls on its
  // own. Narrower, the thread folds into a row over the page and the Inspector keeps its column
  // beside the page, so a selection always shows. On a phone it is one column (Phone): the thread
  // folds into a row over the conversation, the inspector rises as a sheet between the conversation
  // and the composer, and the composer folds away while the sheet is pulled up. A phone on its side
  // (520 px tall or less) is short of height, not width: the conversation keeps the column under
  // the thread row and the composer, or the inspector, takes a column beside it, and the loom's
  // map is most of a screen tall on a page that scrolls. The conversation's format sits by the
  // thread's header: under it in the thread column, else in the folded row.
  import { untrack } from "svelte";
  import { MediaQuery } from "svelte/reactivity";
  import { Icon, IconButton } from "@stdlogits/ledger";
  import Chat from "../Chat.svelte";
  import LoomSidebar from "../loom/LoomSidebar.svelte";
  import Thread from "./Thread.svelte";
  import ConversationFormat from "./ConversationFormat.svelte";
  import Inspector from "../inspector/Inspector.svelte";
  import { workbenchUi, showConversation } from "../../lib/stores/workbench.svelte";
  import { clearSelection, selection, selectionIdentity, selectionStep } from "../../lib/stores/selection.svelte";
  import { loomUiState } from "../../lib/stores/loomUi.svelte";

  let { phone = false, active = true }: { phone?: boolean; active?: boolean } = $props();

  let threadOpen = $state(false);
  /** The three regions side by side, the thread a column of its own. */
  const wideQuery = new MediaQuery("min-width: 1240px");
  const threadColumn = $derived(!phone && wideQuery.current);
  const loomPage = $derived(workbenchUi.page === "loom");
  // The loom page takes the phone's whole column, so the sheet waits until the conversation is back.
  // A steering term opens in place over the composer, not in the sheet.
  const selecting = $derived(selection.current.kind !== "none");
  const sheetShown = $derived(!loomPage && (selecting || workbenchUi.sheet));
  const LOOM_VIEWS: Record<string, string> = { weave: "Weave", map: "Map", path: "Path", options: "Next options", saved: "Starred" };

  $effect(() => {
    // The loom takes the phone's column: opening it folds the thread (its way in), which would
    // otherwise hold most of the column over the map.
    if (phone && loomPage) untrack(() => (threadOpen = false));
  });

  $effect(() => {
    // A fresh selection on the phone rises at the peek height. Clearing the selection to show the
    // instruments in the sheet (the sampling line, the crumb) keeps the sheet where it was asked.
    if (!selecting && !untrack(() => workbenchUi.sheet)) workbenchUi.sheetOpen = false;
  });

  // A new token or unit opens its view at the top (its header, p and alternatives), not at the
  // scroll the Inspector last had for something else. A step of the token cursor keeps the scroll.
  let inspector: HTMLElement | null = $state(null);
  let sheetBody: HTMLElement | null = $state(null);
  const selectionKey = $derived(selectionIdentity(selection.current));
  $effect(() => {
    if (selectionKey === null) return;
    untrack(() => {
      if (selectionStep.stepped) return;
      if (inspector) inspector.scrollTop = 0;
      if (sheetBody) sheetBody.scrollTop = 0;
    });
  });

  function closeSheet(): void {
    clearSelection();
    workbenchUi.sheet = false;
    workbenchUi.sheetOpen = false;
  }
</script>

<div class="workbench" class:phone class:loom-page={loomPage} class:sheet-shown={phone && sheetShown} class:sheet-up={phone && sheetShown && workbenchUi.sheetOpen}>
  <div class="dw-main">
    <div class="thread-fold">
      <button type="button" class="fold" aria-expanded={threadOpen} aria-controls="workbench-thread" onclick={() => (threadOpen = !threadOpen)}>
        <span>Thread</span><span class="chev" class:open={threadOpen}><Icon name="down" /></span>
      </button>
      {#if !threadColumn}<ConversationFormat />{/if}
    </div>
    <aside class="dw-thread" class:folded={!threadOpen} id="workbench-thread" aria-label="Thread">
      <Thread format={threadColumn} />
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
      <aside class="dw-insp" aria-label="Inspector" bind:this={inspector}>
        <Inspector />
      </aside>
    {:else if sheetShown}
      <section class="sheet" class:open={workbenchUi.sheetOpen} aria-label="Inspector">
        <button type="button" class="grab" aria-label={workbenchUi.sheetOpen ? "Collapse the inspector" : "Expand the inspector"}
          aria-expanded={workbenchUi.sheetOpen} onclick={() => (workbenchUi.sheetOpen = !workbenchUi.sheetOpen)}><i></i></button>
        <div class="sheet-body" bind:this={sheetBody}>
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
  .loom[hidden], .conversation[hidden] { display: none; }
  .dw-page { align-content: start; height: calc(100dvh - var(--top-h, 68px) - var(--s4)); min-height: 420px; }
  .dw-page { display: flex; flex-direction: column; gap: var(--s4); }
  .conversation, .loom { flex: 1 1 0; min-height: 0; display: flex; flex-direction: column; }
  .loom-crumbs { flex: none; }

  /* Wide: the three regions side by side, each scrolling on its own under the header. */
  @media (min-width: 1240px) {
    .workbench:not(.phone) { height: calc(100dvh - var(--top-h, 68px)); overflow: hidden; }
    .workbench:not(.phone) .dw-main { height: 100%; flex-wrap: nowrap; align-items: stretch; padding-bottom: 0; }
    .workbench:not(.phone) .thread-fold { display: none; }
    .workbench:not(.phone) .dw-thread,
    .workbench:not(.phone) .dw-insp { overflow-y: auto; overscroll-behavior: contain; padding-bottom: var(--s8); align-content: start; scrollbar-width: thin; }
    .workbench:not(.phone) .dw-page { height: auto; padding-bottom: var(--s5); }
  }

  /* Between the phone and wide: Page | Inspector, the thread folded over the page, each column
   * scrolling on its own, so whatever is selected shows beside the conversation. */
  @media (max-width: 1239.98px) {
    .workbench:not(.phone) { height: calc(100dvh - var(--top-h, 68px)); overflow: hidden; }
    .workbench:not(.phone) .dw-main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) clamp(300px, 42%, 400px);
      grid-template-rows: auto auto minmax(0, 1fr);
      align-items: stretch;
      gap: 0 var(--s6);
      height: 100%;
      padding-bottom: 0;
    }
    .workbench:not(.phone) .thread-fold { grid-column: 1; grid-row: 1; }
    .workbench:not(.phone) .dw-thread { grid-column: 1; grid-row: 2; max-height: 40dvh; overflow-y: auto; overscroll-behavior: contain; padding-bottom: var(--s3); }
    .workbench:not(.phone) .dw-thread.folded { display: none; }
    .workbench:not(.phone) .dw-page { grid-column: 1; grid-row: 3; height: auto; min-height: 0; max-width: 720px; padding-bottom: var(--s5); }
    .workbench:not(.phone) .dw-insp { grid-column: 2; grid-row: 1 / -1; max-width: none; min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding-bottom: var(--s8); scrollbar-width: thin; }
  }

  /* The phone: one column under the header; the conversation, then the sheet, then the composer. */
  .phone { height: calc(100dvh - var(--top-h, 60px)); overflow: hidden; }
  .phone .dw-main { display: flex; flex-direction: column; flex-wrap: nowrap; align-items: stretch; gap: 0; height: 100%; padding: 0; }
  .phone .thread-fold { flex: none; padding: 0 var(--s4); }
  .phone .dw-thread { flex: none; max-height: 40dvh; overflow-y: auto; padding: 0 var(--s4) var(--s3); }
  .phone .dw-thread.folded { display: none; }
  .phone .dw-page, .phone .conversation:not([hidden]) { display: contents; }
  .phone .loom { flex: 1 1 0; min-height: 0; overflow: auto; padding: 0 var(--s4); }
  .phone .loom-crumbs { padding: 0 var(--s4); }
  .phone :global(.chat-panel), .phone :global(.chat) { display: contents; }
  .phone :global(.chat .log), .phone :global(.chat .raw-host) { order: 1; flex: 1 1 0; min-height: 0; }
  .phone .sheet { order: 2; }
  .phone :global(.chat .dock) { order: 3; }
  .phone.sheet-up :global(.chat .dock) { display: none; }
  /* While the sheet is up, the dock keeps the recipe and the composer and drops the status line. */
  .phone.sheet-shown :global(.chat .dock .status-footer) { display: none; }
  .thread-fold { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s1) var(--s3); min-width: 0; }
  .fold { display: inline-flex; align-items: center; gap: var(--s2); height: 44px; padding: 0; border: 0; background: transparent; color: var(--ink3); font: var(--type-control); cursor: pointer; transition: color var(--fast) var(--ease); }
  .fold:hover, .fold[aria-expanded="true"] { color: var(--ink); }
  .chev { display: grid; transition: rotate var(--glide) var(--ease); }
  .chev.open { rotate: 180deg; }

  /* A phone on its side: two columns, the conversation (thread row, log) and beside it the composer
   * at the foot, or the inspector, which takes the whole column while it shows. The loom page
   * scrolls instead, its map most of a screen tall with room above it to scroll back by. */
  @media (max-height: 520px) {
    .phone:not(.loom-page) .dw-main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) clamp(260px, 40%, 360px);
      grid-template-rows: auto auto minmax(0, 1fr) auto;
      column-gap: var(--s4);
    }
    .phone:not(.loom-page) .thread-fold { grid-column: 1; grid-row: 1; }
    .phone:not(.loom-page) .dw-thread { grid-column: 1; grid-row: 2; }
    .phone :global(.chat .log) { grid-column: 1; grid-row: 3 / 5; }
    .phone :global(.chat .raw-host) { grid-column: 1 / -1; grid-row: 3 / 5; }
    .phone.sheet-shown :global(.chat .raw-host) { grid-column: 1; }
    .phone :global(.chat .dock) { grid-column: 2; grid-row: 4; padding-inline: 0 var(--s4); }
    .phone .sheet, .phone .sheet.open { grid-column: 2; grid-row: 1 / -1; height: auto; margin: 0 var(--s4) var(--s4) 0; border-radius: var(--r-row); }
    .phone .grab { display: none; }
    .phone.sheet-shown :global(.chat .dock) { display: none; }
    .phone.loom-page { height: auto; overflow: visible; }
    .phone.loom-page .dw-main { height: auto; }
    .phone.loom-page .loom { flex: none; overflow: visible; }
    .phone.loom-page :global(.tree-scroll) { min-height: calc(100dvh - 72px); }
  }

  /* The sheet shrinks rather than overflow when the thread or the dock leave it less room. */
  .sheet {
    position: relative;
    z-index: 2;
    flex: 0 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    height: min(380px, 50dvh);
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
