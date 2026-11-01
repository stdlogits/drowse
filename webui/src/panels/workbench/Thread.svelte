<script lang="ts">
  // The thread: the loom as the path you are on. One row per turn on the active path (a pip, the
  // turn's first lines, and which of its siblings it is), the chat's other saved threads under
  // Earlier, and the loom's map and weave one step away in the Page region.
  import { onMount } from "svelte";
  import { loomNavigate, loomTree } from "../../lib/stores/loom.svelte";
  import { loomUiState } from "../../lib/stores/loomUi.svelte";
  import { loomNodeSnippet } from "../../lib/loomSearch";
  import { genStatus } from "../../lib/stores/chat.svelte";
  import { sessionState } from "../../lib/stores/session.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import { conversationLibrary, onConversationLibraryChanged, savedConversationState } from "../../lib/stores/savedConversations.svelte";
  import { openDrawer } from "../../lib/stores/drawers.svelte";
  import { showLoom } from "../../lib/stores/workbench.svelte";
  import { openSavedChat } from "../../lib/savedChatOpen";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import type { SavedConversationSummary } from "../../lib/conversationLibrary";
  import type { LoomNodeJSON } from "../../lib/types";

  interface Row {
    node: LoomNodeJSON;
    model: boolean;
    snippet: string;
    meta: string;
  }

  const rows = $derived.by<Row[]>(() => {
    const out: Row[] = [];
    for (const id of loomTree.activePath) {
      const node = loomTree.nodes.get(id);
      if (!node) continue;
      if (node.parent_id === null && node.role === "system" && !node.text) continue;
      const siblings = node.parent_id ? loomTree.children_of.get(node.parent_id) ?? [] : [];
      const parts: string[] = [];
      if (siblings.length > 1) parts.push(`${siblings.indexOf(id) + 1} of ${siblings.length}`);
      if (node.applied_steering) parts.push("steered");
      if (node.role === "system") parts.unshift("system");
      out.push({
        node,
        model: node.role === "assistant",
        snippet: loomNodeSnippet(node, node.id === loomTree.pendingNodeId ? "Writing…" : "Empty"),
        meta: parts.join(" · "),
      });
    }
    return out;
  });

  const branches = $derived.by(() => {
    let leaves = 0;
    for (const [id] of loomTree.nodes) {
      if (id !== loomTree.root_id && (loomTree.children_of.get(id)?.length ?? 0) === 0) leaves += 1;
    }
    return leaves;
  });

  let saved = $state<SavedConversationSummary[]>([]);
  const current = $derived(saved.find(row => row.id === savedConversationState.activeId) ?? null);
  const earlier = $derived(saved
    .filter(row => row.id !== savedConversationState.activeId && row.modelId === sessionState.info?.model_id)
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, 4));
  const name = $derived.by(() => {
    if (current?.name) return current.name;
    if (loomTree.name) return loomTree.name;
    const first = rows.find(row => row.node.role === "user")?.snippet;
    if (!first) return "New conversation";
    return first.length > 40 ? `${first.slice(0, 40).trimEnd()}…` : first;
  });

  async function refreshSaved(): Promise<void> {
    try { saved = (await conversationLibrary.listSummaries()).conversations; }
    catch { saved = []; }
  }

  onMount(() => {
    void refreshSaved();
    return onConversationLibraryChanged(refreshSaved);
  });
  $effect(() => {
    void savedConversationState.activeId;
    void refreshSaved();
  });

  async function go(row: Row): Promise<void> {
    if (row.node.id === loomTree.active_node_id) return;
    try { await loomNavigate(row.node.id); }
    catch (error) { pushToast(userFacingError(error, "That point in the thread could not be opened."), { kind: "error" }); }
  }

  let opening = $state<string | null>(null);
  async function open(row: SavedConversationSummary): Promise<void> {
    if (genStatus.active || opening) return;
    opening = row.id;
    try {
      const { name: opened } = await openSavedChat(row.id);
      pushToast(`Opened “${opened}”.`, { kind: "info" });
    } catch (error) {
      pushToast(userFacingError(error, "This conversation could not be opened. Your current work was left unchanged."), { kind: "error" });
    } finally { opening = null; }
  }

  function view(next: "map" | "weave"): void {
    loomUiState.view = next;
    showLoom();
  }
</script>

<div class="dw-sec-head">
  <span class="dw-thread-name">{name}</span>
  <span class="meta">{branches} {branches === 1 ? "branch" : "branches"}</span>
</div>

{#if rows.length}
  <div class="dw-nodes" aria-label="The path you are on">
    {#each rows as row (row.node.id)}
      <button type="button" class="dw-nodeRow" class:dw-indent={row.model}
        aria-current={row.node.id === loomTree.active_node_id ? "true" : undefined} onclick={() => void go(row)}>
        <span class="dw-pip" class:is-model={row.model}></span>
        <span><span class="dw-snip">{row.snippet}</span>{#if row.meta}<span class="meta">{row.meta}</span>{/if}</span>
      </button>
    {/each}
  </div>
{:else}
  <p class="meta empty">Nothing yet</p>
{/if}

<div class="dw-row">
  <button type="button" class="dw-text" onclick={() => view("map")}>Map</button>
  <button type="button" class="dw-text" onclick={() => view("weave")}>Weave</button>
</div>

<div class="dw-sec">
  <span class="head">Earlier</span>
  {#if earlier.length}
    <div class="dw-nodes">
      {#each earlier as row (row.id)}
        <button type="button" class="dw-nodeRow" aria-busy={opening === row.id} disabled={genStatus.active} onclick={() => void open(row)}>
          <span class="dw-pip"></span>
          <span><span class="dw-snip">{row.name}</span><span class="meta">{row.threadCount} {row.threadCount === 1 ? "branch" : "branches"}</span></span>
        </button>
      {/each}
    </div>
  {/if}
  <button type="button" class="dw-text all" onclick={() => openDrawer("load_conversation")}>Saved chats</button>
</div>

<style>
  .empty { padding: 0 var(--s3); }
  .all { justify-self: start; }
  .dw-nodeRow:disabled { opacity: 0.4; cursor: default; }
</style>
