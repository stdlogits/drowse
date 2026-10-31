<script lang="ts">
  import { registerInterfaceController } from "../lib/workspaceController";
  import { objectSchema, ToolError } from "../lib/webmcp/types";

  import { MorphText } from "@stdlogits/ledger";
  import { onMount } from "svelte";
  import { CONVERSATION_ACCENTS, type ConversationAccent } from "../lib/conversationLibrary";

  import { IconButton } from "@stdlogits/ledger";
  import {
    conversationLibrary,
    closeDrawer,
    genStatus,
    requestPersistentConversationStorage,
    savedConversationState,
  } from "../lib/stores.svelte";
  import {
    defaultConversationName,
    displayModelName,
    randomAvatarSeed,
    ConversationLibraryError,
    SAVED_CONVERSATION_SCHEMA_VERSION,
    type SavedConversationRecord,
  } from "../lib/conversationLibrary";
  import { captureConversationSnapshot } from "../lib/conversationWorkspace";
  import { downloadPreparedChatBackup, encodeChatBackup } from "../lib/chatBackup";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import { flushConversationAutosave, onConversationLibraryChanged } from "../lib/stores/savedConversations.svelte";

  let { embedded = false }: { params?: unknown; embedded?: boolean } = $props();

  let draftBaseline = $state("");
  let current: SavedConversationRecord | null = $state(null);
  let name = $state("");
  let avatarSeed = $state(randomAvatarSeed());
  let accent: ConversationAccent = $state("purple");
  const metadataDirty = $derived(draftBaseline !== "" && JSON.stringify([name, avatarSeed, accent]) !== draftBaseline);

  let error = $state<string | null>(null);
  let loading = $state(true);
  let saving = $state(false);
  let modelName = $state("Current model");
  let turnCount = $state(0);

  onMount(() => {
    void initialize();
  });
  onMount(() => onConversationLibraryChanged(() => { if (!saving && !metadataDirty) return initialize(); }));

  async function initialize(): Promise<void> {
    loading = true;
    error = null;
    try {
      await flushConversationAutosave().catch(() => undefined);
      const snapshot = captureConversationSnapshot();
      modelName = displayModelName(snapshot.model_id);
      turnCount = snapshot.tree.nodes.filter((node) => node.parent_id !== null).length;
      name = defaultConversationName();
      const activeId = savedConversationState.activeId;
      if (activeId) {
        try {
          const saved = await conversationLibrary.get(activeId);
          if (saved.modelId === snapshot.model_id) {
            current = saved;
            name = saved.name;
            avatarSeed = saved.avatarSeed;
            accent = saved.accent ?? "purple";
          } else {
            savedConversationState.activeId = null;
            savedConversationState.avatarSeed = null;
            savedConversationState.accent = "purple";
          }
        } catch (cause) {
          if (cause instanceof ConversationLibraryError && cause.code === "NOT_FOUND") {
            savedConversationState.activeId = null;
            savedConversationState.avatarSeed = null;
            savedConversationState.accent = "purple";
          } else {
            throw cause;
          }
        }
      }
    } catch (cause) {
      error = userFacingError(cause, "This conversation is not ready to save yet.");
    } finally {
      draftBaseline = JSON.stringify([name, avatarSeed, accent]);
      loading = false;
    }
  }

  async function save(asNew = false): Promise<void> {
    error = null;
    if (genStatus.active) {
      error = "Wait for the current reply to finish before saving.";
      return;
    }
    saving = true;
    try {
      await flushConversationAutosave().catch(() => undefined);
      const persistenceProtected = await requestPersistentConversationStorage();
      const snapshot = captureConversationSnapshot();
      const activeId = current?.id ?? savedConversationState.activeId;
      const record = activeId && !asNew
        ? await conversationLibrary.update(activeId, { name, avatarSeed, accent, snapshot })
        : await conversationLibrary.create({ name, avatarSeed, accent, snapshot });
      current = record;
      draftBaseline = JSON.stringify([record.name, record.avatarSeed, record.accent ?? "purple"]);
      name = record.name;
      avatarSeed = record.avatarSeed;
      savedConversationState.activeId = record.id;
      savedConversationState.avatarSeed = record.avatarSeed;
      savedConversationState.accent = record.accent ?? "purple";
      savedConversationState.status = "saved";
      savedConversationState.error = null;
      pushToast(
        persistenceProtected === false
          ? `Saved “${record.name}”. Browser cleanup protection is off.`
          : `Saved “${record.name}”.`,
        { kind: persistenceProtected === false ? "warning" : "info" },
      );
      if (!embedded) closeDrawer();
    } catch (cause) {
      error = userFacingError(cause, "This conversation could not be saved.");
    } finally {
      saving = false;
    }
  }

  let backupState = $state("idle");
  $effect(() => { if (backupState !== "ready") return; const timer = setTimeout(() => backupState = "idle", 1800); return () => clearTimeout(timer); });
  async function downloadCopy(): Promise<void> {
    if (saving || loading || genStatus.active) return;
    backupState = "preparing";
    saving = true;
    error = null;
    try {
      const snapshot = captureConversationSnapshot();
      const now = Date.now();
      const record: SavedConversationRecord = {
        schemaVersion: SAVED_CONVERSATION_SCHEMA_VERSION,
        id: current?.id ?? crypto.randomUUID(),
        name: name.trim() || defaultConversationName(now),
        avatarSeed,
        accent,
        modelId: snapshot.model_id,
        createdAt: current?.createdAt ?? now,
        updatedAt: Math.max(current?.updatedAt ?? now, now),
        snapshot,
      };
      const blob = new Blob([await encodeChatBackup(record)], { type: "application/json" });
      downloadPreparedChatBackup(blob, record.name);
      backupState = "ready";
    } catch (cause) {
      backupState = "idle";
      error = userFacingError(cause, "A backup copy could not be created.");
    } finally { saving = false; }
  }

  onMount(() => registerInterfaceController("chat_identity", {
    schema: objectSchema({ name: { type: "string", maxLength: 120 }, avatar_seed: { type: "string", maxLength: 256 }, accent: { type: "string", enum: [...CONVERSATION_ACCENTS] } }),
    read: () => ({ busy: loading || saving, dirty: metadataDirty, chat_id: current?.id ?? savedConversationState.activeId, values: { name, avatar_seed: avatarSeed, accent } }),
    update: change => {
      if (loading || saving) throw new ToolError("BUSY", "Wait for the saved-chat form.");
      if (change.name !== undefined) name = change.name as string;
      if (change.avatar_seed !== undefined) avatarSeed = change.avatar_seed as string;
      if (change.accent !== undefined) accent = change.accent as ConversationAccent;
    },
  }));
</script>

<section class="drawer-shell" class:embedded aria-label={embedded ? "Save and name chat" : "Save conversation drawer"}>
  <header class="header">
    <h2 class="dw-title"><MorphText text={embedded ? "Save and name chat" : current ? "Update saved chat" : "Save chat"} /></h2>
    {#if !embedded}<IconButton icon="close" label="Close" onclick={closeDrawer} />{/if}
  </header>

  <div class="body">
    <label class="name-field">
      <span class="meta">Name</span>
      <input
        bind:value={name}
        maxlength="120"
        autocomplete="off"
        placeholder="Untitled conversation"
        disabled={loading || saving}
      />
    </label>

    <div class="dw-facts" aria-label="Conversation summary">
      <span><b>{modelName}</b></span>
      <span>{turnCount} {turnCount === 1 ? "turn" : "turns"}</span>
      <span>kept in this browser</span>
    </div>

    {#if genStatus.active}
      <p class="notice">Saving waits for the reply to finish.</p>
    {/if}
    {#if error}
      <p class="error" role="alert">{error}</p>
    {/if}
  </div>

  <footer class="footer">
    <button type="button" class="dw-text" onclick={downloadCopy} disabled={loading || saving || genStatus.active}>
      <MorphText text={backupState === "preparing" ? "Preparing copy…" : backupState === "ready" ? "Download started" : "Download copy"} numbers={false} />
    </button>
    <div class="footer-actions">
      {#if current}
        <button type="button" class="dw-quiet" onclick={() => void save(true)} disabled={loading || saving || genStatus.active}>
          Save as new
        </button>
      {/if}
      <button
        type="button"
        class="dw-button"
        aria-busy={saving}
        onclick={() => void save(false)}
        disabled={loading || saving || genStatus.active || !name.trim()}
      >
        <MorphText text={saving ? "Saving…" : current ? "Update" : "Save"} />
      </button>
    </div>
  </footer>
</section>

<style>
  .drawer-shell { display: flex; flex-direction: column; gap: var(--s5); height: 100%; min-height: 0; padding: var(--s5) var(--s6) var(--s6); color: var(--ink2); }
  .drawer-shell.embedded { height: auto; padding: 0; }
  .header { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); }
  .embedded .dw-title { font: var(--type-lead); letter-spacing: 0; }
  .body { display: flex; flex: 1 1 auto; flex-direction: column; gap: var(--s4); min-height: 0; margin: -4px; padding: 4px; overflow-y: auto; }
  .embedded .body { flex: none; overflow: visible; }
  .name-field { display: grid; gap: var(--s1); min-width: 0; }
  .name-field input { width: 100%; min-width: 0; height: 48px; padding: 0 var(--s4); border: 0; border-radius: var(--r-box); background: var(--wash); color: var(--ink); font: var(--type-label); }
  .notice, .error { color: var(--ink); font: var(--type-small); }
  .footer { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s3) var(--s4); }
  .footer-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2); margin-left: auto; }
  @media (max-width: 720px) {
    .drawer-shell:not(.embedded) { padding: var(--s4) var(--s4) var(--s5); }
    .name-field input { font-size: 16px; }
  }
</style>
