<script lang="ts">
  import { onMount } from "svelte";
  import { IconButton } from "@stdlogits/ledger";
  import { closeDrawer, genStatus } from "../lib/stores.svelte";
  import { conversationLibrary, savedConversationState } from "../lib/stores/savedConversations.svelte";
  import { captureConversationSnapshot } from "../lib/conversationWorkspace";
  import { ConversationLibraryError, defaultConversationName, randomAvatarSeed, SAVED_CONVERSATION_SCHEMA_VERSION } from "../lib/conversationLibrary";
  import { chatBackupFilename, downloadPreparedChatBackup, encodeChatBackup } from "../lib/chatBackup";
  import { userFacingError } from "../lib/runtime/userFacingError";

  let { params: _params }: { params?: unknown } = $props();
  let name = $state("");
  let blob = $state<Blob | null>(null);
  let error = $state<string | null>(null);
  let loading = $state(true);
  let downloading = $state(false);
  const filename = $derived(chatBackupFilename(name));

  onMount(() => { void prepare(); });

  async function prepare(): Promise<void> {
    loading = true;
    error = null;
    blob = null;
    try {
      if (genStatus.active) throw new Error("Wait for the reply to finish, then try again.");
      const snapshot = captureConversationSnapshot();
      const activeId = savedConversationState.activeId;
      let saved = null;
      if (activeId) {
        try { saved = await conversationLibrary.get(activeId); }
        catch (cause) {
          if (!(cause instanceof ConversationLibraryError && cause.code === "NOT_FOUND")) throw cause;
        }
      }
      if (saved?.modelId !== snapshot.model_id) saved = null;
      const now = Date.now();
      const record = {
        schemaVersion: SAVED_CONVERSATION_SCHEMA_VERSION,
        id: saved?.id ?? crypto.randomUUID(),
        name: saved?.name ?? defaultConversationName(now),
        avatarSeed: saved?.avatarSeed ?? randomAvatarSeed(),
        accent: saved?.accent ?? savedConversationState.accent,
        modelId: snapshot.model_id,
        createdAt: saved?.createdAt ?? now,
        updatedAt: Math.max(saved?.updatedAt ?? now, now),
        snapshot,
      };
      name = record.name;
      blob = new Blob([await encodeChatBackup(record)], { type: "application/json" });
    } catch (cause) {
      error = userFacingError(cause, "This chat could not be prepared for download.");
    } finally { loading = false; }
  }

  function download(event: SubmitEvent): void {
    event.preventDefault();
    if (!blob || !name.trim() || downloading) return;
    downloading = true;
    try {
      downloadPreparedChatBackup(blob, name);
      closeDrawer();
    } catch (cause) {
      error = userFacingError(cause, "The download could not start. Try again.");
      downloading = false;
    }
  }
</script>

<form class="download-chat" onsubmit={download}>
  <header>
    <h2 class="dw-title">Download chat</h2>
    <IconButton icon="close" label="Close" onclick={closeDrawer} />
  </header>
  <p class="meta">Every message and branch, the settings and the readings; no model files.</p>
  {#if loading}
    <p class="meta" role="status">Preparing…</p>
  {:else if blob}
    <label class="name-field">
      <span class="meta">File name</span>
      <input id="chat-download-name" bind:value={name} maxlength="120" required autocomplete="off" />
    </label>
    <div class="file-summary">
      <span class="filename">{filename}</span>
      <span class="meta" aria-label="Backup size" data-bytes={blob.size}>{blob.size.toLocaleString()} bytes</span>
    </div>
  {/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
  <footer>
    <button type="button" class="dw-text" onclick={closeDrawer}>Cancel</button>
    {#if error && !blob}
      <button type="button" class="dw-button" onclick={() => void prepare()} disabled={loading}>Try again</button>
    {:else}
      <button type="submit" class="dw-button" disabled={!blob || !name.trim() || downloading}>Download</button>
    {/if}
  </footer>
</form>

<style>
  .download-chat { display: grid; gap: var(--s4); min-width: 0; padding: var(--s5) var(--s6) var(--s6); }
  header { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); }
  .name-field { display: grid; gap: var(--s1); min-width: 0; }
  input { width: 100%; min-width: 0; height: 40px; padding: 0 var(--s3); border: 0; border-radius: var(--r-col); background: var(--wash); color: var(--ink); font: var(--type-body); }
  .file-summary { display: grid; gap: 2px; padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .filename { color: var(--ink); font: var(--type-small); overflow-wrap: anywhere; }
  .error { color: var(--ink); }
  footer { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: var(--s4); padding-top: var(--s2); }
  @media (max-width: 720px) {
    .download-chat { padding: var(--s4) var(--s4) var(--s5); }
    input { font-size: 16px; }
  }
</style>
