<script lang="ts">
  import { openSavedChat } from "../lib/savedChatOpen";
  import { onConversationLibraryChanged } from "../lib/stores/savedConversations.svelte";
  import BaseModelTag from "../lib/ui/BaseModelTag.svelte";
  import { flushConversationAutosave } from "../lib/stores/savedConversations.svelte";
  import { onMount, tick } from "svelte";
  import { downloadChatBackup, importChatBackup } from "../lib/chatBackup";

  import { IconButton } from "@stdlogits/ledger";
  import {
    conversationLibrary,
    closeDrawer,
    genStatus,
    openDrawer,
    requestPersistentConversationStorage,
    savedConversationState,
    sessionState,
  } from "../lib/stores.svelte";
  import {
    displayModelName,
    randomAvatarSeed,
    summarizeConversation,
    type SavedConversationIssue,
    type SavedConversationRecord,
    type SavedConversationSummary,
  } from "../lib/conversationLibrary";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import { pushToast } from "../lib/stores/toasts.svelte";

  let fileInput: HTMLInputElement | null = $state(null);
  let conversations: SavedConversationSummary[] = $state([]);
  let issues: SavedConversationIssue[] = $state([]);
  let query = $state("");
  let loading = $state(true);
  let error = $state<string | null>(null);
  let openingId = $state<string | null>(null);
  let changingId = $state<string | null>(null);
  let duplicatingId = $state<string | null>(null);
  let confirmDeleteId = $state<string | null>(null);
  let editingId = $state<string | null>(null);
  let editingName = $state("");
  let importing = $state(false);
  let renameInput: HTMLInputElement | null = $state(null);
  let persistenceState: "checking" | "protected" | "unprotected" | "unavailable" = $state("checking");
  let persistenceRequesting = $state(false);
  let persistenceAttempted = $state(false);

  const currentModelId = $derived(sessionState.info?.model_id ?? null);
  const filtered = $derived.by(() => {
    const needle = query.trim().toLocaleLowerCase();
    if (!needle) return conversations;
    return conversations.filter((record) =>
      record.name.toLocaleLowerCase().includes(needle) ||
      displayModelName(record.modelId).toLocaleLowerCase().includes(needle)
    );
  });

  onMount(() => {
    void refresh();
    void refreshPersistenceState();
  });

  async function refreshPersistenceState(): Promise<void> {
    if (!navigator.storage?.persisted) {
      persistenceState = "unavailable";
      return;
    }
    try {
      persistenceState = await navigator.storage.persisted()
        ? "protected"
        : "unprotected";
    } catch {
      persistenceState = "unavailable";
    }
  }

  async function protectStorage(): Promise<void> {
    persistenceRequesting = true;
    try {
      const protectedStorage = await requestPersistentConversationStorage();
      persistenceState = protectedStorage === true
        ? "protected"
        : protectedStorage === false
          ? "unprotected"
          : "unavailable";
      persistenceAttempted = protectedStorage !== true;
      if (protectedStorage) pushToast("Saved chats are protected from automatic browser cleanup.", { kind: "info" });
    } finally {
      persistenceRequesting = false;
    }
  }

  onMount(() => onConversationLibraryChanged(refresh));

  async function refresh(): Promise<void> {
    loading = true;
    error = null;
    try {
      const result = await conversationLibrary.listSummaries();
      conversations = result.conversations;
      issues = result.issues;
    } catch (cause) {
      error = userFacingError(cause, "Saved conversations could not be opened.");
    } finally {
      loading = false;
    }
  }

  async function openConversation(record: SavedConversationSummary): Promise<void> {
    if (record.modelId !== currentModelId || genStatus.active) return;
    openingId = record.id;
    error = null;
    try {
      await openSavedChat(record.id);
      pushToast(`Opened “${record.name}”.`, { kind: "info" });
      closeDrawer();
    } catch (cause) {
      error = userFacingError(
        cause,
        "This conversation could not be opened. Your current work was left unchanged.",
      );
    } finally {
      openingId = null;
    }
  }

  function beginRename(record: SavedConversationSummary): void {
    confirmDeleteId = null;
    editingId = record.id;
    editingName = record.name;
    void tick().then(() => {
      renameInput?.focus();
      renameInput?.select();
    });
  }

  async function saveRename(record: SavedConversationSummary): Promise<void> {
    changingId = record.id;
    error = null;
    try {
      const updated = await conversationLibrary.update(record.id, { name: editingName });
      replaceConversation(updated);
      editingId = null;
    } catch (cause) {
      error = userFacingError(cause, "The conversation could not be renamed.");
    } finally {
      changingId = null;
    }
  }

  function onRenameKey(event: KeyboardEvent, record: SavedConversationSummary): void {
    if (event.key === "Enter") {
      event.preventDefault();
      if (editingName.trim() && changingId !== record.id) void saveRename(record);
    } else if (event.key === "Escape") {
      event.preventDefault();
      editingId = null;
    }
  }

  function requestDelete(id: string): void {
    editingId = null;
    confirmDeleteId = id;
  }

  async function deleteConversation(id: string): Promise<void> {
    changingId = id;
    error = null;
    try {
      await conversationLibrary.delete(id);
      conversations = conversations.filter((record) => record.id !== id);
      issues = issues.filter((issue) => issue.id !== id);
      confirmDeleteId = null;
      pushToast("Saved conversation deleted.", { kind: "info" });
    } catch (cause) {
      error = userFacingError(cause, "The saved conversation could not be deleted.");
    } finally {
      changingId = null;
    }
  }

  async function download(summary: SavedConversationSummary): Promise<void> {
    changingId = summary.id;
    error = null;
    try {
      await flushConversationAutosave();
      await downloadChatBackup(conversationLibrary, summary.id);
    } catch (cause) {
      error = userFacingError(cause, "This conversation could not be exported. Your saved chat was not changed.");
    } finally {
      changingId = null;
    }
  }

  async function importFile(event: Event): Promise<void> {
    const target = event.currentTarget as HTMLInputElement;
    const file = target.files?.[0] ?? null;
    target.value = "";
    if (!file || importing) return;
    importing = true;
    error = null;
    try {
      const record = await importChatBackup(file, conversationLibrary);
      conversations = [summarizeConversation(record), ...conversations];
      pushToast(`Imported “${record.name}” as a separate chat.`, { kind: "info" });
      void requestPersistentConversationStorage();
    } catch (cause) {
      error = userFacingError(cause, "This conversation file could not be imported.");
    } finally { importing = false; }
  }

  async function duplicateConversation(record: SavedConversationSummary): Promise<void> {
    if (changingId !== null || importing || openingId !== null) return;
    changingId = record.id;
    duplicatingId = record.id;
    error = null;
    try {
      await flushConversationAutosave();
      const copy = await conversationLibrary.duplicate(record.id);
      conversations = [summarizeConversation(copy), ...conversations];
      pushToast(`Created “${copy.name}”. The original chat is unchanged.`, { kind: "info" });
    } catch (cause) {
      error = userFacingError(cause, "The chat could not be duplicated. The original chat was not changed.");
    } finally {
      changingId = null;
      duplicatingId = null;
    }
  }

  function replaceConversation(updated: SavedConversationRecord): void {
    conversations = conversations.map((record) => record.id === updated.id ? summarizeConversation(updated) : record);
  }

  function relativeTime(timestamp: number): string {
    const elapsed = timestamp - Date.now();
    const units: Array<[Intl.RelativeTimeFormatUnit, number]> = [
      ["year", 365 * 24 * 60 * 60 * 1000],
      ["month", 30 * 24 * 60 * 60 * 1000],
      ["day", 24 * 60 * 60 * 1000],
      ["hour", 60 * 60 * 1000],
      ["minute", 60 * 1000],
    ];
    for (const [unit, duration] of units) {
      if (Math.abs(elapsed) >= duration || unit === "minute") {
        const value = Math.round(elapsed / duration);
        return new Intl.RelativeTimeFormat(undefined, { numeric: "auto" }).format(value, unit);
      }
    }
    return "just now";
  }

  function absoluteTime(timestamp: number): string {
    return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" })
      .format(new Date(timestamp));
  }
</script>

<section class="drawer-shell" aria-label="Saved conversations drawer">
  <header class="header">
    <h2 class="dw-title">Saved chats</h2>
    <IconButton icon="close" label="Close" onclick={closeDrawer} />
  </header>

  <div class="toolbar">
    <label class="search-field">
      <span class="sr-only">Search saved chats</span>
      <input bind:value={query} type="search" placeholder="Search saved chats" />
    </label>
    <input
      bind:this={fileInput}
      class="file-input"
      type="file"
      accept=".drowsechat,.polytheticchat,.saklaschat,.json,application/json"
      aria-label="Import Drowse conversation file"
      onchange={importFile}
      tabindex="-1"
    />
    <div class="dw-row">
      <button type="button" class="dw-button" disabled={genStatus.active} onclick={() => openDrawer("save_conversation")}>Save current chat</button>
      <button type="button" class="dw-quiet" disabled={importing} aria-busy={importing} onclick={() => fileInput?.click()}>{importing ? "Importing…" : "Import file"}</button>
    </div>
  </div>

  {#if !loading && conversations.length > 0 && persistenceState === "unprotected"}
    <div class="storage-protection" role="status">
      <div>
        <strong>Protect saved chats</strong>
        {#if persistenceAttempted}<p class="meta">The browser declined; your chats are still saved here.</p>{/if}
      </div>
      <button
        type="button"
        class="dw-quiet"
        disabled={persistenceRequesting}
        onclick={() => void protectStorage()}
      >{persistenceRequesting ? "Checking…" : persistenceAttempted ? "Try again" : "Protect"}</button>
    </div>
  {/if}

  <div class="body">
    {#if error}<p class="error" role="alert">{error}</p>{/if}
    {#if genStatus.active}<p class="notice">Opening another chat waits for the reply to finish.</p>{/if}

    {#if loading}
      <p class="meta">Loading…</p>
    {:else if filtered.length === 0 && issues.length === 0}
      <p class="meta">{query.trim() ? "No matches" : "No saved chats yet"}</p>
    {:else}
      <div class="conversation-list" role="list" aria-label="Saved conversations">
        {#each filtered as record (record.id)}
          {@const matchesModel = record.modelId === currentModelId}
          <article
            class="conversation-card"
            role="listitem"
            class:current={savedConversationState.activeId === record.id}
            data-saved-conversation={record.id}
          >
            <div class="card-main">
              <div class="card-copy">
                {#if editingId === record.id}
                  <label class="rename-field">
                    <span class="sr-only">Conversation name</span>
                    <input
                      bind:this={renameInput}
                      bind:value={editingName}
                      maxlength="120"
                      onkeydown={(event) => onRenameKey(event, record)}
                      disabled={changingId === record.id}
                    />
                  </label>
                {:else}
                  <strong {...{ "aria-description": (record.name) }} dir="auto">{record.name}</strong>
                {/if}
                <span class="meta card-meta">
                  <span class="model">{displayModelName(record.modelId)}{#if record.modelType === "base" || (matchesModel && sessionState.info?.is_base_model)}<BaseModelTag />{/if}</span>
                  {#if savedConversationState.activeId === record.id}<span>open now</span>{/if}
                </span>
              </div>

              <time class="meta" datetime={new Date(record.updatedAt).toISOString()} {...{ "aria-description": (absoluteTime(record.updatedAt)) }}>
                {relativeTime(record.updatedAt)}
              </time>
            </div>

            {#if editingId === record.id}
              <div class="card-actions">
                <button
                  type="button"
                  class="dw-quiet"
                  onclick={() => void saveRename(record)}
                  disabled={!editingName.trim() || changingId === record.id}
                >Save name</button>
                <button type="button" class="dw-text" onclick={() => (editingId = null)}>Cancel</button>
              </div>
            {:else if confirmDeleteId === record.id}
              <div class="card-actions confirm-row" role="group" aria-label={`Confirm deletion of ${record.name}`}>
                <p>Delete “{record.name}” for good?</p>
                <button
                  type="button"
                  class="dw-button"
                  onclick={() => void deleteConversation(record.id)}
                  disabled={changingId === record.id}
                >Delete</button>
                <button type="button" class="dw-text" onclick={() => (confirmDeleteId = null)}>Cancel</button>
              </div>
            {:else}
              <div class="card-actions">
                <button
                  type="button"
                  class="dw-quiet"
                  onclick={() => void openConversation(record)}
                  disabled={!matchesModel || genStatus.active || openingId !== null}
                  {...{ "aria-description": (matchesModel ? undefined : `Load ${displayModelName(record.modelId)} to open this chat`) }}
                >{openingId === record.id ? "Opening…" : matchesModel ? "Open" : "Different model"}</button>
                <button type="button" class="dw-text" onclick={() => beginRename(record)}>Rename</button>
                <button type="button" class="dw-text" disabled={changingId !== null} onclick={() => void download(record)}>Download</button>
                <button type="button" class="dw-text" disabled={changingId !== null || importing || openingId !== null}
                  aria-label={`Duplicate ${record.name}`} aria-busy={duplicatingId === record.id}
                  onclick={() => void duplicateConversation(record)}>Duplicate</button>
                <button type="button" class="dw-text" onclick={() => requestDelete(record.id)}>Delete</button>
              </div>
            {/if}
          </article>
        {/each}

        {#each issues as issue (issue.id)}
          <article class="conversation-card unavailable" role="listitem">
            <div class="card-main">
              <div class="card-copy">
                <strong>{issue.name || "Unavailable saved chat"}</strong>
                <span class="meta">Could not be read; the data is still in storage</span>
              </div>
            </div>
            {#if confirmDeleteId === issue.id}
              <div class="card-actions confirm-row" role="group" aria-label="Confirm deletion of unavailable saved chat">
                <p>Delete this saved data for good?</p>
                <button type="button" class="dw-button" onclick={() => void deleteConversation(issue.id)}>Delete</button>
                <button type="button" class="dw-text" onclick={() => (confirmDeleteId = null)}>Cancel</button>
              </div>
            {:else}
              <div class="card-actions">
                <span
                  class="meta"
                  {...{ "aria-description": (userFacingError(issue.reason, "This saved chat could not be read. The data is still in storage.")) }}
                >Unreadable</span>
                <button type="button" class="dw-text" onclick={() => requestDelete(issue.id)}>Delete</button>
              </div>
            {/if}
          </article>
        {/each}
      </div>
    {/if}
  </div>
</section>

<style>
  /* Saved chats: a search, then one row per chat, the open one in wash2; its actions as quiet text. */
  .drawer-shell { display: flex; flex-direction: column; gap: var(--s4); height: 100%; min-height: 0; padding: var(--s5) var(--s6) 0; color: var(--ink2); }
  .header { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); }
  .toolbar { display: grid; gap: var(--s3); }
  .search-field input { width: 100%; height: 40px; padding: 0 var(--s4); border: 0; border-radius: var(--r-pill); background: var(--wash); color: var(--ink); font: var(--type-small); }
  .search-field input::-webkit-search-cancel-button { -webkit-appearance: none; }
  .file-input { display: none; }
  .storage-protection { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s2) var(--s4); padding: var(--s3) var(--s3) var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .storage-protection strong { color: var(--ink); font: var(--type-control); }
  .storage-protection .dw-quiet { background: var(--wash2); }
  .body { flex: 1 1 auto; min-height: 0; overflow-y: auto; overscroll-behavior: contain; margin: 0 calc(var(--s3) * -1); padding: 0 0 var(--s6); }
  .body > p { margin: 0 var(--s3) var(--s3); }
  .notice, .error { color: var(--ink); font: var(--type-small); }
  .conversation-list { display: grid; gap: 2px; }
  .conversation-card { display: grid; gap: var(--s2); min-width: 0; padding: var(--s3); border-radius: var(--r-box); transition: background-color var(--fast) var(--ease); }
  .conversation-card:hover { background: var(--wash); }
  .conversation-card.current { background: var(--wash2); }
  .conversation-card.unavailable { color: var(--ink3); }
  .card-main { display: flex; align-items: baseline; justify-content: space-between; gap: var(--s4); min-width: 0; }
  .card-copy { display: grid; gap: 2px; min-width: 0; }
  .card-copy strong { overflow: hidden; color: var(--ink); font: var(--type-label); text-overflow: ellipsis; white-space: nowrap; }
  .card-meta { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s3); }
  .model { display: inline-flex; align-items: center; gap: var(--s2); }
  time { flex: none; white-space: nowrap; }
  .rename-field input { width: 100%; height: 40px; padding: 0 var(--s3); border: 0; border-radius: var(--r-col); background: var(--paper); color: var(--ink); font: var(--type-label); }
  .card-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s4); }
  .card-actions .dw-quiet { height: 32px; padding: 0 var(--s4); }
  .conversation-card:hover .dw-quiet, .conversation-card.current .dw-quiet { background: var(--paper); }
  .confirm-row p { color: var(--ink); font: var(--type-small); }
  @media (max-width: 720px) {
    .drawer-shell { padding: var(--s4) var(--s4) 0; }
    .search-field input, .rename-field input { font-size: 16px; }
    .card-actions .dw-quiet { height: 44px; }
  }
</style>
