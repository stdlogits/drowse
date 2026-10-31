<script lang="ts">
  import { onMount } from "svelte";
  import { registerSystemPromptController } from "../lib/workspaceController";
  import { IconButton, MorphText } from "@stdlogits/ledger";
  import "./manifold/form.css";
  // System-prompt drawer — edit the session's default system prompt.
  // Saves via PATCH /sessions/{id} (``patchSessionDefaults``); cancel
  // closes without writing.

  import {
    sessionState,
    patchSessionDefaults,
    closeDrawer,
  } from "../lib/stores.svelte";
  import { ApiError } from "../lib/runtime/services";
  import { userFacingError } from "../lib/runtime/userFacingError";

  let value = $state(sessionState.info?.config.system_prompt ?? "");
  let busy = $state(false);
  let errorMsg: string | null = $state(null);
  onMount(() => registerSystemPromptController({
    read: () => ({ busy, dirty: value !== (sessionState.info?.config.system_prompt ?? ""), values: { text: value } }),
    sync: () => { value = sessionState.info?.config.system_prompt ?? ""; errorMsg = null; },
  }));

  async function save(): Promise<void> {
    if (busy) return;
    busy = true;
    errorMsg = null;
    try {
      await patchSessionDefaults({ system_prompt: value });
      closeDrawer();
    } catch (e) {
      if (e instanceof ApiError) {
        const detail =
          e.body && typeof e.body === "object" && "detail" in (e.body as object)
            ? String((e.body as { detail: unknown }).detail)
            : e.message;
        errorMsg = userFacingError(e, `Unable to save the system prompt. ${detail}`);
      } else {
        errorMsg = userFacingError(e, "Unable to save the system prompt. Try again.");
      }
    } finally {
      busy = false;
    }
  }
</script>

<section class="mf-tool dialog" aria-label="System prompt drawer">
  <header class="mf-head">
    <h2 class="dw-title" data-tip="The default for new generations in this session. A request's own OpenAI or Ollama system message takes precedence; empty clears it.">System prompt</h2>
    <IconButton icon="close" label="Close" onclick={closeDrawer} />
  </header>

  <label class="field">
    <span class="sr-only">System prompt</span>
    <textarea
      rows="12"
      bind:value={value}
      disabled={busy}
      placeholder="No system prompt"
      spellcheck="false"
    ></textarea>
    <span class="meta count"><MorphText text={`${value.length} char${value.length === 1 ? "" : "s"}`} /></span>
  </label>

  {#if errorMsg}
    <p class="mf-error" role="alert">{errorMsg}</p>
  {/if}

  <div class="mf-actions end">
    <button type="button" class="dw-text" onclick={closeDrawer} disabled={busy}>Cancel</button>
    <button
      type="button"
      class="dw-button"
      aria-busy={busy}
      onclick={save}
      disabled={busy}
    ><MorphText text={busy ? "Saving…" : "Save system prompt"} numbers={false} /></button>
  </div>
</section>

<style>
  .field { display: grid; gap: var(--s2); min-width: 0; }
  textarea { width: 100%; min-height: 240px; padding: var(--s3) var(--s4); border-radius: var(--r-box); color: var(--ink); font: var(--type-body); resize: vertical; }
  .count { justify-self: end; }
</style>
