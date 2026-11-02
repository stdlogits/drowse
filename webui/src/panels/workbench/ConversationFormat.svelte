<script lang="ts">
  // The conversation's format, by the thread's header: the model's chat template (turns and roles)
  // or raw completion (one text, continued as written). It is how the Page draws the conversation
  // and how the next turn is sent, kept per model (lib/stores/chat.svelte.ts::genUiMode), not a
  // sampling field. A base model has no template, so there is no choice to show; the raw buffer's
  // head says so. The workbench places this once: in the thread column when it is a column, else in
  // the row the thread folds into.
  import { Segmented } from "@stdlogits/ledger";
  import { genUiMode, setGenUiMode } from "../../lib/stores/chat.svelte";
  import { sessionState } from "../../lib/stores/session.svelte";

  const FORMATS: { value: "chat" | "raw"; label: string; hint: string }[] = [
    { value: "chat", label: "Chat", hint: "The model's chat template: turns, roles and a system prompt" },
    { value: "raw", label: "Raw", hint: "Raw completion: the text as written, continued" },
  ];
</script>

{#if sessionState.info && sessionState.info.is_base_model !== true}
  <span class="format"><Segmented label="Format" value={genUiMode.mode} onchange={setGenUiMode} options={FORMATS} /></span>
{/if}

<style>
  .format { display: inline-flex; flex: none; }
  /* Phones and touch screens: 44 px targets (Segmented's thumb follows its option's box). */
  @media (max-width: 720px), (pointer: coarse) {
    .format :global(.seg > button) { height: 44px; }
  }
</style>
