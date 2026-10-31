<script lang="ts">
  // Resets the current model's settings to their starting values. ``full`` (the Model room, and
  // drowse.ai's local runtime page) resets everything it holds and asks first; without it (the
  // sampling section) it is one quiet text button that resets generation settings at once.
  import { MorphText } from "@stdlogits/ledger";
  import Button from "./Button.svelte";
  import { genStatus } from "../stores/chat.svelte";
  import { modelDefaultsState } from "../stores/sampling.svelte";
  import { resetSettings, settingsResetState } from "../stores/settingsReset.svelte";
  import { pushToast } from "../stores/toasts.svelte";
  import { userFacingError } from "../runtime/userFacingError";

  let { full = false }: { full?: boolean } = $props();
  let confirming = $state(false);
  let resetDone = $state(false);
  $effect(() => { if (!resetDone) return; const timer = setTimeout(() => resetDone = false, 1800); return () => clearTimeout(timer); });
  let error = $state<string | null>(null);
  async function reset(): Promise<void> {
    error = null;
    try {
      await resetSettings(full);
      confirming = false;
      resetDone = true;
      pushToast(full ? "Model settings reset." : "Generation settings reset.", { kind: "info" });
    } catch (cause) {
      error = userFacingError(cause, "Settings could not be fully reset. Try again.");
    }
  }
  const label = $derived(settingsResetState.busy ? "Resetting…" : resetDone ? "Settings reset" : full ? "Reset settings" : "Reset to original");
</script>

<div class="reset-settings" class:full>
  {#if full}<h3 class="head">Reset settings</h3>{/if}
  {#if confirming}
    <p class="ask">Generation settings, the system prompt, roles, steering and readings go back to how they started. Chats and downloads stay.</p>
    <div class="actions">
      <Button variant="solid" disabled={settingsResetState.busy || genStatus.active} onclick={() => void reset()}>Reset</Button>
      <Button variant="flat" disabled={settingsResetState.busy} onclick={() => confirming = false}>Cancel</Button>
    </div>
  {:else}
    <div class="actions">
      <Button variant={full ? "ghost" : "flat"} size={full ? "md" : "sm"}
        disabled={!modelDefaultsState.info || settingsResetState.busy || genStatus.active}
        onclick={() => full ? confirming = true : void reset()}>
        <MorphText text={label} numbers={false} />
      </Button>
    </div>
  {/if}
  {#if genStatus.active}<p class="meta">After the reply finishes</p>{/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
</div>

<style>
  .reset-settings { display: grid; justify-items: start; gap: var(--s1); min-width: 0; }
  .reset-settings.full { gap: var(--s3); }
  .ask { max-width: 520px; color: var(--ink2); font: var(--type-small); }
  .actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2); }
  .error { color: var(--ink); font: var(--type-small); overflow-wrap: anywhere; }
</style>
