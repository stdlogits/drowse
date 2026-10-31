<script lang="ts">
  import { MorphText } from "@stdlogits/ledger";
  import { Icon } from "@stdlogits/ledger";
  // Pending-queue strip — ghosted bubbles between the streaming
  // assistant turn and the composer.  Each bubble shows the item's
  // label tag + truncated text + an X button that cancels just that
  // item via the per-id ``cancelPendingAction`` store helper.
  //
  // Drains automatically on every WS ``done``; this component is display +
  // cancel only, with no submit logic.

  import {
    pendingActions,
    cancelPendingAction,
    inputHistory,
  } from "../lib/stores.svelte";

  const items = $derived(pendingActions.queue);
  /** Slot index currently pulled into the input via ``↑``.  The
   *  matching bubble renders with the ``editing`` class so the user
   *  can see which queued item their input is editing. */
  const editingSlot = $derived(inputHistory.pulledSlot);

  function truncate(text: string | null, max: number): string {
    if (text === null) return "";
    const flat = text.replace(/\n/g, " ⏎ ");
    return flat.length <= max ? flat : flat.slice(0, max - 1) + "…";
  }
</script>

{#if items.length > 0}
  <div class="pending-strip" role="list" aria-label="Pending queue">
    {#each items as item, idx (item.id)}
      <div class="bubble" class:editing={idx === editingSlot} role="listitem">
        {#if idx === editingSlot}
          <span class="meta" aria-label="editing">editing</span>
        {/if}
        <span class="meta tag"><MorphText text={item.label} numbers={false} /></span>
        {#if item.text !== null}
          <span class="text" {...{ "aria-description": (item.text) }}><MorphText text={truncate(item.text, 80)} numbers={false} /></span>
        {/if}
        <button
          type="button"
          class="cancel"
          onclick={() => cancelPendingAction(item.id)}
          aria-label="Cancel pending {item.label}"
          {...{ "aria-description": "cancel" }}
        ><Icon name="close" /></button>
      </div>
    {/each}
  </div>
{/if}

<style>
  /* Queued turns, waiting above the composer: a quiet wash row each, the one pulled into the field
   * for editing a step deeper. */
  .pending-strip { display: grid; gap: 2px; }
  .bubble { display: flex; align-items: center; gap: var(--s2); min-width: 0; min-height: 36px; padding: 0 var(--s1) 0 var(--s4); border-radius: var(--r-pill); background: var(--wash); color: var(--ink2); font: var(--type-small); transition: background-color var(--fast) var(--ease); }
  .bubble.editing { background: var(--wash2); color: var(--ink); }
  .tag { flex-shrink: 0; }
  .text { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cancel { display: grid; place-items: center; flex-shrink: 0; width: 28px; height: 28px; margin-left: auto; padding: 0; border: 0; border-radius: var(--r-pill); background: transparent; color: var(--ink3); cursor: pointer; transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease); }
  .cancel:hover { background: var(--paper); color: var(--ink); }
  @media (pointer: coarse) { .cancel { width: 44px; height: 44px; } .bubble { min-height: 44px; } }
</style>
