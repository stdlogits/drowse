<script lang="ts">
  // One reading as a row (the Instruments canvas): its name, a mark (a trace, a meter or per-layer
  // bars) and the value, with an optional trailing action. A row that can be chosen is a button,
  // pressed while it is the reading the reply is tinted by. ``tint`` sets the row's depth colour,
  // which its mark fills with. With a ``sub`` (a feature's label) the row takes the unit layout:
  // ID, label, a short meter, the value.
  import type { Snippet } from "svelte";

  let {
    pressed = false,
    onselect,
    tint = null,
    title,
    lead,
    sub,
    mark,
    value,
    action,
  }: {
    pressed?: boolean;
    onselect?: () => void;
    tint?: string | null;
    title?: string;
    lead: Snippet;
    sub?: Snippet;
    mark: Snippet;
    value: Snippet;
    action?: Snippet;
  } = $props();
</script>

{#snippet cells()}
  <span class="lead">{@render lead()}</span>
  {#if sub}<span class="sub">{@render sub()}</span>{/if}
  <span class="mark">{@render mark()}</span>
  <span class="value">{@render value()}</span>
{/snippet}

<div class="row" class:pressed class:unit={!!sub} class:chooses={!!onselect} style:--tint={tint ?? undefined} data-tip={title}>
  {#if onselect}
    <button type="button" class="main" aria-pressed={pressed} onclick={onselect}>{@render cells()}</button>
  {:else}
    <div class="main">{@render cells()}</div>
  {/if}
  {#if action}<span class="action">{@render action()}</span>{/if}
</div>

<style>
  .row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    min-width: 0;
    border-radius: var(--r-box);
    transition: background-color var(--fast) var(--ease);
  }
  .row.chooses:hover { background: var(--wash); }
  .row.pressed, .row.pressed:hover { background: var(--wash2); }
  .main {
    display: grid;
    grid-template-columns: minmax(0, 1fr) var(--mark-width, 140px) 44px;
    gap: var(--s4);
    align-items: center;
    min-width: 0;
    min-height: 44px;
    padding: var(--s2) var(--s3);
    border: 0;
    border-radius: inherit;
    background: transparent;
    color: inherit;
    font: inherit;
    text-align: left;
    transition: scale var(--fast) var(--ease);
  }
  button.main { cursor: pointer; }
  button.main:active { scale: 0.99; }
  .unit .main { grid-template-columns: auto minmax(0, 1fr) 56px 44px; gap: var(--s3); }
  .action { display: flex; align-items: center; padding-right: var(--s1); }

  .lead { display: flex; align-items: baseline; gap: var(--s2); min-width: 0; overflow: hidden; white-space: nowrap; }
  .lead :global(.n) { min-width: 0; overflow: hidden; text-overflow: ellipsis; color: var(--ink); font: var(--type-control); }
  .lead :global(.n.is-id) { font: 600 14px/20px var(--display); letter-spacing: -0.01em; }
  .lead :global(.ws) { color: var(--ink3); }
  .lead :global(.meta) { flex: none; }
  .sub { min-width: 0; overflow: hidden; color: var(--ink2); font: var(--type-small); white-space: nowrap; text-overflow: ellipsis; }
  .mark { display: grid; align-items: center; min-width: 0; }
  .mark :global(.bar) { width: 100%; }
  .value { color: var(--ink2); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }

  @media (max-width: 720px) {
    .main { --mark-width: 88px; gap: var(--s3); }
  }
</style>
