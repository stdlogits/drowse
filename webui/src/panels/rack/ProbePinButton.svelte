<script lang="ts">
  // Shared pin/unpin control for J-lens and SAE discovery/persistent cards.
  // The shape is the pillar identity; the hit target, baseline, hover, and
  // disabled behavior are deliberately identical.

  import RackMarker, { type RackMarkerShape } from "./RackMarker.svelte";

  let {
    shape,
    pinned,
    disabled = false,
    onclick,
    ariaLabel,
    title,
  }: {
    shape: RackMarkerShape;
    pinned: boolean;
    disabled?: boolean;
    onclick: () => void;
    ariaLabel: string;
    title: string;
  } = $props();
</script>

<button
  type="button"
  class="pin"
  class:pinned
  {disabled}
  {onclick}
  {...{ "aria-description": (title) }}
  aria-label={ariaLabel}
  aria-pressed={pinned}
>
  <span class="pin-icon" class:visible={!pinned} aria-hidden="true"><RackMarker {shape} /></span>
  <span class="pin-icon" class:visible={pinned} aria-hidden="true"><RackMarker {shape} filled /></span>
</button>

<style>
  .pin {
    display: inline-grid;
    place-items: center;
    inline-size: 36px;
    block-size: 36px;
    padding: 0;
    color: var(--ink3);
    background: transparent;
    border: 0;
    border-radius: var(--r-pill);
    flex: 0 0 36px;
    cursor: pointer;
    transition:
      color var(--fast) var(--ease),
      background-color var(--fast) var(--ease),
      scale var(--fast) var(--ease);
  }
  .pin:active:not(:disabled) {
    scale: 0.97;
  }
  .pin-icon {
    grid-area: 1 / 1;
    display: inline-grid;
    pointer-events: none;
    opacity: 0;
    scale: 0.25;
    transition:
      opacity var(--glide) var(--ease),
      scale var(--glide) var(--ease);
  }
  .pin-icon.visible {
    opacity: 1;
    scale: 1;
  }
  .pin.pinned {
    color: var(--ink);
  }
  .pin:hover:not(:disabled) {
    color: var(--ink);
    background: var(--wash2);
  }
  .pin:disabled {
    cursor: default;
    opacity: 0.4;
  }
  @media (pointer: coarse) {
    .pin { inline-size: 44px; block-size: 44px; flex-basis: 44px; }
  }
</style>
