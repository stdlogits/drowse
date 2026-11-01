<script lang="ts">
  // A row's one-click way into the recipe (Push, Silence): quiet text at the row's end that adds its
  // term to the strip through lib/steerActions.ts, where its chip opens for the strength. The
  // instruments' rows (a lens word, an SAE feature, a unit in the token view) share it.
  let { label, aria, tip, disabled = false, onclick }: {
    label: string;
    /** The full action, for a screen reader (``Push L20 n512``). */
    aria: string;
    tip?: string;
    disabled?: boolean;
    onclick: () => void;
  } = $props();
</script>

<button type="button" class="steer" {disabled} aria-label={aria} data-tip={tip} onclick={(event) => { event.stopPropagation(); onclick(); }}>{label}</button>

<style>
  .steer {
    flex: none;
    height: 32px;
    padding: 0 var(--s2);
    border: 0;
    border-radius: var(--r-pill);
    background: transparent;
    color: var(--ink3);
    font: var(--type-meta);
    white-space: nowrap;
    cursor: pointer;
    transition: color var(--fast) var(--ease), background-color var(--fast) var(--ease);
  }
  .steer:hover:not(:disabled) { color: var(--ink); background: var(--wash2); }
  .steer:disabled { opacity: 0.4; cursor: default; }
  .steer:focus-visible { outline-offset: -2px; }
  @media (max-width: 720px), (pointer: coarse) {
    .steer { height: 44px; }
  }
</style>
