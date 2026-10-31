<script lang="ts">
  // The one button, in Ledger: a pill with no line, told apart by fill alone.
  //   solid  — ink fill, paper text: the primary action of a surface
  //   ghost  — a wash that steps to wash2 on hover: the default workhorse
  //   flat   — text only, ink3 to ink on hover
  //   danger — a ghost for destructive actions (Ledger has no red: the confirm step carries it)
  //
  // ``accent`` tints the fill (never the text): solid fills with it, ghost and danger take it at
  // 14% in oklch. Busy is announced (aria-busy, the progress cursor); callers own the label and
  // disabling. Path and props are shared with drowse.ai.

  import type { Snippet } from "svelte";

  interface Props {
    children: Snippet;
    variant?: "solid" | "ghost" | "flat" | "danger";
    size?: "sm" | "md";
    /** A fill tint: any CSS colour (a depth on the ramp, usually). */
    accent?: string;
    disabled?: boolean;
    busy?: boolean;
    static?: boolean;
    title?: string;
    ariaLabel?: string;
    type?: "button" | "submit";
    onclick?: (ev: MouseEvent) => void;
    onmousedown?: (ev: MouseEvent) => void;
  }

  let {
    children,
    variant = "ghost",
    size = "md",
    accent,
    disabled = false,
    busy = false,
    static: isStatic = false,
    title,
    ariaLabel,
    type = "button",
    onclick,
    onmousedown,
  }: Props = $props();
</script>

<button
  class="sk-btn {variant} {size}"
  class:accented={accent !== undefined}
  class:static={isStatic}
  style:--btn-accent={accent}
  {disabled}
  {...{ "aria-description": (title) }}
  aria-label={ariaLabel}
  aria-busy={busy || undefined}
  {type}
  {onclick}
  {onmousedown}
>
  {@render children()}
</button>

<style>
  .sk-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--s2);
    max-width: 100%;
    min-height: 36px;
    padding: var(--s2) var(--s4);
    border: 0;
    border-radius: var(--r-pill);
    background: transparent;
    color: var(--ink);
    font: var(--type-control);
    text-align: center;
    text-wrap: balance;
    overflow-wrap: anywhere;
    cursor: pointer;
    transition:
      background-color var(--fast) var(--ease),
      color var(--fast) var(--ease),
      opacity var(--fast) var(--ease),
      scale var(--fast) var(--ease);
  }
  .sk-btn:active:not(:disabled):not(.static) { scale: 0.97; }
  .sk-btn:disabled { opacity: 0.4; cursor: default; }

  .md { font: var(--type-control); }
  .sm { min-height: 32px; padding: var(--s1) var(--s3); font: var(--type-control); }

  .solid { padding-inline: var(--s5); background: var(--ink); color: var(--paper); }
  .solid.sm { padding-inline: var(--s4); }
  .solid.accented { background: var(--btn-accent); }
  .solid:hover:not(:disabled) { opacity: 0.86; }

  .ghost, .danger { background: var(--wash); }
  .ghost:hover:not(:disabled), .danger:hover:not(:disabled) { background: var(--wash2); }
  .ghost.accented, .danger.accented { background: color-mix(in oklch, var(--btn-accent) 14%, var(--wash)); }
  .ghost.accented:hover:not(:disabled), .danger.accented:hover:not(:disabled) { background: color-mix(in oklch, var(--btn-accent) 22%, var(--wash)); }

  .flat, .flat.sm { padding-inline: 0; background: transparent; color: var(--ink3); }
  .flat:hover:not(:disabled) { color: var(--ink); }

  @media (max-width: 720px) {
    .sk-btn, .sm { min-height: 44px; }
  }
</style>
