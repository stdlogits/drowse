<script lang="ts">
  // Advanced-options collapsible — the builder-drawer wrapper around
  // the Ledger ``Expander`` with a fixed "Advanced options" summary and a body
  // layout that matches the extract / manifold forms.  Used in both
  // drawers so the secondary-fields surface reads identically.

  import type { Snippet } from "svelte";
  import { Expander } from "@stdlogits/ledger";

  interface Props {
    expanded?: boolean;
    /** Override the default "Advanced options" summary text. */
    summary?: string;
    children: Snippet;
  }

  let {
    expanded = $bindable(false),
    summary = "Advanced options",
    children,
  }: Props = $props();
</script>

<Expander bind:open={expanded} label={summary}>
  <div class="sk-advanced-body">
    {@render children()}
  </div>
</Expander>

<style>
  .sk-advanced-body { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s4); }
</style>
