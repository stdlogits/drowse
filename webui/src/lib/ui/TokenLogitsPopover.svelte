<script lang="ts">
  import { onMount, untrack } from "svelte";
  import { IconButton, fadeIn, fadeOut, rise, stagger, tokenParts } from "@stdlogits/ledger";
  import type { TokenScore } from "../types";
  import { tokenProbabilityRows } from "../tokenProbabilities";
  import { samplingState } from "../stores/sampling.svelte";
  import { getRuntimeClient } from "../runtime/registry";
  import { tokenAlternativeDefault } from "../runtime/samplingCapabilities";
  import { currentViewport, placePopover, popoverBounds } from "../popoverPlacement";

  let { token, anchor, isCurrent, source = "Model token", contextChanged = false, onclose, ondetails }: {
    token: TokenScore;
    anchor: HTMLElement;
    isCurrent: (anchor: HTMLElement) => boolean;
    source?: string;
    contextChanged?: boolean;
    onclose: () => void;
    ondetails?: () => void;
  } = $props();
  const uid = $props.id();
  const trigger = untrack(() => anchor);
  const rows = $derived(tokenProbabilityRows(token));
  const title = $derived(tokenParts(token.text));
  const alternativeCount = tokenAlternativeDefault(getRuntimeClient().mode);
  /** What the numbers are, for the title's tip and the dialog's description (not a visible line). */
  const about = $derived(`${source} · sampled probabilities, not raw logits`);
  const beforeEdit = "Recorded before your edit, with the original text before it";
  let panel: HTMLDivElement;
  let closing = false;

  function close(restoreFocus = true) {
    if (closing || !isCurrent(trigger)) return;
    closing = true;
    panel.inert = true;
    if (restoreFocus && trigger.isConnected) trigger.focus({ preventScroll: true });
    onclose();
  }

  function place() {
    if (closing) return;
    const view = currentViewport();
    const bounds = popoverBounds(view);
    panel.style.maxWidth = `${bounds.maxWidth}px`;
    panel.style.maxHeight = `${bounds.maxHeight}px`;
    const at = placePopover(trigger.getBoundingClientRect(), panel.getBoundingClientRect(), view);
    panel.style.left = `${at.left}px`;
    panel.style.top = `${at.top}px`;
    panel.style.maxHeight = `${at.maxHeight}px`;
  }

  onMount(() => {
    panel.showPopover();
    place();
    panel.focus({ preventScroll: true });
    const escape = (event: KeyboardEvent) => {
      if (closing || !isCurrent(trigger)) return;
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      close();
    };
    const outside = (event: PointerEvent) => {
      if (!closing && !panel.contains(event.target as Node) && !trigger.contains(event.target as Node)) close(false);
    };
    const focusOutside = (event: FocusEvent) => {
      if (!closing && !panel.contains(event.target as Node) && !trigger.contains(event.target as Node)) close(false);
    };
    const scroll = (event: Event) => {
      if (event.target instanceof Node && panel.contains(event.target)) return;
      place();
    };
    document.addEventListener("keydown", escape, true);
    document.addEventListener("pointerdown", outside, true);
    document.addEventListener("focusin", focusOutside);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", scroll, true);
    window.visualViewport?.addEventListener("resize", place);
    window.visualViewport?.addEventListener("scroll", place);
    return () => {
      document.removeEventListener("keydown", escape, true);
      document.removeEventListener("pointerdown", outside, true);
      document.removeEventListener("focusin", focusOutside);
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", scroll, true);
      window.visualViewport?.removeEventListener("resize", place);
      window.visualViewport?.removeEventListener("scroll", place);
    };
  });
</script>

<div bind:this={panel} class="token-logits-popover" popover="manual" role="dialog" tabindex="-1" aria-labelledby={`${uid}-title`} aria-describedby={`${uid}-about`}
  in:fadeIn|global
  out:fadeOut|global
  onintrostart={() => {
    if (!isCurrent(trigger)) return;
    closing = false;
    panel.inert = false;
    place();
    if (!panel.contains(document.activeElement)) panel.focus({ preventScroll: true });
  }}
  onoutrostart={() => { closing = true; panel.inert = true; }}
>
  <header>
    <!-- A token recorded before an edit keeps the page's dashed mark for it. -->
    <h2 id={`${uid}-title`} class:context-changed={contextChanged} data-tip={about} data-tip-sub={contextChanged ? beforeEdit : undefined}><span class="ws">{title.lead ? "·" : ""}</span>{title.empty ? "∅" : title.text}</h2>
    <IconButton icon="close" label="Close token probabilities" onclick={() => close()} />
  </header>
  <p class="sr-only" id={`${uid}-about`}>{about}{contextChanged ? `. ${beforeEdit}` : ""}</p>
  {#if rows.length}
    <table aria-label="Recorded token probabilities">
      <thead class="sr-only"><tr><th scope="col">Token</th><th scope="col">Probability</th></tr></thead>
      <tbody>{#each rows as row, index (index)}
        {@const parts = tokenParts(row.text)}
        <tr class:chosen={row.chosen} in:rise|global={{ delay: stagger(index) }}>
          <td><span class="tok"><span class="ws">{parts.lead ? "·" : ""}</span>{parts.empty ? "∅" : parts.text}</span>{#if row.chosen}<span class="sr-only"> (this token)</span>{/if}</td>
          <td {...{ "aria-description": (`log probability ${row.logprob.toFixed(4)}`) }}>{Math.exp(row.logprob) >= 0.001 ? Math.exp(row.logprob).toFixed(3) : Math.exp(row.logprob).toExponential(2)}</td>
        </tr>
      {/each}</tbody>
    </table>
  {/if}
  {#if !token.topAlts?.length && alternativeCount > 0 && samplingState.return_top_k === 0}
    <button type="button" class="dw-quiet" onclick={() => samplingState.return_top_k = alternativeCount}>Record alternatives</button>
  {:else if !token.topAlts?.length}
    <p class="meta">No alternatives were recorded</p>
  {/if}
  {#if ondetails}<button type="button" class="dw-text" onclick={() => { trigger.focus({ preventScroll: true }); ondetails?.(); onclose(); }}>Full details</button>{/if}
</div>

<style>
  .token-logits-popover {
    position: fixed; inset: auto; margin: 0; width: min(20rem, calc(100vw - 16px));
    display: grid; gap: var(--s3); justify-items: start;
    box-sizing: border-box; padding: var(--s4); overflow: auto; overscroll-behavior: contain;
    border: 0; border-radius: var(--r-box);
    background: var(--wash2); color: var(--ink);
    font: var(--type-small);
  }
  .token-logits-popover:focus { outline: none; }
  header { display: flex; justify-content: space-between; align-items: center; gap: var(--s3); width: 100%; min-width: 0; }
  h2 { min-width: 0; margin: 0; color: var(--ink); font: var(--type-lead); font-family: var(--display); font-weight: 600; white-space: pre; overflow: hidden; text-overflow: ellipsis; cursor: help; }
  h2.context-changed { text-decoration: underline dashed var(--ink3); text-underline-offset: 4px; }
  header :global(.icon) { background: var(--paper); }
  .ws { color: var(--ink3); }
  .ws:empty { display: none; }
  .meta { overflow-wrap: anywhere; }
  table { width: 100%; border-collapse: separate; border-spacing: 0 2px; table-layout: fixed; }
  td { padding: 2px var(--s2); color: var(--ink2); }
  td:first-child { border-radius: var(--r-col) 0 0 var(--r-col); }
  td:last-child { width: 6rem; border-radius: 0 var(--r-col) var(--r-col) 0; text-align: end; font: var(--type-meta); font-variant-numeric: tabular-nums; }
  .tok { color: var(--ink); white-space: pre-wrap; overflow-wrap: anywhere; }
  /* The token that was chosen: the output's tint (the last layer's), as the inspector marks it. */
  .chosen td { background: color-mix(in oklch, var(--late) 22%, var(--wash2)); color: var(--ink); }
  .token-logits-popover .dw-quiet { background: var(--paper); }
</style>
