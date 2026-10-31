<script lang="ts">
  // The lens at this token: the chosen token's probability by depth (one bar per layer in its own
  // depth colour, dimmed before the layer it leads from), then layer by layer what each layer's
  // residual reads as, then the readout's aggregate across layers. The layer table is the frame
  // the token anatomy fills: "Reads as" comes from the lens; the attention and MLP writes join as
  // columns once the engine replays them (none are drawn until their data exists).
  import { Token, tokenParts } from "@stdlogits/ledger";
  import type { LensTokenReadoutJSON } from "../../lib/types";
  import type { ReplayReadout } from "./readout.svelte";
  import ReadingWait from "./ReadingWait.svelte";
  import EmptyState from "./EmptyState.svelte";
  import JLensMissingState from "@runtime-jlens-missing";
  import { chosenByDepth, depthTint, foldedLayers, formatP, lastLayer, leadLayer, type LensCell } from "./layers";

  let { readout, tokenText, jlensFitted, hasReplayContext, modelId, replayAvailable }: {
    readout: ReplayReadout<LensTokenReadoutJSON>;
    /** The inspected token's text, for the heads. */
    tokenText: string;
    jlensFitted: boolean;
    hasReplayContext: boolean;
    modelId: string | null;
    replayAvailable: boolean;
  } = $props();

  const data = $derived(readout.data);
  const layers = $derived(data?.layers ?? []);
  const tokenId = $derived(data && data.token_id >= 0 ? data.token_id : null);
  const last = $derived(lastLayer(layers));
  const width = $derived(Math.max(0, ...layers.map(row => row.tokens.length)));
  const lead = $derived(leadLayer(layers, tokenId));
  const byDepth = $derived(chosenByDepth(layers, tokenId));
  const peak = $derived(Math.max(1e-9, ...byDepth.map(row => row.p ?? 0)));
  const named = $derived(tokenParts(tokenText));
  /** No layer ranked the token within its top-k: there is nothing to draw over depth. */
  const unmeasured = $derived(byDepth.every(row => row.p === null));
  const depthNote = $derived([
    unmeasured ? `below every layer's top ${width}` : lead !== null ? `leads from L${lead}` : "",
    readout.origin === "replayed" ? "replayed" : "",
  ].filter(Boolean).join(" · "));

  let all = $state(false);
  const shown = $derived(all ? new Set(layers.map(row => row.layer)) : foldedLayers(layers.map(row => row.layer), lead));
  const folds = $derived(shown.size < layers.length || all);
  let open = $state<number | null>(null);
  $effect(() => { void data; open = null; });

  /** Reads as: the layer's first reading, then (while no other column claims the row) the next two. */
  const reads = 3;

  function cellTip(layer: number, cell: LensCell, rank: number): string {
    return `L${layer} · rank ${rank} · p ${formatP(Math.exp(cell.logprob))}`;
  }

  /** Aggregate tokens match layer readings by exact text, so a whitespace variant never counts. */
  function hits(token: string): number {
    return layers.filter(row => row.tokens.some(cell => cell.token === token)).length;
  }
</script>

{#snippet cell(layer: number, item: LensCell, rank: number, faint: boolean)}
  {@const parts = tokenParts(item.token)}
  <span class="chip" class:it={tokenId != null && item.id === tokenId} class:faint data-tip={cellTip(layer, item, rank)}><span class="ws">{parts.lead ? "·" : ""}</span>{parts.empty ? "∅" : parts.text}</span>
{/snippet}

{#if readout.loading}
  <section class="dw-sec" aria-label="Lens">
    <ReadingWait progress={readout.progress} label="Lens readout" blocks={[56, 120]} />
  </section>
{:else if readout.error}
  <section class="dw-sec" aria-label="Lens"><p class="meta" role="alert">{readout.error}</p></section>
{:else if data}
  <section class="dw-sec" aria-label="By depth">
    <div class="dw-sec-head">
      <span class="head"><span class="dw-ws">{named.lead ? "·" : ""}</span>{named.empty ? "∅" : named.text} by depth</span>
      <span class="meta">{depthNote}</span>
    </div>
    {#if !unmeasured}
    <div class="depth">
      <div class="bars" role="img" aria-label={`Probability of ${named.text || "the token"} by layer`}>
        {#each byDepth as row (row.layer)}
          <i class:dim={lead !== null && row.layer < lead} class:none={row.p === null}
            style:height="{row.p === null ? 4 : Math.max(4, (row.p / peak) * 100)}%"
            style:--tint={depthTint(row.layer, last)}
            data-tip={`L${row.layer} · ${row.p === null ? "below the top " + width : "p " + formatP(row.p)}`}></i>
        {/each}
      </div>
      <div class="ends meta"><span>L{layers[0]?.layer ?? 0}</span><span>L{last}</span></div>
    </div>
    {/if}
  </section>

  <section class="dw-sec" aria-label="Layer by layer">
    <div class="dw-sec-head">
      <span class="head">Layer by layer</span>
      {#if folds}<button type="button" class="dw-text" onclick={() => (all = !all)}>{all ? "Every third layer" : `All ${layers.length} layers`}</button>{/if}
    </div>
    <div class="layers">
      <div class="lrow is-head" aria-hidden="true"><span></span><span data-tip={readout.source ? `Through ${readout.source}` : undefined}>Reads as</span></div>
      {#each layers as row (row.layer)}
        {#if shown.has(row.layer)}
          <button type="button" class="lrow" class:lead={row.layer === lead} style:--tint={depthTint(row.layer, last)}
            aria-expanded={row.tokens.length > reads ? open === row.layer : undefined} onclick={() => (open = open === row.layer ? null : row.layer)}>
            <span class="lnum"><span class="dot"></span>L{row.layer}</span>
            <span class="reads">
              {#each row.tokens.slice(0, reads) as item, rank (item.id)}{@render cell(row.layer, item, rank + 1, rank > 0)}{/each}
            </span>
          </button>
          {#if open === row.layer && row.tokens.length > reads}
            <div class="more toks">
              {#each row.tokens as item, rank (item.id)}<Token token={item.token} value={Math.exp(item.logprob)} flags={tokenId != null && item.id === tokenId ? 1 : 0} i={rank} />{/each}
            </div>
          {/if}
        {/if}
      {/each}
    </div>
  </section>

  {#if (data.aggregate ?? []).length > 0}
    <section class="dw-sec" aria-label="Across layers">
      <div class="dw-sec-head"><span class="head">Across layers</span></div>
      <div class="toks" data-link-scope>
        {#each data.aggregate ?? [] as chip, i (i)}
          <span class="agg" data-tip={`strength ${chip.strength.toFixed(3)} · centred at ${chip.com.toFixed(2)} of depth ±${chip.spread.toFixed(2)} · in ${hits(chip.token)}/${layers.length} layers`}>
            <Token token={chip.token} value={chip.strength} flags={chip.token === data.token_text ? 1 : 0} {i} />
          </span>
        {/each}
      </div>
    </section>
  {/if}
{:else if !jlensFitted}
  <section class="dw-sec" aria-label="Lens"><JLensMissingState {modelId} /></section>
{:else if replayAvailable && !hasReplayContext}
  <section class="dw-sec" aria-label="Lens"><EmptyState title="No replay record for this token" /></section>
{/if}

<style>
  .depth { display: grid; gap: var(--s1); }
  .bars { display: flex; align-items: flex-end; gap: 2px; height: 56px; }
  .bars > i {
    flex: 1 1 0;
    min-height: 2px;
    border-radius: 2px;
    background: var(--tint);
    transition: height var(--glide) var(--ease), opacity var(--glide) var(--ease);
  }
  .bars > i.dim { opacity: 0.4; }
  .bars > i.none { background: var(--token-fill); opacity: 1; }
  .ends { display: flex; justify-content: space-between; }

  .layers { display: grid; gap: 2px; }
  .lrow {
    display: grid;
    grid-template-columns: 56px minmax(0, 1fr);
    gap: var(--s2);
    align-items: center;
    width: 100%;
    min-height: 32px;
    padding: 2px var(--s2);
    border: 0;
    border-radius: var(--r-col);
    background: transparent;
    color: inherit;
    font: inherit;
    text-align: left;
    cursor: pointer;
    transition: background-color var(--fast) var(--ease);
  }
  .lrow.is-head { min-height: 24px; color: var(--ink3); font: var(--type-head); cursor: default; }
  @media (hover: hover) { button.lrow:hover { background: var(--token-fill); } }
  .lrow.lead, .lrow[aria-expanded="true"] { background: var(--token-fill); }
  .lnum { display: flex; align-items: center; gap: 6px; color: var(--ink3); font: var(--type-meta); font-variant-numeric: tabular-nums; }
  .dot { flex: none; width: 8px; height: 8px; border-radius: 2px; background: var(--tint); }
  .reads { display: flex; gap: var(--s1); min-width: 0; overflow: hidden; }
  .chip {
    flex: none;
    max-width: 100%;
    overflow: hidden;
    padding: 2px 6px;
    border-radius: var(--r-chip);
    background: var(--token-fill);
    color: var(--ink2);
    font: var(--type-small);
    white-space: pre;
    text-overflow: ellipsis;
  }
  .chip:first-child { flex: 0 1 auto; color: var(--ink); }
  .chip.it { background: color-mix(in oklch, var(--tint) 22%, var(--paper)); color: var(--ink); }
  .chip.faint { opacity: 0.55; }
  .ws { color: var(--ink3); }
  .ws:empty { display: none; }
  .more { padding: var(--s1) var(--s2) var(--s2) calc(56px + var(--s4)); }
  .agg { display: contents; }
  @media (max-width: 720px) {
    .lrow { min-height: 40px; grid-template-columns: 48px minmax(0, 1fr); }
    .more { padding-left: calc(48px + var(--s4)); }
  }
</style>
