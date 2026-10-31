<script lang="ts">
  // The token's alternatives: what the sampler had at this position, as rows (the token, a meter on
  // the last layer's tint, the probability), the chosen one marked. Any other row branches: the
  // node's decode replays with that token swapped in and the rest resampled as a sibling.
  import { Meter, tokenParts } from "@stdlogits/ledger";
  import { instrumentFamily, samplingState } from "../../lib/stores.svelte";
  import type { TokenScore } from "../../lib/types";
  import { getRuntimeClient } from "../../lib/runtime/registry";
  import { tokenAlternativeDefault, tokenAlternativeLimit } from "../../lib/runtime/samplingCapabilities";
  import { formatP } from "./layers";

  let { token, canBranch, busy, onbranch }: {
    token: TokenScore;
    /** The token has a replay boundary and nothing is generating. */
    canBranch: boolean;
    /** A branch from this view is starting. */
    busy: boolean;
    onbranch: (altId: number) => void;
  } = $props();

  const runtimeMode = getRuntimeClient().mode;
  const retainedAlternativeCount = tokenAlternativeDefault(runtimeMode);
  const alternativesAvailable = tokenAlternativeLimit(runtimeMode) > 0;
  const forkAvailable = $derived(
    runtimeMode === "http" ||
      (runtimeMode === "browser" && instrumentFamily("geometry")?.capabilities.token_readout === true),
  );

  /** The server sends ``top_alts`` in descending order, so the index is the rank. */
  const rows = $derived((token.topAlts ?? []).map((alt, i) => {
    const p = Math.exp(alt.logprob);
    return {
      id: alt.id,
      parts: tokenParts(alt.text),
      p,
      chosen: token.tokenId != null && alt.id === token.tokenId,
      tip: `rank ${i + 1} · id ${alt.id} · log p ${alt.logprob.toFixed(3)}${i > 0 ? ` · ${(alt.logprob - (token.topAlts?.[0]?.logprob ?? 0)).toFixed(3)} from the top` : ""}`,
    };
  }));
  const top = $derived(Math.max(1e-9, ...rows.map(row => row.p)));
  /** One row left with all the mass: certain after the sampler, which says nothing about truth. */
  const nearCertain = $derived(rows.length === 1 && rows[0].chosen && rows[0].p >= 0.9995);

  function recordAlternatives(): void {
    if (alternativesAvailable && (samplingState.return_top_k ?? 0) === 0) {
      samplingState.return_top_k = retainedAlternativeCount;
    }
  }
</script>

{#if rows.length > 0}
  <div class="alts" role="list" aria-label="Alternatives at this position">
    {#each rows as row, i (row.id)}
      <div class="alt rise" class:chosen={row.chosen} role="listitem" style:--i={i}>
        <span class="tok" data-tip={nearCertain ? `${row.tip} · after the sampler, not a measure of factual accuracy` : row.tip}><span class="ws">{row.parts.lead ? "·" : ""}</span>{row.parts.empty ? "∅" : row.parts.text}</span>
        <Meter value={row.p / top} label={`p ${formatP(row.p)}`} width={96} />
        <span class="p">{formatP(row.p)}</span>
        {#if !row.chosen && forkAvailable}
          <button type="button" class="dw-text branch" disabled={!canBranch || busy}
            aria-label={`Branch with ${row.parts.text || "this token"}`} onclick={() => onbranch(row.id)}>Branch</button>
        {:else}
          <span></span>
        {/if}
      </div>
    {/each}
  </div>
{:else if alternativesAvailable}
  <div class="dw-row">
    <button type="button" class="dw-quiet" disabled={samplingState.return_top_k > 0} onclick={recordAlternatives}>
      {samplingState.return_top_k > 0 ? "Recorded from the next run" : "Record alternatives"}
    </button>
  </div>
{/if}

<style>
  .alts { display: grid; gap: 2px; }
  .alt {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 96px 48px 56px;
    gap: var(--s3);
    align-items: center;
    min-height: 36px;
    padding: 2px var(--s3);
    border-radius: var(--r-box);
    transition: background-color var(--fast) var(--ease);
    /* The sampler reads the last layer. */
    --tint: var(--late);
  }
  @media (hover: hover) { .alt:hover { background: var(--wash); } }
  .tok {
    justify-self: start;
    max-width: 100%;
    overflow: hidden;
    padding: 2px var(--s2);
    border-radius: var(--r-chip);
    background: var(--wash);
    color: var(--ink);
    font: var(--type-small);
    white-space: pre;
    text-overflow: ellipsis;
    cursor: help;
  }
  .chosen .tok { background: color-mix(in oklch, var(--tint) 22%, var(--paper)); }
  .ws { color: var(--ink3); }
  .ws:empty { display: none; }
  .p { color: var(--ink2); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; }
  /* The meter takes its column's width (the piece sets a fixed one inline). */
  .alt > :global(.meter) { width: 100% !important; }
  .branch { justify-self: end; font: var(--type-meta); }
  /* With a pointer, a row's branch shows on the row it is about. */
  @media (hover: hover) {
    .branch { opacity: 0; transition: opacity var(--fast) var(--ease), color var(--fast) var(--ease); }
    .alt:hover .branch, .branch:focus-visible { opacity: 1; }
  }
  @media (max-width: 720px) {
    .alt { min-height: 44px; grid-template-columns: minmax(0, 1fr) 48px 44px 56px; }
    .branch { min-height: 44px; }
  }
</style>
