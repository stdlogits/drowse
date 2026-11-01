<script lang="ts">
  // One token, layer by layer, through both lenses (the token anatomy's ``reads_as``, the residual
  // each block wrote): what the S lens reads there (input: the words the residual takes in, a
  // footprint z) and what the R lens says (output: the words it pushes the model to say, a
  // probability). Each half fades by its weight in the depth rule, the crossover band (both
  // weighted) sits on a wash, and the cursor's own token is marked wherever it shows.
  import { Token } from "@stdlogits/ledger";
  import { depthTint } from "../../../lib/charts/depth";
  import { halfValue } from "../../../drawers/token/anatomy";
  import type { AnatomyLayer, FamilyReadDict } from "../../../lib/types";

  let { layers, tokenId, shown = 3 }: { layers: readonly AnatomyLayer[]; tokenId: number; shown?: number } = $props();

  const n = $derived(layers.length);
  const tint = (layer: number) => depthTint(n > 1 ? layer / (n - 1) : 0);
  const both = (layer: AnatomyLayer) => layer.reads_as.s.weight > 0 && layer.reads_as.r.weight > 0;
  /** A half's opacity: its weight, never fainter than a read can be made out. */
  const fade = (half: FamilyReadDict) => (half.weight > 0 ? 0.45 + 0.55 * half.weight : 0.3);
</script>

<div class="layers" role="table" aria-label="Each layer through both lenses">
  <div class="row head" role="row">
    <span role="columnheader" class="lay"></span>
    <span role="columnheader" class="meta">Reads <span class="fam">S lens</span></span>
    <span role="columnheader" class="meta">Says <span class="fam">R lens</span></span>
  </div>
  {#each layers as layer (layer.layer)}
    <div class="row" class:band={both(layer)} role="row" style:--tint={tint(layer.layer)}>
      <span role="rowheader" class="lay">
        <b>L{layer.layer}</b>
        <span class="w" aria-hidden="true"><i style:width="{(layer.reads_as.s.weight * 100).toFixed(0)}%"></i><i class="r" style:width="{(layer.reads_as.r.weight * 100).toFixed(0)}%"></i></span>
      </span>
      {#each [layer.reads_as.s, layer.reads_as.r] as half (half.family)}
        <span role="cell" class="cell" style:opacity={fade(half)}>
          {#if half.absent !== null}
            <span class="absent meta" data-tip={half.absent}>–</span>
          {:else}
            {#each half.tokens.slice(0, shown) as t, i (`${i}:${t.token_id}`)}
              <span class="tok" data-tip={halfValue(half, t)}><Token token={t.token} flags={t.token_id === tokenId ? 1 : 0} {i} /></span>
            {/each}
          {/if}
        </span>
      {/each}
    </div>
  {/each}
</div>

<style>
  .layers { display: grid; gap: 2px; min-width: 0; }
  .row { display: grid; grid-template-columns: 72px minmax(0, 1fr) minmax(0, 1fr); gap: var(--s3); align-items: center; min-height: 36px; padding: 0 var(--s3); border-radius: var(--r-box); }
  .row.band { background: var(--wash); }
  .row.head { min-height: 28px; }
  .fam { color: var(--ink3); }
  .lay { display: flex; align-items: center; gap: var(--s2); min-width: 0; }
  .lay b { min-width: 28px; color: var(--ink); font: 600 13px/20px var(--display); font-variant-numeric: tabular-nums; }
  .w { display: flex; flex-direction: column; gap: 2px; width: 28px; }
  .w > i { display: block; height: 3px; border-radius: 2px; background: var(--tint); opacity: 0.5; }
  .w > i.r { opacity: 1; }
  .cell { display: flex; flex-wrap: nowrap; align-items: center; gap: var(--s1); min-width: 0; overflow: hidden; transition: opacity var(--fast) var(--ease); }
  .tok { flex: none; max-width: 100%; }
  .absent { color: var(--ink3); }

  @media (max-width: 520px) {
    .row { grid-template-columns: 52px minmax(0, 1fr) minmax(0, 1fr); gap: var(--s2); padding: 0 var(--s2); }
    .w { display: none; }
  }
</style>
