<script lang="ts">
  // The token, layer by layer (the node replay's anatomy, one request): the chosen token's
  // probability by depth, then per layer what the residual reads as and what attention and the MLP
  // added (and steering, where it injected), each cell the depth-picked read: S early, R late, both
  // in the crossover band. Then the units that pushed the token, signed, and where the top heads
  // read from. A unit selects into the Inspector. (The Main and Phone canvases, token selected.)
  import { Token, tokenParts } from "@stdlogits/ledger";
  import type { AnatomyLayer, AnatomyUnit, TokenAnatomy } from "../../lib/types";
  import { select } from "../../lib/stores/selection.svelte";
  import type { ReplayReadout } from "./readout.svelte";
  import ReadingWait from "./ReadingWait.svelte";
  import { depthTint, foldedLayers, formatP } from "./layers";
  import {
    anatomyByDepth,
    anatomyLead,
    columnRead,
    formatContribution,
    halfValue,
    layerUnits,
    lensTag,
    patternRows,
    pickCell,
    pushRows,
    unitShort,
    weightTip,
    type AnatomyColumn,
  } from "./anatomy";
  import type { UnitLabels } from "./labels.svelte";

  let { readout, tokenText, labels = null }: {
    readout: ReplayReadout<TokenAnatomy>;
    /** The inspected token's text, for the heads. */
    tokenText: string;
    /** The catalog's labels, when the runtime reads them. */
    labels?: UnitLabels | null;
  } = $props();

  const data = $derived(readout.data);
  const layers = $derived<AnatomyLayer[]>(data?.layers ?? []);
  const tokenId = $derived(data?.token.token_id ?? -1);
  const last = $derived(Math.max(0, layers.length - 1));
  const named = $derived(tokenParts(tokenText));

  // ---- by depth ------------------------------------------------------------

  const byDepth = $derived(data ? anatomyByDepth(layers, tokenId, data.token.probability) : []);
  const lead = $derived(anatomyLead(layers, tokenId));
  const peak = $derived(Math.max(1e-9, ...byDepth.map(row => row.p ?? 0)));
  const width = $derived(Math.max(0, ...layers.map(row => row.reads_as.r.tokens.length)));

  // ---- layer by layer ------------------------------------------------------

  const NAMES: Record<AnatomyColumn, string> = { reads_as: "Reads as", attention: "Attention added", mlp: "MLP added", steering: "Steering added" };
  const steered = $derived(layers.some(row => row.steering !== null));
  /** The table's width: three columns from 350 px (steering a fourth from 480, else in the opened
   *  layer); narrower, one group at a time behind a switch (the Phone canvas). */
  let tableWidth = $state(480);
  type Group = "read" | "added" | "steering";
  let group = $state<Group>("read");
  const groups = $derived<Group[]>(steered ? ["read", "added", "steering"] : ["read", "added"]);
  const wide = $derived(tableWidth >= 350);
  const current = $derived<Group>(groups.includes(group) ? group : "read");
  const shownColumns = $derived.by<AnatomyColumn[]>(() => {
    if (wide) return steered && tableWidth >= 480 ? ["reads_as", "attention", "mlp", "steering"] : ["reads_as", "attention", "mlp"];
    return current === "read" ? ["reads_as"] : current === "added" ? ["attention", "mlp"] : ["steering"];
  });
  const nextGroup = $derived(groups[(groups.indexOf(current) + 1) % groups.length]);
  const groupLabel: Record<Group, string> = { read: "Reads as", added: "Attention and MLP", steering: "Steering" };
  const grid = $derived(`var(--lnum) repeat(${shownColumns.length}, minmax(0, 1fr))`);

  let all = $state(false);
  const shown = $derived(all ? new Set(layers.map(row => row.layer)) : foldedLayers(layers.map(row => row.layer), lead));
  const folds = $derived(shown.size < layers.length || all);
  let open = $state<number | null>(null);
  $effect(() => { void data; open = null; });

  // ---- units and heads -----------------------------------------------------

  const pushes = $derived(data ? pushRows(data.units, layers) : []);
  const heads = $derived(data ? patternRows(data.patterns, data.context) : []);
  const credit = $derived(labels?.credit ?? null);

  $effect(() => {
    if (!labels || !data) return;
    labels.want([
      ...pushes.map(row => ({ unit: row.unit, pole: row.pole })),
      ...heads.map(row => ({ unit: row.unit, pole: null })),
    ]);
  });
  $effect(() => {
    if (!labels || open === null) return;
    const row = layers.find(l => l.layer === open);
    if (row) labels.want(layerUnits(row, 4).map(u => ({ unit: u.unit, pole: poleOf(u) })));
  });

  function poleOf(u: AnatomyUnit): "+" | "-" | null {
    return u.kind === "neuron" && u.activation ? (u.activation > 0 ? "+" : "-") : null;
  }
  function unitTip(unit: string, contribution: number, kind: "neuron" | "head", pole: "+" | "-" | null, activation: number | null, label: string | undefined): string {
    const parts = [
      `${unit} · ${formatContribution(contribution)} to the logit`,
      kind === "neuron" ? (pole ? `${pole === "+" ? "+" : "−"} pole${activation != null ? ` · activation ${activation.toFixed(2)}` : ""}` : "")
        : activation != null ? `write norm ${activation.toFixed(2)}` : "",
      label ?? "",
    ];
    return parts.filter(Boolean).join(" · ");
  }
</script>

{#snippet cellView(row: AnatomyLayer, column: AnatomyColumn)}
  {@const read = columnRead(row, column)}
  <span class="cell">
    {#if read === null}
      <span class="chip none" aria-hidden="true"></span>
    {:else}
      {@const cell = pickCell(read)}
      {#if cell.absent !== null}
        <span class="chip none" data-tip={cell.absent}>–</span>
      {:else}
        {#each cell.tokens as item, i (i)}
          {@const parts = tokenParts(item.token)}
          <span class="chip" class:it={item.id === tokenId} class:agree={cell.agree && item.id !== tokenId} class:faint={item.faint} data-tip={item.tip}><span class="ws">{parts.lead ? "·" : ""}</span>{parts.empty ? "∅" : parts.text}</span>
        {/each}
      {/if}
    {/if}
  </span>
{/snippet}

{#snippet unitRow(unit: string, kind: "neuron" | "head", layer: number, contribution: number, share: number, pole: "+" | "-" | null, activation: number | null)}
  {@const label = labels?.label(unit, kind, pole) ?? null}
  <button type="button" class="unit" class:bare={label === null} style:--tint={depthTint(layer, last)}
    data-tip={unitTip(unit, contribution, kind, pole, activation, label?.tip)} onclick={() => select({ kind: "unit", id: unit })}>
    <span class="uid">{unitShort(unit)}</span>
    {#if label}<span class="ulabel">{label.text}</span>{/if}
    <span class="umeter" class:neg={contribution < 0}><i style:width="{Math.max(2, share * 100)}%"></i></span>
    <span class="uval">{formatContribution(contribution)}</span>
  </button>
{/snippet}

{#if readout.loading}
  <section class="dw-sec" aria-label="Anatomy">
    <ReadingWait progress={readout.progress} label="Token anatomy" blocks={[56, 240, 120]} />
  </section>
{:else if readout.error}
  <section class="dw-sec" aria-label="Anatomy"><p class="meta" role="alert">{readout.error}</p></section>
{:else if data}
  <section class="dw-sec" aria-label="By depth">
    <div class="dw-sec-head">
      <span class="head"><span class="dw-ws">{named.lead ? "·" : ""}</span>{named.empty ? "∅" : named.text} by depth</span>
      {#if lead !== null}<span class="meta">leads from L{lead}</span>{/if}
    </div>
    <div class="depth">
      <div class="bars" role="img" aria-label={`Probability of ${named.text || "the token"} by layer`}>
        {#each byDepth as row (row.layer)}
          <i class:dim={lead !== null && row.layer < lead} class:none={row.p === null}
            style:height="{row.p === null ? 4 : Math.max(4, (row.p / peak) * 100)}%"
            style:--tint={depthTint(row.layer, last)}
            data-tip={`L${row.layer} · ${row.p === null ? `below the top ${width}` : `p ${formatP(row.p)}`}${row.final ? " · the model's output" : ""}`}></i>
        {/each}
      </div>
      <div class="ends meta"><span>L{layers[0]?.layer ?? 0}</span><span>L{last}</span></div>
    </div>
  </section>

  <section class="dw-sec" aria-label="Layer by layer">
    <div class="dw-sec-head">
      <span class="head">Layer by layer</span>
      <span class="ctl">
        {#if !wide}<button type="button" class="dw-text" onclick={() => (group = nextGroup)}>{groupLabel[nextGroup]}</button>{/if}
        {#if folds}<button type="button" class="dw-text" onclick={() => (all = !all)}>{all ? "Every third layer" : `All ${layers.length} layers`}</button>{/if}
      </span>
    </div>
    <div class="layers" bind:clientWidth={tableWidth} style:--cols={grid}>
      <div class="lrow is-head" aria-hidden="true"><span></span>{#each shownColumns as column (column)}<span>{NAMES[column]}</span>{/each}</div>
      {#each layers as row (row.layer)}
        {#if shown.has(row.layer)}
          <button type="button" class="lrow" class:lead={row.layer === lead} style:--tint={depthTint(row.layer, last)}
            aria-expanded={open === row.layer} onclick={() => (open = open === row.layer ? null : row.layer)}>
            <span class="lnum"><span class="dot"></span>L{row.layer}<span class="tag" data-tip={weightTip(row.reads_as)}>{lensTag(row.reads_as)}</span></span>
            {#each shownColumns as column (column)}{@render cellView(row, column)}{/each}
          </button>
          {#if open === row.layer}
            {@const own = layerUnits(row, 4)}
            {@const peakHere = Math.max(1e-12, ...own.map(u => Math.abs(u.contribution)))}
            <div class="detail" style:--tint={depthTint(row.layer, last)}>
              {#each ["reads_as", "attention", "mlp", "steering"] as const as column (column)}
                {@const read = columnRead(row, column)}
                {#if read}
                  {@const halves = read.s.weight >= read.r.weight ? [read.s, read.r] : [read.r, read.s]}
                  <div class="drow">
                    <span class="dname">{NAMES[column]}</span>
                    <div class="dreads">
                      {#each halves as half (half.family)}
                        {#if half.weight > 0 && half.tokens.length > 0}
                          <div class="toks" class:faint={half !== halves[0]} data-link-scope data-tip={half.lens ?? undefined}>
                            {#each half.tokens as score, i (score.token_id)}
                              <span class="it-wrap" class:it={score.token_id === tokenId} data-tip={`${half.lens ?? half.family} · ${halfValue(half, score)}`}><Token token={score.token} value={score.value} {i} /></span>
                            {/each}
                          </div>
                        {:else if half.weight > 0 && half.absent}
                          <span class="meta">{half.absent}</span>
                        {/if}
                      {/each}
                    </div>
                  </div>
                {/if}
              {/each}
              {#if own.length > 0}
                <div class="units">
                  {#each own as u (u.unit)}{@render unitRow(u.unit, u.kind, u.layer, u.contribution, Math.abs(u.contribution) / peakHere, poleOf(u), u.activation)}{/each}
                </div>
              {/if}
            </div>
          {/if}
        {/if}
      {/each}
    </div>
  </section>

  {#if pushes.length > 0}
    <section class="dw-sec" aria-label="Units that pushed the token">
      <div class="dw-sec-head">
        <span class="head">Pushed toward <span class="dw-ws">{named.lead ? "·" : ""}</span>{named.empty ? "∅" : named.text}</span>
        {#if credit}<span class="meta">{credit}</span>{/if}
      </div>
      <div class="units">
        {#each pushes as row (row.unit)}{@render unitRow(row.unit, row.kind, row.layer, row.contribution, row.share, row.pole, null)}{/each}
      </div>
    </section>
  {:else if data.units_absent}
    <section class="dw-sec" aria-label="Units that pushed the token">
      <div class="dw-sec-head"><span class="head">Pushed toward <span class="dw-ws">{named.lead ? "·" : ""}</span>{named.empty ? "∅" : named.text}</span></div>
      <p class="meta">{data.units_absent}</p>
    </section>
  {/if}

  {#if heads.length > 0}
    <section class="dw-sec" aria-label="Where the top heads read from">
      <div class="dw-sec-head"><span class="head">Heads read from</span></div>
      <div class="heat">
        {#each heads as row (row.unit)}
          {@const label = labels?.label(row.unit, "head", null) ?? null}
          <div class="hrow" style:--tint={depthTint(row.layer, last)}>
            <button type="button" class="uid hid" data-tip={[`${row.unit} · ${formatContribution(row.contribution)} to the logit`, `${formatP(row.sink)} on the first token`, label?.text ?? ""].filter(Boolean).join(" · ")}
              onclick={() => select({ kind: "unit", id: row.unit })}>{row.short}</button>
            <p class="hline">{#each row.pieces as piece, i (i)}{#if piece.pos < 0}<span class="gap">{" … "}</span>{:else}<span class="ht" class:here={piece.here} style:--a="{piece.tint}%" data-tip={`${formatP(piece.weight)}`}>{piece.text.replace(/\n/g, "⏎")}</span>{/if}{/each}</p>
          </div>
        {/each}
      </div>
    </section>
  {/if}
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

  .ctl { display: flex; flex: none; gap: var(--s4); }

  /* The layer table: a label column (the layer, its dot on the ramp, the lens mark), then a cell per
   * column. The rows are buttons: a click opens the layer below its row. */
  .layers { --lnum: 64px; display: grid; gap: 2px; }
  .lrow {
    display: grid;
    grid-template-columns: var(--cols);
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
  .lrow.is-head > span { min-width: 0; line-height: 16px; overflow-wrap: anywhere; }
  @media (hover: hover) { button.lrow:hover { background: var(--token-fill); } }
  .lrow.lead, .lrow[aria-expanded="true"] { background: var(--token-fill); }
  .lnum { display: flex; align-items: center; gap: 6px; min-width: 0; color: var(--ink3); font: var(--type-meta); font-variant-numeric: tabular-nums; }
  .dot { flex: none; width: 8px; height: 8px; border-radius: 2px; background: var(--tint); }
  .tag { margin-left: auto; padding-right: var(--s1); color: var(--ink2); white-space: nowrap; cursor: help; }
  .cell { display: flex; gap: var(--s1); min-width: 0; overflow: hidden; }
  .chip {
    flex: 0 0 auto;
    max-width: 100%;
    min-width: 0;
    overflow: hidden;
    padding: 2px 6px;
    border-radius: var(--r-chip);
    background: var(--token-fill);
    color: var(--ink);
    font: var(--type-small);
    white-space: pre;
    text-overflow: ellipsis;
  }
  .lrow.lead .chip, .lrow[aria-expanded="true"] .chip { background: var(--paper); }
  .chip.it, .lrow .chip.it { background: color-mix(in oklch, var(--tint) 22%, var(--paper)); }
  .chip.agree, .lrow .chip.agree { background: color-mix(in oklch, var(--tint) 14%, var(--paper)); }
  /* The lighter read gives way first: it shrinks, the heavier keeps its width. */
  .chip.faint { flex: 0 1 auto; color: var(--ink2); opacity: 0.6; }
  .chip.none { min-width: 20px; color: var(--ink3); text-align: center; cursor: help; }
  .chip.none:empty { visibility: hidden; }
  .ws { color: var(--ink3); }
  .ws:empty { display: none; }

  /* A layer opened: each read's tokens, the lighter half faint, then the layer's own units. */
  .detail { display: grid; gap: var(--s3); padding: var(--s2) var(--s2) var(--s3) calc(var(--lnum) + var(--s4)); }
  .drow { display: grid; gap: var(--s1); min-width: 0; }
  .dname { color: var(--ink3); font: var(--type-head); }
  .dreads { display: grid; gap: var(--s1); min-width: 0; }
  .toks { display: flex; flex-wrap: wrap; gap: var(--s1); min-width: 0; }
  .toks.faint { opacity: 0.6; }
  .it-wrap { display: contents; }
  .it-wrap.it > :global(.tok) { background: color-mix(in oklch, var(--tint) 22%, var(--paper)); }
  .detail .units { margin: 0 calc(-1 * var(--s3)); }

  /* A unit that pushed: its ID, its label, a signed meter and the value. A click opens the unit. */
  .units { display: grid; gap: 2px; }
  .unit {
    display: grid;
    grid-template-columns: 88px minmax(0, 1fr) 56px 48px;
    gap: var(--s3);
    align-items: center;
    width: 100%;
    min-height: 36px;
    padding: var(--s2) var(--s3);
    border: 0;
    border-radius: var(--r-box);
    background: transparent;
    color: inherit;
    font: inherit;
    text-align: left;
    cursor: pointer;
    transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease);
  }
  /* No label: the meter takes its place. */
  .unit.bare { grid-template-columns: 88px minmax(0, 1fr) 48px; }
  @media (hover: hover) { .unit:hover { background: var(--token-fill); } }
  .unit:active { scale: 0.99; }
  .uid { color: var(--ink); font: 600 14px/20px var(--display); letter-spacing: -0.01em; white-space: nowrap; }
  .ulabel { overflow: hidden; color: var(--ink2); font: var(--type-small); white-space: nowrap; text-overflow: ellipsis; }
  .umeter { position: relative; height: 4px; border-radius: 2px; background: var(--wash2); overflow: hidden; }
  .unit.bare .umeter { max-width: 160px; }
  .umeter > i { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 2px; background: var(--tint); transition: width var(--glide) var(--ease); }
  .umeter.neg > i { background: var(--ink3); opacity: 0.6; }
  .uval { color: var(--ink2); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; }

  /* Where a head reads from: its ID, then its strongest places in the context, the heat as a tint
   * (Ledger's cap, 22%), the query position in ink, the rest elided. */
  .heat { display: grid; gap: var(--s2); }
  .hrow { display: grid; grid-template-columns: 88px minmax(0, 1fr); gap: var(--s3); align-items: baseline; padding: 0 var(--s3); }
  .hid { padding: 0; border: 0; background: transparent; text-align: left; cursor: pointer; transition: color var(--fast) var(--ease); }
  @media (hover: hover) { .hid:hover { color: var(--ink2); } }
  .hline { margin: 0; min-width: 0; color: var(--ink2); font: var(--type-small); line-height: 24px; white-space: pre-wrap; overflow-wrap: anywhere; }
  /* The highlight grows with a same-colour spread, not padding, so the token stays joined to its word. */
  .ht { --heat: color-mix(in oklch, var(--tint) var(--a), transparent); border-radius: 4px; background: var(--heat); box-shadow: 0 0 0 2px var(--heat); cursor: help; }
  .ht.here { color: var(--ink); }
  .gap { color: var(--ink3); }

  /* Full width against the Inspector's scroll edge: the ring sits inside. */
  .lrow:focus-visible, .unit:focus-visible { outline-offset: -2px; }
  @media (max-width: 720px) {
    /* Phone sizes (the Phone canvas): every target 44 px. */
    .lrow { min-height: 44px; }
    .unit { min-height: 48px; }
    .ctl .dw-text { position: relative; }
    .ctl .dw-text::after { content: ""; position: absolute; inset: -14px -8px; }
    .hid { position: relative; }
    .hid::after { content: ""; position: absolute; inset: -12px -8px; }
  }
</style>
