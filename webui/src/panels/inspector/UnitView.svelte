<script lang="ts">
  // A unit selected (the Unit canvas): a neuron or an attention head, from its weights and on this
  // conversation. Its ID, the catalog's label with its credit and its Neuron Lens page (when the
  // model is catalogued and the labels are fetched), and its facts. A neuron has two poles, read
  // one side at a time: what it writes (its write at the block's output) and what it fires on (its
  // read direction after attention), each through the lens its depth calls for, the other lens's
  // read faded beneath. A head shows what it copies and suppresses (its OV map) and what its query,
  // key and value read. "Hide hub tokens" ranks every read by the layer-contrastive z, so the
  // tokens every unit of the layer reads drop out. Then its activation on each token of the latest
  // reply, and what it can do in the recipe: be probed, pushed (a neuron, either pole) or silenced
  // (exact mean-ablation).
  import { Icon, MorphText, Segmented, Switch, Swap, Token, formatValue } from "@stdlogits/ledger";
  import UnitTrace from "./UnitTrace.svelte";
  import { depthTint } from "../../lib/charts/depth";
  import { genStatus } from "../../lib/stores/chat.svelte";
  import {
    fetchCatalog,
    latestReply,
    loadCatalog,
    nodeActivity,
    requestNodeActivity,
    requestUnitRead,
    retryUnits,
    toggleUnitProbe,
    toggleUnitTerm,
    unitActivityAvailable,
    unitProbeBusy,
    unitProbed,
    unitRead,
    unitTerm,
    unitsState,
    type UnitRead,
  } from "../../lib/stores/units.svelte";
  import { steerRack } from "../../lib/stores/steering.svelte";
  import { formatUnitTerm } from "../../lib/expression";
  import {
    firesOn,
    labelText,
    lensText,
    orderedHalves,
    parseUnit,
    peakIndex,
    poleOf,
    unitFacts,
    unitTitle,
    withAgreement,
    type UnitTerm,
  } from "../../lib/units";
  import type { UnitCatalogBlockJSON, UnitLabelJSON, UnitPlaceJSON } from "../../lib/types";

  let { id }: { id: string } = $props();

  const ref = $derived(parseUnit(id));
  const isHead = $derived(ref?.kind === "head");
  let pole = $state<"+" | "-">("+");
  /** The side the person chose; until then the view opens on the side that fired here. */
  let chosen = $state(false);
  let hideHubs = $state(true);

  $effect(() => {
    if (!unitsState.catalogRead) void loadCatalog();
  });
  $effect(() => {
    if (!genStatus.active) requestUnitRead(id, hideHubs);
  });
  const loaded = $derived(unitRead(id, hideHubs));
  /** The read in the other hub mode stands in while this one loads, so the switch never blanks. */
  const read = $derived<UnitRead | null>(loaded?.value ?? unitRead(id, !hideHubs)?.value ?? null);
  const tint = $derived(depthTint(read?.depth));

  const label = $derived.by((): UnitLabelJSON | null => {
    if (!read) return null;
    if (read.kind === "head") return read.label;
    return poleOf(read, pole)?.label ?? null;
  });
  const catalogBlock = $derived(read && "catalog" in read ? (read as { catalog?: UnitCatalogBlockJSON }).catalog ?? null : null);
  const link = $derived(label?.url ?? (catalogBlock?.bound ? catalogBlock.url : null));
  const credit = $derived(label?.credit ?? (catalogBlock?.bound ? catalogBlock.credit : null) ?? (unitsState.catalog?.bound ? unitsState.catalog.credit : null));
  const facts = $derived(read ? unitFacts(read, pole) : null);
  const canFetch = $derived(unitsState.catalog?.available === true && !unitsState.catalog.bound);

  // ---------- reads
  interface Group { title: string; place: UnitPlaceJSON }
  const groups = $derived.by((): Group[][] => {
    if (!read) return [];
    if (read.kind === "head") {
      return [
        [{ title: "Copies", place: read.copies }, { title: "Suppresses", place: read.suppresses }],
        [{ title: "Query", place: read.inputs.query }, { title: "Key", place: read.inputs.key }, { title: "Value", place: read.inputs.value }],
      ];
    }
    const p = poleOf(read, pole);
    return p ? [[{ title: "Writes", place: p.writes }, { title: "Fires on", place: p.fires_on }]] : [];
  });
  /** Each row's groups, the halves in depth order, with agreement across the row's primary reads. */
  const rows = $derived(groups.map(row => {
    const halves = row.map(g => orderedHalves(g.place));
    const primary = withAgreement(halves.map(h => (h.primary.absent ? [] : h.primary.tokens)));
    const other = withAgreement(halves.map(h => (h.other.absent ? [] : h.other.tokens)));
    return row.map((g, i) => ({ ...g, ...halves[i], primaryTokens: primary[i], otherTokens: other[i] }));
  }));

  // ---------- on this conversation
  const reply = $derived(unitActivityAvailable() ? latestReply() : null);
  $effect(() => {
    if (reply && !genStatus.active) requestNodeActivity(reply.nodeId, { units: [id], top: 1 });
  });
  const activity = $derived(reply ? nodeActivity(reply.nodeId, { units: [id], top: 1 }) : null);
  const trace = $derived(activity?.value?.traces[id] ?? null);
  const traceTokens = $derived(activity?.value?.tokens.map(t => t.token) ?? []);
  const sign = $derived<1 | -1>(!isHead && pole === "-" ? -1 : 1);
  const fired = $derived(trace ? firesOn(trace.map(v => (v === null ? null : Math.max(0, v * sign)))) : 0);
  $effect(() => {
    if (chosen || isHead || !trace) return;
    const at = peakIndex(trace);
    if (at >= 0) pole = (trace[at] ?? 0) < 0 ? "-" : "+";
  });

  // ---------- the recipe
  const term = $derived(unitTerm(id));
  const entry = $derived(steerRack.entries.get(id));
  const expression = $derived(entry?.mode === "unit" && term ? formatUnitTerm(id, entry) : null);
  const probed = $derived(unitProbed(id));
  const actions = $derived<{ term: UnitTerm; label: string }[]>(isHead
    ? [{ term: "silence", label: "Silence" }]
    : [{ term: "plus", label: "Push +" }, { term: "minus", label: "Push −" }, { term: "silence", label: "Silence" }]);

  function factText(v: number): string {
    return v < 0 ? `−${formatValue(-v)}` : formatValue(v);
  }
  function rateText(r: number): string {
    const p = r * 100;
    return `${p !== 0 && p < 0.1 ? "<0.1" : p.toFixed(p >= 10 ? 0 : 1)}%`;
  }
</script>

{#snippet toks(list: { token: string; count: number; agree: boolean }[])}
  <div class="toks">
    {#each list as t, i (`${i}:${t.token}`)}
      <Token token={t.token} count={t.count} flags={t.agree ? 1 : 0} {i} />
    {/each}
  </div>
{/snippet}

<div class="unit" style:--tint={tint ?? undefined}>
  <div class="dw-sec name">
    <h2 class="dw-title">{unitTitle(id)}</h2>
    {#if labelText(label)}<p class="label">{labelText(label)}</p>{/if}
    {#if credit || link || canFetch}
      <div class="dw-row split">
        {#if credit}<span class="meta">{credit}</span>{/if}
        {#if link}
          <a class="out" href={link} target="_blank" rel="noopener">Open in Neuron Lens<Icon name="external" /></a>
        {:else if canFetch}
          <button type="button" class="dw-text" disabled={unitsState.fetching} onclick={() => void fetchCatalog()}>
            <MorphText text={unitsState.fetching ? "Fetching labels…" : "Fetch labels"} />
          </button>
        {/if}
      </div>
    {/if}
    {#if facts}
      <div class="dw-facts">
        {#if facts.typical !== null}<span>typical {isHead ? "write" : "firing"} <b>{factText(facts.typical)}</b></span>{/if}
        {#if facts.rate !== null}<span>fires on <b>{rateText(facts.rate)}</b> of tokens</span>{/if}
        {#if facts.writeNorm !== null}<span>write norm <b>{factText(facts.writeNorm)}</b></span>{/if}
        {#if facts.copy !== null}<span>copy score <b>{factText(facts.copy)}</b></span>{/if}
      </div>
    {:else}
      <div class="ghost-line" aria-hidden="true"></div>
    {/if}
  </div>

  <div class="dw-sec reads">
    <div class="dw-row split">
      {#if !isHead}
        <Segmented label="Side" value={pole} onchange={(v) => { pole = v; chosen = true; }}
          options={[{ value: "+", label: "+ side" }, { value: "-", label: "− side" }]} />
      {/if}
      <span class="hubs"><Switch checked={hideHubs} onchange={(on) => (hideHubs = on)} label="Hide hub tokens" /></span>
    </div>

    {#if read}
      <Swap key={`${pole}|${read === loaded?.value ? "now" : "stale"}`}>
        <div class="stack" class:stale={read !== loaded?.value}>
          {#each rows as row, r (r)}
            <div class="grid" class:three={row.length === 3}>
              {#each row as g (g.title)}
                <div class="group">
                  <div class="gh"><h3 class="head">{g.title}</h3><span class="meta">{lensText(g.primary)}</span></div>
                  {#if g.primary.absent}
                    <span class="meta">{g.primary.absent}</span>
                  {:else}
                    {@render toks(g.primaryTokens)}
                  {/if}
                  {#if !g.other.absent && g.otherTokens.length > 0}
                    <div class="other">
                      <span class="meta">{lensText(g.other)}</span>
                      {@render toks(g.otherTokens.slice(0, 6))}
                    </div>
                  {/if}
                </div>
              {/each}
            </div>
          {/each}
        </div>
      </Swap>
    {:else if loaded?.status === "error"}
      <div class="dw-row"><span class="meta">{loaded.error}</span><button type="button" class="dw-text" onclick={retryUnits}>Try again</button></div>
    {:else}
      <div class="grid" aria-hidden="true">
        {#each [0, 1] as k (k)}<div class="group"><i class="ghost-head"></i><div class="toks">{#each [56, 72, 48, 64, 80, 52] as w, j (j)}<i class="ghost-tok" style:width="{w}px"></i>{/each}</div></div>{/each}
      </div>
    {/if}
  </div>

  {#if reply && (trace || activity?.status === "loading" || activity?.status === "error")}
    <section class="dw-sec" aria-labelledby="unit-here-head">
      <div class="dw-sec-head">
        <span class="head" id="unit-here-head">On this conversation</span>
        {#if trace}<span class="meta">fires on {fired} of {trace.length} tokens</span>{/if}
      </div>
      {#if trace}
        <UnitTrace values={trace} tokens={traceTokens} {sign} label={`${unitTitle(id)} on each of the ${trace.length} tokens of the reply`} />
      {:else if activity?.status === "error"}
        <div class="dw-row"><span class="meta">{activity.error}</span><button type="button" class="dw-text" onclick={retryUnits}>Try again</button></div>
      {:else}
        <i class="ghost-plot" aria-hidden="true"></i>
      {/if}
    </section>
  {/if}

  <div class="dw-sec acts">
    <div class="dw-row">
      <button type="button" class="dw-quiet act" aria-pressed={probed} disabled={unitProbeBusy(id)} onclick={() => void toggleUnitProbe(id)}>
        {#if probed}<Icon name="check" />{/if}{probed ? "Probing" : "Probe it"}
      </button>
      {#each actions as a (a.term)}
        <button type="button" class="dw-quiet act" aria-pressed={term === a.term} onclick={() => toggleUnitTerm(id, a.term)}>
          {#if term === a.term}<Icon name="check" />{/if}{a.label}
        </button>
      {/each}
    </div>
    {#if expression}<span class="meta expr">{expression}</span>{/if}
  </div>
</div>

<style>
  .unit { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s7); min-width: 0; }
  .name { gap: var(--s1); }
  .label { margin: 0; color: var(--ink); font: var(--type-label); }
  .split { justify-content: space-between; }
  .out { display: inline-flex; align-items: center; gap: 6px; height: 36px; color: var(--ink2); font: var(--type-control); text-decoration: none; transition: color var(--fast) var(--ease); }
  .out:hover { color: var(--ink); }
  .name .dw-facts { margin-top: var(--s2); }

  .reads { gap: var(--s5); }
  .stack { display: grid; gap: var(--s5); min-width: 0; transition: opacity var(--fast) var(--ease); }
  .stack.stale { opacity: 0.6; }
  .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s5); min-width: 0; }
  .grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--s4); }
  .group { display: grid; gap: var(--s2); align-content: start; min-width: 0; }
  .gh { display: flex; align-items: baseline; gap: var(--s2); min-height: 16px; }
  .gh h3 { margin: 0; }
  .other { display: grid; gap: var(--s1); margin-top: var(--s1); opacity: 0.45; transition: opacity var(--fast) var(--ease); }
  .other:hover { opacity: 1; }

  .acts { gap: var(--s2); }
  .act[aria-pressed="true"] { background: var(--wash2); }
  .expr { overflow-wrap: anywhere; }

  /* Loading: wash blocks where the reads will land. */
  .ghost-line { width: 60%; height: 16px; margin-top: var(--s2); border-radius: var(--r-chip); background: var(--wash); }
  .ghost-head { display: block; width: 64px; height: 16px; border-radius: var(--r-chip); background: var(--wash); }
  .ghost-tok { display: block; height: 24px; border-radius: var(--r-chip); background: var(--wash); }
  .ghost-plot { display: block; height: 112px; border-radius: var(--r-box); background: var(--wash); }

  /* On the phone's sheet (a wash), the side switch steps up to wash2 like the sheet's other pieces. */
  :global(.sheet-body) .reads :global(.seg) { background: var(--wash2); }
  @media (max-width: 520px) {
    .grid, .grid.three { grid-template-columns: minmax(0, 1fr); }
  }
  @media (max-width: 720px), (pointer: coarse) {
    .hubs :global(.switch) { height: 44px; }
  }
</style>
