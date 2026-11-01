<script lang="ts">
  // The Units room (the Units canvas): the model's neurons and heads on this conversation. The atlas
  // draws every layer as a column of its units (UnitAtlas): the strongest of each layer, or every
  // neuron of it at one token; the units most active over the latest reply, the clearest writers of
  // a word, or the units the catalog labels. The search names a word (the units that write it) or a
  // unit's ID. Below, the units as rows (ID, label, a meter, the value) and the chosen one in brief:
  // its peak here, its activation token by token (a bar chooses the token the atlas reads), what it
  // writes, and the way to the workbench (which selects it) or to its Neuron Lens page.
  import { tick, untrack } from "svelte";
  import { MediaQuery } from "svelte/reactivity";
  import { Icon, MorphText, Segmented, Swap, Token, formatValue, tokenParts } from "@stdlogits/ledger";
  import UnitAtlas, { type AtlasField } from "./UnitAtlas.svelte";
  import { depthTint } from "../../lib/charts/depth";
  import { chatLog, genStatus } from "../../lib/stores/chat.svelte";
  import { goto } from "../../lib/stores/rooms.svelte";
  import { select, selection } from "../../lib/stores/selection.svelte";
  import {
    fetchCatalog,
    latestReply,
    loadCatalog,
    nodeActivity,
    requestNodeActivity,
    requestUnitRead,
    requestUnitsForWord,
    retryUnits,
    unitActivityAvailable,
    unitLabelFor,
    unitRead,
    unitsAvailable,
    unitsForWord,
    unitsState,
  } from "../../lib/stores/units.svelte";
  import {
    decileCuts,
    decodeFloat16,
    FIRES_AT,
    labelText,
    lensText,
    orderedHalves,
    parseQuery,
    parseUnit,
    peakIndex,
    peakPole,
    poleOf,
    scaleText,
    strongestColumns,
    unitTitle,
    type AtlasBar,
  } from "../../lib/units";
  import type { UnitActivityJSON, UnitLabelJSON } from "../../lib/types";

  type Kind = "neuron" | "head";
  type Filter = "active" | "writes" | "labelled";

  let kind = $state<Kind>("neuron");
  let every = $state(false);
  let filter = $state<Filter>("active");
  let query = $state("");
  let committed = $state("");
  let sel = $state<string | null>(null);
  let layer = $state<number | null>(null);
  let focus = $state<number | null>(null);
  let shown = $state(8);
  let picks = $state(0);
  let search: HTMLInputElement | null = $state(null);
  /** A narrow window takes the search's short placeholder. */
  const narrow = new MediaQuery("max-width: 520px");

  const available = $derived(unitsAvailable());
  $effect(() => {
    if (!unitsState.catalogRead) void loadCatalog();
  });
  const bound = $derived(unitsState.catalog?.bound === true);
  const credit = $derived(bound ? unitsState.catalog?.credit ?? null : null);
  const canFetch = $derived(unitsState.catalog?.available === true && !bound);

  // ---------- the conversation and the token the atlas reads
  const reply = $derived(unitActivityAvailable() ? latestReply() : null);
  $effect(() => {
    void reply?.nodeId;
    untrack(() => { focus = null; });
  });
  /** The workbench's selected token, when it is in the latest reply. */
  const workbenchToken = $derived.by((): number | null => {
    const cur = selection.current;
    if (cur.kind !== "token" || !reply) return null;
    const p = cur.params as { turnIdx?: number; tokenIdx?: number; isThinking?: boolean } | null;
    const turn = p?.turnIdx === undefined ? undefined : chatLog.turns[p.turnIdx];
    if (!turn || turn.nodeId !== reply.nodeId || p?.tokenIdx === undefined) return null;
    return (p.isThinking ? turn.thinkingTokens : turn.tokens)?.[p.tokenIdx]?.rawIndex ?? null;
  });
  const lastRaw = $derived(reply?.tokens.at(-1)?.rawIndex ?? null);
  const focusRaw = $derived(focus ?? workbenchToken ?? lastRaw);

  const request = $derived(reply && focusRaw !== null ? { top: 1024, rawIndex: focusRaw } : null);
  $effect(() => {
    if (reply && request && !genStatus.active) requestNodeActivity(reply.nodeId, request);
  });
  const loading = $derived(reply && request ? nodeActivity(reply.nodeId, request) : null);
  /** The last activity that arrived, held while the next token's loads. */
  let held = $state<UnitActivityJSON | null>(null);
  $effect(() => {
    const value = loading?.value;
    if (value) held = value;
    else if (!reply) held = null;
  });
  const data = $derived.by(() => {
    const value = loading?.value ?? held;
    return value && reply && value.node_id === reply.nodeId ? value : null;
  });
  const focusToken = $derived(data?.tokens.find(t => t.raw_index === (data.population?.raw_index ?? focusRaw)) ?? null);

  const pop = $derived.by(() => {
    const p = data?.population;
    if (!p) return null;
    const sorted = [...p.layers].sort((a, b) => a.layer - b.layer);
    const neurons = sorted.map(l => decodeFloat16(l.neurons));
    const heads = sorted.map(l => decodeFloat16(l.heads));
    const flat = new Float32Array(neurons.reduce((n, a) => n + a.length, 0));
    let at = 0;
    for (const a of neurons) { flat.set(a, at); at += a.length; }
    return {
      neurons, heads, layers: sorted.map(l => l.layer), cuts: decileCuts(flat),
      nNeurons: sorted[0]?.n_neurons ?? 0, nHeads: sorted[0]?.n_heads ?? 0,
    };
  });
  const nLayers = $derived(pop?.neurons.length ?? Math.max(0, ...(data?.top ?? []).map(t => t.layer + 1)));

  // ---------- what the search names, and the word the writers write
  let timer: ReturnType<typeof setTimeout> | null = null;
  function typed(value: string): void {
    query = value;
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => { committed = query; }, 500);
  }
  function commit(): void {
    if (timer) clearTimeout(timer);
    committed = query;
  }
  function clear(): void {
    query = "";
    commit();
    search?.focus();
  }
  const q = $derived(parseQuery(committed));
  /** The word whose writers show: the searched word, else (Writes a word) the token the atlas reads. */
  const word = $derived.by((): { ask: string; shown: string } | null => {
    if (q.mode === "word") return { ask: q.word, shown: q.word };
    if (q.mode === "active" && filter === "writes" && focusToken) return { ask: `#${focusToken.token_id}`, shown: focusToken.token };
    return null;
  });
  $effect(() => {
    if (word && available && !genStatus.active) requestUnitsForWord(word.ask, "writes", 48);
  });
  const writers = $derived(word ? unitsForWord(word.ask, "writes", 48) : null);

  // ---------- the rows
  /** A unit as a row; ``pole`` is the side the row is about (a peak's sign, a writer's pole). */
  interface Row { id: string; layer: number; kind: Kind; value: number | null; label: string | null; pole: "+" | "-" }
  const labelOf = (id: string, pole: "+" | "-", label?: UnitLabelJSON | null) => labelText(label) ?? unitLabelFor(id, pole);
  const activeRows = $derived<Row[]>((data?.top ?? []).map(t => {
    const pole = t.kind === "neuron" && t.peak < 0 ? "-" : "+";
    return { id: t.unit, layer: t.layer, kind: t.kind, value: t.peak, label: labelOf(t.unit, pole, t.label), pole };
  }));
  const writerRows = $derived<Row[]>((writers?.value?.hits ?? []).map(h => {
    const pole = h.pole === "-" ? "-" : "+";
    return { id: h.unit, layer: h.layer, kind: "neuron", value: h.score, label: labelOf(h.unit, pole, h.label), pole };
  }));
  const rows = $derived.by((): Row[] => {
    if (q.mode === "id") {
      if (!q.ref) return [];
      const id = q.ref.id;
      const hit = activeRows.find(r => r.id === id);
      return [hit ?? { id, layer: q.ref.layer, kind: q.ref.kind, value: null, label: unitLabelFor(id), pole: "+" }];
    }
    if (word) return kind === "neuron" ? writerRows : [];
    const mine = activeRows.filter(r => r.kind === kind);
    return filter === "labelled" ? mine.filter(r => r.label) : mine;
  });
  const rowsLoading = $derived(word ? writers?.status === "loading" || writers === null : loading?.status === "loading" && !data);
  const rowsError = $derived(word ? (writers?.status === "error" ? writers.error : null) : loading?.status === "error" ? loading.error : null);
  const top = $derived(Math.max(1e-9, ...rows.map(r => Math.abs(r.value ?? 0))));
  const visible = $derived(rows.slice(0, shown));
  $effect(() => {
    void committed; void filter;
    untrack(() => { shown = 8; });
  });

  const heading = $derived.by((): { pre: string; word: string | null; meta: string } => {
    const units = (n: number) => `${n} ${n === 1 ? "unit" : "units"}`;
    if (q.mode === "id") return { pre: "Matches", word: null, meta: units(rows.length) };
    if (word) return { pre: "Writes ", word: word.shown, meta: writers?.value ? units(rows.length) : "" };
    if (filter === "labelled") return { pre: "Labelled here", word: null, meta: units(rows.length) };
    const totals = pop && nLayers ? `${(nLayers * (kind === "neuron" ? pop.nNeurons : pop.nHeads)).toLocaleString()} ${kind === "neuron" ? "neurons" : "heads"}` : "";
    return { pre: "Most active here", word: null, meta: totals };
  });

  // The selected unit: the one chosen, else the first row; choosing a row picks its layer and kind.
  const selected = $derived(sel ?? rows[0]?.id ?? null);
  const selRef = $derived(selected ? parseUnit(selected) : null);
  // On a phone the detail sits under the list: a chosen row brings it into view.
  let detail: HTMLElement | null = $state(null);
  const phone = new MediaQuery("max-width: 720px");
  function pick(row: Row): void {
    if (row.id !== selected) picks += 1;
    sel = row.id;
    layer = row.layer;
    kind = row.kind;
    if (phone.current) {
      void tick().then(() => detail?.scrollIntoView({
        block: "start",
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
      }));
    }
  }
  const atLayer = $derived(layer ?? selRef?.layer ?? 0);

  // ---------- the atlas
  const columns = $derived.by((): AtlasBar[][] => {
    if (!nLayers) return [];
    if (kind === "head") {
      return strongestColumns((data?.top ?? []).filter(t => t.kind === "head"), nLayers, pop?.nHeads ?? 0);
    }
    const source = word && q.mode !== "id"
      ? writerRows.map(r => ({ unit: r.id, layer: r.layer, index: parseUnit(r.id)?.index ?? 0, peak: r.value ?? 0 }))
      : (data?.top ?? []).filter(t => t.kind === "neuron" && (filter !== "labelled" || labelOf(t.unit, t.peak < 0 ? "-" : "+", t.label)));
    return strongestColumns(source, nLayers, 40);
  });
  const rowsPerColumn = $derived(kind === "head" ? Math.max(1, pop?.nHeads ?? 1) : every ? 40 : Math.max(8, ...columns.map(col => col.length)));
  const lit = $derived.by((): Set<string> | null => {
    if (q.mode === "id") return q.ref ? new Set([q.ref.id]) : new Set();
    if (q.mode === "word" && kind === "head") return new Set();
    return null;
  });
  const showAll = $derived(every && kind === "neuron" && pop !== null);
  const field = $derived<AtlasField | null>(showAll && pop ? { layers: pop.neurons, cuts: pop.cuts } : null);
  const hits = $derived(showAll
    ? q.mode === "id" ? (q.ref?.kind === "neuron" ? [{ layer: q.ref.layer, index: q.ref.index }] : [])
      : q.mode === "word" ? writerRows.map(r => ({ layer: r.layer, index: parseUnit(r.id)?.index ?? 0 })) : []
    : []);
  const pin = $derived(showAll && selRef?.kind === "neuron" ? { layer: selRef.layer, index: selRef.index } : null);

  const subset = $derived.by(() => {
    if (!pop) return "";
    if (kind === "head") return `all ${pop.nHeads} heads a layer`;
    const n = pop.nNeurons.toLocaleString();
    if (every) return `all ${n} a layer, in neuron order`;
    if (word && q.mode !== "id") return `the clearest writers of ${n} a layer`;
    if (filter === "labelled") return `the labelled of ${n} a layer`;
    return `the 40 most active of ${n} a layer`;
  });
  const layerFact = $derived.by(() => {
    if (!pop) return "";
    const values = kind === "head" ? pop.heads[atLayer] : pop.neurons[atLayer];
    if (!values) return "";
    if (word && kind === "neuron" && q.mode !== "id") {
      // Until the search answers there is no count to state.
      if (!writers?.value) return "";
      const n = writerRows.filter(r => r.layer === atLayer).length;
      return `L${atLayer} · ${n} write ${tokenParts(word.shown).lead ? "·" : ""}${tokenParts(word.shown).text}`;
    }
    let n = 0;
    for (const v of values) if (Math.abs(v) >= FIRES_AT) n += 1;
    return `L${atLayer} · ${n.toLocaleString()} of ${values.length.toLocaleString()} active here`;
  });

  // ---------- the selected unit, in brief
  $effect(() => {
    if (selected && available && !genStatus.active) requestUnitRead(selected, true);
  });
  $effect(() => {
    if (selected && reply && !genStatus.active) requestNodeActivity(reply.nodeId, { units: [selected], top: 1 });
  });
  const selRead = $derived(selected ? (unitRead(selected, true)?.value ?? null) : null);
  const selTrace = $derived(selected && reply ? nodeActivity(reply.nodeId, { units: [selected], top: 1 })?.value ?? null : null);
  const trace = $derived(selected ? selTrace?.traces[selected] ?? null : null);
  const tokens = $derived(selTrace?.tokens ?? []);
  const peakAt = $derived(trace ? peakIndex(trace) : -1);
  const peakMag = $derived(trace && peakAt >= 0 ? Math.abs(trace[peakAt] ?? 0) : 0);
  /** The side of a neuron the summary reads, as the unit view opens it: the side that fired here
   *  (its trace's peak; the row's sign while the trace loads), or, for a writer of the searched
   *  word, the side that writes it. Its writes, label and link all come from that side. */
  const selRow = $derived(selected ? rows.find(r => r.id === selected) ?? null : null);
  const selPole = $derived.by((): "+" | "-" => {
    if (selRef?.kind !== "neuron") return "+";
    if (word && q.mode !== "id" && selRow) return selRow.pole;
    if (trace && peakAt >= 0) return peakPole(trace);
    return selRow?.pole ?? "+";
  });
  const selSide = $derived(selRead?.kind === "neuron" ? poleOf(selRead, selPole) : null);
  const selLabel = $derived(selRead ? labelText(selRead.kind === "neuron" ? selSide?.label : selRead.label) : selected ? unitLabelFor(selected, selPole) : null);
  const selLink = $derived(selRead ? (selRead.kind === "neuron" ? selSide?.label?.url ?? (selRead.catalog.bound ? selRead.catalog.url : null) : selRead.label?.url ?? null) : null);
  const writes = $derived.by(() => {
    if (!selRead) return null;
    const place = selRead.kind === "neuron" ? selSide?.writes : selRead.copies;
    if (!place) return null;
    const { primary } = orderedHalves(place);
    return { lens: lensText(primary), absent: primary.absent, tokens: primary.tokens };
  });
  const isHit = (t: string) => word !== null && t.trim().toLowerCase() === word.shown.trim().toLowerCase();
  const selFact = $derived.by((): { b: string; pre: string; post: string } | null => {
    if (!selRead) return null;
    if (selRead.kind === "head") {
      const v = selRead.firing.scale?.top;
      const post = selRead.firing.scale?.source === "neutral" ? " on the neutral text" : "";
      return typeof v === "number" ? { pre: "typical write ", b: formatValue(v), post } : null;
    }
    const rate = selSide?.firing.rate;
    if (typeof rate === "number") return { pre: "fires on ", b: `${(rate * 100).toFixed(rate >= 0.1 ? 0 : 1)}%`, post: " of tokens" };
    return { pre: "write norm ", b: formatValue(selRead.scalars.w_norm), post: "" };
  });
  const tint = $derived(selRef && nLayers > 1 ? depthTint(selRef.layer / (nLayers - 1)) : null);

  function signed(v: number): string {
    const t = formatValue(Math.abs(v));
    return v < 0 ? `−${t}` : t;
  }
  function open(): void {
    if (!selected) return;
    select({ kind: "unit", id: selected });
    goto("workbench");
  }
  function onKey(event: KeyboardEvent): void {
    if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) return;
    const target = event.target as HTMLElement | null;
    if (target?.closest("input, textarea, [contenteditable='true']")) return;
    event.preventDefault();
    search?.focus();
  }
</script>

<svelte:window onkeydown={onKey} />

{#snippet ws(token: string)}
  {@const parts = tokenParts(token)}{#if parts.lead}<span class="dw-ws">·</span>{/if}{parts.empty ? "∅" : parts.text}
{/snippet}

<div class="dw-room units">
  {#if !available}
    <div class="dw-empty"><span class="head">Units</span><p>This runtime does not read neurons and heads.</p></div>
  {:else}
    <div class="tools">
      <Segmented label="Units" value={kind} onchange={(v) => { kind = v; if (v === "head" && filter === "writes") filter = "active"; }} options={[{ value: "neuron", label: "Neurons" }, { value: "head", label: "Heads" }]} />
      <span class="seg-wrap" class:off={kind === "head"} inert={kind === "head"}>
        <Segmented label="Which neurons" value={every ? "all" : "strongest"} onchange={(v) => (every = v === "all")}
          options={[{ value: "strongest", label: "Strongest" }, { value: "all", label: "All" }]} />
      </span>
      <Segmented label="Units shown" value={filter} onchange={(v) => (filter = v)}
        options={[{ value: "active", label: "Active here" }, ...(kind === "neuron" ? [{ value: "writes" as const, label: "Writes a word" }] : []), ...(bound ? [{ value: "labelled" as const, label: "Labelled" }] : [])]} />
      <div class="search">
        <Icon name="search" size={18} />
        <label class="sr-only" for="unit-q">Search units</label>
        <input id="unit-q" type="search" autocomplete="off" spellcheck="false" bind:this={search} value={query}
          placeholder={narrow.current ? "A word, or an ID like L20 n344" : "Search a word, or an ID like L20 n344"}
          oninput={(e) => typed((e.currentTarget as HTMLInputElement).value)}
          onkeydown={(e) => { if (e.key === "Enter") commit(); if (e.key === "Escape" && query) { e.preventDefault(); clear(); } }} />
        {#if query}
          <button type="button" class="clear" aria-label="Clear the search" onclick={clear}><Icon name="close" /></button>
        {:else}
          <kbd aria-hidden="true">/</kbd>
        {/if}
      </div>
    </div>

    <section class="atlas-sec" aria-label="Atlas">
      <div class="dw-sec-head">
        <span class="ah">
          {#if focusToken && (kind === "head" || every || !word)}<span class="head">At {@render ws(focusToken.token)}</span>{:else if word}<span class="head">Writes {@render ws(word.shown)}</span>{/if}
          <span class="meta">{subset}</span>
        </span>
        <span class="meta"><MorphText text={layerFact} /></span>
      </div>
      {#if columns.length > 0}
        <UnitAtlas {columns} rows={rowsPerColumn} {lit} selected={selected} layer={atLayer} onlayer={(L) => (layer = L)}
          {field} {hits} {pin} label="Layers" />
      {:else if loading?.status === "error"}
        <div class="dw-row"><span class="meta">{loading.error}</span><button type="button" class="dw-text" onclick={retryUnits}>Try again</button></div>
      {:else if !reply}
        <div class="ghost-atlas none"><span class="meta">Units light up here once the model has written a reply.</span></div>
      {:else}
        <div class="ghost-atlas" aria-hidden="true"></div>
      {/if}
    </section>

    <div class="below">
      <section class="list" aria-label="Results">
        <div class="dw-sec-head">
          <span class="head">{heading.pre}{#if heading.word !== null}{@render ws(heading.word)}{/if}</span>
          <span class="meta">{heading.meta}</span>
        </div>
        {#if visible.length > 0}
          <div class="rows">
            {#each visible as r (r.id)}
              <button type="button" class="row" aria-pressed={r.id === selected}
                style:--tint={depthTint(nLayers > 1 ? r.layer / (nLayers - 1) : null) ?? undefined} onclick={() => pick(r)}>
                <span class="uid">{unitTitle(r.id)}</span>
                <span class="ulabel">{r.label ?? ""}</span>
                <span class="meter" class:neg={(r.value ?? 0) < 0}><i style:width="{Math.round((Math.abs(r.value ?? 0) / top) * 100)}%"></i></span>
                <span class="uval">{r.value === null ? "–" : signed(r.value)}</span>
              </button>
            {/each}
          </div>
        {:else if rowsError}
          <div class="dw-row"><span class="meta">{rowsError}</span><button type="button" class="dw-text" onclick={retryUnits}>Try again</button></div>
        {:else if rowsLoading}
          <div class="ghost-rows" aria-hidden="true">{#each [0, 1, 2, 3] as k (k)}<i></i>{/each}</div>
        {:else if q.mode === "id"}
          <p class="meta none">No unit {committed.trim()}</p>
        {:else if word}
          <p class="meta none">No unit writes {@render ws(word.shown)}</p>
        {:else if data}
          <p class="meta none">{filter === "labelled" ? "No unit here has a label" : kind === "head" ? "No head wrote here" : "No neuron fired here"}</p>
        {/if}
        <div class="split">
          {#if rows.length > shown}<button type="button" class="dw-text" onclick={() => (shown += 16)}>{rows.length - shown} more</button>{/if}
          {#if data && scaleText(data.scale.source)}<span class="meta" data-tip={data.scale.detail}>Values over each unit's {scaleText(data.scale.source)}</span>{/if}
          {#if credit}<span class="meta credit">{credit}</span>{:else if canFetch}
            <button type="button" class="dw-text credit" disabled={unitsState.fetching} onclick={() => void fetchCatalog()}><MorphText text={unitsState.fetching ? "Fetching labels…" : "Fetch labels"} /></button>
          {/if}
        </div>
      </section>

      {#if selected}
        <Swap key={`${selected}:${picks}`}>
          <section class="sum" aria-label="Selected unit" style:--tint={tint ?? undefined} bind:this={detail}>
            <div class="name">
              <h2 class="dw-title">{unitTitle(selected)}</h2>
              {#if selLabel}<p class="label">{selLabel}</p>{/if}
            </div>
            <div class="dw-facts">
              {#if trace && peakAt >= 0}<span>peak here <b>{signed(trace[peakAt] ?? 0)}</b> on {@render ws(tokens[peakAt]?.token ?? "")}</span>{/if}
              {#if selFact}<span>{selFact.pre}<b>{selFact.b}</b>{selFact.post}</span>{/if}
            </div>
            {#if trace && trace.length > 0}
              <div class="strip" style:--n={trace.length} role="group" aria-label={`${unitTitle(selected)} on each of the ${trace.length} tokens; choose one to read the atlas there`}>
                {#each trace as v, i (i)}
                  {@const a = Math.abs(v ?? 0)}
                  <button type="button" class="s" class:on={tokens[i]?.raw_index === focusRaw} aria-label={`${tokens[i]?.token ?? ""} ${signed(v ?? 0)}`}
                    onclick={() => (focus = tokens[i]?.raw_index ?? null)}>
                    <i class:neg={(v ?? 0) < 0} class:lit={peakMag > 0 && a >= 0.4 * peakMag} style:height="{Math.max(2, peakMag > 0 ? (a / peakMag) * 32 : 2)}px"></i>
                  </button>
                {/each}
              </div>
            {/if}
            {#if writes}
              <div class="group">
                <div class="dw-row"><h3 class="head">{selRef?.kind === "head" ? "Copies" : "Writes"}</h3><span class="meta">{writes.lens}</span></div>
                {#if writes.absent}
                  <span class="meta">{writes.absent}</span>
                {:else}
                  <div class="toks">{#each writes.tokens.slice(0, 8) as t, i (`${i}:${t.token}`)}<Token token={t.token} flags={isHit(t.token) ? 1 : 0} {i} />{/each}</div>
                {/if}
              </div>
            {:else}
              <div class="ghost-toks" aria-hidden="true">{#each [56, 72, 48, 64, 80] as w, j (j)}<i style:width="{w}px"></i>{/each}</div>
            {/if}
            <div class="dw-row">
              <button type="button" class="dw-quiet" onclick={open}>Open in workbench<Icon name="right" /></button>
              {#if selLink}<a class="out" href={selLink} target="_blank" rel="noopener">Open in Neuron Lens<Icon name="external" /></a>{/if}
            </div>
          </section>
        </Swap>
      {/if}
    </div>
  {/if}
</div>

<style>
  .units { gap: var(--s6); }
  .tools { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s3); }
  .seg-wrap { display: inline-flex; transition: opacity var(--fast) var(--ease); }
  .seg-wrap.off { opacity: 0.4; }
  .search { flex: 1 1 320px; display: flex; align-items: center; gap: var(--s2); min-width: 0; height: 44px; padding: 0 var(--s2) 0 var(--s4); border-radius: var(--r-pill); background: var(--wash); color: var(--ink3); transition: background-color var(--fast) var(--ease); }
  .search:focus-within { background: var(--wash2); }
  .search input { flex: 1 1 auto; min-width: 0; height: 100%; padding: 0; border: 0; outline: 0; background: transparent; color: var(--ink); font: var(--type-body); }
  .search input::placeholder { color: var(--ink3); }
  .search input::-webkit-search-cancel-button { display: none; }
  .search kbd { display: grid; place-items: center; min-width: 24px; height: 24px; margin-right: var(--s2); border-radius: var(--r-chip); background: var(--paper); color: var(--ink3); font: var(--type-meta); }
  .clear { display: grid; place-items: center; width: 32px; height: 32px; padding: 0; border: 0; border-radius: var(--r-pill); background: transparent; color: var(--ink3); cursor: pointer; transition: color var(--fast) var(--ease), background-color var(--fast) var(--ease); }
  .clear:hover { color: var(--ink); background: var(--paper); }

  .atlas-sec { display: grid; gap: var(--s3); min-width: 0; }
  .ah { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s1) var(--s3); min-width: 0; }
  .ghost-atlas { display: grid; place-items: center; height: 214px; border-radius: var(--r-box); background: var(--wash); }
  .ghost-atlas.none { padding: var(--s4); }

  .below { display: flex; flex-wrap: wrap; align-items: flex-start; gap: var(--s7); }
  .list { flex: 999 1 520px; max-width: 760px; min-width: 0; display: grid; gap: var(--s3); }
  .rows { display: grid; gap: 2px; }
  .row { display: grid; grid-template-columns: 88px minmax(0, 1fr) 56px 44px; gap: var(--s3); align-items: center; height: 36px; padding: 0 var(--s3); border: 0; border-radius: var(--r-box); background: transparent; color: inherit; text-align: left; cursor: pointer; transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .row:hover { background: var(--wash); }
  .row[aria-pressed="true"] { background: var(--wash2); }
  .row:active { scale: 0.99; }
  .uid { color: var(--ink); font: 600 14px/20px var(--display); letter-spacing: -0.01em; white-space: nowrap; }
  .ulabel { overflow: hidden; color: var(--ink2); font: var(--type-small); white-space: nowrap; text-overflow: ellipsis; }
  .row[aria-pressed="true"] .ulabel { color: var(--ink); }
  .meter { position: relative; height: 4px; border-radius: 2px; background: var(--wash2); }
  .row[aria-pressed="true"] .meter { background: var(--paper); }
  .meter > i { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 2px; background: var(--tint); transition: width var(--glide) var(--ease); }
  .meter.neg > i { background: var(--ink3); opacity: 0.6; }
  .uval { color: var(--ink2); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; }
  .split { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2); }
  .split .credit { margin-left: auto; }
  .none { margin: 0; padding: var(--s2) var(--s3); }
  .ghost-rows { display: grid; gap: 2px; }
  .ghost-rows > i { display: block; height: 36px; border-radius: var(--r-box); background: var(--wash); }

  .sum { flex: 1 1 360px; max-width: 520px; min-width: 0; display: grid; gap: var(--s4); align-content: start; }
  .name { display: grid; gap: var(--s1); }
  .label { margin: 0; color: var(--ink); font: var(--type-label); }
  .strip { display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); align-items: end; height: 40px; }
  .s { display: flex; align-items: flex-end; justify-content: center; height: 100%; min-width: 0; padding: 0 0 4px; border: 0; border-radius: 4px; background: transparent; cursor: pointer; transition: background-color var(--fast) var(--ease); }
  .s:hover, .s.on { background: var(--wash); }
  .s > i { display: block; width: 62%; min-height: 2px; border-radius: 2px; background: var(--tint); opacity: 0.4; transition: height var(--glide) var(--ease), opacity var(--glide) var(--ease); }
  .s > i.lit { opacity: 1; }
  .s > i.neg { background: var(--ink3); opacity: 0.4; }
  .group { display: grid; gap: var(--s2); min-width: 0; }
  .group h3 { margin: 0; }
  .ghost-toks { display: flex; flex-wrap: wrap; gap: var(--s1); }
  .ghost-toks > i { display: block; height: 24px; border-radius: var(--r-chip); background: var(--wash); }
  .out { display: inline-flex; align-items: center; gap: 6px; height: 36px; color: var(--ink2); font: var(--type-control); text-decoration: none; transition: color var(--fast) var(--ease); }
  .out:hover { color: var(--ink); }

  /* Phone sizes and touch: every target 44 px (a token's bar is a column of the strip, so the strip
   * grows instead), and no key hint where there is no keyboard. */
  @media (max-width: 720px), (pointer: coarse) {
    .row { height: 44px; }
    .clear { width: 44px; height: 44px; }
    .strip { height: 48px; }
    .split .dw-text { min-height: 44px; }
  }
  @media (pointer: coarse) {
    .search kbd { display: none; }
  }
  @media (max-width: 520px) {
    .row { grid-template-columns: 80px minmax(0, 1fr) 40px 40px; gap: var(--s2); }
  }
</style>
