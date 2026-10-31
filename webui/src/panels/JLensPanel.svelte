<script lang="ts">
  // The lens, live (the Instruments canvas's Live section). A switch turns the readout on; with it
  // on, three depths show what the lens reads there (the top words at an early, a middle and a late
  // layer), then the words it reads most strongly across the fitted layers, pinned ones (persistent
  // ``jlens/<word>`` probes, gate-able) with the rest, in one sort. One field takes a word to watch
  // (pin) or to steer toward (an ``α jlens/<word>`` term in the recipe, edited from its chip). While
  // a token is hovered, everything here reads that token.
  import { flip } from "svelte/animate";
  import { GLIDE, MorphText, Switch, Token, ease, reducedMotion } from "@stdlogits/ledger";
  import JLensProbeCard from "./rack/JLensProbeCard.svelte";
  import { mergeInstrumentProbeRows } from "./rack/probeRows";
  import type { InstrumentProbeRow } from "./rack/probeRows";
  import { depthTint, layerDepth } from "../lib/charts/depth";
  import { apiInstruments, describeError } from "../lib/runtime/services";
  import { goto } from "../lib/stores/rooms.svelte";
  import {
    addJLensToRack,
    activeProbeNames,
    attachProbe,
    lensState,
    lensAggregateForDisplay,
    lensReadoutForDisplay,
    probeRack,
    probeEntryForDisplay,
    sessionState,
    setLensWorkspaceSortMode,
    setLiveLens,
    tokenHoverState,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { ProbeRackEntry, ScalarReadingJSON } from "../lib/types";
  import type { LensWorkspaceSortMode } from "../lib/stores.svelte";

  const fitted = $derived(sessionState.info?.jlens_fitted === true);
  const liveOn = $derived(lensState.layers !== null);
  const displayReadout = $derived(lensReadoutForDisplay());
  const displayAggregate = $derived(lensAggregateForDisplay());
  const displayLayers = $derived.by(() => {
    if (!tokenHoverState.active) return lensState.layers ?? [];
    return Object.keys(displayReadout ?? {}).map(Number).sort((a, b) => a - b);
  });
  const deepest = $derived(Math.max(0, ...displayLayers));

  // ---------- three depths: the top words at an early, a middle and a late layer ----------
  const bands = $derived.by(() => {
    const layers = displayLayers.filter((layer) => (displayReadout?.[String(layer)]?.length ?? 0) > 0);
    if (layers.length === 0) return [];
    const picks = layers.length <= 3
      ? layers.map((layer) => ({ layer, name: `L${layer}` }))
      : [1 / 6, 1 / 2, 5 / 6].map((at, i) => ({
          layer: layers[Math.round(at * (layers.length - 1))],
          name: ["early", "middle", "late"][i],
        }));
    return picks.map(({ layer, name }) => ({
      layer,
      name,
      tint: depthTint(layerDepth(layer, deepest)),
      words: (displayReadout?.[String(layer)] ?? []).slice(0, 3),
    }));
  });

  // ---------- the words: pinned probes and the live aggregate, one roster ----------
  interface WorkspaceRow extends InstrumentProbeRow {
    token: string;
    strength: number;
    com: number | null;
    spread: number | null;
    measured: boolean;
    cells: { layer: number; p: number | null }[];
    pinned: boolean;
  }

  /** A pinned probe's per-layer readings (the store's axis-0 map). */
  function pinnedCells(entry: ProbeRackEntry): { layer: number; p: number | null }[] {
    const perLayer = entry.perLayer;
    if (!perLayer) return [];
    return Object.keys(perLayer)
      .sort((a, b) => Number(a) - Number(b))
      .map((layer) => ({ layer: Number(layer), p: perLayer[layer] ?? null }));
  }

  /** A found word's probability at each streamed layer; null = below that layer's top-k. */
  function readoutCells(token: string): { layer: number; p: number | null }[] {
    return displayLayers.map((layer) => {
      const pairs = displayReadout?.[String(layer)];
      const hit = pairs?.find(([text]) => text === token);
      return { layer, p: hit ? hit[1] : null };
    });
  }

  const pinnedRows = $derived.by((): WorkspaceRow[] => {
    const rows: WorkspaceRow[] = [];
    for (const name of activeProbeNames()) {
      const entry = probeEntryForDisplay(name);
      if (entry?.info.family !== "lens") continue;
      const latest = (entry.aggregate ?? entry.reading) as ScalarReadingJSON | null;
      const word = entry.info.word;
      rows.push({
        key: name,
        sortName: word,
        token: word,
        strength: latest?.value ?? entry.current ?? 0,
        measured: latest !== null || entry.sparkline.length > 0,
        com: latest?.depth?.center?.[0] ?? null,
        spread: latest?.depth?.spread?.[0] ?? null,
        cells: pinnedCells(entry),
        pinned: true,
      });
    }
    return rows;
  });

  const foundRows = $derived.by((): WorkspaceRow[] => {
    const rows = displayAggregate;
    if (!rows || rows.length === 0) return [];
    // A pinned word already has its row.
    return rows
      .filter(([token]) => !pinnedRows.some((row) => row.token === token))
      .map(([token, strength, com, spread]) => ({
        key: `aggregate:${token}`,
        sortName: token.trim(),
        token,
        strength,
        measured: true,
        com,
        spread,
        cells: readoutCells(token),
        pinned: false,
      }));
  });

  const words = $derived(
    mergeInstrumentProbeRows(pinnedRows, liveOn || tokenHoverState.active ? foundRows : [], lensState.workspaceSortMode),
  );

  const SORTS: LensWorkspaceSortMode[] = ["strength", "name", "depth"];
  const nextSort = $derived(SORTS[(SORTS.indexOf(lensState.workspaceSortMode) + 1) % SORTS.length]);

  // ---------- one field: watch a word, or steer toward it ----------
  let word = $state("");
  let busy = $state(false);

  function bareWord(value: string): string {
    return value.trim().replace(/^jlens\//, "");
  }

  async function pinWord(raw: string): Promise<boolean> {
    const bare = bareWord(raw);
    if (!bare || busy) return false;
    const selector = `jlens/${bare}`;
    if (probeRack.active.includes(selector)) return true;
    busy = true;
    try {
      const validated = await apiInstruments.validateLensToken(bare);
      const validatedSelector = `jlens/${validated.word}`;
      await attachProbe(validatedSelector);
      pushToast(`pinned ${validatedSelector}`, { kind: "info" });
      return true;
    } catch (e) {
      pushToast(`Couldn't pin ${selector}: ${describeError(e)}`, { kind: "error" });
      return false;
    } finally {
      busy = false;
    }
  }

  async function watch(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    const submitted = word;
    if ((await pinWord(submitted)) && word === submitted) word = "";
  }

  async function steer(): Promise<void> {
    const submitted = word;
    const bare = bareWord(submitted);
    if (!bare || busy) return;
    busy = true;
    try {
      const validated = await apiInstruments.validateLensToken(bare);
      addJLensToRack(validated.word);
      if (word === submitted) word = "";
    } catch (e) {
      pushToast(`Couldn't steer toward jlens/${bare}: ${describeError(e)}`, { kind: "error" });
    } finally {
      busy = false;
    }
  }
</script>

<div class="jlens" aria-label="Lens">
  {#if fitted}
    <div class="dw-row">
      <span class="live" inert={lensState.busy}>
        <Switch checked={liveOn} onchange={() => void setLiveLens(!liveOn)} label="Lens readout"
          hint="Read the predicted words at every layer while the model writes. Pinned words stay when it finishes." />
      </span>
    </div>

    {#if bands.length > 0}
      <div class="bands" data-link-scope>
        {#each bands as band (band.name)}
          <div class="band" style:--tint={band.tint}>
            <span class="lnum" data-tip={`L${band.layer}`}><span class="dot" aria-hidden="true"></span>{band.name}</span>
            <div class="toks">
              {#each band.words as [text, p], i (i)}<Token token={text} value={p} {i} />{/each}
            </div>
          </div>
        {/each}
      </div>
    {/if}

    {#if tokenHoverState.active && (tokenHoverState.lensLoading || tokenHoverState.lensError || foundRows.length === 0)}
      <p class="meta" role={tokenHoverState.lensError ? "alert" : undefined}>
        {tokenHoverState.lensLoading ? "Reading the hovered token…" : tokenHoverState.lensError ?? "The lens has no reading for this token."}
      </p>
    {/if}

    {#if words.length > 0}
      <div class="rows" role="list" aria-label="J-lens probe tokens">
        {#each words as row (row.key)}
          <div role="listitem" animate:flip={{ duration: reducedMotion() ? 0 : GLIDE, easing: ease }}>
            <JLensProbeCard
              token={row.token}
              probeName={row.pinned ? row.key : undefined}
              strength={row.measured ? row.strength : null}
              com={row.com}
              spread={row.spread}
              cells={row.cells}
              pinned={row.pinned}
              {busy}
              onpin={pinWord}
            />
          </div>
        {/each}
      </div>
    {/if}

    <form class="ask" onsubmit={watch}>
      <label class="sr-only" for="lens-word">A word</label>
      <input id="lens-word" type="text" placeholder="A word" autocomplete="off" spellcheck="false" bind:value={word} />
      <button type="submit" class="dw-text" disabled={busy || !word.trim()}>Watch</button>
      <button type="button" class="dw-text" disabled={busy || !word.trim()} onclick={() => void steer()}>Steer</button>
      {#if words.length > 1}
        <button type="button" class="dw-text sort" onclick={() => setLensWorkspaceSortMode(nextSort)}
          aria-label={`Sorted by ${lensState.workspaceSortMode}. Sort by ${nextSort}`}>by <MorphText text={lensState.workspaceSortMode} /></button>
      {/if}
    </form>
  {:else}
    <div class="dw-row"><button type="button" class="dw-quiet" onclick={() => goto("lens")}>Choose a lens</button></div>
  {/if}
</div>

<style>
  .jlens { display: grid; gap: var(--s3); min-width: 0; }
  .live[inert] { opacity: 0.4; }

  .bands { display: grid; gap: var(--s2); }
  .band { display: grid; grid-template-columns: 64px minmax(0, 1fr); gap: var(--s3); align-items: center; }
  .lnum { display: flex; align-items: center; gap: 6px; color: var(--ink3); font: var(--type-meta); cursor: default; }
  .dot { width: 8px; height: 8px; border-radius: 2px; background: var(--tint); transition: background-color var(--glide) var(--ease); }

  .rows { display: grid; gap: 2px; }

  /* The field: a wash well, then quiet text actions. */
  .ask { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s4); }
  .ask input {
    flex: 1 1 160px;
    min-width: 0;
    height: 36px;
    padding: 0 var(--s4);
    border: 0;
    border-radius: var(--r-pill);
    background: var(--wash);
    color: var(--ink);
    font: var(--type-small);
    transition: background-color var(--fast) var(--ease);
  }
  .ask input::placeholder { color: var(--ink3); }
  .ask input:hover { background: var(--wash2); }
  .ask input:focus { outline: none; }
  .ask input:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
  /* On the phone's sheet (itself a wash) the well steps up to wash2, and hover steps one further. */
  :global(.sheet) .ask input { background: var(--wash2); }
  :global(.sheet) .ask input:hover { background: color-mix(in oklab, var(--wash2), var(--ink) 6%); }
  .sort { margin-left: auto; font: var(--type-meta); }

  /* Phone sizes: every target 44 px (the field, its text buttons, the switch). */
  @media (max-width: 720px), (pointer: coarse) {
    .ask .dw-text { min-height: 44px; padding: 0 var(--s1); }
    .live :global(.switch) { height: 44px; }
  }
  @media (max-width: 720px) {
    .ask input { height: 44px; font: var(--type-body); }
  }
</style>
