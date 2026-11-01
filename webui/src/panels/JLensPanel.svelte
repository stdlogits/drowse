<script lang="ts">
  // The lens, live (the Instruments canvas's Live section). A switch turns the R lens's readout
  // on; with it on, three depths show what the lens reads there (the top words at an early, a
  // middle and a late layer), then the words it reads most strongly across the fitted layers,
  // pinned ones (persistent probes, gate-able) with the rest, in one sort. One field takes a word
  // to watch (pin a probe) or to steer by, each as said (``jlens/<word>``: toward saying it, the R
  // lens) or as read (``slens/<word>``: toward having read it, the S lens), through the one pair
  // the token view shares; a found word pins the same way. A model with one lens family offers it
  // alone. While a token is hovered, everything here reads that token.
  import { flip } from "svelte/animate";
  import { GLIDE, MorphText, Segmented, Switch, Token, ease, reducedMotion } from "@stdlogits/ledger";
  import JLensProbeCard from "./rack/JLensProbeCard.svelte";
  import { foundLensRows, lensReadoutCells, mergeInstrumentProbeRows, pairLensPins } from "./rack/probeRows";
  import type { InstrumentProbeRow } from "./rack/probeRows";
  import { depthTint, layerDepth } from "../lib/charts/depth";
  import { apiInstruments, apiSessions, describeError } from "../lib/runtime/services";
  import { ApiError } from "../lib/runtime/errors";
  import {
    LENS_NAMESPACES,
    lensNamespaceInUse,
    lensNamespaceReady,
    lensPairState,
    type LensNamespace,
  } from "../lib/stores/instruments.svelte";
  import { goto } from "../lib/stores/rooms.svelte";
  import {
    addJLensToRack,
    addSLensToRack,
    activeProbeNames,
    attachProbe,
    lensState,
    lensAggregateForDisplay,
    lensReadoutForDisplay,
    probeRack,
    probeEntryForDisplay,
    setLensWorkspaceSortMode,
    setLiveLens,
    tokenHoverState,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { ProbeRackEntry, ScalarReadingJSON } from "../lib/types";
  import type { LensWorkspaceSortMode } from "../lib/stores.svelte";

  /** The R lens (its live readout, ``jlens/``) and the S lens (``slens/``), each ready or not. */
  const fitted = $derived(lensNamespaceReady("jlens"));
  const readReady = $derived(lensNamespaceReady("slens"));
  /** Both offered: the pair shows, and each pinned row says which it reads. */
  const both = $derived(fitted && readReady);
  const inUse = $derived(lensNamespaceInUse());
  const pairWord = (namespace: LensNamespace) => (both ? (namespace === "jlens" ? "said" : "read") : null);
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
    /** Which lens reads it (a found word is the R lens's read). */
    namespace: LensNamespace;
    /** The vocabulary id (a pinned probe's, when the server named one). */
    id: number | null;
  }

  /** A pinned probe's per-layer readings (the store's axis-0 map). */
  function pinnedCells(entry: ProbeRackEntry): { layer: number; p: number | null }[] {
    const perLayer = entry.perLayer;
    if (!perLayer) return [];
    return Object.keys(perLayer)
      .sort((a, b) => Number(a) - Number(b))
      .map((layer) => ({ layer: Number(layer), p: perLayer[layer] ?? null }));
  }


  const pinnedRows = $derived.by((): WorkspaceRow[] => {
    const rows: WorkspaceRow[] = [];
    for (const name of activeProbeNames()) {
      const entry = probeEntryForDisplay(name);
      if (entry?.info.family !== "lens") continue;
      const live = (entry.aggregate ?? entry.reading) as ScalarReadingJSON | null;
      // After a reload the live reading is gone: the saved reply's aggregate and its last row's
      // reading (band and depth) stand in, so a measured pin never reads "Not measured".
      const latest = live ?? ((entry.savedReading ?? null) as ScalarReadingJSON | null);
      const word = entry.info.word;
      const namespace = entry.info.namespace;
      rows.push({
        key: name,
        sortName: word,
        token: word,
        strength: live?.value ?? entry.savedAggregate ?? latest?.value ?? entry.current ?? 0,
        measured: latest !== null || entry.sparkline.length > 0 || entry.savedAggregate !== null,
        com: latest?.depth?.center?.[0] ?? null,
        spread: latest?.depth?.spread?.[0] ?? null,
        cells: pinnedCells(entry),
        pinned: true,
        namespace,
        id: typeof entry.info.token_id === "number" ? entry.info.token_id : null,
      });
    }
    return rows;
  });

  const foundRows = $derived.by((): WorkspaceRow[] => {
    const rows = displayAggregate;
    if (!rows || rows.length === 0) return [];
    // A token pinned As said already has its row; the rest key by vocabulary id (foundLensRows), and their
    // cells join the per-layer reads by id too.
    return foundLensRows(rows, pinnedRows).map((row) => ({
      ...row,
      measured: true,
      cells: lensReadoutCells(displayReadout, displayLayers, row.id),
      pinned: false,
      namespace: "jlens" as const,
    }));
  });

  const words = $derived(
    pairLensPins(mergeInstrumentProbeRows(pinnedRows, liveOn || tokenHoverState.active ? foundRows : [], lensState.workspaceSortMode)),
  );

  const SORTS: LensWorkspaceSortMode[] = ["strength", "name", "depth"];
  const nextSort = $derived(SORTS[(SORTS.indexOf(lensState.workspaceSortMode) + 1) % SORTS.length]);

  // ---------- one field: watch a word, or steer toward it, as said or as read ----------
  let word = $state("");
  let busy = $state(false);
  const pair = $derived(LENS_NAMESPACES.find((row) => row.value === inUse) ?? LENS_NAMESPACES[0]);

  function bareWord(value: string): string {
    return value.trim().replace(/^[js]lens\//, "");
  }

  /** Whether a probe on vocabulary id ``id`` is attached in ``namespace``. */
  function pinnedId(id: number, namespace: LensNamespace = inUse): boolean {
    return pinnedRows.some((row) => row.id === id && row.namespace === namespace);
  }

  async function pinWord(raw: string): Promise<boolean> {
    const bare = bareWord(raw);
    if (!bare || busy) return false;
    const namespace = inUse;
    const selector = `${namespace}/${bare}`;
    if (probeRack.active.includes(selector)) return true;
    busy = true;
    try {
      const validated = await apiInstruments.validateLensToken(bare);
      if (pinnedId(validated.token_id, namespace)) return true;
      const validatedSelector = `${namespace}/${validated.word}`;
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

  /** Pin a found token by its vocabulary id (``jlens/#<id>`` or ``slens/#<id>``, as the pair
   *  says): its text may not name it (two ids can decode to one text, as byte pieces do). */
  async function pinFound(id: number): Promise<void> {
    const namespace = inUse;
    if (busy || pinnedId(id, namespace)) return;
    busy = true;
    const selector = `${namespace}/#${id}`;
    try {
      await attachProbe(selector);
      pushToast(`pinned ${selector}`, { kind: "info" });
    } catch (e) {
      pushToast(`Couldn't pin ${selector}: ${describeError(e)}`, { kind: "error" });
    } finally {
      busy = false;
    }
  }

  async function watch(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    const submitted = word;
    if ((await pinWord(submitted)) && word === submitted) word = "";
  }

  /** Steer by the word, as the pair says: toward saying it (``jlens/``) or toward having read it
   *  (``slens/``). The atom is checked as composition would resolve it; a busy model cannot check
   *  it now, so it is added and the reply (or its chip's editor) says if it does not resolve. */
  async function steer(): Promise<void> {
    const submitted = word;
    const bare = bareWord(submitted);
    const namespace = inUse;
    if (!bare || busy) return;
    busy = true;
    try {
      const validated = await apiInstruments.validateLensToken(bare);
      const atom = `${namespace}/${validated.word}`;
      try {
        const check = await apiSessions.validateSteering(`0.3 ${atom}@response`);
        if (!check.valid) throw new Error(check.error ?? `${atom} does not resolve`);
      } catch (e) {
        if (!(e instanceof ApiError && e.status === 409)) throw e;
      }
      if (namespace === "jlens") addJLensToRack(validated.word);
      else addSLensToRack(validated.word);
      if (word === submitted) word = "";
    } catch (e) {
      pushToast(`Couldn't steer by ${namespace}/${bare}: ${describeError(e)}`, { kind: "error" });
    } finally {
      busy = false;
    }
  }
</script>

<div class="jlens" aria-label="Lens">
  {#if fitted || readReady}
    {#if fitted}
      <div class="dw-row">
        <span class="live" inert={lensState.busy}>
          <Switch checked={liveOn} onchange={() => void setLiveLens(!liveOn)} label="Lens readout"
            hint="Read the predicted words at every layer while the model writes. Pinned words stay when it finishes." />
        </span>
      </div>
    {/if}

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

    {#if fitted && tokenHoverState.active && (tokenHoverState.lensLoading || tokenHoverState.lensError || foundRows.length === 0)}
      <p class="meta" role={tokenHoverState.lensError ? "alert" : undefined}>
        {tokenHoverState.lensLoading ? "Reading the hovered token…" : tokenHoverState.lensError ?? "The lens has no reading for this token."}
      </p>
    {/if}

    {#if words.length > 0}
      <div class="rows" role="list" aria-label="Lens probe tokens">
        {#each words as row (row.key)}
          <div role="listitem" animate:flip={{ duration: reducedMotion() ? 0 : GLIDE, easing: ease }}>
            <JLensProbeCard
              token={row.token}
              probeName={row.pinned ? row.key : undefined}
              namespace={row.namespace}
              pinAs={inUse}
              pairWord={row.pinned ? pairWord(row.namespace) : null}
              strength={row.measured ? row.strength : null}
              com={row.com}
              spread={row.spread}
              cells={row.cells}
              pinned={row.pinned}
              {busy}
              onpin={row.pinned || row.id === null ? pinWord : () => void pinFound(row.id as number)}
            />
          </div>
        {/each}
      </div>
    {/if}

    <form class="ask" onsubmit={watch}>
      <label class="sr-only" for="lens-word">A word</label>
      <input id="lens-word" type="text" placeholder="A word" autocomplete="off" spellcheck="false" bind:value={word} />
      {#if both}
        <Segmented label="Read the word" value={inUse} options={[...LENS_NAMESPACES]}
          onchange={(value: LensNamespace) => (lensPairState.namespace = value)} />
      {/if}
      <button type="submit" class="dw-text" disabled={busy || !word.trim()} data-tip={`Pin a probe on it: ${pair.hint}`}>Watch</button>
      <button type="button" class="dw-text" disabled={busy || !word.trim()} data-tip={pair.hint} onclick={() => void steer()}>Steer</button>
      {#if words.length > 1}
        <button type="button" class="dw-text sort" onclick={() => setLensWorkspaceSortMode(nextSort)}
          aria-label={`Sorted by ${lensState.workspaceSortMode}. Sort by ${nextSort}`}>by <MorphText text={lensState.workspaceSortMode} /></button>
      {/if}
    </form>
  {:else}
    <div class="dw-row"><button type="button" class="dw-quiet" onclick={() => goto("artifacts", "lenses")}>Choose a lens</button></div>
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
