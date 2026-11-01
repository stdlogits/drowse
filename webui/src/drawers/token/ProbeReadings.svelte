<script lang="ts">
  // The probes at this token, as rows: the probe, its trace over depth (each layer in its own depth
  // colour; a negative is ink at low alpha), and its reading. Geometry probes read the whitened
  // Monitor (a flat probe its signed coordinate, a curved one its nearest node and subspace share);
  // the lens and SAE probes pinned while the token was generated read their one channel. A row
  // opens to the rest: every axis, the share, the nearest nodes and assignment, the residual and
  // tube membership, and any layer's own coordinates.
  import { openDrawer, probeAxisScale, probeRack } from "../../lib/stores.svelte";
  import { drawerAvailability } from "../../lib/runtime/ui-capabilities";
  import type { ProbeReadingJSON, ScalarReadingJSON } from "../../lib/types";
  import type { GeometryTokenReadout, ReplayReadout } from "./readout.svelte";
  import ReadingWait from "./ReadingWait.svelte";
  import { depthTint, formatSigned, readingText } from "./layers";
  import { layersNote, probeNote } from "./readings";

  let { readout, pinned, hasGeometryProbes, hasReplayContext, replayAvailable, returnToToken }: {
    readout: ReplayReadout<GeometryTokenReadout>;
    /** The lens and SAE probes recorded with the token (captured envelopes only). */
    pinned: Record<string, ScalarReadingJSON>;
    hasGeometryProbes: boolean;
    hasReplayContext: boolean;
    replayAvailable: boolean;
    returnToToken: { turnIdx: number; tokenIdx: number; isThinking: boolean };
  } = $props();

  interface Bar { layer: number; value: number | null }
  interface Row {
    name: string;
    kind: "flat" | "curved" | "geometry" | "pinned";
    value: string;
    note: string;
    bars: Bar[];
    scale: number;
    geometry: ProbeReadingJSON | null;
    scalar: ScalarReadingJSON | null;
  }

  const authoring = drawerAvailability("manifold_builder");
  const byName = (a: string, b: string) => a.localeCompare(b, undefined, { sensitivity: "base" });

  function affineOf(name: string): boolean | null {
    const info = probeRack.entries.get(name)?.info;
    return info?.family === "geometry" ? info.is_affine : null;
  }

  /** Axis 0 on a two-node concept is pole-normalized (+1 at the first node), so it takes that name. */
  function axisLabel(name: string, axis: number, rank: number): string {
    const info = probeRack.entries.get(name)?.info;
    const labels = info?.family === "geometry" ? info.node_labels : undefined;
    return rank === 1 && axis === 0 && labels?.length === 2 ? labels[0] : `axis ${axis + 1}`;
  }

  function series(source: Record<string, number | null | undefined>): Bar[] {
    return Object.keys(source)
      .filter(layer => Number.isSafeInteger(Number(layer)))
      .sort((a, b) => Number(a) - Number(b))
      .map(layer => ({ layer: Number(layer), value: source[layer] ?? null }));
  }

  const rows = $derived.by<Row[]>(() => {
    const out: Row[] = [];
    for (const [name, reading] of Object.entries(readout.data?.readings ?? {}).sort(([a], [b]) => byName(a, b))) {
      const affine = affineOf(name);
      const curved = affine === false;
      const bars = curved
        ? series(reading.fraction_per_layer ?? {})
        : series(Object.fromEntries(Object.entries(reading.coords_per_layer ?? {}).map(([layer, c]) => [layer, c[0] ?? null])));
      out.push({
        name,
        kind: affine === null ? "geometry" : curved ? "curved" : "flat",
        value: curved ? reading.fraction.toFixed(2) : formatSigned(reading.coords[0] ?? 0),
        note: probeNote(name, curved ? (reading.nearest?.[0]?.[0] ?? "") : reading.coords.length === 1 ? axisLabel(name, 0, 1) : ""),
        bars,
        scale: curved ? 1 : probeAxisScale(name, 0),
        geometry: reading,
        scalar: null,
      });
    }
    for (const [name, reading] of Object.entries(pinned).sort(([a], [b]) => byName(a, b))) {
      const bars = series(reading.per_layer ?? {});
      out.push({
        name,
        kind: "pinned",
        value: readingText(reading.value, reading.unit),
        note: "",
        bars,
        scale: Math.max(1e-9, ...bars.map(bar => Math.abs(bar.value ?? 0))),
        geometry: null,
        scalar: reading,
      });
    }
    return out;
  });
  const deepest = $derived(Math.max(0, ...rows.flatMap(row => row.bars.map(bar => bar.layer))));
  /** A saved reply's capture shows where nothing replays it: each reading's summary, no layers. */
  const layersNotSaved = $derived(
    readout.origin === "captured" && rows.some(row => row.geometry !== null && row.geometry.coords_per_layer === undefined),
  );

  let open = $state<string | null>(null);
  let pickedLayer = $state<number | null>(null);
  $effect(() => { void readout.data; open = null; pickedLayer = null; });

  function toggle(name: string): void {
    open = open === name ? null : name;
    pickedLayer = null;
  }

  function setupProbe(create = false): void {
    openDrawer(create ? "manifold_builder" : "subspace", { returnToToken, ...(create ? { mode: "discover" } : {}) });
  }

  const UNIT_NAMES: Record<string, string> = {
    mean_token_probability: "mean probability",
    raw_activation: "activation",
    relative_activation: "relative activation",
  };
</script>

{#snippet trace(bars: Bar[], scale: number, picked: number | null, onpick: ((layer: number) => void) | null)}
  <span class="trace" class:pickable={onpick !== null} aria-hidden={onpick === null}>
    {#each bars as bar (bar.layer)}
      {@const height = bar.value === null ? 0 : Math.min(100, (Math.abs(bar.value) / (scale || 1)) * 100)}
      {#if onpick}
        <button type="button" class="bar" class:neg={(bar.value ?? 0) < 0} class:picked={picked === bar.layer}
          style:--tint={depthTint(bar.layer, deepest)} aria-label={`Layer ${bar.layer}`} aria-pressed={picked === bar.layer}
          data-tip={`L${bar.layer} · ${bar.value === null ? "not read" : formatSigned(bar.value, 3)}`} onclick={() => onpick(bar.layer)}><i style:height="{Math.max(height, 4)}%"></i></button>
      {:else}
        <i class="bar" class:neg={(bar.value ?? 0) < 0} style:--tint={depthTint(bar.layer, deepest)} style:height="{Math.max(height, 4)}%"></i>
      {/if}
    {/each}
  </span>
{/snippet}

{#snippet signed(value: number, scale: number)}
  {@const share = Math.min(1, Math.abs(value) / (scale || 1)) * 50}
  <span class="signed" aria-hidden="true"><i class:neg={value < 0} style:left="{value < 0 ? 50 - share : 50}%" style:width="{share}%"></i></span>
{/snippet}

{#snippet detail(row: Row)}
  {#if row.geometry}
    {@const reading = row.geometry}
    {@const rank = reading.coords.length}
    {@const layerCoords = pickedLayer !== null ? reading.coords_per_layer?.[String(pickedLayer)] ?? null : null}
    <div class="detail">
      {#each layerCoords ?? reading.coords as coord, axis (axis)}
        <div class="kv"><span>{axisLabel(row.name, axis, rank)}</span>{@render signed(coord, probeAxisScale(row.name, axis))}<span class="v">{formatSigned(coord, 3)}</span></div>
      {/each}
      {#if pickedLayer !== null}
        {@const fraction = reading.fraction_per_layer?.[String(pickedLayer)]}
        {#if fraction != null}<div class="kv"><span data-tip="The share of the centred activation in this subspace, 0 to 1. Not confidence.">share at L{pickedLayer}</span><span class="share"><i style:width="{Math.min(1, fraction) * 100}%"></i></span><span class="v">{fraction.toFixed(3)}</span></div>{/if}
      {:else}
        <div class="kv"><span data-tip="The share of the centred activation in this subspace, 0 to 1. A larger subspace can hold more; this is not confidence.">share</span><span class="share"><i style:width="{Math.min(1, reading.fraction) * 100}%"></i></span><span class="v">{reading.fraction.toFixed(3)}</span></div>
      {/if}
      {#if row.bars.length > 1}
        <div class="layerpick">
          {@render trace(row.bars, row.scale, pickedLayer, row.kind === "flat" || row.kind === "geometry" ? (layer => (pickedLayer = pickedLayer === layer ? null : layer)) : null)}
          <span class="meta">{pickedLayer !== null ? `L${pickedLayer}` : reading.depth_com?.[0] != null ? `centred at ${reading.depth_com[0].toFixed(2)} ±${(reading.depth_spread?.[0] ?? 0).toFixed(2)}` : ""}</span>
        </div>
      {/if}
      {#if (reading.nearest ?? []).length > 0 || (reading.assignment ?? []).length > 0}
        <div class="toks">
          {#each reading.nearest ?? [] as [label, dist] (label)}
            <span class="chip" data-tip={`${dist.toFixed(3)} typical node spacings away; smaller is closer`}>{label}<span class="n">{dist.toFixed(2)}</span></span>
          {/each}
          {#each reading.assignment ?? [] as [label, prob] (label)}
            <span class="chip soft" data-tip={`Relative geometric fit among the nodes (${(prob * 100).toFixed(1)}%), not the probability the trait is true`}>~{label}<span class="n">{(prob * 100).toFixed(0)}%</span></span>
          {/each}
        </div>
      {/if}
      {#if row.kind === "curved" || reading.residual !== 0 || (row.kind !== "flat" && reading.membership != null)}
        <div class="facts meta">
          {#if row.kind === "curved" || reading.residual !== 0}<span data-tip="Distance off the fitted surface over the in-subspace norm; smaller is closer">residual <b>{reading.residual.toFixed(3)}</b></span>{/if}
          {#if row.kind !== "flat" && reading.membership != null}<span data-tip="Fit inside the learned tube, 0 to 1 (1 without a learned tube); not confidence">membership <b>{reading.membership.toFixed(3)}</b></span>{/if}
        </div>
      {/if}
    </div>
  {:else if row.scalar}
    {@const reading = row.scalar}
    <div class="detail">
      <div class="facts meta">
        <span>{UNIT_NAMES[reading.unit] ?? reading.unit.replace(/_/g, " ")} <b>{reading.value.toFixed(3)}</b></span>
        {#if reading.depth?.center?.[0] != null}<span data-tip={`Depth summary over ${reading.depth.basis.replace(/_/g, " ")}; 0 is the first layer, 1 the last`}>centred at <b>{reading.depth.center[0].toFixed(2)}</b> ±{(reading.depth.spread?.[0] ?? 0).toFixed(2)}</span>{/if}
      </div>
    </div>
  {/if}
{/snippet}

<section class="dw-sec" aria-label="Probes">
  <div class="dw-sec-head">
    <span class="head">Probes</span>
    <span class="meta">{#if readout.origin === "replayed"}replayed{:else if !replayAvailable && hasGeometryProbes && !readout.data}<span data-tip="Geometry recorded during generation still shows">not replayable here</span>{/if}</span>
  </div>
  {#if readout.loading}
    <ReadingWait progress={readout.progress} label="Probe readings" blocks={[44, 44]} />
  {:else if readout.error}
    <p class="meta" role="alert">{readout.error}</p>
  {/if}
  {#if layersNotSaved}
    <p class="meta">A saved reply keeps each probe's reading, not its layers. {layersNote(replayAvailable, hasReplayContext, "the probe is attached")}</p>
  {/if}
  {#if rows.length > 0}
    <div class="rows" role="list">
      {#each rows as row, i (row.name)}
        <div role="listitem" class="item rise" style:--i={i}>
          <button type="button" class="row" aria-expanded={open === row.name} onclick={() => toggle(row.name)}>
            <span class="name"><span class="n1">{#each row.name.split(".") as part, k (k)}{#if k > 0}.<wbr>{/if}{part}{/each}</span>{#if row.note}<span class="meta">{row.note}</span>{/if}</span>
            {@render trace(row.bars, row.scale, null, null)}
            <span class="val">{row.value}</span>
          </button>
          {#if open === row.name}{@render detail(row)}{/if}
        </div>
      {/each}
    </div>
  {:else if !readout.loading && !hasGeometryProbes}
    <div class="dw-row">
      <button type="button" class="dw-quiet" onclick={() => setupProbe()}>Add a probe</button>
      {#if authoring.available}<button type="button" class="dw-quiet" onclick={() => setupProbe(true)}>Create a concept</button>{/if}
    </div>
  {:else if !readout.loading && !readout.error && replayAvailable && !hasReplayContext}
    <p class="meta">No replay record for this token</p>
  {/if}
</section>

<style>
  .rows { display: grid; gap: 2px; }
  .item { display: grid; min-width: 0; }
  .row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 120px 48px;
    gap: var(--s4);
    align-items: center;
    width: 100%;
    min-height: 44px;
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
  @media (hover: hover) { .row:hover { background: var(--token-fill); } }
  .row[aria-expanded="true"] { background: var(--token-fill); }
  /* Full width against the Inspector's scroll edge: the ring sits inside. */
  .row:focus-visible { outline-offset: -2px; }
  .row:active { scale: 0.99; }
  /* The name is the row's identity: it keeps its width and wraps at its pole separator when it
   * must, while the note beside it gives way first. */
  .name { display: flex; align-items: baseline; gap: var(--s2); min-width: 0; }
  .n1 { flex: 0 1 auto; min-width: 0; color: var(--ink); font: var(--type-control); overflow-wrap: anywhere; }
  .name .meta { flex: 0 1000 auto; min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .val { color: var(--ink2); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; }

  .trace { display: flex; align-items: flex-end; gap: 1px; height: 24px; min-width: 0; }
  .trace .bar { flex: 1 1 0; min-width: 0; min-height: 2px; border-radius: 1px; background: var(--tint); }
  .trace .bar.neg { background: var(--ink3); opacity: 0.45; }
  .trace.pickable { height: 40px; }
  .trace button.bar { display: flex; align-items: flex-end; height: 100%; padding: 0; border: 0; background: transparent; cursor: pointer; }
  .trace button.bar > i { display: block; width: 100%; border-radius: 1px; background: var(--tint); transition: opacity var(--fast) var(--ease); }
  .trace button.bar.neg > i { background: var(--ink3); opacity: 0.45; }
  .trace.pickable:has(.picked) button.bar:not(.picked) > i { opacity: 0.35; }

  .detail { display: grid; gap: var(--s2); padding: var(--s3) var(--s3) var(--s4); }
  .kv { display: grid; grid-template-columns: minmax(0, 1fr) 120px 56px; gap: var(--s4); align-items: center; color: var(--ink2); font: var(--type-small); }
  .kv > span:first-child { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .kv [data-tip] { cursor: help; }
  .v { color: var(--ink); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; }
  .signed, .share { position: relative; height: 4px; border-radius: 2px; background: var(--token-fill); overflow: hidden; }
  .signed > i, .share > i { position: absolute; top: 0; bottom: 0; border-radius: 2px; background: var(--tint); transition: left var(--glide) var(--ease), width var(--glide) var(--ease); }
  .share > i { left: 0; }
  .signed > i.neg { background: var(--ink3); opacity: 0.45; }
  .layerpick { display: grid; gap: var(--s1); padding-top: var(--s2); }
  .chip { display: inline-flex; align-items: baseline; gap: 6px; padding: 2px var(--s2); border-radius: var(--r-chip); background: var(--token-fill); color: var(--ink); font: var(--type-small); cursor: help; }
  .chip.soft { color: var(--ink2); }
  .chip .n { color: var(--ink3); font: var(--type-meta); font-variant-numeric: tabular-nums; }
  .facts { display: flex; flex-wrap: wrap; gap: var(--s1) var(--s4); }
  .facts b { color: var(--ink); font-weight: 500; }
  /* A narrow view (the Inspector beside a mid-width page, or the phone) gives the names the room. */
  @container token (max-width: 440px) {
    .row { grid-template-columns: minmax(0, 1fr) 88px 44px; gap: var(--s3); }
    .kv { grid-template-columns: minmax(0, 1fr) 88px 56px; gap: var(--s3); }
  }
  @media (max-width: 720px) {
    .trace button.bar { min-height: 44px; }
  }
</style>
