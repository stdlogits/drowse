<script lang="ts">
  import { MorphText } from "@stdlogits/ledger";
  // A selected probe, in the Inspector (``openDrawer("probe_inspector", { name })``; the crumbs
  // carry the close). Its name and facts; its reading token by token (a scrubbable trace) and layer
  // by layer (a bar per layer in that layer's colour); for a geometry probe, where it sits (the
  // subspace share, the nearest node, one signed meter per coordinate, poles on a concept axis),
  // and its geometry in the whitened (Mahalanobis) frame (ProbeGeometryPlot, shared with the Probes
  // room). The live point and its trail ride the probe's per-token ``subspace_coords_per_layer``
  // from the live token stream only: a loom row keeps each reading's summary, so a saved reply draws
  // no trail, and its "By layer" view replays the token. Everything draws at the probe's depth on
  // the ramp (--tint), which only fills.

  import {
    detachProbe,
    highlightState,
    instrumentFamily,
    probeRack,
    setHighlightTarget,
  } from "../lib/stores.svelte";
  import { clearSelection } from "../lib/stores/selection.svelte";
  import { effectiveRawMode } from "../lib/stores/chat.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import { cachedTokenReadout } from "../lib/runtime/tokenReadoutCache";
  import { readingLayers } from "../lib/tokenRows";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import Bar from "../lib/charts/Bar.svelte";
  import Trace from "../lib/charts/Trace.svelte";
  import TraceDetail from "../lib/charts/TraceDetail.svelte";
  import { depthTint, layerDepth, peakLayer, readingText } from "../lib/charts/depth";
  import { nodeCoordExtent, parseProbeTarget } from "../lib/tokens";
  import { polesOf } from "../lib/concepts";
  import { axisName } from "./token/readings";
  import ManifoldMiniMap from "../panels/manifold/ManifoldMiniMap.svelte";
  import ProbeGeometryPlot from "../panels/manifold/ProbeGeometryPlot.svelte";
  import { probeTitle } from "../panels/rack/probeRows";
  import { readsShare, replyLayers, replyToken, replyTrace } from "../panels/rack/probeTrace";
  import type {
    GeometryProbeInfo,
    ProbeGeometryResponse,
    ProbeLayerGeometry,
    ProbeReadingJSON,
    ScalarReadingJSON,
  } from "../lib/types";

  let { params: input }: { params?: unknown; onclose?: () => void } = $props();
  const params = $derived(input as { name?: string } | null);
  const probeName = $derived(params?.name ?? "");
  const entry = $derived(probeRack.entries.get(probeName) ?? null);
  const info = $derived(entry?.info ?? null);
  /** The resident SAE's hook layer (the layer a feature's ID carries, as its row writes it). */
  const saeLayer = $derived.by((): number | null => {
    const live = instrumentFamily("sae")?.live;
    return live && "layer" in live ? live.layer : null;
  });
  /** The probe's ID as its row writes it: ``L20 f7``, a lens word, a geometry probe's name. */
  const title = $derived(probeTitle(info, probeName, saeLayer));
  const displayName = $derived(title.text);
  const geometric = $derived(info?.family === "geometry");
  const geo = $derived(geometric ? (info as GeometryProbeInfo) : null);

  /** The fitted geometry and the plotted layer, as the plot (ProbeGeometryPlot) loaded and chose them. */
  let geom = $state<ProbeGeometryResponse | null>(null);
  let selectedLayer = $state<number | null>(null);

  // ---------- the reading ----------
  const latest = $derived((entry?.aggregate ?? entry?.reading ?? null) as ProbeReadingJSON | ScalarReadingJSON | null);
  /** The live reading, else the saved reply's (its last measured row: share, coordinates, nearest,
   *  residual and depth, without layers), so a reload shows what the live view showed but the trail. */
  const settled = $derived((latest ?? entry?.savedReading ?? null) as ProbeReadingJSON | ScalarReadingJSON | null);
  /** The reply's settled reading: its aggregate live, else the saved one. */
  const replySettled = $derived((entry?.aggregate ?? entry?.savedReading ?? null) as ProbeReadingJSON | null);
  const geoReading = $derived(geometric ? (settled as ProbeReadingJSON | null) : null);
  const scalarReading = $derived(geometric ? null : (settled as ScalarReadingJSON | null));
  const sparkline = $derived(entry?.sparkline ?? []);
  const affine = $derived(geo?.is_affine ?? false);
  /** A curved probe reads the share on its surface: its trace, its layers and "reads" all follow it. */
  const share = $derived(readsShare(info));
  /** The latest reply token by token, else the live buffer (the same primary reading). */
  const tokenTrace = $derived((probeName ? replyTrace(probeName, { share }) : null) ?? sparkline);
  /** The token the By-token scrubber rests on (null: the latest). */
  let pickedToken = $state<number | null>(null);
  /** A reply token's layers from a replay: a saved reply keeps a geometry probe's summary, not its
   *  per-layer trace, so its layers replay the token (steered, as it was generated). */
  let tokenLayers = $state<{ key: string | null; layers: Record<string, number> | null; replayed: boolean; note: string | null; failed: boolean }>({
    key: null, layers: null, replayed: false, note: null, failed: false,
  });
  const geometryReplay = $derived(instrumentFamily("geometry")?.capabilities.token_readout === true);
  const liveLayers = $derived(entry?.perLayer ?? {});
  /** The token whose layers show: the picked one, or the latest when nothing live has layers. */
  const layerToken = $derived(
    probeName && (pickedToken !== null || Object.keys(liveLayers).length === 0) ? replyToken(pickedToken) : null,
  );
  const savedLayers = $derived(layerToken && probeName ? replyLayers(probeName, pickedToken, { share }) : null);
  $effect(() => {
    const picked = layerToken;
    const name = probeName;
    if (!picked || !name || savedLayers !== null || !geometric) {
      tokenLayers = { key: null, layers: null, replayed: false, note: null, failed: false };
      return;
    }
    const { nodeId, rawIndex } = picked;
    if (!geometryReplay || nodeId === null || rawIndex === null) {
      tokenLayers = {
        key: null, layers: null, replayed: false, failed: false,
        note: "A saved reply keeps each probe's reading, not its layers. They replay while this model is loaded and the probe is attached.",
      };
      return;
    }
    const key = `${nodeId}:${rawIndex}:${name}:${share}`;
    if (tokenLayers.key === key) return;
    tokenLayers = { key, layers: null, replayed: false, note: "Replaying this token's layers…", failed: false };
    cachedTokenReadout("geometry", nodeId, rawIndex, { steered: true, raw: effectiveRawMode() }).then(
      (res) => {
        if (tokenLayers.key !== key) return;
        const layers = readingLayers(res.measurements.instruments.geometry?.readings?.[name], share);
        tokenLayers = layers && Object.keys(layers).length > 0
          ? { key, layers, replayed: true, note: null, failed: false }
          : { key, layers: null, replayed: false, note: "The replay read no layers of this probe. Check that it is attached.", failed: true };
      },
      (error: unknown) => {
        if (tokenLayers.key !== key) return;
        tokenLayers = { key, layers: null, replayed: false, failed: true, note: userFacingError(error, "This token's layers could not be replayed. Try again, or close and reopen the model.") };
      },
    );
  });
  /** By layer: the picked token's per-layer reading (its own, else replayed), else the live latest. */
  const perLayer = $derived.by((): Record<string, number> => {
    if (layerToken === null) return liveLayers;
    return savedLayers ?? tokenLayers.layers ?? (pickedToken === null ? liveLayers : {});
  });
  const layerKeys = $derived(Object.keys(perLayer).map(Number).filter(Number.isFinite).sort((a, b) => a - b));
  const deepest = $derived(Math.max(0, ...(geo?.layers ?? []), ...layerKeys));
  const peak = $derived(peakLayer(perLayer));
  const depthCom = $derived(geoReading?.depth_com?.[0] ?? scalarReading?.depth?.center?.[0] ?? null);
  const depthSpread = $derived(geoReading?.depth_spread?.[0] ?? scalarReading?.depth?.spread?.[0] ?? null);
  const isHighlight = $derived(
    highlightState.target !== null && parseProbeTarget(highlightState.target).base === probeName,
  );
  /** A word as said reads as a probability; as read, as a footprint z. */
  const percent = $derived(info?.family === "lens" && info.namespace === "jlens");
  const value = $derived.by((): number | null => {
    if (!entry) return null;
    if (latest !== null || sparkline.length > 0) return entry.current;
    return entry.savedAggregate ?? null;
  });

  // Geometry: the subspace share, the nearest node, one signed meter per coordinate.
  const fraction = $derived(geoReading?.fraction ?? entry?.savedFraction ?? null);
  const coordVec = $derived(
    geoReading?.coords ?? entry?.savedCoordinates ?? (affine && entry?.savedAggregate != null ? [entry.savedAggregate] : []),
  );
  const nDim = $derived(geo ? (geo.intrinsic_dim > 0 ? geo.intrinsic_dim : coordVec.length) : 0);
  const axes = $derived(
    Array.from({ length: nDim }, (_, i) => ({ i, value: coordVec[i] ?? 0, scale: nodeCoordExtent(geo?.node_coords, i) })),
  );
  const poles = $derived(polesOf(geo?.manifold || probeName));
  const showPoles = $derived(affine && nDim === 1 && (geo?.node_count ?? 0) <= 2);
  const nearest = $derived(entry?.nearest?.[0] ?? null);
  const residual = $derived(replySettled?.residual ?? null);
  const showMiniMap = $derived.by(() => {
    if (!geo || geo.intrinsic_dim !== 2) return false;
    const d = geo.domain as { type?: string };
    return d?.type === "box" && !!geo.node_coords && geo.node_coords.length > 0;
  });
  const layerScale = $derived(affine ? nodeCoordExtent(geo?.node_coords, 0) : percent ? Math.max(...Object.values(perLayer), 1e-12) : 1);

  function toggleHighlight(): void {
    setHighlightTarget(isHighlight ? null : probeName);
  }

  let removing = $state(false);
  async function remove(): Promise<void> {
    if (removing || !probeName) return;
    removing = true;
    const name = probeName;
    try {
      await detachProbe(name);
      pushToast(`${geometric ? "detached" : "unpinned"} ${name}`, { kind: "info" });
      clearSelection();
    } catch (e) {
      pushToast(userFacingError(e, `Unable to remove ${name}. Try again.`), { kind: "error", ttlMs: null });
    } finally {
      removing = false;
    }
  }


  // sorted layer list (ascending) for the share bars
  const layerList = $derived.by<ProbeLayerGeometry[]>(() =>
    geom ? Object.values(geom.layers).sort((a, b) => a.layer - b.layer) : [],
  );
  const activeGeom = $derived(
    geom && selectedLayer !== null
      ? (geom.layers[String(selectedLayer)] ?? null)
      : null,
  );
  const geomDeepest = $derived(Math.max(deepest, ...layerList.map((l) => l.layer)));
  const bestShareLayer = $derived(
    layerList.reduce<ProbeLayerGeometry | null>((best, l) => (best === null || Math.abs(l.mahalanobis_share) > Math.abs(best.mahalanobis_share) ? l : best), null)?.layer ?? null,
  );
  /** The probe's place on the ramp: its measured depth, else the layer its fit carries most. */
  const tint = $derived(depthTint(depthCom ?? (bestShareLayer === null ? null : layerDepth(bestShareLayer, geomDeepest))));

  const rankLabel = $derived.by(() => {
    const r = activeGeom?.rank ?? geo?.intrinsic_dim ?? 0;
    if (r <= 1) return "a line";
    if (r === 2) return "a plane";
    return `rank ${r}`;
  });


</script>

<div class="probe" style:--tint={tint ?? undefined} aria-label="Probe inspector">
  {#if !probeName || !entry}
    <p class="meta">This probe is no longer attached.</p>
  {:else}
    <div class="dw-sec">
      <h2 class="dw-title" data-tip={probeName}>{#if title.lead}<span class="dw-ws">·</span>{/if}{displayName}</h2>
      <div class="dw-facts">
        {#if value !== null}<span>reads <b><MorphText text={readingText(value, { signed: affine, percent })} /></b></span>{/if}
        {#if geo}<span>{affine ? "flat" : "curved"}, {rankLabel}</span>{/if}
        {#if depthCom !== null}<span>depth <b>{depthCom.toFixed(2)}</b>{depthSpread === null ? "" : ` ± ${depthSpread.toFixed(2)}`}</span>{/if}
        {#if !affine && residual !== null && Number.isFinite(residual)}<span>residual <b>{residual.toFixed(2)}</b></span>{/if}
        {#if geom && !geom.rank_uniform}<span>rank varies by layer</span>{/if}
      </div>
      <div class="dw-row">
        <button type="button" class="dw-quiet" aria-pressed={isHighlight} onclick={toggleHighlight}>{isHighlight ? "Tinting the reply" : "Tint the reply"}</button>
        <button type="button" class="dw-quiet" disabled={removing} onclick={() => void remove()}>{geometric ? "Detach" : "Unpin"}</button>
      </div>
    </div>

    <section class="dw-sec">
      <div class="dw-sec-head"><span class="head">By token</span></div>
      {#if tokenTrace.length > 1}
        <TraceDetail points={tokenTrace} percentage={percent} signed={affine} onpick={(i) => (pickedToken = i)} />
      {:else}
        <p class="meta">No reading yet.</p>
      {/if}
    </section>

    {#if layerKeys.length > 1}
      <section class="dw-sec">
        <div class="dw-sec-head"><span class="head">By layer</span><span class="meta">{pickedToken === null ? "" : `token ${pickedToken + 1} · `}{tokenLayers.replayed ? "replayed · " : ""}{peak === null ? "" : `peaks at L${peak}`}</span></div>
        <div class="depth">
          <Trace values={layerKeys.map((l) => perLayer[String(l)])} limit={0} height={56} scale={layerScale}
            colors={layerKeys.map((l) => depthTint(layerDepth(l, deepest)))} cursor={peak === null ? null : layerKeys.indexOf(peak)}
            label={`${displayName} by layer`} />
          <div class="ends"><span>L{layerKeys[0]}</span><span>L{layerKeys[layerKeys.length - 1]}</span></div>
        </div>
      </section>
    {:else if tokenLayers.note}
      <section class="dw-sec">
        <div class="dw-sec-head"><span class="head">By layer</span>{#if pickedToken !== null}<span class="meta">token {pickedToken + 1}</span>{/if}</div>
        <p class="meta" role={tokenLayers.failed ? "alert" : undefined}>{tokenLayers.note}</p>
      </section>
    {/if}

    {#if geo}
      <section class="dw-sec">
        <div class="dw-sec-head"><span class="head">Where it sits</span>{#if nearest}<span class="meta">nearest {nearest[0]}{#if Number.isFinite(nearest[1])}, {nearest[1].toFixed(2)} away{/if}</span>{/if}</div>
        <div class="coords">
          {#if fraction !== null}
            <div class="coord" data-tip="The share of the centred activation in this subspace (0 to 1, not a confidence)">
              <span class="cl">in subspace</span><Bar value={fraction} max={1} width={200} height={4} percentage /><span class="cv">{fraction.toFixed(2)}</span>
            </div>
          {/if}
          {#each axes as ax (ax.i)}
            <div class="coord">
              <span class="cl">{showPoles ? (poles.negative ?? "") : axisName(geo.node_labels, ax.i, nDim)}</span>
              <Bar value={ax.value} max={ax.scale} width={200} height={4} bipolar />
              <span class="cv">{readingText(ax.value, { signed: true })}</span>
              {#if showPoles}<span class="pole">{poles.positive}</span>{/if}
            </div>
          {/each}
        </div>
        {#if showMiniMap}
          <div class="map"><ManifoldMiniMap info={geo} trajectory={entry.trajectory ?? []} settled={replySettled?.coords ?? null} /></div>
        {/if}
      </section>

      <ProbeGeometryPlot name={probeName} label={displayName} trail={entry.subspaceTrail ?? []} {deepest} controller
        bind:geom bind:layer={selectedLayer} />
    {/if}
  {/if}
</div>

<style>
  .probe { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s6); min-width: 0; }

  /* A reading over depth: one flat bar per layer in its own depth colour. */
  .depth { display: grid; gap: var(--s1); }
  .ends { display: flex; justify-content: space-between; color: var(--ink3); font: var(--type-meta); font-variant-numeric: tabular-nums; }

  /* Signed meters, one per coordinate: a label, the meter, the value (and a concept's positive pole). */
  .coords { display: grid; gap: var(--s2); }
  .coord { display: grid; grid-template-columns: 88px minmax(0, 1fr) 48px; gap: var(--s3); align-items: center; min-height: 28px; }
  .coord:has(.pole) { grid-template-columns: 88px minmax(0, 1fr) 48px 88px; }
  .coord :global(.bar) { width: 100%; }
  .cl { overflow: hidden; color: var(--ink2); font: var(--type-small); text-align: right; white-space: nowrap; text-overflow: ellipsis; }
  .cv { color: var(--ink); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; }
  .pole { overflow: hidden; color: var(--ink); font: var(--type-small); white-space: nowrap; text-overflow: ellipsis; }
  .map { display: flex; justify-content: center; padding: var(--s4); border-radius: var(--r-box); background: var(--wash); }
  /* On the phone's sheet (itself a wash) the map steps up to wash2. */
  :global(.sheet) .map { background: var(--wash2); }

  @media (max-width: 720px) {
    .coord { grid-template-columns: 64px minmax(0, 1fr) 44px; }
    .coord:has(.pole) { grid-template-columns: 64px minmax(0, 1fr) 44px 64px; }
  }
</style>
