<script lang="ts">
  import { geometryViewSchema } from "../lib/interfaceSchemas";
  import { registerInterfaceController } from "../lib/workspaceController";
  import { ToolError } from "../lib/webmcp/types";
  import { onMount as onInterfaceMount } from "svelte";

  import { Expander, MorphText } from "@stdlogits/ledger";
  // A selected probe, in the Inspector (``openDrawer("probe_inspector", { name })``; the crumbs
  // carry the close). Its name and facts; its reading token by token (a scrubbable trace) and layer
  // by layer (a bar per layer in that layer's colour); for a geometry probe, where it sits (the
  // subspace share, the nearest node, one signed meter per coordinate, poles on a concept axis),
  // and its geometry in the whitened (Mahalanobis) frame:
  //
  //   rank 1   -> a line: poles + neutral + live point
  //   rank 2   -> a 2D scatter of node centroids (+ curve overlay if 1-D)
  //   rank 3+  -> a drag-orbit 3D scatter on the top-3 subspace PCs (+ curve / surface overlay)
  //
  // The share-by-layer bars (or, on a touch screen, the layer stepper beside them) choose the plotted
  // layer; geometry is fetched once (all layers) and reprojected client-side. The live point and its trail ride the probe's per-token
  // ``subspace_coords_per_layer``, stored across all layers so changing layer is a pure read. They come from the live token stream
  // only: a loom row keeps each reading's summary, so a saved reply draws no trail, and its "By layer" view replays the token.
  // Everything draws at the probe's depth on the ramp (--tint), which only fills.

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
  import { apiProbes } from "../lib/runtime/services";
  import { cachedTokenReadout } from "../lib/runtime/tokenReadoutCache";
  import { readingLayers } from "../lib/tokenRows";
  import { runtimeOperationAvailability } from "../lib/runtime/ui-capabilities";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import Bar from "../lib/charts/Bar.svelte";
  import Trace from "../lib/charts/Trace.svelte";
  import TraceDetail from "../lib/charts/TraceDetail.svelte";
  import { chartValue } from "../lib/charts/chartValues";
  import { depthTint, layerDepth, peakLayer, readingText } from "../lib/charts/depth";
  import {
    renderProbeGeometry,
    orbitDrag,
    DEFAULT_ORBIT_QUAT,
    type OrbitState,
    type GeometryHitPoint,
  } from "../lib/charts/probeGeometry";
  import {
    pinchMetrics,
    scaleFromPinch,
    type GesturePoint,
  } from "../lib/pointerGesture";
  import { nodeCoordExtent, parseProbeTarget } from "../lib/tokens";
  import { polesOf } from "../lib/concepts";
  import ManifoldMiniMap from "../panels/manifold/ManifoldMiniMap.svelte";
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
  const liveTrailAvailability = runtimeOperationAvailability(
    "probe_subspace_trails",
  );

  let geom = $state<ProbeGeometryResponse | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let selectedLayer = $state<number | null>(null);
  const orbit = $state<OrbitState>({ q: DEFAULT_ORBIT_QUAT, zoom: 1.6 });

  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let rafId = 0;
  let hitPoints: GeometryHitPoint[] = [];
  let plotTitle = $state("Hover a node to see its whitened coordinates");
  let loadEpoch = 0;

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

  // --- load geometry on probe change ---
  async function load(name: string, epoch: number): Promise<void> {
    if (!name || !geometric) {
      if (epoch === loadEpoch) {
        geom = null;
        loading = false;
        error = null;
      }
      return;
    }
    loading = true;
    error = null;
    geom = null;
    try {
      const g = await apiProbes.geometry(name);
      if (epoch !== loadEpoch) return;
      geom = g;
      // default to the highest-share layer (the one carrying the most
      // steering budget — also the most concept-bearing to read from)
      let best: number | null = null;
      let bestShare = -Infinity;
      for (const l of Object.values(g.layers)) {
        const sh = Math.abs(l.mahalanobis_share);
        if (sh > bestShare) {
          bestShare = sh;
          best = l.layer;
        }
      }
      selectedLayer = best;
    } catch (e) {
      if (epoch !== loadEpoch) return;
      error = userFacingError(
        e,
        "Unable to load this direction map. Reattach the direction and try again.",
      );
    } finally {
      if (epoch === loadEpoch) loading = false;
    }
  }

  $effect(() => {
    const epoch = ++loadEpoch;
    void load(probeName, epoch);
    return () => {
      if (loadEpoch === epoch) loadEpoch += 1;
    };
  });

  // sorted layer list (ascending) for the share bars
  const layerList = $derived.by<ProbeLayerGeometry[]>(() =>
    geom ? Object.values(geom.layers).sort((a, b) => a.layer - b.layer) : [],
  );
  const activeGeom = $derived(
    geom && selectedLayer !== null
      ? (geom.layers[String(selectedLayer)] ?? null)
      : null,
  );
  const selectedIndex = $derived(layerList.findIndex((l) => l.layer === selectedLayer));
  /** Step the plotted layer (the touch screen's way to choose one; a bar is a few pixels wide). */
  function stepLayer(by: number): void {
    const next = layerList[Math.max(0, Math.min(layerList.length - 1, selectedIndex + by))];
    if (next) selectedLayer = next.layer;
  }
  const maxShare = $derived(
    layerList.reduce((m, l) => Math.max(m, Math.abs(l.mahalanobis_share)), 0),
  );
  const geomDeepest = $derived(Math.max(deepest, ...layerList.map((l) => l.layer)));
  const bestShareLayer = $derived(
    layerList.reduce<ProbeLayerGeometry | null>((best, l) => (best === null || Math.abs(l.mahalanobis_share) > Math.abs(best.mahalanobis_share) ? l : best), null)?.layer ?? null,
  );
  /** The probe's place on the ramp: its measured depth, else the layer its fit carries most. */
  const tint = $derived(depthTint(depthCom ?? (bestShareLayer === null ? null : layerDepth(bestShareLayer, geomDeepest))));

  // --- live point + trail for the selected layer (reprojected client-side) ---
  const layerKey = $derived(selectedLayer !== null ? String(selectedLayer) : "");
  const livePoint = $derived.by<number[] | null>(() => {
    const trail = entry?.subspaceTrail;
    if (!trail || trail.length === 0) return null;
    return trail[trail.length - 1]?.perLayer[layerKey] ?? null;
  });
  const trailPoints = $derived.by<number[][]>(() => {
    const trail = entry?.subspaceTrail ?? [];
    const out: number[][] = [];
    for (const s of trail) {
      const p = s.perLayer[layerKey];
      if (Array.isArray(p)) out.push(p);
    }
    return out;
  });

  // --- canvas render (rAF-coalesced; re-runs on any dependency change) ---
  $effect(() => {
    const canvas = canvasEl;
    const g = activeGeom;
    // touch the reactive deps so the effect re-subscribes
    const live = livePoint;
    const trail = trailPoints;
    const q = orbit.q;
    const zoom = orbit.zoom;
    const labels = geom?.node_labels ?? [];
    if (!canvas || !g) {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      return;
    }
    const redraw = () => {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        rafId = 0;
        hitPoints = renderProbeGeometry(canvas, {
          geom: g,
          nodeLabels: labels,
          live,
          trail,
          orbit: { q, zoom },
        });
      });
    };
    const observer = new ResizeObserver(redraw);
    observer.observe(canvas);
    redraw();
    return () => {
      observer.disconnect();
      if (rafId) cancelAnimationFrame(rafId);
      rafId = 0;
    };
  });

  // --- orbit interaction (rank >= 3 only) ---
  let dragPointerId: number | null = null;
  let lastX = 0;
  let lastY = 0;
  const touchPointers = new Map<number, GesturePoint>();
  let pinchGesture: { distance: number; zoom: number } | null = null;
  const canOrbit = $derived((activeGeom?.rank ?? 0) >= 3);

  function capturePlotPointer(target: HTMLElement, pointerId: number): void {
    try {
      target.setPointerCapture(pointerId);
    } catch {
      // The pointer may already have ended between events.
    }
  }

  function releasePlotPointer(target: HTMLElement, pointerId: number): void {
    if (!target.hasPointerCapture(pointerId)) return;
    try {
      target.releasePointerCapture(pointerId);
    } catch {
      // A cancelled pointer releases capture automatically.
    }
  }

  function touchPair(): [GesturePoint, GesturePoint] | null {
    const points = [...touchPointers.values()];
    return points.length >= 2 ? [points[0], points[1]] : null;
  }

  function beginPlotPinch(target: HTMLElement): void {
    const pair = touchPair();
    if (!pair) return;
    const { distance } = pinchMetrics(pair[0], pair[1]);
    if (distance <= 0) return;
    dragPointerId = null;
    pinchGesture = { distance, zoom: orbit.zoom };
    for (const pointerId of touchPointers.keys()) {
      capturePlotPointer(target, pointerId);
    }
  }

  function onPointerDown(ev: PointerEvent): void {
    if (!canOrbit) return;
    if (ev.pointerType === "touch") {
      touchPointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
      if (touchPointers.size >= 2) {
        beginPlotPinch(ev.currentTarget as HTMLElement);
        ev.preventDefault();
        return;
      }
    }
    dragPointerId = ev.pointerId;
    lastX = ev.clientX;
    lastY = ev.clientY;
    capturePlotPointer(ev.currentTarget as HTMLElement, ev.pointerId);
    ev.preventDefault();
  }
  function onPointerMove(ev: PointerEvent): void {
    if (canvasEl && ev.buttons === 0) {
      const rect = canvasEl.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      let closest: GeometryHitPoint | undefined;
      let distance = 16;
      for (const hit of hitPoints) {
        const d = Math.hypot(x - hit.screen[0], y - hit.screen[1]);
        if (d <= distance) { closest = hit; distance = d; }
      }
      plotTitle = closest
        ? `${closest.label} · whitened coordinates [${closest.point.map((v) => chartValue(v)).join(", ")}]`
        : "Hover a node to see its whitened coordinates";
    }
    if (!canOrbit) return;
    if (ev.pointerType === "touch" && touchPointers.has(ev.pointerId)) {
      touchPointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
      const pair = touchPair();
      if (pair && pinchGesture) {
        orbit.zoom = scaleFromPinch(
          pinchGesture.zoom,
          pinchGesture.distance,
          pinchMetrics(pair[0], pair[1]).distance,
          0.3,
          6,
        );
        ev.preventDefault();
        return;
      }
    }
    if (dragPointerId !== ev.pointerId) return;
    const dx = ev.clientX - lastX;
    const dy = ev.clientY - lastY;
    lastX = ev.clientX;
    lastY = ev.clientY;
    // Trackball: compose a screen-axis rotation onto the accumulated
    // orientation.  No Euler angles → no gimbal lock, no elevation clamp.
    orbit.q = orbitDrag(orbit.q, dx, dy);
  }
  function onPointerUp(ev: PointerEvent): void {
    const target = ev.currentTarget as HTMLElement;
    releasePlotPointer(target, ev.pointerId);
    if (ev.pointerType === "touch") {
      const wasPinching = pinchGesture !== null;
      touchPointers.delete(ev.pointerId);
      if (wasPinching) {
        if (touchPointers.size >= 2) beginPlotPinch(target);
        else pinchGesture = null;
      }
    }
    if (dragPointerId === ev.pointerId) dragPointerId = null;
  }
  // Scroll wheel = intentional zoom.  Multiplicative so each notch is a constant ratio; clamped.
  function onWheel(ev: WheelEvent): void {
    if (!canOrbit) return;
    ev.preventDefault();
    const factor = Math.exp(-ev.deltaY * 0.0015);
    orbit.zoom = Math.max(0.3, Math.min(6, orbit.zoom * factor));
  }

  function rotateBy(dx: number, dy: number): void {
    if (!canOrbit) return;
    orbit.q = orbitDrag(orbit.q, dx, dy);
  }

  function zoomBy(factor: number): void {
    orbit.zoom = Math.max(0.3, Math.min(6, orbit.zoom * factor));
  }

  function resetView(): void {
    orbit.q = DEFAULT_ORBIT_QUAT;
    orbit.zoom = 1.6;
  }

  function formatPoint(point: number[] | null): string {
    if (!point || point.length === 0) return "not available";
    return point.slice(0, 3).map((value) => value.toFixed(3)).join(", ");
  }

  const rankLabel = $derived.by(() => {
    const r = activeGeom?.rank ?? geo?.intrinsic_dim ?? 0;
    if (r <= 1) return "a line";
    if (r === 2) return "a plane";
    return `rank ${r}`;
  });

  onInterfaceMount(() => registerInterfaceController("probe_geometry", {
    schema: geometryViewSchema,
    read: () => ({ values: { layer: selectedLayer, zoom: orbit.zoom, quaternion: orbit.q }, available_layers: layerList.map(row => row.layer), can_rotate: canOrbit, loading }),
    update: (change) => {
      const layer = change.layer as number | undefined;
      if (layer !== undefined && !layerList.some(row => row.layer === layer)) throw new ToolError("NOT_FOUND", "Choose a layer from available_layers.");
      const target = layer === undefined ? activeGeom : geom?.layers[String(layer)];
      if (!target) throw new ToolError("NOT_READY", "Wait for this probe's geometry to load.");
      if ((change.rotate || change.zoom !== undefined) && target.rank < 3) throw new ToolError("UNAVAILABLE", "Orbit and zoom controls require geometry of rank 3 or greater.");
      if (layer !== undefined) selectedLayer = layer;
      if (change.reset) resetView();
      if (change.zoom !== undefined) orbit.zoom = change.zoom as number;
      if (change.rotate) { const delta = change.rotate as { dx: number; dy: number }; rotateBy(delta.dx, delta.dy); }
    },
  }));
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
              <span class="cl">{showPoles ? (poles.negative ?? "") : `axis ${ax.i}`}</span>
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

      <section class="dw-sec">
        <div class="dw-sec-head"><span class="head">Geometry</span>{#if activeGeom}<span class="meta"><MorphText text={`L${selectedLayer} · share ${activeGeom.mahalanobis_share.toFixed(3)}`} /></span>{/if}</div>
        {#if loading}
          <div class="well" aria-busy="true" role="status"><span class="sr-only">Loading probe geometry…</span></div>
        {:else if error}
          <p class="meta" role="alert">Probe geometry failed: {error}</p>
        {:else if !geom || !activeGeom || layerList.length === 0}
          <p class="meta">This probe has no fitted geometry.</p>
        {:else}
          <div class="shares" role="group" aria-label="Share by layer: choose the layer to plot">
            {#each layerList as l (l.layer)}
              <button
                type="button"
                class="share"
                aria-pressed={l.layer === selectedLayer}
                style:--tint={depthTint(layerDepth(l.layer, geomDeepest))}
                data-tip={`L${l.layer} · share ${l.mahalanobis_share.toFixed(3)} · ${chartValue(maxShare > 0 ? Math.abs(l.mahalanobis_share) / maxShare : 0, true)} of the largest`}
                aria-label={`Layer ${l.layer}, share ${l.mahalanobis_share.toFixed(3)}`}
                onclick={() => (selectedLayer = l.layer)}
              ><i style:height={`${Math.max(6, maxShare > 0 ? (Math.abs(l.mahalanobis_share) / maxShare) * 100 : 0)}%`}></i></button>
            {/each}
          </div>
          <div class="ends"><span>L{layerList[0].layer}</span><span>L{layerList[layerList.length - 1].layer}</span></div>
          <div class="step" role="group" aria-label="Plotted layer">
            <button type="button" class="dw-text" disabled={selectedIndex <= 0} onclick={() => stepLayer(-1)} aria-label="Plot the layer before">earlier</button>
            <span class="meta"><MorphText text={`L${selectedLayer}`} /></span>
            <button type="button" class="dw-text" disabled={selectedIndex >= layerList.length - 1} onclick={() => stepLayer(1)} aria-label="Plot the layer after">later</button>
          </div>

          <div class="well" class:orbit={canOrbit}>
            <canvas
              bind:this={canvasEl}
              class="plot"
              {...{ "aria-description": (plotTitle) }}
              aria-label={`Whitened geometry for ${displayName}, layer ${selectedLayer}`}
              aria-describedby="probe-geometry-summary"
              data-orbit-zoom={orbit.zoom.toFixed(2)}
              onpointerdown={onPointerDown}
              onpointermove={onPointerMove}
              onpointerup={onPointerUp}
              onpointercancel={onPointerUp}
              onlostpointercapture={onPointerUp}
              onwheel={onWheel}
            >Whitened geometry for {displayName}, layer {selectedLayer}.</canvas>
          </div>
          <div class="plot-foot">
            {#if trailPoints.length > 0}
              <span class="meta"><MorphText text={`${trailPoints.length} trail points`} /></span>
            {:else if !liveTrailAvailability.available}
              <span class="meta" {...{ "aria-description": (liveTrailAvailability.reason ?? undefined) }}>live trail unavailable here</span>
            {:else}
              <span class="meta">the trail draws while a reply streams</span>
            {/if}
            {#if canOrbit}
              <span class="meta">drag · scroll or pinch</span>
              <span class="plot-controls" role="group" aria-label="Geometry view controls">
                <button type="button" class="dw-text" aria-label="Rotate geometry left" onclick={() => rotateBy(-18, 0)}>left</button>
                <button type="button" class="dw-text" aria-label="Rotate geometry right" onclick={() => rotateBy(18, 0)}>right</button>
                <button type="button" class="dw-text" aria-label="Rotate geometry up" onclick={() => rotateBy(0, -18)}>up</button>
                <button type="button" class="dw-text" aria-label="Rotate geometry down" onclick={() => rotateBy(0, 18)}>down</button>
                <button type="button" class="dw-text" aria-label="Zoom geometry out" onclick={() => zoomBy(0.8)}>−</button>
                <button type="button" class="dw-text" aria-label="Zoom geometry in" onclick={() => zoomBy(1.25)}>+</button>
                <button type="button" class="dw-text" onclick={resetView}>reset</button>
              </span>
            {/if}
          </div>
          <p class="sr-only" id="probe-geometry-summary">
            Neutral [{formatPoint(activeGeom.neutral_white)}]; live point [{formatPoint(livePoint)}];
            {activeGeom.node_white.length} node {activeGeom.node_white.length === 1 ? "centroid" : "centroids"}.
          </p>
          <Expander label="Coordinates">
            <ul class="points">
              <li><span>neutral</span><span>{formatPoint(activeGeom.neutral_white)}</span></li>
              <li><span>live</span><span><MorphText text={formatPoint(livePoint)} identity={selectedLayer} /></span></li>
              {#each activeGeom.node_white as point, index (index)}
                <li><span>{geom.node_labels[index] ?? `node ${index + 1}`}</span><span>{formatPoint(point)}</span></li>
              {/each}
            </ul>
          </Expander>
          {#if !liveTrailAvailability.available}
            <p class="meta" role="note">{liveTrailAvailability.reason}. The fitted geometry remains available.</p>
          {/if}
        {/if}
      </section>
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

  /* The share by layer: each bar a button that plots its layer. */
  .shares { display: flex; align-items: flex-end; gap: 2px; height: 56px; }
  .share { position: relative; display: flex; align-items: flex-end; flex: 1 1 0; min-width: 2px; height: 100%; padding: 0; border: 0; border-radius: 2px; background: transparent; cursor: pointer; }
  .share i { display: block; width: 100%; border-radius: 2px; background: var(--tint); opacity: 0.4; transition: opacity var(--fast) var(--ease), height var(--glide) var(--ease); }
  .share:hover i { opacity: 0.7; }
  .share[aria-pressed="true"] i { opacity: 1; }
  /* On a touch screen a bar is a few pixels wide, so a stepper chooses the layer instead. */
  .step { display: none; align-items: center; justify-content: space-between; gap: var(--s3); }
  .step .dw-text { min-height: 44px; padding: 0 var(--s2); }
  @media (max-width: 720px), (pointer: coarse) {
    .step { display: flex; }
  }

  /* The plot: a square wash the geometry draws on. */
  .well {
    position: relative;
    width: 100%;
    aspect-ratio: 1;
    border-radius: var(--r-box);
    background: var(--wash);
    overflow: hidden;
    /* Read by the canvas renderer: nodes at the probe's depth, neutral in faint ink. */
    --geom-node: var(--tint);
    --geom-neutral: var(--ink3);
  }
  /* On the phone's sheet (itself a wash) the plot and the map step up to wash2. */
  :global(.sheet) .well, :global(.sheet) .map { background: var(--wash2); }
  .well.orbit .plot { cursor: grab; touch-action: none; }
  .well.orbit .plot:active { cursor: grabbing; }
  .plot { position: absolute; inset: 0; display: block; width: 100%; height: 100%; }
  .plot-foot { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s4); }
  .plot-controls { display: flex; flex-wrap: wrap; gap: var(--s3); margin-left: auto; }
  .plot-controls button { font: var(--type-meta); }

  .points { display: grid; gap: 2px; color: var(--ink2); font: var(--type-meta); font-variant-numeric: tabular-nums; }
  .points li { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: var(--s3); }
  .points li span:first-child { overflow: hidden; color: var(--ink); white-space: nowrap; text-overflow: ellipsis; }

  @media (max-width: 720px) {
    .coord { grid-template-columns: 64px minmax(0, 1fr) 44px; }
    .coord:has(.pole) { grid-template-columns: 64px minmax(0, 1fr) 44px 64px; }
    .plot-controls { gap: 0; }
    .plot-controls button { min-width: 44px; min-height: 44px; }
  }
</style>
