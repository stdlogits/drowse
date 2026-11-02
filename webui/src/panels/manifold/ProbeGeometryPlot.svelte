<script lang="ts">
  // A geometry probe's fitted geometry in the whitened (Mahalanobis) frame, layer by layer (the probe
  // inspector's plot, shared with the Probes room):
  //
  //   rank 1   -> a line: poles + neutral + the point
  //   rank 2   -> a 2D scatter of node centroids (+ curve overlay if 1-D)
  //   rank 3+  -> a drag-orbit 3D scatter on the top-3 subspace PCs (+ curve / surface overlay)
  //
  // The share-by-layer bars (or, on a touch screen, the layer stepper beside them) choose the plotted
  // layer; geometry is fetched once (all layers) and reprojected client-side. The point is ``point``
  // (a token's per-layer subspace coordinates) when given, else the live trail's latest; the trail
  // is ``trail`` (the stream's per-token ``subspace_coords_per_layer``). Everything draws at the
  // probe's depth on the ramp (--tint, set by the parent). ``geom`` and ``layer`` bind out: what
  // loaded and the plotted layer. ``controller`` registers the WebMCP geometry view (one mount).
  import { Expander, MorphText } from "@stdlogits/ledger";
  import { geometryViewSchema } from "../../lib/interfaceSchemas";
  import { registerInterfaceController } from "../../lib/workspaceController";
  import { ToolError } from "../../lib/webmcp/types";
  import { apiProbes } from "../../lib/runtime/services";
  import { runtimeOperationAvailability } from "../../lib/runtime/ui-capabilities";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { chartValue, readingWeight } from "../../lib/charts/chartValues";
  import { depthTint, layerDepth } from "../../lib/charts/depth";
  import {
    renderProbeGeometry,
    orbitDrag,
    DEFAULT_ORBIT_QUAT,
    DEFAULT_ORBIT_ZOOM,
    type OrbitState,
    type GeometryHitPoint,
  } from "../../lib/charts/probeGeometry";
  import { pinchMetrics, scaleFromPinch, type GesturePoint } from "../../lib/pointerGesture";
  import type { ProbeGeometryResponse, ProbeLayerGeometry } from "../../lib/types";

  let {
    name,
    label,
    trail = [],
    point = null,
    deepest = 0,
    controller = false,
    geom = $bindable(null),
    layer: selectedLayer = $bindable(null),
  }: {
    name: string;
    label: string;
    trail?: readonly { perLayer: Record<string, number[]> }[];
    point?: Record<string, number[]> | null;
    deepest?: number;
    controller?: boolean;
    geom?: ProbeGeometryResponse | null;
    layer?: number | null;
  } = $props();

  const liveTrailAvailability = runtimeOperationAvailability(
    "probe_subspace_trails",
  );
  let loading = $state(false);
  let error = $state<string | null>(null);
  const orbit = $state<OrbitState>({ q: DEFAULT_ORBIT_QUAT, zoom: DEFAULT_ORBIT_ZOOM });

  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let rafId = 0;
  let hitPoints: GeometryHitPoint[] = [];
  let plotTitle = $state("Hover a node to see its whitened coordinates");
  let loadEpoch = 0;

  // --- load geometry on probe change ---
  async function load(name: string, epoch: number): Promise<void> {
    if (!name) {
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
    void load(name, epoch);
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
  const layerWeights = $derived(layerList.map((l) => l.mahalanobis_share));
  const maxShare = $derived(
    layerList.reduce((m, l) => Math.max(m, Math.abs(l.mahalanobis_share)), 0),
  );
  const geomDeepest = $derived(Math.max(deepest, ...layerList.map((l) => l.layer)));

  // --- live point + trail for the selected layer (reprojected client-side) ---
  const layerKey = $derived(selectedLayer !== null ? String(selectedLayer) : "");
  const livePoint = $derived.by<number[] | null>(() => {
    if (point) return point[layerKey] ?? null;
    if (trail.length === 0) return null;
    return trail[trail.length - 1]?.perLayer[layerKey] ?? null;
  });
  const trailPoints = $derived.by<number[][]>(() => {
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
    orbit.zoom = DEFAULT_ORBIT_ZOOM;
  }

  function formatPoint(point: number[] | null): string {
    if (!point || point.length === 0) return "not available";
    return point.slice(0, 3).map((value) => value.toFixed(3)).join(", ");
  }


  $effect(() => {
    if (!controller) return;
    return registerInterfaceController("probe_geometry", {
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
    });
  });
</script>

  <section class="dw-sec">
    <div class="dw-sec-head"><span class="head">Geometry</span>{#if activeGeom}<span class="meta" data-tip="This layer's weight in the probe's reading (every layer's fitted spread in the subspace, summed to 100%)"><MorphText text={`L${selectedLayer} · ${readingWeight(activeGeom.mahalanobis_share, layerWeights)} of the reading`} /></span>{/if}</div>
    {#if loading}
      <div class="well" aria-busy="true" role="status"><span class="sr-only">Loading probe geometry…</span></div>
    {:else if error}
      <p class="meta" role="alert">Probe geometry failed: {error}</p>
    {:else if !geom || !activeGeom || layerList.length === 0}
      <p class="meta">This probe has no fitted geometry.</p>
    {:else}
      <div class="shares" role="group" aria-label="Weight by layer: choose the layer to plot">
        {#each layerList as l (l.layer)}
          <button
            type="button"
            class="share"
            aria-pressed={l.layer === selectedLayer}
            style:--tint={depthTint(layerDepth(l.layer, geomDeepest))}
            data-tip={`L${l.layer} · ${readingWeight(l.mahalanobis_share, layerWeights)} of the reading`}
            aria-label={`Layer ${l.layer}, ${readingWeight(l.mahalanobis_share, layerWeights)} of the reading`}
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
          aria-label={`Whitened geometry for ${label}, layer ${selectedLayer}`}
          aria-describedby="probe-geometry-summary"
          data-orbit-zoom={orbit.zoom.toFixed(2)}
          onpointerdown={onPointerDown}
          onpointermove={onPointerMove}
          onpointerup={onPointerUp}
          onpointercancel={onPointerUp}
          onlostpointercapture={onPointerUp}
          onwheel={onWheel}
        >Whitened geometry for {label}, layer {selectedLayer}.</canvas>
      </div>
      <div class="plot-foot">
        {#if trailPoints.length > 0}
          <span class="meta"><MorphText text={`${trailPoints.length} trail points`} /></span>
        {:else if !liveTrailAvailability.available}
          <span class="meta" {...{ "aria-description": (liveTrailAvailability.reason ?? undefined) }}>live trail unavailable here</span>
        {/if}
        {#if canOrbit}
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

<style>
  /* The share by layer: each bar a button that plots its layer. */
  .shares { display: flex; align-items: flex-end; gap: 2px; height: 56px; }
  .share { position: relative; display: flex; align-items: flex-end; flex: 1 1 0; min-width: 2px; height: 100%; padding: 0; border: 0; border-radius: 2px; background: transparent; cursor: pointer; }
  .share i { display: block; width: 100%; border-radius: 2px; background: var(--tint); opacity: 0.4; transition: opacity var(--fast) var(--ease), height var(--glide) var(--ease); }
  .share:hover i { opacity: 0.7; }
  .share[aria-pressed="true"] i { opacity: 1; }
  .ends { display: flex; justify-content: space-between; color: var(--ink3); font: var(--type-meta); font-variant-numeric: tabular-nums; }
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
  /* On the phone's sheet (itself a wash) the plot steps up to wash2. */
  :global(.sheet) .well { background: var(--wash2); }
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
    .plot-controls { gap: 0; }
    .plot-controls button { min-width: 44px; min-height: 44px; }
  }
</style>
