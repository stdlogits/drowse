<script lang="ts">
  // The Probes room (#/probes): the attached geometry probes read at the token cursor (TokenStrip).
  // First the linear probes and vectors (one-axis flat fits): each one's value at the token and its
  // trace over the whole conversation, the token marked (a click on the trace moves the cursor
  // there). Then the manifolds (fits of more axes, flat fans or curved): where the token sits (its share of the subspace, its nearest node, one
  // signed meter per coordinate) and the fitted geometry it sits in (ProbeGeometryPlot, the probe
  // inspector's plot, the token placed in it). Then the SAE features the rack pins. Whole, the
  // reply's aggregate and the live trail. A token's row keeps each probe's value; the rest (a
  // manifold's nearest node and whitened place, a probe attached since) replays the token, steered
  // as it was generated.
  import { Icon } from "@stdlogits/ledger";
  import TokenStrip from "./TokenStrip.svelte";
  import ProbeGeometryPlot from "../manifold/ProbeGeometryPlot.svelte";
  import Bar from "../../lib/charts/Bar.svelte";
  import Trace from "../../lib/charts/Trace.svelte";
  import { depthTint, readingText } from "../../lib/charts/depth";
  import { cachedTokenReadout } from "../../lib/runtime/tokenReadoutCache";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { chatLog, effectiveRawMode, genStatus } from "../../lib/stores/chat.svelte";
  import { loomTree } from "../../lib/stores/loom.svelte";
  import { goto } from "../../lib/stores/rooms.svelte";
  import { openDrawer } from "../../lib/stores/drawers.svelte";
  import { probeRack } from "../../lib/stores/probes.svelte";
  import { instrumentFamily } from "../../lib/stores.svelte";
  import { moveCursor, settled, tokenCursor } from "../../lib/stores/tokenCursor.svelte";
  import { segmentTokens } from "../../drawers/token/cursor";
  import { axisName } from "../../drawers/token/readings";
  import { nodeCoordExtent } from "../../lib/tokens";
  import { rowGeometry, withRowValues } from "../../lib/tokenRows";
  import { readsShare, tokenReading } from "../rack/probeTrace";
  import { probeTitle } from "../rack/probeRows";
  import type { GeometryProbeInfo, ProbeGeometryResponse, ProbeInfo, ProbeReadingJSON, TokenScore } from "../../lib/types";

  /** The attached probes this room reads, in the rack's order, by kind. */
  const attached = $derived(probeRack.active
    .map((name) => ({ name, info: probeRack.entries.get(name)?.info ?? null }))
    .filter((row): row is { name: string; info: ProbeInfo } => row.info !== null));
  /** A linear probe or vector: a flat fit of one axis (a 2-node concept, a 1-node direction). */
  const isLinear = (info: ProbeInfo) => info.family === "geometry" && (info as GeometryProbeInfo).is_affine && (info as GeometryProbeInfo).intrinsic_dim <= 1;
  const linear = $derived(attached.filter((p) => isLinear(p.info)));
  /** A manifold: a fit of more than one axis, flat (a fan) or curved. */
  const manifolds = $derived(attached.filter((p) => p.info.family === "geometry" && !isLinear(p.info)));
  const features = $derived(attached.filter((p) => p.info.family === "sae"));
  const saeLayer = $derived.by((): number | null => {
    const live = instrumentFamily("sae")?.live;
    return live && "layer" in live ? live.layer : null;
  });

  // ---------- the walk: every token of the conversation, in the strip's order
  const walk = $derived(tokenCursor.segments.flatMap((s) => segmentTokens(chatLog.turns[s.turnIdx], s.seg)));
  const token = $derived(tokenCursor.token);
  const whole = $derived(token === null);
  const reply = $derived(tokenCursor.reply);
  /** The whole reply's span of the walk. */
  const replySpan = $derived.by((): [number, number] | null => {
    if (!reply) return null;
    let start = 0;
    let first = -1;
    let last = -1;
    for (const s of tokenCursor.segments) {
      if (s.turnIdx === reply.turnIdx) {
        if (first < 0) first = start;
        last = start + s.length - 1;
      }
      start += s.length;
    }
    return first < 0 ? null : [first, last];
  });
  /** A probe's reading of every token of the walk (null where a token has none). */
  function traceOf(name: string, info: ProbeInfo): (number | null)[] {
    const share = readsShare(info);
    return walk.map((t: TokenScore) => tokenReading(t, name, { share }));
  }
  /** The reply's aggregate reading of a geometry probe, as its node keeps it. */
  const aggregate = $derived(reply?.nodeId ? (loomTree.nodes.get(reply.nodeId)?.aggregate_measurements ?? null) : null);

  // ---------- the token's geometry: its row, completed by a replay where it keeps less
  const here = settled(() => tokenCursor.token);
  let replay = $state.raw<{ key: string; readings: Record<string, ProbeReadingJSON> | null; error: string | null }>({ key: "", readings: null, error: null });
  const geometryReplay = $derived(instrumentFamily("geometry")?.capabilities.token_readout === true);
  const want = $derived.by(() => {
    const t = here.current;
    if (!t || !geometryReplay || t.nodeId === null || t.index === null || linear.length + manifolds.length === 0) return null;
    return { nodeId: t.nodeId, index: t.index, row: rowGeometry(t.token.measurements), key: `${t.nodeId}:${t.index}:${effectiveRawMode()}:${[...linear, ...manifolds].map((p) => p.name).join(",")}` };
  });
  $effect(() => {
    const w = want;
    if (!w || genStatus.active || replay.key === w.key) return;
    const key = w.key;
    replay = { key, readings: null, error: null };
    cachedTokenReadout("geometry", w.nodeId, w.index, { steered: true, raw: effectiveRawMode() }).then(
      (res) => {
        if (replay.key !== key) return;
        replay = { key, readings: withRowValues(res.measurements.instruments.geometry?.readings ?? {}, w.row), error: null };
      },
      (error: unknown) => { if (replay.key === key) replay = { key, readings: null, error: userFacingError(error, "This token's probes could not be replayed.") }; },
    );
  });
  /** A geometry probe's reading at the token (replayed, its row's values kept), or, whole, over the reply. */
  function readingOf(name: string): ProbeReadingJSON | null {
    if (whole) return (aggregate?.instruments.geometry?.readings?.[name] as ProbeReadingJSON | undefined) ?? null;
    const replayed = replay.key === want?.key ? replay.readings?.[name] : undefined;
    if (replayed) return replayed;
    const row = rowGeometry(token?.token.measurements)[name];
    return row ? ({ fraction: row.fraction, coords: row.coords } as ProbeReadingJSON) : null;
  }
  /** A probe's value: at the token (its row's), else (whole) its mean over the reply. */
  function valueOf(name: string, trace: (number | null)[]): number | null {
    if (token) return trace[token.ordinal] ?? readingOf(name)?.coords?.[0] ?? null;
    const kept = reply?.turn.aggregateReadings?.[name];
    if (typeof kept === "number") return kept;
    if (!replySpan) return null;
    const values = trace.slice(replySpan[0], replySpan[1] + 1).filter((v): v is number => v !== null);
    return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;
  }

  /** A click on a trace lands the cursor on the token at that place. */
  function jump(event: MouseEvent): void {
    const box = (event.currentTarget as HTMLElement).getBoundingClientRect();
    if (box.width <= 0 || walk.length === 0) return;
    let target = Math.floor(((event.clientX - box.left) / box.width) * walk.length);
    target = Math.max(0, Math.min(walk.length - 1, target));
    let start = 0;
    for (const s of tokenCursor.segments) {
      if (target < start + s.length) {
        moveCursor({ turnIdx: s.turnIdx, seg: s.seg, tokenIdx: target - start });
        return;
      }
      start += s.length;
    }
  }
  const markAt = $derived(token ? token.ordinal : null);

  // ---------- the manifold shown
  let chosen = $state<string | null>(null);
  const shown = $derived(manifolds.find((p) => p.name === chosen) ?? manifolds[0] ?? null);
  let shownGeom = $state<ProbeGeometryResponse | null>(null);
  let shownLayer = $state<number | null>(null);
  const shownInfo = $derived(shown ? (shown.info as GeometryProbeInfo) : null);
  const shownReading = $derived(shown ? readingOf(shown.name) : null);
  const shownAxes = $derived.by(() => {
    if (!shownInfo) return [];
    const coords = shownReading?.coords ?? [];
    const n = shownInfo.intrinsic_dim > 0 ? shownInfo.intrinsic_dim : coords.length;
    return Array.from({ length: n }, (_, i) => ({ i, value: coords[i] ?? 0, scale: nodeCoordExtent(shownInfo.node_coords, i) }));
  });
  const nearest = $derived(shownReading?.nearest?.[0] ?? null);
  const tintOf = (info: ProbeInfo) => {
    const layers = (info as GeometryProbeInfo).layers ?? [];
    return depthTint(layers.length > 0 ? layers[Math.floor(layers.length / 2)] / Math.max(1, ...layers) : null) ?? undefined;
  };
</script>

<div class="dw-room probes">
  <TokenStrip />

  {#if attached.length === 0 || linear.length + manifolds.length + features.length === 0}
    <div class="dw-empty">
      <span class="head">No probes</span>
      <div class="dw-row">
        <button type="button" class="dw-quiet" onclick={() => openDrawer("subspace")}>A concept</button>
        <button type="button" class="dw-quiet" onclick={() => openDrawer("manifolds")}>A manifold</button>
        <button type="button" class="dw-quiet" onclick={() => goto("artifacts", "build")}>Build one<Icon name="right" /></button>
      </div>
    </div>
  {:else}
    {#if replay.error && !whole && replay.key === want?.key}<p class="meta">{replay.error}</p>{/if}
    {#if linear.length > 0}{@render probeRows("Probes", linear)}{/if}

    {#if shown && shownInfo}
      <section class="dw-sec manifolds" aria-labelledby="probes-manifolds-head" style:--tint={tintOf(shown.info)}>
        <div class="dw-sec-head"><span class="head" id="probes-manifolds-head">Manifolds</span>{#if nearest}<span class="meta">nearest {nearest[0]}{#if Number.isFinite(nearest[1])}, {nearest[1].toFixed(2)} away{/if}</span>{/if}</div>
        {#if manifolds.length > 1}
          <div class="pick" role="group" aria-label="Manifold shown">
            {#each manifolds as probe (probe.name)}
              <button type="button" class="dw-quiet" aria-pressed={probe.name === shown.name} onclick={() => (chosen = probe.name)}>{probeTitle(probe.info, probe.name).text}</button>
            {/each}
          </div>
        {/if}
        <div class="split">
          <div class="where">
            <h3 class="dw-title">{probeTitle(shown.info, shown.name).text}</h3>
            {#if shownReading}
              <div class="coords">
                {#if Number.isFinite(shownReading.fraction)}
                  <div class="coord"><span class="cl">in subspace</span><Bar value={shownReading.fraction} max={1} width={200} height={4} percentage /><span class="cv">{shownReading.fraction.toFixed(2)}</span></div>
                {/if}
                {#each shownAxes as ax (ax.i)}
                  <div class="coord">
                    <span class="cl">{axisName(shownGeom?.node_labels ?? undefined, ax.i, shownAxes.length)}</span>
                    <Bar value={ax.value} max={ax.scale} width={200} height={4} bipolar />
                    <span class="cv">{readingText(ax.value, { signed: true })}</span>
                  </div>
                {/each}
              </div>
              {@const trace = traceOf(shown.name, shown.info)}
              <button type="button" class="trace wide" tabindex="-1" aria-label="Its share over the conversation; choose a place to move there" onclick={jump}>
                <Trace values={trace} limit={0} maxBars={240} height={28} cursor={markAt} label="Its share of the subspace over the conversation" />
              </button>
            {:else}
              <div class="ghost" aria-hidden="true">{#each [0, 1, 2] as k (k)}<i></i>{/each}</div>
            {/if}
          </div>
          <div class="plot">
            <!-- At a token, the token's own place (its replay's whitened coordinates); whole, the live trail. -->
            <ProbeGeometryPlot name={shown.name} label={probeTitle(shown.info, shown.name).text}
              trail={whole ? (probeRack.entries.get(shown.name)?.subspaceTrail ?? []) : []}
              point={whole ? null : (shownReading?.subspace_coords_per_layer ?? {})}
              bind:geom={shownGeom} bind:layer={shownLayer} />
          </div>
        </div>
      </section>
    {/if}

    {#if features.length > 0}{@render probeRows("SAE features", features)}{/if}
  {/if}
</div>

{#snippet probeRows(head: string, rows: { name: string; info: ProbeInfo }[])}
  <section class="dw-sec" aria-label={head}>
    <div class="dw-sec-head"><span class="head">{head}</span><span class="meta">{whole ? "over this reply" : ""}</span></div>
    <div class="rows">
      {#each rows as probe (probe.name)}
        {@const trace = traceOf(probe.name, probe.info)}
        {@const value = valueOf(probe.name, trace)}
        {@const title = probeTitle(probe.info, probe.name, saeLayer)}
        <div class="row" style:--tint={tintOf(probe.info)}>
          <span class="pname" data-tip={probe.name}>{title.text}</span>
          <button type="button" class="trace" tabindex="-1" aria-label={`${title.text} over the conversation; choose a place to move there`} onclick={jump}>
            <Trace values={trace} limit={0} maxBars={240} height={28} cursor={markAt} label={`${title.text} over the conversation`} />
            {#if whole && replySpan && walk.length > 0}<span class="span" style:left="{(replySpan[0] / walk.length) * 100}%" style:width="{((replySpan[1] - replySpan[0] + 1) / walk.length) * 100}%"></span>{/if}
          </button>
          <span class="pval">{readingText(value, { signed: probe.info.family === "geometry" })}</span>
        </div>
      {/each}
    </div>
  </section>
{/snippet}

<style>
  .rows { display: grid; gap: 2px; }
  .row { display: grid; grid-template-columns: minmax(96px, 200px) minmax(0, 1fr) 64px; gap: var(--s3); align-items: center; min-height: 44px; padding: 0 var(--s3); border-radius: var(--r-box); }
  .row:hover { background: var(--wash); }
  .pname { overflow: hidden; color: var(--ink); font: var(--type-control); white-space: nowrap; text-overflow: ellipsis; }
  .pval { color: var(--ink); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; }
  .trace { position: relative; display: block; width: 100%; min-width: 0; padding: 0; border: 0; background: transparent; cursor: pointer; }
  .trace .span { position: absolute; bottom: -4px; height: 2px; border-radius: 1px; background: var(--ink2); }
  .trace.wide { margin-top: var(--s2); }

  .pick { display: flex; flex-wrap: wrap; gap: var(--s2); }
  .pick .dw-quiet[aria-pressed="false"] { background: transparent; color: var(--ink2); }
  .split { display: flex; flex-wrap: wrap; align-items: flex-start; gap: var(--s6); }
  .where { flex: 1 1 320px; min-width: 0; display: grid; gap: var(--s3); align-content: start; }
  .plot { flex: 1 1 360px; max-width: 520px; min-width: 0; display: grid; gap: var(--s3); }
  .coords { display: grid; gap: var(--s2); }
  .coord { display: grid; grid-template-columns: 96px minmax(0, 1fr) 52px; gap: var(--s3); align-items: center; min-height: 28px; }
  .coord :global(.bar) { width: 100%; }
  .cl { overflow: hidden; color: var(--ink2); font: var(--type-small); text-align: right; white-space: nowrap; text-overflow: ellipsis; }
  .cv { color: var(--ink); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; }
  .ghost { display: grid; gap: var(--s2); }
  .ghost > i { display: block; height: 28px; border-radius: var(--r-box); background: var(--wash); }

  @media (max-width: 520px) {
    .row { grid-template-columns: minmax(0, 1fr) 56px; grid-template-areas: "name val" "trace trace"; row-gap: var(--s1); padding: var(--s2) var(--s3); }
    .pname { grid-area: name; }
    .pval { grid-area: val; }
    .trace:not(.wide) { grid-area: trace; }
    .coord { grid-template-columns: 72px minmax(0, 1fr) 44px; }
  }
</style>
