<script lang="ts" module>
  export interface AtlasField {
    /** Every neuron's value per layer (``activation_over_scale`` at one token). */
    layers: Float32Array[];
    /** Nine cut points: a cell's shade is its decile among every neuron of the model. */
    cuts: readonly number[];
  }
</script>

<script lang="ts">
  // The atlas (the Units canvas): every layer a column in its own depth colour, its units as short
  // bars shaded by rank (ten shades, the faintest a tenth). Strongest draws each layer's strongest
  // units, sorted; All draws every neuron of each layer as a cell of a canvas, in neuron order,
  // fading in over the bars. A search lights its matches (their shade kept in the top three) and
  // sinks the rest (the bottom four); the selected unit is an ink bar. A bar is its unit's button
  // (All: the cell under the pointer is), and choosing one chooses the unit and its layer; the rest
  // of a column chooses just the layer, and a wash thumb glides under it. The atlas is one keyboard
  // stop: the arrows move along a column and across the layers (lib/units.ts::atlasStep).
  import { tick } from "svelte";
  import { hideTip, showTip } from "@stdlogits/ledger";
  import { depthTint } from "../../lib/charts/depth";
  import {
    SHADES,
    atlasStep,
    fieldCell,
    fieldCellAt,
    fieldGeometry,
    fieldLayout,
    shadeOf,
    signedValue,
    unitRef,
    unitTitle,
    type AtlasBar,
  } from "../../lib/units";

  let {
    columns,
    rows,
    lit,
    selected,
    layer,
    onlayer,
    onunit,
    field = null,
    hits = [],
    pin = null,
    label,
  }: {
    columns: AtlasBar[][];
    /** Bars a column holds at most (the plot's height). */
    rows: number;
    /** Units a search matched (null: not searching). */
    lit: ReadonlySet<string> | null;
    selected: string | null;
    layer: number;
    onlayer: (layer: number) => void;
    /** A unit chosen in the atlas: a bar, or (All) the cell under the pointer. */
    onunit: (id: string, layer: number) => void;
    field?: AtlasField | null;
    /** All mode: the cells a search matched, as (layer, neuron). */
    hits?: readonly { layer: number; index: number }[];
    /** All mode: the selected neuron, pinned by an ink bar. */
    pin?: { layer: number; index: number } | null;
    label: string;
  } = $props();

  const count = $derived(columns.length);
  const depth = (L: number) => depthTint(count > 1 ? L / (count - 1) : 0) ?? "var(--tint)";

  let plot: HTMLDivElement | null = $state(null);
  let canvas: HTMLCanvasElement | null = $state(null);
  let width = $state(0);
  let scheme = $state(0);
  const narrow = $derived(width > 0 && count > 0 && width / count < 14);
  const bar = $derived(narrow ? 2 : 3);
  const plotH = $derived(16 + Math.max(1, rows) * (bar + 2) - 2);
  const all = $derived(field !== null && field.layers.length === count);

  function opacity(shade: number, id: string): number {
    if (lit === null) return SHADES[shade];
    return lit.has(id) ? SHADES[7 + Math.round((shade * 2) / 9)] : SHADES[Math.round((shade * 3) / 9)];
  }

  $effect(() => {
    if (!plot) return;
    const ro = new ResizeObserver(() => { width = plot?.clientWidth ?? 0; });
    ro.observe(plot);
    width = plot.clientWidth;
    // A theme change repaints the field in the new colours.
    const mo = new MutationObserver(() => { scheme += 1; });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class", "style"] });
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onMedia = () => { scheme += 1; };
    media.addEventListener("change", onMedia);
    return () => { ro.disconnect(); mo.disconnect(); media.removeEventListener("change", onMedia); };
  });

  /** A CSS colour as bytes, read through the page (so a depth mix resolves in the current theme). */
  function bytes(ctx: CanvasRenderingContext2D, css: string): [number, number, number] {
    const el = canvas!;
    el.style.color = css;
    const resolved = getComputedStyle(el).color;
    el.style.color = "";
    ctx.clearRect(0, 0, 1, 1);
    ctx.fillStyle = resolved;
    ctx.fillRect(0, 0, 1, 1);
    const d = ctx.getImageData(0, 0, 1, 1).data;
    return [d[0], d[1], d[2]];
  }

  // Paint the field: each cell its layer's colour at its decile's shade, matches as small full
  // squares. Strongest leaves the last field in place, so it fades out under the bars.
  $effect(() => {
    const cv = canvas;
    const f = field;
    void scheme;
    if (!cv || !f || !all || width <= 0) return;
    const h = plotH;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(width * dpr);
    const H = Math.round(h * dpr);
    cv.width = W;
    cv.height = H;
    const ctx = cv.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    const tones = Array.from({ length: count }, (_, L) => bytes(ctx, depth(L)));
    ctx.clearRect(0, 0, W, H);
    const { colW, share, top, fh } = fieldLayout(width, h, count, narrow);
    const img = ctx.createImageData(W, H);
    const px = img.data;
    const searching = hits.length > 0;
    const fill = (x0: number, y0: number, x1: number, y1: number, rgb: readonly number[], a: number) => {
      for (let y = Math.max(0, y0); y < y1 && y < H; y++) {
        for (let x = Math.max(0, x0); x < x1 && x < W; x++) {
          const at = (y * W + x) * 4;
          if (px[at + 3] >= a) continue;
          px[at] = rgb[0];
          px[at + 1] = rgb[1];
          px[at + 2] = rgb[2];
          px[at + 3] = a;
        }
      }
    };
    const geom = f.layers.map(values => fieldGeometry(colW * share, fh, values.length));
    for (let L = 0; L < count; L++) {
      const values = f.layers[L];
      const { g, cw, ch } = geom[L];
      const x0 = (L + (1 - share) / 2) * colW;
      for (let i = 0; i < values.length; i++) {
        const sh = shadeOf(values[i], f.cuts);
        const a = Math.round(255 * (searching ? SHADES[Math.round((sh * 3) / 9)] : SHADES[sh]));
        const cx = i % g;
        const cy = (i - cx) / g;
        const xa = Math.floor((x0 + cx * cw) * dpr);
        const xb = Math.max(xa + 1, Math.floor((x0 + (cx + 1) * cw) * dpr));
        const ya = Math.floor((top + cy * ch) * dpr);
        const yb = Math.max(ya + 1, Math.floor((top + (cy + 1) * ch) * dpr));
        fill(xa, ya, xb, yb, tones[L], a);
      }
    }
    for (const hit of hits) {
      const values = f.layers[hit.layer];
      if (!values) continue;
      const { g, cw, ch } = geom[hit.layer];
      const x = (hit.layer + (1 - share) / 2) * colW + ((hit.index % g) + 0.5) * cw;
      const y = top + (Math.floor(hit.index / g) + 0.5) * ch;
      const s = Math.round(1.5 * dpr);
      const cx = Math.round(x * dpr);
      const cy = Math.round(y * dpr);
      for (let yy = cy - s; yy < cy + s; yy++) for (let xx = cx - s; xx < cx + s; xx++) {
        if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
        const at = (yy * W + xx) * 4;
        px[at] = tones[hit.layer][0];
        px[at + 1] = tones[hit.layer][1];
        px[at + 2] = tones[hit.layer][2];
        px[at + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  });

  const layout = $derived(fieldLayout(width, plotH, count, narrow));
  const totals = $derived(field?.layers.map(values => values.length) ?? []);

  const pinBox = $derived.by(() => {
    if (!all || !pin || !field || width <= 0) return null;
    const total = totals[pin.layer];
    if (!total) return null;
    const cell = fieldCell(layout, pin.layer, pin.index, total);
    const wide = narrow ? 1 : 0.86;
    return { x: (pin.layer + (1 - wide) / 2) * layout.colW, y: cell.y + cell.h / 2 - bar / 2, w: layout.colW * wide };
  });

  // ---------- the one keyboard stop: a column's button (row -1) or one of its bars
  let cursor = $state<{ layer: number; row: number } | null>(null);
  const lengths = $derived(columns.map(col => col.length));
  /** The tab stop: the last place focused, else the selected unit's bar, else the layer's button. */
  const stop = $derived.by((): { layer: number; row: number } => {
    if (cursor && cursor.layer < count && cursor.row < (lengths[cursor.layer] ?? 0)) return cursor;
    for (let L = 0; L < count; L++) {
      const row = columns[L].findIndex(b => b.id === selected);
      if (row >= 0) return { layer: L, row };
    }
    return { layer: Math.max(0, Math.min(count - 1, layer)), row: -1 };
  });
  const isStop = (L: number, row: number) => stop.layer === L && stop.row === row;

  function onKey(event: KeyboardEvent, L: number, row: number): void {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    const next = atlasStep(lengths, { layer: L, row }, event.key);
    if (!next) return;
    event.preventDefault();
    cursor = next;
    void tick().then(() => plot?.querySelector<HTMLElement>(`[data-at="${next.layer}:${next.row}"]`)?.focus());
  }

  /** All: the cell a bar's neuron is drawn at, ringed while that bar has the keyboard. */
  let focused = $state<{ layer: number; index: number } | null>(null);
  const cursorBox = $derived.by(() => {
    if (!all || !focused || width <= 0) return null;
    const total = totals[focused.layer];
    if (!total) return null;
    const cell = fieldCell(layout, focused.layer, focused.index, total);
    const side = Math.max(3, cell.w, cell.h);
    return { x: cell.x + cell.w / 2 - side / 2, y: cell.y + cell.h / 2 - side / 2, side };
  });
  function focusBar(L: number, row: number, b: AtlasBar): void {
    cursor = { layer: L, row };
    focused = { layer: L, index: b.index };
    if (!all || !plot) return;
    const total = totals[L];
    if (!total) return;
    const cell = fieldCell(layout, L, b.index, total);
    const box = plot.getBoundingClientRect();
    showTip(new DOMRect(box.left + cell.x, box.top + cell.y, Math.max(1, cell.w), Math.max(1, cell.h)), { body: unitTitle(b.id), sub: signedValue(b.value) });
  }

  /** All: the neuron under a point of the page, from where it is in the plot. */
  function cellAt(x: number, y: number): { layer: number; index: number } | null {
    if (!all || !plot) return null;
    const box = plot.getBoundingClientRect();
    return fieldCellAt(layout, x - box.left, y - box.top, totals);
  }
  /** Where the pointer went down. A click reports whole pixels and a pointer event fractions of
   *  one, and a cell can be narrower than a pixel: the click takes the press's point, so it chooses
   *  the cell the hover named. */
  let pressAt: { x: number; y: number } | null = null;
  function pointField(event: PointerEvent): void {
    const hit = cellAt(event.clientX, event.clientY);
    if (!hit || !field || !plot) { hideTip(); return; }
    const cell = fieldCell(layout, hit.layer, hit.index, totals[hit.layer]);
    const box = plot.getBoundingClientRect();
    showTip(new DOMRect(box.left + cell.x, box.top + cell.y, Math.max(1, cell.w), Math.max(1, cell.h)),
      { body: unitTitle(unitRef("neuron", hit.layer, hit.index)), sub: signedValue(field.layers[hit.layer][hit.index] ?? 0) });
  }
  /** A column's own button: its layer, or (All, by pointer) the neuron in the cell under it. */
  function chooseColumn(event: MouseEvent, L: number): void {
    const at = pressAt ?? { x: event.clientX, y: event.clientY };
    pressAt = null;
    const hit = event.detail > 0 ? cellAt(at.x, at.y) : null;
    if (hit) onunit(unitRef("neuron", hit.layer, hit.index), hit.layer);
    else onlayer(L);
  }

  const ends = $derived(Array.from({ length: count }, (_, L) => {
    const on = L === layer;
    return { on, text: on || ((L === 0 || L === count - 1) && Math.abs(L - layer) > 2) ? `L${L}` : "" };
  }));
</script>

<div class="atlas" style:--count={count} role="group" aria-label={label}>
  <div class="plot" class:all class:narrow bind:this={plot} style:height="{plotH}px" style:--bar="{bar}px">
    <span class="thumb" aria-hidden="true" style:transform="translateX({layer * 100}%)"></span>
    {#each columns as col, L (L)}
      <div class="col" style:--c={depth(L)} style:--L={L}>
        <button type="button" class="lay" aria-label={`Layer ${L}`} aria-pressed={L === layer} data-at="{L}:-1" tabindex={isStop(L, -1) ? 0 : -1}
          onclick={(e) => chooseColumn(e, L)} onkeydown={(e) => onKey(e, L, -1)} onfocus={() => { cursor = { layer: L, row: -1 }; focused = null; }}
          onpointerdown={(e) => (pressAt = { x: e.clientX, y: e.clientY })}
          onpointermove={all ? pointField : undefined} onpointerleave={all ? hideTip : undefined}></button>
        {#each col as b, row (b.id)}
          <button type="button" class="u" class:sel={b.id === selected} style:--o={opacity(b.shade, b.id)}
            aria-label={`${unitTitle(b.id)}, ${signedValue(b.value)}`} aria-pressed={b.id === selected} data-at="{L}:{row}" tabindex={isStop(L, row) ? 0 : -1}
            data-tip={all ? undefined : unitTitle(b.id)} data-tip-sub={all ? undefined : signedValue(b.value)}
            onclick={() => onunit(b.id, L)} onkeydown={(e) => onKey(e, L, row)} onfocus={() => focusBar(L, row, b)} onblur={() => { focused = null; if (all) hideTip(); }}><i></i></button>
        {/each}
      </div>
    {/each}
    <canvas class="field" bind:this={canvas} style:height="{plotH}px" aria-hidden="true"></canvas>
    {#if pinBox}<span class="pin" aria-hidden="true" style:width="{pinBox.w}px" style:height="{bar}px" style:transform="translate({pinBox.x}px, {pinBox.y}px)"></span>{/if}
    {#if cursorBox}<span class="cursor" aria-hidden="true" style:width="{cursorBox.side}px" style:height="{cursorBox.side}px" style:transform="translate({cursorBox.x}px, {cursorBox.y}px)"></span>{/if}
  </div>
  <div class="scale" aria-hidden="true">
    {#each ends as t, L (L)}<span class:on={t.on}><b>{t.text}</b></span>{/each}
  </div>
</div>

<style>
  .atlas { display: grid; gap: var(--s1); min-width: 0; user-select: none; -webkit-user-select: none; }
  .plot { position: relative; isolation: isolate; display: grid; grid-template-columns: repeat(var(--count), minmax(0, 1fr)); overflow: clip; transition: height var(--glide) var(--ease); }
  .thumb { position: absolute; left: 0; top: 0; bottom: 0; z-index: -1; width: calc(100% / var(--count)); border-radius: var(--r-col); background: var(--wash); pointer-events: none; transition: transform var(--glide) var(--ease); }
  .col { position: relative; display: grid; align-content: start; min-width: 0; padding: var(--s2) 0; }
  /* The column's own button lies under its bars: the space around them chooses the layer. */
  .lay { position: absolute; inset: 0; padding: 0; border: 0; border-radius: var(--r-col); background: transparent; cursor: pointer; transition: background-color var(--fast) var(--ease); }
  @media (hover: hover) { .col:hover > .lay:not([aria-pressed="true"]) { background: var(--wash); } }
  /* A bar's button is its row of the column: the bar and the gap under it, edge to edge. */
  .u { position: relative; display: flex; width: 100%; justify-content: center; align-items: flex-start; height: calc(var(--bar) + 2px); padding: 0; border: 0; border-radius: 2px; background: transparent; cursor: pointer; }
  .u > i { display: block; width: 62%; height: var(--bar); border-radius: 2px; background: var(--c); opacity: var(--o); transition: width var(--glide) var(--ease), background-color var(--glide) var(--ease), opacity var(--glide) var(--ease); animation: arrive var(--in) var(--ease) backwards; animation-delay: calc(var(--L) * 16ms); transform-origin: left center; }
  @media (hover: hover) { .u:hover > i { width: 86%; opacity: 1; } }
  .u.sel > i { width: 86%; background: var(--ink); opacity: 1; }
  .narrow .u > i { width: 70%; border-radius: 1px; }
  .narrow .u.sel > i, .narrow .u:hover > i { width: 100%; }
  @keyframes arrive { from { opacity: 0; transform: scaleX(0.2); } }
  .field { position: absolute; left: 0; top: 0; display: block; width: 100%; opacity: 0; pointer-events: none; transition: opacity var(--glide) var(--ease); }
  .all .field { opacity: 1; }
  /* All: the cells answer the pointer (through the column's button); a bar keeps the keyboard and
   * rings its neuron's cell instead of itself. */
  .all .u { pointer-events: none; }
  .all .u > i, .all .u.sel > i { opacity: 0; }
  .all .u:focus-visible { outline: none; }
  .pin { position: absolute; left: 0; top: 0; border-radius: 2px; background: var(--ink); pointer-events: none; transition: transform var(--glide) var(--ease), width var(--glide) var(--ease); }
  .cursor { position: absolute; left: 0; top: 0; display: none; border-radius: 2px; pointer-events: none; }
  .plot:has(.u:focus-visible) .cursor { display: block; outline: 2px solid var(--ink); outline-offset: 2px; }
  .scale { display: grid; grid-template-columns: repeat(var(--count), minmax(0, 1fr)); height: 16px; }
  .scale > span { position: relative; }
  .scale b { position: absolute; left: 50%; top: 0; translate: -50% 0; color: var(--ink3); font: var(--type-meta); font-weight: 400; white-space: nowrap; transition: color var(--glide) var(--ease); }
  .scale .on b { color: var(--ink); font-weight: 600; }
  @media (prefers-reduced-motion: reduce) { .u > i { animation: none; } }
  @media (forced-colors: active) { .u > i { background: CanvasText; } .u.sel > i { background: Highlight; } }
</style>
