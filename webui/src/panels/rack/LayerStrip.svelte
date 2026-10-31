<script lang="ts">
  // Where a steering term acts: one cell per layer of the model, a bar on the depth ramp as tall
  // as the term's share there. The layers it is fitted at take a selection, drawn solid; the rest
  // are stubs. Press and drag across the cells to paint a range (a stroke adds when it starts on an
  // unselected cell and removes otherwise); Enter or Space toggles one. Every change writes the
  // term's clause (lib/steering/layers.ts): every fitted layer writes none.
  import {
    layerRunsText,
    layerTint,
    paintLayer,
    scopeForSelection,
    selectedGain,
  } from "../../lib/steering/layers";
  import type { LayerScope } from "../../lib/types";

  let {
    nLayers,
    fitted,
    share,
    selected,
    scope,
    onchange,
  }: {
    /** The model's depth. */
    nLayers: number;
    /** The layers the term can act on, ascending. */
    fitted: readonly number[];
    /** Its share at each fitted layer (aligned with ``fitted``). */
    share: readonly number[];
    /** The layers it acts on now. */
    selected: readonly number[];
    /** Its clause as written (``workspace`` shows as the band the server resolved). */
    scope: LayerScope;
    onchange: (scope: LayerScope) => void;
  } = $props();

  const uid = $props.id();
  const top = $derived(Math.max(0, ...share.filter(Number.isFinite)));
  const gain = $derived(selectedGain(fitted, share, selected));
  const cells = $derived(Array.from({ length: nLayers }, (_, L) => {
    const at = fitted.indexOf(L);
    const height = at < 0 ? 0 : top > 0 ? Math.max(8, Math.round(((share[at] ?? 0) / top) * 880) / 10) : 88;
    return { layer: L, fitted: at >= 0, on: selected.includes(L), height, colour: layerTint(L, nLayers) };
  }));
  const allFitted = $derived(scope == null);
  const none = $derived(Array.isArray(scope) && scope.length === 0);
  const summary = $derived(selected.length === 0
    ? "No layers"
    : `L${layerRunsText(selected)} · ${selected.length} ${selected.length === 1 ? "layer" : "layers"}`);

  /** A cell's tip: its layer, and on a selected one the push composition gives it there. */
  function tipFor(layer: number): string {
    const at = gain.get(layer);
    return at === undefined || stroke ? `L${layer}` : `L${layer} · ${at.toFixed(2)}`;
  }

  // A stroke: whether it adds, the selection so far, and the layer it last reached.
  let stroke = $state<{ on: boolean; pick: number[]; last: number } | null>(null);
  const shown = $derived(stroke ? stroke.pick : selected);

  function commit(pick: number[]): void {
    onchange(scopeForSelection(pick, fitted));
  }

  /** The fitted layer under the pointer, by the cells' own boxes. */
  function layerAt(strip: HTMLElement, x: number): number | null {
    for (const cell of strip.querySelectorAll<HTMLElement>("[data-layer]")) {
      const box = cell.getBoundingClientRect();
      if (x >= box.left - 2 && x <= box.right + 2) return Number(cell.dataset.layer);
    }
    return null;
  }

  function begin(event: PointerEvent, layer: number): void {
    if (event.button !== 0) return;
    event.preventDefault();
    (event.currentTarget as HTMLElement).closest<HTMLElement>(".cells")?.setPointerCapture(event.pointerId);
    const on = !selected.includes(layer);
    stroke = { on, pick: paintLayer(selected, layer, on), last: layer };
  }

  function sweep(event: PointerEvent): void {
    if (!stroke) return;
    const layer = layerAt(event.currentTarget as HTMLElement, event.clientX);
    if (layer === null || !fitted.includes(layer) || layer === stroke.last) return;
    // A quick stroke passes cells between two moves: it paints every layer it crossed.
    const [lo, hi] = layer < stroke.last ? [layer, stroke.last] : [stroke.last, layer];
    let pick = stroke.pick;
    for (const L of fitted) if (L >= lo && L <= hi) pick = paintLayer(pick, L, stroke.on);
    stroke = { ...stroke, pick, last: layer };
  }

  function end(): void {
    if (!stroke) return;
    const pick = stroke.pick;
    stroke = null;
    commit(pick);
  }

  function key(event: MouseEvent, layer: number): void {
    // A pointer press already painted; a click with no pointer (Enter, Space) toggles.
    if (event.detail !== 0) return;
    commit(paintLayer(selected, layer, !selected.includes(layer)));
  }
</script>

<div class="where">
  <div class="dw-sec-head">
    <span class="head" id="{uid}-where">Where</span>
    <div class="presets">
      <button type="button" class="dw-text" aria-pressed={allFitted} onclick={() => onchange(null)}>All fitted</button>
      <button type="button" class="dw-text" aria-pressed={scope === "workspace"} onclick={() => onchange("workspace")}>Workspace</button>
      <button type="button" class="dw-text" aria-pressed={none} onclick={() => onchange([])}>Clear</button>
    </div>
  </div>
  <div class="cells" role="group" aria-labelledby="{uid}-where" style:--n={nLayers}
    onpointermove={sweep} onpointerup={end} onpointercancel={end} onlostpointercapture={end}>
    {#each cells as cell (cell.layer)}
      {#if cell.fitted}
        <button type="button" class="cell" data-layer={cell.layer} aria-pressed={shown.includes(cell.layer)}
          aria-label="L{cell.layer}" style:--bar={cell.colour}
          onpointerdown={(event) => begin(event, cell.layer)} onclick={(event) => key(event, cell.layer)}>
          <i style:height="{cell.height}%"></i>
          <b style:bottom="calc({cell.height}% + 6px)">{tipFor(cell.layer)}</b>
        </button>
      {:else}
        <span class="cell is-off" aria-hidden="true"><i></i></span>
      {/if}
    {/each}
  </div>
  <div class="ends meta"><span>L0</span><span class="sum">{summary}</span><span>L{nLayers - 1}</span></div>
</div>

<style>
  .where { display: grid; gap: var(--s2); min-width: 0; }
  .presets { display: flex; flex-wrap: wrap; gap: var(--s2) var(--s4); }
  .cells {
    display: grid;
    grid-template-columns: repeat(var(--n), minmax(0, 1fr));
    gap: 4px;
    height: 120px;
    user-select: none;
    -webkit-user-select: none;
    touch-action: none;
  }
  .cell {
    position: relative;
    display: flex;
    align-items: flex-end;
    min-width: 0;
    height: 100%;
    padding: 0;
    border: 0;
    border-radius: 2px;
    background: transparent;
    cursor: pointer;
  }
  .cell > i {
    display: block;
    width: 100%;
    border-radius: 2px;
    background: var(--bar);
    opacity: 0.24;
    transition: opacity var(--fast) var(--ease), height var(--glide) var(--ease);
  }
  .cell[aria-pressed="true"] > i { opacity: 1; }
  .cell.is-off { cursor: default; }
  .cell.is-off > i { height: 4px; background: var(--wash2); opacity: 1; }
  .cell > b {
    position: absolute;
    left: 50%;
    translate: -50% 0;
    padding: 2px 6px;
    border-radius: var(--r-chip);
    background: var(--paper);
    color: var(--ink);
    font: var(--type-meta);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity var(--fast) var(--ease);
  }
  .cell:focus-visible > b { opacity: 1; }
  @media (hover: hover) {
    .cell[aria-pressed="false"]:hover > i { opacity: 0.48; }
    .cell:hover > b { opacity: 1; }
  }
  .ends { display: grid; grid-template-columns: 1fr auto 1fr; }
  .ends > span:last-child { text-align: right; }
  .sum { color: var(--ink2); }
  @media (max-width: 720px) {
    .cells { gap: 2px; height: 96px; }
    .presets .dw-text { position: relative; }
    .presets .dw-text::after { content: ""; position: absolute; inset: -14px -6px; }
  }
</style>
