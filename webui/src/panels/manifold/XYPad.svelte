<script lang="ts">
  import { MorphText } from "@stdlogits/ledger";
  // Manifold position picker.  One slider per intrinsic dimension,
  // uniformly: dim 1 -> one slider, dim 2 -> two sliders, dim N -> N
  // sliders.  (The old 2D draggable xy-pad special case was retired in
  // favor of this consistent control across every manifold rank.)
  //
  // Periodic axes wrap rather than clamp: a value that runs off one
  // edge re-enters at the other.  Each axis carries its own [lo, hi]
  // range; the parent owns the canonical ``coords`` array and receives
  // updates through ``onchange``.

  import type { ManifoldInfo } from "../../lib/types";
  import Slider from "../../lib/Slider.svelte";
  import { quotientDescription } from "../../lib/manifolds/surfaceGeometry";

  interface Props {
    manifold: ManifoldInfo;
    /** Current authoring coordinates — one per intrinsic dimension. */
    coords: number[];
    onchange: (coords: number[]) => void;
    /** When true, the sliders display the coords but reject input —
     *  used by ``ManifoldSteerCard`` for label-form terms, where the
     *  engine takes the position from the named node and the surfaced
     *  coords would be ignored.  Keeping the controls visible (instead
     *  of hiding them) gives the user a read-out of where the snapped
     *  node actually sits in the manifold. */
    locked?: boolean;
  }

  let { manifold, coords, onchange, locked = false }: Props = $props();

  interface AxisRange {
    name: string;
    lo: number;
    hi: number;
    periodic: boolean;
  }

  // Per-axis range + label.  Box domains carry explicit axes; sphere /
  // custom domains have no authored bounds, so we derive each axis as
  // a symmetric ``[-R, R]`` around 0 where ``R = max(1, ceil(max|v|))``
  // over the per-axis node coords.  Centering on 0 lines up the slider
  // midpoint with the (0, 0, ...) centroid where every freshly-racked
  // manifold term starts, and ``ceil(max-magnitude)`` keeps the range a
  // clean whole number that still envelops every node (a persona node
  // at ``c0 = -10`` lands at exactly the left endpoint with ``R = 10``).
  // The ``max(1, ...)`` floor keeps the slider usable when every node
  // sits very close to 0 on a particular axis.  Falls back to ``[-1, 1]``
  // when no fitted coords are available (the unfitted-manifold pre-fit
  // state).
  const axes = $derived.by<AxisRange[]>(() => {
    const d = manifold.domain;
    if (d.type === "klein") return [
      { name: "u (rad)", lo: 0, hi: 2 * Math.PI, periodic: false },
      { name: "v (rad)", lo: 0, hi: 2 * Math.PI, periodic: false },
    ];
    if (d.type === "projective") return [
      { name: "Polar (rad)", lo: 0, hi: Math.PI, periodic: false },
      { name: "Azimuth (rad)", lo: 0, hi: 2 * Math.PI, periodic: false },
    ];
    if (d.type === "box") {
      return d.axes.map((a) => ({
        name: a.name,
        lo: a.lo,
        hi: a.hi,
        periodic: a.periodic,
      }));
    }
    const n = manifold.intrinsic_dim;
    const nodeCoords = manifold.node_coords ?? [];
    return Array.from({ length: n }, (_, i) => {
      let r = 1;
      if (nodeCoords.length > 0) {
        let mag = 0;
        for (const row of nodeCoords) {
          const v = row?.[i];
          if (Number.isFinite(v)) {
            const a = Math.abs(v);
            if (a > mag) mag = a;
          }
        }
        if (mag > 0) r = Math.max(1, Math.ceil(mag));
      }
      return { name: `c${i}`, lo: -r, hi: r, periodic: false };
    });
  });

  /** Clamp (open axis) or wrap (periodic axis) a raw coordinate into
   *  the axis range. */
  function fit(axis: AxisRange, raw: number): number {
    const span = axis.hi - axis.lo;
    if (span <= 0) return axis.lo;
    if (axis.periodic) {
      let v = (raw - axis.lo) % span;
      if (v < 0) v += span;
      return axis.lo + v;
    }
    return Math.min(axis.hi, Math.max(axis.lo, raw));
  }

  function setCoord(idx: number, raw: number): void {
    const next = coords.slice();
    next[idx] = fit(axes[idx], raw);
    onchange(next);
  }

  // Two axes also draw as a plane on a wash: the nodes in faint ink and the term's position as a
  // tint dot, which a press or drag moves (the sliders stay the keyboard's way in).
  const PAD = 120;
  const unit = (axis: AxisRange, v: number) => axis.hi > axis.lo ? Math.min(1, Math.max(0, (v - axis.lo) / (axis.hi - axis.lo))) : 0;
  const plane = $derived(axes.length === 2);
  const planeNodes = $derived(plane ? (manifold.node_coords ?? []).flatMap((row, i) =>
    Array.isArray(row) && Number.isFinite(row[0]) && Number.isFinite(row[1])
      ? [{ key: i, x: unit(axes[0], row[0]) * PAD, y: (1 - unit(axes[1], row[1])) * PAD }]
      : []) : []);
  const here = $derived(plane ? { x: unit(axes[0], coords[0] ?? axes[0].lo) * PAD, y: (1 - unit(axes[1], coords[1] ?? axes[1].lo)) * PAD } : null);
  let dragging = false;
  function place(event: PointerEvent): void {
    const rect = (event.currentTarget as SVGElement).getBoundingClientRect();
    const fx = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const fy = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
    onchange([
      fit(axes[0], axes[0].lo + fx * (axes[0].hi - axes[0].lo)),
      fit(axes[1], axes[1].lo + (1 - fy) * (axes[1].hi - axes[1].lo)),
      ...coords.slice(2),
    ]);
  }

  function fmt(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "0.00";
  }
</script>

<div class="xypad" class:locked>
  {#if quotientDescription(manifold.domain)}
    <p>{quotientDescription(manifold.domain)}</p>
  {/if}
  {#if plane && here}
    <svg class="plane" class:live={!locked} width={PAD} height={PAD} viewBox="0 0 {PAD} {PAD}" aria-hidden="true"
      onpointerdown={(event) => { if (locked) return; dragging = true; (event.currentTarget as SVGElement).setPointerCapture(event.pointerId); place(event); }}
      onpointermove={(event) => { if (dragging) place(event); }}
      onpointerup={() => (dragging = false)}
      onpointercancel={() => (dragging = false)}>
      <line class="cross" x1={unit(axes[0], 0) * PAD} x2={unit(axes[0], 0) * PAD} y1="0" y2={PAD} />
      <line class="cross" y1={(1 - unit(axes[1], 0)) * PAD} y2={(1 - unit(axes[1], 0)) * PAD} x1="0" x2={PAD} />
      {#each planeNodes as node (node.key)}<circle class="node" cx={node.x} cy={node.y} r="2.5" />{/each}
      <circle class="here" cx={here.x} cy={here.y} r="5" />
    </svg>
  {/if}
  <div class="sliders">
    {#each axes as axis, i (i)}
      <label class="axis-row">
        <span class="axis-name">
          {axis.name}{axis.periodic ? " ↻" : ""}
        </span>
        <Slider
          value={coords[i] ?? axis.lo}
          displayValue={fmt(coords[i] ?? axis.lo)}
          min={axis.lo}
          max={axis.hi}
          step={(axis.hi - axis.lo) / 100 || 0.01}
          oninput={(v) => setCoord(i, v)}
          ariaLabel="{axis.name} coordinate"
          disabled={locked}
        />
        <span class="axis-val"><MorphText text={fmt(coords[i] ?? axis.lo)} /></span>
      </label>
    {/each}
  </div>
</div>

<style>
  .xypad { display: grid; gap: var(--s2); min-width: 0; }
  .xypad p { margin: 0; color: var(--ink3); font: var(--type-meta); }
  /* The disabled prop already dims each Slider's input; also dim the
   * surrounding axis name and readout so the whole row reads locked. */
  .xypad.locked .sliders { opacity: 0.7; }
  .sliders { display: grid; gap: var(--s3); }
  /* It sits in a steer card, a wash that steps up to wash2 under the pointer, so the plane is paper:
   * the one fill that differs from the card in both states. The origin's cross is wash2 on it. */
  .plane { border-radius: var(--r-box); background: var(--paper); touch-action: none; }
  .plane.live { cursor: crosshair; }
  .cross { stroke: var(--wash2); stroke-width: 1; }
  .node { fill: var(--ink3); }
  .here { fill: var(--tint); stroke: var(--paper); stroke-width: 2; transition: cx var(--fast) var(--ease), cy var(--fast) var(--ease); }
  .axis-row { display: grid; grid-template-columns: minmax(40px, auto) minmax(0, 1fr) 48px; align-items: center; gap: var(--s3); min-width: 0; }
  .axis-name { color: var(--ink3); font: var(--type-meta); }
  .axis-val { color: var(--ink2); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: end; }
</style>
