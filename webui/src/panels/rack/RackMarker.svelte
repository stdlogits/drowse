<script lang="ts">
  // One optical marker system for every rack card. Unicode shape glyphs
  // have wildly different intrinsic boxes, so a shared outer box plus
  // per-shape optical compensation keeps circles, diamonds, squares, and
  // triangles the same apparent size and baseline across pillar switches.

  export type RackMarkerShape = "circle" | "diamond" | "square" | "triangle";

  let {
    shape,
    filled = false,
  }: {
    shape: RackMarkerShape;
    filled?: boolean;
  } = $props();

  const glyph = $derived.by(() => {
    if (shape === "circle") return filled ? "●" : "○";
    if (shape === "diamond") return filled ? "◆" : "◇";
    if (shape === "triangle") return filled ? "▲" : "△";
    return filled ? "■" : "□";
  });
</script>

<span class="marker {shape}" aria-hidden="true">{glyph}</span>

<style>
  .marker {
    display: inline-grid;
    place-items: center;
    inline-size: 18px;
    block-size: 18px;
    flex: 0 0 18px;
    font: 500 12px/1 var(--text);
    text-align: center;
  }
  /* Optical, not layout, compensation: the four shapes keep one 18 px box while their visible
     ink reads equally large. */
  .triangle {
    transform: translateY(-0.25px);
  }
</style>
