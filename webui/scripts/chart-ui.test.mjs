import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "vite";

const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { barExtent, barTooltip, chartValue } = await server.ssrLoadModule("/src/lib/charts/chartValues.ts");
  assert.equal(barExtent(-0.5, 1), 0.5);
  assert.equal(barExtent(2, 1), 1);
  for (const [value, max] of [[NaN, 1], [Infinity, 1], [1, 0], [1, -1], [1, Infinity]]) {
    assert.equal(barExtent(value, max), 0);
  }
  assert.equal(barTooltip(-0.5, 1), "-0.5 · 50% of scale (1)");
  assert.equal(barTooltip(2, 1), "2 · 200% of scale (1)", "tooltip preserves overflow despite clipped fill");
  assert.equal(barTooltip(0.125, 1, true), "12.5%");
  assert.equal(barTooltip(NaN, 1), "Not available");
  assert.equal(barTooltip(1, 0), "1 · scale unavailable");
  assert.equal(chartValue(0.000001, true), "<0.01%");

  // The depth ramp and the trace mark's arithmetic.
  const { depthTint, layerDepth, peakLayer, traceBars, readingText, bucketed, aggregateReadingText } = await server.ssrLoadModule("/src/lib/charts/depth.ts");
  // Regression: the Readings list printed raw values ("0" for a jlens/ probability of 2e-6).
  assert.equal(aggregateReadingText("jlens/Paris", 2.4e-6), "<0.1%");
  // Regression: the probe inspector's Geometry head read "L31 · share 20.01" beside a tip defining
  // share as 0 to 1. A layer's fitted weight reads as its part of the reading.
  {
    const { readingWeight } = await server.ssrLoadModule("/src/lib/charts/chartValues.ts");
    assert.equal(readingWeight(20.01, [20.01, 60.03, 20.01]), "20%");
    assert.equal(readingWeight(-1, [1, -1, 2]), "25%");
    assert.equal(readingWeight(1, [0, 0]), "0%");
  }
  // Regression: a probe DLS fitted on a subset of layers (confident.uncertain kept 23 of 32) showed
  // no depth label; the label reads the model's layer count, falling back to a full fit's.
  {
    const { depthLayer } = await server.ssrLoadModule("/src/lib/charts/depth.ts");
    assert.equal(depthLayer(0.434, 32, [3, 4, 5, 9, 30]), 13);
    assert.equal(depthLayer(0.5, null, [0, 1, 2, 3, 4]), 2, "a full fit stands in for the model");
    assert.equal(depthLayer(0.5, null, [3, 4, 5]), null, "a subset fit alone gives no depth");
    assert.equal(depthLayer(null, 32), null);
  }
  assert.equal(aggregateReadingText("slens/Paris", 0.0538), "+0.05");
  assert.equal(aggregateReadingText("default/verbose.concise", -0.31), "−0.31");
  assert.deepEqual(bucketed([1, -3, null, 2, NaN], 3), { values: [-3, 2, null], size: 2 }, "a folded bar keeps its largest magnitude, sign and all");
  assert.deepEqual(bucketed([1, null], 4), { values: [1, null], size: 1 });
  assert.equal(depthTint(0), "color-mix(in oklch, var(--deep) 100%, var(--late))");
  assert.equal(depthTint(1), "color-mix(in oklch, var(--deep) 0%, var(--late))");
  assert.equal(depthTint(0.25), "color-mix(in oklch, var(--deep) 75%, var(--late))");
  assert.equal(depthTint(2), depthTint(1), "a depth past the last layer clamps");
  assert.equal(depthTint(null), null, "no depth keeps the inherited tint");
  assert.equal(depthTint(NaN), null);
  assert.equal(layerDepth(6, 24), 0.25);
  assert.equal(layerDepth(3, 0), null, "a single layer has no depth to place");
  assert.equal(peakLayer({ 2: 0.1, 10: -0.9, 5: 0.4, x: 9 }), 10, "the peak is by magnitude, ignoring non-layer keys");
  assert.equal(peakLayer({}), null);
  const bars = traceBars([0.5, -1, null, 0, NaN], { scale: 1 });
  assert.deepEqual(bars.map((b) => b.height), [50, 100, 0, 0, 0]);
  assert.deepEqual(bars.map((b) => b.negative), [false, true, false, false, false]);
  assert.deepEqual(bars.map((b) => b.missing), [false, false, true, false, true]);
  assert.equal(traceBars([0.001, 1]).at(0).height, 8, "a nonzero reading keeps a sliver so it never reads as missing");
  assert.equal(traceBars([1, 2, 4]).at(-1).height, 100, "without a scale the largest shown fills");
  assert.equal(traceBars(Array.from({ length: 60 }, (_, i) => i), { limit: 32 }).length, 32, "only the most recent readings show");
  assert.equal(traceBars([1, 2, 3], { limit: 0 }).length, 3, "limit 0 shows them all");
  assert.equal(readingText(-0.5), "−0.50");
  assert.equal(readingText(0.5, { signed: true }), "+0.50");
  assert.equal(readingText(-0.001), "0.00", "a reading that rounds to zero drops its sign");
  assert.equal(readingText(0.2345, { percent: true }), "23%");
  assert.equal(readingText(0.0004, { percent: true }), "<0.1%");
  assert.equal(readingText(null), "–");

  const { render } = await server.ssrLoadModule("svelte/server");
  const { default: Trace } = await server.ssrLoadModule("/src/lib/charts/Trace.svelte");
  const trace = render(Trace, { props: { values: [0.5, -0.25, null], scale: 0.5, label: "probe by token", colors: ["var(--deep)", null, null] } }).body;
  assert.match(trace, /aria-label="probe by token"/);
  assert.match(trace, /height: 100%; background: var\(--deep\)/, "a bar takes the colour it is given");
  assert.match(trace, /class="[^"]*\bneg"[^>]*style="height: 50%;"/, "a negative bar keeps faint ink, not a depth colour");
  assert.match(trace, /class="[^"]*\bmissing"/);
  // The By-token readout writes a reading as the rows and the facts do: two decimals, a true minus.
  const { default: TraceDetail } = await server.ssrLoadModule("/src/lib/charts/TraceDetail.svelte");
  const detail = render(TraceDetail, { props: { points: [0.5, -6.808], signed: true } }).body;
  assert.match(detail, /−6\.81/, "a true minus and two decimals");
  assert.doesNotMatch(detail, /-6\.808/);
  assert.match(detail, /aria-valuetext="token 2"/, "the scrubber names the token, as its readout does");
  assert.match(render(TraceDetail, { props: { points: [0.1, 0.5], signed: true } }).body, /\+0\.50/);
  const { default: Bar } = await server.ssrLoadModule("/src/lib/charts/Bar.svelte");
  const negative = render(Bar, { props: { value: -0.5, max: 1, bipolar: true } }).body;
  assert.match(negative, /left: 25%; width: 25%/);
  assert.match(negative, /aria-label="-0.5 · 50% of scale \(1\)"/);
  const invalid = render(Bar, { props: { value: NaN, max: 1 } }).body;
  assert.doesNotMatch(invalid, /NaN/);
  const { default: Sparkline } = await server.ssrLoadModule("/src/lib/charts/Sparkline.svelte");
  const spark = render(Sparkline, { props: { points: [0.1, null, NaN, 0.25], percentage: true } }).body;
  assert.match(spark, /Latest 25% · range 10% to 25% · 2 readings/);
  assert.match(spark, /stroke-linecap="round"/);
  assert.match(render(Sparkline, { props: { points: [] } }).body, /No readings yet/);
  const { default: MiniMap } = await server.ssrLoadModule("/src/panels/manifold/ManifoldMiniMap.svelte");
  const mapProps = {
    info: { name: "test", domain: { type: "box", axes: [{ name: "x", lo: -1, hi: 1 }, { name: "y", lo: -1, hi: 1 }] },
      node_labels: ["node"], node_coords: [[0.25, -0.5]] },
    trajectory: [[0.1, 0.2]], settled: null,
  };
  assert.match(render(MiniMap, { props: mapProps }).body, /node · x 0.25 · y -0.5/);
  assert.match(render(MiniMap, { props: mapProps }).body, /Live coordinates \[0.1, 0.2\]/);
  assert.match(render(MiniMap, { props: { ...mapProps, settled: [0.3, 0.4] } }).body, /Final coordinates \[0.3, 0.4\]/);

  const source = await readFile(new URL("../src/drawers/ProbeInspectorDrawer.svelte", import.meta.url), "utf8");
  assert.match(source, /new ResizeObserver\(redraw\)/);
  assert.match(source, /observer\.observe\(canvas\)/);
  assert.match(source, /observer\.disconnect\(\)/);
  assert.match(source, /aria-pressed=\{l.layer === selectedLayer\}/);
  const barSource = await readFile(new URL("../src/lib/charts/Bar.svelte", import.meta.url), "utf8");
  assert.doesNotMatch(barSource, /preserveAspectRatio/);

  const { renderProbeGeometry, DEFAULT_ORBIT_QUAT, DEFAULT_ORBIT_ZOOM } = await server.ssrLoadModule("/src/lib/charts/probeGeometry.ts");
  const textCalls = [];
  const ctx = new Proxy({
    measureText: (text) => ({ width: text.length * 6 }),
    fillText: (text, x, y) => textCalls.push({ text, x, y }),
  }, { get: (target, key) => target[key] ?? (() => {}) });
  let width = 240, height = 600;
  const canvas = { getContext: () => ctx, getBoundingClientRect: () => ({ width, height }) };
  const oldWindow = globalThis.window, oldStyle = globalThis.getComputedStyle;
  globalThis.window = { devicePixelRatio: 2 };
  globalThis.getComputedStyle = () => ({ getPropertyValue: () => "#888" });
  try {
    const input = {
      geom: { rank: 2, node_white: [[1, 0], [0, 1]], neutral_white: [0, 0], overlay: null },
      nodeLabels: ["right_edge_name", "y"], live: null, trail: [], orbit: { q: DEFAULT_ORBIT_QUAT, zoom: 1 },
    };
    for (const size of [[240, 600], [600, 240], [400, 400]]) {
      [width, height] = size;
      const hits = renderProbeGeometry(canvas, input);
      const label = textCalls.findLast((call) => call.text === "right_edge_name");
      assert.ok(label.x >= 8 && label.x + label.text.length * 6 <= width - 8,
        "node labels move inward before they clip against the chart edge");
      assert.equal(canvas.width, width * 2);
      assert.equal(canvas.height, height * 2);
      const [neutral, x, y] = hits;
      assert.ok(Math.abs((x.screen[0] - neutral.screen[0]) - (neutral.screen[1] - y.screen[1])) < 1e-9,
        "equal data distances have equal pixel lengths at every aspect ratio");
      assert.deepEqual(x.point, [1, 0]);
    }
    for (const rank of [1, 3]) {
      input.geom = { ...input.geom, rank, pca_rotation: [[1, 0, 0], [0, 1, 0]] };
      assert.equal(renderProbeGeometry(canvas, input).length, 3, `rank ${rank} exposes neutral and both nodes`);
    }

    // Wave 3 finding 20: a ring off to one side of neutral (the months manifold) opened zoomed past
    // the fit, so its farthest nodes and their labels fell off the canvas. At the opening zoom, in
    // any orientation, every node and every drawn label fits. Wave 5 (a9's decision 5): the plot
    // pivots on the centre of everything drawn, not on neutral, where the ring sat in one corner.
    const months = ["january", "february", "march", "april", "may", "june", "july", "august",
      "september", "october", "november", "december"];
    const ring = months.map((_, i) => [3 + Math.cos((2 * Math.PI * i) / 12), Math.sin((2 * Math.PI * i) / 12), 0.4 * Math.cos((4 * Math.PI * i) / 12)]);
    const ringInput = {
      geom: { rank: 3, node_white: ring, neutral_white: [0, 0, 0], overlay: null, pca_rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] },
      nodeLabels: months, live: null, trail: [], orbit: { q: DEFAULT_ORBIT_QUAT, zoom: DEFAULT_ORBIT_ZOOM },
    };
    const turns = [DEFAULT_ORBIT_QUAT, [1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0.92, 0, 0.38, 0], [0.7, 0.7, 0, 0], [0.27, 0.65, 0.27, 0.65]];
    for (const size of [[360, 360], [520, 300], [300, 520]]) {
      [width, height] = size;
      for (const q of turns) {
        textCalls.length = 0;
        const hits = renderProbeGeometry(canvas, { ...ringInput, orbit: { q, zoom: DEFAULT_ORBIT_ZOOM } });
        const live = { ...ringInput, live: [8, 0.5, 0], trail: [[6, 0, 0], [7, 0.2, 0]], orbit: { q, zoom: DEFAULT_ORBIT_ZOOM } };
        for (const hit of renderProbeGeometry(canvas, live)) {
          assert.ok(hit.screen[0] >= 6 && hit.screen[0] <= width - 6 && hit.screen[1] >= 6 && hit.screen[1] <= height - 6,
            `with a reading past the ring, ${hit.label} is still inside the ${width}x${height} plot`);
        }
        // The label checks below read this render's text (the reading's render above drew its own).
        textCalls.length = 0;
        renderProbeGeometry(canvas, { ...ringInput, orbit: { q, zoom: DEFAULT_ORBIT_ZOOM } });
        assert.ok(textCalls.some((call) => call.text === "PC2"), "the label checks have the render's text");
        for (const hit of hits) {
          assert.ok(hit.screen[0] >= 6 && hit.screen[0] <= width - 6 && hit.screen[1] >= 6 && hit.screen[1] <= height - 6,
            `${hit.label} is drawn inside the ${width}x${height} plot`);
        }
        for (const call of textCalls) {
          assert.ok(call.x >= 0 && call.x + call.text.length * 6 <= width && call.y - 11 >= 0 && call.y <= height,
            `the label ${call.text} fits the ${width}x${height} plot`);
        }
        // Regression (W5): the neutral ring was drawn over the PC2 name. A triad name sits past
        // its axis's end, clear of every drawn point.
        for (const call of textCalls.filter((c) => /^PC\d$/.test(c.text))) {
          const box = [call.x - 3, call.y - 11 - 3, call.x + call.text.length * 6 + 3, call.y + 3];
          for (const hit of hits) {
            const inside = hit.screen[0] >= box[0] && hit.screen[0] <= box[2] && hit.screen[1] >= box[1] && hit.screen[1] <= box[3];
            assert.ok(!inside, `${hit.label} is not drawn under ${call.text} (${width}x${height})`);
          }
        }
        // Regression (W4): neighbouring nodes' labels overlapped (december, november, january).
        const drawn = textCalls.filter((call) => months.includes(call.text));
        for (const [a, b] of drawn.flatMap((a, i) => drawn.slice(i + 1).map((b) => [a, b]))) {
          const apart = a.x + a.text.length * 6 <= b.x || b.x + b.text.length * 6 <= a.x || a.y - 14 >= b.y || b.y - 14 >= a.y;
          assert.ok(apart, `the labels ${a.text} and ${b.text} do not overlap`);
        }
      }
      // Seen face on, the box's centre (the midpoint of neutral and january, x = 4) is the plot's
      // centre, and the cloud fills the plot (neutral at the centre gave it under half).
      const face = renderProbeGeometry(canvas, { ...ringInput, orbit: { q: [1, 0, 0, 0], zoom: DEFAULT_ORBIT_ZOOM } });
      const neutral = face.find((hit) => hit.label === "neutral");
      const january = face.find((hit) => hit.label === "january");
      assert.ok(Math.abs((neutral.screen[0] + january.screen[0]) / 2 - width / 2) < 1e-9 && Math.abs((neutral.screen[1] + january.screen[1]) / 2 - height / 2) < 1e-9,
        "the pivot is the centre of everything drawn");
      assert.ok(january.screen[0] - neutral.screen[0] > 0.8 * (Math.min(width, height) - 56), "the ring and neutral fill the plot");
      // A reading inside the box moves nothing.
      const inside = renderProbeGeometry(canvas, { ...ringInput, live: [3, 0.5, 0], trail: [[2, 0, 0]], orbit: { q: [1, 0, 0, 0], zoom: DEFAULT_ORBIT_ZOOM } });
      assert.deepEqual(inside.find((hit) => hit.label === "january").screen, january.screen, "a reading inside the box leaves the cloud still");
    }
  } finally {
    if (oldWindow === undefined) delete globalThis.window; else globalThis.window = oldWindow;
    if (oldStyle === undefined) delete globalThis.getComputedStyle; else globalThis.getComputedStyle = oldStyle;
  }
  console.log("Chart values, bars, responsive geometry, and projected hover coordinates passed");
} finally {
  await server.close();
}
