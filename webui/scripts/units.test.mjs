import assert from "node:assert/strict";
import { createServer } from "vite";

// Units: the arithmetic of the unit view and the Units room (lib/units.ts), the units store over a
// stub runtime, the recipe's unit terms, and the two views rendered on the server in a catalogued
// model's state (labels, the credit, the Neuron Lens link) and an uncatalogued one's.
const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

/** float16 little-endian base64, as the server packs a population. */
function f16(values) {
  const out = new Uint8Array(values.length * 2);
  values.forEach((v, i) => {
    const sign = v < 0 ? 0x8000 : 0;
    const a = Math.abs(v);
    let h;
    if (a === 0) h = 0;
    else {
      let e = Math.floor(Math.log2(a));
      let m = a / 2 ** e - 1;
      if (e < -14) { h = Math.round(a / 2 ** -24); }
      else { h = ((e + 15) << 10) | Math.round(m * 1024); }
    }
    h |= sign;
    out[2 * i] = h & 0xff;
    out[2 * i + 1] = h >> 8;
  });
  return Buffer.from(out).toString("base64");
}

const CREDIT = "Labels by Claude Haiku 5.5";
const half = (family, weight, lens, tokens, absent = null) => ({
  family, weight, unit: family === "R" ? "probability" : "footprint_z", lens,
  tokens: absent ? [] : tokens.map((token, i) => ({ token_id: 100 + i, token, value: 1 - i / 10 })), absent,
});
const place = (layer, site, r, s) => ({ layer, site, depth: layer / 35, hubs: "hidden", r, s });
const scaleOf = (rate, top) => (rate === null
  ? { source: "neutral", detail: "96 neutral statements", top, value: top * 1.03 }
  : { source: "catalog", detail: "neuron-lens catalog stdlogits/neuron-lens@b5bf11a2b9e3", top, value: top * 1.03 });
const firing = (rate, act_top) => ({
  rate, rate_absent: rate === null ? "no neuron-lens catalog for this model" : null, scale: scaleOf(rate, act_top),
  neutral: { act_top: act_top / 10, mean: 0.1, documents: 96 }, neutral_absent: null, catalog: rate === null ? null : { rate }, catalog_absent: null,
});
function neuronRead({ catalogued }) {
  const label = (fires, url = "https://neuron-lens.com/qwen3-8b/L31/n4410") => (catalogued ? { fires, fires_conf: 0.9, writes: "star words", credit: CREDIT, url } : null);
  return {
    unit: "neuron/L31n4410", kind: "neuron", layer: 31, index: 4410, depth: 31 / 35,
    scalars: { w_norm: 0.82, cos_ab: -0.1, null_share: 0.01, mean_act: 0.1 },
    poles: [
      {
        pole: "+",
        writes: place(31, "out", half("R", 0.9, "R:out", [" stars", " starlight", " Stars", " moonlight"]), half("S", 0.1, "S:out", [" night", " sky"])),
        fires_on: place(31, "mid", half("R", 0.9, "R:mid", [" stars", " bright"]), half("S", 0.1, "S:mid", [], "no S lens at mid")),
        vocab_skew: 1, vocab_kurt: 2, firing: firing(catalogued ? 0.006 : null, 3.4), label: label("night-sky and starlight words"),
      },
      {
        pole: "-",
        writes: place(31, "out", half("R", 0.9, "R:out", [" daylight", " sunshine"]), half("S", 0.1, "S:out", [" sun"])),
        fires_on: place(31, "mid", half("R", 0.9, "R:mid", [" sun"]), half("S", 0.1, "S:mid", [" morning"])),
        vocab_skew: -1, vocab_kurt: 2, firing: firing(catalogued ? 0.004 : null, 2.1), label: label("daylight words", "https://neuron-lens.com/qwen3-8b/L31/n4410?side=minus"),
      },
    ],
    catalog: catalogued
      ? { bound: true, slug: "qwen3-8b", url: "https://neuron-lens.com/qwen3-8b/L31/n4410", credit: CREDIT, reason: null }
      : { bound: false, slug: null, url: null, credit: null, reason: "no neuron-lens catalog for this model" },
  };
}
const headRead = {
  unit: "head/L28h12", kind: "head", layer: 28, index: 12, kv_head: 3, depth: 0.8,
  copies: place(28, "mid", half("R", 0.8, "R:mid", [" stars", " sea"]), half("S", 0.2, "S:mid", [" water"])),
  suppresses: place(28, "mid", half("R", 0.8, "R:mid", [" the"]), half("S", 0.2, "S:mid", [" a"])),
  inputs: {
    query: place(27, "out", half("R", 0.8, "R:out", [" sky"]), half("S", 0.2, "S:out", [" dark"])),
    key: place(27, "out", half("R", 0.8, "R:out", [" stars"]), half("S", 0.2, "S:out", [" lights"])),
    value: place(27, "out", half("R", 0.8, "R:out", [" sea"]), half("S", 0.2, "S:out", [" waves"])),
  },
  copy_score: { R: 0.87, S: 0.4 }, firing: { ...firing(null, 1.6), neutral: { write_top: 1.6, mean_z_norm: 0.4, documents: 96 } }, label: null,
};

try {
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  let catalogued = true;
  let flip = false;
  const asked = [];
  const tokens = [" Above", " the", " stars", " hung", "."].map((token, raw_index) => ({ raw_index, token_id: 200 + raw_index, token }));
  installRuntimeClient({
    mode: "http",
    sessions: {
      // A unit acts at its own layer (the composer's term_layers), which the term editor's strip shows.
      validateSteering: async (expression) => {
        const unit = /(neuron|head)\/L(\d+)[nh]\d+/u.exec(expression);
        if (!unit) return { valid: true, expression, error: null };
        const layer = Number(unit[2]);
        return { valid: true, expression, error: null, n_layers: 36, terms: [{ key: unit[0], fitted: [layer], share: [1], layers: [layer] }] };
      },
    },
    profiles: {}, manifolds: {}, templates: {}, tree: {}, events: {},
    probes: {
      attach: async (req) => ({ family: "units", name: req.selector, unit: req.selector, kind: "neuron", layer: 31, index: 4410, layers: [31], intrinsic_dim: 1, feature_space: "units" }),
      detach: async () => ({}),
      list: async () => ({ probes: [] }),
    },
    instruments: {
      unitRead: async (unit, options) => { asked.push(["read", unit, options.hideHubs]); return unit.startsWith("head/") ? headRead : neuronRead({ catalogued }); },
      unitTop: async (token) => ({ token_id: 7, token, by: "writes", hits: [
        { unit: "neuron/L31n4410", layer: 31, index: 4410, pole: "+", score: 7.9, r: { z: 7.9, weight: 1, lens: "R:out" }, s: { z: 0, weight: 0, lens: "S:out" }, label: catalogued ? { fires: "night-sky and starlight words", credit: CREDIT, url: "u" } : null },
        { unit: "neuron/L24n9127", layer: 24, index: 9127, pole: "+", score: 6.8, r: { z: 6.8, weight: 1, lens: "R:out" }, s: { z: 0, weight: 0, lens: "S:out" }, label: null },
      ], layers: {} }),
      unitCatalog: async () => (catalogued
        ? { model_id: "Qwen/Qwen3-8B", available: true, bound: true, slug: "qwen3-8b", repo: "stdlogits/neuron-lens", revision: "r", credit: CREDIT, reason: null, binding: { revision: "r", fetched_at: null, files: 37 } }
        : { model_id: "HuggingFaceTB/SmolLM2-360M-Instruct", available: false, bound: false, slug: null, repo: "stdlogits/neuron-lens", revision: "r", credit: null, reason: "no neuron-lens catalog for this model", binding: null }),
      unitActivity: async (nodeId, options) => {
        asked.push(["activity", nodeId, options.top, options.units ?? null, options.rawIndex ?? null]);
        // ``flip``: the strongest neuron fired on its − pole here.
        const sign = flip ? -1 : 1;
        const traces = {};
        for (const unit of options.units ?? []) traces[unit] = [0.1, 0, 3.9, 1.1, -0.3].map(v => v * sign);
        const layers = Array.from({ length: 4 }, (_, layer) => ({
          layer, n_neurons: 8, n_heads: 2,
          neurons: f16(Array.from({ length: 8 }, (_, i) => (i === 3 && layer === 3 ? 3.5 : (i - 4) / 8))),
          heads: f16([0.5, layer / 4]),
        }));
        return {
          node_id: nodeId, steering: null, unit: "activation_over_scale", scale: { source: "catalog", detail: "neuron-lens catalog stdlogits/neuron-lens@b5bf11a2b9e3" }, tokens, sink_raw_index: null,
          top: [
            { unit: "neuron/L3n3", kind: "neuron", layer: 3, index: 3, peak: 3.9 * sign, raw_index: 2, label: catalogued ? { fires: flip ? "daylight words" : "night-sky and starlight words", credit: CREDIT, url: "u" } : null },
            { unit: "head/L2h1", kind: "head", layer: 2, index: 1, peak: 3.1, raw_index: 2 },
            { unit: "neuron/L1n6", kind: "neuron", layer: 1, index: 6, peak: -2.2, raw_index: 4 },
          ],
          traces,
          population: options.rawIndex === undefined ? null : { raw_index: options.rawIndex, encoding: "float16-le-base64", layers },
        };
      },
    },
    dispose: async () => {},
  });

  const units = await server.ssrLoadModule("/src/lib/units.ts");
  const {
    parseUnit, unitTitle, lensText, orderedHalves, mergedTokens, withAgreement, unitFacts, labelText, scaleText,
    firesOn, peakIndex, decodeFloat16, decileCuts, shadeOf, strongestColumns, fieldGeometry, parseQuery,
    peakPole,
  } = units;
  const { unitTermProblem } = await server.ssrLoadModule("/src/lib/unitTerms.ts");

  // IDs: every spelling a person types or a route names, one reference.
  for (const text of ["neuron/L31n4410", "L31 n4410", "l31n4410", "L31/n4410", " L31  n4410 "]) {
    assert.deepEqual(parseUnit(text), { kind: "neuron", layer: 31, index: 4410, id: "neuron/L31n4410" }, text);
  }
  assert.equal(parseUnit("head/L28h12").id, "head/L28h12");
  assert.equal(parseUnit("L28 h12").kind, "head");
  assert.equal(parseUnit("neuron/L28h12"), null, "a neuron reference names a neuron");
  assert.equal(parseUnit("stars"), null);
  assert.equal(unitTitle("neuron/L31n4410"), "L31 n4410");
  assert.equal(unitTitle("head/L28h12"), "L28 h12");

  // Lenses: the depth's family first, the other faded; an absent primary yields.
  assert.equal(lensText({ lens: "R:out", family: "R" }), "R · out");
  assert.equal(lensText({ lens: "logit", family: "R" }), "logit");
  const read = neuronRead({ catalogued: true });
  assert.equal(orderedHalves(read.poles[0].writes).primary.family, "R");
  assert.equal(orderedHalves({ r: half("R", 0.2, "R:out", ["a"]), s: half("S", 0.8, "S:out", ["b"]) }).primary.family, "S");
  assert.equal(orderedHalves({ r: half("R", 0.2, "R:out", ["a"]), s: half("S", 0.8, "S:out", [], "no S lens") }).primary.family, "R");
  assert.equal(orderedHalves({ r: half("R", 0.5, "R:out", ["a"]), s: half("S", 0.5, "S:out", ["b"]) }).primary.family, "R", "R on a tie");

  // Variants merge under their stem, counted; a token two groups name agrees.
  assert.deepEqual(mergedTokens(read.poles[0].writes.r.tokens).map(t => [t.token, t.count]), [[" stars", 2], [" starlight", 1], [" moonlight", 1]]);
  const [w, f] = withAgreement([read.poles[0].writes.r.tokens, read.poles[0].fires_on.r.tokens]);
  assert.deepEqual(w.filter(t => t.agree).map(t => t.token), [" stars"]);
  assert.deepEqual(f.map(t => [t.token, t.agree]), [[" stars", true], [" bright", false]]);

  // Facts: the shown pole's typical firing and the catalog's rate; a head's typical write and copy score.
  // The typical firing is the reading scale readings divide by (act_top under the soft floor), named.
  const facts = ({ scale, ...rest }) => ({ ...rest, source: scale?.source ?? null });
  assert.deepEqual(facts(unitFacts(read, "+")), { typical: 3.4 * 1.03, rate: 0.006, writeNorm: 0.82, copy: null, source: "catalog" });
  assert.deepEqual(facts(unitFacts(read, "-")), { typical: 2.1 * 1.03, rate: 0.004, writeNorm: 0.82, copy: null, source: "catalog" });
  // Regression: it showed act_top, so a pole that never fired on the neutral text read "typical
  // firing 0" beside a reading of −36 (which divides by the floor, 0.14).
  const unfired = neuronRead({ catalogued: false });
  unfired.poles[1].firing.scale = { source: "neutral", detail: "96 neutral statements", top: 0, value: 0.141602 };
  assert.equal(unitFacts(unfired, "-").typical, 0.141602);
  assert.equal(unitFacts(neuronRead({ catalogued: false }), "+").rate, null, "no catalog, no rate");
  assert.equal(unitFacts(neuronRead({ catalogued: false }), "+").scale.source, "neutral");
  assert.deepEqual(facts(unitFacts(headRead)), { typical: 1.6 * 1.03, rate: null, writeNorm: null, copy: 0.87, source: "neutral" });
  assert.equal(scaleText("neutral"), "typical firing on the neutral text");
  assert.equal(scaleText("catalog"), "typical firing in text (Neuron Lens)");
  assert.equal(scaleText(null), null);
  assert.equal(labelText(read.poles[0].label), "night-sky and starlight words");
  assert.equal(labelText({ writes: " star words ", credit: CREDIT, url: "u" }), "star words");
  assert.equal(labelText(null), null);

  // Activity: firing counts from a quarter of the typical high firing, either pole; the peak by magnitude.
  assert.equal(firesOn([0.1, 0.3, -0.5, null, 0.24]), 2);
  assert.equal(peakIndex([0.1, -4, 3.9, null]), 1);
  assert.equal(peakIndex([0, null]), -1);
  // The side that fired: the peak's sign (the unit view and the room's summary read the same one).
  assert.equal(peakPole([0.1, -4, 3.9, null]), "-");
  assert.equal(peakPole([0.1, 4, -3.9]), "+");
  assert.equal(peakPole([0, null]), "+", "a silent trace reads its + side");

  assert.deepEqual([...decodeFloat16(f16([1, -2, 0.5, 0, 65504, 3.5]))], [1, -2, 0.5, 0, 65504, 3.5]);
  assert.ok(Math.abs(decodeFloat16(f16([0.1]))[0] - 0.1) < 1e-3);
  const cuts = decileCuts(Float32Array.from({ length: 100 }, (_, i) => i - 50));
  assert.equal(cuts.length, 9);
  assert.equal(shadeOf(0, cuts), 0);
  assert.equal(shadeOf(-50, cuts), 9, "a shade is by magnitude");
  assert.deepEqual(decileCuts([]), new Array(9).fill(Infinity));

  // The strongest of each layer, sorted, shaded by rank among every bar shown.
  const cols = strongestColumns([
    { unit: "a", layer: 0, index: 1, peak: 1 }, { unit: "b", layer: 0, index: 2, peak: -5 },
    { unit: "c", layer: 1, index: 3, peak: 3 }, { unit: "d", layer: 1, index: 4, peak: 2 }, { unit: "e", layer: 7, index: 0, peak: 9 },
  ], 2, 1);
  assert.deepEqual(cols.map(col => col.map(bar => bar.id)), [["b"], ["c"]], "one a layer, the strongest; an out-of-range layer drops");
  assert.deepEqual(cols.map(col => col.map(bar => bar.shade)), [[5], [0]]);
  const g = fieldGeometry(20, 100, 2560);
  assert.ok(g.g * g.rows >= 2560 && g.cw * g.g <= 20.0001 && g.ch * g.rows <= 100.0001);

  // Regression (a9, the Units atlas): a click on a row or a pixel chose only the whole layer. A
  // point of All's field names the neuron drawn there (the layout the atlas paints with), a bar is
  // its unit's button, and the atlas is one keyboard stop moved by the arrows.
  {
    const { fieldLayout, fieldCell, fieldCellAt, unitRef, signedValue, atlasStep } = units;
    const layout = fieldLayout(320, 214, 4, false);
    const totals = [2560, 2560, 0, 2560];
    for (const [layer, index] of [[0, 0], [1, 17], [3, 2559], [1, 1300]]) {
      const cell = fieldCell(layout, layer, index, totals[layer]);
      assert.deepEqual(fieldCellAt(layout, cell.x + cell.w / 2, cell.y + cell.h / 2, totals), { layer, index }, `cell ${layer}:${index}`);
    }
    assert.equal(fieldCellAt(layout, 2, 100, totals), null, "the column's margin beside the cells chooses no neuron");
    assert.equal(fieldCellAt(layout, 200, 100, totals), null, "a layer with no population has no cells");
    assert.equal(fieldCellAt(layout, 40, 3, totals), null, "above the cells");
    assert.equal(unitRef("neuron", 12, 344), "neuron/L12n344");
    assert.equal(unitRef("head", 20, 7), "head/L20h7");
    assert.equal(signedValue(-15.2), "−15");
    const lengths = [3, 0, 2];
    assert.deepEqual(atlasStep(lengths, { layer: 0, row: -1 }, "ArrowDown"), { layer: 0, row: 0 }, "from the layer into its bars");
    assert.deepEqual(atlasStep(lengths, { layer: 0, row: 0 }, "ArrowUp"), { layer: 0, row: -1 }, "back up to the layer");
    assert.deepEqual(atlasStep(lengths, { layer: 0, row: 2 }, "ArrowDown"), { layer: 0, row: 2 }, "the last bar holds");
    assert.deepEqual(atlasStep(lengths, { layer: 0, row: 2 }, "ArrowRight"), { layer: 1, row: -1 }, "an empty column: its layer");
    assert.deepEqual(atlasStep(lengths, { layer: 0, row: 2 }, "End"), { layer: 2, row: 1 }, "a shorter column: its last bar");
    assert.deepEqual(atlasStep(lengths, { layer: 2, row: 1 }, "Home"), { layer: 0, row: 1 }, "the same rank");
    assert.equal(atlasStep(lengths, { layer: 0, row: 0 }, "Enter"), null, "Enter is the button's own");
  }

  // The search: nothing, an ID (or a malformed one), a word (a · or a space asks for the word start).
  assert.deepEqual(parseQuery("  "), { mode: "active" });
  assert.equal(parseQuery("L31 n4410").ref.id, "neuron/L31n4410");
  assert.deepEqual(parseQuery("L31 n"), { mode: "id", ref: null, text: "L31 n" });
  assert.deepEqual(parseQuery("stars"), { mode: "word", word: "stars" });
  assert.deepEqual(parseQuery("·stars"), { mode: "word", word: " stars" });

  // The recipe: a unit holds one term; push ±, silence, toggled off; serialized in the grammar.
  const store = await server.ssrLoadModule("/src/lib/stores/units.svelte.ts");
  const steering = await server.ssrLoadModule("/src/lib/stores/steering.svelte.ts");
  const { serializeExpression } = await server.ssrLoadModule("/src/lib/expression.ts");
  store.toggleUnitTerm("neuron/L31n4410", "plus");
  assert.equal(store.unitTerm("neuron/L31n4410"), "plus");
  assert.equal(steering.currentSteeringExpression(), "0.2 neuron/L31n4410");
  steering.patchSteerEntry("neuron/L31n4410", { alpha: 0.35 }, ["unit"]);
  store.toggleUnitTerm("neuron/L31n4410", "minus");
  assert.equal(steering.currentSteeringExpression(), "-0.35 neuron/L31n4410", "the − pole keeps the strength");
  store.toggleUnitTerm("head/L28h12", "silence");
  assert.equal(steering.currentSteeringExpression(), "-0.35 neuron/L31n4410 + !head/L28h12");
  store.toggleUnitTerm("neuron/L31n4410", "minus");
  assert.equal(store.unitTerm("neuron/L31n4410"), null, "pressing the held term takes it out");
  assert.equal(steering.currentSteeringExpression(), "!head/L28h12");
  assert.equal(serializeExpression(new Map([["neuron/L8n3", { mode: "unit", ablate: true, alpha: 0.5, trigger: "RESPONSE", enabled: true }]])), "0.5 !neuron/L8n3@response");
  store.toggleUnitTerm("head/L28h12", "silence");
  assert.equal(steering.steerRack.entries.size, 0);

  // The term card's push/ablate switch: an ablation runs at α 1 (the neutral mean), never at the
  // push's signed α (−0.2 would amplify the unit's deviation); a push comes back at the default.
  store.toggleUnitTerm("neuron/L30n637", "minus");
  assert.equal(steering.currentSteeringExpression(), "-0.2 neuron/L30n637");
  store.setUnitOperation("neuron/L30n637", true);
  assert.deepEqual([steering.steerRack.entries.get("neuron/L30n637").ablate, steering.steerRack.entries.get("neuron/L30n637").alpha], [true, 1]);
  assert.equal(steering.currentSteeringExpression(), "!neuron/L30n637");
  assert.equal(store.unitTerm("neuron/L30n637"), "silence");
  steering.patchSteerEntry("neuron/L30n637", { alpha: 0.5 }, ["unit"]);
  store.setUnitOperation("neuron/L30n637", true);
  assert.equal(steering.currentSteeringExpression(), "0.5 !neuron/L30n637", "ablating an ablation keeps its blend");
  store.setUnitOperation("neuron/L30n637", false);
  assert.equal(steering.currentSteeringExpression(), "0.2 neuron/L30n637");
  store.toggleUnitTerm("head/L28h12", "silence");
  store.setUnitOperation("head/L28h12", false);
  assert.equal(store.unitTerm("head/L28h12"), "silence", "a head has no push");
  steering.removeSteerEntry("neuron/L30n637");
  steering.removeSteerEntry("head/L28h12");
  assert.equal(steering.steerRack.entries.size, 0);

  // One rule for a unit row (the snapshot validator's and WebMCP's): a neuron pushed either pole or
  // ablated, a head only ablated, an ablation's α never negative.
  assert.equal(unitTermProblem("neuron/L31n4410", { ablate: false, alpha: -0.2 }), null);
  assert.equal(unitTermProblem("neuron/L31n4410", { ablate: true, alpha: 1 }), null);
  assert.equal(unitTermProblem("neuron/L31n4410", { ablate: true, alpha: 0 }), null);
  assert.match(unitTermProblem("neuron/L31n4410", { ablate: true, alpha: -0.2 }), /must not be negative/);
  assert.match(unitTermProblem("head/L28h12", { ablate: false, alpha: 0.2 }), /no push/);
  assert.equal(unitTermProblem("head/L28h12", { ablate: true, alpha: 1 }), null);
  assert.match(unitTermProblem("neuron/L031n4", { ablate: true, alpha: 1 }), /names no unit/);
  assert.match(unitTermProblem("neuron/L3n4", { ablate: "yes", alpha: 1 }), /ablate/);
  assert.match(unitTermProblem("neuron/L3n4", { ablate: true, alpha: Infinity }), /finite/);

  // Probes: a unit attaches by its reference and lists in the rack.
  const probes = await server.ssrLoadModule("/src/lib/stores/probes.svelte.ts");
  await store.toggleUnitProbe("neuron/L31n4410");
  assert.equal(store.unitProbed("neuron/L31n4410"), true);
  assert.deepEqual(probes.probeRequestFromInfo(probes.probeRack.entries.get("neuron/L31n4410").info), { selector: "neuron/L31n4410", name: "neuron/L31n4410" });
  await store.toggleUnitProbe("neuron/L31n4410");
  assert.equal(store.unitProbed("neuron/L31n4410"), false);

  // Rooms: Units shows only where the runtime reads units.
  const rooms = await server.ssrLoadModule("/src/lib/stores/rooms.svelte.ts");
  assert.ok(rooms.shownRooms().some(room => room.value === "units"));
  // The rooms in order; the Manifolds room's tools are the Artifacts room's sections (one spelling,
  // no alias), beside its panels (the unit catalog, lenses, SAEs). A section's value names its room.
  assert.deepEqual(rooms.ROOMS.map(room => room.value), ["workbench", "units", "lens", "probes", "artifacts", "model"]);
  assert.deepEqual(rooms.ROOM_SECTIONS.artifacts.map(row => row.value), ["units", "lenses", "build", "packs", "combine", "templates", "roles", "saes"]);
  assert.deepEqual(rooms.ROOM_SECTIONS.artifacts.filter(row => row.drawer === null).map(row => row.value), ["units", "lenses", "saes"]);
  assert.equal(rooms.ROOM_SECTIONS.manifolds, undefined);
  assert.deepEqual(rooms.roomFor("manifold_pack"), { room: "artifacts", section: "packs" });
  const sectionValues = Object.values(rooms.ROOM_SECTIONS).flatMap(rows => rows.map(row => row.value));
  assert.equal(new Set(sectionValues).size, sectionValues.length);

  // The views on the server: a catalogued model's state, then an uncatalogued one's.
  const { render } = await server.ssrLoadModule("svelte/server");
  const { chatLog } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const { default: UnitView } = await server.ssrLoadModule("/src/panels/inspector/UnitView.svelte");
  const { default: UnitsRoom } = await server.ssrLoadModule("/src/panels/rooms/UnitsRoom.svelte");
  const settle = () => new Promise(resolve => setTimeout(resolve, 0));
  // The Artifacts room draws a panel section (here the unit catalog) under its section tabs.
  const { default: ArtifactsRoom } = await server.ssrLoadModule("/src/panels/rooms/ArtifactsRoom.svelte");
  rooms.goto("artifacts", "units");
  const artifactsHtml = render(ArtifactsRoom, {}).body;
  for (const label of ["Units", "Lenses", "Build", "Packs", "SAEs"]) assert.match(artifactsHtml, new RegExp(`>${label}<`));
  assert.match(artifactsHtml, /Unit catalog/);
  assert.match(artifactsHtml, />Read units</);
  rooms.goto("workbench");
  chatLog.turns = [
    { role: "user", text: "Write one sentence." },
    { role: "assistant", text: "Above the stars hung.", generated: true, nodeId: "n1", tokens: tokens.map(t => ({ text: t.token, rawIndex: t.raw_index })) },
  ];

  await store.loadCatalog();
  store.requestUnitRead("neuron/L31n4410", true);
  store.requestNodeActivity("n1", { units: ["neuron/L31n4410"], top: 1 });
  await settle();
  assert.equal(store.unitLabelFor("neuron/L31n4410"), "night-sky and starlight words");
  let html = render(UnitView, { props: { id: "neuron/L31n4410" } }).body;
  assert.match(html, /<h2 class="dw-title[^"]*">L31 n4410<\/h2>/);
  assert.match(html, /night-sky and starlight words/);
  assert.match(html, new RegExp(CREDIT));
  assert.match(html, /href="https:\/\/neuron-lens\.com\/qwen3-8b\/L31\/n4410"[^>]*>Open in Neuron Lens/);
  assert.match(html, /typical firing <b[^>]*>3\.5<\/b>/, "the reading scale (act_top 3.4 under the floor)");
  assert.match(html, /fires on <b[^>]*>0\.6%<\/b> of tokens/);
  assert.match(html, /write norm <b[^>]*>0\.82<\/b>/);
  assert.match(html, />R · out</);
  assert.match(html, />R · mid</);
  assert.match(html, /On this conversation/);
  assert.match(html, /fires on 2 of 5 tokens/, "the shown side fires");
  assert.match(html, /Push \+/);
  assert.match(html, /Silence/);
  assert.doesNotMatch(html, /no S lens at mid/, "an absent faded half stays out");

  // The unit's term in the recipe's term editor: the one rule (lib/unitTerms.ts), the catalog label
  // of the pole it pushes, a way to its unit view, and its own layer on a fixed strip.
  const { default: TermEditor } = await server.ssrLoadModule("/src/panels/rack/TermEditor.svelte");
  const { sessionState: session } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
  session.info = { model_id: "Qwen/Qwen3-8B" };
  const editor = (name, entry) => render(TermEditor, { props: { name, entry, onclose: () => {}, onremove: () => {} } }).body;
  const push = { mode: "unit", ablate: false, alpha: 0.2, trigger: "BOTH", enabled: true };
  editor("neuron/L31n4410", push);
  await settle();
  await settle();
  html = editor("neuron/L31n4410", push);
  assert.match(html, /MLP neuron · L31 · night-sky and starlight words/, "the facts carry the catalog label");
  assert.match(html, />Open L31 n4410</, "the editor opens the unit view");
  assert.match(html, />Push</);
  assert.match(html, />its own layer</, "a unit's strip shows its layer, fixed");
  assert.doesNotMatch(html, />All fitted</, "no clause control on a unit");
  assert.equal((html.match(/class="cell is-fixed[^"]*\bis-on\b/g) ?? []).length, 1, "one layer lit");
  const text = (markup) => markup.replace(/<[^>]*>/g, "");
  assert.match(text(html), /0\.2 neuron\/L31n4410\s*Copy/, "the term as the expression writes it");
  const silenced = { mode: "unit", ablate: true, alpha: 1, trigger: "BOTH", enabled: true };
  editor("head/L28h12", silenced);
  await settle();
  await settle();
  html = editor("head/L28h12", silenced);
  assert.match(html, /attention head · L28/);
  assert.doesNotMatch(html, />Push</, "a head has no push");
  assert.match(html, /All of it/, "an ablation at 1 is exact mean-ablation");
  assert.match(text(html), /!head\/L28h12\s*Copy/);
  session.info = null;

  store.requestUnitRead("head/L28h12", true);
  await settle();
  html = render(UnitView, { props: { id: "head/L28h12" } }).body;
  for (const title of ["Copies", "Suppresses", "Query", "Key", "Value"]) assert.match(html, new RegExp(`>${title}<`));
  assert.match(html, /copy score <b[^>]*>0\.87<\/b>/);
  assert.doesNotMatch(html, /Push/, "a head has no push");
  assert.doesNotMatch(html, /\+ side/);

  // The room reads at the token cursor (lib/stores/tokenCursor): with no token selected, the whole
  // reply, each unit at its peak over it; at a token, the strongest units there (its population).
  const selection = await server.ssrLoadModule("/src/lib/stores/selection.svelte.ts");
  const { default: TokenStrip } = await server.ssrLoadModule("/src/panels/rooms/TokenStrip.svelte");
  store.requestNodeActivity("n1", { top: 1024 });
  store.requestUnitRead("neuron/L3n3", true);
  store.requestNodeActivity("n1", { units: ["neuron/L3n3"], top: 1 });
  await settle();
  html = render(UnitsRoom, {}).body;
  assert.match(html, /aria-pressed="true"[^>]*>Whole reply</, "no token selected: the whole reply");
  assert.match(html, /Most active in this reply/);
  assert.match(html, />L3 n3</);
  assert.match(html, /−2\.2/, "a unit at its peak over the reply");
  assert.match(html, /This reply, at each unit.s peak/);
  assert.ok(asked.some(([kind, , top, , raw]) => kind === "activity" && top === 1024 && raw === null), "whole, no population");
  // The strip: the reply's tokens on one line, the whole reply lit, its seat opening it.
  let strip = render(TokenStrip, {}).body;
  assert.equal((strip.match(/class="t [^"]*\blit\b/g) ?? []).length, 5);
  assert.match(strip, /class="seat[^"]*"[^>]*>assistant</);
  // The workbench selects the reply's last token: the cursor is there, and the strip marks it.
  selection.select({ kind: "token", params: { turnIdx: 1, tokenIdx: 4 } });
  strip = render(TokenStrip, {}).body;
  assert.match(strip, /class="t [^"]*\bon\b[^"]*"[^>]*aria-label="\. \(token 5\)"/);
  assert.match(strip, /token 5 of 5/);
  store.requestNodeActivity("n1", { top: 1024, rawIndex: 4 });
  await settle();
  html = render(UnitsRoom, {}).body;
  assert.match(html, /Most active here/);
  assert.match(html, /32 neurons/);
  // Regression (a9): each of the atlas's bars is its unit's button, labelled and tipped with its ID
  // and value, pressed when it is the unit shown; one place in the atlas takes the Tab key.
  {
    const atlas = html.slice(html.indexOf('aria-label="Layers"'), html.indexOf('aria-label="Results"'));
    assert.match(atlas, /<button[^>]*class="u\b[^"]*\bsel\b[^"]*"[^>]*aria-label="L3 n3, 3\.5"[^>]*aria-pressed="true"[^>]*data-tip="L3 n3"[^>]*data-tip-sub="3\.5"/, "the selected unit's bar, its value at the token");
    assert.ok((atlas.match(/<button[^>]*class="u\b/g) ?? []).length >= 2, "every bar a button");
    assert.match(atlas, /<button[^>]*class="lay[^"]*"[^>]*aria-label="Layer 3"[^>]*aria-pressed="true"/, "the layer the unit is on");
    assert.equal((atlas.match(/tabindex="0"/g) ?? []).length, 1, "one keyboard stop");
    assert.match(atlas, /class="u\b[^"]*\bsel\b[^"]*"[^>]*tabindex="0"/, "on the selected unit's bar");
    // Live, All's hover named L13 n675 and the click chose n674: a click reports whole pixels, a
    // pointer event fractions, and a cell is narrower than a pixel. The click takes the press's point.
    const source = (await import("node:fs")).readFileSync(new URL("../src/panels/rooms/UnitAtlas.svelte", import.meta.url), "utf8");
    assert.match(source, /onpointerdown=\{\(e\) => \(pressAt = \{ x: e\.clientX, y: e\.clientY \}\)\}/);
    assert.match(source, /const at = pressAt \?\? \{ x: event\.clientX, y: event\.clientY \};[^]*?cellAt\(at\.x, at\.y\)/);
  }
  assert.match(html, />L3 n3</);
  assert.doesNotMatch(html, />L2 h1</, "neurons list neurons");
  assert.doesNotMatch(html, /−2\.2/, "at the token, not at the peak");
  assert.match(html, />Labelled</, "a bound catalog offers the labelled units");
  assert.match(html, /Values over each unit&#39;s typical firing in text \(Neuron Lens\)|Values over each unit.s typical firing in text \(Neuron Lens\)/, "the room names the scale source");
  assert.match(html, new RegExp(CREDIT));
  assert.match(html, /the 40 most active of 8 a layer/);
  assert.match(html, /L3 · 6 of 8 active here/);
  assert.match(html, /Open in workbench/);
  let summary = html.slice(html.indexOf('aria-label="Selected unit"'));
  assert.match(summary, /peak here <b[^>]*>3\.9<\/b>/);
  assert.match(summary, />at [^]*?<b[^>]*>−0\.30<\/b>/, "its value at the cursor's token");
  assert.match(summary, /night-sky and starlight words/);
  assert.match(summary, /moonlight/, "the + side's writes");
  assert.equal(store.unitLabelFor("neuron/L3n3", "-"), "daylight words", "labels are kept per pole");

  // A neuron that fired on its − pole: the summary reads that side (its writes, label, link and
  // rate), the side the unit view opens on, and the row and the summary wear the one label.
  flip = true;
  selection.clearSelection();
  store.forgetUnits();
  await store.loadCatalog();
  store.requestNodeActivity("n1", { top: 1024 });
  store.requestUnitRead("neuron/L3n3", true);
  store.requestNodeActivity("n1", { units: ["neuron/L3n3"], top: 1 });
  await settle();
  html = render(UnitsRoom, {}).body;
  summary = html.slice(html.indexOf('aria-label="Selected unit"'));
  assert.match(summary, /peak here <b[^>]*>−3\.9<\/b>/);
  assert.match(summary, /daylight words/);
  assert.doesNotMatch(summary, /night-sky/);
  assert.match(summary, /sunshine/, "the − side's writes");
  assert.doesNotMatch(summary, /moonlight/);
  assert.match(summary, /href="https:\/\/neuron-lens\.com\/qwen3-8b\/L31\/n4410\?side=minus"/);
  assert.match(summary, /fires on <b[^>]*>0\.4%<\/b> of tokens/);
  assert.doesNotMatch(html.slice(0, html.indexOf('aria-label="Selected unit"')), /night-sky/, "the row wears the − side's label");
  flip = false;

  // Uncatalogued: no label, no credit, no link, no Labelled filter.
  catalogued = false;
  store.forgetUnits();
  await store.loadCatalog();
  store.requestUnitRead("neuron/L31n4410", true);
  store.requestNodeActivity("n1", { top: 1024, rawIndex: 4 });
  await settle();
  html = render(UnitView, { props: { id: "neuron/L31n4410" } }).body;
  assert.doesNotMatch(html, new RegExp(CREDIT));
  assert.doesNotMatch(html, /Neuron Lens/);
  assert.doesNotMatch(html, /fires on <b/, "no catalog, no rate");
  assert.match(html, /typical firing <b[^>]*>3\.5<\/b>/);
  html = render(UnitsRoom, {}).body;
  assert.doesNotMatch(html, />Labelled</);
  assert.doesNotMatch(html, new RegExp(CREDIT));
  assert.ok(asked.some(([kind, , top, , raw]) => kind === "activity" && top === 1024 && raw === 4), "the atlas reads one token's population");

  // WebMCP's rack edit holds the same rule: switching a unit to ablation takes α 1 unless alpha says
  // otherwise, and an ablation at a negative α (or a head push) is refused.
  const { sessionState } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
  const { loomTree } = await server.ssrLoadModule("/src/lib/stores/loom.svelte.ts");
  const { createWorkspaceTools } = await server.ssrLoadModule("/src/lib/webmcp/workspaceTools.ts");
  sessionState.info = { model_id: "HuggingFaceTB/SmolLM2-360M-Instruct" };
  loomTree.loaded = true;
  const editRack = createWorkspaceTools().find(t => t.name === "drowse_edit_steer_rack");
  const context = { signal: new AbortController().signal, jobs: { list: () => [] } };
  const edit = (input) => editRack.execute({ action: "upsert", ...input }, context);
  await edit({ selector: "neuron/L30n637", alpha: -0.2 });
  assert.equal(steering.currentSteeringExpression(), "-0.2 neuron/L30n637");
  await edit({ selector: "neuron/L30n637", ablate: true });
  assert.equal(steering.currentSteeringExpression(), "!neuron/L30n637", "an ablation runs at α 1");
  await assert.rejects(edit({ selector: "neuron/L30n637", alpha: -0.2 }), /must not be negative/);
  await assert.rejects(edit({ selector: "neuron/L30n637", ablate: true, alpha: -0.5 }), /must not be negative/);
  await assert.rejects(edit({ selector: "head/L28h12", ablate: false }), /no push/);
  await edit({ selector: "neuron/L30n637", ablate: false });
  assert.equal(steering.currentSteeringExpression(), "0.2 neuron/L30n637", "a push comes back at the default");
  await edit({ selector: "neuron/L30n637", ablate: true, alpha: 0.5 });
  assert.equal(steering.currentSteeringExpression(), "0.5 !neuron/L30n637", "an explicit alpha wins");

  // Wave 3 (Qwen, 375): the boot skeleton is fixed over the room, and the Units room drawn under it
  // showed its Neurons/Heads and Strongest/All pills through the blocks. The room hides (still laid
  // out, so a gliding thumb measures) and is inert until boot ends.
  const { readFileSync } = await import("node:fs");
  const app = readFileSync(new URL("../src/App.svelte", import.meta.url), "utf8");
  assert.match(app, /class="rooms-host" class:waiting=\{bootStatus !== "ready"\} inert=\{[^}]*bootStatus !== "ready"\}/);
  assert.match(app, /\.rooms-host\.waiting > \.room \{ visibility: hidden; opacity: 0; \}/);
  // Wave 4: the skeleton faded out over the Model room's content. It leaves on the frame the room
  // arrives, and the room fades in.
  assert.match(app, /<div class="booting"[^>]*in:fadeIn>/);
  assert.doesNotMatch(app, /<div class="booting"[^>]*out:/);
  // Wave 4: the atlas head read "At to", the token run into the sentence. A token in the Units
  // room's prose is a Ledger Token chip, as everywhere else in the dashboard.
  const room = readFileSync(new URL("../src/panels/rooms/UnitsRoom.svelte", import.meta.url), "utf8");
  assert.match(room, /At <Token token=\{focusToken\.token\} \/>/);
  assert.doesNotMatch(room, /@render ws\(/, "no token is spliced into the prose as bare text");
  // Wave 5: a facts row with a Token chip ("peak here −64 on [augh]") lifted its plain sibling
  // ("write norm 5.7") 4px off the baseline; a facts row aligns its items on the baseline.
  const shared = readFileSync(new URL("../src/lib/style/drowse.css", import.meta.url), "utf8");
  assert.match(/^\.dw-facts \{([^}]*)\}/m.exec(shared)?.[1] ?? "", /align-items: baseline/);

  console.log("units: ids, reads, activity, atlas, search, recipe, probes, WebMCP and both catalog states");
} finally {
  await server.close();
}
