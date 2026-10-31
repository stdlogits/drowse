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
const firing = (rate, act_top) => ({
  rate, rate_absent: rate === null ? "no neuron-lens catalog for this model" : null,
  neutral: { act_top, mean: 0.1, documents: 96 }, neutral_absent: null, catalog: rate === null ? null : { rate }, catalog_absent: null,
});
function neuronRead({ catalogued }) {
  const label = (fires) => (catalogued ? { fires, fires_conf: 0.9, writes: "star words", credit: CREDIT, url: "https://neuron-lens.com/qwen3-8b/L31/n4410" } : null);
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
        vocab_skew: -1, vocab_kurt: 2, firing: firing(catalogued ? 0.004 : null, 2.1), label: label("daylight words"),
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
  const asked = [];
  const tokens = [" Above", " the", " stars", " hung", "."].map((token, raw_index) => ({ raw_index, token_id: 200 + raw_index, token }));
  installRuntimeClient({
    mode: "http",
    sessions: {}, profiles: {}, manifolds: {}, templates: {}, tree: {}, events: {},
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
        const traces = {};
        for (const unit of options.units ?? []) traces[unit] = [0.1, 0, 3.9, 1.1, -0.3];
        const layers = Array.from({ length: 4 }, (_, layer) => ({
          layer, n_neurons: 8, n_heads: 2,
          neurons: f16(Array.from({ length: 8 }, (_, i) => (i === 3 && layer === 3 ? 3.5 : (i - 4) / 8))),
          heads: f16([0.5, layer / 4]),
        }));
        return {
          node_id: nodeId, steering: null, unit: "activation_over_scale", tokens, sink_raw_index: null,
          top: [
            { unit: "neuron/L3n3", kind: "neuron", layer: 3, index: 3, peak: 3.9, raw_index: 2, label: catalogued ? { fires: "night-sky and starlight words", credit: CREDIT, url: "u" } : null },
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
    parseUnit, unitTitle, lensText, orderedHalves, mergedTokens, withAgreement, unitFacts, labelText,
    firesOn, peakIndex, decodeFloat16, decileCuts, shadeOf, strongestColumns, fieldGeometry, parseQuery, unitTermExpression,
  } = units;

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
  assert.deepEqual(unitFacts(read, "+"), { typical: 3.4, rate: 0.006, writeNorm: 0.82, copy: null });
  assert.deepEqual(unitFacts(read, "-"), { typical: 2.1, rate: 0.004, writeNorm: 0.82, copy: null });
  assert.equal(unitFacts(neuronRead({ catalogued: false }), "+").rate, null, "no catalog, no rate");
  assert.deepEqual(unitFacts(headRead), { typical: 1.6, rate: null, writeNorm: null, copy: 0.87 });
  assert.equal(labelText(read.poles[0].label), "night-sky and starlight words");
  assert.equal(labelText({ writes: " star words ", credit: CREDIT, url: "u" }), "star words");
  assert.equal(labelText(null), null);

  // Activity: firing counts from a quarter of the typical high firing, either pole; the peak by magnitude.
  assert.equal(firesOn([0.1, 0.3, -0.5, null, 0.24]), 2);
  assert.equal(peakIndex([0.1, -4, 3.9, null]), 1);
  assert.equal(peakIndex([0, null]), -1);
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

  // The search: nothing, an ID (or a malformed one), a word (a · or a space asks for the word start).
  assert.deepEqual(parseQuery("  "), { mode: "active" });
  assert.equal(parseQuery("L31 n4410").ref.id, "neuron/L31n4410");
  assert.deepEqual(parseQuery("L31 n"), { mode: "id", ref: null, text: "L31 n" });
  assert.deepEqual(parseQuery("stars"), { mode: "word", word: "stars" });
  assert.deepEqual(parseQuery("·stars"), { mode: "word", word: " stars" });
  assert.equal(unitTermExpression("neuron/L31n4410", "plus"), "0.2 neuron/L31n4410");
  assert.equal(unitTermExpression("neuron/L31n4410", "minus", 0.3), "-0.3 neuron/L31n4410");
  assert.equal(unitTermExpression("head/L28h12", "silence"), "!head/L28h12");
  assert.throws(() => unitTermExpression("head/L28h12", "plus"), /no push/);

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

  // The views on the server: a catalogued model's state, then an uncatalogued one's.
  const { render } = await server.ssrLoadModule("svelte/server");
  const { chatLog } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const { default: UnitView } = await server.ssrLoadModule("/src/panels/inspector/UnitView.svelte");
  const { default: UnitsRoom } = await server.ssrLoadModule("/src/panels/rooms/UnitsRoom.svelte");
  const settle = () => new Promise(resolve => setTimeout(resolve, 0));
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
  assert.match(html, /typical firing <b[^>]*>3\.4<\/b>/);
  assert.match(html, /fires on <b[^>]*>0\.6%<\/b> of tokens/);
  assert.match(html, /write norm <b[^>]*>0\.82<\/b>/);
  assert.match(html, />R · out</);
  assert.match(html, />R · mid</);
  assert.match(html, /On this conversation/);
  assert.match(html, /fires on 2 of 5 tokens/, "the shown side fires");
  assert.match(html, /Push \+/);
  assert.match(html, /Silence/);
  assert.doesNotMatch(html, /no S lens at mid/, "an absent faded half stays out");

  store.requestUnitRead("head/L28h12", true);
  await settle();
  html = render(UnitView, { props: { id: "head/L28h12" } }).body;
  for (const title of ["Copies", "Suppresses", "Query", "Key", "Value"]) assert.match(html, new RegExp(`>${title}<`));
  assert.match(html, /copy score <b[^>]*>0\.87<\/b>/);
  assert.doesNotMatch(html, /Push/, "a head has no push");
  assert.doesNotMatch(html, /\+ side/);

  store.requestNodeActivity("n1", { top: 1024, rawIndex: 4 });
  store.requestUnitRead("neuron/L3n3", true);
  store.requestNodeActivity("n1", { units: ["neuron/L3n3"], top: 1 });
  await settle();
  html = render(UnitsRoom, {}).body;
  assert.match(html, /Most active here/);
  assert.match(html, /32 neurons/);
  assert.match(html, />L3 n3</);
  assert.doesNotMatch(html, />L2 h1</, "neurons list neurons");
  assert.match(html, /−2\.2/);
  assert.match(html, />Labelled</, "a bound catalog offers the labelled units");
  assert.match(html, new RegExp(CREDIT));
  assert.match(html, /the 40 most active of 8 a layer/);
  assert.match(html, /L3 · 6 of 8 active here/);
  assert.match(html, /Open in workbench/);

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
  assert.match(html, /typical firing <b[^>]*>3\.4<\/b>/);
  html = render(UnitsRoom, {}).body;
  assert.doesNotMatch(html, />Labelled</);
  assert.doesNotMatch(html, new RegExp(CREDIT));
  assert.ok(asked.some(([kind, , top, , raw]) => kind === "activity" && top === 1024 && raw === 4), "the atlas reads one token's population");

  console.log("units: ids, reads, activity, atlas, search, recipe, probes and both catalog states");
} finally {
  await server.close();
}
