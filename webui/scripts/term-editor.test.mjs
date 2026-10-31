import assert from "node:assert/strict";
import { createServer } from "vite";

// The term editor's logic: the layer clause and the depth strip's arithmetic, the expression it
// writes, unit terms, where a term acts as the server reports it, the chips' words, and the
// snapshot rows that persist a scoped or unit term.
const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
try {
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  // validateSteering answers from a table the blocks below fill; every asked text is recorded.
  const asked = [];
  const answers = new Map();
  installRuntimeClient({
    mode: "http",
    profiles: { list: async () => ({ profiles: [] }), correlation: async () => null },
    sessions: {
      validateSteering: async (expression) => {
        asked.push(expression);
        const answer = answers.get(expression);
        if (answer instanceof Error) throw answer;
        return answer ?? { valid: true, expression, error: null };
      },
    },
  });

  const layers = await server.ssrLoadModule("/src/lib/steering/layers.ts");
  const { layerRuns, formatLayerIndices, layerRunsText, layerClause, termIsInert, scopeForSelection, paintLayer, depthSpan, peakLayer, selectedGain, layerTint } = layers;

  // ---- the clause and the strip's arithmetic
  assert.deepEqual(layerRuns([20, 4, 5, 6, 7, 8, 8]), [[4, 8], [20, 20]]);
  assert.equal(formatLayerIndices([5, 4]), "4-5", "a run of two is a range, as the grammar writes it");
  assert.equal(formatLayerIndices([20, 4, 6, 5]), "4-6,20");
  assert.equal(layerRunsText([14, 15, 16, 31]), "14–16, 31");
  assert.equal(layerClause(null), "");
  assert.equal(layerClause(undefined), "");
  assert.equal(layerClause("workspace"), "{workspace}");
  assert.equal(layerClause([8, 4, 5, 6, 7, 20]), "{4-8,20}");
  assert.equal(layerClause([]), "");
  assert.ok(termIsInert([]) && !termIsInert(null) && !termIsInert("workspace") && !termIsInert([3]));
  assert.equal(scopeForSelection([3, 1, 2], [1, 2, 3]), null, "every fitted layer writes no clause");
  assert.deepEqual(scopeForSelection([3, 1], [1, 2, 3]), [1, 3]);
  assert.deepEqual(scopeForSelection([], [1, 2]), []);
  assert.deepEqual(paintLayer([1, 3], 2, true), [1, 2, 3]);
  assert.deepEqual(paintLayer([1, 3], 3, false), [1]);
  assert.deepEqual(paintLayer([1, 3], 3, true), [1, 3], "painting a layer already on changes nothing");
  assert.equal(layerTint(0, 36), "color-mix(in oklch, var(--deep) 100%, var(--late))");
  assert.equal(layerTint(35, 36), "color-mix(in oklch, var(--deep) 0%, var(--late))");
  const span = depthSpan([14, 15, 24], 36);
  assert.deepEqual([span.left, span.width], [38.9, 30.6], "the canvas's wonder chip: L14–24 of 36");
  assert.match(span.background, /^linear-gradient\(90deg, color-mix\(in oklch, var\(--deep\) 60%, var\(--late\)\), color-mix\(in oklch, var\(--deep\) 31%, var\(--late\)\)\)$/);
  assert.equal(depthSpan([31], 36).background, layerTint(31, 36), "one layer is one colour");
  assert.equal(depthSpan([], 36), null);
  assert.equal(peakLayer([1, 2, 3], [0.5, 2, 0.5]), 2);
  assert.equal(peakLayer([1, 2, 3], [0.5, 2, 0.5], [1, 3]), 1, "the peak among the selected layers");
  const gain = selectedGain([1, 2, 3], [0.5, 1.5, 1], [1, 2]);
  assert.deepEqual([...gain.entries()], [[1, 0.5], [2, 1.5]], "mean 1 over the selection, as composition weighs it");

  // ---- the expression a scoped term writes
  const { serializeExpression, formatTerm } = await server.ssrLoadModule("/src/lib/expression.ts");
  const rack = new Map([
    ["default/honest", { mode: "subspace", ablate: false, coords: [1], label: "honest", variant: "raw", trigger: "BOTH", enabled: true, layers: [12, 13, 14, 20] }],
    ["jlens/storm", { mode: "jlens", ablate: true, alpha: 1, trigger: "BOTH", enabled: true, layers: [20, 21, 22] }],
    ["sae/7", { mode: "sae", ablate: false, alpha: 0.3, trigger: "BOTH", enabled: true, layers: [] }],
    ["neuron/L31n4410", { mode: "unit", ablate: false, alpha: -0.2, trigger: "RESPONSE", enabled: true }],
    ["head/L28h12", { mode: "unit", ablate: false, alpha: 1, trigger: "BOTH", enabled: true }],
    ["default/emotions", { mode: "manifold", blend: 0.35, onto: 0, coords: [0], label: "wonder", variant: "raw", trigger: "RESPONSE", enabled: true, layers: "workspace" }],
  ]);
  assert.equal(serializeExpression(rack, 0.5),
    "0.5 default/honest%honest{12-14,20} + 1 !jlens/storm{20-22} - 0.2 neuron/L31n4410@response + 1 !head/L28h12 + 0.35 default/emotions%wonder{workspace}@response",
    "the clause follows the selector and precedes the trigger; a term at no layer sits out; a unit takes no clause and a head is only silenced");
  assert.equal(formatTerm("sae/7", rack.get("sae/7"), 0.5), "0.3 sae/7", "on its own, an inert term writes no clause");

  // ---- rack edits: the clause and units
  const steering = await server.ssrLoadModule("/src/lib/stores/steering.svelte.ts");
  const { steerRack, setSteerLayers, addUnitToRack, unitKey, addJLensToRack, currentSteeringExpression, removeSteerEntry } = steering;
  assert.equal(unitKey("NEURON/l08N0344"), "neuron/L8n344", "a unit's key is the grammar's canonical form");
  assert.equal(unitKey("head/L20h17"), "head/L20h17");
  assert.equal(unitKey("neuron/L8h3"), null, "the letter must match the family");
  assert.equal(unitKey("neuron/8"), null);
  assert.equal(addUnitToRack("head/L2h0", { ablate: false }), "head/L2h0");
  assert.deepEqual(steerRack.entries.get("head/L2h0"), { mode: "unit", ablate: true, alpha: 1, trigger: "BOTH", enabled: true }, "a head is silenced, fully by default");
  addUnitToRack("neuron/L1n3");
  assert.deepEqual(steerRack.entries.get("neuron/L1n3"), { mode: "unit", ablate: false, alpha: 0.2, trigger: "BOTH", enabled: true });
  addJLensToRack("storm");
  setSteerLayers("jlens/storm", [4, 5]);
  setSteerLayers("neuron/L1n3", [1]);
  assert.equal(steerRack.entries.get("neuron/L1n3").layers, undefined, "a unit takes no clause");
  assert.equal(currentSteeringExpression(), "0.3 jlens/storm{4-5} + 1 !head/L2h0 + 0.2 neuron/L1n3");
  setSteerLayers("jlens/storm", null);
  assert.equal(currentSteeringExpression(), "0.3 jlens/storm + 1 !head/L2h0 + 0.2 neuron/L1n3");
  for (const name of ["jlens/storm", "neuron/L1n3", "head/L2h0"]) removeSteerEntry(name);

  // ---- where a term acts, as the server says
  const { placeOf, layersProbe, forgetTermLayers } = await server.ssrLoadModule("/src/lib/steering/termLayers.svelte.ts");
  const entry = { mode: "manifold", blend: 0.35, onto: 0.2, coords: [0], label: "wonder", variant: "raw", trigger: "BEFORE", enabled: true, layers: [14, 15, 16] };
  const unscoped = layersProbe("default/emotions", entry, false);
  const scoped = layersProbe("default/emotions", entry, true);
  assert.equal(unscoped, "1 default/emotions%wonder@response", "a probe steers @response at strength 1, so it never drops the cached prefix");
  assert.equal(scoped, "1 default/emotions%wonder{14-16}@response");
  const where = (fitted, share, picked, n = 36) => ({ valid: true, expression: "", error: null, n_layers: n, terms: [{ key: "k", fitted, share, layers: picked }] });
  const fitted = [4, 5, 14, 15, 16, 31];
  answers.set(unscoped, where(fitted, [0.5, 0.5, 1.5, 1.5, 1.5, 0.5], fitted));
  answers.set(scoped, where(fitted, [0.5, 0.5, 1.5, 1.5, 1.5, 0.5], [14, 15, 16]));
  assert.equal(placeOf("default/emotions", entry, "m").status, "loading");
  await tick(); await tick();
  let place = placeOf("default/emotions", entry, "m");
  assert.equal(place.status, "ready");
  assert.deepEqual(place.where.fitted, fitted);
  assert.deepEqual(place.selected, [14, 15, 16], "while the scoped term is asked, the clause's own layers show");
  assert.equal(place.where.nLayers, 36);
  await tick(); await tick();
  assert.deepEqual(asked, [unscoped, scoped], "the extent first, then the clause");
  assert.deepEqual(placeOf("default/emotions", entry, "m").selected, [14, 15, 16]);
  const before = asked.length;
  placeOf("default/emotions", { ...entry, blend: -0.6, trigger: "AFTER" }, "m");
  await tick();
  assert.equal(asked.length, before, "a strength or a trigger asks nothing new");

  const workspace = { ...entry, layers: "workspace" };
  answers.set(layersProbe("default/emotions", workspace, true), where(fitted, [1, 1, 1, 1, 1, 1], [14, 15, 16, 31]));
  placeOf("default/emotions", workspace, "m");
  await tick(); await tick();
  assert.deepEqual(placeOf("default/emotions", workspace, "m").selected, [14, 15, 16, 31], "the workspace band is the server's");

  const refused = { ...entry, layers: [2] };
  answers.set(layersProbe("default/emotions", refused, true), { valid: false, expression: "", error: "steering term 'x{2}' selects layer(s) 2 it is not fitted at" });
  placeOf("default/emotions", refused, "m");
  await tick(); await tick();
  place = placeOf("default/emotions", refused, "m");
  assert.equal(place.status, "ready", "the strip still draws the term's extent");
  assert.deepEqual(place.selected, []);
  assert.match(place.error, /not fitted at/);

  const other = { mode: "jlens", ablate: false, alpha: 0.3, trigger: "BOTH", enabled: true };
  answers.set(layersProbe("jlens/sea", other, false), { valid: true, expression: "1 jlens/sea@response", error: null });
  placeOf("jlens/sea", other, "m");
  await tick(); await tick();
  assert.equal(placeOf("jlens/sea", other, "m").status, "unavailable", "a runtime without layers hides the strip");
  assert.equal(placeOf("jlens/sea", other, "another-model").status, "loading", "another model asks again");
  answers.set(layersProbe("jlens/sky", other, false), new Error("offline"));
  placeOf("jlens/sky", other, "m");
  await tick(); await tick();
  assert.equal(placeOf("jlens/sky", other, "m").status, "invalid");
  forgetTermLayers();
  assert.equal(placeOf("default/emotions", entry, "m").status, "loading", "a catalog change forgets every answer");
  await tick(); await tick();

  // ---- what a chip and the editor say
  const terms = await server.ssrLoadModule("/src/lib/steering/terms.ts");
  assert.equal(terms.coefficientText(-0.2), "−0.2");
  assert.equal(terms.coefficientText(0.35, { fixed: true }), "0.35");
  assert.equal(terms.coefficientText(-0.004, { fixed: true }), "0.00");
  assert.equal(terms.chipStrength("jlens/storm", { mode: "jlens", ablate: true, alpha: 1 }, 0.5), "without");
  assert.equal(terms.chipStrength("head/L2h0", { mode: "unit", ablate: false, alpha: 1 }, 0.5), "without");
  assert.equal(terms.chipStrength("default/honest", { mode: "subspace", ablate: false }, 0.5), "0.5", "a flat term shows the shared strength");
  assert.equal(terms.termTitle("neuron/L31n4410", { mode: "unit" }), "L31 n4410");
  assert.equal(terms.unitLayer("head/L28h12"), 28);
  assert.equal(terms.triggerWord("AFTER"), "after thinking");
  assert.equal(terms.triggerWord("BOTH"), "");
  assert.deepEqual(terms.strengthDial("default/emotions", entry, 0.5), { value: 0.35, min: -1, max: 1, step: 0.05, shared: false });
  assert.equal(terms.strengthDial("default/honest", { mode: "subspace" }, 0.7).shared, true);
  assert.deepEqual([terms.strengthDial("jlens/storm", { mode: "jlens", ablate: true, alpha: 0.4 }, 0.5).min, terms.strengthSays("jlens/storm", { mode: "jlens", ablate: true, alpha: 0.4 }, 0.4)], [0, "40% of it"]);
  assert.equal(terms.strengthSays("jlens/storm", { mode: "jlens", ablate: false }, -0.3), "Away from storm");
  assert.ok(terms.canRemove("neuron/L1n3", { mode: "unit" }, null) && !terms.canRemove("head/L1h3", { mode: "unit" }, null));
  assert.ok(!terms.canRemove("default/emotions", entry, { node_count: 20 }), "a curved position cannot be removed");
  assert.ok(terms.canRemove("default/honest", { mode: "subspace" }, { node_count: 2 }));

  // ---- snapshot rows keep a clause and a unit
  const { validateConversationSnapshot } = await server.ssrLoadModule("/src/lib/conversationSnapshot.ts");
  const snapshot = steerRows => ({
    version: 7, savedAt: "2026-10-31T00:00:00Z", model_id: "m", session_id: "s",
    tree: { tree_format: 2, drowse_version: "6", root_id: "r", active_node_id: "r", rev: 0, model_id: "m", session_id: "s", nodes: [], children_of: {}, cast: {} },
    steerRack: steerRows, subspaceAlong: 0.5, customSteeringExpression: null,
    probeRack: { sortMode: "name", active: [], entries: [] },
    highlightState: { target: null, compareTarget: null, compareTwo: false, smoothBlend: false },
    samplingState: {
      temperature: 0.7, top_p: 0.95, top_k: null, max_tokens: 256, seed: null, system_prompt: "",
      stop_sequences: "", logit_bias_text: "", presence_penalty: 0, frequency_penalty: 0,
      thinking: false, return_top_k: 8, user_role: "user", assistant_role: "assistant",
    },
  });
  const ok = rows => {
    const value = snapshot(rows);
    validateConversationSnapshot(value, Object.keys(value.samplingState));
  };
  ok([
    { name: "jlens/storm", ...rack.get("jlens/storm") },
    { name: "default/emotions", ...rack.get("default/emotions") },
    { name: "neuron/L31n4410", ...rack.get("neuron/L31n4410") },
    { name: "head/L28h12", mode: "unit", ablate: true, alpha: 1, trigger: "BOTH", enabled: true },
    { name: "sae/7", ...rack.get("sae/7") },
  ]);
  for (const [row, why] of [
    [{ name: "jlens/storm", ...rack.get("jlens/storm"), layers: [5, 4] }, "layers out of order"],
    [{ name: "jlens/storm", ...rack.get("jlens/storm"), layers: [-1] }, "a negative layer"],
    [{ name: "jlens/storm", ...rack.get("jlens/storm"), layers: "all" }, "a clause the rack never writes"],
    [{ name: "head/L28h12", mode: "unit", ablate: false, alpha: 1, trigger: "BOTH", enabled: true }, "a pushed head"],
    [{ name: "neuron/L1h3", mode: "unit", ablate: true, alpha: 1, trigger: "BOTH", enabled: true }, "a neuron named as a head"],
    [{ name: "neuron/L1n3", mode: "unit", ablate: true, alpha: 1, trigger: "BOTH", enabled: true, layers: [1] }, "a unit with a clause"],
  ]) assert.throws(() => ok([row]), TypeError, why);
  console.log("Term editor: the layer clause, unit terms, where a term acts, the chips' words and the snapshot rows passed");
} finally { await server.close(); }
