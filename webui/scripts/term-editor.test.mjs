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
    ["head/L28h12", { mode: "unit", ablate: true, alpha: 1, trigger: "BOTH", enabled: true }],
    ["default/emotions", { mode: "manifold", blend: 0.35, onto: 0, coords: [0], label: "wonder", variant: "raw", trigger: "RESPONSE", enabled: true, layers: "workspace" }],
  ]);
  assert.equal(serializeExpression(rack, 0.5),
    "0.5 default/honest%honest{12-14,20} + !jlens/storm{20-22} - 0.2 neuron/L31n4410@response + !head/L28h12 + 0.35 default/emotions%wonder{workspace}@response",
    "the clause follows the selector and precedes the trigger; a term at no layer sits out; a unit takes no clause");
  assert.equal(formatTerm("sae/7", rack.get("sae/7"), 0.5), "0.3 sae/7", "on its own, an inert term writes no clause");

  // ---- rack edits: the clause and units
  const steering = await server.ssrLoadModule("/src/lib/stores/steering.svelte.ts");
  const { steerRack, setSteerLayers, setUnitTerm, patchSteerEntry, addJLensToRack, currentSteeringExpression, removeSteerEntry } = steering;
  const { UNIT_ABLATE_ALPHA, UNIT_PUSH_ALPHA } = await server.ssrLoadModule("/src/lib/unitTerms.ts");
  assert.equal(setUnitTerm("NEURON/l08N0344", { ablate: false, alpha: UNIT_PUSH_ALPHA }), "neuron/L8n344", "a unit's key is the grammar's canonical form");
  removeSteerEntry("neuron/L8n344");
  assert.equal(setUnitTerm("neuron/L8h3", { ablate: true, alpha: 1 }), null, "the letter must match the family");
  assert.equal(setUnitTerm("neuron/8", { ablate: true, alpha: 1 }), null);
  // The one rule (lib/unitTerms.ts): a head is only ablated, an ablation never negative.
  assert.equal(setUnitTerm("head/L2h0", { ablate: false, alpha: UNIT_PUSH_ALPHA }), null, "a head push is refused");
  assert.equal(setUnitTerm("neuron/L1n3", { ablate: true, alpha: -0.2 }), null, "a negative ablation is refused");
  assert.equal(steerRack.entries.size, 0, "a refused term changes nothing");
  assert.equal(setUnitTerm("head/L2h0", { ablate: true, alpha: UNIT_ABLATE_ALPHA }), "head/L2h0");
  assert.deepEqual(steerRack.entries.get("head/L2h0"), { mode: "unit", ablate: true, alpha: 1, trigger: "BOTH", enabled: true }, "exact mean-ablation");
  setUnitTerm("neuron/L1n3", { ablate: false, alpha: UNIT_PUSH_ALPHA });
  assert.deepEqual(steerRack.entries.get("neuron/L1n3"), { mode: "unit", ablate: false, alpha: 0.2, trigger: "BOTH", enabled: true });
  // Every edit keeps to the rule: the editor's switch and dial go through patchSteerEntry.
  patchSteerEntry("head/L2h0", { ablate: false, alpha: UNIT_PUSH_ALPHA }, ["unit"]);
  assert.equal(steerRack.entries.get("head/L2h0").ablate, true, "a head cannot be switched to a push");
  patchSteerEntry("head/L2h0", { alpha: -0.5 }, ["unit"]);
  assert.equal(steerRack.entries.get("head/L2h0").alpha, 1, "an ablation's strength never goes negative");
  patchSteerEntry("neuron/L1n3", { alpha: -0.4 }, ["unit"]);
  assert.equal(steerRack.entries.get("neuron/L1n3").alpha, -0.4, "a push takes either pole");
  patchSteerEntry("neuron/L1n3", { alpha: UNIT_PUSH_ALPHA }, ["unit"]);
  addJLensToRack("storm");
  setSteerLayers("jlens/storm", [4, 5]);
  setSteerLayers("neuron/L1n3", [1]);
  assert.equal(steerRack.entries.get("neuron/L1n3").layers, undefined, "a unit takes no clause");
  assert.equal(currentSteeringExpression(), "0.3 jlens/storm{4-5} + !head/L2h0 + 0.2 neuron/L1n3");
  setSteerLayers("jlens/storm", null);
  assert.equal(currentSteeringExpression(), "0.3 jlens/storm + !head/L2h0 + 0.2 neuron/L1n3");
  for (const name of ["jlens/storm", "neuron/L1n3", "head/L2h0"]) removeSteerEntry(name);

  // ---- where a term acts, as the server says
  const { placeOf, layersProbe, forgetTermLayers, SETTLE_MS } = await server.ssrLoadModule("/src/lib/steering/termLayers.svelte.ts");
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
  placeOf("default/emotions", { ...entry, blend: 0.6, trigger: "AFTER" }, "m");
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
  // Regression: a 409 (the model busy with a reply or a fit) was cached as the term's refusal.
  const { ApiError } = await server.ssrLoadModule("/src/lib/runtime/errors.ts");
  answers.set(layersProbe("jlens/sun", other, false), new ApiError(409, "/steering/validate", "", { detail: "a model operation is in flight; retry shortly" }));
  placeOf("jlens/sun", other, "m");
  await tick(); await tick();
  assert.equal(placeOf("jlens/sun", other, "m").status, "loading", "busy is not a refusal");
  forgetTermLayers();
  place = placeOf("default/emotions", entry, "m");
  assert.equal(place.status, "loading", "a catalog change forgets every answer");
  assert.deepEqual([place.where?.fitted, place.selected], [fitted, [14, 15, 16]], "and the term keeps its place until the new one comes");
  await tick(); await tick();

  // ---- a moved term keeps its place, and asks once its position rests
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const moving = { mode: "manifold", blend: 0.5, onto: 0, coords: [0.1], label: null, variant: "raw", trigger: "BOTH", enabled: true, layers: "workspace" };
  const at = x => ({ ...moving, coords: [x] });
  for (const [x, share] of [[0.1, [1, 1, 1]], [0.2, [0.5, 1, 1.5]], [0.3, [1.5, 1, 0.5]]]) {
    answers.set(layersProbe("default/moods", at(x), false), where([2, 3, 4], share, [2, 3, 4], 8));
    answers.set(layersProbe("default/moods", at(x), true), where([2, 3, 4], share, [3, 4], 8));
  }
  assert.equal(placeOf("default/moods", at(0.1), "m").where, null, "a term's first probe has nothing to show yet");
  await tick(); await tick();
  assert.deepEqual(placeOf("default/moods", at(0.1), "m").selected, [3, 4]);
  let mark = asked.length;
  for (const x of [0.2, 0.3]) {
    place = placeOf("default/moods", at(x), "m");
    assert.equal(place.status, "loading");
    assert.deepEqual([place.where.share, place.selected], [[1, 1, 1], [3, 4]], "a moved term keeps the strip and the span it showed");
    await tick(); await tick();
  }
  assert.equal(asked.length, mark, "a position in motion asks nothing");
  await sleep(SETTLE_MS + 30);
  await tick(); await tick();
  assert.deepEqual(asked.slice(mark), [layersProbe("default/moods", at(0.3), false), layersProbe("default/moods", at(0.3), true)],
    "the rested position is asked once, the passed one never");
  place = placeOf("default/moods", at(0.3), "m");
  assert.deepEqual([place.status, place.where.share, place.selected], ["ready", [1.5, 1, 0.5], [3, 4]]);

  // A new clause is asked at once, and until the workspace band answers the term keeps the layers
  // it showed (never none).
  const calm = { ...moving, coords: [0], layers: null };
  const calmBand = { ...calm, layers: "workspace" };
  answers.set(layersProbe("default/calm", calm, false), where([2, 3, 4], [1, 1, 1], [2, 3, 4], 8));
  answers.set(layersProbe("default/calm", calmBand, true), where([2, 3, 4], [1, 1, 1], [3, 4], 8));
  placeOf("default/calm", calm, "m");
  await tick(); await tick();
  assert.deepEqual(placeOf("default/calm", calm, "m").selected, [2, 3, 4]);
  mark = asked.length;
  assert.deepEqual(placeOf("default/calm", calmBand, "m").selected, [2, 3, 4], "the layers last shown, while the band is asked");
  await tick(); await tick();
  assert.equal(asked.length, mark + 1, "a new clause is asked without waiting");
  assert.deepEqual(placeOf("default/calm", calmBand, "m").selected, [3, 4]);

  // ---- what a chip and the editor say
  const terms = await server.ssrLoadModule("/src/lib/steering/terms.ts");
  assert.equal(terms.coefficientText(-0.2), "−0.2");
  assert.equal(terms.coefficientText(0.35, { fixed: true }), "0.35");
  assert.equal(terms.coefficientText(-0.004, { fixed: true }), "0.00");
  // A projection's subtitle is its relation alone (it read "all but the part along another concept").
  const notAlong = { mode: "projection", base: "a", operator: "|", onto: "default/warm.clinical", alpha: 0.5, trigger: "BOTH", enabled: true };
  assert.equal(terms.termKind(notAlong, null), "");
  assert.equal(terms.termSource("a|default/warm.clinical", notAlong), "not along warm.clinical");
  assert.equal(terms.chipStrength("jlens/storm", { mode: "jlens", ablate: true, alpha: 1 }, 0.5), "without");
  assert.equal(terms.chipStrength("head/L2h0", { mode: "unit", ablate: true, alpha: 1 }, 0.5), "without");
  assert.equal(terms.chipStrength("default/honest", { mode: "subspace", ablate: false }, 0.5), "0.5", "a flat term shows the shared strength");
  assert.equal(terms.termTitle("neuron/L31n4410", { mode: "unit" }), "L31 n4410");
  assert.equal(terms.unitLayer("head/L28h12"), 28);
  assert.equal(terms.triggerWord({ trigger: "AFTER" }), "after thinking");
  assert.equal(terms.triggerWord({ trigger: "BOTH" }), "");
  assert.equal(terms.triggerWord({ trigger: "FIRST_N", tokens: 8 }), "first 8 tokens");
  assert.equal(terms.triggerWord({ trigger: "RESPONSE", gate: { probe: "default/calm.anxious", op: "<=", threshold: -1.5 } }), "while calm.anxious ≤ −1.5",
    "a bare gate is the response's own window, so the chip says only the gate");
  assert.equal(terms.triggerWord({ trigger: "THINKING", gate: { probe: "jlens/storm", op: ">", threshold: 0.1 } }), "thinking while storm > 0.1");
  assert.deepEqual([terms.whenChoice("AFTER_N"), terms.whenChoice("FIRST_N"), terms.whenChoice("AFTER")], ["TOKENS", "TOKENS", "AFTER"]);
  assert.deepEqual(terms.strengthDial("default/emotions", entry, 0.5), { value: 0.35, min: 0, max: 1, step: 0.05, shared: false },
    "a curved term slides toward its position only: composition refuses a negative along");
  assert.deepEqual(terms.strengthDial("jlens/storm", { mode: "jlens", ablate: false, alpha: 0.3 }, 0.5), { value: 0.3, min: -1, max: 1, step: 0.05, shared: false });
  assert.equal(terms.strengthDial("default/honest", { mode: "subspace" }, 0.7).shared, true);
  assert.deepEqual([terms.strengthDial("jlens/storm", { mode: "jlens", ablate: true, alpha: 0.4 }, 0.5).min, terms.strengthSays("jlens/storm", { mode: "jlens", ablate: true, alpha: 0.4 }, 0.4)], [-1, "40% of it"],
    "an atom's removal is signed: below zero it adds to the component");
  assert.equal(terms.strengthDial("default/honest", { mode: "subspace" }, -0.7).min, -2, "the shared flat strength pushes away below zero");
  const without = { mode: "subspace", ablate: true, coords: [1], label: "confident", variant: "raw", trigger: "BOTH", enabled: true };
  assert.equal(terms.strengthDial("default/confident.uncertain", without, 1.5).max, 2, "a removed flat term rides the shared dial");
  assert.deepEqual([1, 1.5, 2].map(v => terms.strengthSays("default/confident.uncertain", without, v)), ["All of it", "150% of it", "200% of it"],
    "past 1 a removal takes more than all of it, and says so");
  assert.equal(terms.strengthSays("jlens/storm", { mode: "jlens", ablate: false }, -0.3), "Away from storm");
  const line = "0.5 default/confident.uncertain%confident{4-8,20}@response";
  assert.deepEqual(terms.wrapPieces(line), ["0.5 default/", "confident.uncertain", "%confident", "{4-8,", "20}", "@response"],
    "the expression wraps between a selector's parts, never inside a name, and inside a clause only at a comma");
  assert.equal(terms.wrapPieces(line).join(""), line);
  assert.ok(terms.canRemove("neuron/L1n3", { mode: "unit" }, null) && !terms.canRemove("head/L1h3", { mode: "unit" }, null),
    "a neuron switches between push and ablation; a head (no push) does not");
  assert.deepEqual(terms.strengthDial("head/L1h3", { mode: "unit", ablate: true, alpha: 1 }, 0.5), { value: 1, min: 0, max: 1, step: 0.05, shared: false },
    "an ablation's dial runs from none of it to exact mean-ablation, never below zero");
  assert.deepEqual([terms.strengthDial("neuron/L1n3", { mode: "unit", ablate: false, alpha: -0.2 }, 0.5).min, terms.strengthSays("neuron/L1n3", { mode: "unit", ablate: false }, -0.2)],
    [-1, "Away from L1 n3"], "a push runs either pole");
  assert.ok(!terms.canRemove("default/emotions", entry, { node_count: 20 }), "a curved position cannot be removed");
  assert.ok(terms.canRemove("default/honest", { mode: "subspace" }, { node_count: 2 }));

  // ---- snapshot rows keep a clause and a unit
  const { validateConversationSnapshot } = await server.ssrLoadModule("/src/lib/conversationSnapshot.ts");
  const snapshot = steerRows => ({
    version: 8, savedAt: "2026-10-31T00:00:00Z", model_id: "m", session_id: "s",
    tree: { tree_format: 2, drowse_version: "6", root_id: "r", active_node_id: "r", rev: 0, model_id: "m", session_id: "s", nodes: [], children_of: {}, cast: {} },
    steerRack: steerRows, subspaceAlong: 0.5, readOnlyRecipe: null,
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
  // ---- an S-lens token term: toward having read the word, the jlens term's twin
  const read = { mode: "slens", ablate: false, alpha: 0.3, trigger: "RESPONSE", enabled: true, layers: [2, 3, 4] };
  assert.equal(formatTerm("slens/Paris", read, 0.5), "0.3 slens/Paris{2-4}@response", "an slens term takes a clause like a jlens one");
  assert.equal(serializeExpression(new Map([
    ["sae/7", { mode: "sae", ablate: false, alpha: 0.3, trigger: "BOTH", enabled: true }],
    ["slens/Paris", read],
    ["jlens/storm", { mode: "jlens", ablate: false, alpha: 0.3, trigger: "BOTH", enabled: true }],
  ]), 0.5), "0.3 jlens/storm + 0.3 slens/Paris{2-4}@response + 0.3 sae/7", "lens tokens emit together, said before read");
  assert.equal(terms.termTitle("slens/Paris", read), "Paris");
  // Regression: a term from the token view's Steer toward it (jlens/#282) read "#282" on its chip.
  const said = { mode: "jlens", ablate: false, alpha: 0.3, trigger: "BOTH", enabled: true };
  const known = new Map([[282, " of"]]);
  assert.equal(terms.termTitle("jlens/#282", said, null, known), "of");
  assert.equal(terms.termTitle("slens/#282", { ...said, mode: "slens" }, null, known), "of");
  assert.equal(terms.termTitle("jlens/#283", said, null, known), "#283", "an id it was not added for shows the id");
  assert.equal(terms.termSource("jlens/#282", said), "lens token #282, as said", "the id stays in the meta");
  assert.equal(terms.termSource("jlens/storm", said), "lens token, as said");
  // Regression: a free months position read "6" beside "months", which reads as a duration.
  const free = { mode: "manifold", blend: 0.5, onto: 0, coords: [6], label: null, variant: "raw", trigger: "BOTH", enabled: true };
  const monthsInfo = { domain: { type: "box", axes: [{ name: "month", lo: 0, hi: 12, periodic: true }] } };
  assert.equal(terms.termTitle("default/months", free, monthsInfo), "month 6.00");
  assert.equal(terms.termTitle("default/months", free), "point 6.00", "without the manifold's axes it is a point");
  assert.equal(terms.termTitle("default/months", { ...free, label: "july" }, monthsInfo), "july");
  assert.equal(terms.strengthSays("default/months", free, 0.5, monthsInfo), "Toward month 6.00", "the strength names it the same way");
  assert.equal(terms.termSource("slens/Paris", read), "lens token, as read");
  assert.equal(terms.termSource("jlens/Paris", { ...read, mode: "jlens" }), "lens token, as said", "the pair's twin wording");
  // The one pair the Lens panel and the token view share: its choice when that lens is ready, the
  // other when only that one is, so a model with one lens family offers it alone.
  const lens = await server.ssrLoadModule("/src/lib/stores/instruments.svelte.ts");
  const { sessionState } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
  const sessionBefore = sessionState.info;
  sessionState.info = { ...(sessionBefore ?? {}), jlens_fitted: true };
  lens.lensSourceState.sources = [];
  assert.equal(lens.lensNamespaceReady("slens"), false, "no S source, no slens/");
  lens.lensPairState.namespace = "slens";
  assert.equal(lens.lensNamespaceInUse(), "jlens", "the R lens alone is offered alone");
  lens.lensSourceState.sources = [{ source: "local:s", family: "S", active: true, sites: ["mid"] }];
  assert.equal(lens.lensNamespaceReady("slens"), false, "slens/ needs the S lens's out site");
  lens.lensSourceState.sources = [{ source: "local:s", family: "S", active: true, sites: ["out", "mid"], compatible: true }];
  assert.equal(lens.lensNamespaceInUse(), "slens");
  sessionState.info = { ...(sessionBefore ?? {}), jlens_fitted: false };
  lens.lensPairState.namespace = "jlens";
  assert.equal(lens.lensNamespaceInUse(), "slens", "the S lens alone is offered alone");
  lens.lensSourceState.sources = [{ source: "local:s", family: "S", active: true, sites: ["out"], compatible: false }];
  assert.equal(lens.lensNamespaceReady("slens"), false, "an S source not loaded for these weights");
  assert.deepEqual(lens.LENS_NAMESPACES.map(row => row.label), ["As said", "As read"]);
  sessionState.info = sessionBefore;
  lens.lensSourceState.sources = [];
  lens.lensPairState.namespace = "jlens";
  assert.ok(terms.canRemove("slens/Paris", read, null));
  assert.equal(layersProbe("slens/Paris", read, true), "1 slens/Paris{2-4}@response");
  const { addSLensToRack } = steering;
  addSLensToRack("slens/Paris");
  setSteerLayers("slens/Paris", [3]);
  assert.equal(currentSteeringExpression(), "0.3 slens/Paris{3}");
  removeSteerEntry("slens/Paris");
  ok([{ name: "slens/Paris", ...read }]);
  assert.throws(() => ok([{ name: "jlens/Paris", ...read }]), TypeError, "an slens row names an slens atom");

  // Escape closes the open term editor (a selection) before it stops a streaming reply. Regression:
  // the window's Escape stopped the reply first, so closing the editor mid-reply cut it off.
  const { escapeAction } = await server.ssrLoadModule("/src/lib/escapeKey.ts");
  const idle = { paletteOpen: false, modalOpen: false, drawerOpen: false, selectionOpen: false, generating: false };
  assert.equal(escapeAction({ ...idle, selectionOpen: true, generating: true }), "selection");
  assert.equal(escapeAction({ ...idle, generating: true }), "stop");
  assert.equal(escapeAction({ ...idle, drawerOpen: true, selectionOpen: true, generating: true }), "drawer");
  assert.equal(escapeAction({ ...idle, modalOpen: true, generating: true }), "modal");
  assert.equal(escapeAction({ ...idle, paletteOpen: true, modalOpen: true }), "palette");
  assert.equal(escapeAction(idle), "none");

  // Regression (Wave 4): the position rows' grid (a 44px label column) was an unscoped `.dial` rule
  // after the strength dial's, so the strength readout (124px, 40px type) sat in a 44px column over
  // its slider, and the squeezed range ran "-2" into "0". Only the strength dial is a bare `.dial`.
  const { readFileSync } = await import("node:fs");
  const editor = readFileSync(new URL("../src/panels/rack/TermEditor.svelte", import.meta.url), "utf8");
  const style = editor.slice(editor.indexOf("<style>"));
  assert.deepEqual([...style.matchAll(/^\s*\.dial \{([^}]*)\}/gm)].map((m) => m[1].trim()), ["display: flex; align-items: center; gap: var(--s4); min-width: 0;"]);
  assert.match(style, /\.position \.dial \{ display: grid;/);
  console.log("Term editor: the layer clause, unit terms, slens terms, where a term acts, the chips' words, the snapshot rows and Escape's order passed");
} finally { await server.close(); }
