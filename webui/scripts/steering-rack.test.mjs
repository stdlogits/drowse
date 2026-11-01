import assert from "node:assert/strict";
import { createServer } from "vite";

const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  // The profiles service the correlation-freshness block below drives.
  let roster = ["a"];
  let pendingCorrelation = null;
  installRuntimeClient({
    mode: "http",
    profiles: {
      list: async () => ({ profiles: roster.map(name => ({ name, layers: [0], metadata: {} })) }),
      correlation: () => new Promise(resolve => { pendingCorrelation = resolve; }),
    },
  });
  const steering = await server.ssrLoadModule("/src/lib/stores/steering.svelte.ts");
  const { genStatus } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const { pendingActions, drainNextPendingAction } = await server.ssrLoadModule("/src/lib/stores/pending.svelte.ts");
  const { steerRack, patchSteerEntry, setSteerCoords, setSteerLabel, setSteerTiming, relateSteerEntry, removeSteerEntry, addSubspaceToRack, addManifoldToRack, addSaeToRack, setUnitTerm, currentSteeringExpression } = steering;
  steerRack.catalog = [{
    namespace: "default", name: "emotions", node_count: 3, node_labels: ["calm", "angry", "glad"],
    node_coords: [[0, 0], [1, 0], [0, 1]], intrinsic_dim: 2, domain: { type: "custom" },
  }, {
    namespace: "default", name: "formal.casual", node_count: 2, node_labels: ["formal", "casual"],
    node_coords: [[1], [-1]], intrinsic_dim: 1, domain: { type: "custom" },
  }, {
    namespace: "default", name: "honest.deceptive", node_count: 2, node_labels: ["honest", "deceptive"],
    node_coords: [[1], [-1]], intrinsic_dim: 1, domain: { type: "custom" },
  }];
  addSubspaceToRack("default/formal.casual");
  addManifoldToRack("default/emotions");
  addSaeToRack(17);

  // Label binding mirrors node coords; coords drop the label; both only touch geometric terms.
  setSteerLabel("default/emotions", "glad");
  assert.deepEqual(steerRack.entries.get("default/emotions"), { ...steerRack.entries.get("default/emotions"), label: "glad", coords: [0, 1] });
  setSteerCoords("default/emotions", [0.5, 0.25]);
  assert.equal(steerRack.entries.get("default/emotions").label, null);
  setSteerLabel("default/emotions", "unknown");
  assert.deepEqual(steerRack.entries.get("default/emotions").coords, [0.5, 0.25], "an uncatalogued label keeps the coords");
  assert.equal(steerRack.entries.get("default/emotions").label, "unknown", "and resolves server-side");
  setSteerLabel("default/emotions", null);
  setSteerLabel("sae/17", "glad");
  assert.equal(steerRack.entries.get("sae/17").label, undefined, "atoms have no geometry");

  // Mode-restricted patches leave other families alone.
  patchSteerEntry("default/formal.casual", { blend: 0.9 }, ["manifold"]);
  assert.equal(steerRack.entries.get("default/formal.casual").blend, undefined);
  patchSteerEntry("default/emotions", { blend: 0.8, onto: 0.2 }, ["manifold"]);
  patchSteerEntry("sae/17", { alpha: 0.4, ablate: true }, ["sae"]);
  setSteerTiming("default/formal.casual", { trigger: "BEFORE" });
  assert.equal(currentSteeringExpression(),
    "0.5 default/formal.casual%formal@before + 0.4 !sae/17 + 0.8,0.2 default/emotions%0.5,0.25");

  // Edits during a generation queue and coalesce into one bubble.
  genStatus.active = true;
  patchSteerEntry("sae/17", { enabled: false });
  patchSteerEntry("sae/17", { alpha: 0.6 }, ["sae"]);
  assert.equal(pendingActions.queue.length, 1);
  assert.equal(steerRack.entries.get("sae/17").enabled, true);
  genStatus.active = false;
  await drainNextPendingAction();
  assert.deepEqual([steerRack.entries.get("sae/17").enabled, steerRack.entries.get("sae/17").alpha], [false, 0.6]);
  removeSteerEntry("sae/17");
  assert.equal(steerRack.entries.has("sae/17"), false);

  // Timing has one path: a window, its count only while it counts, and a gate, each replacing the
  // last; a unit's push and ablation keep the term's timing.
  setSteerTiming("default/formal.casual", { trigger: "AFTER_N", tokens: 6, gate: { probe: "default/formal.casual", op: "<", threshold: -0.5 } });
  assert.equal(currentSteeringExpression(), "0.5 default/formal.casual%formal@after:6&when:default/formal.casual<-0.5 + 0.8,0.2 default/emotions%0.5,0.25");
  setSteerTiming("default/formal.casual", { trigger: "RESPONSE", gate: steerRack.entries.get("default/formal.casual").gate });
  assert.equal(steerRack.entries.get("default/formal.casual").tokens, undefined, "a phase drops the count");
  assert.ok(!("tokens" in steerRack.entries.get("default/formal.casual")));
  assert.match(currentSteeringExpression(), /^0\.5 default\/formal\.casual%formal@when:default\/formal\.casual<-0\.5 \+/);
  setSteerTiming("default/formal.casual", { trigger: "BOTH", gate: null });
  assert.ok(!("gate" in steerRack.entries.get("default/formal.casual")), "no gate leaves no field");
  setUnitTerm("neuron/L3n4", { ablate: false, alpha: 0.2 });
  setSteerTiming("neuron/L3n4", { trigger: "FIRST_N", tokens: 3, gate: { probe: "neuron/L3n4", op: ">", threshold: 1 } });
  setUnitTerm("neuron/L3n4", { ablate: true, alpha: 1 });
  assert.deepEqual(steerRack.entries.get("neuron/L3n4"),
    { mode: "unit", ablate: true, alpha: 1, trigger: "FIRST_N", tokens: 3, gate: { probe: "neuron/L3n4", op: ">", threshold: 1 }, enabled: true });
  removeSteerEntry("neuron/L3n4");

  // A relation re-keys a flat concept as a projection at its own signed strength (its pole's sign
  // on the shared one), keeping its timing, layers and switch; all of it again puts it back on the
  // shared strength at the pole its sign points to. The new key reaches the caller to select.
  addSubspaceToRack("default/honest.deceptive");
  setSteerLabel("default/honest.deceptive", "deceptive");
  steering.setSteerLayers("default/honest.deceptive", [3, 4]);
  setSteerTiming("default/honest.deceptive", { trigger: "THINKING" });
  const selected = [];
  relateSteerEntry("default/honest.deceptive", { operator: "~", onto: "default/formal.casual" }, (key) => selected.push(key));
  assert.deepEqual(selected, ["default/honest.deceptive~default/formal.casual"]);
  assert.equal(steerRack.entries.has("default/honest.deceptive"), false);
  assert.deepEqual(steerRack.entries.get("default/honest.deceptive~default/formal.casual"), {
    mode: "projection", base: "default/honest.deceptive", operator: "~", onto: "default/formal.casual",
    alpha: -0.5, trigger: "THINKING", layers: [3, 4], enabled: true,
  });
  assert.match(currentSteeringExpression(), / - 0\.5 default\/honest\.deceptive~default\/formal\.casual\{3-4\}@thinking/);
  relateSteerEntry("default/honest.deceptive~default/formal.casual", { operator: "|", onto: "default/formal.casual" }, (key) => selected.push(key));
  assert.equal(selected.at(-1), "default/honest.deceptive|default/formal.casual");
  addSubspaceToRack("default/honest.deceptive");
  relateSteerEntry("default/honest.deceptive|default/formal.casual", null, (key) => selected.push(key));
  assert.equal(selected.length, 2, "a relation that would land on a chip already there changes nothing");
  removeSteerEntry("default/honest.deceptive");
  relateSteerEntry("default/honest.deceptive|default/formal.casual", null, (key) => selected.push(key));
  assert.deepEqual(steerRack.entries.get("default/honest.deceptive"), {
    mode: "subspace", ablate: false, coords: [-1], label: "deceptive", variant: "raw", trigger: "THINKING", layers: [3, 4], enabled: true,
  });
  removeSteerEntry("default/honest.deceptive");
  // An agent may write the concepts bare; all of it again keys the chip by its catalog name.
  steerRack.entries.set("honest.deceptive~formal.casual", {
    mode: "projection", base: "honest.deceptive", operator: "~", onto: "formal.casual", alpha: 0.3, trigger: "BOTH", enabled: true,
  });
  relateSteerEntry("honest.deceptive~formal.casual", null, (key) => selected.push(key));
  assert.equal(selected.at(-1), "default/honest.deceptive");
  assert.equal(steerRack.entries.get("default/honest.deceptive").label, "honest");
  removeSteerEntry("default/honest.deceptive");

  // Regression: the round trip kept its direction only at a positive shared strength and a labelled
  // pole. Back from a projection the pole is the side of its strength over the shared one's sign;
  // forward, a free coordinate's side signs the strength (it pushed toward the first pole before).
  const sharedAlong = steerRack.subspaceAlong;
  const roundTrip = (along, place) => {
    steerRack.subspaceAlong = along;
    addSubspaceToRack("default/honest.deceptive");
    place();
    const before = currentSteeringExpression();
    relateSteerEntry("default/honest.deceptive", { operator: "~", onto: "default/formal.casual" });
    const related = steerRack.entries.get("default/honest.deceptive~default/formal.casual").alpha;
    relateSteerEntry("default/honest.deceptive~default/formal.casual", null);
    const after = steerRack.entries.get("default/honest.deceptive");
    removeSteerEntry("default/honest.deceptive");
    return { before, related, label: after.label };
  };
  for (const along of [0.5, -0.5]) {
    for (const [pole, side] of [["honest", 1], ["deceptive", -1]]) {
      const trip = roundTrip(along, () => setSteerLabel("default/honest.deceptive", pole));
      assert.equal(trip.related, side * along, `${pole} at ${along}: strength toward honest`);
      assert.equal(trip.label, pole, `${pole} at ${along}: the round trip keeps the pole`);
    }
    const free = roundTrip(along, () => setSteerCoords("default/honest.deceptive", [-0.8]));
    assert.equal(free.related, -along, `a free coordinate on deceptive's side at ${along}`);
    assert.equal(free.label, "deceptive");
  }
  steerRack.subspaceAlong = sharedAlong;

  // Correlation freshness: nothing fetches per generation, but a generation
  // (which can register vectors and fill null cells) or a vector-roster change
  // leaves the cached matrix stale, so the open drawer refetches.
  const { correlationCache, markCorrelationStale, refreshCorrelation, refreshVectorList } = steering;
  const matrix = names => ({ names, matrix: {}, layers_shared: {} });
  const fresh = () => correlationCache.fetched === correlationCache.revision;
  await refreshVectorList();
  let request = refreshCorrelation(null);
  pendingCorrelation(matrix(["a"]));
  await request;
  assert.deepEqual(steerRack.correlation.names, ["a"]);
  assert.ok(fresh());

  markCorrelationStale();
  assert.deepEqual(steerRack.correlation.names, ["a"], "a stale matrix stays on screen");
  assert.ok(!fresh(), "but the open drawer refetches it");
  request = refreshCorrelation(null);
  markCorrelationStale();
  pendingCorrelation(matrix(["stale"]));
  await request;
  assert.deepEqual(steerRack.correlation.names, ["a"], "a response older than the last stale mark is dropped");
  assert.ok(!fresh());

  await refreshVectorList();
  assert.ok(steerRack.correlation, "an unchanged vector roster keeps the matrix");
  roster = ["a", "b"];
  await refreshVectorList();
  assert.equal(steerRack.correlation, null, "a registered vector joins the pool");
  // Regression: in server mode a reload restored nothing from the chat's autosave record, so the
  // steer rack came back empty and the autosave that followed wrote the empty rack over the saved
  // one. The page's own workspace now comes back from the record (the server keeps the tree).
  {
    const { restoreWorkspaceSettings } = await server.ssrLoadModule("/src/lib/conversationWorkspace.ts");
    const { samplingState } = await server.ssrLoadModule("/src/lib/stores/sampling.svelte.ts");
    const { highlightState, probeRack } = await server.ssrLoadModule("/src/lib/stores/probes.svelte.ts");
    const sampling = { ...JSON.parse(JSON.stringify(samplingState)), temperature: 1.3 };
    const saved = {
      version: 8, savedAt: "2026-11-01T00:00:00.000Z", model_id: "fixture/model", session_id: "default",
      tree: { tree_format: 2, drowse_version: "fixture", model_id: "fixture/model", session_id: "default", name: null,
        root_id: "root", active_node_id: "root", rev: 141, nodes: [], children_of: {}, cast: {} },
      steerRack: [{ name: "default/formal.casual", mode: "subspace", ablate: false, coords: [1], label: "formal",
        variant: "raw", trigger: "BOTH", enabled: true }],
      subspaceAlong: 0.5, readOnlyRecipe: null,
      probeRack: { sortMode: "value", active: [], entries: [] },
      highlightState: { target: null, compareTarget: null, compareTwo: false, smoothBlend: true },
      samplingState: sampling,
    };
    steerRack.entries.clear();
    restoreWorkspaceSettings(saved);
    assert.deepEqual([...steerRack.entries.keys()], ["default/formal.casual"], "the saved rack comes back");
    assert.equal(steerRack.subspaceAlong, 0.5);
    assert.equal(currentSteeringExpression(), "0.5 default/formal.casual%formal");
    assert.equal(samplingState.temperature, 1.3);
    assert.equal(highlightState.smoothBlend, true);
    assert.equal(probeRack.sortMode, "value");
  }
  // Regression (a9's decision: no recipe typed as text): the automatic comparison's custom field
  // took steering=<expression> and the server applied it. It changes sampling only now.
  {
    const { autoRegenState, currentRecipeOverride, customComparisonProblem } = await server.ssrLoadModule("/src/lib/stores/ab.svelte.ts");
    autoRegenState.enabled = true;
    autoRegenState.mode = "custom";
    autoRegenState.custom = "seed=42, temperature=1.5";
    assert.equal(currentRecipeOverride(), "seed=42, temperature=1.5");
    for (const text of ["steering=0.5 !formal.casual", "seed=4, steering = 0.2 calm", " steering="]) {
      autoRegenState.custom = text;
      assert.ok(customComparisonProblem(text), text);
      assert.equal(currentRecipeOverride(), null, `${text} is never sent`);
    }
    assert.equal(customComparisonProblem("seed=42"), null);
    autoRegenState.enabled = false;
  }
  console.log("Steering rack: one patch path for every term, label binding, mode guards, queued edits, and correlation freshness passed");
} finally { await server.close(); }
