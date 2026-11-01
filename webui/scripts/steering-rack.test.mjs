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
    namespace: "default", name: "verbose.concise", node_count: 2, node_labels: ["verbose", "concise"],
    node_coords: [[1], [-1]], intrinsic_dim: 1, domain: { type: "custom" },
  }];
  addSubspaceToRack("default/formal.casual");
  addManifoldToRack("default/emotions");
  // Regression: a curved term started at a free point in the middle of its nodes; it starts at its
  // first node, titled by it.
  assert.deepEqual([steerRack.entries.get("default/emotions").label, steerRack.entries.get("default/emotions").coords], ["calm", [0, 0]]);
  assert.match(currentSteeringExpression(), /default\/emotions%calm/);
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
  addSubspaceToRack("default/verbose.concise");
  setSteerLabel("default/verbose.concise", "concise");
  steering.setSteerLayers("default/verbose.concise", [3, 4]);
  setSteerTiming("default/verbose.concise", { trigger: "THINKING" });
  const selected = [];
  relateSteerEntry("default/verbose.concise", { operator: "~", onto: "default/formal.casual" }, (key) => selected.push(key));
  assert.deepEqual(selected, ["default/verbose.concise~default/formal.casual"]);
  assert.equal(steerRack.entries.has("default/verbose.concise"), false);
  assert.deepEqual(steerRack.entries.get("default/verbose.concise~default/formal.casual"), {
    mode: "projection", base: "default/verbose.concise", operator: "~", onto: "default/formal.casual",
    alpha: -0.5, trigger: "THINKING", layers: [3, 4], enabled: true,
  });
  assert.match(currentSteeringExpression(), / - 0\.5 default\/verbose\.concise~default\/formal\.casual\{3-4\}@thinking/);
  relateSteerEntry("default/verbose.concise~default/formal.casual", { operator: "|", onto: "default/formal.casual" }, (key) => selected.push(key));
  assert.equal(selected.at(-1), "default/verbose.concise|default/formal.casual");
  addSubspaceToRack("default/verbose.concise");
  relateSteerEntry("default/verbose.concise|default/formal.casual", null, (key) => selected.push(key));
  assert.equal(selected.length, 2, "a relation that would land on a chip already there changes nothing");
  removeSteerEntry("default/verbose.concise");
  relateSteerEntry("default/verbose.concise|default/formal.casual", null, (key) => selected.push(key));
  assert.deepEqual(steerRack.entries.get("default/verbose.concise"), {
    mode: "subspace", ablate: false, coords: [-1], label: "concise", variant: "raw", trigger: "THINKING", layers: [3, 4], enabled: true,
  });
  removeSteerEntry("default/verbose.concise");
  // An agent may write the concepts bare; all of it again keys the chip by its catalog name.
  steerRack.entries.set("verbose.concise~formal.casual", {
    mode: "projection", base: "verbose.concise", operator: "~", onto: "formal.casual", alpha: 0.3, trigger: "BOTH", enabled: true,
  });
  relateSteerEntry("verbose.concise~formal.casual", null, (key) => selected.push(key));
  assert.equal(selected.at(-1), "default/verbose.concise");
  assert.equal(steerRack.entries.get("default/verbose.concise").label, "verbose");
  removeSteerEntry("default/verbose.concise");

  // Regression: a one-node concept (the bundled ``deceptive``, ``refusing``, ``manipulative``) was
  // added at the middle of its nodes, ``default/deceptive%0``: the neutral end, no push. It starts at its node, and a
  // relation and back keeps it there.
  steerRack.catalog = [...steerRack.catalog, {
    namespace: "default", name: "deceptive", node_count: 1, node_labels: ["deceptive"],
    node_coords: [[1]], intrinsic_dim: 1, domain: { type: "custom" },
  }];
  addSubspaceToRack("default/deceptive");
  assert.deepEqual([steerRack.entries.get("default/deceptive").label, steerRack.entries.get("default/deceptive").coords], ["deceptive", [1]]);
  assert.match(currentSteeringExpression(), /default\/deceptive%deceptive(?![\d,])/);
  relateSteerEntry("default/deceptive", { operator: "|", onto: "default/formal.casual" }, (key) => selected.push(key));
  assert.equal(steerRack.entries.get("default/deceptive|default/formal.casual").alpha, steerRack.subspaceAlong);
  relateSteerEntry("default/deceptive|default/formal.casual", null, (key) => selected.push(key));
  assert.equal(steerRack.entries.get("default/deceptive").label, "deceptive");
  removeSteerEntry("default/deceptive");

  // Regression: the side of a pole came from the catalog's display coordinate, whose sign is the
  // layout's own. SmolLM2 draws ``formal`` at -4.1 and ``casual`` at +21.6, so "Not along" turned
  // ``0.5`` toward formal into ``-0.5``, a push toward casual. A projection's positive strength is
  // toward the first pole (each layer's axis is oriented to node 0), so the side is the pole order,
  // and a free position's side is its coordinate's against the first pole's.
  steerRack.catalog = [...steerRack.catalog, {
    namespace: "default", name: "warm.clinical", node_count: 2, node_labels: ["warm", "clinical"],
    node_coords: [[-4.1], [21.6]], intrinsic_dim: 1, domain: { type: "custom" },
  }];
  const { strengthSays } = await server.ssrLoadModule("/src/lib/steering/terms.ts");
  const mirrored = steerRack.catalog.at(-1);
  const flipAlong = steerRack.subspaceAlong;
  steerRack.subspaceAlong = 0.5;
  for (const [place, side, pole] of [
    [() => setSteerLabel("default/warm.clinical", "warm"), 1, "warm"],
    [() => setSteerLabel("default/warm.clinical", "clinical"), -1, "clinical"],
    [() => setSteerCoords("default/warm.clinical", [-2]), 1, "warm"],
    [() => setSteerCoords("default/warm.clinical", [10]), -1, "clinical"],
  ]) {
    addSubspaceToRack("default/warm.clinical");
    place();
    relateSteerEntry("default/warm.clinical", { operator: "|", onto: "default/formal.casual" });
    const projection = steerRack.entries.get("default/warm.clinical|default/formal.casual");
    assert.equal(projection.alpha, side * 0.5, `${pole}: the projection pushes toward ${pole}`);
    assert.equal(strengthSays("default/warm.clinical|default/formal.casual", projection, projection.alpha, mirrored), `Toward ${pole}`);
    relateSteerEntry("default/warm.clinical|default/formal.casual", null);
    assert.equal(steerRack.entries.get("default/warm.clinical").label, pole, `${pole}: and back at ${pole}`);
    removeSteerEntry("default/warm.clinical");
  }
  steerRack.subspaceAlong = flipAlong;

  // Regression: the round trip kept its direction only at a positive shared strength and a labelled
  // pole. Back from a projection the pole is the side of its strength over the shared one's sign;
  // forward, a free coordinate's side signs the strength (it pushed toward the first pole before).
  const sharedAlong = steerRack.subspaceAlong;
  const roundTrip = (along, place) => {
    steerRack.subspaceAlong = along;
    addSubspaceToRack("default/verbose.concise");
    place();
    const before = currentSteeringExpression();
    relateSteerEntry("default/verbose.concise", { operator: "~", onto: "default/formal.casual" });
    const related = steerRack.entries.get("default/verbose.concise~default/formal.casual").alpha;
    relateSteerEntry("default/verbose.concise~default/formal.casual", null);
    const after = steerRack.entries.get("default/verbose.concise");
    removeSteerEntry("default/verbose.concise");
    return { before, related, label: after.label };
  };
  for (const along of [0.5, -0.5]) {
    for (const [pole, side] of [["verbose", 1], ["concise", -1]]) {
      const trip = roundTrip(along, () => setSteerLabel("default/verbose.concise", pole));
      assert.equal(trip.related, side * along, `${pole} at ${along}: strength toward verbose`);
      assert.equal(trip.label, pole, `${pole} at ${along}: the round trip keeps the pole`);
    }
    const free = roundTrip(along, () => setSteerCoords("default/verbose.concise", [-0.8]));
    assert.equal(free.related, -along, `a free coordinate on concise's side at ${along}`);
    assert.equal(free.label, "concise");
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

    // Regression: a tree with no turns gets no autosave record, so a recipe built before the first
    // message vanished on reload. It is kept as the draft of its model and root until a record
    // takes over, and comes back like a record's workspace.
    const { saveWorkspaceDraft, readWorkspaceDraft, clearWorkspaceDraft } = await server.ssrLoadModule("/src/lib/workspaceDraft.ts");
    const store = new Map();
    const storage = { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) };
    assert.equal(saveWorkspaceDraft(saved, storage), true);
    assert.equal(readWorkspaceDraft("fixture/model", "other-root", storage), null, "another tree");
    assert.equal(readWorkspaceDraft("other/model", "root", storage), null, "another model");
    steerRack.entries.clear();
    restoreWorkspaceSettings(readWorkspaceDraft("fixture/model", "root", storage));
    assert.equal(currentSteeringExpression(), "0.5 default/formal.casual%formal", "the draft rack comes back");
    // Regression: a rack change made just before a reload was lost (autosave is debounced and
    // asynchronous). The page keeps it as the draft as it hides; the next load takes the draft
    // over the record only when the draft is the newer.
    const { newerWorkspaceDraft } = await server.ssrLoadModule("/src/lib/workspaceDraft.ts");
    assert.ok(newerWorkspaceDraft("fixture/model", "root", "2026-10-31T23:59:59.000Z", storage), "a newer draft wins");
    assert.equal(newerWorkspaceDraft("fixture/model", "root", "2026-11-01T00:00:00.000Z", storage), null, "the record is as new");
    assert.equal(newerWorkspaceDraft("fixture/model", "other-root", "2026-10-31T00:00:00.000Z", storage), null);
    clearWorkspaceDraft(storage);
    assert.equal(readWorkspaceDraft("fixture/model", "root", storage), null);
    store.set("drowse.workspace-draft.v1", "{not json");
    assert.equal(readWorkspaceDraft("fixture/model", "root", storage), null, "an unreadable draft is none");
    assert.equal(saveWorkspaceDraft(saved, { ...storage, setItem: () => { throw new Error("quota"); } }), false);
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
  // Regression: the name sort compared raw keys, so ``default/months`` sorted by ``default``.
  {
    const { probeRack, activeProbeNames } = await server.ssrLoadModule("/src/lib/stores/probes.svelte.ts");
    const before = { active: [...probeRack.active], sortMode: probeRack.sortMode };
    probeRack.active = ["direct.indirect", "default/months", "curious.disinterested", "jlens/Paris", "confident.uncertain"];
    probeRack.sortMode = "name";
    assert.deepEqual(activeProbeNames(), ["confident.uncertain", "curious.disinterested", "direct.indirect", "default/months", "jlens/Paris"]);
    probeRack.active = before.active;
    probeRack.sortMode = before.sortMode;
  }
  console.log("Steering rack: one patch path for every term, label binding, mode guards, queued edits, and correlation freshness passed");
} finally { await server.close(); }
