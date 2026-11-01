import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const flush = () => new Promise((resolve) => setImmediate(resolve));
const envelope = { measurements: { version: 1, scope: "replay", provenance: "replayed", instruments: {
  lens: { binding: { source: "test", steering: null }, readings: {}, readout: { layers: [], aggregate: [] } },
  sae: { binding: { source: "test", steering: null, layer: 0 }, readings: {}, readout: { features: [] } },
  geometry: { binding: { source: "test", steering: null }, readings: {} },
} } };

try {
  const requests = [];
  let replay = (...args) => new Promise((resolve, reject) => requests.push({ args, resolve, reject }));
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "browser", instruments: {
    tokenReadout: (...args) => replay(...args),
  } });
  const { cachedTokenReadout, invalidateTokenReadoutCache, tokenReadoutCacheEpoch } =
    await server.ssrLoadModule("/src/lib/runtime/tokenReadoutCache.ts");

  const active = cachedTokenReadout("lens", "active", 1, {});
  const queued = cachedTokenReadout("lens", "obsolete", 2, {});
  const unaffected = cachedTokenReadout("sae", "other-family", 3, {});
  const rejectedActive = assert.rejects(active, /conversation or reading source changed/);
  const rejectedQueued = assert.rejects(queued, /conversation or reading source changed/);
  invalidateTokenReadoutCache("lens");
  await Promise.all([rejectedActive, rejectedQueued]);
  const options = { topK: 5, steered: true };
  const fresh = cachedTokenReadout("lens", "active", 1, options);
  options.steered = false;
  assert.equal(requests.length, 1, "invalidation must not overlap GPU jobs");
  requests[0].resolve(envelope);
  await flush();
  assert.equal(requests.length, 2);
  assert.equal(requests[1].args[1], "active");
  assert.equal(requests[1].args[3].steered, true, "queued options retain their keyed identity");
  requests[1].resolve(envelope);
  await fresh;
  await flush();
  assert.equal(requests[2].args[1], "other-family");
  requests[2].resolve(envelope);
  await unaffected;
  await flush();
  assert.equal(requests.some(({ args }) => args[1] === "obsolete"), false);
  await cachedTokenReadout("lens", "active", 1, { topK: 5, steered: true });
  assert.equal(requests.length, 3, "only the fresh result is reused");

  const { ReplayReadout } = await server.ssrLoadModule("/src/drawers/token/readout.svelte.ts");
  const view = new ReplayReadout();
  view.clear();
  let attempts = 0;
  replay = async () => {
    if (++attempts === 1) throw new Error("temporary read failure");
    return envelope;
  };
  const map = () => ({ data: { strength: 0.4 }, source: "test-source" });
  view.replay("lens", "retry", 0, {}, map);
  await flush();
  assert.match(view.error, /temporary read failure/);
  view.replay("lens", "retry", 0, {}, map);
  await flush();
  assert.equal(attempts, 2);
  assert.deepEqual(view.data, { strength: 0.4 });
  assert.equal(view.origin, "replayed");
  // A replay that completes a saved row stands on the row's capture: the summary shows (as
  // captured) while it loads and stays beside the error when it fails (a 422 for a gated recipe,
  // a 409 while busy), rather than leaving only the error.
  const saved = { strength: 0.2, layers: [] };
  replay = async () => { throw new Error("422: the recipe is gated on a probe"); };
  view.replay("lens", "saved-row", 3, { steered: true }, map, { data: saved, source: "row-source" });
  assert.deepEqual(view.data, saved, "the saved summary shows while the replay loads");
  assert.equal(view.origin, "captured");
  assert.equal(view.loading, true);
  await flush();
  assert.deepEqual(view.data, saved, "a failed replay keeps the saved summary");
  assert.equal(view.origin, "captured");
  assert.equal(view.source, "row-source");
  assert.match(view.error, /gated on a probe/);
  replay = async () => envelope;
  view.replay("lens", "saved-row", 3, { steered: true }, map, { data: saved, source: "row-source" });
  await flush();
  assert.deepEqual(view.data, { strength: 0.4 }, "a replay that lands replaces it");
  assert.equal(view.origin, "replayed");
  assert.equal(view.error, null);
  view.replay("lens", "unsaved-row", 3, { steered: false }, map, { data: saved, source: "row-source" });
  view.clear();
  replay = async () => { throw new Error("busy"); };
  view.replay("lens", "unsaved-row", 4, { steered: false }, map);
  await flush();
  assert.equal(view.data, null, "a base never outlives the call that set it");
  assert.equal(view.origin, null);
  replay = async () => envelope;
  const { layersNote } = await server.ssrLoadModule("/src/drawers/token/readings.ts");
  assert.match(layersNote(true, true, "the probe is attached"), /replay while this model is loaded and the probe is attached/);
  assert.match(layersNote(true, false, "x"), /no replay record/);
  assert.match(layersNote(false, true, "x"), /does not replay them/);
  view.dispose();
  assert.equal(view.data, null);
  replay = async () => ({ measurements: { ...envelope.measurements, instruments: {} } });
  await assert.rejects(cachedTokenReadout("lens", "missing-result", 0, {}), /No reading was returned/);
  await flush();
  replay = async () => envelope;
  assert.deepEqual(await cachedTokenReadout("lens", "missing-result", 0, {}), envelope,
    "a missing result is neither measured zero nor a reusable cache hit");
  await flush();

  const { applyTreeSnapshot, applyTreeDelta, loomTree } =
    await server.ssrLoadModule("/src/lib/stores/loom.svelte.ts");
  const node = (id, parent_id) => ({
    id, parent_id, role: parent_id ? "assistant" : "system", text: id,
    recipe: null, applied_steering: null, finish_reason: "stop", tokens: [],
  });
  const snapshot = {
    tree_format: 2, drowse_version: "test", session_id: "default", model_id: "fixture",
    root_id: "root", active_node_id: "a", rev: 1,
    nodes: [node("root", null), node("a", "root")],
    children_of: { root: ["a"], a: [] }, cast: {},
  };
  applyTreeSnapshot(snapshot);
  const epoch = tokenReadoutCacheEpoch("lens");
  applyTreeDelta({ rev: 2, updated: [{ ...loomTree.nodes.get("a"), starred: true }] });
  assert.equal(tokenReadoutCacheEpoch("lens"), epoch, "stars do not alter replay inputs");
  applyTreeDelta({ rev: 3, updated: [{ ...loomTree.nodes.get("root"), text: "new system prompt" }] });
  assert.equal(tokenReadoutCacheEpoch("lens"), epoch + 1, "ancestor edits invalidate descendant replays");
  applyTreeSnapshot({ ...snapshot, rev: 4 });
  assert.equal(tokenReadoutCacheEpoch("lens"), epoch + 2, "same node IDs in restored chats are not cache hits");
  applyTreeSnapshot({ ...snapshot, rev: 5, model_id: "other-model" }, { reconcileEdgeLabels: true });
  assert.equal(tokenReadoutCacheEpoch("lens"), epoch + 3, "model swaps invalidate every family");

  const { highlightScoreFor } = await server.ssrLoadModule("/src/lib/highlight.ts");
  assert.equal(highlightScoreFor({ text: "x", score: 0.9, probes: { old: 0.9 } }, "new"), undefined);
  assert.equal(highlightScoreFor({ text: "x", probes: { probe: 0 } }, "probe"), 0);
  assert.equal(highlightScoreFor({ text: "x", coordsByProbe: { probe: [0.1, -0.2] } }, "probe[1]"), -0.2);

  const { probeRack, highlightScale, probeEntryForDisplay, highlightState } =
    await server.ssrLoadModule("/src/lib/stores/probes.svelte.ts");
  const { tokenHoverState } = await server.ssrLoadModule("/src/lib/stores/instruments.svelte.ts");
  probeRack.entries.set("word alias", { info: { family: "lens", namespace: "jlens", name: "word alias", word: "word", layers: [0] },
    current: 0.5, previous: 0.5, sparkline: [0.5], reading: null, aggregate: null });
  probeRack.entries.set("read alias", { info: { family: "lens", namespace: "slens", name: "read alias", word: "word", layers: [0] },
    current: 1.5, previous: 1.5, sparkline: [1.5], reading: null, aggregate: null });
  probeRack.entries.set("feature alias", { info: { family: "sae", name: "feature alias", feature_id: 7, max_act: 10, layers: [0] },
    current: 5, previous: 5, sparkline: [5], reading: { value: 5, unit: "raw_activation", per_layer: {}, depth: null }, aggregate: null });
  probeRack.active = ["word alias", "feature alias"];
  assert.equal(highlightScale("word alias"), 1);
  assert.equal(highlightScale("feature alias"), 5, "reading units override newly cached feature metadata");
  const { highlightStyleFor } = await server.ssrLoadModule("/src/lib/highlight.ts");
  highlightState.target = "word alias";
  assert.match(highlightStyleFor({ text: "x", probes: { "word alias": 0.5 } }).backgroundColor, /var\(--tint\)/, "a lens word tints at the depth tint");
  tokenHoverState.active = true;
  tokenHoverState.lensAggregate = [[" word", 0.9, 0.5, 0.1]];
  assert.equal(probeEntryForDisplay("word alias").reading, null, "hover cannot borrow a whitespace variant's probability");
  tokenHoverState.lensAggregate = [["word", 0.2, 0.5, 0.1]];
  assert.equal(probeEntryForDisplay("word alias").reading.value, 0.2);
  // A word as read is the S lens's footprint z: never filled from the R lens's probability, and
  // re-attached as the slens/ probe it is.
  assert.equal(probeEntryForDisplay("read alias").reading, null, "an slens/ probe never borrows the R read");
  tokenHoverState.probes = { "read alias": 2.5 };
  assert.equal(probeEntryForDisplay("read alias").reading.unit, "mean_footprint_z");
  tokenHoverState.probes = null;
  const { probeRequestFromInfo } = await server.ssrLoadModule("/src/lib/stores/probes.svelte.ts");
  assert.deepEqual(probeRequestFromInfo(probeRack.entries.get("read alias").info), { selector: "slens/word", name: "read alias" });
  assert.deepEqual(probeRequestFromInfo(probeRack.entries.get("word alias").info), { selector: "jlens/word", name: "word alias" });
  tokenHoverState.saeReadout = [{ id: 7, activation: 3, max_act: null }];
  assert.equal(probeEntryForDisplay("feature alias").reading.unit, "raw_activation");
  assert.equal(probeEntryForDisplay("feature alias").reading.value, 3, "historical rows are not normalized using current metadata");
  // A captured reading without its traces (a node's summary) draws its layer strip from the token's
  // replay, its value from the capture.
  probeRack.entries.set("calm", { info: { family: "geometry", name: "calm", is_affine: true, layers: [3, 7] },
    current: 0, previous: 0, sparkline: [], reading: null, aggregate: null });
  const calmSummary = { fraction: 0.4, nearest: [["calm", 1]], coords: [0.25], residual: 0, assignment: [], membership: 1, depth_com: [], depth_spread: [] };
  const calmReplayed = { ...calmSummary, coords: [0.3], depth_com: [0.5], coords_per_layer: { 3: [0.1], 7: [0.3] }, fraction_per_layer: {}, residual_per_layer: {} };
  tokenHoverState.probeReadings = { calm: calmSummary };
  tokenHoverState.replayedReadings = null;
  assert.deepEqual(probeEntryForDisplay("calm").perLayer, {}, "a summary has no strip of its own");
  assert.equal(probeEntryForDisplay("calm").current, 0.25);
  tokenHoverState.replayedReadings = { calm: calmReplayed };
  assert.deepEqual(probeEntryForDisplay("calm").perLayer, { 3: 0.1, 7: 0.3 }, "the strip comes from the replay");
  assert.equal(probeEntryForDisplay("calm").current, 0.25, "the value stays the capture's");
  // A generated row keeps the value (its geometry coordinates) and nothing else: the card shows it at
  // once, and the replayed reading (its depth, its strip) once it lands, beside the row's value.
  tokenHoverState.probeReadings = null;
  tokenHoverState.replayedReadings = null;
  tokenHoverState.rowGeometry = { calm: { fraction: 0.4, coords: [0.25] } };
  tokenHoverState.probes = { calm: 0.25 };
  assert.equal(probeEntryForDisplay("calm").current, 0.25, "a generated row's value shows at once");
  tokenHoverState.replayedReadings = { calm: calmReplayed };
  assert.equal(probeEntryForDisplay("calm").reading.depth_com[0], 0.5, "the replay brings the rest of the reading");
  assert.deepEqual(probeEntryForDisplay("calm").perLayer, { 3: 0.1, 7: 0.3 });
  assert.equal(probeEntryForDisplay("calm").current, 0.25, "the value stays the row's");
  tokenHoverState.probes = null;
  tokenHoverState.rowGeometry = null;
  tokenHoverState.replayedReadings = null;
  tokenHoverState.active = false;
  probeRack.active = [];
  probeRack.entries.clear();
  highlightState.target = null;

  const { sparklinePaths } = await server.ssrLoadModule("/src/lib/charts/sparklinePath.ts");
  assert.deepEqual(sparklinePaths([], 60, 16, 1), { line: "", area: "" });
  assert.deepEqual(sparklinePaths([null], 60, 16, 1), { line: "", area: "" });
  assert.equal(sparklinePaths([1], 60, 16, 1).line, "M 0 0.00 L 60 0.00");
  assert.equal(sparklinePaths([0], 60, 16, 1).line, "M 0 8.00 L 60 8.00");
  const gaps = sparklinePaths([1, 0.5, null, 0.25, 0], 40, 16, 1);
  assert.equal(gaps.line, "M 0.00 0.00 L 10.00 4.00 M 30.00 6.00 L 40.00 8.00");
  assert.equal((gaps.area.match(/Z/g) ?? []).length, 2, "area fills cannot bridge missing measurements");
  assert.deepEqual(sparklinePaths([NaN, Infinity], 60, 16, 1), { line: "", area: "" });

  const { render } = await server.ssrLoadModule("svelte/server");
  const { default: Alternatives } = await server.ssrLoadModule("/src/drawers/token/Alternatives.svelte");
  const chosen = { text: "x", tokenId: 1, logprob: Math.log(0.9996),
    topAlts: [{ text: "x", id: 1, logprob: Math.log(0.9996) }] };
  const altProps = { canBranch: true, busy: false, onbranch() {} };
  const logitHtml = render(Alternatives, { props: { token: chosen, ...altProps } }).body;
  assert.match(logitHtml, /not a measure of factual accuracy/);
  assert.doesNotMatch(render(Alternatives, { props: { token: { ...chosen, tokenId: 2 }, ...altProps } }).body,
    /factual accuracy/, "a top alternative's probability is not the chosen token's probability");
  assert.doesNotMatch(logitHtml, /Branch with/, "the chosen token offers no branch to itself");
  const alts = { text: " sky", tokenId: 2, logprob: Math.log(0.3), topAlts: [
    { text: " moon", id: 1, logprob: Math.log(0.5) }, { text: " sky", id: 2, logprob: Math.log(0.3) },
  ] };
  const altHtml = render(Alternatives, { props: { token: alts, ...altProps } }).body;
  assert.doesNotMatch(altHtml, /Branch with/, "a browser runtime without token replay cannot fork");
  assert.equal((altHtml.match(/class="alt rise [^"]*chosen/g) ?? []).length, 1, "the chosen alternative is marked once, by id");

  const layers = await server.ssrLoadModule("/src/drawers/token/layers.ts");
  const read = [
    { layer: 0, tokens: [{ id: 9, token: " the", logprob: Math.log(0.4) }] },
    { layer: 1, tokens: [{ id: 2, token: " sky", logprob: Math.log(0.2) }, { id: 9, token: " the", logprob: Math.log(0.1) }] },
    { layer: 2, tokens: [{ id: 5, token: "sky", logprob: Math.log(0.6) }] },
    { layer: 3, tokens: [{ id: 2, token: " sky", logprob: Math.log(0.7) }] },
  ];
  assert.deepEqual(layers.chosenByDepth(read, 2).map(row => row.p === null ? null : +row.p.toFixed(2)), [null, 0.2, null, 0.7],
    "a layer that never ranked the token is unmeasured, and a whitespace variant never stands in for it");
  assert.equal(layers.leadLayer(read, 2), 3, "the lead is where the token stays first to the last layer");
  assert.equal(layers.leadLayer(read, 9), null, "a token the last layer does not read first has no lead");
  assert.equal(layers.leadLayer(read, null), null);
  assert.equal(layers.depthTint(0, 35), "color-mix(in oklch, var(--deep) 100%, var(--late))");
  assert.equal(layers.depthTint(35, 35), "color-mix(in oklch, var(--deep) 0%, var(--late))");
  assert.equal(layers.lastLayer(read), 3);
  const many = Array.from({ length: 36 }, (_, i) => i);
  const folded = layers.foldedLayers(many, 25);
  assert.ok(folded.has(0) && folded.has(33) && folded.has(35) && folded.has(25) && !folded.has(1), "every third, the last and the lead");
  assert.equal(layers.foldedLayers([0, 1, 2], null).size, 3, "a short readout does not fold");
  assert.equal(layers.formatP(0.4123), "0.41");
  assert.equal(layers.formatP(0.0042), "0.004");
  assert.equal(layers.formatP(null), "–");
  assert.equal(layers.formatSigned(-0.5), "−0.50");
  assert.equal(layers.formatSigned(0.25), "+0.25");
  assert.equal(layers.formatSigned(-0.001), "0.00", "a reading that rounds to zero carries no sign");

  const readings = await server.ssrLoadModule("/src/drawers/token/readings.ts");
  const pin = (value) => ({ value, unit: "mean_token_probability", per_layer: {}, depth: null });
  const captured = { "lens-pin": pin(0.25) };
  const replayedLens = { "lens-pin": pin(0.24) };
  const sae = { "sae-pin": pin(0.5) };
  const tokenPins = (o) => readings.tokenPins({
    anatomy: false, steered: true, captured: null, anatomyPins: null, lensPins: null, saePins: null, ...o,
  });
  assert.deepEqual(tokenPins({ lensPins: captured, saePins: sae }), { ...captured, ...sae },
    "each family's pins are its shown read's (a capture's)");
  assert.deepEqual(tokenPins({ lensPins: replayedLens }), replayedLens,
    "a replayed lens readout lists the pins it read: a generated row keeps none of its own");
  assert.deepEqual(tokenPins({}), {}, "no read yet, no pins");
  // Where the anatomy stands in for the readout: steered, the pins captured with a live token, else
  // the anatomy's own (a generated row, an unsteered view).
  assert.deepEqual(tokenPins({ anatomy: true, captured, anatomyPins: replayedLens }), captured);
  assert.deepEqual(tokenPins({ anatomy: true, captured: null, anatomyPins: replayedLens }), replayedLens,
    "a generated row's steered anatomy lists the pins it replayed");
  assert.deepEqual(tokenPins({ anatomy: true, steered: false, captured, anatomyPins: replayedLens }), replayedLens,
    "an unsteered anatomy lists its own (unsteered) pins, never the captured ones");
  assert.deepEqual(tokenPins({ anatomy: true, lensPins: replayedLens }), {}, "the hidden readout's pins never show");
  // Steered, a replayed pin keeps the value the row kept (its score); unsteered, it is all replay.
  const scores = { "lens-pin": 0.25 };
  assert.equal(tokenPins({ anatomy: true, anatomyPins: replayedLens, scores })["lens-pin"].value, 0.25);
  assert.equal(tokenPins({ anatomy: true, steered: false, anatomyPins: replayedLens, scores })["lens-pin"].value, 0.24);
  assert.equal(tokenPins({ saePins: { "lens-pin": pin(0.3) }, saeSteered: false, scores })["lens-pin"].value, 0.3);
  const { withRowValues } = await server.ssrLoadModule("/src/lib/tokenRows.ts");
  const replayedGeometry = { calm: { fraction: 0.41, nearest: [["calm", 1]], coords: [0.26], residual: 0, assignment: [], membership: 1, depth_com: [0.5], depth_spread: [0.1] } };
  assert.deepEqual(withRowValues(replayedGeometry, { calm: { fraction: 0.4, coords: [0.25] } }).calm,
    { ...replayedGeometry.calm, fraction: 0.4, coords: [0.25] }, "a generated row's share and coordinates stand; the rest replays");
  assert.deepEqual(withRowValues(replayedGeometry, {}), replayedGeometry);
  // The lens side replays as the anatomy (any active lens, an S lens alone included) or as the
  // readout (the R lens only); the view and the WebMCP inspector gate Unsteered on the same answer.
  const gate = (o) => readings.lensReplay({ anatomy: false, tokenReadout: false, rLens: false, activeLens: false, replayContext: true, ...o });
  assert.deepEqual(gate({ anatomy: true, activeLens: true }), { available: true, ready: true, anatomy: true },
    "an S lens alone is enough for the anatomy");
  assert.deepEqual(gate({ tokenReadout: true, activeLens: true }), { available: true, ready: false, anatomy: false },
    "the readout needs the R lens");
  assert.deepEqual(gate({ rLens: true, activeLens: true }), { available: false, ready: true, anatomy: false });
  assert.deepEqual(gate({ anatomy: true, tokenReadout: true }), { available: true, ready: false, anatomy: false },
    "no lens ready: the readout's view (the capture, or the missing-lens state) stands in");
  assert.deepEqual(gate({ anatomy: true, tokenReadout: true, rLens: true, replayContext: false }), { available: true, ready: true, anatomy: false },
    "a token without its replay record keeps the lens readout's captured view");
  assert.equal(readings.probeNote("confident.uncertain", "confident"), "", "a first pole beside its own name says nothing");
  assert.equal(readings.probeNote("default/Formal.casual", "formal"), "");
  assert.equal(readings.probeNote("individualist.collectivist", "axis 1"), "axis 1");
  assert.equal(readings.probeNote("emotions", "joy"), "joy", "a curved probe's nearest node is a reading");
  assert.equal(readings.probeNote("emotions", ""), "");

  const { default: LensReading } = await server.ssrLoadModule("/src/drawers/token/LensReading.svelte");
  const { default: ProbeReadings } = await server.ssrLoadModule("/src/drawers/token/ProbeReadings.svelte");
  const { default: FeatureReadings } = await server.ssrLoadModule("/src/drawers/token/FeatureReadings.svelte");
  const readout = {
    data: { token_id: 2, token_text: " word", steering: null,
      aggregate: [{ token: " word", id: 11, strength: 0.1, com: 0.5, spread: 0.1 }],
      layers: [{ layer: 0, tokens: [{ token: "word", id: 1, logprob: Math.log(0.8) }] }],
      layer: 0, features: [],
    },
    origin: "captured", source: "original", progress: null, loading: false, error: null,
  };
  const lensHtml = render(LensReading, { props: {
    readout, tokenText: " word", jlensFitted: true, hasReplayContext: true, modelId: "fixture", replayAvailable: true,
  } }).body;
  assert.match(lensHtml, /in 0\/1 layers/, "a whitespace variant is not the aggregate token's layer probability");
  assert.match(lensHtml, /Reads as/);
  assert.doesNotMatch(lensHtml, /Attention added|MLP added/, "the anatomy's other columns wait for their data");
  assert.match(lensHtml, /below every layer's top 1/, "a token no layer ranked is unmeasured, not zero, and draws no depth");
  assert.doesNotMatch(lensHtml, /class="bars/);
  assert.match(lensHtml, /\(this token\)/, "the inspected token is named as such across layers");
  assert.doesNotMatch(lensHtml, /two reads agree/, "the inspected token does not borrow Token's agreement flag");
  const unfitted = render(LensReading, { props: {
    readout: { ...readout, data: null, origin: null }, tokenText: " word", jlensFitted: false, hasReplayContext: true, modelId: "org/fixture", replayAvailable: true,
  } }).body;
  assert.match(unfitted, /No lens for fixture/);
  const scalar = { value: 0.25, unit: "mean_token_probability", per_layer: { 0: 0.1, 1: 0.4 }, depth: null };
  const geometry = { ...readout, data: { steering: null, readings: {} } };
  const probeProps = { readout: geometry, hasGeometryProbes: false, hasReplayContext: true, replayAvailable: true,
    returnToToken: { turnIdx: 0, tokenIdx: 0, isThinking: false } };
  assert.match(render(ProbeReadings, { props: { ...probeProps, pinned: { "original-probe": scalar } } }).body, /original-probe/);
  assert.match(render(ProbeReadings, { props: { ...probeProps, pinned: {} } }).body, /Add a probe/, "an empty roster offers to add one");
  const featureProps = { readout, saeLoaded: true, availability: "available", hasReplayContext: true, replayAvailable: true, lastLayer: 3 };
  assert.match(render(FeatureReadings, { props: featureProps }).body, /None fired here/);
  const fired = { ...readout, data: { ...readout.data, layer: 2, features: [
    { id: 7, activation: 3, label: "night sky", max_act: 6 }, { id: 9, activation: 2, label: null, max_act: null },
  ] } };
  const featureHtml = render(FeatureReadings, { props: { ...featureProps, readout: fired } }).body;
  assert.match(featureHtml, /L2 f7/);
  assert.match(featureHtml, /night sky/);
  assert.match(featureHtml, />0\.50</, "a feature with a reference maximum reads relative to it");
  assert.match(featureHtml, />2\.0</, "a feature without one reads raw");
  assert.match(render(FeatureReadings, { props: { ...featureProps, readout: { ...readout, data: null }, saeLoaded: false } }).body,
    /Choose an SAE/, "no SAE leads to the Lens room");

  const { default: TokenLogitsPopover } = await server.ssrLoadModule("/src/lib/ui/TokenLogitsPopover.svelte");
  const popover = render(TokenLogitsPopover, { props: {
    token: { text: " sky", tokenId: 2, logprob: Math.log(0.3), topAlts: [{ id: 2, text: " sky", logprob: Math.log(0.3) }] },
    anchor: {}, isCurrent: () => true, contextChanged: true, onclose() {},
  } }).body;
  assert.match(popover, /data-tip="Model token · sampled probabilities, not raw logits"/);
  assert.match(popover, /data-tip-sub="Recorded before your edit, with the original text before it"/);
  assert.doesNotMatch(popover, /<p class="meta[^"]*">[^<]*(not raw logits|Recorded before your edit)/,
    "what the numbers are goes in the title's tip, not a line in the working view");

  // A probe's trace is the latest generated reply's token rows (what the reply is tinted by), so it
  // covers the whole reply and survives a reload; its layers follow the token asked for, from the
  // token's own reading. A saved row keeps a geometry reading's summary only (no per-layer trace):
  // there the layers are null and the inspector replays the token (``replyToken`` is its anchor).
  const { chatLog } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const { replyTrace, replyLayers, replyToken } = await server.ssrLoadModule("/src/panels/rack/probeTrace.ts");
  const flatReading = (coord, coordsPerLayer) => ({ version: 2, scope: "token", provenance: "captured", instruments: {
    geometry: { readings: { calm: {
      fraction: 0.4, nearest: [], coords: [coord], residual: 0, assignment: [], membership: 1, depth_com: [], depth_spread: [],
      ...(coordsPerLayer ? { coords_per_layer: coordsPerLayer, fraction_per_layer: {}, residual_per_layer: {} } : {}),
    } } },
    lens: { binding: { sources: { R: "local:r" }, steering: null }, readings: {
      "jlens/sky": { value: 0.3, unit: "mean_token_probability", per_layer: { 5: 0.1, 6: 0.5 }, depth: null },
    } },
  } });
  chatLog.turns = [
    { role: "assistant", text: "older", generated: true, tokens: [{ text: "a", probes: { calm: 9 } }] },
    { role: "user", text: "hi", tokens: [{ text: "hi", probes: { calm: 7 } }] },
    { role: "assistant", text: "abc", generated: true, nodeId: "n1", tokens: [
      { text: "a", rawIndex: 0, probes: { calm: 0.5 }, measurements: flatReading(0.5, { 3: [0.2, 9], 7: [-0.4, 9] }) },
      { text: "b", rawIndex: 1, coordsByProbe: { calm: [-0.25, 1] } },
      { text: "c", rawIndex: 2, probes: { calm: NaN }, measurements: flatReading(0.9, null) },
    ] },
  ];
  assert.deepEqual(replyTrace("calm"), [0.5, -0.25, null], "the latest reply only, coordinates first, non-finite as gaps");
  assert.deepEqual(replyTrace("calm[1]"), [null, 1, null], "an axis target reads that coordinate");
  assert.equal(replyTrace("absent"), null, "a reply without the probe gives no trace");
  assert.deepEqual(replyLayers("calm", 0), { 3: 0.2, 7: -0.4 }, "a live token's layers are its reading's axis 0");
  assert.equal(replyLayers("calm"), null, "a saved summary keeps no layers: they replay");
  assert.deepEqual(replyToken(), { token: chatLog.turns[2].tokens[2], nodeId: "n1", rawIndex: 2 }, "the replay anchor defaults to the last token");
  assert.equal(replyLayers("calm", 1), null);
  assert.deepEqual(replyLayers("jlens/sky"), { 5: 0.1, 6: 0.5 }, "a scalar reading keeps its few layers on the row");

  // A curved probe reads the share on its surface: its trace and layers come from each token's
  // geometry reading, never from the rows' axis-0 views (which would draw a coordinate against the
  // share's 0..1 scale). A token without that reading is a gap; an axis target still reads its axis.
  const { readsShare } = await server.ssrLoadModule("/src/panels/rack/probeTrace.ts");
  const shareReading = (fraction, perLayer) => ({ version: 2, scope: "token", provenance: "captured", instruments: {
    geometry: { readings: { "default/months": { fraction, fraction_per_layer: perLayer, coords: [2.64, -1], nearest: [], residual: 0, coords_per_layer: { 3: [2.1, 0] }, residual_per_layer: {}, assignment: [], membership: 1, depth_com: [], depth_spread: [] } } },
  } });
  chatLog.turns = [
    { role: "assistant", text: "abc", generated: true, tokens: [
      { text: "a", probes: { "default/months": 2.64 }, coordsByProbe: { "default/months": [2.64, -1] },
        measurements: shareReading(0.39, { 3: 0.2, 7: 0.41 }) },
      { text: "b", probes: { "default/months": 1.5 } },
      { text: "c", probes: { "default/months": 3 }, measurements: shareReading(0.5, { 3: 0.25 }) },
    ] },
  ];
  assert.deepEqual(replyTrace("default/months", { share: true }), [0.39, null, 0.5], "a curved probe's trace is its share, token by token");
  assert.deepEqual(replyTrace("default/months"), [2.64, 1.5, 3], "without share the trace reads axis 0");
  assert.deepEqual(replyTrace("default/months[1]", { share: true }), [-1, null, null], "an axis target reads that axis even on a curved probe");
  assert.deepEqual(replyLayers("default/months", 0, { share: true }), { 3: 0.2, 7: 0.41 }, "a curved probe's layers are its share by layer");
  assert.deepEqual(replyLayers("default/months", 0), { 3: 2.1 });
  assert.equal(replyLayers("default/months", 1, { share: true }), null, "a token without a geometry reading keeps no share");
  chatLog.turns = [{ role: "assistant", text: "b", generated: true, tokens: [{ text: "b", probes: { "default/months": 1.5 } }] }];
  assert.equal(replyTrace("default/months", { share: true }), null, "with no per-token share the row falls back to its live buffer");
  assert.equal(readsShare({ family: "geometry", is_affine: false }), true);
  assert.equal(readsShare({ family: "geometry", is_affine: true }), false);
  assert.equal(readsShare({ family: "lens" }), false);
  chatLog.turns = [];

  // A probe's title is its ID as its row writes it.
  const { probeTitle, foundLensRows, lensReadoutCells } = await server.ssrLoadModule("/src/panels/rack/probeRows.ts");
  // Regression: two byte pieces that both decode to U+FFFD are two found rows with two keys (keyed
  // by text they collided and the keyed each threw each_key_duplicate mid-reply).
  const found = foundLensRows([["\uFFFD", 0.3, 0.5, 0.1, 5], ["\uFFFD", 0.2, 0.6, 0.1, 6], [" Paris", 0.1, 0.9, 0.1, 7]], [{ namespace: "jlens", id: 7 }]);
  assert.deepEqual(found.map(row => row.key), ["aggregate:5", "aggregate:6"], "keyed by id; a pinned id keeps its own row");
  assert.deepEqual(found.map(row => row.id), [5, 6]);
  assert.equal(foundLensRows([["\uFFFD", 0.3, 0.5, 0.1, 5]], [{ namespace: "jlens", id: 6 }]).length, 1, "a pinned id hides only itself, not its text");
  // Regression: an As read (slens/) pin hid the word's R discovery row and its As said pin action.
  assert.deepEqual(foundLensRows([[" Paris", 0.1, 0.9, 0.1, 42]], [{ namespace: "slens", id: 42 }]).map(row => row.id), [42],
    "an slens/ pin leaves the found probability row");
  assert.deepEqual(foundLensRows([[" Paris", 0.1, 0.9, 0.1, 42]], [{ namespace: "slens", id: 42 }, { namespace: "jlens", id: 42 }]), [],
    "a jlens/ pin of the same id takes its place");
  const { lensReadoutSnapshot } = await server.ssrLoadModule("/src/lib/stores/instruments.svelte.ts");
  const lensSnap = lensReadoutSnapshot({
    layers: [{ layer: 0, tokens: [{ token: "\uFFFD", id: 5, logprob: Math.log(0.8) }, { token: "\uFFFD", id: 6, logprob: Math.log(0.1) }] }],
    aggregate: [{ token: "\uFFFD", id: 5, strength: 0.3, com: 0.5, spread: 0.1 }],
  });
  assert.deepEqual(lensSnap.aggregate, [["\uFFFD", 0.3, 0.5, 0.1, 5]], "the snapshot keeps the wire's id");
  assert.deepEqual(lensSnap.readout["0"].map(([, , id]) => id), [5, 6], "per-layer reads keep their ids");
  // Regression: cells were joined by text, so id 6 showed id 5's probability.
  const cells = (id) => lensReadoutCells(lensSnap.readout, [0, 1], id).map(cell => cell.p === null ? null : Number(cell.p.toFixed(3)));
  assert.deepEqual(cells(5), [0.8, null]);
  assert.deepEqual(cells(6), [0.1, null]);
  assert.deepEqual(probeTitle({ family: "sae", feature_id: 7 }, "sae/7", 20), { lead: false, text: "L20 f7" });
  assert.deepEqual(probeTitle({ family: "sae", feature_id: 7 }, "sae/7"), { lead: false, text: "f7" }, "no resident layer, no L");
  assert.deepEqual(probeTitle({ family: "lens", word: "sea" }, "jlens/sea"), { lead: false, text: "sea" });
  assert.deepEqual(probeTitle({ family: "lens", word: " sea" }, "jlens/ sea"), { lead: true, text: "sea" }, "a word start is marked, not spelled");
  assert.deepEqual(probeTitle({ family: "geometry" }, "default/confident.uncertain"), { lead: false, text: "confident.uncertain" });

  const { default: JLensProbeCard } = await server.ssrLoadModule("/src/panels/rack/JLensProbeCard.svelte");
  const { default: SaeProbeCard } = await server.ssrLoadModule("/src/panels/rack/SaeProbeCard.svelte");
  const card = { token: "word", probeName: "custom lens alias", strength: null,
    com: null, spread: null, series: [], cells: [], pinned: true };
  // Regression: the pair suffix sat flush against the word ("Parisread"), and the two pins of a
  // word sorted read before said, the reverse of the As said | As read toggle.
  assert.match(render(JLensProbeCard, { props: { ...card, pairWord: "read" } }).body, /<span class="pair[^"]*"> read<\/span>/, "the suffix keeps its space");
  const { pairLensPins } = await server.ssrLoadModule("/src/panels/rack/probeRows.ts");
  const pins = [
    { key: "other", token: "sea", pinned: true, namespace: "jlens" },
    { key: "read", token: "Paris", pinned: true, namespace: "slens" },
    { key: "found", token: "Paris", pinned: false, namespace: "jlens" },
    { key: "said", token: "Paris", pinned: true, namespace: "jlens" },
  ];
  assert.deepEqual(pairLensPins(pins).map(row => row.key), ["other", "said", "read", "found"]);
  const lensCard = render(JLensProbeCard, { props: card }).body;
  assert.match(lensCard, /Not measured/);
  assert.match(lensCard, /Unpin probe custom lens alias/);
  assert.doesNotMatch(lensCard, /Strength 0/);
  const saeCard = render(SaeProbeCard, { props: {
    id: 7, probeName: "custom feature alias", layer: 0, value: 0, measured: false,
    series: [], pinned: true,
  } }).body;
  assert.match(saeCard, /Not measured/);
  assert.match(saeCard, /Unpin probe custom feature alias/);
  assert.doesNotMatch(saeCard, /Activation 0/);

  // Regression: after a reload the rack, the Lens panel's pins and the inspector lost what the live
  // view showed (a curved probe's share, a lens pin's value, nearest, residual, depth), though the
  // tree keeps it: the node's flat aggregate and its aggregate reading (aggregate_measurements; a
  // generated row keeps only its scores and geometry shares and coordinates). The hydrate reads both.
  {
    const { hydrateProbeRackFromActiveNode, probeRack: rack } =
      await server.ssrLoadModule("/src/lib/stores/probes.svelte.ts");
    const { applyTreeSnapshot: applySnapshot } = await server.ssrLoadModule("/src/lib/stores/loom.svelte.ts");
    const months = { family: "geometry", name: "default/months", is_affine: false, intrinsic_dim: 2, layers: [0, 1] };
    const paris = { family: "lens", namespace: "slens", name: "slens/Paris", word: "Paris", layers: [0, 1] };
    const blank = { sparkline: [], current: 0, previous: 0, perLayer: {}, reading: null, aggregate: null,
      savedAggregate: null, savedReading: null, nearest: [], trajectory: [], subspaceTrail: [] };
    rack.entries.clear();
    rack.entries.set("default/months", { request: { selector: "default/months" }, info: months, ...blank });
    rack.entries.set("slens/Paris", { request: { selector: "slens/Paris" }, info: paris, ...blank });
    const monthsReading = { fraction: 0.341, nearest: [["october", 24.19]], coords: [2.37, 0.5], residual: 1.02,
      assignment: [["october", 0.9]], membership: 0.8, depth_com: [0.5], depth_spread: [0.26] };
    const parisReading = { value: 0.9205, unit: "mean_footprint_z", per_layer: { 0: 0.8, 1: 1.04 },
      depth: { center: [0.6], spread: [0.1], basis: "footprint_z" } };
    const row = (fraction) => ({ text: "x", raw_index: 0, measurements: { version: 2, scope: "token", provenance: "captured",
      instruments: { geometry: { readings: { "default/months": { fraction, coords: [2.3, 0.4] } } } },
      scores: { "default/months": 2.3, "slens/Paris": 0.1 } } });
    const aggregate = { version: 2, scope: "aggregate", provenance: "captured", instruments: {
      geometry: { readings: { "default/months": monthsReading } },
      lens: { binding: { sources: { R: null, S: "s" }, steering: null }, readings: { "slens/Paris": parisReading } } } };
    applySnapshot({
      tree_format: 2, drowse_version: "test", session_id: "default", model_id: "fixture",
      root_id: "root", active_node_id: "reply", rev: 9, children_of: { root: ["reply"], reply: [] }, cast: {},
      nodes: [
        { id: "root", parent_id: null, role: "system", text: "", recipe: null, applied_steering: null, finish_reason: null, tokens: null, thinking_tokens: null, aggregate_readings: {} },
        { id: "reply", parent_id: "root", role: "assistant", text: "x y", recipe: null, applied_steering: null, finish_reason: "stop",
          aggregate_readings: { "default/months": 2.3732, "slens/Paris": 0.9205 },
          aggregate_measurements: aggregate,
          tokens: [row(0.2), row(0.3), { text: "\n" }],
          thinking_tokens: null },
      ],
    });
    hydrateProbeRackFromActiveNode();
    const m = rack.entries.get("default/months");
    assert.equal(m.savedFraction, 0.341, "a curved probe's share comes from the node's aggregate reading");
    assert.equal(m.current, 0.341);
    assert.deepEqual(m.savedCoordinates, [2.3732, 0.5], "axis 0 from the exact aggregate, the rest from its reading");
    assert.equal(m.savedReading.residual, 1.02);
    assert.deepEqual(m.nearest, [["october", 24.19]]);
    const p = rack.entries.get("slens/Paris");
    assert.equal(p.savedAggregate, 0.9205, "a lens pin keeps its saved value");
    assert.equal(p.savedReading.depth.center[0], 0.6);
    assert.deepEqual(p.perLayer, { 0: 0.8, 1: 1.04 }, "a scalar reading's band comes back from the aggregate");
    rack.entries.clear();
  }

  console.log("Faithful readout checks passed (cache lifetimes, provenance, token identity, missing data, and aliases)");
} finally {
  await server.close();
}
