import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "vite";

// Steering in one place (Wave 6, a9): every term lands as a chip in the composer's strip through
// lib/steerActions.ts, opened so its editor shows (the selection stays); "+ Steer" picks every kind in the priority
// order (neurons and heads, lens tokens, concepts, manifold positions, SAE features), the chips sort
// the same way, the instruments keep only one-click Push and Silence, and the Inspector reads the
// units first.
const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({
    mode: "http",
    sessions: { validateSteering: async (expression) => ({ valid: true, expression, error: null }) },
    profiles: { list: async () => ({ profiles: [] }), correlation: async () => null },
    manifolds: { list: async () => ({ manifolds: [] }) },
    templates: {}, tree: {}, events: {},
    probes: { list: async () => ({ probes: [] }) },
    instruments: {
      unitRead: async () => { throw new Error("not asked"); },
      validateSaeFeature: async (id) => ({ id, layer: 12 }),
      sources: async () => ({ sources: [] }),
    },
  });
  const actions = await server.ssrLoadModule("/src/lib/steerActions.ts");
  const steering = await server.ssrLoadModule("/src/lib/stores/steering.svelte.ts");
  const { selection, select, clearSelection, termOpen, closeTermEditor } = await server.ssrLoadModule("/src/lib/stores/selection.svelte.ts");
  const { sessionState } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
  const { steerRack, currentSteeringExpression, removeSteerEntry, patchSteerEntry, lensTokenText } = steering;
  const opened = () => termOpen.name;

  // ---- the helpers: each adds its term and opens its chip
  assert.equal(actions.pushLensToken({ word: " stars" }, "jlens"), "jlens/stars");
  assert.equal(opened(), "jlens/stars");
  assert.equal(actions.pushLensToken({ id: 282, text: " of" }, "slens"), "slens/#282");
  assert.equal(lensTokenText.get(282), " of", "a token named by id reads its text on the chip");
  assert.equal(opened(), "slens/#282");
  assert.equal(actions.pushLensToken({ word: "jlens/" }, "jlens"), null, "no word, no term");
  // A lens term that removes its token turns into a push at the atom's default.
  patchSteerEntry("jlens/stars", { ablate: true, alpha: 1 }, ["jlens"]);
  actions.pushLensToken({ word: "stars" }, "jlens");
  assert.deepEqual([steerRack.entries.get("jlens/stars").ablate, steerRack.entries.get("jlens/stars").alpha], [false, 0.3]);
  assert.equal(actions.pushFeature(17), "sae/17");
  assert.equal(opened(), "sae/17");
  assert.equal(actions.pushFeature(-1), null);
  assert.equal(actions.pushFeature(1.5), null);

  steerRack.catalog = [{
    namespace: "default", name: "emotions", node_count: 3, node_labels: ["calm", "angry", "glad"],
    node_coords: [[0, 0], [1, 0], [0, 1]], intrinsic_dim: 2, domain: { type: "custom" },
    fit_mode: "spectral", resolved_fit_mode: "spectral", fitted_for_session: true,
  }, {
    namespace: "default", name: "formal.casual", node_count: 2, node_labels: ["formal", "casual"],
    node_coords: [[1], [-1]], intrinsic_dim: 1, domain: { type: "custom" },
    fit_mode: "pca", resolved_fit_mode: "pca", fitted_for_session: true,
  }];
  assert.equal(actions.pushConcept("default/formal.casual", "raw", "casual"), "default/formal.casual");
  assert.equal(steerRack.entries.get("default/formal.casual").label, "casual", "a concept's pole");
  assert.equal(opened(), "default/formal.casual");
  assert.equal(actions.pushPosition("default/emotions", "raw", "glad"), "default/emotions");
  assert.deepEqual([steerRack.entries.get("default/emotions").label, steerRack.entries.get("default/emotions").coords], ["glad", [0, 1]]);
  actions.pushPosition("default/emotions", "raw", "angry");
  assert.equal(steerRack.entries.get("default/emotions").label, "angry", "a racked position moves to the node");
  assert.equal(actions.pushUnit("neuron/L3n7", "-"), "neuron/L3n7");
  assert.equal(actions.silenceUnit("L2 h1"), "head/L2h1");
  assert.equal(currentSteeringExpression(),
    "0.5 default/formal.casual%casual + 0.3 jlens/stars + 0.3 slens/#282 + 0.3 sae/17 - 3 neuron/L3n7 + !head/L2h1 + 0.5 default/emotions%angry",
    "the expression keeps its canonical order (the chips sort by priority)");

  // ---- a Push or Silence from the Inspector's token or unit view opens the chip and keeps the view
  // (W6 review: the term had been a selection, so pushing from a token threw the token away).
  const token = { kind: "token", params: { turnIdx: 1, seg: "response", tokenIdx: 2 } };
  select(token);
  actions.pushUnit("neuron/L3n9", "+");
  assert.deepEqual([selection.current, opened()], [token, "neuron/L3n9"], "the token stays selected, the chip opens");
  select({ kind: "unit", id: "head/L2h4" });
  actions.silenceUnit("head/L2h4");
  assert.deepEqual([selection.current, opened()], [{ kind: "unit", id: "head/L2h4" }, "head/L2h4"]);
  removeSteerEntry("neuron/L3n9");
  removeSteerEntry("head/L2h4");

  // ---- the strip: units, lens tokens, flat terms, curved terms, SAE features
  const { render } = await server.ssrLoadModule("svelte/server");
  clearSelection();
  closeTermEditor();
  const { default: RecipeBar } = await server.ssrLoadModule("/src/panels/RecipeBar.svelte");
  const strip = render(RecipeBar, {}).body;
  const names = [...strip.matchAll(/<span class="dw-term-name[^"]*">([^<]*)<\/span>/g)].map((m) => m[1]);
  assert.deepEqual(names, ["L2 h1", "L3 n7", "stars", "of", "casual", "angry", "feature 17"]);
  assert.match(strip, /class="dw-term[^"]*\bis-unit\b/, "a unit's chip names it as every unit row does");

  // ---- "+ Steer": every kind this runtime reads, in the priority order
  sessionState.info = {
    model_id: "m", jlens_fitted: true, architecture: { model_type: "llama", layers: 32, hidden_size: 960, neurons_per_layer: 2560, heads_per_layer: 15, parameters: 1 },
    instruments: [{ family: "sae", source: "s", live: { layer: 12 } }],
  };
  const { default: SteerPicker } = await server.ssrLoadModule("/src/panels/rack/SteerPicker.svelte");
  const picker = render(SteerPicker, { props: { onpicked: () => {} } }).body;
  const tabs = picker.slice(0, picker.indexOf("<form"));
  const kinds = [...tabs.matchAll(/<button[^>]*>([^<]+)</g)].map((m) => m[1]);
  assert.deepEqual(kinds, ["Neuron or head", "Lens token", "Concept", "Manifold position", "SAE feature"]);
  assert.doesNotMatch(picker, /<p class="meta"/, "no caption explains it");
  // W6 integration: a kind whose instrument is missing stays listed and leads to the Artifacts
  // section that prepares it (no lens, no SAE), where the picker had left it out.
  sessionState.info = { model_id: "m" };
  const bare = render(SteerPicker, { props: { onpicked: () => {} } }).body;
  const bareKinds = [...bare.slice(0, bare.indexOf("<form")).matchAll(/<button[^>]*>([^<]+)</g)].map((m) => m[1]);
  assert.deepEqual(bareKinds, ["Neuron or head", "Lens token", "Concept", "Manifold position", "SAE feature"], "no lens, no SAE: still listed");
  const pickerSource = readFileSync(new URL("../src/panels/rack/SteerPicker.svelte", import.meta.url), "utf8");
  assert.match(pickerSource, /section: "lenses"/);
  assert.match(pickerSource, /section: "saes"/);
  assert.match(pickerSource, /goto\("artifacts", missing\.section\)/);
  sessionState.info = {
    model_id: "m", jlens_fitted: true, architecture: { model_type: "llama", layers: 32, hidden_size: 960, neurons_per_layer: 2560, heads_per_layer: 15, parameters: 1 },
    instruments: [{ family: "sae", source: "s", live: { layer: 12 } }],
  };

  // ---- the instruments: Units, Lens, Probes, Features; the panels add no term of their own
  const { default: Instruments } = await server.ssrLoadModule("/src/panels/inspector/Instruments.svelte");
  const inst = render(Instruments, {}).body;
  const heads = [...inst.matchAll(/<span class="head[^"]*" id="instrument-[a-z]+-head">([^<]+)<\/span>/g)].map((m) => m[1]);
  assert.deepEqual(heads.slice(0, 4), ["Units", "Lens", "Probes", "Features"]);
  for (const file of ["JLensPanel", "SaePanel"]) {
    const source = readFileSync(new URL(`../src/panels/${file}.svelte`, import.meta.url), "utf8");
    assert.doesNotMatch(source, />Steer</, `${file} has no Steer field: "+ Steer" picks, a row pushes`);
    assert.doesNotMatch(source, /addJLensToRack|addSLensToRack|addSaeToRack/, `${file} adds terms only through lib/steerActions.ts`);
    assert.match(source, /onpush=\{/, `${file}'s rows push`);
  }
  // W6 review: the concept and manifold dialogs ("+ Steer" → All concepts / All manifolds, the
  // Probes room's A concept / A manifold) added terms with the rack's own mutators and their chips
  // never opened. They steer through lib/steerActions.ts like every other surface.
  const rackDrawer = readFileSync(new URL("../src/drawers/RackDrawer.svelte", import.meta.url), "utf8");
  assert.doesNotMatch(rackDrawer, /addSubspaceToRack|addManifoldToRack|setSteerLabel/, "RackDrawer adds terms only through lib/steerActions.ts");
  assert.match(rackDrawer, /pushConcept\(rowKey\(m\), choice\.variant, label\)/);
  assert.match(rackDrawer, /pushPosition\(rowKey\(m\), choice\.variant\)/);
  // The token view: the units that pushed the token and the heads it read from come first, each
  // with Push (a neuron) and Silence.
  const anatomy = readFileSync(new URL("../src/drawers/token/TokenAnatomy.svelte", import.meta.url), "utf8");
  const markup = anatomy.slice(anatomy.indexOf("{:else if data}"));
  assert.ok(markup.indexOf('aria-label="Units that pushed the token"') < markup.indexOf('aria-label="By depth"'));
  assert.ok(markup.indexOf('aria-label="Where the top heads read from"') < markup.indexOf('aria-label="By depth"'));
  assert.match(anatomy, /pushUnit\(unit, pole \?\? "\+"\)/);
  assert.match(anatomy, /\{@render unitActs\(row\.unit, "head", null\)\}/);
  // W6 review: a − pole push read "Push" here and "Push −" in + Steer and the Units room.
  assert.match(anatomy, /<SteerButton label=\{pole === "-" \? "Push −" : "Push"\}/);
  // W6 review (live): a clicked strip token kept focus while the arrows moved the cursor, so its
  // focus ring marked a second token. A strip token and the track take no focus from a click.
  const tokenStrip = readFileSync(new URL("../src/panels/rooms/TokenStrip.svelte", import.meta.url), "utf8");
  assert.match(tokenStrip, /class="t"[^>]*onmousedown=\{keepFocus\}/);
  assert.match(tokenStrip, /class="track"[^>]*onmousedown=\{keepFocus\}/);
  assert.match(tokenStrip, /function keepFocus\(event: MouseEvent\): void \{\s*event\.preventDefault\(\);/);
  // The unit view's actions put a term in and open it; only the strip takes one out.
  const unitView = readFileSync(new URL("../src/panels/inspector/UnitView.svelte", import.meta.url), "utf8");
  assert.doesNotMatch(unitView, /toggleUnitTerm/);
  assert.ok(unitView.indexOf('class="dw-sec acts"') < unitView.indexOf('class="dw-sec reads"'), "what a unit can do comes first");

  // ---- behind a running reply the last click wins (W6 review): Silence then Push + on a pushed
  // neuron ends pushed, Push then Silence ends silenced, and Push − then Push + on a silenced one
  // ends on the + pole at the default. Each queued edit decides against the rack as the edits
  // before it leave it, not against the rack as it was when clicked.
  const { genStatus } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const pending = await server.ssrLoadModule("/src/lib/stores/pending.svelte.ts");
  const drain = async () => {
    genStatus.active = false;
    while (pending.pendingActions.queue.length > 0) await pending.drainNextPendingAction();
  };
  const unitTerm = (id) => {
    const entry = steerRack.entries.get(id);
    return entry ? [entry.ablate, entry.alpha] : null;
  };
  actions.pushUnit("neuron/L4n2", "+");
  patchSteerEntry("neuron/L4n2", { alpha: 0.7 }, ["unit"]);
  genStatus.active = true;
  actions.silenceUnit("neuron/L4n2");
  actions.pushUnit("neuron/L4n2", "+");
  assert.deepEqual(unitTerm("neuron/L4n2"), [false, 0.7], "nothing lands while the reply runs");
  await drain();
  assert.deepEqual(unitTerm("neuron/L4n2"), [false, 3], "Silence then Push +: pushed, at the default (it was silenced when the push landed)");
  genStatus.active = true;
  actions.pushUnit("neuron/L4n2", "-");
  actions.silenceUnit("neuron/L4n2");
  await drain();
  assert.deepEqual(unitTerm("neuron/L4n2"), [true, 1], "Push then Silence: silenced");
  genStatus.active = true;
  actions.pushUnit("neuron/L4n2", "-");
  actions.pushUnit("neuron/L4n2", "+");
  await drain();
  assert.deepEqual(unitTerm("neuron/L4n2"), [false, 3], "Push − then Push +: the + pole, keeping the strength the first push set");
  genStatus.active = true;
  actions.pushUnit("neuron/L4n2", "+");
  await drain();
  assert.deepEqual(unitTerm("neuron/L4n2"), [false, 3], "a push the unit already holds changes nothing");

  for (const name of [...steerRack.entries.keys()]) removeSteerEntry(name);
  sessionState.info = null;
  console.log("Steer actions: the helpers, the strip's order, + Steer's kinds, the instruments' order and their one-click shortcuts passed");
} finally {
  await server.close();
}
