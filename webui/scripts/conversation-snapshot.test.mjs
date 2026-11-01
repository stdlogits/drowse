import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const {
    buildConversationImportPlan,
    isConversationSnapshot,
    runConversationImportTransaction,
  } = await server.ssrLoadModule("/src/lib/conversationSnapshot.ts");

  const samplingState = {
    temperature: 0.7,
    top_p: 0.95,
    top_k: null,
    max_tokens: 256,
    seed: null,
    system_prompt: "",
    stop_sequences: "",
    logit_bias_text: "",
    presence_penalty: 0,
    frequency_penalty: 0,
    thinking: false,
    return_top_k: 8,
    user_role: "user",
    assistant_role: "assistant",
  };
  const snapshot = {
    version: 8,
    savedAt: "2026-08-29T12:00:00.000Z",
    model_id: "fixture/model",
    session_id: "default",
    tree: {
      tree_format: 2,
      drowse_version: "fixture",
      model_id: "fixture/model",
      session_id: "default",
      name: null,
      root_id: "root",
      active_node_id: "root",
      rev: 0,
      nodes: [],
      children_of: {},
      cast: {},
    },
    steerRack: [{
      name: "local/calm",
      mode: "subspace",
      ablate: true,
      coords: [1],
      label: "calm",
      variant: "raw",
      trigger: "BOTH",
      enabled: true,
    }, {
      name: "jlens/fixture",
      mode: "jlens",
      ablate: true,
      alpha: 0.3,
      trigger: "RESPONSE",
      enabled: true,
    }, {
      name: "sae/17",
      mode: "sae",
      ablate: false,
      alpha: 0.4,
      trigger: "BOTH",
      enabled: true,
    }, {
      name: "local/emotions",
      mode: "manifold",
      blend: 0.4,
      onto: 0.7,
      coords: [0.2, 0.8],
      label: "happy",
      variant: "sae-release-a",
      trigger: "RESPONSE",
      enabled: true,
    }],
    subspaceAlong: 0.5,
    readOnlyRecipe: null,
    probeRack: {
      sortMode: "change",
      active: ["mood alias", "latent alias"],
      entries: [{
        name: "mood alias",
        request: {
          selector: "local/emotions:sae-release-a",
          name: "mood alias",
          top_n: 4,
        },
        sparkline: [0.1, 0.2],
        current: 0.2,
        previous: 0.1,
      }, {
        name: "latent alias",
        request: { selector: "sae/17", name: "latent alias" },
        sparkline: [0.3],
        current: 0.3,
        previous: 0.25,
      }],
    },
    highlightState: {
      target: "mood alias",
      compareTarget: "latent alias",
      compareTwo: true,
      smoothBlend: false,
    },
    samplingState,
  };

  assert.equal(isConversationSnapshot(snapshot, Object.keys(samplingState)), true);

  const pythonSnapshot = structuredClone(snapshot);
  pythonSnapshot.tree = JSON.parse(readFileSync(new URL(
    "../../tests/fixtures/loom_tree_v2.json", import.meta.url,
  ), "utf8"));
  pythonSnapshot.tree.model_id = pythonSnapshot.model_id;
  assert.equal(pythonSnapshot.tree.session_id, null);
  assert.equal(isConversationSnapshot(pythonSnapshot, Object.keys(samplingState)), true);
  assert.equal(buildConversationImportPlan(pythonSnapshot).snapshot.tree.session_id, null);
  for (const sessionId of [undefined, "unrelated-session"]) {
    const invalidIdentity = structuredClone(pythonSnapshot);
    invalidIdentity.tree.session_id = sessionId;
    assert.equal(isConversationSnapshot(invalidIdentity, Object.keys(samplingState)), false);
  }
  const wrongModel = structuredClone(pythonSnapshot);
  wrongModel.tree.model_id = "different/model";
  assert.equal(isConversationSnapshot(wrongModel, Object.keys(samplingState)), false);
  const plan = buildConversationImportPlan(snapshot);
  assert.equal(
    plan.steeringExpression,
    "0.5 !local/calm + 0.3 !jlens/fixture@response + 0.4 sae/17 + 0.4,0.7 local/emotions:sae-release-a%happy@response",
  );
  assert.deepEqual(plan.probeRequests, [
    {
      selector: "local/emotions:sae-release-a",
      name: "mood alias",
      top_n: 4,
    },
    { selector: "sae/17", name: "latent alias" },
  ]);
  assert.equal(plan.steerEntries.get("local/emotions").variant, "sae-release-a");
  assert.equal(plan.steerEntries.get("jlens/fixture").ablate, true);

  // v8 is exact: every ablatable row carries its ablation flag, triggers are the five the chips
  // offer (no grammar aliases), and there is no typed text recipe.
  for (const index of [0, 1, 2]) {
    const noAblation = structuredClone(snapshot);
    delete noAblation.steerRack[index].ablate;
    assert.equal(isConversationSnapshot(noAblation, Object.keys(samplingState)), false, `row ${index} without ablate`);
  }
  for (const alias of ["PROMPT", "GENERATED"]) {
    const aliasTrigger = structuredClone(snapshot);
    aliasTrigger.steerRack[0].trigger = alias;
    assert.equal(isConversationSnapshot(aliasTrigger, Object.keys(samplingState)), false, alias);
  }
  const v7 = structuredClone(snapshot);
  v7.version = 7;
  delete v7.readOnlyRecipe;
  v7.customSteeringExpression = "0.3 jlens/orange";
  assert.equal(isConversationSnapshot(v7, Object.keys(samplingState)), false, "a v7 text recipe is refused");
  const textRecipe = structuredClone(snapshot);
  textRecipe.customSteeringExpression = null;
  assert.equal(isConversationSnapshot(textRecipe, Object.keys(samplingState)), false, "the v7 field is refused in v8");
  const held = structuredClone(snapshot);
  held.steerRack = [];
  held.readOnlyRecipe = "0.4 local/a~local/b";
  assert.equal(isConversationSnapshot(held, Object.keys(samplingState)), true);
  assert.equal(buildConversationImportPlan(held).steeringExpression, "0.4 local/a~local/b");
  const heldBesideChips = structuredClone(held);
  heldBesideChips.steerRack = structuredClone(snapshot.steerRack);
  assert.equal(isConversationSnapshot(heldBesideChips, Object.keys(samplingState)), false, "chips beside a read-only recipe");
  const emptyHeld = structuredClone(held);
  emptyHeld.readOnlyRecipe = "";
  assert.equal(isConversationSnapshot(emptyHeld, Object.keys(samplingState)), false, "an empty recipe is no recipe (null)");
  const unknownTrigger = structuredClone(snapshot);
  unknownTrigger.steerRack[0].trigger = "SOMETIMES";
  assert.equal(isConversationSnapshot(unknownTrigger, Object.keys(samplingState)), false);

  // Unit terms: a neuron pushed either pole or ablated, a head only ablated; the reference exact.
  const unitRows = structuredClone(snapshot);
  unitRows.steerRack.push(
    { name: "neuron/L31n4410", mode: "unit", ablate: false, alpha: -0.2, trigger: "BOTH", enabled: true },
    { name: "head/L28h12", mode: "unit", ablate: true, alpha: 1, trigger: "RESPONSE", enabled: true },
  );
  assert.equal(isConversationSnapshot(unitRows, Object.keys(samplingState)), true);
  const unitPlan = buildConversationImportPlan(unitRows);
  assert.match(unitPlan.steeringExpression, /- 0\.2 neuron\/L31n4410 \+ !head\/L28h12@response/);
  for (const [name, ablate, alpha] of [
    ["head/L28h12", false, 0.2], ["neuron/L31h4", true, 0.2], ["neuron/L031n4", false, 0.2], ["unit/L1n2", true, 0.2],
    // An ablation blends toward the neutral mean; a negative α would amplify the unit instead.
    ["neuron/L30n637", true, -0.2], ["head/L28h12", true, -1],
  ]) {
    const bad = structuredClone(snapshot);
    bad.steerRack.push({ name, mode: "unit", ablate, alpha, trigger: "BOTH", enabled: true });
    assert.equal(isConversationSnapshot(bad, Object.keys(samplingState)), false, `${name} ${ablate} ${alpha}`);
  }
  const blend = structuredClone(snapshot);
  blend.steerRack.push({ name: "neuron/L30n637", mode: "unit", ablate: true, alpha: 0.5, trigger: "BOTH", enabled: true });
  assert.equal(isConversationSnapshot(blend, Object.keys(samplingState)), true, "a partial ablation blends");

  const oldVersion = structuredClone(snapshot);
  oldVersion.version = 6;
  assert.equal(isConversationSnapshot(oldVersion, Object.keys(samplingState)), false);
  const lostAlias = structuredClone(snapshot);
  delete lostAlias.probeRack.entries[0].request.name;
  assert.equal(isConversationSnapshot(lostAlias, Object.keys(samplingState)), false);
  const lostCurveVariant = structuredClone(snapshot);
  delete lostCurveVariant.steerRack[3].variant;
  assert.equal(isConversationSnapshot(lostCurveVariant, Object.keys(samplingState)), false);

  const preflightOrder = [];
  await assert.rejects(
    runConversationImportTransaction({
      capture() {
        preflightOrder.push("capture");
        return "old";
      },
      async preflight() {
        preflightOrder.push("preflight");
        throw new Error("missing instrument");
      },
      async apply() { preflightOrder.push("apply"); },
      async rollback() { preflightOrder.push("rollback"); },
    }),
    /missing instrument/,
  );
  assert.deepEqual(preflightOrder, ["capture", "preflight"]);

  let workspace = "original";
  const rollbackOrder = [];
  await assert.rejects(
    runConversationImportTransaction({
      capture() {
        rollbackOrder.push("capture");
        return workspace;
      },
      async preflight() { rollbackOrder.push("preflight"); },
      async apply() {
        rollbackOrder.push("apply");
        workspace = "partial import";
        throw new Error("attach failed");
      },
      async rollback(previous) {
        rollbackOrder.push("rollback");
        workspace = previous;
      },
    }),
    /attach failed/,
  );
  assert.equal(workspace, "original");
  assert.deepEqual(rollbackOrder, ["capture", "preflight", "apply", "rollback"]);

  process.stdout.write("conversation snapshot tests passed\n");
} finally {
  await server.close();
}
