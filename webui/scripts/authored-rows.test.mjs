// Authored (prompt) token rows: slim like a generated row, replayed through their node's
// ``prompt_capture``. A prompt token's replay address is its place among the node's captured rows
// (thinking rows, then visible rows); it never forks, the token view reads it under the steering
// its capture ran, and the hover replays it like a generated token, keeping the row's own values.
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const flush = () => new Promise((resolve) => setImmediate(resolve));

try {
  const { authoredRowIndex, replayIndex, promptCaptureProblem, wholeEnvelope } =
    await server.ssrLoadModule("/src/lib/tokenRows.ts");
  const { replayAnchor } = await server.ssrLoadModule("/src/drawers/token/readings.ts");

  const capture = {
    steering: "0.3 calm", system_prompt: null, thinking: true, raw: false,
    start: 9, thinking_start: 4, digest: "a".repeat(64),
  };
  const slim = (token_id, text, fraction) => ({
    token_id, text, logprob: null, perplexity: null,
    measurements: { version: 2, scope: "token", provenance: "captured",
      instruments: { geometry: { readings: { calm: { fraction, coords: [0.25] } } } }, scores: { calm: 0.25 } },
  });
  const blank = { role_label: null, applied_steering: null, aggregate_readings: {}, aggregate_measurements: null,
    starred: false, notes: "", created_at: 1, edited_at: null, edit_count: 0, mean_logprob: null, mean_surprise: null };
  const root = { ...blank, id: "root", parent_id: null, role: "system", text: "", thinking_text: null, recipe: null,
    finish_reason: null, tokens: null, thinking_tokens: null, raw_token_ids: null, prompt_capture: null };
  const user = { ...blank, id: "u", parent_id: "root", role: "user", text: "Hi there", thinking_text: "hm",
    recipe: null, finish_reason: null, raw_token_ids: null, prompt_capture: capture,
    thinking_tokens: [slim(5, "hm", 0.1)], tokens: [slim(10, "Hi", 0.2), slim(11, " there", 0.3)] };
  const reply = { ...blank, id: "r", parent_id: "u", role: "assistant", text: "ok", thinking_text: null,
    recipe: { steering: "0.3 calm", sampling: null, thinking: null, seed: null, probes: [], probe_hashes: {} },
    applied_steering: "0.3 calm", finish_reason: "stop", raw_token_ids: [7], prompt_capture: null,
    thinking_tokens: null, tokens: [{ ...slim(7, "ok", 0.4), raw_index: 0 }] };

  // The helpers: one index space per authored node, thinking first; a generated row keeps its own.
  assert.equal(promptCaptureProblem(user), null);
  assert.equal(promptCaptureProblem(reply), null);
  assert.equal(wholeEnvelope(user.tokens[0].measurements), null, "an authored row is the generated row's projection");
  assert.equal(authoredRowIndex(user, true, 0), 0);
  assert.equal(authoredRowIndex(user, false, 0), 1);
  assert.equal(authoredRowIndex(user, false, 1), 2);
  assert.equal(authoredRowIndex({ ...user, prompt_capture: null }, false, 0), null, "an uncaptured turn has no anchor");
  assert.equal(authoredRowIndex(reply, false, 0), null, "a generated row is addressed by its raw index");
  assert.equal(replayIndex({ rawIndex: 4, promptIndex: null }), 4);
  assert.equal(replayIndex({ promptIndex: 2 }), 2);
  assert.equal(replayIndex({}), null);
  assert.equal(replayIndex(null), null);
  // Regression: the old whole authored rows (no anchor) are refused, as by the loom reader.
  assert.match(promptCaptureProblem({ ...user, prompt_capture: null }), /replay anchor/);
  assert.match(promptCaptureProblem({ ...user, tokens: [{ ...slim(10, "Hi", 0.2), raw_index: 0 }] }), /raw index/);

  // The token view's replay target and the steering it reads under (the Steered/Unsteered switch
  // shows where there is one): a prompt token's is its capture's.
  assert.deepEqual(replayAnchor(user, { promptIndex: 2 }, null), { index: 2, steering: "0.3 calm" });
  assert.deepEqual(replayAnchor({ ...user, prompt_capture: { ...capture, steering: null } }, { promptIndex: 2 }, null),
    { index: 2, steering: null });
  assert.deepEqual(replayAnchor(reply, { rawIndex: 0 }, "ignored"), { index: 0, steering: "0.3 calm" });
  assert.deepEqual(replayAnchor({ ...reply, recipe: { ...reply.recipe, steering: "" } }, { rawIndex: 0 }, null),
    { index: 0, steering: null }, "an empty expression is none");
  assert.deepEqual(replayAnchor(user, { text: "x" }, null), { index: null, steering: "0.3 calm" });

  // A reload rehydrates the prompt turn's tokens with their replay address and no fork address.
  const requests = [];
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "browser", instruments: {
    tokenReadout: async (family, nodeId, rawIndex, options) => {
      requests.push([family, nodeId, rawIndex, options.steered]);
      return { measurements: { version: 2, scope: "replay", provenance: "replayed", instruments: {
        geometry: { binding: { source: null, steering: "0.3 calm" }, readings: { calm: {
          fraction: 0.29, nearest: [["calm", 1.2]], coords: [0.26], residual: 0.1, assignment: [["calm", 1]],
          membership: 1, depth_com: [0.5], depth_spread: [0.1],
          fraction_per_layer: { 3: 0.29 }, coords_per_layer: { 3: [0.26] }, residual_per_layer: { 3: 0.1 },
        } } },
      } } };
    },
  } });
  const { applyTreeSnapshot } = await server.ssrLoadModule("/src/lib/stores/loom.svelte.ts");
  const { chatLog } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  applyTreeSnapshot({
    tree_format: 2, drowse_version: "test", session_id: "default", model_id: "fixture", name: null,
    root_id: "root", active_node_id: "r", rev: 3, nodes: [root, user, reply],
    children_of: { root: ["u"], u: ["r"], r: [] }, cast: {},
  });
  const userTurn = chatLog.turns.find((turn) => turn.nodeId === "u");
  const replyTurn = chatLog.turns.find((turn) => turn.nodeId === "r");
  assert.deepEqual(userTurn.thinkingTokens.map((t) => t.promptIndex), [0]);
  assert.deepEqual(userTurn.tokens.map((t) => t.promptIndex), [1, 2]);
  assert.equal([...userTurn.tokens, ...userTurn.thinkingTokens].some((t) => t.rawIndex != null), false,
    "a prompt token never forks");
  assert.deepEqual(replyTurn.tokens.map((t) => [t.rawIndex, t.promptIndex]), [[0, undefined]]);

  // The hover on a prompt token: the row's share at once, the rest replayed at its prompt index.
  const { sessionState } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
  const { beginTokenHover, endTokenHover, tokenHoverState } =
    await server.ssrLoadModule("/src/lib/stores/instruments.svelte.ts");
  sessionState.info = { ...(sessionState.info ?? {}), model_id: "fixture", instruments: [
    { family: "geometry", source: null, capabilities: { token_readout: true } },
  ] };
  beginTokenHover(userTurn.tokens[1], "u");
  assert.equal(tokenHoverState.rowGeometry.calm.fraction, 0.3, "the row's own share shows at once");
  assert.equal(tokenHoverState.probeReadings, null, "a slim row is not a whole envelope");
  await new Promise((resolve) => setTimeout(resolve, 400));
  await flush();
  assert.deepEqual(requests, [["geometry", "u", 2, true]], "the hover replays the prompt token at its index");
  assert.deepEqual(tokenHoverState.replayedReadings.calm.nearest, [["calm", 1.2]], "the rest comes from the replay");
  assert.deepEqual(tokenHoverState.replayedReadings.calm.coords_per_layer, { 3: [0.26] });
  endTokenHover();

  console.log("Authored row checks passed (prompt index, replay target, rehydration, hover replay)");
} finally {
  await server.close();
}
