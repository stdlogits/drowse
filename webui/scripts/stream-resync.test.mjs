import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

// A reader that lags its token frames gets one ``resync`` (``ws_stream._OutboundQueue``): the
// server dropped frames while generation went on into the tree. The client rebuilds the streaming
// turn from the authoritative tree, holds the frames that arrive meanwhile, and dedupes them by
// raw index: no token twice, none lost, and the live rows it already drew are kept as they are.
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const tick = () => new Promise((resolve) => setImmediate(resolve));
try {
  const words = Array.from({ length: 16 }, (_, index) => `w${index} `);
  // A curved probe (months: its share is the primary reading) and a flat one (calm: its score).
  // The tree's rows are the generated row's slim projection (``token_row_measurements``: the
  // scores, each geometry reading's fraction and coords); a live frame carries the envelope whole.
  const share = (index) => 0.5 + index / 100;
  const scores = (index) => ({ months: index / 10, calm: -index / 4 });
  const envelope = (readings, index) => ({
    version: 2, scope: "token", provenance: "captured",
    instruments: { geometry: { readings } }, scores: scores(index),
  });
  const slim = (index) => envelope({ months: { fraction: share(index), coords: [index / 10, 1] } }, index);
  const whole = (index) => envelope({ months: {
    fraction: share(index), coords: [index / 10, 1], nearest: [["may", 0.4]], assignment: { may: 1 },
    residual: 0.2, membership: 0.8, depth: { "3": share(index) },
  } }, index);
  const row = (index) => ({ token_id: index, text: words[index], raw_index: index, logprob: -0.5, measurements: slim(index) });
  const reply = (rows, finish = null, text = "") => ({
    id: "reply", parent_id: "ask", role: "assistant", text, role_label: null, thinking_text: null,
    aggregate_readings: {}, applied_steering: null, finish_reason: finish, starred: false, notes: "",
    created_at: 0, edited_at: null, edit_count: 0, mean_logprob: null, mean_surprise: null,
    recipe: { steering: null, sampling: null, thinking: false, seed: null, probes: [], probe_hashes: {} },
    tokens: rows, thinking_tokens: [], raw_token_ids: null, children: [], aggregate_measurements: null,
  });
  const ask = {
    id: "ask", parent_id: "root", role: "user", text: "tell me", role_label: null, thinking_text: null,
    aggregate_readings: {}, applied_steering: null, finish_reason: null, starred: false, notes: "",
    created_at: 0, edited_at: null, edit_count: 0, mean_logprob: null, mean_surprise: null, recipe: null,
    tokens: [], thinking_tokens: [], raw_token_ids: null, aggregate_measurements: null,
  };
  const root = { ...ask, id: "root", parent_id: null, role: "system", text: "" };
  const tree = (rev, nodes, active) => ({
    tree_format: 2, drowse_version: "test", session_id: "default", model_id: "fixture", name: null,
    root_id: "root", active_node_id: active, rev,
    nodes, children_of: Object.fromEntries(nodes.map((node) => [node.id, nodes.filter((child) => child.parent_id === node.id).map((child) => child.id)])),
    cast: {},
  });

  let snapshot = tree(1, [root, ask], "ask");
  let pendingRead = null;
  let reads = 0;
  const listeners = new Set();
  const stateListeners = new Set();
  let open = false;
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({
    mode: "http",
    sessions: { get: async () => ({}) },
    profiles: { list: async () => ({ profiles: [] }) },
    probes: { list: async () => ({ probes: [] }) },
    manifolds: { list: async () => ({ manifolds: [] }) },
    templates: {},
    tree: {
      get: () => {
        reads += 1;
        if (reads === 1) return Promise.resolve(structuredClone(snapshot));
        return new Promise((resolve) => { pendingRead = () => resolve(structuredClone(snapshot)); });
      },
    },
    events: {
      get isOpen() { return open; },
      async open() { open = true; },
      subscribe: (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
      subscribeState: (listener) => { stateListeners.add(listener); return () => stateListeners.delete(listener); },
      send() {},
      async stop() {},
      acknowledgeSnapshot() {},
      close() { open = false; },
    },
  });
  const { ensureRuntimeChannel } = await server.ssrLoadModule("/src/lib/stores/ws.svelte.ts");
  const { chatLog, genStatus } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const { loomTree } = await server.ssrLoadModule("/src/lib/stores/loom.svelte.ts");
  const { turnTrace } = await server.ssrLoadModule("/src/panels/rack/probeTrace.ts");
  await ensureRuntimeChannel({ sessionFresh: true });
  const emit = (message) => { for (const listener of listeners) listener(message); };
  const token = (index) => ({
    type: "token", text: words[index], thinking: false, token_id: index, node_id: "reply", raw_index: index, logprob: -0.5,
    measurements: whole(index),
  });
  const turn = () => chatLog.turns.find((candidate) => candidate.nodeId === "reply");

  emit({ type: "started", generation_id: "g", node_id: null, sibling_index: 0, sibling_count: 1 });
  emit({ type: "tree_mutated", op: "begin_assistant", rev: 2, added: [reply([])], removed: [], updated: [], active_node_id: "reply", cast: {} });
  for (const index of [0, 1, 2, 3]) emit(token(index));
  const live = turn().tokens.slice();
  assert.equal(turn().text, words.slice(0, 4).join(""));

  // Frames 4..9 were dropped; the tree holds rows 0..10 when the client reads it.
  snapshot = tree(2, [root, ask, reply([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(row))], "reply");
  emit({ type: "resync", dropped: 6 });
  // Live again while the read is in flight: 10 is in the snapshot too, 11 is not.
  emit(token(10));
  emit(token(11));
  assert.equal(turn().tokens.length, 4, "frames wait for the snapshot");
  assert.equal(typeof pendingRead, "function", "a resync reads the authoritative tree");
  pendingRead();
  await tick();
  await tick();
  assert.deepEqual(turn().tokens.map((score) => score.rawIndex), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
  assert.equal(turn().text, words.slice(0, 12).join(""));
  assert.ok(live.every((score, index) => turn().tokens[index] === score), "the drawn rows are kept as they are");
  assert.equal(genStatus.tokensSoFar, 12);
  // The rack's trace and the thread's tint read the caught-up slim rows as they read the live
  // ones: a curved probe's share, a flat probe's score, no gap where the frames were dropped.
  const indices = [...Array(12).keys()];
  assert.deepEqual(turnTrace(turn(), "months", { share: true }), indices.map(share), "the share reads every row");
  assert.deepEqual(turnTrace(turn(), "calm"), indices.map((index) => scores(index).calm), "the score reads every row");
  assert.deepEqual(turnTrace(turn(), "months[1]"), indices.map(() => 1), "a curved probe's second axis reads every row");
  assert.ok(turn().tokens.slice(4, 11).every((score) => Object.keys(score.measurements.instruments.geometry.readings.months).sort().join() === "coords,fraction"),
    "the caught-up rows are the tree's slim rows");
  assert.equal(genStatus.active, true);

  emit(token(12));
  const final = reply([...Array(13).keys()].map(row), "length", words.slice(0, 13).join(""));
  emit({ type: "tree_mutated", op: "finalize_assistant", rev: 3, added: [], removed: [], updated: [final], active_node_id: "reply", cast: {} });
  emit({ type: "done", node_id: "reply", sibling_index: 0, sibling_count: 1, result: {
    text: final.text, tokens: 13, finish_reason: "length", applied_steering: null, usage: {}, mean_logprob: -0.5, mean_surprise: 0.5,
  } });
  assert.equal(turn().text, final.text);
  assert.equal(turn().finishReason, "length");
  assert.equal(genStatus.active, false);
  assert.equal(loomTree.rev, 3);

  // A stateless reply (the A/B comparison) has no node to catch up from: its gapped token rows
  // give way to the whole text on ``done``, so the bubble (which draws rows whenever it has any)
  // never shows a reply with words missing.
  const { sendShadowGenerate, abState } = await server.ssrLoadModule("/src/lib/stores/ab.svelte.ts");
  turn().generated = true;
  const steeredIdx = chatLog.turns.indexOf(turn());
  await sendShadowGenerate(steeredIdx);
  assert.equal(abState.processingAb, true);
  const shadowToken = (index) => ({ ...token(index), node_id: null, raw_index: index });
  emit({ type: "started", generation_id: "s", node_id: null, sibling_index: 0, sibling_count: 1 });
  emit(shadowToken(0));
  snapshot = tree(3, [root, ask, final], "reply");
  pendingRead = null;
  emit({ type: "resync", dropped: 1 });
  assert.equal(typeof pendingRead, "function");
  pendingRead();
  await tick();
  await tick();
  emit(shadowToken(2));
  const shadow = () => chatLog.turns[steeredIdx].abPair;
  assert.deepEqual(shadow().tokens.map((score) => score.text), [words[0], words[2]], "the stream has the gap");
  emit({ type: "done", node_id: null, sibling_index: 0, sibling_count: 1, result: {
    text: words.slice(0, 3).join(""), tokens: 3, finish_reason: "length", applied_steering: null, usage: {},
  } });
  await tick();
  const compared = chatLog.turns[steeredIdx].abPair;
  assert.equal(compared.text, words.slice(0, 3).join(""), "the comparison's text is whole");
  assert.equal(compared.tokens?.length ?? 0, 0, "its gapped rows give way to the text");

  // A server connection that drops mid-reply says so (not "the local model is not ready") and
  // reconnects, which reads the tree: the page shows the reply as the server kept it.
  const { toasts } = await server.ssrLoadModule("/src/lib/stores/toasts.svelte.ts");
  const second = { ...reply([0, 1].map(row)), id: "reply2", text: "" };
  emit({ type: "started", generation_id: "g2", node_id: null, sibling_index: 0, sibling_count: 1 });
  emit({ type: "tree_mutated", op: "begin_assistant", rev: 4, added: [second], removed: [], updated: [], active_node_id: "reply2", cast: {} });
  emit({ ...token(0), node_id: "reply2" });
  const kept = { ...second, tokens: [0, 1, 2, 3, 4].map(row), text: words.slice(0, 5).join(""), finish_reason: "cancelled" };
  snapshot = tree(5, [root, ask, final, kept], "reply2");
  pendingRead = null;
  open = false;
  for (const listener of [...stateListeners]) listener({ state: "closed", expected: false, reason: "closed (1011)", code: "RUNTIME_CONNECTION_LOST" });
  assert.ok(toasts.entries.some((entry) => /connection to the drowse server dropped/.test(entry.message)),
    "the toast names the dropped connection");
  assert.ok(!toasts.entries.some((entry) => /local model is not ready/.test(entry.message)));
  await new Promise((resolve) => setTimeout(resolve, 5));
  await tick();
  assert.equal(typeof pendingRead, "function", "the client reconnects and reads the tree");
  pendingRead();
  await tick();
  await tick();
  const caught = chatLog.turns.find((candidate) => candidate.nodeId === "reply2");
  assert.equal(caught.tokens.length, 5, "the reply shows every row the server kept");
  assert.equal(caught.finishReason, "cancelled");
  console.log("stream resync: a lagging reader rebuilds its reply from the tree, no token lost or doubled");
} finally {
  await server.close();
}
