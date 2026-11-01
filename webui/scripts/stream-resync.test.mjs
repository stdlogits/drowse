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
  const row = (index) => ({ token_id: index, text: words[index], raw_index: index, logprob: -0.5 });
  const reply = (rows, finish = null, text = "") => ({
    id: "reply", parent_id: "ask", role: "assistant", text, role_label: null, thinking_text: null,
    aggregate_readings: {}, applied_steering: null, finish_reason: finish, starred: false, notes: "",
    created_at: 0, edited_at: null, edit_count: 0, mean_logprob: null, mean_surprise: null,
    recipe: { steering: null, sampling: null, thinking: false, seed: null, probes: [], probe_hashes: {} },
    tokens: rows, thinking_tokens: [], raw_token_ids: null, children: [],
  });
  const ask = {
    id: "ask", parent_id: "root", role: "user", text: "tell me", role_label: null, thinking_text: null,
    aggregate_readings: {}, applied_steering: null, finish_reason: null, starred: false, notes: "",
    created_at: 0, edited_at: null, edit_count: 0, mean_logprob: null, mean_surprise: null, recipe: null,
    tokens: [], thinking_tokens: [], raw_token_ids: null,
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
      subscribeState: () => () => {},
      send() {},
      async stop() {},
      acknowledgeSnapshot() {},
      close() { open = false; },
    },
  });
  const { ensureRuntimeChannel } = await server.ssrLoadModule("/src/lib/stores/ws.svelte.ts");
  const { chatLog, genStatus } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const { loomTree } = await server.ssrLoadModule("/src/lib/stores/loom.svelte.ts");
  await ensureRuntimeChannel({ sessionFresh: true });
  const emit = (message) => { for (const listener of listeners) listener(message); };
  const token = (index) => ({
    type: "token", text: words[index], thinking: false, token_id: index, node_id: "reply", raw_index: index, logprob: -0.5,
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
  console.log("stream resync: a lagging reader rebuilds its reply from the tree, no token lost or doubled");
} finally {
  await server.close();
}
