import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

// Page bootstrap reads the tree, the session, the profiles, the probes and the manifolds once each.
// Regression (Wave 3 finding 24): bootstrap fetched them all, then opened the event channel, whose
// rehydrate fetched them all again, so every load asked for each twice (a 2.4 MB tree among them).
// Bootstrap now opens the channel and lets its rehydrate read; if the channel fails it reads once.
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const tree = JSON.parse(readFileSync(new URL("../../tests/fixtures/loom_tree_v2.json", import.meta.url), "utf8"));
  tree.model_id = "fixture/model";
  const info = {
    id: "default", model_id: "fixture/model", device: "cpu", dtype: "torch.float32", created: 0,
    config: { temperature: 1, top_p: 0.9, top_k: null, max_tokens: 256, system_prompt: null, thinking: null },
    profiles: [], probes: [], history_length: 0, supports_thinking: false, thinking_is_optional: false,
    is_base_model: false, jlens_fitted: false, instruments: [], default_steering: null,
    role_substitution_supported: true, user_role_supported: true, default_assistant_role: "assistant",
    default_user_role: "user", scene_mode: true, thinking_input_supported: false, strips_history_thinking: false,
  };
  const calls = new Map();
  const hit = (key) => calls.set(key, (calls.get(key) ?? 0) + 1);
  let refuse = false;
  let open = false;
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({
    mode: "http",
    sessions: { get: async () => { hit("session"); return structuredClone(info); } },
    profiles: { list: async () => { hit("profiles"); return { profiles: [] }; } },
    probes: { list: async () => { hit("probes"); return { probes: [] }; } },
    manifolds: { list: async () => { hit("manifolds"); return { manifolds: [] }; } },
    tree: { get: async () => { hit("tree"); return structuredClone(tree); } },
    templates: {},
    events: {
      get isOpen() { return open; },
      async open() { hit("channel"); if (refuse) throw new Error("the socket was refused"); open = true; },
      subscribe: () => () => {},
      subscribeState: () => () => {},
      send() {},
      async stop() {},
      acknowledgeSnapshot() {},
      close() { open = false; },
    },
  });
  const { bootstrap } = await server.ssrLoadModule("/src/lib/stores/bootstrap.svelte.ts");
  const { disconnectRuntimeChannel } = await server.ssrLoadModule("/src/lib/stores/ws.svelte.ts");
  const { loomTree } = await server.ssrLoadModule("/src/lib/stores/loom.svelte.ts");
  const once = { channel: 1, session: 1, tree: 1, profiles: 1, probes: 1, manifolds: 1 };

  await bootstrap();
  assert.deepEqual(Object.fromEntries(calls), once, "one read of each on a load, through the channel's rehydrate");
  assert.equal(open, true, "bootstrap leaves the channel open for the first send");
  assert.equal(loomTree.root_id, tree.root_id, "the rehydrate's tree is the page's tree");

  // The channel fails: the page still opens, reading each once itself.
  disconnectRuntimeChannel();
  calls.clear();
  refuse = true;
  await bootstrap();
  assert.deepEqual(Object.fromEntries(calls), once, "a failed channel falls back to one read of each");
  console.log("bootstrap: each read once per load, through the channel or the fallback");
} finally {
  await server.close();
}
