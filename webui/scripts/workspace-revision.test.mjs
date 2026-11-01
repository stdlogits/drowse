import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

// The workspace revision an agent's edit checks against (drowse_set_steering and the other workspace
// tools). Regression: the autosave's status and the library write it makes counted toward it, so a
// steering edit sent right after a page load or a send answered STALE_STATE and only a retry passed.
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "http", sessions: {}, profiles: {}, manifolds: {}, templates: {}, tree: {}, events: {}, probes: {} });
  const { workspaceRevision } = await server.ssrLoadModule("/src/lib/webmcp/workspaceTools.ts");
  const { sessionState } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
  const { conversationLibrary, savedConversationState } = await server.ssrLoadModule("/src/lib/stores/savedConversations.svelte.ts");
  sessionState.info = { model_id: "HuggingFaceTB/SmolLM2-360M-Instruct" };
  savedConversationState.activeId = "chat-1";
  const before = workspaceRevision();

  for (const status of ["pending", "saving", "saved", "error", "idle"]) {
    savedConversationState.status = status;
    savedConversationState.error = status === "error" ? "Storage full" : null;
    assert.equal(workspaceRevision(), before, `autosave status ${status} leaves the revision alone`);
  }
  conversationLibrary.revision += 3;
  assert.equal(workspaceRevision(), before, "the autosave's library write leaves the revision alone");

  savedConversationState.activeId = "chat-2";
  assert.notEqual(workspaceRevision(), before, "opening another saved chat still changes the revision");
  console.log("workspace revision: autosave status and writes do not count; the open chat does");
} finally {
  await server.close();
}
