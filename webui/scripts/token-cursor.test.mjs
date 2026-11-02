import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

// The token cursor (lib/stores/tokenCursor.svelte.ts): one position over the whole conversation's
// tokens, prompt tokens included, that follows the workbench's token selection and moves it.
const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({ root, optimizeDeps: { noDiscovery: true, include: [] }, appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
try {
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({
    mode: "fake", sessions: {}, profiles: {}, probes: {}, manifolds: {}, templates: {}, tree: {}, instruments: {}, events: {},
    dispose: async () => {},
  });
  const { chatLog } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const selection = await server.ssrLoadModule("/src/lib/stores/selection.svelte.ts");
  const cursor = await server.ssrLoadModule("/src/lib/stores/tokenCursor.svelte.ts");
  const { tokenCursor } = cursor;
  chatLog.turns = [
    // A captured prompt: its rows replay at their place in the capture (promptIndex).
    { role: "user", text: "Hi there", nodeId: "u1", tokens: [{ text: "Hi", promptIndex: 0 }, { text: " there", promptIndex: 1 }] },
    { role: "assistant", text: "Yes.", generated: true, nodeId: "a1", thinking: true,
      thinkingTokens: [{ text: "hm", rawIndex: 0, thinking: true }], tokens: [{ text: "Yes", rawIndex: 1 }, { text: ".", rawIndex: 2 }] },
    // A prompt no generation measured: no tokens, so the walk passes over it.
    { role: "user", text: "More", nodeId: "u2" },
    { role: "assistant", text: "Ok", generated: true, nodeId: "a2", tokens: [{ text: "Ok", rawIndex: 0 }] },
  ];
  const at = () => tokenCursor.at && { ...tokenCursor.at };

  // Nothing selected: the whole latest reply.
  assert.equal(tokenCursor.at, null);
  assert.equal(tokenCursor.reply.turnIdx, 3);
  assert.equal(tokenCursor.total, 6);
  assert.equal(cursor.canStep(-1), true);

  // A step from the whole reply lands on its first token, as a step: it selects that token.
  cursor.stepToken(1);
  assert.deepEqual(at(), { turnIdx: 3, seg: "response", tokenIdx: 0 });
  assert.deepEqual(selection.selection.current, { kind: "token", params: { turnIdx: 3, tokenIdx: 0, isThinking: false } });
  assert.equal(selection.selectionStep.stepped, true, "the Inspector keeps its scroll for a step");
  assert.equal(cursor.canStep(1), false, "the conversation's last token");

  // ← rolls across the unmeasured prompt to the earlier reply's last token, then through its
  // thinking into the captured prompt's tokens.
  cursor.stepToken(-1);
  assert.deepEqual(at(), { turnIdx: 1, seg: "response", tokenIdx: 1 });
  assert.equal(tokenCursor.token.index, 2);
  assert.equal(tokenCursor.token.ordinal, 4);
  cursor.stepToken(-1);
  cursor.stepToken(-1);
  assert.deepEqual(at(), { turnIdx: 1, seg: "thinking", tokenIdx: 0 });
  cursor.stepToken(-1);
  assert.deepEqual(at(), { turnIdx: 0, seg: "response", tokenIdx: 1 });
  assert.equal(tokenCursor.token.nodeId, "u1");
  assert.equal(tokenCursor.token.index, 1, "a prompt token replays at its captured row");
  assert.equal(tokenCursor.reply.turnIdx, 1, "a prompt's whole reply is the reply that read it");
  cursor.segmentEdge("first");
  assert.deepEqual(at(), { turnIdx: 0, seg: "response", tokenIdx: 0 });
  assert.equal(cursor.canStep(-1), false);

  // A unit or probe selected: the cursor rests on the last token.
  selection.select({ kind: "unit", id: "neuron/L1n1" });
  assert.deepEqual(at(), { turnIdx: 0, seg: "response", tokenIdx: 0 });
  // Whole reply: the reply after the prompt the cursor rested in, not the latest.
  cursor.showWholeReply();
  assert.equal(tokenCursor.at, null);
  assert.equal(tokenCursor.reply.turnIdx, 1);
  assert.deepEqual(selection.selection.current, { kind: "unit", id: "neuron/L1n1" }, "the unit stays selected");

  // A token selected in the workbench is the cursor (a fresh selection, not a step).
  selection.select({ kind: "token", params: { turnIdx: 3, tokenIdx: 0 } });
  assert.deepEqual(at(), { turnIdx: 3, seg: "response", tokenIdx: 0 });
  assert.equal(selection.selectionStep.stepped, false);
  // Clearing that selection reads its turn's whole reply.
  selection.clearSelection();
  assert.equal(tokenCursor.at, null);
  assert.equal(tokenCursor.reply.turnIdx, 3);
  // Whole reply from a selected token deselects it.
  selection.select({ kind: "token", params: { turnIdx: 1, tokenIdx: 1 } });
  cursor.showWholeReply();
  assert.equal(selection.selection.current.kind, "none");
  assert.equal(tokenCursor.reply.turnIdx, 1);

  // A trace's place (a replay index) back to a position.
  assert.deepEqual(cursor.cursorAtIndex(1, 0), { turnIdx: 1, seg: "thinking", tokenIdx: 0 });
  assert.deepEqual(cursor.cursorAtIndex(1, 2), { turnIdx: 1, seg: "response", tokenIdx: 1 });
  assert.deepEqual(cursor.cursorAtIndex(0, 1), { turnIdx: 0, seg: "response", tokenIdx: 1 });
  assert.equal(cursor.cursorAtIndex(3, 5), null);

  // A selection past a turn's end (the turn's token list shrank) clamps onto the walk.
  selection.select({ kind: "token", params: { turnIdx: 1, tokenIdx: 9 } });
  assert.deepEqual(at(), { turnIdx: 1, seg: "response", tokenIdx: 1 });

  // Regression (W6 review): the resting place was positional only and outlived the conversation.
  // After Clear chat and a new reply, the rooms opened at a token of the new reply nobody selected.
  // The position holds only while its turn holds the node it was made on.
  selection.select({ kind: "token", params: { turnIdx: 1, tokenIdx: 1 } });
  selection.select({ kind: "unit", id: "neuron/L1n3" });
  selection.clearSelection();
  assert.deepEqual(at(), { turnIdx: 1, seg: "response", tokenIdx: 1 }, "resting on the token while nothing is selected");
  chatLog.turns = [];
  selection.reconcileTokenSelection();
  assert.equal(tokenCursor.at, null, "no tokens: nothing to rest on");
  assert.equal(tokenCursor.reply, null);
  assert.deepEqual([selection.tokenRest.params, selection.tokenRest.turnIdx, selection.tokenRest.nodeId], [null, null, null], "Clear chat drops where it rested");
  const fresh = [
    { role: "user", text: "new q", nodeId: "u-new" },
    { role: "assistant", text: "new", generated: true, nodeId: "a-new", tokens: [{ text: "n0", rawIndex: 0 }, { text: "n1", rawIndex: 1 }, { text: "n2", rawIndex: 2 }] },
  ];
  chatLog.turns = fresh;
  assert.equal(tokenCursor.at, null, "a new reply opens whole");
  assert.equal(tokenCursor.reply.nodeId, "a-new");
  // Before the projection reconciles (the turns replaced in place), the cursor already reads
  // another branch's reply whole; the reconcile then drops the stale token selection.
  selection.select({ kind: "token", params: { turnIdx: 1, tokenIdx: 2 } });
  assert.equal(selection.tokenRest.nodeId, "a-new");
  chatLog.turns = [fresh[0], { ...fresh[1], nodeId: "a-sibling" }];
  assert.equal(tokenCursor.at, null, "another branch's reply: no token nobody selected");
  assert.equal(tokenCursor.reply.nodeId, "a-sibling");
  selection.reconcileTokenSelection();
  assert.equal(selection.selection.current.kind, "none", "the stale token selection drops");
  chatLog.turns = [];

  // Regression (W6 review): with the A/B twin open, clicking another token of the same turn kept
  // the twin, so the Inspector read the twin's token for the primary token the thread marked. A
  // fresh selection or another turn opens the turn as generated; a step within the turn keeps the
  // side.
  const { branchAfter } = await server.ssrLoadModule("/src/drawers/token/cursor.ts");
  assert.equal(branchAfter("shadow", { sameTurn: true, stepped: false }), "primary", "another token clicked in the same turn");
  assert.equal(branchAfter("shadow", { sameTurn: true, stepped: true }), "shadow", "an arrow, Home/End or a context step");
  assert.equal(branchAfter("shadow", { sameTurn: false, stepped: true }), "primary", "a step into another turn");
  const { readFileSync } = await import("node:fs");
  const drawer = readFileSync(new URL("../src/drawers/TokenDrilldownDrawer.svelte", import.meta.url), "utf8");
  assert.match(drawer, /branchAfter\(untrack\(\(\) => branch\), \{ sameTurn, stepped: untrack\(\(\) => selectionStep\.stepped\) \}\)/);
  assert.match(drawer, /const p = inputParams;/, "the reset follows each selection, not only a turn change");

  console.log("token cursor: walk, prompt tokens, whole reply, selection follow, steps and clamping passed");
} finally {
  await server.close();
}
