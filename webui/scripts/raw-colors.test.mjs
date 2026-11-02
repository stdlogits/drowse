import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { withoutComments } from "./markup.mjs";

// Raw completion keeps its colour selector. 6.0 moved the reading tools ("Color words by") into the
// chat's dock, which raw mode does not draw, so a base model's completion could not change what its
// tokens are coloured by, though RawBuffer still tints them and explains an empty colour. The tools
// are one snippet in Chat: under the composer in a chat, under the completion's head in raw mode,
// behind one Colors state.
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "http", instruments: {} });
  const { render } = await server.ssrLoadModule("svelte/server");
  const { createRawSnippet } = await server.ssrLoadModule("svelte");
  const { genUiMode } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const { default: Chat } = await server.ssrLoadModule("/src/panels/Chat.svelte");
  const { default: RawBuffer } = await server.ssrLoadModule("/src/panels/RawBuffer.svelte");
  const colorsToggle = /<button[^>]*aria-expanded="false"[^>]*aria-controls="chat-reading-tools"[^>]*>Colors<\/button>/;

  genUiMode.mode = "chat";
  const chat = withoutComments(render(Chat).body);
  assert.match(chat, colorsToggle, "a chat offers Colors in its composer row");
  assert.ok(!chat.includes('aria-label="Completion buffer"'));

  genUiMode.mode = "raw";
  const raw = withoutComments(render(Chat).body);
  assert.ok(raw.includes('aria-label="Completion buffer"'), "raw mode draws the completion buffer");
  const head = raw.slice(raw.indexOf('class="raw-head'), raw.indexOf('id="completion-hint"'));
  assert.match(head, colorsToggle, "raw completion offers Colors in its head");
  genUiMode.mode = "chat";

  const tools = createRawSnippet(() => ({ render: () => '<div id="chat-reading-tools">tools</div>' }));
  const closed = withoutComments(render(RawBuffer, { props: { colors: tools } }).body);
  assert.ok(!closed.includes("tools</div>"), "closed, the tools wait behind Colors");
  const open = withoutComments(render(RawBuffer, { props: { colors: tools, colorsOpen: true } }).body);
  assert.match(open, /aria-expanded="true"[^>]*>Colors<\/button>/);
  assert.ok(open.indexOf("tools</div>") > open.indexOf('class="raw-head'), "open, they sit under the head");
  assert.ok(open.indexOf("tools</div>") < open.indexOf('class="surface'), "above the completion");
  assert.doesNotMatch(withoutComments(render(RawBuffer).body), />Colors<\/button>/, "no tools, no toggle");

  // Chat hands raw mode the same snippet and Colors state, with the completion's own label.
  const source = readFileSync(new URL("../src/panels/Chat.svelte", import.meta.url), "utf8");
  const snippet = source.slice(source.indexOf("{#snippet readingTools()}"), source.indexOf("{/snippet}"));
  assert.match(snippet, /ariaLabel=\{rawMode \? "Color completion tokens by" : "Color generated words by"\}/);
  assert.match(source, /<RawBuffer colors=\{readingTools\} bind:colorsOpen=\{readingOpen\} \/>/);
  assert.match(source, /\{#if readingOpen\}\{@render readingTools\(\)\}\{\/if\}/);
  console.log("raw completion keeps the colour selector");
} finally {
  await server.close();
}
