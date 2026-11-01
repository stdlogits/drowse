import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "vite";

// Regression: with the status line and the rack morphing a value per decoded token, the served page
// read about 15 token frames a second and the server cut long replies (a 1013 close, recorded as
// cancelled). Values that change per token are plain text while a reply streams and morph once it
// settles.
const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "http" });
  const { render } = await server.ssrLoadModule("svelte/server");
  const { genStatus } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const { default: LiveText } = await server.ssrLoadModule("/src/lib/ui/LiveText.svelte");

  genStatus.active = true;
  const streaming = render(LiveText, { props: { text: "0.42" } }).body;
  assert.match(streaming, /0\.42/);
  assert.doesNotMatch(streaming, /morph-text/, "a streaming value is plain text, never a letter morph");
  genStatus.active = false;
  const settled = render(LiveText, { props: { text: "0.42" } }).body;
  assert.match(settled, /class="morph-text[^"]*"/, "a settled value morphs");
  assert.match(settled, /0\.42/);

  // Every value that changes per decoded token goes through LiveText.
  const read = (path) => readFile(new URL(`../src/${path}`, import.meta.url), "utf8");
  const perToken = {
    "panels/StatusFooter.svelte": ["genStatus.tokensSoFar", "tokPerSec.toFixed(1)", "elapsedSec.toFixed(1)", "ppl.toFixed(2)"],
    "panels/rack/ProbeCard.svelte": ["readingText(reading, { signed: affine })"],
    "panels/rack/UnitProbeCard.svelte": ['readingText(reading, { signed: info.kind === "neuron" })'],
    "panels/rack/SaeProbeCard.svelte": ["readingText(strength ?? rawValue)"],
  };
  for (const [path, values] of Object.entries(perToken)) {
    const source = await read(path);
    for (const value of values) {
      assert.ok(source.includes(`<LiveText text={${value}} />`), `${path}: ${value} streams through LiveText`);
      assert.ok(!source.includes(`<MorphText text={${value}}`), `${path}: ${value} must not morph per token`);
    }
  }
  console.log("Live text: per-token values are plain while a reply streams and morph once it settles");
} finally {
  await server.close();
}
