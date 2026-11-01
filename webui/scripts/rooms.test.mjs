import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { createServer } from "vite";

// Rooms and navigation: the Artifacts room's sections (what is prepared, what to do next), the
// command palette (the rooms, then the instruments in the order of priority, then every section by
// what it does, a hosted tool listed once), and every link to setup going to its Artifacts section.
const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const src = new URL("../src/", import.meta.url);
const read = (path) => readFileSync(new URL(path, src), "utf8");

try {
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  let catalog = { model_id: "Qwen/Qwen3-8B", available: true, bound: false, slug: "qwen3-8b", repo: "stdlogits/neuron-lens", revision: "b5bf11a2b9e3", credit: null, reason: null, binding: null };
  installRuntimeClient({
    mode: "http",
    sessions: {}, profiles: {}, manifolds: {}, templates: {}, tree: {}, events: {}, probes: {},
    instruments: {
      unitRead: async () => { throw new Error("not read here"); },
      unitCatalog: async () => catalog,
    },
    dispose: async () => {},
  });

  // The sections: Artifacts in the order of priority (units, lenses, the probe tools, SAEs), each
  // with what it does; the HTTP runtime's Model room has health and API access, not the browser's.
  const rooms = await server.ssrLoadModule("/src/lib/stores/rooms.svelte.ts");
  assert.deepEqual(rooms.ROOMS.map(room => room.label), ["Workbench", "Units", "Lens", "Probes", "Artifacts", "Model"]);
  assert.deepEqual(rooms.shownSections("artifacts").map(row => row.value), ["units", "lenses", "build", "packs", "combine", "templates", "roles", "saes"]);
  assert.ok(!rooms.shownSections("model").some(row => row.value === "runtime"), "the HTTP runtime has no local runtime section");
  for (const rows of Object.values(rooms.ROOM_SECTIONS)) {
    for (const row of rows) assert.ok(row.does && row.keywords, `${row.value} says what it does`);
  }

  // The palette: rooms first in header order, the instruments by priority, the Inspector's order
  // (W6 integration: the unit probes first, where units read; the palette had left them out), then
  // each section by what it does. A tool a room hosts is listed once.
  const { paletteCommands, searchCommands } = await server.ssrLoadModule("/src/lib/commands.ts");
  const commands = paletteCommands();
  assert.deepEqual(commands.filter(c => c.group === "Rooms").map(c => c.label), ["Workbench", "Units", "Lens", "Probes", "Artifacts", "Model"]);
  assert.deepEqual(commands.filter(c => c.group === "Instruments").map(c => c.action.tab), ["units", "lens", "subspace", "sae"]);
  const instrumentIds = readFileSync(new URL("../src/lib/commands.ts", import.meta.url), "utf8");
  assert.match(instrumentIds, /tab === "units" \? "instrument-units"/, "the unit probes scroll to the Inspector's Units");
  assert.deepEqual(commands.filter(c => c.group === "Artifacts").map(c => c.action.section), ["units", "lenses", "build", "packs", "combine", "templates", "roles", "saes"]);
  const { roomFor } = rooms;
  assert.deepEqual(commands.filter(c => c.action.kind === "drawer" && roomFor(c.action.drawer)).map(c => c.label), [], "a hosted tool is its section");
  assert.equal(new Set(commands.map(c => `${c.group} ${c.label}`)).size, commands.length);
  const first = (query) => searchCommands(commands, query)[0]?.action;
  assert.deepEqual(first("fit a lens"), { kind: "section", room: "artifacts", section: "lenses" });
  assert.deepEqual(first("fetch the unit catalog"), { kind: "section", room: "artifacts", section: "units" });
  assert.deepEqual(first("build a probe"), { kind: "section", room: "artifacts", section: "build" });
  assert.deepEqual(first("install a pack"), { kind: "section", room: "artifacts", section: "packs" });
  assert.deepEqual(first("train an sae"), { kind: "section", room: "artifacts", section: "saes" });
  assert.deepEqual(first("create a concept"), { kind: "section", room: "artifacts", section: "build" }, "the builder's old name still finds it");
  assert.deepEqual(first("api key"), { kind: "section", room: "model", section: "access" });
  assert.ok(!searchCommands(commands, "fit a lens").some(c => c.action.kind === "room" && c.action.room === "lens"), "the Lens room reads; it does not fit");

  // The Units section: a catalogued model's labels to fetch (the one primary action), then bound
  // (what is installed: the catalog, its commit and files, the credit), then an uncatalogued
  // model's reason. Each leads to the Units room.
  const { render } = await server.ssrLoadModule("svelte/server");
  const units = await server.ssrLoadModule("/src/lib/stores/units.svelte.ts");
  const { default: UnitCatalog } = await server.ssrLoadModule("/src/panels/rooms/artifacts/UnitCatalog.svelte");
  await units.loadCatalog();
  let html = render(UnitCatalog).body;
  assert.match(html, /labels to fetch/);
  assert.match(html, /class="dw-button[ "][^>]*><span class="morph-text[^"]*" data-text="Fetch labels"/);
  assert.match(html, /at b5bf11a/);
  assert.match(html, />Read units</);
  catalog = { ...catalog, bound: true, credit: "Labels by Claude Haiku 5.5", binding: { revision: "b5bf11a2b9e3", fetched_at: "2026-10-30T12:00:00Z", files: 37 } };
  await units.loadCatalog();
  html = render(UnitCatalog).body;
  assert.match(html, />labelled</);
  assert.match(html, /37 files/);
  assert.match(html, /Labels by Claude Haiku 5\.5/);
  assert.doesNotMatch(html, /Fetch labels/);
  catalog = { model_id: "HuggingFaceTB/SmolLM2-360M-Instruct", available: false, bound: false, slug: null, repo: "stdlogits/neuron-lens", revision: "r", credit: null, reason: "no neuron-lens catalog for this model", binding: null };
  await units.loadCatalog();
  html = render(UnitCatalog).body;
  assert.match(html, /no neuron-lens catalog for this model/);
  assert.doesNotMatch(html, /Fetch labels/);

  // Regression (a9, Wave 6): setup lived in the Lens room, so "Choose a lens", "Choose an SAE" and
  // "Fit a lens here" led to a readout. Every link to setup goes to its Artifacts section.
  const files = [];
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      const path = new URL(name, dir);
      if (statSync(path).isDirectory()) walk(new URL(`${name}/`, dir));
      else if (name.endsWith(".svelte")) files.push(path);
    }
  };
  walk(src);
  const setup = { "Choose a lens": "lenses", "Choose an SAE": "saes", "Fit a lens here": "lenses", "Fetch labels": "units", "Build a probe": "build",
    "Build one": "build", "Lenses": "lenses", "Unit catalog": "units" };
  let links = 0;
  for (const file of files) {
    const text = readFileSync(file, "utf8");
    for (const match of text.matchAll(/goto\(([^)]*)\)[^>]*>([^<{]+)</g)) {
      const want = setup[match[2].trim()];
      if (!want) continue;
      links += 1;
      assert.equal(match[1], `"artifacts", "${want}"`, `${file.pathname.split("/src/")[1]}: ${match[2].trim()}`);
    }
  }
  assert.ok(links >= 9, `found ${links} setup links`);
  // W6 integration: a readout room missing its instrument leads to the Artifacts section that
  // prepares it. The Probes room with nothing attached offered only "Workbench"; it attaches a
  // concept or a manifold in place, or builds one.
  const probesRoom = read("panels/rooms/ProbesRoom.svelte");
  const empty = probesRoom.slice(probesRoom.indexOf(">No probes<"), probesRoom.indexOf("{:else}", probesRoom.indexOf(">No probes<")));
  assert.match(empty, /openDrawer\("subspace"\)\}>A concept</);
  assert.match(empty, /openDrawer\("manifolds"\)\}>A manifold</);
  assert.match(empty, /goto\("artifacts", "build"\)\}>Build one</);
  assert.match(read("panels/rooms/LensRoom.svelte"), /goto\("artifacts", "lenses"\)\}>Lenses</);
  const model = read("panels/rooms/ModelFacts.svelte");
  assert.ok(model.includes(`goto("artifacts", "lenses")}>{canFit ? "Fit a lens here"`));
  assert.ok(model.includes(`goto("artifacts", "saes")}>{sae ? "SAEs" : "Choose an SAE"}`));
  assert.ok(model.includes(`goto("artifacts", "units")}>{catalog?.available && !catalog.bound ? "Fetch labels"`));

  // The shell: on a phone the rooms are in the menu and a crumb over each room leads back; the
  // rooms draw no crumb of their own.
  const app = read("App.svelte");
  assert.match(app, /\{#if phone && route\.room !== "workbench"\}\s*<nav class="dw-crumbs room-crumbs"/);
  assert.match(app, /label: "Rooms", items: shownRooms\(\)/);
  assert.doesNotMatch(read("panels/rooms/RoomSections.svelte"), /dw-crumbs/);

  // Help names the readout rooms as token by token and Artifacts as where things are prepared.
  const { default: Help } = await server.ssrLoadModule("/src/drawers/HelpDrawer.svelte");
  const help = render(Help).body;
  assert.match(help, /<dt[^>]*>Artifacts<\/dt>\s*<dd[^>]*>Where what the model is read and steered with is prepared/);
  console.log("Rooms (Artifacts sections, palette by what each does, setup links, the phone crumb) passed");
} finally { await server.close(); }
