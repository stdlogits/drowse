import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({ root, optimizeDeps: { noDiscovery: true, include: [] }, appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
try {
  const lens = await server.ssrLoadModule("/src/lib/lensRoom.ts");

  // The server's band: S below the crossover, both inside it, R above (10 layers, S weights).
  const sWeights = [1, 1, 0.75, 0.5, 0.25, 0, 0, 0, 0, 0];
  const band = { lo: 0.2, hi: 0.5, weights: sWeights.map(s => [s, 1 - s]) };
  assert.equal(lens.layerCount(band, []), 10);
  assert.deepEqual(lens.familyWeights(band, "S"), sWeights);
  assert.deepEqual(lens.familyWeights(band, "R"), sWeights.map(s => 1 - s));
  assert.equal(lens.familyWeights(null, "S"), null, "no band, no weights: the room hides the axis");
  assert.deepEqual(lens.familySpan(band, "S"), { first: 0, last: 4 });
  assert.deepEqual(lens.familySpan(band, "R"), { first: 2, last: 9 });
  assert.deepEqual(lens.crossover(band), { first: 2, last: 4 });
  assert.equal(lens.spanText(lens.crossover(band)), "L2–L4");
  assert.equal(lens.spanText({ first: 3, last: 3 }), "L3");
  const hard = { lo: 0.3, hi: 0.3, weights: [[1, 0], [1, 0], [0, 1]] };
  assert.equal(lens.crossover(hard), null, "a hard switch has no crossover");

  // Without a band, the layer count comes from the deepest layer any source holds.
  const local = {
    source: "local:r", kind: "local", family: "R", lens: "R", active: true, sites: ["out", "mid"],
    layers: { out: [0, 1, 2], mid: [0, 1, 2, 3] }, n_prompts: 64,
    split_half: { out: { 0: 0.25, 1: 0.125, 2: 0.0625 } },
  };
  assert.equal(lens.layerCount(null, [local]), 4);
  assert.equal(lens.layerCount(null, []), 0);

  // A site's series: held layers and the split-half difference where the fit recorded it.
  const out = lens.siteSeries(local, "out", 4);
  assert.deepEqual(out.covered, [true, true, true, false]);
  assert.deepEqual(out.split, [0.25, 0.125, 0.0625, null]);
  assert.equal(out.measured, true);
  const mid = lens.siteSeries(local, "mid", 4);
  assert.deepEqual(mid.covered, [true, true, true, true]);
  assert.equal(mid.measured, false, "a site with no record is held but unmeasured");
  assert.equal(lens.coveredText(mid), "L0–L3");
  assert.equal(lens.siteSeries(null, "out", 2).covered.some(Boolean), false);
  assert.equal(lens.rangeText(out.split), "0.06–0.25");
  assert.equal(lens.rangeText([0.1, null, 0.1]), "0.10");
  assert.equal(lens.rangeText([null]), "");
  assert.equal(lens.sharedScale([local, null]), 0.25);
  assert.equal(lens.sharedScale([{ source: "rs-r" }]), null, "imports record no split-half");

  // The words around a source.
  assert.equal(lens.foreignKind("J", "R"), "J-lens");
  assert.equal(lens.foreignKind("R", "R"), "");
  assert.equal(lens.documentsText(1000), "1,000 documents");
  assert.equal(lens.documentsText(1), "1 document");
  assert.equal(lens.documentsText(undefined), "");
  assert.equal(lens.sitesText(["out", "mid"]), "out and mid");
  assert.equal(lens.sitesText(["out"]), "out only");
  assert.equal(lens.sitesText(undefined), "");
  assert.equal(lens.originText(local, null), "local");
  assert.equal(lens.originText({ source: "rs-r", kind: "huggingface", repo_id: "stdlogits/rs-lenses" }, null), "stdlogits/rs-lenses");
  assert.equal(lens.originText(null, { repo_id: "neuronpedia/jacobian-lens" }), "neuronpedia/jacobian-lens");

  // A family's sources: every provider in the server's order (prepared or still to fetch), then
  // the local fits.
  const providers = [
    { source: "rs-r", lens: "R", family: "R", sites: ["out", "mid"], repo_id: "stdlogits/rs-lenses" },
    { source: "neuronpedia", lens: "J", family: "R", sites: ["out"], repo_id: "neuronpedia/jacobian-lens" },
    { source: "rs-s", lens: "S", family: "S", sites: ["out", "mid"], repo_id: "stdlogits/rs-lenses" },
  ];
  const fetched = { source: "neuronpedia", kind: "huggingface", family: "R", lens: "J", active: false };
  const items = lens.familyItems([local, fetched], providers, "R");
  assert.deepEqual(items.map(item => item.source), ["rs-r", "neuronpedia", "local:r"]);
  assert.equal(items[0].row, null, "rs-r is still to fetch");
  assert.equal(items[1].row, fetched);
  assert.equal(items[2].provider, null);
  assert.deepEqual(lens.familyItems([local], providers, "S").map(item => item.source), ["rs-s"]);
  assert.deepEqual(lens.familyItems([local], [], "R").map(item => item.source), ["local:r"], "a runtime with no providers lists its rows");

  assert.equal(lens.sourceFamily("rs-s", [], providers), "S");
  assert.equal(lens.sourceFamily("local:k-chat", [], []), "S");
  assert.equal(lens.sourceFamily("local:j", [], []), "R");
  assert.equal(lens.sourceFamily(null, [], []), null);
  assert.equal(lens.sourceFamily("mystery", [], []), null);

  // The fit's documents: the server's default, doubled and halved.
  const ladder = lens.documentLadder(64);
  assert.deepEqual(ladder, [8, 16, 32, 64, 128, 256, 512, 1024]);
  assert.ok(ladder.includes(64), "the server default is a rung");
  assert.equal(lens.stepDocuments(ladder, 64, 1), 128);
  assert.equal(lens.stepDocuments(ladder, 64, -1), 32);
  assert.equal(lens.stepDocuments(ladder, 1024, 1), 1024);
  assert.equal(lens.stepDocuments(ladder, 8, -1), 8);
  assert.equal(lens.stepDocuments(ladder, 100, -1), 64, "an off-ladder count steps to the nearest rung");
  assert.deepEqual(lens.documentLadder(2), [1, 2, 4, 8, 16, 32]);

  // A refused fit offers its replacement only for the server's refusal to replace, never for an
  // unrelated failure (an unreachable Hub) or another family's fit.
  const sent = { prompts: 64, kind: "r", sites: ["out", "mid"] };
  const refused = { state: "error", source: "local:r", replaceable: true };
  assert.equal(lens.replaceTarget(refused, sent), "local:r");
  // Regression: the room showed the server's CLI and Python wording ("force=True (`-f`)").
  assert.equal(lens.replaceMessage("local:s", 64, 32), "local:s already holds a lens over 64 documents. This fit, over 32 documents, would replace it.");
  assert.doesNotMatch(lens.replaceMessage("local:s", undefined, 32), /force|`/);
  assert.equal(lens.replaceTarget({ ...refused, replaceable: false }, sent), null, "ConnectionError: no replace");
  assert.equal(lens.replaceTarget({ state: "error", source: "local:r" }, sent), null, "a runtime that does not mark it offers none");
  assert.equal(lens.replaceTarget({ ...refused, state: "running" }, sent), null);
  assert.equal(lens.replaceTarget({ ...refused, source: "local:s" }, sent), null, "another fit's refusal");
  assert.equal(lens.replaceTarget(refused, null), null, "nothing sent from this view, nothing to resend");

  // A running fit names its stage until the first document is in, then counts.
  assert.equal(lens.fitProgressText(0, 8, "streaming 8 documents from wikitext…"), "Streaming 8 documents from wikitext…");
  assert.equal(lens.fitProgressText(3, 8, "prompt 3/8 (12 tokens)"), "3 of 8 documents");
  assert.equal(lens.fitProgressText(0, 8, null), "0 of 8 documents");

  // A selected source that does not load for these weights.
  assert.equal(lens.unloaded({ source: "rs-r", active: true, compatible: false }), true);
  assert.equal(lens.unloaded({ source: "rs-r", active: true, compatible: true }), false);
  assert.equal(lens.unloaded({ source: "rs-r", active: true }), false, "a runtime that does not say counts as loaded");
  assert.equal(lens.unloaded({ source: "rs-r", active: false, compatible: false }), false);
  assert.equal(lens.unloaded(null), false);

  // The pointer's layer: the closest column carrying data-layer.
  const column = { dataset: { layer: "7" } };
  assert.equal(lens.layerAt({ target: { closest: () => column } }), 7);
  assert.equal(lens.layerAt({ target: { closest: () => null } }), -1);
  assert.equal(lens.layerAt({ target: null }), -1);

  // Regression (a9, unnecessary text): the depth axis carried an Out/Mid legend ("neuron outputs,
  // attention inputs"). It is gone; what a site reads is the tip on each card's site.
  {
    const { render } = await server.ssrLoadModule("svelte/server");
    const { default: LensDepth } = await server.ssrLoadModule("/src/panels/rooms/lens/LensDepth.svelte");
    const { default: LensCard } = await server.ssrLoadModule("/src/panels/rooms/lens/LensCard.svelte");
    const depth = render(LensDepth, { props: { band, model: "fixture" } }).body;
    assert.match(depth, /Which lens, by depth/);
    assert.doesNotMatch(depth, /neuron outputs|attention inputs|class="sites/);
    for (const site of ["out", "mid"]) {
      const card = render(LensCard, { props: { family: "R", site, row: local, provider: null, n: 4, weights: null, span: null, scale: 0.25 } }).body;
      assert.match(card, new RegExp(`<span class="meta[^"]*" data-tip="${lens.SITE_TIPS[site].replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}">${site}</span>`));
    }
    // Regression (W6 review): with no lens prepared each card read "None · out · used L0–L10", a
    // layer range for a lens that does not exist. The range shows only for a source that holds it.
    const span = { first: 0, last: 10 };
    const none = render(LensCard, { props: { family: "R", site: "out", row: null, provider: null, n: 4, weights: null, span, scale: null } }).body;
    assert.doesNotMatch(none, /used L/);
    const held = render(LensCard, { props: { family: "R", site: "out", row: local, provider: null, n: 4, weights: null, span, scale: 0.25 } }).body;
    assert.match(held, /used L0–L10/);
    assert.match(lens.SITE_TIPS.out, /neuron outputs and attention inputs/);
    assert.match(lens.SITE_TIPS.mid, /neuron inputs and attention outputs/);
  }

  // The 2x2 without the fit: drowse.ai's hosted section renders LensQuad (its runtime fits nothing,
  // and its build refuses the fit's words), so nothing LensQuad reaches imports the fit job or
  // LensFit or carries the fit's markup; drowse's own section (LensFit columns) still shows the fit.
  {
    const fitMarkers = ["Lens fit progress", "Generation pauses while documents are fitted", "Fit locally", "drowse lens fit"];
    const count = (text, needle) => text.split(needle).length - 1;
    const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
    installRuntimeClient({
      mode: "http", sessions: {}, profiles: {}, manifolds: {}, templates: {}, tree: {}, events: {}, probes: {}, instruments: {},
      dispose: async () => {},
    });
    const { render } = await server.ssrLoadModule("svelte/server");
    const { default: LensQuad } = await server.ssrLoadModule("/src/panels/rooms/lens/LensQuad.svelte");
    const graph = server.environments.ssr.moduleGraph;
    const reached = new Set();
    const queue = [await graph.getModuleByUrl("/src/panels/rooms/lens/LensQuad.svelte")];
    while (queue.length > 0) {
      const module = queue.shift();
      if (reached.has(module)) continue;
      reached.add(module);
      queue.push(...module.importedModules);
    }
    const sources = new URL("../src/", import.meta.url).pathname;
    const files = [...new Set([...reached].map(module => module.file).filter(file => file?.startsWith(sources)))];
    assert.ok(files.some(file => file.endsWith("/rooms/lens/LensFamily.svelte")), "the walk reaches the columns");
    const fitModule = (file) => file.includes("/stores/instrumentAuthoring.") || file.endsWith("/rooms/lens/LensFit.svelte");
    assert.deepEqual(files.filter(fitModule), [], "LensQuad imports no fit");
    for (const file of files) {
      const text = readFileSync(file, "utf8");
      for (const marker of fitMarkers) assert.ok(!text.includes(marker), `${file} carries "${marker}"`);
    }

    const { sessionState } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
    const { lensSourceState } = await server.ssrLoadModule("/src/lib/stores/instruments.svelte.ts");
    sessionState.info = { model_id: "m", instruments: [{ family: "lens", capabilities: { source_switch: true, preparations: ["fetch", "fit"] } }] };
    const otherS = { source: "local:s", kind: "local", family: "S", lens: "S", active: false, sites: ["out"], layers: { out: [0, 1] }, n_prompts: 32 };
    lensSourceState.sources = [local, otherS];
    lensSourceState.fitDocuments = 64;
    const quad = render(LensQuad, {}).body;
    assert.match(quad, /aria-label="The four lenses"/);
    assert.deepEqual([count(quad, ">S-lens<"), count(quad, ">R-lens<"), count(quad, ">Sources<")], [1, 1, 2]);
    for (const marker of fitMarkers) assert.ok(!quad.includes(marker), `the plain 2x2 shows "${marker}"`);

    const { lensFit } = await server.ssrLoadModule("/src/lib/stores/instrumentAuthoring.svelte.ts");
    const { default: NativeSection } = await server.ssrLoadModule("/src/panels/JLensSourceSection.svelte");
    let native = render(NativeSection, {}).body;
    assert.equal(count(native, ">Fit locally<"), 2, "idle: each family offers a local fit");
    assert.doesNotMatch(native, /Lens fit progress/);
    assert.doesNotMatch(native, /aria-pressed="false"\s+disabled/, "idle: every source can be chosen");
    // An R fit running: its progress among R's sources (after S's form), S's form still there, and
    // a source not in use waits for it (the running flag LensFamily takes).
    Object.assign(lensFit.state, { running: true, current: 3, total: 8, message: "prompt 3/8", latest: { state: "running", source: "local:r" } });
    native = render(NativeSection, {}).body;
    assert.equal(count(native, 'aria-label="Lens fit progress"'), 1);
    assert.match(native, /data-tip="Generation pauses while documents are fitted"/);
    assert.match(native, /data-tip="3 of 8 documents"/);
    assert.match(native, />Cancel</);
    assert.equal(count(native, ">Fit locally<"), 1);
    assert.ok(native.indexOf(">Fit locally<") < native.indexOf("Lens fit progress"), "S's column, then R's");
    assert.match(native, /aria-pressed="false"\s+disabled/, "local:s waits while the fit runs");
    // A failed R fit: its error under R's sources, and both forms back.
    Object.assign(lensFit.state, { running: false, error: "no documents", latest: { state: "error", source: "local:r" } });
    native = render(NativeSection, {}).body;
    assert.match(native, /<p class="meta err[^"]*" role="alert">no documents<\/p>/);
    assert.equal(count(native, ">Fit locally<"), 2);
    Object.assign(lensFit.state, { running: false, current: 0, total: 0, message: null, error: null, latest: null });
    lensSourceState.sources = [];
    lensSourceState.fitDocuments = null;
    sessionState.info = null;
  }
  console.log("Lens room: band spans, site series, family sources, the fit ladder, the fit's status and the 2x2 without the fit passed");
} finally {
  await server.close();
}
