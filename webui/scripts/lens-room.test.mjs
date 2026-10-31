import assert from "node:assert/strict";
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

  // The pointer's layer: the closest column carrying data-layer.
  const column = { dataset: { layer: "7" } };
  assert.equal(lens.layerAt({ target: { closest: () => column } }), 7);
  assert.equal(lens.layerAt({ target: { closest: () => null } }), -1);
  assert.equal(lens.layerAt({ target: null }), -1);
  console.log("Lens room: band spans, site series, family sources and the fit ladder passed");
} finally {
  await server.close();
}
