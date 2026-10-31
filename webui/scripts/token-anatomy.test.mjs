// The token anatomy's view math (drawers/token/anatomy.ts) and its place in the token-readout
// queue: the depth-picked cell, the chosen token over depth, the pushed units' poles, the heads'
// heat rows, the label text, and an anatomy that waits behind a running readout.
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const flush = () => new Promise((resolve) => setImmediate(resolve));

const score = (token_id, token, value) => ({ token_id, token, value });
const half = (family, weight, tokens, extra = {}) => ({
  family, weight, unit: family === "R" ? "probability" : "footprint_z",
  lens: `${family}:out`, centred: false, tokens, absent: null, ...extra,
});
const read = (layer, sWeight, s, r) => ({
  layer, site: "out", depth: layer / 3, of: "residual",
  s: half("S", sWeight, s), r: half("R", 1 - sWeight, r),
});
const layer = (n, readsAs, extra = {}) => ({
  layer: n, depth: n / 3, reads_as: readsAs, attention: readsAs, mlp: readsAs, steering: null,
  neurons: [], heads: [], ...extra,
});

try {
  const a = await server.ssrLoadModule("/src/drawers/token/anatomy.ts");

  // S early, R late: outside the band one token, from the heavy half.
  const early = a.pickCell(read(0, 1, [score(5, " sea", 4.7)], [score(9, " the", 0.3)]));
  assert.deepEqual(early.tokens.map(t => [t.token, t.family, t.faint]), [[" sea", "S", false]]);
  assert.match(early.tokens[0].tip, /^S:out · z 4\.70$/);
  const late = a.pickCell(read(3, 0, [score(5, " sea", 4.7)], [score(7, " stars", 0.41)]));
  assert.deepEqual(late.tokens.map(t => t.token), [" stars"]);
  assert.match(late.tokens[0].tip, /p 0\.41/);
  // In the band both show, the heavier first and the lighter faint; agreeing reads merge.
  const band = a.pickCell(read(1, 0.4, [score(5, " sea", 4.7)], [score(7, " stars", 0.2)]));
  assert.deepEqual(band.tokens.map(t => [t.token, t.faint]), [[" stars", false], [" sea", true]]);
  const agree = a.pickCell(read(1, 0.6, [score(7, " stars", 4.7)], [score(7, " stars", 0.2)]));
  assert.equal(agree.agree, true);
  assert.equal(agree.tokens.length, 1);
  // A heavy half that came back absent gives way to a weighted light one, never to weight 0.
  const absentHeavy = read(1, 0.3, [score(5, " sea", 4.7)], []);
  absentHeavy.r.absent = "no R lens at this layer";
  assert.deepEqual(a.pickCell(absentHeavy).tokens.map(t => [t.token, t.faint]), [[" sea", true]]);
  const absentOnly = read(3, 0, [score(5, " sea", 4.7)], []);
  absentOnly.r.absent = "the R lens stops at L2";
  assert.equal(a.pickCell(absentOnly).absent, "the R lens stops at L2");
  assert.equal(a.pickCell(absentOnly).absentFamily, "R");
  assert.equal(a.pickCell(absentOnly).tokens.length, 0);
  assert.equal(early.absentFamily, null);

  assert.equal(a.lensTag(read(0, 1, [], [])), "S");
  assert.equal(a.lensTag(read(1, 0.6, [], [])), "S R");
  assert.equal(a.lensTag(read(2, 0.25, [], [])), "R S");
  assert.equal(a.lensTag(read(3, 0, [], [])), "R");
  assert.equal(a.weightTip(read(1, 0.6, [], [])), "S-lens 0.60 · R-lens 0.40");

  // By depth: R probabilities, null below the top k; a last layer the R lens misses reads as
  // the model's output. The lead passes over absent layers.
  const stars = 7;
  const lastAbsent = read(3, 0, [], []);
  lastAbsent.r.absent = "the R lens stops at L2";
  const layers = [
    layer(0, read(0, 1, [score(5, " sea", 1)], [score(9, " the", 0.3)])),
    layer(1, read(1, 0.5, [score(5, " sea", 1)], [score(9, " the", 0.2), score(stars, " stars", 0.1)])),
    layer(2, read(2, 0, [], [score(stars, " stars", 0.4)])),
    layer(3, lastAbsent),
  ];
  const depth = a.anatomyByDepth(layers, stars, 0.55);
  assert.deepEqual(depth.map(d => d.p), [null, 0.1, 0.4, 0.55]);
  assert.deepEqual(depth.map(d => d.final), [false, false, false, true]);
  assert.equal(a.anatomyLead(layers, stars), 2);
  assert.equal(a.anatomyLead(layers.slice(0, 2), stars), null);

  // An absent cell's tip says where the family reads instead of the engine's layer list.
  const raw = "layer 3 not in fitted R:out lens layers [0, 1, 2]..[0, 1, 2]";
  lastAbsent.r.absent = raw;
  const sLate = read(0, 1, [], [score(9, " the", 0.3)]);
  sLate.s.absent = "layer 0 not in fitted S:out lens layers [1]..[1]";
  const covered = [layer(0, sLate), ...layers.slice(1)];
  const coverage = a.lensCoverage(covered);
  assert.deepEqual(coverage.get("reads_as:R"), [0, 1, 2]);
  assert.deepEqual(coverage.get("reads_as:S"), [1, 2, 3]);
  assert.deepEqual(coverage.get("steering:R"), [], "no steering read anywhere");
  assert.equal(a.pickCell(lastAbsent).absent, raw, "the cell keeps the engine's reason");
  assert.equal(a.absentTip(coverage, "reads_as", "R", 3), "The R lens stops at L2");
  assert.equal(a.absentTip(coverage, "reads_as", "S", 0), "The S lens starts at L1");
  const gappy = new Map([["reads_as:R", [0, 2]], ["attention:R", []], ["reads_as:S", []], ["attention:S", []]]);
  assert.equal(a.absentTip(gappy, "reads_as", "R", 1), "The R lens skips L1");
  assert.equal(a.absentTip(gappy, "attention", "R", 1), "The R lens does not read attention");
  assert.equal(a.absentTip(gappy, "reads_as", "S", 1), "No S lens is selected");

  // Pushed units: ids as the data writes them, a neuron's pole from its layer's row.
  assert.equal(a.unitShort("neuron/L31n4410"), "L31 n4410");
  assert.equal(a.unitShort("head/L28h12"), "L28 h12");
  const unit = (u, kind, L, index, contribution, activation) => ({ unit: u, kind, layer: L, index, contribution, activation });
  const withUnits = [
    layer(2, read(2, 0, [], []), {
      neurons: [unit("neuron/L2n4", "neuron", 2, 4, -1.5, -3.2)],
      heads: [unit("head/L2h1", "head", 2, 1, 0.75, 2.0)],
    }),
  ];
  const rows = a.pushRows([unit("neuron/L2n4", "neuron", 2, 4, -1.5, null), unit("head/L2h1", "head", 2, 1, 0.75, null)], withUnits);
  assert.deepEqual(rows.map(r => [r.short, r.pole, r.share]), [["L2 n4", "-", 1], ["L2 h1", null, 0.5]]);
  assert.deepEqual(a.layerUnits(withUnits[0], 1).map(u => u.unit), ["neuron/L2n4"]);
  assert.equal(a.formatContribution(-1.53), "−1.5");
  assert.equal(a.formatContribution(0.034), "+0.03");
  assert.equal(a.formatContribution(12.6), "+13");

  // Heat rows: the sink only counts as itself; the strongest places with a neighbour each, the
  // query at the end, gaps elided; the tint is Ledger's 9% for an attended token at its strongest
  // (22% is a unit's firing), graded below it.
  const context = ["<s>", "Above", " the", " black", " water", ",", " and", " the"];
  const weights = [0.6, 0.02, 0.01, 0.15, 0.1, 0.01, 0.01, 0.1];
  const heat = a.heatRow(weights, context);
  assert.equal(heat.sink, 0.6);
  assert.deepEqual(heat.pieces.map(p => p.text), ["…", " the", " black", " water", ",", " and", " the"]);
  assert.equal(a.HEAT_CAP, 9);
  assert.equal(Math.max(...heat.pieces.map(p => p.tint)), 9);
  assert.equal(heat.pieces.find(p => p.text === " water").tint, 6, "graded: two thirds of the peak");
  assert.equal(heat.pieces.at(-1).here, true);
  const gapped = a.heatRow([0.5, 0.3, 0, 0, 0, 0, 0, 0.2], context, { keep: 1 });
  assert.deepEqual(gapped.pieces.map(p => p.text), ["Above", " the", "…", " and", " the"]);

  // Labels: what a unit writes first, the rest in the tip.
  assert.deepEqual(a.labelText({ writes: "night-sky words", fires: "stars", credit: "c", url: "u" }, "neuron"),
    { text: "night-sky words", tip: "fires on stars · writes night-sky words" });
  assert.equal(a.labelText({ attends: "the subject", credit: "c", url: "u" }, "head").text, "the subject");
  assert.equal(a.labelText(null, "head"), null);

  // The anatomy queues behind a running readout (the server refuses overlapping model work).
  const calls = [];
  const envelope = { measurements: { version: 2, scope: "replay", provenance: "replayed", instruments: {
    lens: { binding: { source: "t", steering: null }, readings: {}, readout: { layers: [], aggregate: [] } },
  } } };
  const anatomyEnvelope = { measurements: { ...envelope.measurements, anatomy: { token: {}, layers: [] } } };
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "browser", instruments: {
    tokenReadout: (...args) => new Promise((resolve) => calls.push({ kind: "readout", args, resolve })),
    tokenAnatomy: (...args) => new Promise((resolve) => calls.push({ kind: "anatomy", args, resolve })),
  } });
  const cache = await server.ssrLoadModule("/src/lib/runtime/tokenReadoutCache.ts");
  const readout = cache.cachedTokenReadout("lens", "n1", 3, {});
  const anatomy = cache.cachedTokenAnatomy("n1", 3, { k: 8, steered: false });
  await flush();
  assert.deepEqual(calls.map(c => c.kind), ["readout"], "the anatomy must wait for the running readout");
  calls[0].resolve(envelope);
  await readout;
  await flush();
  assert.deepEqual(calls.map(c => c.kind), ["readout", "anatomy"]);
  assert.deepEqual(calls[1].args.slice(0, 3), ["n1", 3, { k: 8, steered: false }]);
  calls[1].resolve(anatomyEnvelope);
  assert.equal(await anatomy, anatomyEnvelope);
  assert.equal(await cache.cachedTokenAnatomy("n1", 3, { k: 8, steered: false }), anatomyEnvelope, "a settled anatomy is reused");
  assert.equal(calls.length, 2);
  // Unsteered is its own read; a lens change invalidates the anatomy.
  const steered = cache.cachedTokenAnatomy("n1", 3, { k: 8, steered: true });
  await flush();
  assert.equal(calls.length, 3);
  calls[2].resolve({ measurements: envelope.measurements });
  await assert.rejects(steered, /No anatomy came back/);
  cache.invalidateTokenReadoutCache("lens");
  void cache.cachedTokenAnatomy("n1", 3, { k: 8, steered: false });
  await flush();
  assert.equal(calls.length, 4, "a lens change drops the kept anatomy");
  console.log("token anatomy: ok");
} finally {
  await server.close();
}
