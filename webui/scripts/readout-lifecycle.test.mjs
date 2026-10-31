import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const flush = () => new Promise(resolve => setImmediate(resolve));
const envelope = { measurements: { instruments: { geometry: { readings: {} } } } };
try {
  const requests = [];
  const metadataRequests = [];
  const sourceRequests = [];
  let replay = (...args) => new Promise((resolve, reject) => requests.push({ args, resolve, reject }));
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "browser", instruments: {
    tokenReadout: (...args) => replay(...args),
    saeFeaturesMetadata: ids => new Promise((resolve, reject) => metadataRequests.push({ ids, resolve, reject })),
    sources: family => { sourceRequests.push(family); return Promise.resolve({ sources: [], releases: [] }); },
  } });
  const { cachedTokenReadout, invalidateTokenReadoutCache } =
    await server.ssrLoadModule("/src/lib/runtime/tokenReadoutCache.ts");
  const accepted = [];
  for (let index = 0; index < 97; index += 1) {
    accepted.push(cachedTokenReadout("geometry", `queued-${index}`, 0, {}).catch(error => error));
  }
  assert.equal(requests.length, 1, "replays stay serialized");
  for (let index = 0; index < 500; index += 1) {
    await assert.rejects(cachedTokenReadout("geometry", `overflow-${index}`, 0, {}), /queue is full/);
  }
  const duplicate = cachedTokenReadout("geometry", "queued-96", 0, {}).catch(error => error);
  invalidateTokenReadoutCache();
  const cancelled = await Promise.all([...accepted, duplicate]);
  assert.ok(cancelled.every(error => /reading source changed/.test(error.message)));
  requests[0].resolve(envelope);
  await flush();
  assert.equal(requests.length, 1, "invalidated queued work never reaches the model");
  replay = () => { throw new Error("transport unavailable"); };
  await assert.rejects(cachedTokenReadout("geometry", "sync-error", 0, {}), /transport unavailable/);
  replay = async () => envelope;
  assert.deepEqual(await cachedTokenReadout("geometry", "recovered", 0, {}), envelope,
    "synchronous transport failures must release the active slot");
  const { sessionState } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
  const { saeState, recordSaeReadoutFrame, backfillSaeMeta, rehydrateInstrumentsFromSession } =
    await server.ssrLoadModule("/src/lib/stores/instruments.svelte.ts");
  const selectPack = source => {
    sessionState.info = { instruments: [{ family: "sae", source, live: { enabled: true, layer: 13 } }] };
    rehydrateInstrumentsFromSession();
  };
  selectPack("first-dictionary");
  recordSaeReadoutFrame([{ id: 2286, activation: 1, label: null, max_act: 42 }]);
  const firstMetadata = backfillSaeMeta();
  assert.deepEqual(metadataRequests[0].ids, [2286], "a maximum without a label still needs descriptions");
  metadataRequests[0].resolve({ features: { "2286": { label: "Reuters", max_act: 42 } } });
  await firstMetadata;
  recordSaeReadoutFrame([{ id: 2286, activation: 2, label: null, max_act: 42 }]);
  assert.equal(saeState.meta.get(2286).label, "Reuters", "later unlabeled frames preserve fetched descriptions");
  recordSaeReadoutFrame([{ id: 396, activation: 3, label: null, max_act: null }]);
  const staleMetadata = backfillSaeMeta();
  selectPack("second-dictionary");
  metadataRequests[1].resolve({ features: { "396": { label: "wrong dictionary", max_act: null } } });
  await staleMetadata;
  assert.equal(saeState.meta.has(396), false, "late responses cannot label another dictionary's features");
  recordSaeReadoutFrame([{ id: 396, activation: 4, label: null, max_act: null }]);
  const offline = backfillSaeMeta();
  metadataRequests[2].reject(new Error("offline"));
  await offline;
  const retry = backfillSaeMeta();
  metadataRequests[3].resolve({ features: { "396": { label: "correct dictionary", max_act: null } } });
  await retry;
  assert.equal(saeState.meta.get(396).label, "correct dictionary");
  recordSaeReadoutFrame([{ id: 17, activation: 1, label: null, max_act: 20 }]);
  const partial = backfillSaeMeta();
  metadataRequests[4].resolve({ features: {} });
  await partial;
  const partialRetry = backfillSaeMeta();
  assert.deepEqual(metadataRequests[5].ids, [17], "an omitted feature is a retryable lookup failure");
  metadataRequests[5].resolve({ features: { "17": { label: "recovered label", max_act: null } } });
  await partialRetry;
  assert.deepEqual(saeState.meta.get(17), { label: "recovered label", max_act: 20 });
  console.log("SAE metadata lifecycle: missing-label backfill, frame preservation, pack-switch isolation, and offline retry passed");

  // The SAE source picker discovers releases through the one sources refresh. A model with no
  // SAELens release answers an empty list, and that must stay one request (regression: the picker
  // re-fetched whenever its release list came back empty, about 1,200 requests a second).
  const { saeSourceState, refreshSaeSources } = await server.ssrLoadModule("/src/lib/stores/instruments.svelte.ts");
  await Promise.all([refreshSaeSources(), refreshSaeSources()]);
  for (let tick = 0; tick < 20; tick += 1) await flush();
  assert.deepEqual(sourceRequests, ["sae"], "an empty release list is one request, and a concurrent refresh adds none");
  assert.deepEqual([saeSourceState.sources, saeSourceState.releases], [[], []]);
  const { readFile } = await import("node:fs/promises");
  const picker = await readFile(new URL("../src/panels/SaeSourceSection.svelte", import.meta.url), "utf8");
  assert.doesNotMatch(picker, /apiInstruments|\.sources\(/, "the picker reads saeSourceState and never fetches on its own");
  console.log("SAE source discovery: one request for an empty release list passed");
  console.log("Readout lifecycle: bounded pending work, 500 overflow attempts, deduplication, invalidation, and transport recovery passed");
} finally {
  await server.close();
}
