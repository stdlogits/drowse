import assert from "node:assert/strict";
import { createServer } from "vite";

// The correlation drawer's omitted probes: why each active probe is not one residual direction,
// in 6.0's names (the R lens reads ``jlens/`` atoms, the S lens ``slens/``).
const server = await createServer({
  configFile: false, appType: "custom", logLevel: "error",
  optimizeDeps: { noDiscovery: true, include: [] },
  server: { middlewareMode: true, watch: null, hmr: false, ws: false },
});
try {
  const { omittedAnalyticsProbes, comparisonProfileNames } = await server.ssrLoadModule("/src/lib/profileAnalytics.ts");
  const entry = (info) => ({ info });
  const entries = new Map([
    ["calm", entry({ family: "geometry", is_affine: true, intrinsic_dim: 1 })],
    ["jlens/happy", entry({ family: "lens", namespace: "jlens", is_affine: true, intrinsic_dim: 1 })],
    ["slens/happy", entry({ family: "lens", namespace: "slens", is_affine: true, intrinsic_dim: 1 })],
    ["sae/12", entry({ family: "sae", is_affine: true, intrinsic_dim: 1 })],
    ["neuron/L3n7", entry({ family: "units", is_affine: true, intrinsic_dim: 1 })],
    ["months", entry({ family: "geometry", is_affine: false, intrinsic_dim: 2 })],
    ["personas", entry({ family: "geometry", is_affine: true, intrinsic_dim: 3 })],
  ]);
  const omitted = omittedAnalyticsProbes([...entries.keys(), "gone"], entries, ["calm"]);
  assert.deepEqual(Object.fromEntries(omitted.map(({ name, reason }) => [name, reason])), {
    "gone": "not exposed as a direction by this runtime",
    "jlens/happy": "R-lens readout, not a residual direction",
    "months": "curved manifold, not one direction",
    "neuron/L3n7": "a neuron or head, not a residual direction",
    "personas": "multidimensional subspace, not one direction",
    "sae/12": "SAE readout, not a residual direction",
    "slens/happy": "S-lens readout, not a residual direction",
  });
  assert.ok(!omitted.some(({ reason }) => /J-lens/.test(reason)), "6.0 calls the jlens/ family the R lens");
  assert.deepEqual(comparisonProfileNames(["b"], ["calm", "jlens/happy"], entries, true), ["b", "calm"]);
  console.log("profile analytics: omitted probes say why, in R/S lens names");
} finally { await server.close(); }
