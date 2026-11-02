import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({ root, optimizeDeps: { noDiscovery: true, include: [] }, appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
try {
  const { readWorkProgress, remainingWorkMs, elapsedLabel, etaLabel } = await server.ssrLoadModule("/src/lib/manifoldProgress.ts");
  const { EXAMPLES_PER_ROUND, EXAMPLE_BUDGETS } = await server.ssrLoadModule("/src/lib/exampleBudget.ts");
  assert.equal(EXAMPLES_PER_ROUND, 48);
  assert.deepEqual(EXAMPLE_BUDGETS.map(item => item.rounds * EXAMPLES_PER_ROUND), [48, 96, 192]);
  const concepts = ["pirate", "assistant"];
  assert.deepEqual(readWorkProgress({ message: 'Generating "pirate" response 1/48...' }, concepts), {
    stage: "generating", label: "Generating examples", completed: 0, total: 96, detail: "pirate: example 1 of 48",
  });
  assert.equal(readWorkProgress({ message: 'Generating "assistant" response 7/48...' }, concepts).completed, 54);
  assert.equal(readWorkProgress({ message: "Generating 'pirate' responses 9-16/48..." }, concepts).completed, 8, "Python batch starts count only completed work");
  assert.equal(readWorkProgress({ stage: "capturing", completed: 12, total: 48 }).label, "Reading examples");
  assert.equal(readWorkProgress({ stage: "fitting", completed: NaN, total: 4 }).completed, null);
  assert.equal(readWorkProgress(null), null);
  const samples = [{ at: 0, completed: 0 }, { at: 3000, completed: 3 }];
  assert.equal(remainingWorkMs(samples, 10, 3000), 7000);
  assert.equal(remainingWorkMs(samples, 10, 4000), 6000);
  assert.equal(remainingWorkMs(samples, 10, 30000), null, "stalled jobs stop presenting a stale countdown");
  assert.equal(remainingWorkMs(samples.slice(0, 1), 10, 3000), null);
  assert.equal(remainingWorkMs(samples, 3, 3000), null);
  assert.equal(elapsedLabel(326000), "5m 26s");
  assert.equal(etaLabel(61000), "About 2 minutes left");

  const { manifoldJobs, beginManifoldJob, updateManifoldJob, finishManifoldJob, failManifoldJob, dismissManifoldJob } = await server.ssrLoadModule("/src/lib/stores/manifoldJobs.svelte.ts");
  const generating = beginManifoldJob("local", "pirate", "generating", concepts, true);
  const startedAt = manifoldJobs.current.startedAt;
  assert.throws(() => beginManifoldJob("local", "other", "fitting"), /Another creation job/);
  updateManifoldJob(generating, { event: "progress", data: { message: 'Generating "pirate" response 5/48...' } });
  assert.equal(manifoldJobs.current.progress.completed, 4);
  dismissManifoldJob();
  assert.equal(manifoldJobs.current.id, generating, "running progress cannot be dismissed");
  finishManifoldJob(generating);
  assert.equal(manifoldJobs.current.status, "waiting");
  const fitting = beginManifoldJob("local", "pirate", "fitting");
  assert.equal(manifoldJobs.current.startedAt, startedAt, "the two stages share one elapsed timer");
  updateManifoldJob(generating, { event: "progress", data: { message: 'Generating "pirate" response 9/48...' } });
  assert.equal(manifoldJobs.current.progress.stage, "fitting", "stale events cannot rewrite a new stage");
  updateManifoldJob(fitting, { event: "progress", data: { stage: "capturing", completed: 40, total: 96 } });
  updateManifoldJob(fitting, { event: "progress", data: { stage: "fitting", completed: 1, total: 4 } });
  assert.equal(manifoldJobs.current.samples.length, 1, "each step has its own timing samples");
  finishManifoldJob(fitting);
  assert.equal(manifoldJobs.current.status, "complete");
  assert.equal(manifoldJobs.current.progress.completed, manifoldJobs.current.progress.total);
  const cancelled = beginManifoldJob("local", "pirate", "fitting");
  failManifoldJob(cancelled, { code: "FITTING_CANCELLED" });
  assert.equal(manifoldJobs.current.status, "cancelled");
  const failed = beginManifoldJob("local", "pirate", "fitting");
  failManifoldJob(failed, new Error("Test fit error"));
  assert.equal(manifoldJobs.current.status, "failed");
  assert.match(manifoldJobs.current.error, /Test fit error/);
  dismissManifoldJob();
  assert.equal(manifoldJobs.current, null);

  // The band's note is a state, never an instruction: what to do next rides the library button's tip.
  const { jobNote } = await server.ssrLoadModule("/src/lib/manifolds/jobNote.ts");
  const job = (phase, status, fitAfterwards = false) => ({ phase, status, fitAfterwards });
  assert.deepEqual(jobNote(job("generating", "running", true), true), { note: "Step 1 of 2 · Keep this tab open", tip: "" });
  assert.deepEqual(jobNote(job("fitting", "waiting", true), false), { note: "Step 2 of 2", tip: "" });
  assert.deepEqual(jobNote(job("fitting", "cancelling"), false), { note: "", tip: "" });
  assert.equal(jobNote(job("fitting", "complete"), false).note, "Saved and ready to steer or probe.", "drowse.ai's creation e2e matches this sentence");
  assert.equal(jobNote(job("extracting", "complete"), false).note, "Saved and ready to steer or probe.");
  for (const phase of ["generating", "scoring", "installing"]) {
    const { note, tip } = jobNote(job(phase, "complete"), false);
    assert.equal(note, "", `${phase}: the title already says it is done`);
    assert.ok(tip.length > 0, `${phase}: what to do next is a tip`);
  }
  for (const status of ["failed", "cancelled"]) assert.deepEqual(jobNote(job("fitting", status), false), { note: "Saved concept groups are kept", tip: "Reopen the creator or library to try again" });
  for (const phase of ["generating", "fitting", "extracting", "installing", "scoring"]) for (const status of ["running", "waiting", "cancelling", "complete", "cancelled", "failed"]) {
    const { note } = jobNote(job(phase, status, true), true);
    assert.ok(!/\. \S|: see|\bwhen ready\b|try again/.test(note), `${phase}/${status}: "${note}" tells rather than shows`);
  }
  console.log("Manifold progress: counts, batch compatibility, ETA, stalls, stage chaining, cancellation, failures and the band's notes passed");
} finally { await server.close(); }
