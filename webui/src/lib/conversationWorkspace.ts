import {
  buildConversationImportPlan,
  CONVERSATION_SAMPLING_KEYS,
  CONVERSATION_SNAPSHOT_VERSION,
  runConversationImportTransaction,
  validateConversationSnapshot,
  type ConversationImportPlan,
  type ConversationProbeRow,
  type ConversationSnapshotV8,
} from "./conversationSnapshot";
import { runtimeClient } from "./runtime/client";
import { apiSessions, apiTree } from "./runtime/services";
import {
  attachProbe,
  currentLoomTreeSnapshot,
  detachProbe,
  highlightState,
  probeRack,
  refreshLoomTree,
  restoreRack,
  samplingState,
  sessionState,
  steerRack,
} from "./stores.svelte";
import type { SamplingState } from "./stores.svelte";
import type { LoomTreeJSON, ProbeRackEntry, ProbeRequest, SteerEntry } from "./types";

interface WorkspaceRollback {
  tree: LoomTreeJSON;
  steerEntries: Array<[string, SteerEntry]>;
  subspaceAlong: number;
  readOnlyRecipe: string | null;
  probeRequests: ProbeRequest[];
  probeEntries: Array<[string, ProbeRackEntry]>;
  probeSortMode: typeof probeRack.sortMode;
  highlight: {
    target: string | null;
    compareTarget: string | null;
    compareTwo: boolean;
    smoothBlend: boolean;
  };
  sampling: SamplingState;
}

export interface ConversationRestoreSummary {
  turns: number;
  terms: number;
  probes: number;
}

export function captureConversationSnapshot(now = new Date()): ConversationSnapshotV8 {
  return snapshotWith((tree) => clone(tree), now);
}

/** The workspace alone (the tree reduced to its identity, no nodes): what the page keeps
 *  synchronously as it hides (``workspaceDraft``), when a change has not reached autosave yet.
 *  Cheap whatever the tree's size. */
export function captureWorkspaceSnapshot(now = new Date()): ConversationSnapshotV8 {
  return snapshotWith((tree) => ({ ...tree, nodes: [], children_of: {}, active_node_id: tree.root_id }), now);
}

function snapshotWith(
  treeOf: (tree: NonNullable<ReturnType<typeof currentLoomTreeSnapshot>>) => ConversationSnapshotV8["tree"],
  now: Date,
): ConversationSnapshotV8 {
  const info = sessionState.info;
  const tree = currentLoomTreeSnapshot();
  if (!info || !tree) throw new Error("The conversation is still loading");
  const snapshot: ConversationSnapshotV8 = {
    version: CONVERSATION_SNAPSHOT_VERSION,
    savedAt: now.toISOString(),
    model_id: info.model_id,
    session_id: info.id,
    tree: treeOf(tree),
    steerRack: [...steerRack.entries.entries()].map(([name, entry]) => ({
      name,
      ...clone(entry),
    })),
    subspaceAlong: steerRack.subspaceAlong,
    readOnlyRecipe: steerRack.readOnlyRecipe,
    probeRack: {
      sortMode: probeRack.sortMode,
      active: [...probeRack.active],
      entries: [...probeRack.entries.entries()].map(([name, entry]) => ({
        name,
        request: clone(entry.request),
        sparkline: [...entry.sparkline],
        current: entry.current,
        previous: entry.previous,
      })),
    },
    highlightState: clone({ ...highlightState }),
    samplingState: clone({ ...samplingState }),
  };
  validateConversationSnapshot(snapshot, CONVERSATION_SAMPLING_KEYS);
  return snapshot;
}

export async function restoreConversationSnapshot(
  snapshot: ConversationSnapshotV8,
): Promise<ConversationRestoreSummary> {
  const portableSnapshot = clone(snapshot);
  validateConversationSnapshot(portableSnapshot, CONVERSATION_SAMPLING_KEYS);
  const plan = buildConversationImportPlan(portableSnapshot);
  await runConversationImportTransaction({
    capture: captureWorkspace,
    async preflight() {
      const info = sessionState.info;
      if (!info || portableSnapshot.model_id !== info.model_id) {
        throw new Error("This conversation belongs to a different model");
      }
      const validation = await apiSessions.validateSteering(
        plan.steeringExpression,
        undefined,
        runtimeClient.mode === "browser"
          ? { probeRequests: plan.probeRequests, tree: plan.snapshot.tree }
          : undefined,
      );
      if (!validation.valid) throw new Error(validation.error ?? "Steering preflight failed");
    },
    async apply() {
      await apiTree.restore(plan.snapshot.tree);
      await replaceProbeRoster(plan.probeRequests);
      applyLocalPlan(plan);
      await refreshLoomTree();
    },
    rollback: rollbackWorkspace,
  });
  return {
    turns: portableSnapshot.tree.nodes.filter((node) => node.parent_id !== null).length,
    terms: portableSnapshot.steerRack.length,
    probes: portableSnapshot.probeRack.active.length,
  };
}

/** A reload in server mode: the server still holds the tree and the probe roster, but the
 *  workspace the page kept (the steer rack, its read-only recipe, sampling, the highlight and the
 *  probe sort) lived only in the page. Put it back from the autosave record of this tree, so the
 *  autosave that follows never writes an empty rack over the saved one. The rack validates as it
 *  restores (``restoreRack``): a recipe that no longer resolves stays read-only with its reason. */
export function restoreWorkspaceSettings(snapshot: ConversationSnapshotV8): void {
  const portableSnapshot = clone(snapshot);
  validateConversationSnapshot(portableSnapshot, CONVERSATION_SAMPLING_KEYS);
  const plan = buildConversationImportPlan(portableSnapshot);
  restoreRack([...plan.steerEntries].map(([name, entry]) => [name, clone(entry)]), plan.snapshot.subspaceAlong, plan.snapshot.readOnlyRecipe);
  Object.assign(samplingState, clone(plan.snapshot.samplingState));
  Object.assign(highlightState, clone(plan.snapshot.highlightState));
  probeRack.sortMode = plan.snapshot.probeRack.sortMode;
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

async function captureWorkspace(): Promise<WorkspaceRollback> {
  const probeEntries = [...probeRack.entries].map(([name, entry]) =>
    [name, clone(entry)] as [string, ProbeRackEntry]
  );
  const byName = new Map(probeEntries);
  return {
    tree: await apiTree.get(),
    steerEntries: [...steerRack.entries].map(([name, entry]) =>
      [name, clone(entry)] as [string, SteerEntry]
    ),
    subspaceAlong: steerRack.subspaceAlong,
    readOnlyRecipe: steerRack.readOnlyRecipe,
    probeRequests: probeRack.active.map((name) => clone(byName.get(name)!.request)),
    probeEntries,
    probeSortMode: probeRack.sortMode,
    highlight: clone({ ...highlightState }),
    sampling: clone({ ...samplingState }),
  };
}

async function replaceProbeRoster(requests: readonly ProbeRequest[]): Promise<void> {
  for (const name of [...probeRack.active]) await detachProbe(name);
  for (const request of requests) {
    await attachProbe(request.selector, { name: request.name, top_n: request.top_n });
  }
}

function restoreProbeRows(
  rows: Iterable<[string, ProbeRackEntry | ConversationProbeRow]>,
): void {
  for (const [name, saved] of rows) {
    const attached = probeRack.entries.get(name);
    if (!attached) throw new Error(`Probe ${name} was not restored`);
    if ("info" in saved) {
      probeRack.entries.set(name, { ...clone(saved), info: attached.info });
    } else {
      probeRack.entries.set(name, {
        ...attached,
        request: clone(saved.request),
        sparkline: [...saved.sparkline],
        current: saved.current,
        previous: saved.previous,
      });
    }
  }
}

function applyLocalPlan(plan: ConversationImportPlan): void {
  restoreRack([...plan.steerEntries].map(([name, entry]) => [name, clone(entry)]), plan.snapshot.subspaceAlong, plan.snapshot.readOnlyRecipe);
  Object.assign(samplingState, clone(plan.snapshot.samplingState));
  Object.assign(highlightState, clone(plan.snapshot.highlightState));
  probeRack.sortMode = plan.snapshot.probeRack.sortMode;
  restoreProbeRows(plan.probeRows);
}

async function rollbackWorkspace(previous: WorkspaceRollback): Promise<void> {
  const failures: unknown[] = [];
  try {
    await apiTree.restore(previous.tree);
  } catch (error) {
    failures.push(error);
  }
  try {
    await replaceProbeRoster(previous.probeRequests);
  } catch (error) {
    failures.push(error);
  }
  restoreRack(previous.steerEntries.map(([name, entry]) => [name, clone(entry)]), previous.subspaceAlong, previous.readOnlyRecipe);
  Object.assign(samplingState, previous.sampling);
  Object.assign(highlightState, previous.highlight);
  probeRack.sortMode = previous.probeSortMode;
  try {
    restoreProbeRows(previous.probeEntries);
  } catch (error) {
    failures.push(error);
  }
  try {
    await refreshLoomTree();
  } catch (error) {
    failures.push(error);
  }
  if (failures.length > 0) throw new AggregateError(failures, "Workspace rollback failed");
}
