import type { RuntimeProgressEvent, TokenAnatomyOptions } from "./contracts";
import { apiInstruments } from "./services";
import type { InstrumentFamily, MeasurementsEnvelopeJSON } from "../types";

type TokenReadoutOptions = {
  topK?: number;
  steered?: boolean;
  raw?: boolean;
  layers?: string;
};

type TokenReadoutResponse = { measurements: MeasurementsEnvelopeJSON };
type TokenReadoutPriority = "interactive" | "background";

interface CacheEntry {
  key: string;
  /** The family whose epoch invalidates the entry (the anatomy reads through the lens). */
  family: InstrumentFamily;
  nodeId: string;
  rawIndex: number;
  options: TokenReadoutOptions;
  priority: TokenReadoutPriority;
  promise: Promise<TokenReadoutResponse>;
  resolve: (value: TokenReadoutResponse) => void;
  reject: (error: unknown) => void;
  progress: RuntimeProgressEvent | null;
  listeners: Set<(event: RuntimeProgressEvent) => void>;
  state: "queued" | "active" | "settled";
  invalidated: boolean;
  /** The runtime call, with the check its answer must pass. */
  call: (entry: CacheEntry) => Promise<TokenReadoutResponse>;
}

const MAX_TOKEN_READOUTS = 96;
const MAX_PENDING_TOKEN_READOUTS = 96;
const cache = new Map<string, CacheEntry>();
const pending: CacheEntry[] = [];
const epochs: Record<InstrumentFamily, number> = {
  geometry: 0,
  lens: 0,
  sae: 0,
  units: 0,
};
let active: CacheEntry | null = null;

export function cachedTokenReadout(
  family: InstrumentFamily,
  nodeId: string,
  rawIndex: number,
  options: TokenReadoutOptions,
  onProgress?: (event: RuntimeProgressEvent) => void,
  priority: TokenReadoutPriority = "interactive",
): Promise<TokenReadoutResponse> {
  const key = tokenReadoutKey(family, nodeId, rawIndex, options);
  return queued(key, family, nodeId, rawIndex, { ...options }, readoutCall, onProgress, priority);
}

/** One token's anatomy (the node replay), through the same one-at-a-time queue as the readouts:
 *  both run model work, which the server refuses to overlap. Invalidated with the lens. */
export function cachedTokenAnatomy(
  nodeId: string,
  rawIndex: number,
  options: TokenAnatomyOptions,
  onProgress?: (event: RuntimeProgressEvent) => void,
  priority: TokenReadoutPriority = "interactive",
): Promise<TokenReadoutResponse> {
  const fixed = { ...options };
  const key = JSON.stringify([
    "anatomy",
    epochs.lens,
    nodeId,
    rawIndex,
    fixed.k ?? null,
    fixed.units ?? null,
    fixed.patterns ?? null,
    fixed.steered ?? true,
    fixed.raw ?? false,
  ]);
  const call = async (entry: CacheEntry): Promise<TokenReadoutResponse> => {
    const read = apiInstruments.tokenAnatomy;
    if (!read) throw new Error("This runtime has no token anatomy.");
    const value = await read.call(apiInstruments, entry.nodeId, entry.rawIndex, fixed);
    if (!value.measurements.anatomy) throw new Error("No anatomy came back for this token. Select it again to retry.");
    return value;
  };
  return queued(key, "lens", nodeId, rawIndex, {}, call, onProgress, priority);
}

function queued(
  key: string,
  family: InstrumentFamily,
  nodeId: string,
  rawIndex: number,
  options: TokenReadoutOptions,
  call: CacheEntry["call"],
  onProgress: ((event: RuntimeProgressEvent) => void) | undefined,
  priority: TokenReadoutPriority,
): Promise<TokenReadoutResponse> {
  let entry = cache.get(key);
  if (!entry) {
    if (pending.length >= MAX_PENDING_TOKEN_READOUTS) {
      return Promise.reject(new Error("The token reading queue is full. Wait for a reading to finish, then select this token again."));
    }
    let resolve!: (value: TokenReadoutResponse) => void;
    let reject!: (error: unknown) => void;
    const promise = new Promise<TokenReadoutResponse>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    entry = {
      key,
      family,
      nodeId,
      rawIndex,
      options,
      priority,
      promise,
      resolve,
      reject,
      progress: null,
      listeners: new Set(),
      state: "queued",
      invalidated: false,
      call,
    };
    cache.set(key, entry);
    enqueue(entry);
  } else {
    touch(key, entry);
    if (entry.state === "queued" && priority === "interactive") promote(entry);
  }

  if (onProgress) {
    entry.listeners.add(onProgress);
    if (entry.progress) onProgress(entry.progress);
  }
  pump();
  updateQueuedProgress();

  if (!onProgress) return entry.promise;
  const selected = entry;
  return entry.promise.finally(() => selected.listeners.delete(onProgress));
}

async function readoutCall(entry: CacheEntry): Promise<TokenReadoutResponse> {
  const value = await apiInstruments.tokenReadout(
    entry.family,
    entry.nodeId,
    entry.rawIndex,
    entry.options,
    undefined,
    (event) => notify(entry, event),
  );
  const block = value.measurements.instruments[entry.family];
  // Geometry and units answer readings only; the lens and SAE answer a ranked readout too.
  const readingsOnly = entry.family === "geometry" || entry.family === "units";
  if (!block || (!readingsOnly && (!("readout" in block) || !block.readout))) {
    throw new Error("No reading was returned for this token. Check the active reading source and try again.");
  }
  return value;
}

export function tokenReadoutCacheEpoch(family: InstrumentFamily): number {
  return epochs[family];
}

export function invalidateTokenReadoutCache(family?: InstrumentFamily): void {
  const families: InstrumentFamily[] = family
    ? [family]
    : ["geometry", "lens", "sae"];
  for (const current of families) epochs[current] += 1;
  for (const [key, entry] of cache) {
    if (!families.includes(entry.family)) continue;
    cache.delete(key);
    entry.invalidated = true;
    if (entry.state !== "settled") {
      entry.reject(new Error("The conversation or reading source changed. Select the token again to get a current reading."));
      entry.listeners.clear();
    }
  }
  for (let index = pending.length - 1; index >= 0; index -= 1) {
    if (pending[index].invalidated) pending.splice(index, 1);
  }
  updateQueuedProgress();
}

function enqueue(entry: CacheEntry): void {
  if (entry.priority === "interactive") pending.unshift(entry);
  else pending.push(entry);
}

function promote(entry: CacheEntry): void {
  entry.priority = "interactive";
  const index = pending.indexOf(entry);
  if (index <= 0) return;
  pending.splice(index, 1);
  pending.unshift(entry);
}

function pump(): void {
  if (active || pending.length === 0) return;
  const entry = pending.shift()!;
  active = entry;
  entry.state = "active";
  void run(entry);
}

async function run(entry: CacheEntry): Promise<void> {
  try {
    const value = await entry.call(entry);
    if (entry.invalidated) return;
    entry.state = "settled";
    if (cache.get(entry.key) === entry) touch(entry.key, entry);
    entry.resolve(value);
  } catch (error) {
    if (cache.get(entry.key) === entry) cache.delete(entry.key);
    entry.reject(error);
  } finally {
    entry.listeners.clear();
    if (active === entry) active = null;
    trim();
    pump();
    updateQueuedProgress();
  }
}

function updateQueuedProgress(): void {
  const activeCount = active === null ? 0 : 1;
  for (let index = 0; index < pending.length; index += 1) {
    const entry = pending[index];
    const ahead = activeCount + index;
    const message = ahead === 0
      ? "Starting token reading"
      : `Waiting for ${ahead} ${ahead === 1 ? "token reading" : "token readings"}`;
    notify(entry, {
      event: "progress",
      data: {
        kind: "token_readout",
        family: entry.family,
        nodeId: entry.nodeId,
        rawIndex: entry.rawIndex,
        phase: "queued",
        completed: 0,
        total: entry.rawIndex + 1,
        progress: 0,
        message,
      },
    });
  }
}

function notify(entry: CacheEntry, event: RuntimeProgressEvent): void {
  if (entry.invalidated) return;
  entry.progress = event;
  for (const listener of entry.listeners) listener(event);
}

function tokenReadoutKey(
  family: InstrumentFamily,
  nodeId: string,
  rawIndex: number,
  options: TokenReadoutOptions,
): string {
  return JSON.stringify([
    family,
    epochs[family],
    nodeId,
    rawIndex,
    options.topK ?? null,
    options.steered ?? true,
    options.raw ?? false,
    options.layers ?? null,
  ]);
}

function touch(key: string, entry: CacheEntry): void {
  cache.delete(key);
  cache.set(key, entry);
}

function trim(): void {
  if (cache.size <= MAX_TOKEN_READOUTS) return;
  for (const [key, entry] of cache) {
    if (cache.size <= MAX_TOKEN_READOUTS) return;
    if (entry.state === "settled") cache.delete(key);
  }
}
