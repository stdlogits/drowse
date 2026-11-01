// Shared captured-or-replay readout resource for the drilldown's
// instrument tabs.
//
// Every replay-capable family (geometry / sae / lens) reads the same
// way: prefer the token's loom-owned ``measurements`` envelope (original
// capture), otherwise hit the family's ``token-readout`` replay endpoint.
// Replays are keyed and shared so pending and completed token results survive
// navigation and duplicate hover/drilldown requests reuse the same work.
// This class is the one implementation the three tabs share.

import { describeError } from "../../lib/runtime/services";
import type { RuntimeProgressEvent, TokenAnatomyOptions } from "../../lib/runtime/contracts";
import {
  cachedTokenAnatomy,
  cachedTokenReadout,
  tokenReadoutCacheEpoch,
} from "../../lib/runtime/tokenReadoutCache";
import type { InstrumentFamily } from "../../lib/types";
import type {
  MeasurementsEnvelopeJSON,
  ProbeReadingJSON,
  ScalarReadingJSON,
} from "../../lib/types";

export type ReadoutOrigin = "captured" | "replayed" | null;

export interface ReadoutProgress {
  phase: "queued" | "context" | "readout" | "complete";
  completed: number;
  total: number;
  progress: number;
  message: string;
}

interface ReplayEntry<T> {
  state: "loading" | "ready" | "failed";
  data: T | null;
  source: string | null;
  pins: Pins | null;
  error: string | null;
  progress: ReadoutProgress | null;
}

/** The lens or SAE probes pinned while the token was read, by name: a capture's, or a replay's
 *  (the lens and SAE replays and the anatomy return the attached probes' readings). */
export type Pins = Record<string, ScalarReadingJSON>;

/** What a family read maps its envelope to: the view, its source and its pinned probes. */
export interface ReadoutView<T> {
  data: T;
  source: string | null;
  pins?: Pins | null;
}

const MAX_REPLAY_CACHE_ENTRIES = 48;

/** The geometry tab's render shape — the Monitor-roster readings plus
 *  the steering the read ran under (captured: the recipe; replayed: the
 *  binding the endpoint reports). */
export interface GeometryTokenReadout {
  steering: string | null;
  readings: Record<string, ProbeReadingJSON>;
}

export class ReplayReadout<T> {
  data: T | null = $state(null);
  loading: boolean = $state(false);
  error: string | null = $state(null);
  origin: ReadoutOrigin = $state(null);
  source: string | null = $state(null);
  /** The probes the shown read pinned (captured or replayed with it), else null. */
  pins: Pins | null = $state(null);
  progress: ReadoutProgress | null = $state(null);
  #selectedKey: string | null = null;
  /** The saved capture a replay stands on: shown while the replay loads and kept beside its error
   *  if it fails, so a failed replay never takes away what the row saved. */
  #base: ReadoutView<T> | null = null;
  #entries = new Map<string, ReplayEntry<T>>();
  #disposed = false;

  /** Adopt the loom-captured envelope view directly (no fetch). */
  adopt(data: T, source: string | null, pins: Pins | null = null): void {
    this.#selectedKey = null;
    this.#base = null;
    this.data = data;
    this.origin = "captured";
    this.source = source;
    this.pins = pins;
    this.loading = false;
    this.error = null;
    this.progress = null;
  }

  /** Drop the current view without cancelling or forgetting other token jobs. */
  clear(): void {
    this.#selectedKey = null;
    this.#base = null;
    this.data = null;
    this.origin = null;
    this.source = null;
    this.pins = null;
    this.loading = false;
    this.error = null;
    this.progress = null;
  }

  dispose(): void {
    this.#disposed = true;
    this.clear();
    this.#entries.clear();
  }

  /** Fetch the family's token-readout replay and map the returned
   *  measurements envelope into the tab's render shape.  ``map`` may
   *  return a null source; the caller-supplied ``fallbackSource`` covers
   *  bindings from older servers. ``base`` is the saved capture the replay
   *  completes (a row's summary without its layers): it shows, as captured,
   *  until the replay is ready, and stays beside the error if it fails. */
  replay(
    family: InstrumentFamily,
    nodeId: string,
    rawIndex: number,
    opts: { topK?: number; steered?: boolean; raw?: boolean; layers?: string },
    map: (env: MeasurementsEnvelopeJSON) => ReadoutView<T>,
    base: ReadoutView<T> | null = null,
  ): void {
    const key = replayKey(family, nodeId, rawIndex, opts);
    this.#base = base;
    this.#start(key, rawIndex, (onProgress) => cachedTokenReadout(family, nodeId, rawIndex, opts, onProgress), map);
  }

  /** One token's anatomy (the node replay), queued and kept as the family replays are. */
  anatomy(
    nodeId: string,
    rawIndex: number,
    opts: TokenAnatomyOptions,
    map: (env: MeasurementsEnvelopeJSON) => ReadoutView<T>,
  ): void {
    const key = JSON.stringify([
      "anatomy", tokenReadoutCacheEpoch("lens"), nodeId, rawIndex,
      opts.k ?? null, opts.units ?? null, opts.patterns ?? null, opts.steered ?? true, opts.raw ?? false,
    ]);
    this.#base = null;
    this.#start(key, rawIndex, (onProgress) => cachedTokenAnatomy(nodeId, rawIndex, opts, onProgress), map);
  }

  #start(
    key: string,
    rawIndex: number,
    request: (onProgress: (event: RuntimeProgressEvent) => void) => Promise<{ measurements: MeasurementsEnvelopeJSON }>,
    map: (env: MeasurementsEnvelopeJSON) => ReadoutView<T>,
  ): void {
    this.#selectedKey = key;
    const cached = this.#entries.get(key);
    if (cached && cached.state !== "failed") {
      this.#entries.delete(key);
      this.#entries.set(key, cached);
      this.#show(key, cached);
      return;
    }
    const entry: ReplayEntry<T> = {
      state: "loading",
      data: null,
      source: null,
      pins: null,
      error: null,
      progress: {
        phase: "queued",
        completed: 0,
        total: rawIndex + 1,
        progress: 0,
        message: "Waiting for the current model task to finish",
      },
    };
    this.#entries.set(key, entry);
    this.#show(key, entry);
    request((event) => {
        const progress = readoutProgress(event);
        if (progress === null) return;
        entry.progress = progress;
        if (this.#selectedKey === key) this.#show(key, entry);
      })
      .then((res) => {
        const { data, source, pins } = map(res.measurements);
        entry.state = "ready";
        entry.data = data;
        entry.source = source;
        entry.pins = pins ?? null;
        entry.error = null;
        entry.progress = null;
      })
      .catch((e) => {
        entry.state = "failed";
        entry.error = describeError(e);
        entry.progress = null;
      })
      .finally(() => {
        this.#trimCache();
        if (this.#selectedKey === key) this.#show(key, entry);
      });
  }

  #show(key: string, entry: ReplayEntry<T>): void {
    if (this.#disposed || this.#selectedKey !== key) return;
    const base = entry.state === "ready" ? null : this.#base;
    this.loading = entry.state === "loading";
    this.data = base ? base.data : entry.data;
    this.error = entry.error;
    this.origin = entry.state === "ready" ? "replayed" : base ? "captured" : null;
    this.source = base ? base.source : entry.source;
    this.pins = base ? base.pins ?? null : entry.pins;
    this.progress = entry.progress;
  }

  #trimCache(): void {
    if (this.#entries.size <= MAX_REPLAY_CACHE_ENTRIES) return;
    for (const [key, entry] of this.#entries) {
      if (this.#entries.size <= MAX_REPLAY_CACHE_ENTRIES) break;
      if (entry.state === "loading" || key === this.#selectedKey) continue;
      this.#entries.delete(key);
    }
  }
}

function replayKey(
  family: InstrumentFamily,
  nodeId: string,
  rawIndex: number,
  opts: { topK?: number; steered?: boolean; raw?: boolean; layers?: string },
): string {
  return JSON.stringify([
    family,
    tokenReadoutCacheEpoch(family),
    nodeId,
    rawIndex,
    opts.topK ?? null,
    opts.steered ?? true,
    opts.raw ?? false,
    opts.layers ?? null,
  ]);
}

function readoutProgress(event: RuntimeProgressEvent): ReadoutProgress | null {
  if (event.event !== "progress" || event.data === null || typeof event.data !== "object") {
    return null;
  }
  const value = event.data as Record<string, unknown>;
  if (
    value.kind !== "token_readout" ||
    !["queued", "context", "readout", "complete"].includes(String(value.phase)) ||
    !Number.isInteger(value.completed) ||
    !Number.isInteger(value.total) ||
    typeof value.progress !== "number" ||
    !Number.isFinite(value.progress) ||
    typeof value.message !== "string"
  ) {
    return null;
  }
  return {
    phase: value.phase as ReadoutProgress["phase"],
    completed: Math.max(0, Number(value.completed)),
    total: Math.max(0, Number(value.total)),
    progress: Math.max(0, Math.min(1, value.progress)),
    message: value.message,
  };
}
