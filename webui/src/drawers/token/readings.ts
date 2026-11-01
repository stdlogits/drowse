// The token view's reading rules beside the layer math: which pinned probes show with a readout,
// when the lens side can replay, and the note a probe row keeps beside its name. Pure functions, so
// the view only draws.

import { polesOf } from "../../lib/concepts";
import { withRowValues } from "../../lib/tokenRows";
import type { MeasurementsEnvelopeJSON, RowProbeReadingJSON, ScalarReadingJSON } from "../../lib/types";
import type { GeometryTokenReadout, Pins, ReadoutView } from "./readout.svelte";

/** The geometry replay's map from a replayed envelope to the view, bound to the token it was asked
 *  for: ``row`` (its generated row's share and coordinates) and ``captured`` (a live token's
 *  capture) are taken when the replay is requested, never read again when it lands, so a replay
 *  that resolves after the selection moved caches the requested token's values, not the newly
 *  selected one's. Steered, a probe detached since keeps its capture, and a generated row's share
 *  and coordinates stay as the row kept them. */
export function geometryReplayView(opts: {
  steered: boolean;
  captured: GeometryTokenReadout | null;
  row: Record<string, RowProbeReadingJSON>;
}): (m: MeasurementsEnvelopeJSON) => ReadoutView<GeometryTokenReadout> {
  const { steered, captured, row } = opts;
  return (m) => {
    const geometry = m.instruments.geometry;
    if (!geometry) throw new Error("No probe readings came back. Check that the probes are attached.");
    const kept = steered && captured ? captured.readings : {};
    const replayed = geometry.readings ?? {};
    const readings = steered && !captured ? withRowValues(replayed, row) : replayed;
    return { data: { steering: geometry.binding?.steering ?? null, readings: { ...kept, ...readings } }, source: null };
  };
}

/** The lens and SAE probes the token view lists beside its probes: each family's pins as its shown
 *  read pinned them (a capture's, or the replay's: the lens and SAE replays and the anatomy return
 *  the attached probes' readings). Where the anatomy stands in for the lens readout, a steered view
 *  keeps the pins captured with the token (a live token's; a generated row keeps none), else the
 *  anatomy's own. A steered family's pin keeps the value the token's row kept (``scores``: what the
 *  thread and the rack read), its band and depth as replayed; an unsteered one is all replay. */
export function tokenPins(opts: {
  anatomy: boolean;
  steered: boolean;
  saeSteered?: boolean;
  captured: Pins | null | undefined;
  anatomyPins: Pins | null;
  lensPins: Pins | null;
  saePins: Pins | null;
  scores?: Record<string, number> | null;
}): Record<string, ScalarReadingJSON> {
  const captured = opts.steered ? opts.captured ?? null : null;
  const lens = opts.anatomy ? captured ?? opts.anatomyPins : opts.lensPins;
  const kept = (pins: Pins | null, steered: boolean): Pins => {
    const out: Pins = {};
    for (const [name, pin] of Object.entries(pins ?? {})) {
      const value = steered ? opts.scores?.[name] : undefined;
      out[name] = typeof value === "number" && Number.isFinite(value) ? { ...pin, value } : pin;
    }
    return out;
  };
  return { ...kept(lens, opts.steered), ...kept(opts.saePins, opts.saeSteered ?? opts.steered) };
}

/** Whether the lens side replays (as the anatomy, where the runtime has it, else as the lens
 *  readout), whether a lens is ready for it, and whether the token view shows the anatomy. The
 *  anatomy reads through any active lens, an S lens alone included; the readout needs the R lens.
 *  The anatomy shows only for a token with its replay record and a lens ready; otherwise the lens
 *  readout's view stands in (the capture, or why there is none). The view and the WebMCP inspector
 *  share it. */
export function lensReplay(opts: { anatomy: boolean; tokenReadout: boolean; rLens: boolean; activeLens: boolean; replayContext: boolean }): { available: boolean; ready: boolean; anatomy: boolean } {
  const ready = opts.anatomy ? opts.rLens || opts.activeLens : opts.rLens;
  return {
    available: opts.anatomy || opts.tokenReadout,
    ready,
    anatomy: opts.anatomy && ready && opts.replayContext,
  };
}

const plain = (text: string) => text.trim().toLowerCase().replace(/[\s_-]+/g, "_");

/** A probe row's note, unless it only repeats the probe's name. A two-node concept's first axis
 *  takes its first pole's name ("confident" beside confident.uncertain), which the name already
 *  says; a curved probe's nearest node is a reading and stays. */
export function probeNote(name: string, note: string): string {
  if (!note) return "";
  const bare = name.slice(name.lastIndexOf("/") + 1);
  return plain(polesOf(bare).positive) === plain(note) ? "" : note;
}

/** What a saved row's missing layers wait on: a replay where this runtime has one and the token its
 *  replay record, else why nothing will bring them (a prompt-side row with no record keeps its own
 *  layers, so this reads only for a row that dropped them). */
export function layersNote(replayAvailable: boolean, hasReplayContext: boolean, needs: string): string {
  if (!replayAvailable) return "This runtime does not replay them.";
  if (!hasReplayContext) return "This token has no replay record to rebuild them from.";
  return `They replay while this model is loaded and ${needs}.`;
}
