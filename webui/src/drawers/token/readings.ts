// The token view's reading rules beside the layer math: which pinned probes show with a readout,
// when the lens side can replay, and the note a probe row keeps beside its name. Pure functions, so
// the view only draws.

import { polesOf } from "../../lib/concepts";
import type { MeasurementInstrumentsJSON, ScalarReadingJSON } from "../../lib/types";
import type { ReadoutOrigin } from "./readout.svelte";

/** The lens and SAE probes pinned while the token was generated. They belong to that capture, so
 *  each family's pins show only while its readout is the captured one: a replay reads the
 *  discovery readout, not the pinned roster, and must not pass the old pins off as its own. */
export function pinnedReadings(
  instruments: MeasurementInstrumentsJSON | null | undefined,
  origins: { lens: ReadoutOrigin; sae: ReadoutOrigin },
): Record<string, ScalarReadingJSON> {
  return {
    ...(origins.lens === "captured" ? (instruments?.lens?.readings ?? {}) : {}),
    ...(origins.sae === "captured" ? (instruments?.sae?.readings ?? {}) : {}),
  };
}

/** The origin the lens pins go by. The lens readout's own, where it shows; the anatomy is always a
 *  replay, so where it stands in for the readout the pins follow the steering switch: steered, the
 *  replay reads the generation the pins were captured in; unsteered, they would misstate it. */
export function lensPinOrigin(anatomy: boolean, steered: boolean, readout: ReadoutOrigin): ReadoutOrigin {
  if (!anatomy) return readout;
  return steered ? "captured" : null;
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
