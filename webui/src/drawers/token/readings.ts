// The token view's reading rules beside the layer math: which pinned probes show with a readout,
// and the note a probe row keeps beside its name. Pure functions, so the view only draws.

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

const plain = (text: string) => text.trim().toLowerCase().replace(/[\s_-]+/g, "_");

/** A probe row's note, unless it only repeats the probe's name. A two-node concept's first axis
 *  takes its first pole's name ("confident" beside confident.uncertain), which the name already
 *  says; a curved probe's nearest node is a reading and stays. */
export function probeNote(name: string, note: string): string {
  if (!note) return "";
  const bare = name.slice(name.lastIndexOf("/") + 1);
  return plain(polesOf(bare).positive) === plain(note) ? "" : note;
}
