// The catalog's labels for the units a token view shows, asked once per unit (both poles of a
// neuron come back together) and kept for the view's life. Labels are disk reads on the server, so
// they skip the model queue. An unbound catalog is asked again only after a while, so binding one in
// the Units room reaches an open token view.

import { SvelteMap } from "svelte/reactivity";
import { apiInstruments } from "../../lib/runtime/services";
import type { UnitLabelJSON } from "../../lib/types";
import { labelText } from "./anatomy";

const RETRY_UNBOUND_MS = 30_000;
const MAX_PER_REQUEST = 128;

const keyOf = (unit: string, pole: "+" | "-" | null) => `${unit}|${pole ?? ""}`;

export class UnitLabels {
  /** The catalog's credit while it is bound ("Labels by Claude Haiku 5.5"). */
  credit: string | null = $state(null);
  #rows = new SvelteMap<string, UnitLabelJSON | null>();
  #asked = new Set<string>();
  #unboundAt = -Infinity;

  /** The label a row shows: a neuron's by its pole (none when the pole is unknown), a head's. */
  label(unit: string, kind: "neuron" | "head", pole: "+" | "-" | null): { text: string; tip: string } | null {
    if (kind === "neuron" && pole === null) return null;
    return labelText(this.#rows.get(keyOf(unit, kind === "head" ? null : pole)), kind);
  }

  /** Ask for the units not asked yet (one request). */
  want(units: readonly { unit: string }[]): void {
    const read = apiInstruments.unitLabels;
    if (!read || Date.now() - this.#unboundAt < RETRY_UNBOUND_MS) return;
    const fresh = [...new Set(units.map(u => u.unit))].filter(unit => !this.#asked.has(unit)).slice(0, MAX_PER_REQUEST);
    if (fresh.length === 0) return;
    for (const unit of fresh) this.#asked.add(unit);
    read.call(apiInstruments, fresh)
      .then(answer => {
        if (!answer.catalog.bound) {
          this.credit = null;
          this.#unboundAt = Date.now();
          for (const unit of fresh) this.#asked.delete(unit);
          return;
        }
        this.credit = answer.catalog.credit;
        for (const row of answer.labels) this.#rows.set(keyOf(row.unit, row.pole), row.label);
      })
      .catch(() => {
        // Labels are optional: a failed read leaves the rows unlabelled and is not retried.
      });
  }
}
