import type { DiffTokenSpanJSON } from "../../lib/types";

/** One side's tokens from a per-token diff, each once and in order. The server's byte-offset walk
 *  emits a span per step, and while the two sides drift apart one side's token repeats across the
 *  steps that advance the other, so a pane keeps the first span for each of its indices. */
export function diffSideTokens(spans: readonly DiffTokenSpanJSON[], side: "a" | "b"): DiffTokenSpanJSON[] {
  const seen = new Set<number>();
  const out: DiffTokenSpanJSON[] = [];
  for (const span of spans) {
    const index = side === "a" ? span.a_index : span.b_index;
    if (index < 0 || seen.has(index)) continue;
    seen.add(index);
    out.push(span);
  }
  return out;
}
