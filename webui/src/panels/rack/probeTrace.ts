// A probe's reading token by token, from the latest generated reply's token rows (the same
// per-token scores the reply is tinted by), so the trace covers the whole reply and survives a
// reload. While a reply streams, its rows grow and so does the trace.
import { chatLog } from "../../lib/stores/chat.svelte";
import { probeScoreForTarget } from "../../lib/tokens";
import type { ChatTurn } from "../../lib/types";

/** The readings of ``target`` over the turn's response tokens; null where a token has none. */
export function turnTrace(turn: Pick<ChatTurn, "tokens"> | undefined, target: string): (number | null)[] {
  return (turn?.tokens ?? []).map((token) => {
    const v = probeScoreForTarget(token, target);
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  });
}

/** The latest generated reply's trace for ``target``, or null when that reply has no reading of it. */
export function replyTrace(target: string): (number | null)[] | null {
  for (let i = chatLog.turns.length - 1; i >= 0; i -= 1) {
    const turn = chatLog.turns[i];
    if (!turn.generated || !(turn.tokens?.length)) continue;
    const trace = turnTrace(turn, target);
    return trace.some((v) => v !== null) ? trace : null;
  }
  return null;
}

/** ``target``'s reading at each layer for one token of the latest generated reply (the last token
 *  by default), or null when that token kept no per-layer readings of it. */
export function replyLayers(target: string, index: number | null = null): Record<string, number> | null {
  for (let i = chatLog.turns.length - 1; i >= 0; i -= 1) {
    const turn = chatLog.turns[i];
    const tokens = turn.tokens ?? [];
    if (!turn.generated || tokens.length === 0) continue;
    const token = tokens[index === null ? tokens.length - 1 : Math.max(0, Math.min(tokens.length - 1, index))];
    const out: Record<string, number> = {};
    for (const [layer, scores] of Object.entries(token?.perLayerScores ?? {})) {
      const v = scores?.[target];
      if (typeof v === "number" && Number.isFinite(v)) out[layer] = v;
    }
    return Object.keys(out).length > 0 ? out : null;
  }
  return null;
}
