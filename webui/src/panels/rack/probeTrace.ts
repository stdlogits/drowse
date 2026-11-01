// A probe's reading token by token, from the latest generated reply's token rows, so the trace
// covers the whole reply and survives a reload. While a reply streams, its rows grow and so does
// the trace.
//
// A probe reads one primary scalar (the store's sparkline and "reads" value): a flat probe its
// signed axis-0 coordinate, a lens or SAE probe its one channel, and a curved probe the share of
// the activation on its surface. The rows' flat ``probes`` view carries axis 0 for every probe,
// so a curved probe's share comes from the token's geometry reading instead (``share``); a row
// without that reading (Live was off) leaves a gap rather than an axis.
import { chatLog } from "../../lib/stores/chat.svelte";
import { mergedReadings } from "../../lib/stores/instruments.svelte";
import { readingLayers } from "../../lib/tokenRows";
import { parseProbeTarget, probeScoreForTarget } from "../../lib/tokens";
import type { ChatTurn, ProbeInfo, TokenScore } from "../../lib/types";

type TokenRow = Pick<TokenScore, "probes" | "coordsByProbe" | "measurements">;

export interface TraceOptions {
  /** Read the share on the probe's surface (a curved probe's primary reading), not a coordinate. */
  share?: boolean;
}

/** True when ``info``'s primary reading is the share on its surface (a curved geometry probe). */
export function readsShare(info: ProbeInfo | null | undefined): boolean {
  return info?.family === "geometry" && !info.is_affine;
}

/** The share applies to the probe as a whole; an axis target (``months[3]``) reads its coordinate. */
function shareBase(target: string, share: boolean): string | null {
  if (!share) return null;
  const { base } = parseProbeTarget(target);
  return base === target ? base : null;
}

const finite = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** One token's reading of ``target``, or null when the token has none. */
export function tokenReading(token: TokenRow | undefined, target: string, { share = false }: TraceOptions = {}): number | null {
  if (!token) return null;
  const base = shareBase(target, share);
  if (base !== null) return finite(token.measurements?.instruments?.geometry?.readings?.[base]?.fraction);
  return finite(probeScoreForTarget(token, target));
}

/** The readings of ``target`` over the turn's response tokens; null where a token has none. */
export function turnTrace(turn: Pick<ChatTurn, "tokens"> | undefined, target: string, options: TraceOptions = {}): (number | null)[] {
  return (turn?.tokens ?? []).map((token) => tokenReading(token, target, options));
}

/** The latest generated reply's trace for ``target``, or null when that reply has no reading of it. */
export function replyTrace(target: string, options: TraceOptions = {}): (number | null)[] | null {
  for (let i = chatLog.turns.length - 1; i >= 0; i -= 1) {
    const turn = chatLog.turns[i];
    if (!turn.generated || !(turn.tokens?.length)) continue;
    const trace = turnTrace(turn, target, options);
    return trace.some((v) => v !== null) ? trace : null;
  }
  return null;
}

/** One token of the latest generated reply (the last by default): the token, and the loom anchor
 *  (node id and raw index) a replay of it needs, null where the reply has none. */
export function replyToken(index: number | null = null): { token: TokenScore; nodeId: string | null; rawIndex: number | null } | null {
  for (let i = chatLog.turns.length - 1; i >= 0; i -= 1) {
    const turn = chatLog.turns[i];
    const tokens = turn.tokens ?? [];
    if (!turn.generated || tokens.length === 0) continue;
    const token = tokens[index === null ? tokens.length - 1 : Math.max(0, Math.min(tokens.length - 1, index))];
    return { token, nodeId: turn.nodeId ?? null, rawIndex: token.rawIndex ?? null };
  }
  return null;
}

/** ``target``'s reading at each layer for one token of the latest generated reply (the last token
 *  by default), from the token's own reading, or null when that token kept no per-layer trace of
 *  it. A saved reply's geometry readings keep their summary only: their layers replay the token
 *  (``replyToken`` gives its anchor). */
export function replyLayers(target: string, index: number | null = null, { share = false }: TraceOptions = {}): Record<string, number> | null {
  const picked = replyToken(index);
  if (!picked) return null;
  const { base } = parseProbeTarget(target);
  const layers = readingLayers(mergedReadings(picked.token.measurements)?.[base], shareBase(target, share) !== null);
  return layers !== null && Object.keys(layers).length > 0 ? layers : null;
}
