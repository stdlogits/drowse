// The token cursor: one position over the whole conversation's tokens, or the whole reply. The
// Units, Lens and Probes rooms read at it (TokenStrip draws and moves it) and the workbench's token
// view is it. The walk is drawers/token/cursor.ts's: every turn's thinking, then response tokens,
// prompt tokens included (an authored turn's captured rows replay through its prompt capture), and
// a step rolls across segments and turns.
//
// It follows the workbench's token selection: a selected token is the cursor; while a probe, unit
// or term is selected it rests on the last token selected; once a token selection is cleared it
// reads the whole reply (selection.svelte.ts keeps where it rests). Moving it selects the token it
// lands on, as a step, so the thread marks it and the Inspector shows it without jumping to the top.

import type { ChatTurn, TokenScore } from "../types";
import { chatLog } from "./chat.svelte";
import { loomTree } from "./loom.svelte";
import { clearSelection, select, selection, tokenRest } from "./selection.svelte";
import { replayIndex } from "../tokenRows";
import {
  clampCursor,
  segmentTokens,
  segmentsOf,
  stepCursor,
  type SegmentRef,
  type TokenCursor,
} from "../../drawers/token/cursor";

export type { TokenCursor } from "../../drawers/token/cursor";

/** The token under the cursor, with where it replays. */
export interface CursorToken {
  at: TokenCursor;
  turn: ChatTurn;
  token: TokenScore;
  /** The loom node of the token's turn, when it has one. */
  nodeId: string | null;
  /** Where the token replays: a generated token's raw decode index, a prompt token's captured row
   *  (``replayIndex``); null for a token with neither. */
  index: number | null;
  /** The token's place in the whole walk, from 0. */
  ordinal: number;
}

/** The reply the whole view reads. */
export interface WholeReply {
  turnIdx: number;
  turn: ChatTurn;
  nodeId: string | null;
}

/** A selection's token params as a cursor (``{ turnIdx, tokenIdx, isThinking? }``). */
export function paramsCursor(params: unknown): TokenCursor | null {
  const p = params as { turnIdx?: unknown; tokenIdx?: unknown; isThinking?: unknown } | null;
  if (!p || typeof p.turnIdx !== "number" || typeof p.tokenIdx !== "number") return null;
  return { turnIdx: p.turnIdx, seg: p.isThinking ? "thinking" : "response", tokenIdx: p.tokenIdx };
}

/** A cursor as a selection's token params. */
export function cursorParams(at: TokenCursor): { turnIdx: number; tokenIdx: number; isThinking: boolean } {
  return { turnIdx: at.turnIdx, tokenIdx: at.tokenIdx, isThinking: at.seg === "thinking" };
}

/** The loom node behind a turn of the conversation: its own, else the active path's node at that
 *  place (the root system node, when empty, is not a turn). */
export function turnNodeId(turnIdx: number): string | null {
  const turn = chatLog.turns[turnIdx];
  if (turn?.nodeId) return turn.nodeId;
  if (turnIdx < 0 || loomTree.activePath.length === 0) return null;
  const visible = loomTree.activePath
    .map((id) => loomTree.nodes.get(id))
    .filter((node) => node && !(node.parent_id === null && node.role === "system" && !node.text));
  return visible[turnIdx]?.id ?? null;
}

/** Whether a turn is a reply the whole view can read: generated, with tokens. */
function isReply(turn: ChatTurn | undefined): boolean {
  return turn !== undefined && turn.generated === true && ((turn.tokens?.length ?? 0) + (turn.thinkingTokens?.length ?? 0)) > 0;
}

/** The reply a turn's whole view reads: the turn itself when it is one, else the next reply after
 *  it (the one that read its prompt), else the latest; with no turn, the latest. */
export function replyTurnFor(turnIdx: number | null): number | null {
  const turns = chatLog.turns;
  if (turnIdx !== null) {
    for (let i = Math.max(0, turnIdx); i < turns.length; i += 1) if (isReply(turns[i])) return i;
  }
  for (let i = turns.length - 1; i >= 0; i -= 1) if (isReply(turns[i])) return i;
  return null;
}

/** Where ``at`` sits in the walk (its ordinal), given the segments. */
function ordinalOf(segments: readonly SegmentRef[], at: TokenCursor): number {
  let n = 0;
  for (const s of segments) {
    if (s.turnIdx === at.turnIdx && s.seg === at.seg) return n + at.tokenIdx;
    n += s.length;
  }
  return n;
}

class TokenCursorStore {
  /** The walk: every turn's inspectable segments, in order. */
  readonly segments = $derived(segmentsOf(chatLog.turns));
  readonly total = $derived(this.segments.reduce((n, s) => n + s.length, 0));
  /** The position asked for: the selected token, else where the cursor rests (null: the whole reply). */
  readonly requested = $derived.by((): TokenCursor | null => {
    const current = selection.current;
    return paramsCursor(current.kind === "token" ? current.params : tokenRest.params);
  });
  /** The position, clamped onto the walk (token lists grow while streaming and change with a
   *  branch); null reads the whole reply. */
  readonly at = $derived(this.requested ? clampCursor(this.segments, this.requested) : null);
  readonly token = $derived.by((): CursorToken | null => {
    const at = this.at;
    if (!at) return null;
    const turn = chatLog.turns[at.turnIdx];
    const token = segmentTokens(turn, at.seg)[at.tokenIdx];
    if (!turn || !token) return null;
    return {
      at,
      turn,
      token,
      nodeId: turnNodeId(at.turnIdx),
      index: replayIndex(token),
      ordinal: ordinalOf(this.segments, at),
    };
  });
  /** The reply the whole view reads: the cursor's turn's (or, whole, the cleared token's turn's). */
  readonly reply = $derived.by((): WholeReply | null => {
    const from = this.at ? this.at.turnIdx : tokenRest.turnIdx;
    const turnIdx = replyTurnFor(from);
    if (turnIdx === null) return null;
    return { turnIdx, turn: chatLog.turns[turnIdx], nodeId: turnNodeId(turnIdx) };
  });
}

export const tokenCursor = new TokenCursorStore();

/** Moves the cursor to ``next`` (a step: it selects that token). */
export function moveCursor(next: TokenCursor | null): void {
  if (!next) return;
  select({ kind: "token", params: cursorParams(next) }, { step: true });
}

/** One token back or on; from the whole reply, its last or first token. */
export function stepToken(delta: 1 | -1): void {
  const at = tokenCursor.at;
  if (at) {
    moveCursor(stepCursor(tokenCursor.segments, at, delta));
    return;
  }
  const reply = tokenCursor.reply;
  if (!reply) return;
  const own = tokenCursor.segments.filter((s) => s.turnIdx === reply.turnIdx);
  const edge = delta > 0 ? own[0] : own.at(-1);
  if (edge) moveCursor({ turnIdx: edge.turnIdx, seg: edge.seg, tokenIdx: delta > 0 ? 0 : edge.length - 1 });
}

/** Whether a step that way lands anywhere. */
export function canStep(delta: 1 | -1): boolean {
  const at = tokenCursor.at;
  if (!at) return tokenCursor.reply !== null;
  return stepCursor(tokenCursor.segments, at, delta) !== null;
}

/** The first or last token of the cursor's segment (a turn's thinking or its response). */
export function segmentEdge(where: "first" | "last"): void {
  const at = tokenCursor.at;
  if (!at) {
    stepToken(where === "first" ? 1 : -1);
    return;
  }
  const seg = tokenCursor.segments.find((s) => s.turnIdx === at.turnIdx && s.seg === at.seg);
  if (seg) moveCursor({ ...at, tokenIdx: where === "first" ? 0 : seg.length - 1 });
}

/** ``read()`` once it has held still for ``ms``: a room asks for a replay where the cursor rests,
 *  not for every token a held arrow key passes. Call it while a component initialises. */
export function settled<T>(read: () => T, ms = 140): { readonly current: T } {
  let value = $state.raw(read());
  $effect(() => {
    const next = read();
    const timer = setTimeout(() => { value = next; }, ms);
    return () => clearTimeout(timer);
  });
  return { get current() { return value; } };
}

/** The cursor position of the token of turn ``turnIdx`` that replays at ``index`` (a unit trace's
 *  or a probe trace's place), or null. */
export function cursorAtIndex(turnIdx: number, index: number): TokenCursor | null {
  const turn = chatLog.turns[turnIdx];
  for (const seg of ["thinking", "response"] as const) {
    const tokenIdx = segmentTokens(turn, seg).findIndex((token) => replayIndex(token) === index);
    if (tokenIdx >= 0) return { turnIdx, seg, tokenIdx };
  }
  return null;
}

/** The whole reply instead of a token: a selected token is deselected; otherwise the cursor stops
 *  resting on one. */
export function showWholeReply(): void {
  if (selection.current.kind === "token") {
    clearSelection();
    return;
  }
  const at = tokenCursor.at;
  tokenRest.params = null;
  tokenRest.turnIdx = at ? at.turnIdx : tokenRest.turnIdx;
}
