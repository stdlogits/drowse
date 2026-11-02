// What the workbench's Inspector shows: whatever is selected. Nothing selected shows the
// instruments; a token its drilldown; a probe its trace and geometry; a unit (a neuron or a head,
// ``{ kind: "unit", id: "neuron/L8n344" }``) its view. The steering term whose editor is open
// under the recipe's chips (``termOpen``) is apart from the selection: opening one (a chip, or a
// Push or Silence on an instrument's row) keeps the view the Inspector shows.
//
// The selection is presentation state only. Opening the token drilldown or the probe inspector
// (``openDrawer("token_drilldown" | "probe_inspector", params)``) selects instead of opening a
// dialog, so every older entry point lands in the Inspector.

/** The loom node of a conversation turn, bound by lib/stores/tokenCursor.svelte.ts (which reads the
 *  chat log): this store stays free of the chat log's imports, so it loads before a runtime is
 *  installed. Unbound, no turn names a node and a token selection is never dropped. */
let turnNode: (turnIdx: number) => string | null = () => null;

export function bindTurnNodes(read: (turnIdx: number) => string | null): void {
  turnNode = read;
}

export type Selection =
  | { kind: "none" }
  | { kind: "token"; params: unknown }
  | { kind: "probe"; params: unknown }
  | { kind: "unit"; id: string };

export type SelectionKind = Selection["kind"];

export const selection: { current: Selection } = $state({ current: { kind: "none" } });

/** The steering term whose editor is open under the recipe's chips (its rack key), or null. */
export const termOpen: { name: string | null } = $state({ name: null });

export function openTermEditor(name: string): void {
  termOpen.name = name;
}

export function closeTermEditor(): void {
  termOpen.name = null;
}

/** Where the rooms' token cursor (lib/stores/tokenCursor.svelte.ts) rests while the selection holds
 *  no token: the last token selected (``params``), kept while a probe or unit is selected;
 *  ``null`` once a token selection is cleared, which reads the whole reply of ``turnIdx`` (the
 *  cleared token's turn; null: the latest reply). ``nodeId`` is the loom node of the turn the
 *  selected or resting token (or ``turnIdx``) names, recorded when it was selected: the position is
 *  only good while that turn still holds that node (``tokenTurnHolds``), so a cleared chat, another
 *  conversation or another branch never reads a token nobody selected. */
export const tokenRest: { params: unknown; turnIdx: number | null; nodeId: string | null } = $state({
  params: null, turnIdx: null, nodeId: null,
});

/** The turn a token selection's params name, or null. */
function paramsTurn(params: unknown): number | null {
  const turn = (params as { turnIdx?: unknown } | null)?.turnIdx;
  return typeof turn === "number" ? turn : null;
}

/** Whether turn ``turnIdx`` still holds the node the token selection was made on (a turn with no
 *  node when it was selected, a reply still streaming, cannot tell and holds). */
export function tokenTurnHolds(turnIdx: number): boolean {
  return tokenRest.nodeId === null || turnNode(turnIdx) === tokenRest.nodeId;
}

/** Whether the latest selection change was a step of the token cursor (the Inspector keeps its
 *  scroll for a step; a fresh selection opens at the top). */
export const selectionStep: { stepped: boolean } = $state({ stepped: false });

export function select(next: Selection, options: { step?: boolean } = {}): void {
  selectionStep.stepped = options.step === true;
  selection.current = next;
  if (next.kind === "token") {
    tokenRest.params = next.params;
    const turn = paramsTurn(next.params);
    tokenRest.nodeId = turn === null ? null : turnNode(turn);
  }
}

export function clearSelection(): void {
  const current = selection.current;
  if (current.kind === "token") {
    tokenRest.params = null;
    tokenRest.turnIdx = paramsTurn(current.params);
  }
  selectionStep.stepped = false;
  selection.current = { kind: "none" };
}

/** Drop a token selection, and where the cursor rests, once the conversation no longer holds the
 *  node they were made on (Clear chat, another conversation, another branch): the rooms then read
 *  the latest reply whole. ``syncChatLogFromTree`` runs it after each projection. */
export function reconcileTokenSelection(): void {
  if (tokenRest.nodeId === null) return;
  const current = selection.current;
  const turn = current.kind === "token" ? paramsTurn(current.params) : paramsTurn(tokenRest.params) ?? tokenRest.turnIdx;
  if (turn !== null && tokenTurnHolds(turn)) return;
  if (current.kind === "token") {
    selectionStep.stepped = false;
    selection.current = { kind: "none" };
  }
  tokenRest.params = null;
  tokenRest.turnIdx = null;
  tokenRest.nodeId = null;
}

/** What a token, probe or unit selection names (``null`` for none), so a view
 *  can tell a new thing selected from the same one re-selected. Plain-data params only. */
export function selectionIdentity(current: Selection): string | null {
  switch (current.kind) {
    case "token":
    case "probe":
      return `${current.kind}:${JSON.stringify(current.params ?? null)}`;
    case "unit":
      return `unit:${current.id}`;
    default:
      return null;
  }
}

/** The selection's params when it is of ``kind``, else null. */
export function selectedParams(kind: "token" | "probe"): unknown {
  const current = selection.current;
  return current.kind === kind ? current.params : null;
}
