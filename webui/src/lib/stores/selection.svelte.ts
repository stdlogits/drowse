// What the workbench's Inspector shows: whatever is selected. Nothing selected shows the
// instruments; a token its drilldown; a probe its trace and geometry; a unit (a neuron or a head,
// ``{ kind: "unit", id: "neuron/L8n344" }``) its view. The steering term whose editor is open
// under the recipe's chips (``termOpen``) is apart from the selection: opening one (a chip, or a
// Push or Silence on an instrument's row) keeps the view the Inspector shows.
//
// The selection is presentation state only. Opening the token drilldown or the probe inspector
// (``openDrawer("token_drilldown" | "probe_inspector", params)``) selects instead of opening a
// dialog, so every older entry point lands in the Inspector.

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
 *  cleared token's turn; null: the latest reply). */
export const tokenRest: { params: unknown; turnIdx: number | null } = $state({ params: null, turnIdx: null });

/** Whether the latest selection change was a step of the token cursor (the Inspector keeps its
 *  scroll for a step; a fresh selection opens at the top). */
export const selectionStep: { stepped: boolean } = $state({ stepped: false });

export function select(next: Selection, options: { step?: boolean } = {}): void {
  selectionStep.stepped = options.step === true;
  selection.current = next;
  if (next.kind === "token") tokenRest.params = next.params;
}

export function clearSelection(): void {
  const current = selection.current;
  if (current.kind === "token") {
    const turn = (current.params as { turnIdx?: unknown } | null)?.turnIdx;
    tokenRest.params = null;
    tokenRest.turnIdx = typeof turn === "number" ? turn : null;
  }
  selectionStep.stepped = false;
  selection.current = { kind: "none" };
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
