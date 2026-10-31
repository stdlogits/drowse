// What the workbench's Inspector shows: whatever is selected. Nothing selected shows the
// instruments; a token its drilldown; a probe its trace and geometry; a steering term its card;
// a unit (a neuron or a head, ``{ kind: "unit", id: "neuron/L8n344" }``) its view.
//
// The selection is presentation state only. Opening the token drilldown or the probe inspector
// (``openDrawer("token_drilldown" | "probe_inspector", params)``) selects instead of opening a
// dialog, so every older entry point lands in the Inspector.

export type Selection =
  | { kind: "none" }
  | { kind: "token"; params: unknown }
  | { kind: "probe"; params: unknown }
  | { kind: "term"; name: string }
  | { kind: "unit"; id: string };

export type SelectionKind = Selection["kind"];

export const selection: { current: Selection } = $state({ current: { kind: "none" } });

export function select(next: Selection): void {
  selection.current = next;
}

export function clearSelection(): void {
  selection.current = { kind: "none" };
}

/** The selection's params when it is of ``kind``, else null. */
export function selectedParams(kind: "token" | "probe"): unknown {
  const current = selection.current;
  return current.kind === kind ? current.params : null;
}
