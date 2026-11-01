// The workbench's own presentation state: what the Page region shows (the conversation, or the loom
// one step away from the thread), whether the phone's inspector sheet is up while nothing is
// selected, and how far it is pulled; and "+ Steer": the kind its picker opens on (the last one
// chosen) and each request to open it (the command palette's Steer entries).

export type WorkbenchPage = "conversation" | "loom";

export const workbenchUi: { page: WorkbenchPage; sheet: boolean; sheetOpen: boolean } = $state({
  page: "conversation",
  sheet: false,
  sheetOpen: false,
});

export function showLoom(): void {
  workbenchUi.page = "loom";
}

export function showConversation(): void {
  workbenchUi.page = "conversation";
}

/** What "+ Steer" adds, in the order of priority: a neuron or head, a lens token, a concept, a
 *  manifold position, an SAE feature. */
export type SteerKind = "unit" | "lens" | "concept" | "position" | "feature";

/** The kind "+ Steer" opens on (this page's memory of the last one chosen), and a count of the
 *  requests to open it, which RecipeBar follows. */
export const steerPicker: { kind: SteerKind; requested: number } = $state({ kind: "unit", requested: 0 });

/** Open "+ Steer" on ``kind`` (the workbench's conversation, where the recipe is). */
export function openSteerPicker(kind: SteerKind): void {
  showConversation();
  steerPicker.kind = kind;
  steerPicker.requested += 1;
}
