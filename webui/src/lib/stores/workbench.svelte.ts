// The workbench's own presentation state: what the Page region shows (the conversation, or the loom
// one step away from the thread), whether the phone's inspector sheet is up while nothing is
// selected, and how far it is pulled.

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
