// What the window's Escape key does, in order. One press does one thing, the nearest first: the
// palette, then a drawer, then what is open in the workbench (the term editor under the chips, the
// inspector's selection), and only then the streaming reply. Closing the term editor mid-reply must
// not stop the reply; a second Escape does. A loom modal handles its own Escape.

export type EscapeAction = "palette" | "modal" | "drawer" | "selection" | "stop" | "none";

export interface EscapeState {
  paletteOpen: boolean;
  modalOpen: boolean;
  drawerOpen: boolean;
  /** The workbench is showing and something in it is selected (a term, a token, a probe). */
  selectionOpen: boolean;
  generating: boolean;
}

export function escapeAction(state: EscapeState): EscapeAction {
  if (state.paletteOpen) return "palette";
  if (state.modalOpen) return "modal";
  if (state.drawerOpen) return "drawer";
  if (state.selectionOpen) return "selection";
  if (state.generating) return "stop";
  return "none";
}
