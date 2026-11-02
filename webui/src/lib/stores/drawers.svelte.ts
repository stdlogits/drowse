import type { DrawerName, DrawerState } from "../types";
import { drawerAvailability } from "../runtime/ui-capabilities";
import { pushToast } from "./toasts.svelte";
import { clearSelection, select, selection } from "./selection.svelte";
import { goto, roomFor, route } from "./rooms.svelte";

// One entry point for every tool: ``openDrawer(name, params)``. Where a tool opens depends on what
// it is. The token drilldown and the probe inspector are selections, shown in the workbench's
// Inspector; the tools a room hosts (the manifold tools, the model views) go to their room; the
// rest open as a dialog over the page, which ``drawerState`` names.

export const drawerState: DrawerState = $state({
  open: null,
  params: null,
});

/** The token drilldown as the Inspector shows it: always docked there, visible while a token is
 *  selected. Kept as a view over the selection for the callers that ask about it. */
export const tokenInspectorUi = {
  get docked(): boolean { return true; },
  get visible(): boolean { return selection.current.kind === "token"; },
  get params(): unknown { return selection.current.kind === "token" ? selection.current.params : null; },
};

/** Shows a token's details in the Inspector (the current token when no params are given). */
export function dockTokenDetails(params?: unknown): void {
  const current = params ?? (drawerState.open === "token_drilldown" ? drawerState.params : tokenInspectorUi.params);
  if (current) select({ kind: "token", params: current });
}

export function hideTokenDetails(): void {
  if (selection.current.kind === "token") clearSelection();
}

let opener: HTMLElement | null = null;

export function openDrawer(name: DrawerName, params: unknown = null): void {
  const availability = drawerAvailability(name);
  if (!availability.available) {
    pushToast(availability.reason ?? "This tool is unavailable.", {
      kind: "warning",
    });
    return;
  }
  if (name === "token_drilldown") {
    if (route.room !== "workbench") goto("workbench");
    select({ kind: "token", params });
    return;
  }
  if (name === "probe_inspector") {
    if (route.room !== "workbench") goto("workbench");
    select({ kind: "probe", params });
    return;
  }
  const home = roomFor(name);
  if (home) {
    if (drawerState.open !== null) closeDrawer();
    goto(home.room, home.section, params);
    return;
  }
  if (
    drawerState.open === null &&
    typeof document !== "undefined" &&
    document.activeElement instanceof HTMLElement
  ) {
    opener = document.activeElement;
  }
  drawerState.open = name;
  drawerState.params = params;
}

/** Closes the open dialog; with none open, a room's hosted tool closing returns to the workbench. */
export function closeDrawer(): void {
  if (drawerState.open === null) {
    if (route.room !== "workbench") goto("workbench");
    return;
  }
  const restore = opener;
  opener = null;
  drawerState.open = null;
  drawerState.params = null;
  queueMicrotask(() => {
    if (restore?.isConnected) restore.focus();
  });
}
