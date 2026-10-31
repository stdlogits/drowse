// The command palette's index (⌘K, the header's search): the rooms, the instruments, then every
// tool in the drawer registry.
//
// The tool categories are NOT declared here: they are derived from the drawer registry
// (``drawers/index.ts``), the one place a drawer's component, sizing and reachability are written
// down. A tool a room hosts opens its room (``openDrawer`` routes it, lib/stores/rooms.svelte.ts).

import { RAIL_CATEGORIES } from "../drawers";
import { ROOMS, type Room } from "./stores/rooms.svelte";
import type { DrawerName } from "./types";
import type { InspectorTab } from "./stores.svelte";

export type { RailCategory, RailTool } from "../drawers";
export { RAIL_CATEGORIES } from "../drawers";

export type PaletteAction =
  | { kind: "room"; room: Room }
  | { kind: "drawer"; drawer: DrawerName }
  | { kind: "instrument"; tab: InspectorTab };

export interface PaletteCommand {
  label: string;
  group: string;
  action: PaletteAction;
  keywords?: string;
}

const ROOM_KEYWORDS: Record<Room, string> = {
  workbench: "home conversation chat thread page inspector",
  manifolds: "concepts scales packs build combine templates roles",
  units: "neurons heads atlas",
  lens: "jacobian j-lens r-lens s-lens sources fit fetch",
  model: "health device runtime api access key",
};

/** The flattened index: the rooms first, then the instruments, then every registry tool. */
export function paletteCommands(): PaletteCommand[] {
  const cmds: PaletteCommand[] = ROOMS.map(room => ({
    label: room.label,
    group: "Rooms",
    action: { kind: "room", room: room.value },
    keywords: ROOM_KEYWORDS[room.value],
  }));
  cmds.push(
    { label: "Probes", group: "Instruments", action: { kind: "instrument", tab: "subspace" }, keywords: "concepts readings flat affine subspace curved manifold moods scales" },
    { label: "Live lens readout", group: "Instruments", action: { kind: "instrument", tab: "lens" }, keywords: "layer predictions jacobian jlens workspace token" },
    { label: "Features", group: "Instruments", action: { kind: "instrument", tab: "sae" }, keywords: "sparse autoencoder sae features" },
  );
  for (const cat of RAIL_CATEGORIES) {
    for (const tool of cat.tools) {
      cmds.push({
        label: tool.label.replace(/…$/, ""),
        group: cat.label,
        action: { kind: "drawer", drawer: tool.drawer },
        keywords: tool.keywords,
      });
    }
  }
  return cmds;
}
