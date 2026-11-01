// The command palette's index (⌘K, the header's search): the rooms, the ways to steer ("+ Steer"
// on each kind, neurons and heads first), the instruments, every room's sections (findable by what
// they do: "fit a lens", "fetch the unit catalog", "build a probe"), then every other tool in the
// drawer registry.
//
// The tool categories are NOT declared here: they are derived from the drawer registry
// (``drawers/index.ts``), the one place a drawer's component, sizing and reachability are written
// down. A tool a room hosts is listed once, as its room's section (``ROOM_SECTIONS``), and opening
// it goes to that section (lib/stores/rooms.svelte.ts).

import { tick } from "svelte";
import { RAIL_CATEGORIES } from "../drawers";
import { ROOMS, goto, roomFor, shownRooms, shownSections, type Room } from "./stores/rooms.svelte";
import { unitsAvailable } from "./stores/units.svelte";
import type { DrawerName } from "./types";
import { setInspectorTab, type InspectorTab } from "./stores.svelte";
import { clearSelection } from "./stores/selection.svelte";
import { openSteerPicker, showConversation, type SteerKind } from "./stores/workbench.svelte";

export type { RailCategory, RailTool } from "../drawers";
export { RAIL_CATEGORIES } from "../drawers";

export type PaletteAction =
  | { kind: "room"; room: Room }
  | { kind: "section"; room: Room; section: string }
  | { kind: "drawer"; drawer: DrawerName }
  | { kind: "instrument"; tab: InspectorTab | "units" }
  | { kind: "steer"; with: SteerKind };

export interface PaletteCommand {
  label: string;
  group: string;
  action: PaletteAction;
  keywords?: string;
}

const ROOM_KEYWORDS: Record<Room, string> = {
  workbench: "home conversation chat thread page inspector steer steering",
  units: "neurons heads atlas activations read token by token",
  lens: "jacobian j-lens r-lens s-lens layer by layer predictions read token by token",
  probes: "readings vectors concepts scales manifolds read token by token",
  artifacts: "setup fit fetch install build download catalog lenses saes packs templates roles",
  model: "health device runtime api access key layers neurons heads shape",
};

/** "+ Steer"'s kinds in the palette, in the order of priority (a runtime that reads no units
 *  leaves the first out, as the picker does). */
const STEER_COMMANDS: readonly { with: SteerKind; label: string; keywords: string }[] = [
  { with: "unit", label: "Steer a neuron or head", keywords: "push silence ablate steer steering neuron head unit" },
  { with: "lens", label: "Steer toward a lens token", keywords: "push steer steering lens word token jlens slens say read" },
  { with: "concept", label: "Steer with a concept", keywords: "push steer steering concept vector pole flat" },
  { with: "position", label: "Steer to a manifold position", keywords: "push steer steering manifold position node curved" },
  { with: "feature", label: "Steer with an SAE feature", keywords: "push steer steering sae feature sparse autoencoder" },
];

/** The flattened index: the rooms first, then the ways to steer (neurons and heads first), then
 *  the instruments (in the dashboard's order of priority, the Inspector's: the unit probes where
 *  units read, the lens, the probes, the SAE's features), then the rooms' sections, then every
 *  other registry tool. */
export function paletteCommands(): PaletteCommand[] {
  const cmds: PaletteCommand[] = shownRooms().map(room => ({
    label: room.label,
    group: "Rooms",
    action: { kind: "room", room: room.value },
    keywords: ROOM_KEYWORDS[room.value],
  }));
  for (const steer of STEER_COMMANDS) {
    if (steer.with === "unit" && !unitsAvailable()) continue;
    cmds.push({ label: steer.label, group: "Steer", action: { kind: "steer", with: steer.with }, keywords: steer.keywords });
  }
  if (unitsAvailable()) {
    cmds.push({ label: "Unit probes", group: "Instruments", action: { kind: "instrument", tab: "units" }, keywords: "neurons heads units probed readings" });
  }
  cmds.push(
    { label: "Live lens readout", group: "Instruments", action: { kind: "instrument", tab: "lens" }, keywords: "layer predictions jacobian jlens workspace token" },
    { label: "Probes", group: "Instruments", action: { kind: "instrument", tab: "subspace" }, keywords: "concepts readings flat affine subspace curved manifold moods scales" },
    { label: "Features", group: "Instruments", action: { kind: "instrument", tab: "sae" }, keywords: "sparse autoencoder sae features" },
  );
  for (const room of ROOMS) {
    for (const section of shownSections(room.value)) {
      cmds.push({
        label: section.does,
        group: room.label,
        action: { kind: "section", room: room.value, section: section.value },
        keywords: `${section.label} ${section.keywords}`,
      });
    }
  }
  for (const cat of RAIL_CATEGORIES) {
    for (const tool of cat.tools) {
      if (roomFor(tool.drawer)) continue;
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

/** The commands a query finds, best first: every word of the query in the label, group or keywords,
 *  ranked by the label (the query itself, its start, inside it), then the keywords holding the
 *  whole query, then the label holding every word. */
export function searchCommands(commands: readonly PaletteCommand[], query: string): PaletteCommand[] {
  const q = query.trim().toLowerCase();
  if (!q) return [...commands];
  const terms = q.split(/\s+/);
  const rank = (command: PaletteCommand): number => {
    const label = command.label.toLowerCase();
    if (label === q) return 0;
    if (label.startsWith(q)) return 1;
    if (label.includes(q)) return 2;
    if ((command.keywords ?? "").toLowerCase().includes(q)) return 3;
    if (terms.every(term => label.includes(term))) return 4;
    return 5;
  };
  return commands
    .filter(command => terms.every(term => `${command.label} ${command.group} ${command.keywords ?? ""}`.toLowerCase().includes(term)))
    .sort((a, b) => rank(a) - rank(b));
}

/** Opens "+ Steer" on ``kind`` in the workbench. */
export function showSteerPicker(kind: SteerKind): void {
  goto("workbench");
  openSteerPicker(kind);
}

/** Shows the workbench's instruments at one of them (the Inspector with nothing selected). */
export function showInstrument(tab: InspectorTab | "units"): void {
  const id = tab === "units" ? "instrument-units" : tab === "lens" ? "instrument-lens" : tab === "sae" ? "instrument-sae" : "instrument-probes";
  if (tab !== "units") setInspectorTab(tab);
  clearSelection();
  goto("workbench");
  showConversation();
  void tick().then(() => document.getElementById(id)?.scrollIntoView({ block: "start", behavior: "smooth" }));
}
