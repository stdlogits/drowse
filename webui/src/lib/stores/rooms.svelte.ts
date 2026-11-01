// The dashboard's rooms, as routes. The workbench is home; Units, Lens and Probes read the
// conversation (neurons and heads, the lenses, the probes); Artifacts is where what they read with is
// fitted, fetched, built and installed; Model is the model's facts. A room is reached from the
// header's tabs (the phone's menu), the command palette or a link (``#/lens``). A room may hold
// sections (``#/artifacts/lenses``). A section is either a tool that used to open as a drawer
// (opening that drawer, ``openDrawer("manifold_pack")``, goes to its room) or a panel its room draws
// itself (``drawer: null``; the room maps the section's value to a component). Every link to setup
// goes to its Artifacts section: ``goto("artifacts", "lenses")``.
import type { DrawerName } from "../types";
import { getRuntimeClient } from "../runtime/registry";
import { unitReadsAvailable } from "../runtime/contracts";
import { drawerAvailability } from "../runtime/ui-capabilities";

export type Room = "workbench" | "units" | "lens" | "probes" | "artifacts" | "model";

export const ROOMS: readonly { value: Room; label: string }[] = [
  { value: "workbench", label: "Workbench" },
  { value: "units", label: "Units" },
  { value: "lens", label: "Lens" },
  { value: "probes", label: "Probes" },
  { value: "artifacts", label: "Artifacts" },
  { value: "model", label: "Model" },
];

/** A room's section: a hosted drawer, or (``drawer: null``) a panel its room draws by ``value``.
 *  ``does`` is what it is for, as the command palette lists it; ``keywords`` find it there. */
export interface RoomSection {
  value: string;
  label: string;
  drawer: DrawerName | null;
  does: string;
  keywords: string;
}

/** The rooms this runtime has: Units needs a runtime that reads units (``unitReadsAvailable``). */
export function shownRooms(): readonly { value: Room; label: string }[] {
  return ROOMS.filter(room => room.value !== "units" || unitReadsAvailable(getRuntimeClient()));
}

/** Each room's sections, in display order. A section's value is unique across rooms (an agent's
 *  ``section`` alone names its room). Artifacts runs in the dashboard's order of priority: neurons
 *  and heads, the lenses, the probes (built, installed, combined), then the SAEs. ``local_runtime``
 *  and ``health`` are the two runtimes' model views (browser and HTTP); a room shows the one this
 *  runtime has (``sectionShown``). */
export const ROOM_SECTIONS: Partial<Record<Room, readonly RoomSection[]>> = {
  artifacts: [
    { value: "units", label: "Units", drawer: null, does: "Fetch the unit catalog", keywords: "neuron-lens labels neurons heads catalog units" },
    { value: "lenses", label: "Lenses", drawer: null, does: "Fit or fetch a lens", keywords: "fit a lens fetch a lens jacobian j-lens r-lens s-lens sources" },
    { value: "build", label: "Build", drawer: "manifold_builder", does: "Build a probe", keywords: "create a concept or scale extract author discover vector manifold fit" },
    { value: "packs", label: "Packs", drawer: "manifold_pack", does: "Install a pack", keywords: "packs manage downloaded controls download hub huggingface catalog search installed" },
    { value: "combine", label: "Combine", drawer: "manifold_merge", does: "Combine probes", keywords: "merge union corpora concepts response controls" },
    { value: "templates", label: "Templates", drawer: "template_lab", does: "Test a prompt template", keywords: "templates score completion slot restricted choice" },
    { value: "roles", label: "Roles", drawer: "cast", does: "Set up a role", keywords: "roles cast speaker behavior guidance" },
    { value: "saes", label: "SAEs", drawer: null, does: "Fetch or train an SAE", keywords: "sparse autoencoder features saelens fetch train" },
  ],
  model: [
    { value: "health", label: "Health", drawer: "health", does: "Check the model's health", keywords: "device dtype diagnostics" },
    { value: "runtime", label: "Runtime", drawer: "local_runtime", does: "Model settings", keywords: "local device model storage gpu webgpu offline delete unload switch diagnostics" },
    { value: "access", label: "API access", drawer: "session_admin", does: "Set an API key", keywords: "auth api access key bearer" },
  ],
};

/** Whether this runtime has a section: a drawer section its availability allows (the HTTP runtime's
 *  health and API access, the browser's local runtime), a panel section always but the unit
 *  catalog, which needs a runtime that reads units. */
export function sectionShown(row: RoomSection): boolean {
  const client = getRuntimeClient();
  if (row.drawer === null) return row.value !== "units" || unitReadsAvailable(client);
  if (client.mode === "http" && row.drawer === "local_runtime") return false;
  if (client.mode !== "http" && (row.drawer === "health" || row.drawer === "session_admin")) return false;
  return drawerAvailability(row.drawer).available;
}

/** A room's sections this runtime has, in display order. */
export function shownSections(room: Room): readonly RoomSection[] {
  return (ROOM_SECTIONS[room] ?? []).filter(sectionShown);
}

/** The room and section that host a drawer, if one does. */
export function roomFor(drawer: DrawerName): { room: Room; section: string } | null {
  for (const [room, sections] of Object.entries(ROOM_SECTIONS) as [Room, readonly RoomSection[]][]) {
    const section = sections.find(row => row.drawer === drawer);
    if (section) return { room, section: section.value };
  }
  return null;
}

export const route: { room: Room; section: string | null; params: unknown } = $state({
  room: "workbench",
  section: null,
  params: null,
});

function isRoom(value: string): value is Room {
  return ROOMS.some(room => room.value === value);
}

function parse(hash: string): { room: Room; section: string | null } {
  const [room = "", section = ""] = hash.replace(/^#\/?/, "").split("/");
  if (!isRoom(room)) return { room: "workbench", section: null };
  const known = ROOM_SECTIONS[room]?.some(row => row.value === section);
  return { room, section: known ? section : null };
}

function hashFor(room: Room, section: string | null): string {
  if (room === "workbench") return "";
  return section ? `#/${room}/${section}` : `#/${room}`;
}

/** Goes to a room (and section), carrying in-memory params a hosted drawer reads. */
export function goto(room: Room, section: string | null = null, params: unknown = null): void {
  route.room = room;
  route.section = section;
  route.params = params;
  if (typeof window === "undefined") return;
  const next = hashFor(room, section);
  if (window.location.hash === next || (!next && !window.location.hash)) return;
  const url = `${window.location.pathname}${window.location.search}${next}`;
  window.history.pushState(null, "", url);
}

/** Follows the address bar (back, forward, a pasted link); returns the stop function. */
export function startRouting(): () => void {
  const follow = () => {
    const next = parse(window.location.hash);
    if (next.room !== route.room || next.section !== route.section) route.params = null;
    route.room = next.room;
    route.section = next.section;
  };
  follow();
  window.addEventListener("popstate", follow);
  window.addEventListener("hashchange", follow);
  return () => {
    window.removeEventListener("popstate", follow);
    window.removeEventListener("hashchange", follow);
  };
}
