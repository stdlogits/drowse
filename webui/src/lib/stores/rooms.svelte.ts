// The dashboard's rooms, as routes. The workbench is home; Manifolds, Units, Lens and Model are
// rooms beside it, reached from the header's tabs, the command palette or a link (``#/lens``).
// A room may hold sections (``#/manifolds/packs``), and the tools that used to open as drawers
// live in them: opening one of those drawers (``openDrawer("manifold_pack")``) goes to its room.
import type { DrawerName } from "../types";

export type Room = "workbench" | "manifolds" | "units" | "lens" | "model";

export const ROOMS: readonly { value: Room; label: string }[] = [
  { value: "workbench", label: "Workbench" },
  { value: "manifolds", label: "Manifolds" },
  { value: "units", label: "Units" },
  { value: "lens", label: "Lens" },
  { value: "model", label: "Model" },
];

export interface RoomSection {
  value: string;
  label: string;
  drawer: DrawerName;
}

/** The drawers each room hosts, in display order. ``local_runtime`` and ``health`` are the two
 *  runtimes' model views (browser and HTTP); a room shows the one this runtime has. */
export const ROOM_SECTIONS: Partial<Record<Room, readonly RoomSection[]>> = {
  manifolds: [
    { value: "build", label: "Build", drawer: "manifold_builder" },
    { value: "packs", label: "Packs", drawer: "manifold_pack" },
    { value: "combine", label: "Combine", drawer: "manifold_merge" },
    { value: "templates", label: "Templates", drawer: "template_lab" },
    { value: "roles", label: "Roles", drawer: "cast" },
  ],
  model: [
    { value: "health", label: "Health", drawer: "health" },
    { value: "runtime", label: "Runtime", drawer: "local_runtime" },
    { value: "access", label: "API access", drawer: "session_admin" },
  ],
};

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
