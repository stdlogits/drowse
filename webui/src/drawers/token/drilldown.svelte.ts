// Page-session state for the token view.
//
// The view reads in one order (alternatives, the lens by depth and layer, probes, features); there
// are no tabs. ``tab`` is the section last brought into view, which the WebMCP token inspector
// reads and sets (setting it scrolls that section into view). Module scope = page session.

export type DrilldownTab = "geometry" | "logits" | "sae" | "lens";

export const drilldownUi: { tab: DrilldownTab } = $state({ tab: "logits" });
