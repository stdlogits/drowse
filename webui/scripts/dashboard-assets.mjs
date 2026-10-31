// Which files of webui/public ship in the dashboard bundle that `drowse serve`
// mounts (and so in the wheel).  The public dir also holds the hosted site's
// brand set, which drowse.ai copies into its own public-hosted dir and checks
// against these sources: social preview art, the PWA icons, the tab icons and
// the old favicons.  The dashboard ships none of it: its tab icon is the
// static drowse mark (a tile on the depth ramp), one file per theme.
const dashboardMarks = ["drowse-mark-light.svg", "drowse-mark-dark.svg"];

/** Whether a public-dir file (path relative to the public dir, `/`-separated)
 *  belongs in the dashboard build. */
export function shipsInDashboard(path) {
  if (path.startsWith("social/") || path.startsWith("icons/")) return false;
  // The brand set's favicons, and the licence of the Fluent icons its tab rasters are drawn with.
  if (/^favicon\./.test(path) || path.startsWith("LICENSE-Fluent")) return false;
  return true;
}

/** Icons the dashboard references at runtime; the build must contain them. */
export const requiredDashboardIcons = dashboardMarks;
