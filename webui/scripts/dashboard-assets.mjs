// Which files of webui/public ship in the dashboard bundle that `drowse serve`
// mounts (and so in the wheel).  The public dir also holds drowse's brand set,
// which drowse.ai copies into its own public-hosted dir: the social card
// (social/), favicon.svg (both themes in one file) and the PWA and home-screen
// rasters (icons/).  The dashboard ships only the static marks its index.html
// links, one per theme (drowse-mark-{light,dark}.svg: a tile on the depth ramp,
// a paper-coloured closed eye), and favicon.ico, which browsers request whatever
// the page links.
const dashboardIcons = ["drowse-mark-light.svg", "drowse-mark-dark.svg", "favicon.ico"];

/** Whether a public-dir file (path relative to the public dir, `/`-separated)
 *  belongs in the dashboard build. */
export function shipsInDashboard(path) {
  if (path.startsWith("social/") || path.startsWith("icons/")) return false;
  return path !== "favicon.svg";
}

/** Icons the dashboard serves; the build must contain them. */
export const requiredDashboardIcons = dashboardIcons;
