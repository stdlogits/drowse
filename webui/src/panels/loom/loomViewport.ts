export interface LoomCamera {
  x: number;
  y: number;
  zoom: number;
}

export interface LoomSize {
  width: number;
  height: number;
}

export interface LoomPoint {
  x: number;
  y: number;
}

export const LOOM_MIN_ZOOM = 0.01;
export const LOOM_MAX_ZOOM = 1.4;

export function clampLoomZoom(value: number): number {
  return Math.min(LOOM_MAX_ZOOM, Math.max(LOOM_MIN_ZOOM, value));
}

export function fitLoomCamera(
  viewport: LoomSize,
  graph: LoomSize,
  padding = Math.min(48, viewport.width / 10, viewport.height / 10),
): LoomCamera {
  if (viewport.width <= 0 || viewport.height <= 0 || graph.width <= 0 || graph.height <= 0) {
    return { x: 0, y: 0, zoom: 1 };
  }
  const availableWidth = Math.max(1, viewport.width - padding * 2);
  const availableHeight = Math.max(1, viewport.height - padding * 2);
  const zoom = clampLoomZoom(Math.min(
    1,
    availableWidth / graph.width,
    availableHeight / graph.height,
  ));
  const scaledWidth = graph.width * zoom;
  const scaledHeight = graph.height * zoom;
  return {
    x: scaledWidth <= availableWidth ? (viewport.width - scaledWidth) / 2 : padding,
    y: scaledHeight <= availableHeight ? (viewport.height - scaledHeight) / 2 : padding,
    zoom,
  };
}

export function centerLoomRect(
  viewport: LoomSize,
  rect: LoomPoint & LoomSize,
  zoom: number,
): LoomCamera {
  const nextZoom = clampLoomZoom(zoom);
  return {
    x: viewport.width / 2 - (rect.x + rect.width / 2) * nextZoom,
    y: viewport.height / 2 - (rect.y + rect.height / 2) * nextZoom,
    zoom: nextZoom,
  };
}

export type LoomRect = LoomPoint & LoomSize;

/** What ``settleLoomCenter`` needs from the map. */
export interface LoomCenterHost {
  /** The target's placement now, or null when it is gone (or a newer move superseded this one). */
  placement(): LoomRect | null;
  /** Move the camera onto a placement and return the camera that move set. */
  center(rect: LoomRect): LoomCamera;
  /** The camera now: a pan, zoom or pinch since ``center`` shows up here. */
  camera(): LoomCamera;
  /** Resolves once the turns the last move brought into view have measured themselves. */
  remeasured(): Promise<void>;
}

const sameRect = (a: LoomRect, b: LoomRect) =>
  Math.abs(a.x - b.x) < 0.5 && Math.abs(a.y - b.y) < 0.5 &&
  Math.abs(a.width - b.width) < 0.5 && Math.abs(a.height - b.height) < 0.5;
const sameCamera = (a: LoomCamera, b: LoomCamera) =>
  Math.abs(a.x - b.x) < 0.5 && Math.abs(a.y - b.y) < 0.5 && Math.abs(a.zoom - b.zoom) < 1e-6;

/** Centre the camera on a turn and keep it there while the map settles. Turns enter the layout at
 *  an estimated height and take their measured one once drawn, so the turns a move brings into view
 *  can shift the target; each time they do, the camera follows, until the placement holds, someone
 *  else moves the camera, or ``passes`` run out. Returns how many moves it made. */
export async function settleLoomCenter(host: LoomCenterHost, passes = 4): Promise<number> {
  let rect = host.placement();
  if (!rect) return 0;
  let set = host.center(rect);
  let moves = 1;
  for (let pass = 0; pass < passes; pass += 1) {
    await host.remeasured();
    if (!sameCamera(host.camera(), set)) break;
    const next = host.placement();
    if (!next || sameRect(next, rect)) break;
    rect = next;
    set = host.center(rect);
    moves += 1;
  }
  return moves;
}

export function resizeLoomCamera(
  camera: LoomCamera,
  previous: LoomSize,
  viewport: LoomSize,
): LoomCamera {
  return {
    x: camera.x + (viewport.width - previous.width) / 2,
    y: camera.y + (viewport.height - previous.height) / 2,
    zoom: camera.zoom,
  };
}

export function zoomLoomAt(
  camera: LoomCamera,
  nextZoom: number,
  anchor: LoomPoint,
): LoomCamera {
  const zoom = clampLoomZoom(nextZoom);
  const worldX = (anchor.x - camera.x) / camera.zoom;
  const worldY = (anchor.y - camera.y) / camera.zoom;
  return {
    x: anchor.x - worldX * zoom,
    y: anchor.y - worldY * zoom,
    zoom,
  };
}
