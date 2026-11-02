// Where a popover opens beside its anchor (the token probabilities, TokenLogitsPopover), store-free
// so a script test can check it fits, styles written included. It keeps 8 px inside the visual viewport (a pinch-zoomed page
// included), starts at the anchor's left edge, and opens 6 px below the anchor, or above it when the
// panel fits better there; it never covers the anchor, and a panel taller than its side scrolls.

/** The visual viewport, in layout px: ``window.visualViewport``'s offset and size. */
export interface Viewport {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Where to put the panel: its corner and the height it may take before it scrolls. */
export interface Placement {
  left: number;
  top: number;
  maxHeight: number;
}

const MARGIN = 8;
const GAP = 6;

/** The viewport the window has now. */
export function currentViewport(): Viewport {
  const viewport = window.visualViewport;
  return {
    left: viewport?.offsetLeft ?? 0,
    top: viewport?.offsetTop ?? 0,
    width: viewport?.width ?? innerWidth,
    height: viewport?.height ?? innerHeight,
  };
}

/** The largest box a panel may take in ``view`` (its max-width and max-height before placing). */
export function popoverBounds(view: Viewport): { maxWidth: number; maxHeight: number } {
  return { maxWidth: view.width - 2 * MARGIN, maxHeight: view.height - 2 * MARGIN };
}

/**
 * Place a panel of ``size`` (measured within :func:`popoverBounds`) beside ``anchor`` in ``view``.
 * Below when it fits there or there is at least as much room below as above, else above.
 */
export function placePopover(
  anchor: { left: number; top: number; bottom: number },
  size: { width: number; height: number },
  view: Viewport,
): Placement {
  const bounds = popoverBounds(view);
  const width = Math.min(size.width, bounds.maxWidth);
  const left = Math.max(view.left + MARGIN, Math.min(anchor.left, view.left + view.width - width - MARGIN));
  const below = Math.max(0, view.top + view.height - anchor.bottom - GAP - MARGIN);
  const above = Math.max(0, anchor.top - view.top - GAP - MARGIN);
  const opensBelow = size.height <= below || below >= above;
  const maxHeight = Math.min(bounds.maxHeight, opensBelow ? below : above);
  const height = Math.min(size.height, maxHeight);
  const top = Math.max(
    view.top + MARGIN,
    Math.min(opensBelow ? anchor.bottom + GAP : anchor.top - height - GAP, view.top + view.height - height - MARGIN),
  );
  return { left, top, maxHeight };
}

/**
 * Size ``panel`` to ``view``, measure it, and put it beside ``anchor``: TokenLogitsPopover's place().
 * The max size goes on before the measure, so a long panel is measured as the box it will be.
 */
export function applyPopoverPlacement(
  panel: {
    style: Pick<CSSStyleDeclaration, "left" | "top" | "maxWidth" | "maxHeight">;
    getBoundingClientRect(): { width: number; height: number };
  },
  anchor: { getBoundingClientRect(): { left: number; top: number; bottom: number } },
  view: Viewport,
): Placement {
  const bounds = popoverBounds(view);
  panel.style.maxWidth = `${bounds.maxWidth}px`;
  panel.style.maxHeight = `${bounds.maxHeight}px`;
  const at = placePopover(anchor.getBoundingClientRect(), panel.getBoundingClientRect(), view);
  panel.style.left = `${at.left}px`;
  panel.style.top = `${at.top}px`;
  panel.style.maxHeight = `${at.maxHeight}px`;
  return at;
}
