// Focus containment for modal surfaces (drawer dialog, command palette, loom
// dialogs, bottom sheet): one definition of a tab stop, one wrap rule and one
// rule for where focus goes back when a surface closes.

const CANDIDATES = "a[href], button, input, select, textarea, [tabindex]";

/** A visible, enabled, non-inert element that Tab reaches. */
function isTabStop(element: HTMLElement): boolean {
  return element.tabIndex >= 0 &&
    !(element as HTMLButtonElement).disabled &&
    element.offsetParent !== null &&
    !element.closest("[inert]");
}

/** The visible, enabled, non-inert tab stops inside ``container``, in DOM order. */
export function tabStops(container: HTMLElement): HTMLElement[] {
  return [...container.querySelectorAll<HTMLElement>(CANDIDATES)].filter(isTabStop);
}

/** Keep Tab and Shift+Tab inside ``container``.  Focus wraps from the last
 *  stop to the first (and back); focus on the container itself, or anywhere
 *  outside its stops, enters at the near end; a container with no stops
 *  holds focus on itself.  Call from the container's keydown handler. */
export function containTab(event: KeyboardEvent, container: HTMLElement): void {
  if (event.key !== "Tab") return;
  const stops = tabStops(container);
  if (stops.length === 0) {
    event.preventDefault();
    container.focus({ preventScroll: true });
    return;
  }
  const index = stops.indexOf(document.activeElement as HTMLElement);
  if (event.shiftKey ? index <= 0 : index < 0 || index === stops.length - 1) {
    event.preventDefault();
    stops[event.shiftKey ? stops.length - 1 : 0].focus({ preventScroll: true });
  }
}

/** What has focus as a modal surface opens: where focus goes back when it closes. */
export function focusOpener(): HTMLElement | null {
  const active = document.activeElement as HTMLElement | null;
  return active && active !== document.body ? active : null;
}

/** Give focus back as a modal surface closes: to ``opener`` while it is still a tab stop (a
 *  button on the page), else to ``fallback`` (the header's Menu, whose item that opened the
 *  surface has gone with the menu).  Focus that already moved out of ``surface`` (left on the
 *  body, it would be lost) stays where it went.  Call once the page behind is no longer inert. */
export function returnFocus(opener: HTMLElement | null, fallback: (() => void) | null, surface: HTMLElement | null): void {
  const active = document.activeElement as HTMLElement | null;
  if (active && active !== document.body && !surface?.contains(active)) return;
  if (opener?.isConnected && isTabStop(opener)) {
    opener.focus({ preventScroll: true });
    if (document.activeElement === opener) return;
  }
  fallback?.();
}
