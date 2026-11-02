import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "vite";
const server = await createServer({
  configFile: false, appType: "custom", logLevel: "error",
  optimizeDeps: { noDiscovery: true, include: [] },
  server: { middlewareMode: true, watch: null, hmr: false, ws: false },
});
try {
  // One focus-containment rule for every modal surface.
  const { containTab, tabStops, focusOpener, returnFocus } = await server.ssrLoadModule("/src/lib/focusTrap.ts");
  const previousDocument = globalThis.document;
  const element = (name, props = {}) => ({
    name, tabIndex: 0, disabled: false, offsetParent: {}, closest: () => null,
    focus() { globalThis.document.activeElement = this; }, ...props,
  });
  const stops = [element("first"), element("hidden", { offsetParent: null }), element("skipped", { tabIndex: -1 }),
    element("disabled", { disabled: true }), element("inert", { closest: selector => selector === "[inert]" ? {} : null }), element("last")];
  const container = { querySelectorAll: () => stops, focus() { globalThis.document.activeElement = this; } };
  const key = (shiftKey = false) => ({ key: "Tab", shiftKey, prevented: false, preventDefault() { this.prevented = true; } });
  try {
    globalThis.document = { activeElement: container };
    assert.deepEqual(tabStops(container).map(stop => stop.name), ["first", "last"]);
    let event = key();
    containTab(event, container);
    assert.equal(document.activeElement.name, "first", "Tab from the container enters at the first stop");
    event = key(true);
    containTab(event, container);
    assert.equal(document.activeElement.name, "last", "Shift+Tab from the first stop wraps to the last");
    event = key();
    containTab(event, container);
    assert.equal(document.activeElement.name, "first", "Tab from the last stop wraps to the first");
    document.activeElement = stops[0];
    event = key();
    containTab(event, container);
    assert.equal(event.prevented, false, "Tab between interior stops is left to the browser");
    const empty = { querySelectorAll: () => [], focus() { globalThis.document.activeElement = this; } };
    event = key();
    containTab(event, empty);
    assert.equal(document.activeElement, empty, "a surface with no stops holds focus");

    // Closing a dialog gives focus back: Escape on Menu > Feedback left it on the body (drowse.ai's
    // accessibility check). To the opener while it is still a tab stop, else to Menu.
    const body = { name: "body" };
    const dialog = { contains: (node) => node?.name === "inside" };
    const inside = element("inside");
    let menuFocused = 0;
    const menu = () => { menuFocused += 1; document.activeElement = { name: "menu" }; };
    globalThis.document = { body, activeElement: body };
    assert.equal(focusOpener(), null, "nothing focused: no opener");
    const opener = element("sampling button", { isConnected: true });
    document.activeElement = opener;
    assert.equal(focusOpener(), opener, "the focused button is the opener");
    document.activeElement = body;
    returnFocus(opener, menu, dialog);
    assert.equal(document.activeElement.name, "sampling button", "a button that opened the dialog takes focus back");
    document.activeElement = body;
    returnFocus(element("menu item", { isConnected: false }), menu, dialog);
    assert.equal(document.activeElement.name, "menu", "a menu item that went with its menu: focus goes to Menu");
    document.activeElement = body;
    returnFocus(element("hidden", { isConnected: true, offsetParent: null }), menu, dialog);
    assert.equal(document.activeElement.name, "menu", "an opener no longer shown: Menu");
    document.activeElement = body;
    returnFocus(null, menu, dialog);
    assert.equal(document.activeElement.name, "menu", "no opener: Menu");
    document.activeElement = inside;
    returnFocus(null, menu, dialog);
    assert.equal(document.activeElement.name, "menu", "focus still in the leaving dialog comes back too");
    const elsewhere = element("elsewhere");
    document.activeElement = elsewhere;
    const before = menuFocused;
    returnFocus(opener, menu, dialog);
    assert.equal(document.activeElement, elsewhere, "focus that already moved on stays there");
    assert.equal(menuFocused, before);
  } finally { globalThis.document = previousDocument; }

  // App wires it: the opener is taken as the dialog opens and focus goes back after the close's
  // tick (the header is inert until then), with Menu's own focus() as the fallback.
  const app = readFileSync(new URL("../src/App.svelte", import.meta.url), "utf8");
  assert.match(app, /if \(previousDialog === null\) dialogOpener = focusOpener\(\);/);
  assert.match(app, /void tick\(\)\.then\(\(\) => returnFocus\(opener, \(\) => headerMenu\?\.focus\(\), leaving\)\)/);
  assert.match(app, /<HeaderMenu bind:this=\{headerMenu\}/);
  assert.match(readFileSync(new URL("../src/lib/ui/HeaderMenu.svelte", import.meta.url), "utf8"),
    /export function focus\(\): void \{\s*button\?\.focus\(\{ preventScroll: true \}\);/);
  console.log("focus containment wraps, enters, skips hidden/disabled/inert stops; a closing dialog gives focus back");
} finally { await server.close(); }
