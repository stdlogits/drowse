import assert from "node:assert/strict";
import { createServer } from "vite";
const server = await createServer({
  configFile: false, appType: "custom", logLevel: "error",
  optimizeDeps: { noDiscovery: true, include: [] },
  server: { middlewareMode: true, watch: null, hmr: false, ws: false },
});
try {
  // One focus-containment rule for every modal surface.
  const { containTab, tabStops } = await server.ssrLoadModule("/src/lib/focusTrap.ts");
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
  } finally { globalThis.document = previousDocument; }
  console.log("focus containment wraps, enters, skips hidden/disabled/inert stops");
} finally { await server.close(); }
