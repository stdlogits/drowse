import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "vite";

// The dialog host and the token popover, the surfaces drowse.ai's browser checks used to cover:
// - an open drawer makes the page behind it inert, and a leaving one takes no input while it fades;
// - the drawer dialog (Sampling settings) and the token probabilities stay inside the viewport at
//   phone, landscape and desktop sizes, the close button in view and the fields inside the body.
// Rendered on the server, read from the components' styles, and placed by lib/popoverPlacement.ts.
const src = new URL("../src/", import.meta.url);
const read = (path) => readFileSync(new URL(path, src), "utf8");
const VIEWPORTS = [[320, 780], [390, 844], [667, 375], [900, 900], [1440, 900]];

// ---- a component's <style>, as rules with their media ---------------------------------------
function styleRules(source) {
  const css = source.slice(source.indexOf("<style>") + 7, source.indexOf("</style>")).replace(/\/\*[^]*?\*\//g, "");
  const rules = [];
  (function walk(text, media) {
    let i = 0;
    while (i < text.length) {
      const open = text.indexOf("{", i);
      if (open < 0) break;
      let depth = 1;
      let close = open + 1;
      for (; depth > 0; close += 1) depth += text[close] === "{" ? 1 : text[close] === "}" ? -1 : 0;
      const head = text.slice(i, open).trim();
      const body = text.slice(open + 1, close - 1);
      if (head.startsWith("@")) walk(body, head);
      else {
        const decls = {};
        for (const part of body.split(";")) {
          const colon = part.indexOf(":");
          if (colon > 0) decls[part.slice(0, colon).trim()] = part.slice(colon + 1).trim();
        }
        rules.push({ media, selectors: head.split(",").map(s => s.trim()), decls });
      }
      i = close;
    }
  })(css, null);
  return rules;
}

/** The declarations ``classes`` gets at ``width`` (max-width media only; later and more specific wins). */
function computed(rules, classes, width) {
  const applies = (media) => media === null || (media.startsWith("@media") && media.split(",").some(query => {
    const max = query.match(/max-width:\s*(\d+)px/);
    return max !== null && !/pointer|prefers/.test(query) && width <= Number(max[1]);
  }));
  const hits = [];
  rules.forEach((rule, order) => {
    if (!applies(rule.media)) return;
    for (const selector of rule.selectors) {
      const needed = selector.split(".").filter(Boolean);
      if (/^\.[\w.-]+$/.test(selector) && needed.every(name => classes.includes(name))) hits.push({ order, specificity: needed.length, decls: rule.decls });
    }
  });
  hits.sort((a, b) => a.specificity - b.specificity || a.order - b.order);
  return Object.assign({}, ...hits.map(hit => hit.decls));
}

// ---- lengths: px, vw/vh, %, var(--token), calc(), min(), max() ------------------------------
const ledgerTokens = Object.fromEntries([...readFileSync(new URL("../node_modules/@stdlogits/ledger/src/styles/tokens.css", import.meta.url), "utf8")
  .slice(0, 4000).matchAll(/(--s\d):\s*(\d+)px;/g)].map(([, name, value]) => [name, Number(value)]));
function length(text, { vw, vh, percent }) {
  const tokens = text.replace(/var\((--[\w-]+)\)/g, (_, name) => {
    assert.ok(name in ledgerTokens, `unknown length token ${name}`);
    return `${ledgerTokens[name]}px`;
  }).match(/[\d.]+(?:px|vw|vh|%)?|[a-z]+\(|[()+\-*/,]/g);
  let at = 0;
  const unit = (n) => Number.parseFloat(n) * (n.endsWith("vw") ? vw / 100 : n.endsWith("vh") ? vh / 100 : n.endsWith("%") ? percent / 100 : 1);
  function expr() {
    let value = term();
    while (tokens[at] === "+" || tokens[at] === "-") value = tokens[at++] === "+" ? value + term() : value - term();
    return value;
  }
  function term() {
    let value = atom();
    while (tokens[at] === "*" || tokens[at] === "/") value = tokens[at++] === "*" ? value * atom() : value / atom();
    return value;
  }
  function atom() {
    const token = tokens[at++];
    if (token === "calc(" || token === "(") { const value = expr(); at += 1; return value; }
    if (token === "min(" || token === "max(") {
      const args = [expr()];
      while (tokens[at] === ",") { at += 1; args.push(expr()); }
      at += 1;
      return token === "min(" ? Math.min(...args) : Math.max(...args);
    }
    return unit(token);
  }
  const value = expr();
  assert.equal(at, tokens.length, `could not read the length ${text}`);
  return value;
}
const splitTop = (value) => value.match(/(?:[^\s(]+|\([^)]*\))+/g);

/** A fixed box from its insets and width, the way the dialog lays out. */
function fixedBox(decls, vw, vh) {
  const sides = splitTop(decls.inset);
  const [top, right, bottom, left] = [sides[0], sides[1] ?? sides[0], sides[2] ?? sides[0], sides[3] ?? sides[1] ?? sides[0]];
  const at = (side, extent) => side === "auto" ? null : length(side, { vw, vh, percent: extent });
  const width = length(decls.width, { vw, vh, percent: vw });
  const r = at(right, vw);
  const l = at(left, vw) ?? vw - r - width;
  return { left: l, right: l + width, top: at(top, vh), bottom: vh - at(bottom, vh), width };
}

const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "http", instruments: {} });
  const { render } = await server.ssrLoadModule("svelte/server");
  const { openDrawer, closeDrawer } = await server.ssrLoadModule("/src/lib/stores.svelte.ts");
  const { loadDrawer } = await server.ssrLoadModule("/src/drawers/index.ts");
  const { default: App } = await server.ssrLoadModule("/src/App.svelte");
  const appSource = read("App.svelte");
  const tagOf = (html, cls) => html.match(new RegExp(`<[a-z]+ [^>]*class="${cls}[ "][^>]*>`))?.[0] ?? "";

  // ---- an open drawer makes the page behind it inert ------------------------------------------
  {
    const closed = render(App).body;
    assert.doesNotMatch(closed, /role="dialog"/);
    for (const cls of ["dw-top", "notices"]) assert.doesNotMatch(tagOf(closed, cls), /\binert\b/, `.${cls} takes input with no drawer open`);
    openDrawer("advanced_sampling");
    // The first open shows DrawerOutlet's loading blocks until the drawer's chunk arrives.
    const loading = render(App).body;
    assert.match(loading, /<div class="dialog[^"]*" role="dialog" aria-modal="true" aria-label="Sampling settings" tabindex="-1">/);
    assert.match(loading, /class="drawer-loading[^"]*" role="status" aria-label="Loading"/);
    await loadDrawer("advanced_sampling");
    const open = render(App).body;
    for (const cls of ["dw-top", "notices"]) assert.match(tagOf(open, cls), /\binert=""/, `.${cls} is inert behind the drawer`);
    // On the server the rooms host is inert anyway (it boots in the browser): read its rule instead.
    assert.match(appSource, /<main id="main" class="rooms-host"[^>]*\binert=\{dialogOpen \|\|/);
    // Loaded, the drawer renders in the dialog: its head (with Close) above the body that scrolls.
    const dialog = open.slice(open.indexOf('role="dialog"'));
    assert.ok(dialog.indexOf('class="dlg-head') >= 0 && dialog.indexOf('class="dlg-head') < dialog.indexOf('aria-label="Close"'));
    assert.ok(dialog.indexOf('aria-label="Close"') < dialog.indexOf('class="body'), "Close sits in the head, outside the scroller");
    closeDrawer();
    assert.doesNotMatch(render(App).body, /role="dialog"/);
  }

  // ---- a leaving drawer and popover take no input -------------------------------------------
  {
    // Svelte makes an element inert when its outro starts and gives it back its own inert if it
    // returns mid-exit; the dialog and its scrim leave through an out: transition, so a click or Tab
    // during the fade reaches the page, not the closing drawer.
    const transitions = readFileSync(new URL("../node_modules/svelte/src/internal/client/dom/elements/transitions.js", import.meta.url), "utf8");
    assert.match(transitions, /\bout\(fn\)\s*\{[^]*?element\.inert = true;/, "svelte: an outro makes its element inert");
    assert.match(transitions, /\bin\(\)\s*\{\s*element\.inert = inert;/, "svelte: an intro restores it");
    const dialogTag = appSource.slice(appSource.indexOf('<div bind:this={dialogEl} class="dialog"'), appSource.indexOf("<ManifoldJobProgress compact />\n        <DrawerOutlet"));
    assert.match(dialogTag, /aria-modal="true"/);
    assert.match(dialogTag, /containTab\(event, dialogEl\)/, "Tab stays inside the open dialog");
    assert.match(dialogTag, /\bout:fadeOut\b/);
    assert.match(appSource, /<div class="scrim"[^>]*\bout:fadeOut\b/);
    // The token popover goes inert the moment it closes (before its parent's conditional drops it),
    // hands focus back to its token, and comes back to life if reopened mid-fade.
    const popover = read("lib/ui/TokenLogitsPopover.svelte");
    assert.match(popover, /function close\([^)]*\) \{\n[^\n]*\n\s*closing = true;\s*panel\.inert = true;\s*if \(restoreFocus && trigger\.isConnected\) trigger\.focus\([^\n]*\n\s*onclose\(\);\n\s*\}/);
    assert.match(popover, /out:fadeOut\|global/);
    assert.match(popover, /onoutrostart=\{\(\) => \{ closing = true; panel\.inert = true; \}\}/);
    assert.match(popover, /onintrostart=\{\(\) => \{[^}]*panel\.inert = false;/);
  }

  // ---- the drawer dialog fits the viewport ------------------------------------------------------
  {
    const appRules = styleRules(appSource);
    for (const [vw, vh] of VIEWPORTS) {
      for (const classes of [["dialog"], ["dialog", "narrow"]]) {
        const decls = computed(appRules, classes, vw);
        assert.equal(decls.position, "fixed");
        const box = fixedBox(decls, vw, vh);
        const where = `.${classes.join(".")} at ${vw}x${vh}`;
        assert.ok(box.left >= 0 && box.right <= vw, `${where}: ${box.left}..${box.right} wide`);
        assert.ok(box.top >= 0 && box.bottom <= vh && box.bottom - box.top >= vh / 2, `${where}: ${box.top}..${box.bottom} tall`);
        if (vw <= 720) assert.equal(box.width, vw, `${where}: a sheet across the phone`);
      }
    }
    // The dialog clips and stacks its drawer; Sampling settings fills it, its head (Close) holds
    // still while its body scrolls, and its fields share the body's width rather than overflow it.
    const dialog = computed(appRules, ["dialog"], 1440);
    assert.deepEqual([dialog.display, dialog["flex-direction"], dialog.overflow], ["flex", "column", "hidden"]);
    const sampling = styleRules(read("drawers/AdvancedSamplingDrawer.svelte"));
    for (const vw of [320, 1440]) {
      const at = (cls) => computed(sampling, [cls], vw);
      assert.deepEqual([at("sampling-settings").display, at("sampling-settings")["flex-direction"], at("sampling-settings").height, at("sampling-settings")["min-height"]], ["flex", "column", "100%", "0"]);
      assert.equal(at("dlg-head").flex, "0 0 auto");
      assert.deepEqual([at("body").overflow, at("body")["min-height"], at("body")["grid-template-columns"]], ["auto", "0", "minmax(0, 1fr)"]);
      assert.match(at("fields")["grid-template-columns"], /^repeat\(2, minmax\(0, 1fr\)\)$/);
      assert.equal(at("field")["min-width"], "0");
    }
    assert.ok(sampling.some(rule => /@container sampling-settings \(max-width: 560px\)/.test(rule.media ?? "")
      && rule.selectors.includes(".fields") && rule.decls["grid-template-columns"] === "minmax(0, 1fr)"), "one column when the dialog is narrow");
  }

  // ---- the token popover fits the viewport ------------------------------------------------------
  {
    const { placePopover, popoverBounds } = await server.ssrLoadModule("/src/lib/popoverPlacement.ts");
    const views = [
      ...VIEWPORTS.map(([width, height]) => ({ left: 0, top: 0, width, height })),
      { left: 140, top: 260, width: 180, height: 300 }, // pinch-zoomed into a phone
    ];
    const sizes = [{ width: 320, height: 160 }, { width: 320, height: 640 }, { width: 2000, height: 2000 }];
    for (const view of views) {
      const bounds = popoverBounds(view);
      for (const fx of [0, 0.5, 0.97]) {
        for (const fy of [0, 0.3, 0.5, 0.9]) {
          const left = view.left + fx * (view.width - 24);
          const top = view.top + fy * (view.height - 20);
          const anchor = { left, top, bottom: top + 20 };
          for (const natural of sizes) {
            const size = { width: Math.min(natural.width, bounds.maxWidth), height: Math.min(natural.height, bounds.maxHeight) };
            const at = placePopover(anchor, size, view);
            const height = Math.min(size.height, at.maxHeight);
            const where = `${size.width}x${size.height} at (${left}, ${top}) in ${JSON.stringify(view)}`;
            assert.ok(at.left >= view.left + 8 && at.left + size.width <= view.left + view.width - 8 + 1e-9, `${where}: across`);
            assert.ok(at.top >= view.top + 8 && at.top + height <= view.top + view.height - 8 + 1e-9, `${where}: down`);
            assert.ok(height > 0, `${where}: some room`);
            assert.ok(at.top >= anchor.bottom || at.top + height <= anchor.top, `${where}: the token stays visible`);
            const below = view.top + view.height - anchor.bottom - 14;
            if (size.height <= below) assert.equal(at.top, anchor.bottom + 6, `${where}: opens below when it fits`);
          }
        }
      }
    }
  }
  console.log("Drawers make the page inert and leave inert; the drawer dialog and the token popover fit every viewport");
} finally { await server.close(); }
