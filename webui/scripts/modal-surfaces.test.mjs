import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "vite";

// The dialog host and the token popover, the surfaces drowse.ai's browser checks used to cover:
// - an open drawer makes the page behind it inert, and a leaving one takes no input while it fades;
// - the drawer dialog (Sampling settings, and DrawerOutlet loading it) and the token probabilities
//   stay inside the viewport at phone, landscape and desktop sizes, the close button in view and the
//   fields inside the body.
// Rendered on the server, read from the components' styles, and placed by lib/popoverPlacement.ts.
const src = new URL("../src/", import.meta.url);
const read = (path) => readFileSync(new URL(path, src), "utf8");
const VIEWPORTS = [[320, 780], [390, 844], [667, 375], [900, 900], [1440, 900]];

// ---- a stylesheet or a component's <style>, as rules with their media -----------------------
const styleRules = (source) => cssRules(source.slice(source.indexOf("<style>") + 7, source.indexOf("</style>")));
function cssRules(text) {
  const css = text.replace(/\/\*[^]*?\*\//g, "");
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

/**
 * A fixed box in a ``vw`` x ``vh`` viewport, laid out as CSS lays out a fixed, non-replaced box with
 * no margins: per axis from its insets, size and min/max size. A box the model cannot place (sized
 * by its content, at its static position, or styled with a property it does not read) fails.
 */
function fixedBox(decls, vw, vh) {
  const unread = Object.keys(decls).filter(name => /^(margin|transform|translate|scale|rotate|inset-|block-size|inline-size|(min|max)-(block|inline)-size|aspect-ratio)/.test(name));
  assert.deepEqual(unread, [], "a fixed box styled with properties the fit model does not read");
  const sides = splitTop(decls.inset ?? "auto");
  const inset = { top: sides[0], right: sides[1] ?? sides[0], bottom: sides[2] ?? sides[0], left: sides[3] ?? sides[1] ?? sides[0] };
  for (const side of Object.keys(inset)) {
    if (decls[side] === undefined) continue;
    assert.equal(decls.inset, undefined, `both inset and ${side}: which one wins depends on rule order`);
    inset[side] = decls[side];
  }
  // CSS 2.1 10.3.7 and 10.6.4: an auto size spans the insets; min/max clamp it; when both insets and
  // the size are set the end inset gives way (left-to-right, top-to-bottom).
  function axis(start, end, size, min, max, extent, name) {
    const at = (value) => value === undefined || value === "auto" ? null : length(value, { vw, vh, percent: extent });
    const [s, e] = [at(start), at(end)];
    let span = at(size);
    if (span === null) {
      assert.ok(s !== null && e !== null, `${name} is sized by its content`);
      span = extent - s - e;
    }
    if (max !== undefined && max !== "none") span = Math.min(span, at(max));
    if (min !== undefined) span = Math.max(span, at(min) ?? 0);
    assert.ok(s !== null || e !== null, `${name} sits at its static position`);
    const from = s ?? extent - e - span;
    return [from, from + span];
  }
  const [left, right] = axis(inset.left, inset.right, decls.width, decls["min-width"], decls["max-width"], vw, "width");
  const [top, bottom] = axis(inset.top, inset.bottom, decls.height, decls["min-height"], decls["max-height"], vh, "height");
  return { left, right, top, bottom, width: right - left };
}

// The model, on boxes it must place and must refuse.
{
  const box = (decls) => fixedBox(decls, 400, 800);
  assert.deepEqual(box({ inset: "16px 0 0", width: "100%" }), { left: 0, right: 400, top: 16, bottom: 800, width: 400 });
  assert.deepEqual(box({ inset: "auto 0 0", width: "100%", height: "50vh" }), { left: 0, right: 400, top: 400, bottom: 800, width: 400 });
  assert.equal(box({ inset: "0 0 0 auto", width: "200px", height: "120vh" }).bottom, 960, "a set height outgrows the viewport");
  assert.equal(box({ inset: "0 0 0 auto", width: "200px", "min-height": "1000px" }).bottom, 1000, "min-height outgrows it");
  assert.equal(box({ inset: "0", width: "200px", "max-height": "100px" }).bottom, 100);
  assert.deepEqual(box({ inset: "0 0 0 auto", width: "200px" }), { left: 200, right: 400, top: 0, bottom: 800, width: 200 });
  assert.throws(() => box({ inset: "auto 0 0", width: "100%" }), /sized by its content/);
  assert.throws(() => box({ inset: "0", width: "100%", margin: "auto" }), /does not read/);
  assert.throws(() => box({ inset: "0", top: "auto", width: "100%" }), /rule order/);
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
    let shortest = Infinity;
    for (const [vw, vh] of VIEWPORTS) {
      for (const classes of [["dialog"], ["dialog", "narrow"]]) {
        const decls = computed(appRules, classes, vw);
        assert.equal(decls.position, "fixed");
        const box = fixedBox(decls, vw, vh);
        const where = `.${classes.join(".")} at ${vw}x${vh}`;
        assert.ok(box.left >= 0 && box.right <= vw, `${where}: ${box.left}..${box.right} wide`);
        assert.ok(box.top >= 0 && box.bottom <= vh && box.bottom - box.top >= vh / 2, `${where}: ${box.top}..${box.bottom} tall`);
        if (vw <= 720) assert.equal(box.width, vw, `${where}: a sheet across the phone`);
        shortest = Math.min(shortest, box.bottom - box.top);
      }
    }
    // DrawerOutlet's loading state, wash bars in the dialog's padding until the drawer's chunk
    // arrives, takes the dialog's width and fits its shortest height.
    const outlet = read("drawers/DrawerOutlet.svelte");
    const loadingTag = outlet.match(/<div class="drawer-loading dw-blocks"[^>]*>((?:<i><\/i>)+)<\/div>/);
    assert.ok(loadingTag, "DrawerOutlet's loading blocks");
    const bars = loadingTag[1].length / "<i></i>".length;
    const loading = computed(styleRules(outlet), ["drawer-loading", "dw-blocks"], 320);
    const shared = cssRules(readFileSync(new URL("lib/style/drowse.css", src), "utf8"));
    const blocks = Object.assign({}, ...shared.filter(rule => rule.media === null && rule.selectors.includes(".dw-blocks")).map(rule => rule.decls));
    const bar = Object.assign({}, ...shared.filter(rule => rule.media === null && rule.selectors.includes(".dw-blocks > i")).map(rule => rule.decls));
    for (const decls of [loading, blocks, bar]) {
      assert.deepEqual(Object.keys(decls).filter(name => /width|^position$|^inset|^margin/.test(name)), [], "the loading blocks take the dialog's width");
    }
    const lengthOf = (text) => length(text, { vw: 320, vh: 780, percent: 0 });
    const loadingHeight = 2 * lengthOf(loading.padding) + bars * lengthOf(bar.height) + (bars - 1) * lengthOf(blocks.gap);
    assert.ok(loadingHeight <= shortest, `DrawerOutlet's loading state (${loadingHeight}px) fits the shortest dialog (${shortest}px)`);
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
    // TokenLogitsPopover places itself on open and on every resize or scroll through
    // applyPopoverPlacement; here it runs over a panel that measures as a border-box element does
    // (its natural size, clamped by the max-width and max-height written on it).
    const popover = read("lib/ui/TokenLogitsPopover.svelte");
    assert.match(popover, /function place\(\) \{\s*if \(closing\) return;\s*applyPopoverPlacement\(panel, trigger, currentViewport\(\)\);\s*\}/);
    assert.match(popover, /onMount\(\(\) => \{\s*panel\.showPopover\(\);\s*place\(\);/, "placed as it opens");
    for (const listener of [/window\.addEventListener\("resize", place\)/, /window\.visualViewport\?\.addEventListener\("resize", place\)/,
      /window\.visualViewport\?\.addEventListener\("scroll", place\)/, /const scroll = [^]*?\n\s*place\(\);\n\s*\};/]) {
      assert.match(popover, listener, "placed again when the viewport moves");
    }
    const { applyPopoverPlacement } = await server.ssrLoadModule("/src/lib/popoverPlacement.ts");
    const px = (value) => {
      if (value === undefined) return Infinity;
      assert.match(value, /^-?\d+(\.\d+)?(e-?\d+)?px$/);
      return Number.parseFloat(value);
    };
    const panelOf = (natural) => {
      const style = {};
      return { style, getBoundingClientRect: () => ({ width: Math.min(natural.width, px(style.maxWidth)), height: Math.min(natural.height, px(style.maxHeight)) }) };
    };
    const views = [
      ...VIEWPORTS.map(([width, height]) => ({ left: 0, top: 0, width, height })),
      { left: 140, top: 260, width: 180, height: 300 }, // pinch-zoomed into a phone
    ];
    const sizes = [{ width: 320, height: 160 }, { width: 320, height: 640 }, { width: 2000, height: 2000 }];
    for (const view of views) {
      for (const fx of [0, 0.5, 0.97]) {
        for (const fy of [0, 0.3, 0.5, 0.9]) {
          const left = view.left + fx * (view.width - 24);
          const top = view.top + fy * (view.height - 20);
          const anchor = { left, top, bottom: top + 20 };
          for (const natural of sizes) {
            const panel = panelOf(natural);
            applyPopoverPlacement(panel, { getBoundingClientRect: () => anchor }, view);
            const at = { left: px(panel.style.left), top: px(panel.style.top) };
            const { width, height } = panel.getBoundingClientRect();
            const where = `${natural.width}x${natural.height} at (${left}, ${top}) in ${JSON.stringify(view)}`;
            assert.ok(at.left >= view.left + 8 && at.left + width <= view.left + view.width - 8 + 1e-9, `${where}: across`);
            assert.ok(at.top >= view.top + 8 && at.top + height <= view.top + view.height - 8 + 1e-9, `${where}: down`);
            assert.ok(height > 0, `${where}: some room`);
            assert.ok(at.top >= anchor.bottom || at.top + height <= anchor.top, `${where}: the token stays visible`);
            const below = view.top + view.height - anchor.bottom - 14;
            if (Math.min(natural.height, view.height - 16) <= below) assert.equal(at.top, anchor.bottom + 6, `${where}: opens below when it fits`);
          }
        }
      }
    }
  }
  console.log("Drawers make the page inert and leave inert; the drawer dialog and the token popover fit every viewport");
} finally { await server.close(); }
