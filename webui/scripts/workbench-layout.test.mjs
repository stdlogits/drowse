import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { cssRules, styleRules } from "./markup.mjs";

// The workbench on a phone, read from its CSS. drowse.ai's WebKit iPhone checks found two layouts
// that left no room:
// - a phone on its side (667x375, a touch phone) gave the conversation log 39 px between the Thread
//   row and the dock, the reply scrolled under the Thread toggle, which took every tap; at 844x390
//   (wider than the phone breakpoint) the Page | Inspector layout left the log 14 px;
// - the Loom page at 440x796 left the map 96 px from y=740 (the thread, open from its Map link, held
//   the column, and the loom's tools clipped the map), and 0 px at 844x390.
// A phone is narrow or short now; a short one lays the conversation out in two columns and lets the
// loom's page scroll to a map most of a screen tall; opening the loom folds the thread.
const src = new URL("../src/", import.meta.url);
const read = (path) => readFileSync(new URL(path, src), "utf8");
const PHONE = "@media (max-width: 720px), (max-height: 520px)";
const SHORT = "@media (max-height: 520px)";

/** The rule in ``rules`` holding ``selector`` with ``prop``, or a failure naming them. */
function rule(rules, selector, prop) {
  const found = rules.filter(r => r.selectors.includes(selector) && prop in r.decls);
  assert.ok(found.length > 0, `no ${selector} { ${prop} }`);
  return found.at(-1);
}
/** A length at a viewport ``height``: px, dvh, calc() and min() (what these rules use). */
function at(text, height) {
  const expr = text.replace(/calc\(/g, "(").replace(/min\(/g, "Math.min(").replace(/max\(/g, "Math.max(")
    .replace(/([\d.]+)dvh/g, (_, n) => String(Number(n) * height / 100)).replace(/([\d.]+)px/g, "$1");
  assert.match(expr, /^(?:[\d.\s()+\-*/,]|Math\.(?:min|max))+$/, `unexpected length ${text}`);
  return Function(`return ${expr}`)();
}

// ---- one phone: narrow, or short (a phone on its side) --------------------------------------------
const app = read("App.svelte");
assert.match(app, /window\.matchMedia\("\(max-width: 720px\), \(max-height: 520px\)"\)/,
  "App's phone layout (rooms in the menu, one workbench column, sheet dialogs) covers short screens");
for (const [file, selector, prop] of [
  ["App.svelte", ".dw-top", "flex-wrap"],
  ["lib/style/drowse.css", ".dw-main", "padding-left"],
  ["panels/Chat.svelte", ".log", "padding"],
  ["panels/Chat.svelte", ".extras.folded", "display"],
  ["panels/RawBuffer.svelte", ".raw-buffer", "padding"],
  ["panels/loom/LoomSidebar.svelte", ".loom-sidebar", "min-height"],
]) {
  const rules = file.endsWith(".css") ? cssRules(read(file)) : styleRules(read(file));
  assert.equal(rule(rules, selector, prop).media, PHONE, `${file} styles the phone App lays out`);
}

// ---- a phone on its side: the conversation and the dock side by side -----------------------------
const workbench = styleRules(read("panels/workbench/Workbench.svelte"));
const short = workbench.filter(r => r.media === SHORT);
const decl = (selector, prop) => rule(short, selector, prop).decls[prop];
const grid = ".phone:not(.loom-page) .dw-main";
assert.equal(decl(grid, "display"), "grid");
const tracks = decl(grid, "grid-template-rows").split(/\s+(?![^(]*\))/);
assert.deepEqual(tracks, ["auto", "auto", "minmax(0, 1fr)", "auto"]);
const place = (selector) => ({ column: decl(selector, "grid-column"), row: decl(selector, "grid-row") });
assert.deepEqual(place(".phone:not(.loom-page) .thread-fold"), { column: "1", row: "1" });
assert.deepEqual(place(".phone:not(.loom-page) .dw-thread"), { column: "1", row: "2" });
assert.deepEqual(place(".phone :global(.chat .log)"), { column: "1", row: "3 / 5" }, "the log runs to the foot");
assert.deepEqual(place(".phone :global(.chat .dock)"), { column: "2", row: "4" }, "the dock sits beside it");
assert.deepEqual(place(".phone .sheet"), { column: "2", row: "1 / -1" }, "the inspector takes the side column");
assert.equal(rule(short, ".phone.sheet-shown :global(.chat .dock)", "display").decls.display, "none",
  "while it shows, in place of the dock");
assert.equal(rule(short, ".phone.sheet-shown :global(.chat .raw-host)", "grid-column").decls["grid-column"], "1",
  "raw completion spans both columns until the inspector takes the second");
assert.equal(decl(".phone .sheet", "height"), "auto");

// The rows at three phones on their side, with the chrome WebKit measured at 667x375: header 76,
// the Thread row 52, the dock 224 in its column (it was 208 under the log, which left the log 39).
const HEADER = 76, THREAD_ROW = 52, DOCK = 224;
for (const [width, height] of [[667, 375], [844, 390], [932, 430]]) {
  const column = height - HEADER;
  const rows = [THREAD_ROW, 0, null, DOCK];
  rows[2] = Math.max(0, column - rows[0] - rows[1] - rows[3]);
  const log = rows[2] + rows[3];
  assert.ok(log >= 200, `${width}x${height}: the log is ${log} px`);
  assert.equal(log, column - THREAD_ROW, `${width}x${height}: the log takes the column under the Thread row`);
  assert.ok(HEADER + THREAD_ROW <= height - log, "nothing sits over the log: it starts under the Thread row");
}

// ---- the loom on a phone --------------------------------------------------------------------------
const source = read("panels/workbench/Workbench.svelte");
assert.match(source, /if \(phone && loomPage\) untrack\(\(\) => \(threadOpen = false\)\);/,
  "opening the loom on a phone folds the thread (its Map link opened it over the map)");
assert.match(source, /class:loom-page=\{loomPage\}/);
const loom = styleRules(read("panels/loom/LoomSidebar.svelte"));
const sidebar = rule(loom, ".loom-sidebar", "min-height");
assert.equal(sidebar.decls.height, "auto", "the column grows with the tools and the map instead of clipping the map");
assert.equal(sidebar.decls["min-height"], "100%");
assert.equal(rule(loom, ".loom-sidebar", "overflow-y").decls["overflow-y"], "clip", "(the sidebar clips: it must grow)");
const mapMin = rule(loom, ".tree-scroll", "min-height");
assert.equal(mapMin.media, PHONE);
assert.equal(at(mapMin.decls["min-height"], 796), 240, "a portrait phone's map keeps at least 240 px");
assert.ok(at(mapMin.decls["min-height"], 568) >= 240);

// A phone on its side: the loom's page scrolls to a map most of a screen tall, with a strip above it
// to scroll back by (the map takes its own drags).
assert.equal(decl(".phone.loom-page", "height"), "auto");
assert.equal(decl(".phone.loom-page", "overflow"), "visible");
assert.equal(decl(".phone.loom-page .loom", "overflow"), "visible");
for (const height of [375, 390, 430]) {
  const map = at(decl(".phone.loom-page :global(.tree-scroll)", "min-height"), height);
  assert.ok(map >= 0.75 * height && height - map >= 44, `${height} px tall: a ${map} px map and a strip to scroll by`);
}
console.log("workbench on a phone: a short one in two columns, the loom's map with room at every size");
