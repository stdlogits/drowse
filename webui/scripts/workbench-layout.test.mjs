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
// A phone is narrow or short now; a short one at least 480 px wide lays the conversation out in two
// columns (narrower, two would leave the log a 44 px column), any short one lets the loom's page
// scroll to a map most of a screen tall, and opening the loom folds the thread.
const src = new URL("../src/", import.meta.url);
const read = (path) => readFileSync(new URL(path, src), "utf8");
const PHONE = "@media (max-width: 720px), (max-height: 520px)";
const SIDE_BY_SIDE = "@media (max-height: 520px) and (min-width: 480px)";
const SHORT = "@media (max-height: 520px)";

/** Whether a media head of ``(max|min)-(width|height): Npx`` terms (``,`` = or, ``and``) matches. */
function matches(media, width, height) {
  const size = { width, height };
  return media.replace(/^@media\s+/, "").split(",").some(query => query.split(/\band\b/).every(term => {
    const m = term.trim().match(/^\((max|min)-(width|height): (\d+)px\)$/);
    assert.ok(m, `unexpected media term ${term}`);
    return m[1] === "max" ? size[m[2]] <= Number(m[3]) : size[m[2]] >= Number(m[3]);
  }));
}

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
// Which screens are phones, and which of them put the dock beside the log. A short desktop window
// (1280x500) is a phone too: rooms move into the menu (a decision a9 signs off). A short portrait
// screen (320x480) is a phone in one column: two would leave the log 12 px of text.
for (const [size, phone, sideBySide] of [
  [[375, 667], true, false], [[320, 480], true, false], [[480, 320], true, true], [[568, 320], true, true],
  [[667, 375], true, true], [[844, 390], true, true], [[932, 430], true, true], [[1280, 500], true, true],
  [[1024, 768], false, false], [[1280, 800], false, false],
]) {
  assert.equal(matches(PHONE, ...size), phone, `${size.join("x")} is ${phone ? "" : "not "}a phone`);
  assert.equal(matches(SIDE_BY_SIDE, ...size), sideBySide,
    `${size.join("x")} ${sideBySide ? "puts" : "does not put"} the dock beside the log`);
}

const workbench = styleRules(read("panels/workbench/Workbench.svelte"));
const sideBySide = workbench.filter(r => r.media === SIDE_BY_SIDE);
const short = workbench.filter(r => r.media === SHORT);
const decl = (rules, selector, prop) => rule(rules, selector, prop).decls[prop];
const grid = ".phone:not(.loom-page) .dw-main";
assert.equal(decl(sideBySide, grid, "display"), "grid");
const tracks = (text) => text.split(/\s+(?![^(]*\))/);
const rowTracks = tracks(decl(sideBySide, grid, "grid-template-rows"));
assert.deepEqual(rowTracks, ["auto", "auto", "minmax(0, 1fr)"]);
const place = (selector) => ({
  column: decl(sideBySide, selector, "grid-column"), row: decl(sideBySide, selector, "grid-row"),
});
assert.deepEqual(place(".phone:not(.loom-page) .thread-fold"), { column: "1", row: "1" });
assert.deepEqual(place(".phone:not(.loom-page) .dw-thread"), { column: "1", row: "2" });
assert.deepEqual(place(".phone :global(.chat .dock)"), { column: "2", row: "1 / -1" },
  "the dock has the side column to itself: the Thread row does not push it down");
assert.equal(decl(sideBySide, ".phone :global(.chat .dock)", "align-self"), "end", "at its foot");
assert.deepEqual(place(".phone .sheet"), { column: "2", row: "1 / -1" }, "the inspector takes the side column");
assert.equal(decl(sideBySide, ".phone.sheet-shown :global(.chat .dock)", "display"), "none",
  "while it shows, in place of the dock");
assert.equal(decl(sideBySide, ".phone.sheet-shown :global(.chat .raw-host)", "grid-column"), "1",
  "raw completion spans both columns until the inspector takes the second");
assert.equal(decl(sideBySide, ".phone .sheet", "height"), "auto");

// The log's box at each short screen that lays it beside the dock, from the CSS: its height from
// the grid's row tracks and its placement, its width from the column template, the gap token and the
// log's padding. The chrome WebKit measured (iPhone emulation, a reply shown): the header 76, the
// Thread row 52 (100 at 480 wide, where it wraps beside the Chat | Raw switch; the thread folded),
// the dock 224 in its column. In one column at 667x375 the dock (208) left the log 39.
const HEADER = 76, THREAD = 0, DOCK = 224;
const threadRow = (width) => (width < 560 ? 100 : 52);
const s4 = Number(read("../node_modules/@stdlogits/ledger/src/styles/tokens.css")
  .match(/--s4:\s*(\d+)px/)[1]);
const px = (text) => Number(text.replace("var(--s4)", `${s4}px`).replace(/px$/, ""));
const logPadding = decl(styleRules(read("panels/Chat.svelte")).filter(r => r.media === PHONE), ".log", "padding")
  .split(/\s+/).map(px);
const [, side] = tracks(decl(sideBySide, grid, "grid-template-columns"));
const sideAt = (width) => {
  const [lo, pct, hi] = side.match(/^clamp\((\d+)px, (\d+)%, (\d+)px\)$/).slice(1).map(Number);
  return Math.min(hi, Math.max(lo, width * pct / 100));
};
const gap = px(decl(sideBySide, grid, "column-gap"));
const logPlace = place(".phone :global(.chat .log)");
assert.equal(logPlace.column, "1");
const [rowStart, rowEnd = rowStart + 1] = logPlace.row.split("/").map(Number);
// Measured log boxes (height x text width): 480x320 144x172, 568x320 192x260, 667x375 247x352,
// 1280x500 380x872; the dock fit its column at each (at 568x320 it ran 32 px past the foot while
// it sat under the Thread row).
for (const [width, height, minLog] of [
  [480, 320, 140], [568, 320, 190], [667, 375, 240], [844, 390, 260], [932, 430, 300], [1280, 500, 370],
]) {
  assert.ok(matches(SIDE_BY_SIDE, width, height));
  const column = height - HEADER;
  const rows = [threadRow(width), THREAD, null];
  rows[2] = Math.max(0, column - rows[0] - rows[1]);
  assert.equal(rowTracks[2], "minmax(0, 1fr)", "the log's row takes what the Thread row leaves");
  const logHeight = rows.slice(rowStart - 1, rowEnd - 1).reduce((a, b) => a + b, 0);
  const logText = width - sideAt(width) - gap - logPadding[1] * 2;
  assert.ok(logHeight >= minLog, `${width}x${height}: the log is ${logHeight} px tall`);
  assert.ok(logText >= 170, `${width}x${height}: the log keeps ${logText} px of text`);
  // What sits above the dock in its column: the auto rows before its start line (the log's row
  // shrinks to nothing first).
  const dockStart = Number(place(".phone :global(.chat .dock)").row.split("/")[0]);
  const above = rows.slice(0, dockStart - 1).reduce((a, b, i) => a + (rowTracks[i] === "auto" ? b : 0), 0);
  assert.ok(above + DOCK <= column, `${width}x${height}: the dock fits its column (${above} px above it)`);
}
// Just under the width floor two columns would cramp the log, and one column is what it gets.
assert.ok(479 - sideAt(479) - gap - logPadding[1] * 2 < 172);
assert.ok(!matches(SIDE_BY_SIDE, 479, 320) && !matches(SIDE_BY_SIDE, 320, 480));

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
assert.equal(decl(short, ".phone.loom-page", "height"), "auto");
assert.equal(decl(short, ".phone.loom-page", "overflow"), "visible");
assert.equal(decl(short, ".phone.loom-page .loom", "overflow"), "visible");
for (const height of [375, 390, 430]) {
  const map = at(decl(short, ".phone.loom-page :global(.tree-scroll)", "min-height"), height);
  assert.ok(map >= 0.75 * height && height - map >= 44, `${height} px tall: a ${map} px map and a strip to scroll by`);
}
console.log("workbench on a phone: a short one in two columns, the loom's map with room at every size");
