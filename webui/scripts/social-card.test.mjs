import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { textOf } from "./markup.mjs";

// The link card (public/social/drowse.png) is drawn in Ledger from scripts/social-card.html by
// scripts/social-card.mjs: the dashboard in miniature, not the 5.x shader orb. It keeps the size the
// old card had; the PNG is the only file in social/ (the orb's SVG source is gone).
const url = (path) => new URL(path, import.meta.url);
const png = readFileSync(url("../public/social/drowse.png"));
assert.equal(png.subarray(1, 4).toString(), "PNG");
assert.deepEqual([png.readUInt32BE(16), png.readUInt32BE(20)], [1200, 630], "a 1200 x 630 card");
assert.deepEqual(readdirSync(url("../public/social/")), ["drowse.png"]);

const html = readFileSync(url("./social-card.html"), "utf8");
const css = html.slice(html.indexOf("<style>"), html.indexOf("</style>"));
const script = html.slice(html.indexOf("<script>"), html.indexOf("</script>"));
assert.match(html, /href="\.\.\/node_modules\/@stdlogits\/ledger\/src\/styles\/fonts\.css"/, "Ledger's faces");
assert.match(html, /href="\.\.\/node_modules\/@stdlogits\/ledger\/src\/styles\/tokens\.css"/, "and tokens");
assert.match(html, /<img src="\.\.\/public\/drowse-mark-light\.svg" alt="">drowse<\/div>/, "the mark and the name");
// Flat, one accent, no sentences: no borders, outlines or shadows but a marked token's 2 px spread
// in its own colour, colours only from the tokens (the accent mixed in oklch), no capitals.
assert.doesNotMatch(css, /\bborder:|outline|#[0-9a-f]{3,6}\b|rgb|text-transform|monospace/i);
for (const shadow of css.matchAll(/box-shadow:([^;]*);/g)) {
  assert.match(shadow[1], /^\s*0 -2px 0 0 (color-mix\(.+\)), 0 2px 0 0 \1\s*$/, "only the marked token's spread");
}
for (const mix of [...css.matchAll(/color-mix\(in (\w+)/g), ...script.matchAll(/color-mix\(in (\w+)/g)]) assert.equal(mix[1], "oklch");
const words = [...script.matchAll(/\["(\w+)", [\d.]+\]/g)].map(match => match[1]);
assert.equal(words.join(" "), "the harbour goes quiet as the last boats come in");
const shown = `${textOf(html.slice(html.indexOf("<body>"), html.indexOf("<script>")))} ${words.join(" ")}`;
assert.doesNotMatch(shown, /[A-Z!?]|\.(?!\d)/, "no capitals and no sentences on the card (a coefficient's point aside)");
// The alt text drowse.ai publishes with it is recorded beside the drawing.
assert.match(html, /Alt text: "The drowse mark and name, the steering term 0\.4 calm, a reply whose words are\s+tinted by how calm the model reads them, and one word's calm reading across the model's\s+layers\."/);
console.log("social card: 1200 x 630, drawn in Ledger from scripts/social-card.html");
