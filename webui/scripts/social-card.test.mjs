import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { textOf } from "./markup.mjs";

// The link card (public/social/drowse.png) is a painting with the name set over it, as Ledger draws a card:
// scripts/social-card.mjs paints it (scripts/paint/harbour.py) and sets its type (scripts/social-card.html),
// both from scripts/social-card.json. It keeps the size the cards before it had; the PNG is the only file
// in social/.
const url = (path) => new URL(path, import.meta.url);
const png = readFileSync(url("../public/social/drowse.png"));
assert.equal(png.subarray(1, 4).toString(), "PNG");
assert.deepEqual([png.readUInt32BE(16), png.readUInt32BE(20)], [1200, 630], "a 1200 x 630 card");
assert.deepEqual(readdirSync(url("../public/social/")), ["drowse.png"]);

const html = readFileSync(url("./social-card.html"), "utf8");
const css = html.slice(html.indexOf("<style>"), html.indexOf("</style>"));
const body = html.slice(html.indexOf("<body>"), html.indexOf("<script>"));
const script = html.slice(html.indexOf("<script>"), html.indexOf("</script>"));
assert.match(html, /<html lang="en" data-theme="dark">/, "a painting is a dark surface");
assert.match(html, /href="\.\.\/node_modules\/@stdlogits\/ledger\/src\/styles\/card\.css"/, "Ledger's card type, which brings its faces and tokens");
// Art, wordmark, accents: the name set in type, and no mark tile (the painting holds the mark's figure).
assert.match(body, /<div class="name">drowse<\/div>/, "the name");
assert.doesNotMatch(body, /<svg|drowse-mark|<img(?! class="art")/, "no mark on the card");
// Colours only from the tokens (the accent mixed in oklch); nothing drawn here that card.css doesn't draw.
assert.doesNotMatch(css, /\bborder:|outline|box-shadow|#[0-9a-f]{3,6}\b|rgb|text-transform|monospace/i);
for (const mix of [...css.matchAll(/color-mix\(in (\w+)/g), ...script.matchAll(/color-mix\(in (\w+)/g)]) assert.equal(mix[1], "oklch");

// The readings are one file, which the painting and the type both read.
const card = JSON.parse(readFileSync(url("./social-card.json"), "utf8"));
const words = card.words.map(([word]) => word);
assert.equal(words.join(" "), "the harbour goes quiet as the last boats come in");
assert.ok(card.words.every(([, reading]) => reading >= 0 && reading <= 1), "each word's reading, 0 to 1");
assert.deepEqual(card.words.find(([, reading]) => reading === 1), [card.word, 1], "the word read layer by layer is the strongest");
assert.ok(card.reading.length > 1 && card.reading.every((reading) => reading >= 0 && reading <= 1), "a reading for each layer");
assert.ok(0 <= card.steered[0] && card.steered[0] <= card.steered[1] && card.steered[1] < card.reading.length, "the term's layers are the model's");
const draws = readFileSync(url("./social-card.mjs"), "utf8");
assert.match(draws, /paint\/harbour\.py/);
assert.match(draws, /social-card\.json/);
const painting = readFileSync(url("./paint/harbour.py"), "utf8");
assert.match(painting, /CARD\["reading"\]/, "the quay's reflections are the readings");
assert.match(painting, /CARD\["steered"\]/);
assert.doesNotMatch(painting, /rgb\(|#[0-9a-f]{6}\b/i, "its colours are Ledger's (dusk.py)");

const shown = `${textOf(body)} ${card.term.alpha} ${card.term.concept} ${words.join(" ")}`;
assert.doesNotMatch(shown, /[A-Z!?]|\.(?!\d)/, "no capitals and no sentences on the card (a coefficient's point aside)");
// The alt text drowse.ai publishes with it is recorded beside the drawing.
assert.match(html, /Alt text: "The name drowse over a painted harbour at dusk, with the steering term 0\.4 calm and a\s+reply, the harbour goes quiet as the last boats come in, whose words are tinted by how calm the\s+model reads them\. A lamp on the quay for each of the model's layers lays a reflection as long as\s+one word reads calm there, and a last boat comes in\."/);
console.log("social card: 1200 x 630, a painting with the name set over it, from scripts/social-card.json");
