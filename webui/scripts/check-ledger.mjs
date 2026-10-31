// The Ledger ratchet. Counts, per source file, the uses of the bridge's old token names
// (src/lib/style/bridge.css) and the departures from Ledger (DESIGN.md in @stdlogits/ledger: flat
// and borderless, the one shadow under the tip (which lives in Ledger, not here), Wix Madefor
// only, no capitals, one curve, nothing looping, colours only from tokens and mixed in oklch, the
// depth accent only as a fill), and fails if any count rises above webui/ledger-ratchet.json. The
// committed ratchet must equal today's counts, and counts only go down:
//
//   node scripts/check-ledger.mjs           check (npm run check)
//   node scripts/check-ledger.mjs --write   lower the ratchet to today's counts (never raises one)
//   node scripts/check-ledger.mjs --cut     the 6.0 cut: every count zero and the bridge deleted
//
// A category the committed ratchet does not list yet is new: check fails until --write records
// it, and --write takes today's count as its first ceiling. After that it only goes down.
import { existsSync } from "node:fs";
import { readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const webui = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(webui, "src");
const bridgePath = join(src, "lib/style/bridge.css");
const ratchetPath = join(webui, "ledger-ratchet.json");
const mode = process.argv.includes("--write") ? "write" : process.argv.includes("--cut") ? "cut" : "check";

export const CATEGORIES = {
  bridge: "uses of a bridge token (an old name mapped onto Ledger)",
  border: "borders other than 0 or none",
  outline: "outlines other than a focus ring (outside :focus and :focus-visible rules)",
  shadow: "shadows other than a marked token's 2px same-colour spread (the tip's --float is Ledger's own)",
  mono: "monospace",
  uppercase: "text-transform: uppercase or small capitals",
  tracking: "letter-spaced text (positive letter-spacing)",
  easing: "easing other than Ledger's one curve",
  loop: "animations that loop (infinite)",
  colour: "colours written outside the tokens",
  mix: "colour mixes outside oklch and oklab (the accent mixes in oklch; a grey mixes in oklab, since Chrome gives a near-grey's powerless oklch hue a red cast)",
  tintText: "text coloured by the depth accent (--tint, --deep, --late or a bridge name for them)",
};

const SKIP = new Set(["lib/types.gen.ts", "lib/style/bridge.css"]);
const SYSTEM_COLOURS = /\b(Canvas|CanvasText|Highlight|HighlightText|ButtonText|ButtonFace|ButtonBorder|GrayText|LinkText|Field|FieldText|Mark|MarkText)\b/;
const NAMED_COLOURS = /(?<![-\w])(white|black|red|green|blue|gray|grey|yellow|orange|purple|pink|silver|navy|teal|olive|maroon)(?![-\w])/g;

async function sourceFiles(directory) {
  const out = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) out.push(...await sourceFiles(path));
    else if ([".css", ".svelte", ".ts"].includes(extname(entry.name))) out.push(path);
  }
  return out.sort();
}

function stripComments(text, ext) {
  let out = text.replace(/\/\*[\s\S]*?\*\//g, " ");
  if (ext === ".svelte") out = out.replace(/<!--[\s\S]*?-->/g, " ");
  if (ext !== ".css") out = out.replace(/(^|[^:"'`\\])\/\/[^\n]*/g, "$1");
  return out;
}

/** The parts of a file that are CSS: a stylesheet, a component's <style>, and style attributes. */
function styleRegions(text, ext) {
  if (ext === ".css") return [text];
  if (ext !== ".svelte") return [];
  const regions = [...text.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)].map(m => m[1]);
  // A style directive names its property outside the value (style:border={…}), so it is rebuilt
  // as a declaration; a style attribute is already one.
  for (const m of text.matchAll(/\sstyle(?::([\w-]+))?=("[^"]*"|\{[^}]*\})/g)) {
    regions.push(m[1] ? `${m[1]}: ${m[2].replace(/^["{]|["}]$/g, "")};` : m[2]);
  }
  return regions;
}

const declarations = (css, property) =>
  [...css.matchAll(new RegExp(`(?<![-\\w])(${property})\\s*:\\s*([^;}{]+)`, "g"))].map(m => ({ property: m[1], value: m[2].trim() }));

/** Each innermost rule's selector and body (an at-rule's own braces never match, its rules do).
 *  A style attribute's declarations come back with no selector. */
function rules(css) {
  const out = [];
  let rest = css;
  for (const m of css.matchAll(/([^{}]*)\{([^{}]*)\}/g)) {
    out.push({ selector: m[1].trim(), body: m[2] });
    rest = rest.replace(m[0], " ");
  }
  out.push({ selector: "", body: rest });
  return out;
}

/** A selector list split at its top-level commas (not those inside :is(…) and the like). */
function selectorParts(selector) {
  const parts = [""];
  let depth = 0;
  for (const char of selector) {
    if (char === "(") depth += 1;
    if (char === ")") depth -= 1;
    if (char === "," && depth === 0) parts.push("");
    else parts[parts.length - 1] += char;
  }
  return parts;
}

const isFocusRule = selector =>
  selector !== "" && selectorParts(selector).every(part => /:focus(-visible|-within)?\b/.test(part));

function countFile(text, ext, bridgeNames, tintNames) {
  const code = stripComments(text, ext);
  const css = styleRegions(code, ext).join("\n");
  const counts = Object.fromEntries(Object.keys(CATEGORIES).map(key => [key, 0]));

  for (const m of code.matchAll(/var\(\s*(--[\w-]+)/g)) if (bridgeNames.has(m[1])) counts.bridge += 1;

  for (const { selector, body } of rules(css)) {
    if (isFocusRule(selector)) continue;
    for (const { value } of declarations(body, "outline(?:-(?:width|style|color))?")) {
      const plain = value.replace(/\s*!important$/, "");
      if (/^(0|0px|none|none 0|0 none)$/.test(plain) || SYSTEM_COLOURS.test(plain)) continue;
      counts.outline += 1;
    }
  }

  // A component's own accent property (--card-accent, --family, …) is set from the depth accent by
  // its callers, so text coloured through one counts too.
  const ownAccent = name => !bridgeNames.has(name) && /^--(?:[\w-]*-)?(?:accent|tint|family|hue)(?:-[\w-]*)?$/.test(name);
  for (const { value } of declarations(css, "color")) {
    const names = [...value.matchAll(/var\(\s*(--[\w-]+)/g)].map(m => m[1]);
    if (names.some(name => tintNames.has(name) || ownAccent(name))) counts.tintText += 1;
  }

  counts.mix += [...code.matchAll(/color-mix\(\s*in\s+(?!(?:oklch|oklab)\b)[\w-]+/g)].length;

  for (const { value } of declarations(css, "animation|animation-iteration-count")) {
    if (/(?<![-\w])infinite(?![-\w])/.test(value)) counts.loop += 1;
  }
  counts.loop += [...code.matchAll(/iterations\s*:\s*Infinity\b/g)].length;

  const borderProps = "border(?:-(?:top|right|bottom|left|block|inline|block-start|block-end|inline-start|inline-end))?(?:-(?:width|style|color))?";
  for (const { value } of declarations(css, borderProps)) {
    const plain = value.replace(/\s*!important$/, "");
    if (/^(0|0px|none|none 0|0 none)$/.test(plain) || SYSTEM_COLOURS.test(plain)) continue;
    counts.border += 1;
  }

  for (const { value } of declarations(css, "box-shadow|text-shadow")) {
    const plain = value.replace(/\s*!important$/, "");
    if (plain === "none" || SYSTEM_COLOURS.test(plain)) continue;
    if (/^0 0 0 2px \S.*$/.test(plain) && !plain.includes(",")) continue;
    counts.shadow += 1;
  }

  counts.mono += [...code.matchAll(/(?<![-\w])(monospace|ui-monospace)(?![-\w])|Martian Mono|var\(\s*--font-(?:mono|data)\b/gi)].length;

  for (const { value } of declarations(css, "text-transform")) if (/uppercase/.test(value)) counts.uppercase += 1;
  for (const { value } of declarations(css, "font-variant|font-variant-caps")) if (/small-caps|all-small-caps/.test(value)) counts.uppercase += 1;

  for (const { value } of declarations(css, "letter-spacing")) {
    const number = Number.parseFloat(value);
    if (Number.isFinite(number) && number > 0) counts.tracking += 1;
  }

  for (const m of code.matchAll(/cubic-bezier\(([^)]*)\)/g)) {
    const points = m[1].split(",").map(part => Number.parseFloat(part));
    const ledger = [0.2, 0, 0, 1];
    if (!points.every((point, i) => Math.abs(point - ledger[i]) < 1e-9)) counts.easing += 1;
  }
  for (const { value } of declarations(css, "transition|transition-timing-function|animation|animation-timing-function")) {
    counts.easing += [...value.matchAll(/(?<![-\w])(ease|ease-in|ease-out|ease-in-out|linear|step-start|step-end|steps\()(?![-\w])/g)].length;
  }
  counts.easing += [...code.matchAll(/from\s+["']svelte\/easing["']/g)].length;

  counts.colour += [...code.matchAll(/(?<![&\w])#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})(?![\w-])/g)].length;
  counts.colour += [...code.matchAll(/(?<![-\w])(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch)\(/g)].length;
  counts.colour += [...css.matchAll(NAMED_COLOURS)].length;
  return counts;
}

async function measure() {
  const bridgeNames = new Set();
  // The depth accent's names: Ledger's three, and any bridge name whose value reads one of them.
  const tintNames = new Set(["--tint", "--deep", "--late"]);
  if (existsSync(bridgePath)) {
    const bridge = stripComments(await readFile(bridgePath, "utf8"), ".css");
    for (const m of bridge.matchAll(/(--[\w-]+)\s*:\s*([^;]*);/g)) {
      bridgeNames.add(m[1]);
      if (/var\(\s*--(tint|deep|late)\b/.test(m[2])) tintNames.add(m[1]);
    }
  }
  const files = {};
  for (const path of await sourceFiles(src)) {
    const rel = relative(src, path).split("\\").join("/");
    if (SKIP.has(rel)) continue;
    const counts = countFile(await readFile(path, "utf8"), extname(path), bridgeNames, tintNames);
    const nonzero = Object.fromEntries(Object.entries(counts).filter(([, n]) => n > 0));
    if (Object.keys(nonzero).length) files[`src/${rel}`] = nonzero;
  }
  return { files, bridge: existsSync(bridgePath) };
}

function totals(files) {
  const sum = Object.fromEntries(Object.keys(CATEGORIES).map(key => [key, 0]));
  for (const counts of Object.values(files)) for (const [key, n] of Object.entries(counts)) sum[key] += n;
  return sum;
}

function compare(current, ratchet, keys = Object.keys(CATEGORIES)) {
  const rises = [];
  const falls = [];
  const names = new Set([...Object.keys(current), ...Object.keys(ratchet)]);
  for (const file of [...names].sort()) {
    for (const key of keys) {
      const now = current[file]?.[key] ?? 0;
      const allowed = ratchet[file]?.[key] ?? 0;
      if (now > allowed) rises.push(`  ${file}: ${key} ${allowed} → ${now} (${CATEGORIES[key]})`);
      else if (now < allowed) falls.push(`  ${file}: ${key} ${allowed} → ${now}`);
    }
  }
  return { rises, falls };
}

const current = await measure();
const sum = totals(current.files);
const summary = Object.entries(sum).map(([key, n]) => `${key} ${n}`).join(", ");
const saved = existsSync(ratchetPath) ? JSON.parse(await readFile(ratchetPath, "utf8")) : null;
// Categories the committed ratchet already records; any other is new and has no ceiling yet.
const recorded = Object.keys(CATEGORIES).filter(key => key in (saved?.categories ?? {}));
const unrecorded = Object.keys(CATEGORIES).filter(key => !recorded.includes(key));

if (mode === "cut") {
  const left = Object.values(sum).reduce((a, b) => a + b, 0);
  if (left > 0 || current.bridge) {
    console.error(`Ledger cut not ready: ${summary}${current.bridge ? "; src/lib/style/bridge.css still exists" : ""}.`);
    process.exit(1);
  }
  console.log("Ledger cut ready: no bridge, no departures.");
} else if (mode === "write") {
  if (saved) {
    const { rises } = compare(current.files, saved.files, recorded);
    if (rises.length) {
      console.error(`The Ledger ratchet only goes down; these rose:\n${rises.join("\n")}`);
      process.exit(1);
    }
  }
  const body = {
    about: "Ledger ratchet: per-file counts that may only go down (scripts/check-ledger.mjs). The 6.0 cut needs every count at zero and src/lib/style/bridge.css deleted.",
    categories: CATEGORIES,
    totals: sum,
    files: current.files,
  };
  await writeFile(ratchetPath, `${JSON.stringify(body, null, 2)}\n`);
  console.log(`Ledger ratchet written: ${summary}.`);
} else {
  if (!saved) {
    console.error("webui/ledger-ratchet.json is missing: run node scripts/check-ledger.mjs --write.");
    process.exit(1);
  }
  if (unrecorded.length) {
    console.error(`Ledger check: the ratchet does not record ${unrecorded.join(", ")} yet. Record today's counts with node scripts/check-ledger.mjs --write.`);
    process.exit(1);
  }
  const { rises, falls } = compare(current.files, saved.files);
  if (rises.length) {
    console.error(`Ledger check failed: counts rose above the ratchet.\n${rises.join("\n")}\nWrite it in Ledger (DESIGN.md in @stdlogits/ledger) instead.`);
    process.exit(1);
  }
  if (falls.length) {
    console.error(`Ledger check: counts fell below the ratchet. Lower it with node scripts/check-ledger.mjs --write:\n${falls.join("\n")}`);
    process.exit(1);
  }
  console.log(`Ledger ratchet holds: ${summary}.`);
}
