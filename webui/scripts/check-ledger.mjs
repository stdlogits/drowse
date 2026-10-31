// The Ledger ratchet. Counts, per source file, the uses of the bridge's old token names
// (src/lib/style/bridge.css) and the departures from Ledger (DESIGN.md in @stdlogits/ledger: flat
// and borderless, the tip's frosted shadow only, Wix Madefor only, no capitals, one curve, colours
// only from tokens), and fails if any count rises above webui/ledger-ratchet.json. The committed
// ratchet must equal today's counts, and counts only go down:
//
//   node scripts/check-ledger.mjs           check (npm run check)
//   node scripts/check-ledger.mjs --write   lower the ratchet to today's counts (never raises one)
//   node scripts/check-ledger.mjs --cut     the 6.0 cut: every count zero and the bridge deleted
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
  shadow: "shadows other than the tip's --float or a marked token's 2px same-colour spread",
  mono: "monospace",
  uppercase: "text-transform: uppercase or small capitals",
  tracking: "letter-spaced text (positive letter-spacing)",
  easing: "easing other than Ledger's one curve",
  colour: "colours written outside the tokens",
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
  for (const m of text.matchAll(/\sstyle(?::[\w-]+)?=("[^"]*"|\{[^}]*\})/g)) regions.push(m[1]);
  return regions;
}

const declarations = (css, property) =>
  [...css.matchAll(new RegExp(`(?<![-\\w])(${property})\\s*:\\s*([^;}{]+)`, "g"))].map(m => ({ property: m[1], value: m[2].trim() }));

function countFile(text, ext, bridgeNames) {
  const code = stripComments(text, ext);
  const css = styleRegions(code, ext).join("\n");
  const counts = Object.fromEntries(Object.keys(CATEGORIES).map(key => [key, 0]));

  for (const m of code.matchAll(/var\(\s*(--[\w-]+)/g)) if (bridgeNames.has(m[1])) counts.bridge += 1;

  const borderProps = "border(?:-(?:top|right|bottom|left|block|inline|block-start|block-end|inline-start|inline-end))?(?:-(?:width|style|color))?";
  for (const { value } of declarations(css, borderProps)) {
    const plain = value.replace(/\s*!important$/, "");
    if (/^(0|0px|none|none 0|0 none)$/.test(plain) || SYSTEM_COLOURS.test(plain)) continue;
    counts.border += 1;
  }

  for (const { value } of declarations(css, "box-shadow|text-shadow")) {
    const plain = value.replace(/\s*!important$/, "");
    if (plain === "none" || plain === "var(--float)" || SYSTEM_COLOURS.test(plain)) continue;
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
  if (existsSync(bridgePath)) {
    for (const m of (await readFile(bridgePath, "utf8")).matchAll(/(--[\w-]+)\s*:/g)) bridgeNames.add(m[1]);
  }
  const files = {};
  for (const path of await sourceFiles(src)) {
    const rel = relative(src, path).split("\\").join("/");
    if (SKIP.has(rel)) continue;
    const counts = countFile(await readFile(path, "utf8"), extname(path), bridgeNames);
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

function compare(current, ratchet) {
  const rises = [];
  const falls = [];
  const names = new Set([...Object.keys(current), ...Object.keys(ratchet)]);
  for (const file of [...names].sort()) {
    for (const key of Object.keys(CATEGORIES)) {
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

if (mode === "cut") {
  const left = Object.values(sum).reduce((a, b) => a + b, 0);
  if (left > 0 || current.bridge) {
    console.error(`Ledger cut not ready: ${summary}${current.bridge ? "; src/lib/style/bridge.css still exists" : ""}.`);
    process.exit(1);
  }
  console.log("Ledger cut ready: no bridge, no departures.");
} else if (mode === "write") {
  if (saved) {
    const { rises } = compare(current.files, saved.files);
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
