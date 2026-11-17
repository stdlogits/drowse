#!/usr/bin/env node
// Draws the link card, public/social/drowse.png (1200 x 630): a painting with the name set over it, as
// Ledger draws a card. scripts/paint/harbour.py paints it and scripts/social-card.html sets the type, both
// from scripts/social-card.json (the card's readings). drowse.ai copies the card into its
// public-hosted/social and derives its jpg there. Run it when the card changes. Painting needs Python
// with numpy, scipy and Pillow (PYTHON, or python3); Chromium comes from Playwright
// (`npx playwright install chromium`).
//
//   node scripts/social-card.mjs

import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { chromium } from "@playwright/test";

const here = (path) => fileURLToPath(new URL(path, import.meta.url));
const card = new URL("./social-card.html", import.meta.url);
const readings = here("./social-card.json");
const out = here("../public/social/drowse.png");

const folder = await mkdtemp(join(tmpdir(), "drowse-card-"));
const browser = await chromium.launch();
try {
  // The painting, at twice the card's size and reduced.
  const art = join(folder, "harbour.png");
  const { stdout } = await promisify(execFile)(process.env.PYTHON ?? "python3", [here("./paint/harbour.py"), readings, "2", art]);
  process.stdout.write(stdout);

  const page = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1, colorScheme: "dark" });
  await page.goto(card.href);
  await page.evaluate((data) => window.draw(data), { art: pathToFileURL(art).href, ...JSON.parse(await readFile(readings, "utf8")) });
  await page.evaluate(() => document.fonts.ready);
  const missing = await page.evaluate(() => ["600 80px 'Wix Madefor Display'", "400 40px 'Wix Madefor Text'"].filter(font => !document.fonts.check(font)));
  if (missing.length) throw new Error(`fonts did not load: ${missing.join(", ")}`);
  await writeFile(out, await page.screenshot({ type: "png" }));
  console.log(`wrote ${out}`);
} finally {
  await browser.close();
  await rm(folder, { recursive: true });
}
