#!/usr/bin/env node
// Draws the link card, public/social/drowse.png (1200 x 630), from scripts/social-card.html in light
// colours. drowse.ai copies it into its public-hosted/social and derives its jpg there. Run it when
// the card or the mark changes; Chromium comes from Playwright (`npx playwright install chromium`).
//
//   node scripts/social-card.mjs

import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const card = new URL("./social-card.html", import.meta.url);
const out = fileURLToPath(new URL("../public/social/drowse.png", import.meta.url));

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1, colorScheme: "light" });
  await page.goto(card.href);
  await page.evaluate(() => document.fonts.ready);
  const missing = await page.evaluate(() => ["600 32px 'Wix Madefor Display'", "400 44px 'Wix Madefor Text'"].filter(font => !document.fonts.check(font)));
  if (missing.length) throw new Error(`fonts did not load: ${missing.join(", ")}`);
  await writeFile(out, await page.screenshot({ type: "png" }));
  console.log(`wrote ${out}`);
} finally {
  await browser.close();
}
