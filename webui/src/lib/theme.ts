import { createTheme, type Choice } from "@stdlogits/ledger";
import { migrateLegacyStorageItem } from "./brandMigration";

// The dashboard's theme: Ledger's auto → the system's opposite → the system's own look.
// theme-preflight.js (served from the package, see vite.config.ts) applies a stored choice before
// first paint under the same key; createTheme owns it after.
export const THEME_STORAGE_KEY = "drowse.theme";

export const theme = createTheme({ key: THEME_STORAGE_KEY });

/** Carries a choice stored under the earlier brand names over, then follows the system. */
export function startTheme(): () => void {
  try {
    migrateLegacyStorageItem(window.localStorage, THEME_STORAGE_KEY);
  } catch {
    // Storage blocked: auto still follows the system.
  }
  const stop = theme.start();
  if (theme.choice !== "auto") document.documentElement.dataset.theme = theme.choice;
  return stop;
}

export function setTheme(choice: Choice): void {
  theme.set(choice);
}
