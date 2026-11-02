// Applies a stored light or dark choice before first paint; with none (auto) tokens.css follows the
// system. Load it as a classic, blocking script in <head>, served from the product's own origin (a
// strict CSP allows no inline script), naming the product's storage key:
//   <script src="/theme-preflight.js" data-ledger-theme-key="neuron-lens.theme"></script>
// createTheme({ key }) (src/lib/theme.svelte.ts) owns the theme after, under the same key.
(function () {
  try {
    var script = document.currentScript;
    var key = script && script.getAttribute("data-ledger-theme-key");
    var mode = key && localStorage.getItem(key);
    if (mode === "light" || mode === "dark") document.documentElement.dataset.theme = mode;
  } catch (error) {}
})();
