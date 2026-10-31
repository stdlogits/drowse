import assert from "node:assert/strict";
import { createServer } from "vite";

// The shared form primitives and the Model room's pages, rendered on the server: the props drowse.ai
// relies on keep their meaning, and the pages carry what they show.
const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "http" });
  const { render } = await server.ssrLoadModule("svelte/server");
  const { createRawSnippet } = await server.ssrLoadModule("svelte");
  const label = (text) => createRawSnippet(() => ({ render: () => `<span>${text}</span>` }));

  // Button: the four variants on one skeleton; busy is announced; an accent tints the fill.
  const { default: Button } = await server.ssrLoadModule("/src/lib/ui/Button.svelte");
  const plain = render(Button, { props: { children: label("Go") } }).body;
  assert.match(plain, /class="sk-btn ghost md[^"]*"/);
  assert.match(plain, /type="button"/);
  assert.doesNotMatch(plain, /aria-busy/);
  const busy = render(Button, { props: { children: label("Go"), variant: "solid", size: "sm", busy: true, type: "submit" } }).body;
  assert.match(busy, /class="sk-btn solid sm[^"]*"/);
  assert.match(busy, /aria-busy="true"/);
  assert.match(busy, /type="submit"/);
  const tinted = render(Button, { props: { children: label("Go"), accent: "var(--tint)" } }).body;
  assert.match(tinted, /accented/);
  assert.match(tinted, /--btn-accent: var\(--tint\)/);
  for (const variant of ["flat", "danger"]) {
    assert.match(render(Button, { props: { children: label("Go"), variant } }).body, new RegExp(`class="sk-btn ${variant} md`));
  }

  // BaseModelTag: a word, not capitals; ``plain`` drops the chip.
  const { default: BaseModelTag } = await server.ssrLoadModule("/src/lib/ui/BaseModelTag.svelte");
  assert.match(render(BaseModelTag).body, />base</);
  assert.doesNotMatch(render(BaseModelTag).body, /BASE/);
  assert.match(render(BaseModelTag, { props: { plain: true } }).body, /class="[^"]*plain/);

  // Select: closed, the field shows the chosen label (or the placeholder) and no list.
  const { default: Select } = await server.ssrLoadModule("/src/lib/Select.svelte");
  const options = [{ value: "a", label: "Alpha" }, { value: "b", label: "Beta" }];
  const closed = render(Select, { props: { value: "b", options, ariaLabel: "Letter" } }).body;
  assert.match(closed, /aria-expanded="false"/);
  assert.doesNotMatch(closed, /role="listbox"/);
  assert.match(closed, /Beta/);
  assert.match(render(Select, { props: { value: "z", options, placeholder: "Pick one" } }).body, /is-placeholder/);
  assert.match(render(Select, { props: { value: "a", options, invalid: true } }).body, /data-invalid/);

  // Slider: the value it announces follows the step's precision, or the caller's text.
  const { default: Slider } = await server.ssrLoadModule("/src/lib/Slider.svelte");
  assert.match(render(Slider, { props: { value: 0.5, min: 0, max: 2, step: 0.05 } }).body, /aria-valuetext="0.50"/);
  assert.match(render(Slider, { props: { value: 3, min: 0, max: 10, step: 1, displayValue: "three" } }).body, /aria-valuetext="three"/);
  assert.doesNotMatch(render(Slider, { props: { value: 1 } }).body, /popover/);

  // NumberInput: at rest the value is drawn over the field; disabled hides the steppers.
  const { default: NumberInput } = await server.ssrLoadModule("/src/lib/NumberInput.svelte");
  const number = render(NumberInput, { props: { value: 512, min: 1, max: 1024 } }).body;
  assert.match(number, /number-rest/);
  assert.match(number, /sk-number-steppers/);
  assert.doesNotMatch(render(NumberInput, { props: { value: 512, disabled: true } }).body, /sk-number-steppers/);
  assert.doesNotMatch(render(NumberInput, { props: { value: null, allowEmpty: true } }).body, /number-rest/);

  // ResetSettings: the sampling section's one-step reset, the Model room's ask-first reset.
  const { default: ResetSettings } = await server.ssrLoadModule("/src/lib/ui/ResetSettings.svelte");
  assert.match(render(ResetSettings).body, /Reset to original/);
  const full = render(ResetSettings, { props: { full: true } }).body;
  assert.match(full, /Reset settings/);
  assert.doesNotMatch(full, /Chats and downloads stay/, "the scope is said when it asks, not before");

  // Health: five figures, five checks, warnings and the reset, without a dialog header in a room.
  const { default: Health } = await server.ssrLoadModule("/src/drawers/HealthDrawer.svelte");
  const room = render(Health, { props: { params: { embedded: true } } }).body;
  assert.doesNotMatch(room, /class="dlg-head/);
  assert.equal((room.match(/class="figure[ "]/g) ?? []).length, 5);
  assert.equal((room.match(/class="row[ "]/g) ?? []).length, 5);
  for (const head of ["Now", "Checks", "Warnings"]) assert.match(room, new RegExp(`>${head}<`));
  assert.match(render(Health, { props: { params: null } }).body, /class="dlg-head/);

  // Help: the rooms, the keys the shell actually binds, and the grammar as rows.
  const { default: Help } = await server.ssrLoadModule("/src/drawers/HelpDrawer.svelte");
  const help = render(Help).body;
  for (const room of ["Workbench", "Manifolds", "Units", "Lens", "Model", "Loom"]) assert.match(help, new RegExp(`<dt[^>]*>${room}</dt>`));
  assert.match(help, /Searches everything/);
  assert.doesNotMatch(help, /without generating/, "the composer has no add-without-generating key");
  assert.equal((help.match(/class="forms/g) ?? []).length, 7);
  assert.doesNotMatch(help, /<pre/);

  console.log("Form primitives (button, tag, select, slider, number, reset) and the Model room's health and help render as drawn");
} finally { await server.close(); }
