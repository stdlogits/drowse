import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "vite";

// The chips and the server's grammar, over tests/fixtures/recipe_chips.json (shared with
// tests/web/test_recipe_chips.py, which checks each expression is format_expr's own and each parts
// list is what validation answers). Here: each rack's chips write the expression, and the chips
// rebuilt from its parts write it again; a recipe the chips cannot hold stays one read-only chip.
const fixture = JSON.parse(readFileSync(new URL("../../tests/fixtures/recipe_chips.json", import.meta.url), "utf8"));
const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { serializeExpression, formatCoeff } = await server.ssrLoadModule("/src/lib/expression.ts");
  const { recipeAsChips } = await server.ssrLoadModule("/src/lib/steering/recipe.ts");
  const rack = (rows) => new Map(rows.map(({ name, ...entry }) => [name, entry]));

  for (const c of fixture.cases) {
    assert.equal(serializeExpression(rack(c.rack), c.subspaceAlong), c.expression, `${c.name}: the chips' expression`);
    const made = recipeAsChips(c.parts, fixture.catalog, 0.5);
    assert.ok("chips" in made, `${c.name}: ${made.problem}`);
    assert.equal(serializeExpression(made.chips.entries, made.chips.subspaceAlong), c.expression, `${c.name}: rebuilt from its parts`);
    assert.deepEqual([...made.chips.entries.keys()].sort(), c.rack.map(row => row.name).sort(), `${c.name}: one chip a term`);
  }
  for (const c of fixture.readOnly) {
    const made = recipeAsChips(c.parts, fixture.catalog, 0.5);
    assert.ok("problem" in made, `${c.expression} should stay read-only`);
    assert.ok(made.problem.includes(c.problem), `${c.expression}: ${made.problem}`);
  }

  // The rebuilt chips keep what the rack keeps: a label's node coordinates, the shared flat
  // strength (or the current one when the recipe has no flat term), a unit's own rule.
  const pushes = recipeAsChips(fixture.cases[2].parts, fixture.catalog, 0.5);
  assert.deepEqual(pushes.chips.entries.get("default/emotions").coords, [0.3, 0.8]);
  assert.equal(pushes.chips.entries.get("default/emotions").label, "wonder");
  assert.equal(recipeAsChips(fixture.cases[0].parts.slice(1), fixture.catalog, 0.7).chips.subspaceAlong, 0.7);
  const headPush = {
    text: "0.2 head/L20h3", op: "push", target: "head/L20h3", coeff: 0.2, onto: 0, position: null, layers: null,
    trigger: null, phase: "both", first_n: null, after_n: null, gate: null, projection: null,
  };
  assert.ok("problem" in recipeAsChips([headPush], fixture.catalog, 0.5), "a head is never pushed");

  // A term's timing and a projection's parts come from the server's fields, never from parsing its
  // text: a gated, counted term keeps its window, count and gate as fields; a projection its base,
  // relation and second concept, at its own signed strength.
  const byName = (c) => new Map(c.rack.map(({ name, ...entry }) => [name, entry]));
  const gates = fixture.cases.find(c => c.name === "gates and counted windows");
  const gated = recipeAsChips(gates.parts, fixture.catalog, 0.5).chips.entries;
  for (const [name, entry] of byName(gates)) assert.deepEqual(gated.get(name), entry, `${name} as a chip`);
  const relations = fixture.cases.find(c => c.name === "projections relate a concept to a second one");
  const related = recipeAsChips(relations.parts, fixture.catalog, 0.5).chips.entries;
  for (const [name, entry] of byName(relations)) assert.deepEqual(related.get(name), entry, `${name} as a chip`);
  const { triggerWord, strengthDial, strengthSays, chipStrength, gateProbeKey } = await server.ssrLoadModule("/src/lib/steering/terms.ts");
  assert.equal(triggerWord(gated.get("sae/7")), "first 16 tokens while storm ≤ 0.05");
  assert.equal(triggerWord(gated.get("jlens/storm")), "while verbose.concise > 1", "a bare gate runs on the response");
  // Regression: a curved gate's chip showed its ``:fraction`` channel; the expression keeps it.
  assert.equal(triggerWord({ trigger: "RESPONSE", gate: { probe: "default/months:fraction", op: "<", threshold: 0.3 } }), "while months reads < 0.3");
  assert.equal(triggerWord(gated.get("neuron/L31n4410")), "while L8n344 ≥ −0.5");
  assert.equal(triggerWord(gated.get("slens/tower")), "after 4 tokens");
  // Every dial is signed but a curved term's and a unit removal's.
  const signed = fixture.cases.find(c => c.name.startsWith("a signed strength"));
  const away = byName(signed);
  assert.deepEqual([strengthDial("default/verbose.concise", away.get("default/verbose.concise"), -0.5).min, strengthDial("x", related.values().next().value, 0.5).min], [-2, -2]);
  assert.equal(strengthDial("jlens/storm", away.get("jlens/storm"), 0.5).min, -1, "an atom's removal adds below zero");
  assert.equal(strengthDial("head/L20h3", gated.get("head/L20h3"), 0.5).min, 0, "a unit's removal never goes negative");
  assert.equal(strengthDial("default/emotions", gated.get("default/emotions"), 0.5).min, 0, "a curved position has no opposite");
  assert.equal(strengthSays("default/verbose.concise", away.get("default/verbose.concise"), -0.5), "Away from verbose");
  assert.equal(strengthSays("slens/tower", away.get("slens/tower"), -0.5), "50% more of it");
  const poles = fixture.catalog.find(row => row.name === "calm.anxious");
  const projection = related.get("default/calm.anxious|default/verbose.concise");
  assert.deepEqual([0.6, -0.6, 0].map(v => strengthSays("x", projection, v, poles)), ["Toward calm", "Toward anxious", "Off"], "a projection pushes toward a pole");
  assert.equal(chipStrength("slens/tower", away.get("slens/tower"), 0.5), "+50%");
  assert.equal(chipStrength("default/calm.anxious", away.get("default/calm.anxious"), -0.5), "+50%", "a flat removal on a negative shared strength");
  // Regression: a removal chip dropped its amount (0.5 and 1.5 both read "without").
  const removal = away.get("default/calm.anxious");
  assert.deepEqual([1, 0.5, 1.5, 0].map(v => chipStrength("default/calm.anxious", removal, v)), ["without", "without 50%", "without 150%", "off"]);
  // A gate reads the number a probe's card shows: a curved probe's share, any other's value.
  assert.equal(gateProbeKey("default/emotions", { family: "geometry", is_affine: false }), "default/emotions:fraction");
  assert.equal(gateProbeKey("default/verbose.concise", { family: "geometry", is_affine: true }), "default/verbose.concise");
  assert.equal(gateProbeKey("local/a.b", { family: "geometry", is_affine: false }), "local/a.b", "a dotted name takes only the axis forms");
  assert.equal(gateProbeKey("jlens/storm", { family: "jlens" }), "jlens/storm");

  // The store: a recipe from elsewhere becomes chips when they hold it, else stays one read-only
  // chip with the reason; a late answer never overwrites a newer rack; a chip replaces it.
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  const answers = new Map([...fixture.cases, ...fixture.readOnly].map(c => [c.expression, c.parts]));
  let busy = false;
  let hold = null;
  installRuntimeClient({
    mode: "http",
    profiles: { list: async () => ({ profiles: [] }), correlation: async () => null },
    sessions: {
      validateSteering: async (expression) => {
        if (hold) await hold;
        if (busy) throw Object.assign(new Error("busy"), { status: 409 });
        const parts = answers.get(expression);
        return parts ? { valid: true, expression, error: null, n_layers: 32, terms: [], parts } : { valid: false, expression, error: "nope", };
      },
    },
  });
  const steering = await server.ssrLoadModule("/src/lib/stores/steering.svelte.ts");
  const { steerRack, applyRecipe, retryRecipe, restoreRack, addJLensToRack, currentSteeringExpression } = steering;
  steerRack.catalog = fixture.catalog;
  const settle = () => new Promise(resolve => setTimeout(resolve, 0));

  applyRecipe(fixture.cases[2].expression);
  assert.equal(steerRack.readOnlyRecipe, fixture.cases[2].expression, "read-only while it is checked");
  assert.equal(steerRack.recipeNote, null);
  await settle();
  assert.equal(steerRack.readOnlyRecipe, null, "chips hold it");
  assert.equal(currentSteeringExpression(), fixture.cases[2].expression);
  assert.equal(steerRack.entries.size, 7);

  applyRecipe(fixture.readOnly[0].expression);
  await settle();
  assert.equal(steerRack.readOnlyRecipe, fixture.readOnly[0].expression);
  assert.equal(steerRack.entries.size, 0);
  assert.match(steerRack.recipeNote.text, /different strengths/);
  assert.equal(steerRack.recipeNote.retry, false);
  assert.equal(currentSteeringExpression(), fixture.readOnly[0].expression, "sent as written");

  busy = true;
  applyRecipe(fixture.cases[0].expression);
  await settle();
  assert.equal(steerRack.recipeNote.retry, true, "a busy model leaves it read-only, to check again");
  busy = false;
  retryRecipe();
  await settle();
  assert.equal(steerRack.readOnlyRecipe, null);
  assert.equal(currentSteeringExpression(), fixture.cases[0].expression);

  let release;
  hold = new Promise(resolve => { release = resolve; });
  applyRecipe(fixture.cases[1].expression);
  addJLensToRack("late");
  hold = null;
  release();
  await settle();
  assert.equal(currentSteeringExpression(), "0.3 jlens/late", "a chip added meanwhile wins over the late check");

  applyRecipe("");
  assert.equal(steerRack.readOnlyRecipe, null);
  assert.equal(currentSteeringExpression(), "", "an empty recipe is unsteered");
  restoreRack([], 0.5, "0.3 nowhere");
  await settle();
  assert.equal(steerRack.recipeNote.text, "nope", "a recipe that does not resolve stays read-only with the server's reason");

  // Python's %g, the number format_expr writes.
  for (const [value, text] of [
    [0.3, "0.3"], [0.1 + 0.2, "0.3"], [1, "1"], [-0.35, "-0.35"], [100, "100"], [123456, "123456"],
    [1234567, "1.23457e+06"], [999999.5, "1e+06"], [0.0001, "0.0001"], [0.00001, "1e-05"],
    [0.000123456789, "0.000123457"], [1.5e-7, "1.5e-07"], [2.5e100, "2.5e+100"], [0, "0"],
    // Regression: an exact binary tie at the seventh digit rounds to even, as %g does (a fraction
    // as well as an integer; toExponential rounds it up).
    [1.015625, "1.01562"], [0.01953125, "0.0195312"], [1234565, "1.23456e+06"], [-1.015625, "-1.01562"],
    [-0, "-0"], [5e-324, "4.94066e-324"],
  ]) assert.equal(formatCoeff(value), text, `formatCoeff(${value})`);

  // Wave 3 finding 26: on a phone a tap shows a tip (Ledger's touch rule), and the read-only recipe
  // chip's tip stayed over the reason the opened panel gives. The tip goes while the panel shows.
  const recipeBar = readFileSync(new URL("../src/panels/RecipeBar.svelte", import.meta.url), "utf8");
  assert.match(recipeBar, /class="dw-term held" aria-expanded=\{recipeOpen\} data-tip=\{recipeOpen \? undefined : "A recipe from elsewhere, as it was written"\}/);
  // Regression (a9, unnecessary text): "+ Steer" offered its kinds and a line saying where the
  // other kinds steer from. Wave 6 (a9: all steering in one place): "+ Steer" is the picker of every
  // kind, with no caption beside it (scripts/steer-actions.test.mjs renders it).
  const steerMenu = recipeBar.slice(recipeBar.indexOf("{#if adding}"), recipeBar.indexOf("{/if}", recipeBar.indexOf("{#if adding}")));
  assert.match(steerMenu, /<SteerPicker /);
  assert.doesNotMatch(steerMenu, /class="meta"|steer from the instruments/);
} finally {
  await server.close();
}
