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
  const headPush = { text: "0.2 head/L20h3", op: "push", target: "head/L20h3", coeff: 0.2, onto: 0, position: null, layers: null, trigger: null };
  assert.ok("problem" in recipeAsChips([headPush], fixture.catalog, 0.5), "a head is never pushed");

  // Python's %g, the number format_expr writes.
  for (const [value, text] of [
    [0.3, "0.3"], [0.1 + 0.2, "0.3"], [1, "1"], [-0.35, "-0.35"], [100, "100"], [123456, "123456"],
    [1234567, "1.23457e+06"], [999999.5, "1e+06"], [0.0001, "0.0001"], [0.00001, "1e-05"],
    [0.000123456789, "0.000123457"], [1.5e-7, "1.5e-07"], [2.5e100, "2.5e+100"], [0, "0"],
  ]) assert.equal(formatCoeff(value), text, `formatCoeff(${value})`);
} finally {
  await server.close();
}
