<script lang="ts">
  // The steering recipe as chips, right above the composer (the canvas's recipe row). Each racked
  // term is a chip: its strength (or "without", for a removal), its name, where it comes from, a
  // span over the model's depth where it acts, its layers when it is scoped and its trigger. A chip
  // selects its term and opens its editor in place, right under the chips (TermEditor, the Steering
  // canvas); the selected chip closes it. "+ Steer" adds a concept or a manifold position. Recipes
  // are written with the chips, never typed: a recipe from elsewhere (an agent's, a saved
  // conversation's, the command line's) becomes chips when they can hold it, and otherwise shows
  // as one read-only chip that says why and can be cleared.
  import { MorphText, expand, reducedMotion } from "@stdlogits/ledger";
  import TermEditor from "./rack/TermEditor.svelte";
  import {
    steerRack,
    applyRecipe,
    retryRecipe,
    removeSteerEntry,
    manifoldByName,
    lensTokenText,
  } from "../lib/stores/steering.svelte";
  import { openDrawer } from "../lib/stores/drawers.svelte";
  import { clearSelection, select, selection } from "../lib/stores/selection.svelte";
  import { sessionState } from "../lib/stores/session.svelte";
  import { unitTermLabel } from "../lib/stores/units.svelte";
  import { depthSpan, layerRunsText } from "../lib/steering/layers";
  import { placeOf } from "../lib/steering/termLayers.svelte";
  import { chipStrength, termSource, termTitle, triggerWord, unitLayer, wrapPieces } from "../lib/steering/terms";
  import type { SteerEntry } from "../lib/types";

  /** The order families show in: flat terms, curved terms, then atoms and units. */
  const ORDER: Record<SteerEntry["mode"], number> = { subspace: 0, projection: 0, manifold: 1, jlens: 2, slens: 2, sae: 2, unit: 3 };

  const model = $derived(sessionState.info?.model_id);
  const chips = $derived([...steerRack.entries.entries()]
    .sort(([a, x], [b, y]) => ORDER[x.mode] - ORDER[y.mode] || a.localeCompare(b))
    .map(([name, entry]) => {
      const place = placeOf(name, entry, model);
      const unitAt = entry.mode === "unit" ? unitLayer(name) : null;
      const scoped = entry.mode !== "unit" && entry.layers != null;
      const layers = unitAt !== null ? `${unitAt}` : scoped ? (place.selected.length ? layerRunsText(place.selected) : "no layers") : "";
      return {
        name,
        enabled: entry.enabled,
        strength: chipStrength(name, entry, steerRack.subspaceAlong),
        title: termTitle(name, entry, entry.mode === "manifold" ? manifoldByName(name) : null, lensTokenText),
        source: (entry.mode === "unit" ? unitTermLabel(name, entry) : null) ?? termSource(name, entry),
        span: place.where ? depthSpan(place.selected, place.where.nLayers) : null,
        meta: [layers, triggerWord(entry)].filter(Boolean).join(" · "),
      };
    }));
  const recipe = $derived(steerRack.readOnlyRecipe);
  const note = $derived(steerRack.recipeNote);
  const open = $derived(selection.current.kind === "term" && steerRack.entries.has(selection.current.name) ? selection.current.name : null);
  const openEntry = $derived(open === null ? undefined : steerRack.entries.get(open));

  $effect(() => {
    // A term taken off the rack elsewhere (a recipe from an agent, a tool) leaves nothing to edit.
    const current = selection.current;
    if (current.kind === "term" && !steerRack.entries.has(current.name)) clearSelection();
  });

  let adding = $state(false);
  /** The read-only recipe's panel. */
  let recipeOpen = $state(false);
  /** The term just removed, until the next change: the row offers it back. */
  let removed: { name: string; entry: SteerEntry; title: string } | null = $state(null);
  $effect(() => {
    if (recipe === null) recipeOpen = false;
  });

  function toggle(name: string): void {
    removed = null;
    if (open === name) clearSelection();
    else select({ kind: "term", name });
  }

  function remove(name: string): void {
    const entry = steerRack.entries.get(name);
    if (!entry) return;
    removed = { name, entry, title: termTitle(name, entry) };
    removeSteerEntry(name);
    clearSelection();
  }

  function undo(): void {
    if (!removed || steerRack.readOnlyRecipe !== null) return;
    steerRack.entries.set(removed.name, removed.entry);
    select({ kind: "term", name: removed.name });
    removed = null;
  }

  let copied = $state(false);
  async function copy(): Promise<void> {
    if (recipe === null) return;
    try {
      await navigator.clipboard.writeText(recipe);
      copied = true;
    } catch {
      copied = false;
    }
  }
  $effect(() => {
    void recipe;
    copied = false;
  });

  function add(kind: "subspace" | "manifolds"): void {
    adding = false;
    removed = null;
    openDrawer(kind);
  }
</script>

<div class="recipe">
  <div class="dw-recipe" role="group" aria-label="Steering">
    {#if recipe !== null}
      <button type="button" class="dw-term held" aria-expanded={recipeOpen} data-tip="A recipe from elsewhere, as it was written"
        onclick={() => { recipeOpen = !recipeOpen; adding = false; }}>
        <span class="dw-alpha">recipe</span><span class="dw-term-name">{recipe}</span>
      </button>
    {:else}
      {#each chips as chip (chip.name)}
        <button type="button" class="dw-term" class:is-off={!chip.enabled} aria-pressed={open === chip.name}
          aria-expanded={open === chip.name} data-tip={chip.name} onclick={() => toggle(chip.name)}>
          <span class="dw-alpha"><MorphText text={chip.strength} /></span>
          <span class="dw-term-name">{chip.title}</span>
          {#if chip.source}<span class="meta">{chip.source}</span>{/if}
          <span class="dw-span" aria-hidden="true">{#if chip.span}<i style:left="{chip.span.left}%" style:width="{chip.span.width}%" style:background={chip.span.background}></i>{/if}</span>
          {#if chip.meta}<span class="meta">{chip.meta}</span>{/if}
        </button>
      {/each}
      {#if removed}
        <span class="meta gone">{removed.title} removed</span>
        <button type="button" class="dw-text undo" onclick={undo}>Undo</button>
      {/if}
    {/if}
    <button type="button" class="dw-add" aria-expanded={adding} onclick={() => { adding = !adding; recipeOpen = false; removed = null; }}>+ Steer</button>
  </div>

  {#if open !== null && openEntry}
    <div class="editor" transition:expand onintroend={(event) => (event.currentTarget as HTMLElement).scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" })}>
      {#key open}
        <TermEditor name={open} entry={openEntry} onclose={clearSelection} onremove={() => { if (open !== null) remove(open); }} />
      {/key}
    </div>
  {/if}

  {#if recipeOpen && recipe !== null}
    <section class="held-recipe" aria-label="Recipe" transition:expand>
      <div class="expr">
        <p class="text">{#each wrapPieces(recipe) as piece, i}{#if i}<wbr>{/if}{piece}{/each}</p>
        <button type="button" class="dw-text" onclick={() => void copy()}><MorphText text={copied ? "Copied" : "Copy"} numbers={false} /></button>
      </div>
      <p class="meta" role="status">{note === null ? "Checking whether chips can hold it…" : note.text} Adding a term replaces it.</p>
      <div class="foot">
        {#if note?.retry}<button type="button" class="dw-text" onclick={retryRecipe}>Check again</button>{:else}<span></span>{/if}
        <button type="button" class="dw-text" onclick={() => applyRecipe("")}>Clear</button>
      </div>
    </section>
  {/if}

  {#if adding}
    <div class="dw-row more" transition:expand>
      <button type="button" class="dw-quiet" onclick={() => add("subspace")}>A concept</button>
      <button type="button" class="dw-quiet" onclick={() => add("manifolds")}>A manifold position</button>
      <span class="meta">lens tokens and SAE features steer from the instruments</span>
    </div>
  {/if}
</div>

<style>
  .recipe { display: grid; gap: var(--s2); min-width: 0; }
  .held .dw-term-name { max-width: 48ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .gone { padding: 0 var(--s2) 0 var(--s3); white-space: nowrap; }
  .undo { height: 36px; padding: 0 var(--s2); }
  .dw-span > i { transition: left var(--glide) var(--ease), width var(--glide) var(--ease); }
  .editor { min-width: 0; padding-top: var(--s2); }
  .more { padding: var(--s1) 0; }
  /* The read-only recipe, open: the term editor's surface, its expression on paper. */
  .held-recipe { display: grid; gap: var(--s3); min-width: 0; padding: var(--s5); border-radius: var(--r-row); background: var(--wash); }
  .held-recipe .meta { margin: 0; overflow-wrap: anywhere; }
  .expr { display: flex; align-items: center; gap: var(--s4); min-width: 0; padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--paper); }
  .text { flex: 1 1 0; min-width: 0; margin: 0; color: var(--ink); font: var(--type-body); font-variant-numeric: tabular-nums; overflow-wrap: break-word; }
  .foot { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); }
  @media (max-width: 720px) {
    .dw-recipe { flex-wrap: nowrap; overflow-x: auto; overscroll-behavior-x: contain; scrollbar-width: none; margin: 0 calc(-1 * var(--s4)); padding: 0 var(--s4); }
    .dw-recipe::-webkit-scrollbar { display: none; }
    .dw-recipe > * { flex: none; }
    .dw-recipe .dw-term, .dw-recipe .dw-add, .undo { height: 44px; }
    .held-recipe { padding: var(--s4); }
    .held-recipe .dw-text { position: relative; }
    .held-recipe .dw-text::after { content: ""; position: absolute; inset: -14px -8px; }
  }
</style>
