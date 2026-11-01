<script lang="ts">
  // The steering recipe as chips, right above the composer (the canvas's recipe row). Each racked
  // term is a chip: its strength (or "without", for a removal), its name, where it comes from, a
  // span over the model's depth where it acts, its layers when it is scoped and its trigger. A chip
  // selects its term and opens its editor in place, right under the chips (TermEditor, the Steering
  // canvas); the selected chip closes it. This strip is the one place steering lives: "+ Steer"
  // (SteerPicker) adds a neuron or head, a lens token, a concept, a manifold position or an SAE
  // feature, and the instruments' one-click Push and Silence (lib/steerActions.ts) land here too,
  // their chip opened. The chips sort the same way, units first. Recipes are written with the chips,
  // never typed: a recipe from elsewhere (an agent's, a saved conversation's, the command line's)
  // becomes chips when they can hold it, and otherwise shows as one read-only chip that says why and
  // can be cleared.
  import { untrack } from "svelte";
  import { MorphText, expand, reducedMotion } from "@stdlogits/ledger";
  import TermEditor from "./rack/TermEditor.svelte";
  import SteerPicker from "./rack/SteerPicker.svelte";
  import {
    steerRack,
    applyRecipe,
    retryRecipe,
    removeSteerEntry,
    manifoldByName,
    lensTokenText,
  } from "../lib/stores/steering.svelte";
  import { closeTermEditor, openTermEditor, termOpen } from "../lib/stores/selection.svelte";
  import { sessionState } from "../lib/stores/session.svelte";
  import { unitTermLabel } from "../lib/stores/units.svelte";
  import { rackEditQueued } from "../lib/stores/pending.svelte";
  import { steerPicker } from "../lib/stores/workbench.svelte";
  import { UnitLabels } from "../drawers/token/labels.svelte";
  import { depthSpan, layerRunsText } from "../lib/steering/layers";
  import { placeOf } from "../lib/steering/termLayers.svelte";
  import { chipStrength, termSource, termTitle, triggerWord, unitLayer, wrapPieces } from "../lib/steering/terms";
  import type { SteerEntry } from "../lib/types";

  /** The order families show in, as everywhere an order is chosen: units (neurons and heads), lens
   *  tokens, flat terms, curved terms, then SAE features. */
  const ORDER: Record<SteerEntry["mode"], number> = { unit: 0, jlens: 1, slens: 1, subspace: 2, projection: 2, manifold: 3, sae: 4 };

  /** The catalog's labels for the unit chips (each unit asked once). */
  const unitLabels = new UnitLabels();
  const unitNames = $derived([...steerRack.entries.keys()].filter((name) => steerRack.entries.get(name)?.mode === "unit"));
  $effect(() => {
    if (unitNames.length > 0) unitLabels.want(unitNames.map((unit) => ({ unit })));
  });
  function unitChipLabel(name: string, entry: SteerEntry): string | null {
    if (entry.mode !== "unit") return null;
    const head = name.startsWith("head/");
    const pole = head || entry.ablate || entry.alpha >= 0 ? "+" : "-";
    return unitLabels.label(name, head ? "head" : "neuron", pole)?.text ?? unitTermLabel(name, entry);
  }

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
        unit: entry.mode === "unit",
        source: unitChipLabel(name, entry) ?? termSource(name, entry),
        span: place.where ? depthSpan(place.selected, place.where.nLayers) : null,
        meta: [layers, triggerWord(entry)].filter(Boolean).join(" · "),
      };
    }));
  const recipe = $derived(steerRack.readOnlyRecipe);
  const note = $derived(steerRack.recipeNote);
  const open = $derived(termOpen.name !== null && steerRack.entries.has(termOpen.name) ? termOpen.name : null);
  const openEntry = $derived(open === null ? undefined : steerRack.entries.get(open));

  $effect(() => {
    // A term taken off the rack elsewhere (a recipe from an agent, a tool) leaves nothing to edit. A
    // term still queued behind a running reply (a unit pushed from an instrument) opens once it lands.
    const name = termOpen.name;
    if (name !== null && !steerRack.entries.has(name) && !rackEditQueued()) closeTermEditor();
  });

  let adding = $state(false);
  // The palette's Steer entries open "+ Steer" on their kind (stores/workbench.svelte.ts).
  let seenRequest = steerPicker.requested;
  $effect(() => {
    const requested = steerPicker.requested;
    if (requested === seenRequest) return;
    seenRequest = requested;
    untrack(() => {
      adding = true;
      recipeOpen = false;
      closeTermEditor();
    });
  });
  /** The read-only recipe's panel. */
  let recipeOpen = $state(false);
  /** The term just removed, until the next change: the row offers it back. */
  let removed: { name: string; entry: SteerEntry; title: string } | null = $state(null);
  $effect(() => {
    if (recipe === null) recipeOpen = false;
  });

  function toggle(name: string): void {
    removed = null;
    if (open === name) closeTermEditor();
    else openTermEditor(name);
  }

  function remove(name: string): void {
    const entry = steerRack.entries.get(name);
    if (!entry) return;
    removed = { name, entry, title: termTitle(name, entry) };
    removeSteerEntry(name);
    closeTermEditor();
  }

  function undo(): void {
    if (!removed || steerRack.readOnlyRecipe !== null) return;
    steerRack.entries.set(removed.name, removed.entry);
    openTermEditor(removed.name);
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

  function toggleAdding(): void {
    adding = !adding;
    recipeOpen = false;
    removed = null;
    // The picker and an open term share the space under the chips: opening one closes the other.
    if (adding) closeTermEditor();
  }
  $effect(() => {
    if (open !== null) adding = false;
  });
</script>

<div class="recipe">
  <div class="dw-recipe" role="group" aria-label="Steering">
    {#if recipe !== null}
      <!-- The tip says what the open panel says at length: it goes while the panel shows, so a tap
           (which shows a tip on touch) never leaves it over the read-only reason. -->
      <button type="button" class="dw-term held" aria-expanded={recipeOpen} data-tip={recipeOpen ? undefined : "A recipe from elsewhere, as it was written"}
        onclick={() => { recipeOpen = !recipeOpen; adding = false; }}>
        <span class="dw-alpha">recipe</span><span class="dw-term-name">{recipe}</span>
      </button>
    {:else}
      {#each chips as chip (chip.name)}
        <button type="button" class="dw-term" class:is-off={!chip.enabled} class:is-unit={chip.unit} aria-pressed={open === chip.name}
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
    <button type="button" class="dw-add" aria-expanded={adding} onclick={toggleAdding}>+ Steer</button>
  </div>

  {#if open !== null && openEntry}
    <div class="editor" transition:expand onintroend={(event) => (event.currentTarget as HTMLElement).scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" })}>
      {#key open}
        <TermEditor name={open} entry={openEntry} onclose={closeTermEditor} onremove={() => { if (open !== null) remove(open); }} />
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
    <div class="adding" transition:expand>
      <SteerPicker onpicked={() => (adding = false)} />
    </div>
  {/if}
</div>

<style>
  .recipe { display: grid; gap: var(--s2); min-width: 0; }
  .held .dw-term-name { max-width: 48ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  /* A unit's chip names it by its ID, as every unit row does. */
  .is-unit .dw-term-name { font: 600 14px/20px var(--display); letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }
  .gone { padding: 0 var(--s2) 0 var(--s3); white-space: nowrap; }
  .undo { height: 36px; padding: 0 var(--s2); }
  .dw-span > i { transition: left var(--glide) var(--ease), width var(--glide) var(--ease); }
  .editor { min-width: 0; padding-top: var(--s2); }
  .adding { min-width: 0; padding-top: var(--s2); }
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
    /* "+ Steer" stays at the strip's end in view while the chips scroll under it. */
    .dw-recipe > .dw-add { position: sticky; right: 0; z-index: 1; background: var(--paper); }
    .dw-recipe > .dw-add:hover, .dw-recipe > .dw-add[aria-expanded="true"] { background: var(--wash); }
    /* The open editor is taller than a phone's dock can hold, and the workbench does not scroll:
     * it scrolls on its own, so Where, the expression, Remove and Done stay reachable. */
    .editor { max-height: 52dvh; overflow-y: auto; overscroll-behavior: contain; }
    .held-recipe { padding: var(--s4); }
    .held-recipe .dw-text { position: relative; }
    .held-recipe .dw-text::after { content: ""; position: absolute; inset: -14px -8px; }
  }
</style>
