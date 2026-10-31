<script lang="ts">
  // The steering recipe as chips, right above the composer (the canvas's recipe row). Each racked
  // term is a chip: its strength (or "without", for a removal), its name, where it comes from, a
  // span over the model's depth where it acts, its layers when it is scoped and its trigger. A chip
  // selects its term and opens its editor in place, right under the chips (TermEditor, the Steering
  // canvas); the selected chip closes it. "+ Steer" adds a concept or a manifold position; "Edit as
  // text" shows the expression the next reply is generated with, editable in the steering grammar
  // (the server's parser is the only parser).
  import { tick } from "svelte";
  import { MorphText, expand, reducedMotion } from "@stdlogits/ledger";
  import TermEditor from "./rack/TermEditor.svelte";
  import {
    steerRack,
    currentSteeringExpression,
    applyCustomSteeringExpression,
    removeSteerEntry,
  } from "../lib/stores/steering.svelte";
  import { openDrawer } from "../lib/stores/drawers.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import { clearSelection, select, selection } from "../lib/stores/selection.svelte";
  import { sessionState } from "../lib/stores/session.svelte";
  import { unitTermLabel } from "../lib/stores/units.svelte";
  import { depthSpan, layerRunsText } from "../lib/steering/layers";
  import { placeOf } from "../lib/steering/termLayers.svelte";
  import { chipStrength, termSource, termTitle, triggerWord, unitLayer } from "../lib/steering/terms";
  import type { SteerEntry } from "../lib/types";

  /** The order families show in: flat terms, curved terms, then atoms and units. */
  const ORDER: Record<SteerEntry["mode"], number> = { subspace: 0, manifold: 1, jlens: 2, sae: 2, unit: 3 };

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
        title: termTitle(name, entry),
        source: (entry.mode === "unit" ? unitTermLabel(name, entry) : null) ?? termSource(name, entry),
        span: place.where ? depthSpan(place.selected, place.where.nLayers) : null,
        meta: [layers, triggerWord(entry.trigger)].filter(Boolean).join(" · "),
      };
    }));
  const expression = $derived(currentSteeringExpression());
  const custom = $derived(steerRack.customExpression !== null);
  const open = $derived(selection.current.kind === "term" && steerRack.entries.has(selection.current.name) ? selection.current.name : null);
  const openEntry = $derived(open === null ? undefined : steerRack.entries.get(open));

  $effect(() => {
    // A term taken off the rack elsewhere (a text recipe, a tool) leaves nothing to edit.
    const current = selection.current;
    if (current.kind === "term" && !steerRack.entries.has(current.name)) clearSelection();
  });

  let adding = $state(false);
  let editing = $state(false);
  let draft = $state("");
  let field: HTMLTextAreaElement | null = $state(null);
  /** The term just removed, until the next change: the row offers it back. */
  let removed: { name: string; entry: SteerEntry; title: string } | null = $state(null);

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
    if (!removed || steerRack.customExpression !== null) return;
    steerRack.entries.set(removed.name, removed.entry);
    select({ kind: "term", name: removed.name });
    removed = null;
  }

  async function edit(): Promise<void> {
    editing = !editing;
    adding = false;
    removed = null;
    if (!editing) return;
    if (open !== null) clearSelection();
    draft = expression;
    await tick();
    field?.focus();
  }

  function apply(): void {
    applyCustomSteeringExpression(draft.trim());
    editing = false;
  }

  async function copy(): Promise<void> {
    try { await navigator.clipboard.writeText(expression); pushToast("Copied the recipe.", { kind: "info" }); }
    catch { pushToast("Could not copy the recipe. Select its text and copy it.", { kind: "error" }); }
  }

  function add(kind: "subspace" | "manifolds"): void {
    adding = false;
    removed = null;
    openDrawer(kind);
  }
</script>

<div class="recipe">
  <div class="dw-recipe" role="group" aria-label="Steering">
    {#if custom}
      <button type="button" class="dw-term custom" onclick={() => void edit()} aria-expanded={editing}>
        <span class="dw-alpha">text</span><span class="dw-term-name">{expression || "unsteered"}</span>
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
    <button type="button" class="dw-add" aria-expanded={adding} onclick={() => { adding = !adding; editing = false; removed = null; }}>+ Steer</button>
    <button type="button" class="dw-text end" aria-expanded={editing} onclick={() => void edit()}>{editing ? "Done" : "Edit as text"}</button>
  </div>

  {#if open !== null && openEntry}
    <div class="editor" transition:expand onintroend={(event) => (event.currentTarget as HTMLElement).scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" })}>
      {#key open}
        <TermEditor name={open} entry={openEntry} onclose={clearSelection} onremove={() => { if (open !== null) remove(open); }} />
      {/key}
    </div>
  {/if}

  {#if adding}
    <div class="dw-row more" transition:expand>
      <button type="button" class="dw-quiet" onclick={() => add("subspace")}>A concept</button>
      <button type="button" class="dw-quiet" onclick={() => add("manifolds")}>A manifold position</button>
      <span class="meta">lens tokens and SAE features steer from the instruments</span>
    </div>
  {/if}

  {#if editing}
    <form class="text" transition:expand onsubmit={(event) => { event.preventDefault(); apply(); }}>
      <label class="sr-only" for="recipe-text">Steering expression</label>
      <textarea id="recipe-text" bind:this={field} bind:value={draft} rows="2" spellcheck="false"
        placeholder="0.4 honest + 0.2 emotions%wonder{'{'}14-24{'}'}@response"
        onkeydown={(event) => { if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) { event.preventDefault(); apply(); } }}></textarea>
      <div class="dw-row">
        <button type="submit" class="dw-quiet">Use this recipe</button>
        <button type="button" class="dw-text" disabled={!expression} onclick={() => void copy()}>Copy</button>
        {#if custom}<button type="button" class="dw-text" onclick={() => { applyCustomSteeringExpression(""); editing = false; }}>Clear</button>{/if}
      </div>
    </form>
  {/if}
</div>

<style>
  .recipe { display: grid; gap: var(--s2); min-width: 0; }
  .end { margin-left: auto; height: 36px; padding: 0 var(--s2); }
  .custom .dw-term-name { max-width: 48ch; }
  .gone { padding: 0 var(--s2) 0 var(--s3); white-space: nowrap; }
  .undo { height: 36px; padding: 0 var(--s2); }
  .dw-span > i { transition: left var(--glide) var(--ease), width var(--glide) var(--ease); }
  .editor { min-width: 0; padding-top: var(--s2); }
  .more { padding: var(--s1) 0; }
  .text { display: grid; gap: var(--s2); padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .text textarea { width: 100%; min-height: 48px; padding: 0; border: 0; outline: 0; resize: vertical; background: transparent; color: var(--ink); font: var(--type-body); font-variant-numeric: tabular-nums; }
  @media (max-width: 720px) {
    .dw-recipe { flex-wrap: nowrap; overflow-x: auto; overscroll-behavior-x: contain; scrollbar-width: none; margin: 0 calc(-1 * var(--s4)); padding: 0 var(--s4); }
    .dw-recipe::-webkit-scrollbar { display: none; }
    .dw-recipe > * { flex: none; }
    .dw-recipe .dw-term, .dw-recipe .dw-add, .end, .undo { height: 44px; }
  }
</style>
