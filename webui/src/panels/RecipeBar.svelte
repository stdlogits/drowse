<script lang="ts">
  // The steering recipe as chips, right above the composer (the canvas's recipe row). Each racked
  // term is a chip: its strength (or "without", for an ablation), its name, where it comes from,
  // and a strip across depth. A chip selects its term, so the Inspector shows its card. "+ Steer"
  // adds a concept or a manifold position; "Edit as text" shows the expression the next reply is
  // generated with, editable in the steering grammar (the server's parser is the only parser).
  import { tick } from "svelte";
  import { MorphText, expand } from "@stdlogits/ledger";
  import {
    steerRack,
    currentSteeringExpression,
    applyCustomSteeringExpression,
    ATOM_PREFIX,
  } from "../lib/stores/steering.svelte";
  import { openDrawer } from "../lib/stores/drawers.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import { select, selection } from "../lib/stores/selection.svelte";
  import type { SteerEntry } from "../lib/types";

  interface Chip {
    name: string;
    alpha: string;
    label: string;
    source: string;
    when: string;
    order: number;
    enabled: boolean;
  }

  const WHEN: Record<string, string> = { BEFORE: "prompt", AFTER: "after the prompt", THINKING: "thinking", RESPONSE: "response" };

  function chipFor(name: string, entry: SteerEntry): Chip {
    const when = WHEN[entry.trigger] ?? "";
    const [selector, ...rest] = name.split("%");
    const bare = selector.split("/").at(-1) ?? selector;
    switch (entry.mode) {
      case "subspace":
        return {
          name, when, order: 0, enabled: entry.enabled,
          alpha: entry.ablate ? "without" : steerRack.subspaceAlong.toFixed(2),
          label: entry.label ?? (rest.join("%") || bare),
          source: entry.label && entry.label !== bare ? bare : "",
        };
      case "manifold":
        return {
          name, when, order: 2, enabled: entry.enabled,
          alpha: entry.blend.toFixed(2),
          label: entry.label ?? entry.coords.map(value => value.toFixed(2)).join(", "),
          source: bare,
        };
      case "jlens":
      case "sae":
        return {
          name, when, order: 1, enabled: entry.enabled,
          alpha: entry.ablate ? "without" : entry.alpha.toFixed(2),
          label: name.slice(ATOM_PREFIX[entry.mode].length),
          source: entry.mode === "jlens" ? "lens token" : "SAE feature",
        };
    }
  }

  const chips = $derived([...steerRack.entries.entries()]
    .map(([name, entry]) => chipFor(name, entry))
    .sort((a, b) => a.order - b.order || a.name.localeCompare(b.name)));
  const expression = $derived(currentSteeringExpression());
  const custom = $derived(steerRack.customExpression !== null);
  const selected = $derived(selection.current.kind === "term" ? selection.current.name : null);

  let adding = $state(false);
  let editing = $state(false);
  let draft = $state("");
  let field: HTMLTextAreaElement | null = $state(null);

  async function edit(): Promise<void> {
    editing = !editing;
    adding = false;
    if (!editing) return;
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
        <button type="button" class="dw-term" class:is-off={!chip.enabled} aria-pressed={selected === chip.name}
          data-tip={chip.name} onclick={() => select({ kind: "term", name: chip.name })}>
          <span class="dw-alpha"><MorphText text={chip.alpha} /></span>
          <span class="dw-term-name">{chip.label}</span>
          {#if chip.source}<span class="meta">{chip.source}</span>{/if}
          <span class="dw-span" aria-hidden="true"><i class="ramp"></i></span>
          {#if chip.when}<span class="meta">{chip.when}</span>{/if}
        </button>
      {/each}
    {/if}
    <button type="button" class="dw-add" aria-expanded={adding} onclick={() => { adding = !adding; editing = false; }}>+ Steer</button>
    <button type="button" class="dw-text end" aria-expanded={editing} onclick={() => void edit()}>{editing ? "Done" : "Edit as text"}</button>
  </div>

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
        placeholder="0.4 honest + 0.2 emotions%wonder@response"
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
  .ramp { left: 0; right: 0; background: linear-gradient(90deg in oklch, var(--deep), var(--late)); opacity: 0.8; }
  .custom .dw-term-name { max-width: 48ch; }
  .more { padding: var(--s1) 0; }
  .text { display: grid; gap: var(--s2); padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .text textarea { width: 100%; min-height: 48px; padding: 0; border: 0; outline: 0; resize: vertical; background: transparent; color: var(--ink); font: var(--type-body); font-variant-numeric: tabular-nums; }
  @media (max-width: 720px) {
    .dw-recipe { flex-wrap: nowrap; overflow-x: auto; overscroll-behavior-x: contain; scrollbar-width: none; margin: 0 calc(-1 * var(--s4)); padding: 0 var(--s4); }
    .dw-recipe::-webkit-scrollbar { display: none; }
    .dw-recipe > * { flex: none; }
    .dw-recipe .dw-term, .dw-recipe .dw-add, .end { height: 44px; }
  }
</style>
