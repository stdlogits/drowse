<script lang="ts">
  // Search everything (⌘K, the header's search): the rooms, the instruments and every tool, in one
  // filterable list, drawn as Ledger's search (a field on a wash, the chosen option under a wash2
  // thumb that glides as the arrow keys move).
  import { tick } from "svelte";
  import { Icon, fadeIn, fadeOut, glide, rise } from "@stdlogits/ledger";
  import { paletteState, closePalette } from "../lib/stores/palette.svelte";
  import { paletteCommands, type PaletteCommand } from "../lib/commands";
  import { openDrawer, setInspectorTab } from "../lib/stores.svelte";
  import { goto } from "../lib/stores/rooms.svelte";
  import { clearSelection } from "../lib/stores/selection.svelte";
  import { showConversation } from "../lib/stores/workbench.svelte";
  import { containTab } from "../lib/focusTrap";

  const COMMANDS = paletteCommands();

  let query = $state("");
  let selected = $state(0);
  let inputEl: HTMLInputElement | null = $state(null);
  let listEl: HTMLElement | null = $state(null);
  let paletteEl: HTMLElement | null = $state(null);

  const filtered = $derived.by(() => {
    const q = query.trim().toLowerCase();
    if (!q) return COMMANDS;
    const terms = q.split(/\s+/);
    const rank = (command: PaletteCommand): number => {
      const label = command.label.toLowerCase();
      if (label === q) return 0;
      if (label.startsWith(q)) return 1;
      if (label.includes(q)) return 2;
      if ((command.keywords ?? "").toLowerCase().includes(q)) return 3;
      return 4;
    };
    return COMMANDS
      .filter((c) => terms.every((t) => `${c.label} ${c.group} ${c.keywords ?? ""}`.toLowerCase().includes(t)))
      .sort((a, b) => rank(a) - rank(b));
  });

  // Reset and focus on every open; keep the selection inside the list as it narrows.
  $effect(() => {
    if (!paletteState.open) return;
    query = "";
    selected = 0;
    void tick().then(() => inputEl?.focus());
  });
  $effect(() => {
    if (selected >= filtered.length) selected = Math.max(0, filtered.length - 1);
  });
  $effect(() => {
    void selected;
    listEl?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: "nearest" });
  });

  function run(cmd: PaletteCommand): void {
    closePalette();
    switch (cmd.action.kind) {
      case "room":
        goto(cmd.action.room);
        break;
      case "drawer": {
        // The palette returns focus to its opener first; opening in the next microtask lets the
        // tool remember that opener and return there when it closes.
        const drawer = cmd.action.drawer;
        queueMicrotask(() => openDrawer(drawer));
        break;
      }
      case "instrument": {
        const id = cmd.action.tab === "lens" ? "instrument-lens" : cmd.action.tab === "sae" ? "instrument-sae" : "instrument-probes";
        setInspectorTab(cmd.action.tab);
        clearSelection();
        goto("workbench");
        showConversation();
        void tick().then(() => document.getElementById(id)?.scrollIntoView({ block: "start", behavior: "smooth" }));
        break;
      }
    }
  }

  function onKey(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      ev.stopPropagation();
      closePalette();
    } else if (ev.key === "ArrowDown") {
      ev.preventDefault();
      selected = Math.min(filtered.length - 1, selected + 1);
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      selected = Math.max(0, selected - 1);
    } else if (ev.key === "Enter") {
      ev.preventDefault();
      const cmd = filtered[selected];
      if (cmd) run(cmd);
    }
  }
</script>

{#if paletteState.open}
  <div class="scrim" role="button" tabindex="-1" aria-label="Close search" onclick={closePalette}
    onkeydown={(ev) => { if (ev.key === "Enter" || ev.key === " ") closePalette(); }} in:fadeIn out:fadeOut></div>
  <div bind:this={paletteEl} class="palette" role="dialog" aria-modal="true" aria-label="Search everything" tabindex="-1"
    onkeydown={(event) => { if (paletteEl) containTab(event, paletteEl); }} in:rise out:fadeOut>
    <div class="box">
      <span class="glass"><Icon name="search" size={18} /></span>
      <input
        bind:this={inputEl}
        bind:value={query}
        type="search"
        role="combobox"
        autocomplete="off"
        spellcheck="false"
        aria-autocomplete="list"
        aria-controls="command-palette-listbox"
        aria-expanded="true"
        aria-activedescendant={filtered.length > 0 ? `command-palette-option-${selected}` : undefined}
        placeholder="Search rooms, instruments and tools"
        aria-label="Search everything"
        onkeydown={onKey}
      />
      <kbd>esc</kbd>
    </div>
    <div id="command-palette-listbox" class="options" bind:this={listEl} role="listbox" aria-label="Results" use:glide>
      {#if filtered.length === 0}
        <p class="none meta" role="status">Nothing matches “{query.trim()}”</p>
      {:else}
        {#each filtered as cmd, i (cmd.group + cmd.label)}
          <button id={`command-palette-option-${i}`} type="button" class="option" role="option" tabindex="-1"
            aria-selected={i === selected} onclick={() => run(cmd)} onmousemove={() => (selected = i)}>
            <span class="label">{cmd.label}</span>
            <span class="meta">{cmd.group}</span>
          </button>
        {/each}
      {/if}
    </div>
  </div>
{/if}

<style>
  .scrim { position: fixed; inset: 0; z-index: 52; background: color-mix(in oklab, var(--wash2) 88%, transparent); -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px); }
  .palette {
    position: fixed;
    top: 12dvh;
    left: 50%;
    z-index: 53;
    display: grid;
    gap: var(--s2);
    width: min(560px, calc(100vw - var(--s4) * 2));
    max-height: 72dvh;
    padding: var(--s2);
    translate: -50% 0;
    border-radius: var(--r-row);
    background: var(--paper);
  }
  .palette:focus { outline: none; }
  .box { position: relative; height: 48px; border-radius: var(--r-box); background: var(--wash); color: var(--ink3); transition: color var(--fast) var(--ease); }
  .box:focus-within { color: var(--ink); }
  .glass { position: absolute; inset: 0 auto 0 var(--s4); display: grid; place-items: center; pointer-events: none; }
  input { display: block; width: 100%; height: 100%; padding: 0 56px 0 46px; border: 0; border-radius: var(--r-box); background: transparent; color: var(--ink); font: var(--type-body); outline: none; appearance: none; }
  input::placeholder { color: var(--ink3); opacity: 1; }
  input::-webkit-search-cancel-button, input::-webkit-search-decoration { appearance: none; }
  .box:has(input:focus-visible) { outline: 2px solid var(--ink); outline-offset: 2px; }
  kbd { position: absolute; top: 12px; right: var(--s3); display: grid; place-items: center; min-width: 24px; height: 24px; padding: 0 var(--s2); border-radius: var(--r-chip); background: var(--paper); color: var(--ink3); font: var(--type-meta); pointer-events: none; }
  .options { --glide-bg: var(--wash2); --glide-radius: var(--r-col); display: grid; min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding: 0 var(--s1) var(--s1); }
  .option { display: flex; align-items: baseline; justify-content: space-between; gap: var(--s3); min-height: 40px; padding: 10px var(--s3); border: 0; border-radius: var(--r-col); background: transparent; color: var(--ink2); font: var(--type-small); text-align: left; cursor: pointer; transition: color var(--fast) var(--ease); }
  .option[aria-selected="true"] { color: var(--ink); }
  .label { min-width: 0; overflow: hidden; font-weight: 500; white-space: nowrap; text-overflow: ellipsis; }
  .option .meta { flex: none; }
  .none { margin: 0; padding: var(--s2) var(--s3) var(--s3); }
  @media (max-width: 720px) { .palette { top: var(--s4); } }
</style>
