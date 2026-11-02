<script lang="ts" module>
  export interface MenuItem {
    label: string;
    run: () => void;
    disabled?: boolean;
    /** The item for the page you are on (a room, on the phone). */
    current?: boolean;
  }
  export interface MenuGroup {
    label?: string;
    items: MenuItem[];
  }
</script>

<script lang="ts">
  import { tick, type Snippet } from "svelte";
  import { Icon, fadeOut, rise } from "@stdlogits/ledger";

  // The header's menu: the conversation's files, settings, help and feedback (and on a phone the
  // rooms), as a wash2 surface under its button. No shadow: the surface's fill sets it apart, as on
  // the phone canvas. Escape, a click outside or picking an item closes it; focus returns to the
  // button.
  let { groups, label = "Menu", extra }: { groups: MenuGroup[]; label?: string; extra?: Snippet } = $props();

  let open = $state(false);
  let button: HTMLButtonElement | null = $state(null);
  let menu: HTMLElement | null = $state(null);
  const id = $props.id();

  async function toggle(): Promise<void> {
    open = !open;
    if (!open) return;
    await tick();
    menu?.querySelector<HTMLElement>("[role='menuitem']:not(:disabled)")?.focus();
  }

  /** Focus the menu's button (where a dialog its item opened gives focus back). */
  export function focus(): void {
    button?.focus({ preventScroll: true });
  }

  function close(restore = true): void {
    open = false;
    if (restore) button?.focus({ preventScroll: true });
  }

  function pick(item: MenuItem): void {
    close(false);
    item.run();
  }

  function onKey(event: KeyboardEvent): void {
    const items = [...(menu?.querySelectorAll<HTMLElement>("[role='menuitem']:not(:disabled)") ?? [])];
    const at = items.indexOf(document.activeElement as HTMLElement);
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      close();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const step = event.key === "ArrowDown" ? 1 : -1;
      items[(at + step + items.length) % items.length]?.focus();
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      items[event.key === "Home" ? 0 : items.length - 1]?.focus();
    } else if (event.key === "Tab") {
      close(false);
    }
  }
</script>

<div class="menu-host">
  <button bind:this={button} type="button" class="menu-button" aria-label={label} data-tip={label}
    aria-haspopup="menu" aria-expanded={open} aria-controls={id} onclick={() => void toggle()}>
    <Icon name="menu" />
  </button>
  {#if open}
    <button type="button" class="scrim" tabindex="-1" aria-label="Close the menu" onclick={() => close()}></button>
    <!-- svelte-ignore a11y_interactive_supports_focus -->
    <div bind:this={menu} class="menu" id={id} role="menu" aria-label={label} tabindex="-1" onkeydown={onKey} in:rise out:fadeOut>
      {#each groups as group, g (g)}
        {#if group.items.length}
          <div class="group" role="group" aria-label={group.label}>
            {#if group.label}<span class="head" aria-hidden="true">{group.label}</span>{/if}
            {#each group.items as item (item.label)}
              <button type="button" role="menuitem" class="item" aria-current={item.current ? "page" : undefined}
                disabled={item.disabled} onclick={() => pick(item)}>
                <span>{item.label}</span>{#if item.current}<Icon name="check" />{/if}
              </button>
            {/each}
          </div>
        {/if}
      {/each}
      {#if extra}<div class="extra">{@render extra()}</div>{/if}
    </div>
  {/if}
</div>

<style>
  .menu-host { position: relative; }
  .menu-button {
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
    padding: 0;
    border: 0;
    border-radius: var(--r-pill);
    background: var(--wash);
    color: var(--ink2);
    cursor: pointer;
    transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease), scale var(--fast) var(--ease);
  }
  .menu-button:active { scale: 0.94; }
  .menu-button[aria-expanded="true"] { background: var(--wash2); color: var(--ink); }
  @media (hover: hover) { .menu-button:hover { background: var(--wash2); color: var(--ink); } }
  .scrim { position: fixed; inset: 0; z-index: 40; padding: 0; border: 0; background: transparent; cursor: default; }
  .menu {
    position: absolute;
    top: calc(100% + var(--s2));
    right: 0;
    z-index: 41;
    display: grid;
    gap: var(--s2);
    width: 248px;
    max-width: calc(100vw - var(--s4) * 2);
    max-height: calc(100dvh - 96px);
    overflow-y: auto;
    padding: var(--s1);
    border-radius: var(--r-box);
    background: var(--wash2);
  }
  .menu:focus { outline: none; }
  .group { display: grid; }
  .head { padding: var(--s2) var(--s3) var(--s1); }
  .item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--s2);
    min-height: 40px;
    padding: 0 var(--s3);
    border: 0;
    border-radius: var(--r-col);
    background: transparent;
    color: var(--ink2);
    font: var(--type-control);
    text-align: left;
    cursor: pointer;
    transition: color var(--fast) var(--ease), background-color var(--fast) var(--ease), scale var(--fast) var(--ease);
  }
  .item[aria-current="page"] { color: var(--ink); }
  .item:active:not(:disabled) { scale: 0.97; }
  .item:disabled { opacity: 0.4; cursor: default; }
  .item:focus-visible { outline-offset: -2px; }
  @media (hover: hover) { .item:hover:not(:disabled) { color: var(--ink); background: var(--paper); } }
  .extra { display: flex; align-items: center; justify-content: space-between; padding: var(--s1) var(--s1) var(--s1) var(--s3); color: var(--ink2); font: var(--type-control); }
  @media (pointer: coarse) { .menu-button { width: 44px; height: 44px; } .item { min-height: 44px; } }
</style>
