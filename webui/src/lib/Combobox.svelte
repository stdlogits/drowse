<script lang="ts">
  import { Icon } from "@stdlogits/ledger";
  // The editable combobox: free text (a role label, say) over a roster to pick from. A wash field
  // like every text field; its list opens as the same flat wash2 surface as Select's.
  import { onMount, tick } from "svelte";
  import { dropdownMotion } from "./dropdownMotion.svelte";

  interface Option { value: string; label: string; }
  interface Props {
    value: string;
    options: Option[];
    placeholder?: string;
    disabled?: boolean;
    invalid?: boolean;
    ariaLabel?: string;
    ariaDescribedby?: string;
    title?: string;
    spellcheck?: boolean;
    onchange?: (value: string) => void;
  }

  let {
    value = $bindable(), options, placeholder = "", disabled = false,
    invalid = false, ariaLabel, ariaDescribedby, title, spellcheck = false, onchange,
  }: Props = $props();

  let open = $state(false);
  const presence = dropdownMotion();
  let filtering = $state(false);
  let highlight = $state(-1);
  let input: HTMLInputElement | null = $state(null);
  let listbox: HTMLUListElement | null = $state(null);
  let popoverStyle = $state("");
  const uid = $props.id();

  $effect(() => { if (disabled && open) closePopover(); });

  const visible = $derived.by(() => {
    if (!filtering || !value.trim()) return options;
    const needle = value.trim().toLowerCase();
    return options.filter((opt) => opt.label.toLowerCase().includes(needle));
  });

  async function openPopover(filter = false): Promise<void> {
    if (disabled) return;
    filtering = filter;
    open = true;
    presence.mount();
    highlight = visible.findIndex((opt) => opt.value === value);
    if (highlight < 0 && visible.length > 0) highlight = 0;
    await tick();
    if (!open) return;
    try { listbox?.showPopover(); } catch { /* fixed fallback */ }
    placeListbox();
    await tick();
    if (open && listbox) presence.show(listbox);
  }

  function closePopover(): void {
    if (!open) return;
    open = false;
    presence.close(listbox);
  }

  async function commit(index: number): Promise<void> {
    const opt = visible[index];
    if (!opt) return;
    value = opt.value;
    onchange?.(value);
    closePopover();
    await tick();
    input?.focus();
    input?.setSelectionRange(value.length, value.length);
  }

  function onInput(ev: Event): void {
    value = (ev.currentTarget as HTMLInputElement).value;
    onchange?.(value);
    void openPopover(true);
  }

  function move(dir: 1 | -1): void {
    if (visible.length === 0) return;
    highlight = (highlight + dir + visible.length) % visible.length;
    const row = listbox?.children[highlight] as HTMLElement | undefined;
    row?.scrollIntoView({ block: "nearest" });
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      ev.preventDefault();
      if (!open) void openPopover(false);
      else move(ev.key === "ArrowDown" ? 1 : -1);
    } else if (ev.key === "Enter" && open) {
      ev.preventDefault();
      if (highlight >= 0) void commit(highlight);
      else closePopover();
    } else if (ev.key === "Escape" && open) {
      ev.preventDefault();
      ev.stopPropagation();
      closePopover();
    } else if (ev.key === "Tab") {
      closePopover();
    }
  }

  function placeListbox(): void {
    if (!input || !listbox) return;
    const r = input.parentElement?.getBoundingClientRect()
      ?? input.getBoundingClientRect();
    const gutter = 8, gap = 2;
    const viewport = window.visualViewport;
    const viewportLeft = viewport?.offsetLeft ?? 0;
    const viewportTop = viewport?.offsetTop ?? 0;
    const viewportWidth = viewport?.width ?? window.innerWidth;
    const viewportBottom = viewportTop + (viewport?.height ?? window.innerHeight);
    const desired = Math.min(240, Math.max(32, listbox.scrollHeight));
    const below = viewportBottom - r.bottom - gutter - gap;
    const above = r.top - viewportTop - gutter - gap;
    const upward = below < Math.min(desired, 160) && above > below;
    const maxHeight = Math.max(32, Math.min(240, upward ? above : below));
    const height = Math.min(desired, maxHeight);
    const width = Math.min(r.width, viewportWidth - gutter * 2);
    const left = Math.max(viewportLeft + gutter, Math.min(r.left, viewportLeft + viewportWidth - width - gutter));
    const top = upward
      ? Math.max(viewportTop + gutter, r.top - height - gap)
      : Math.min(viewportBottom - height - gutter, r.bottom + gap);
    popoverStyle = `left:${left}px;top:${top}px;width:${width}px;max-height:${maxHeight}px`;
    listbox.dataset.origin = upward ? "bottom-left" : "top-left";
  }

  function onDocumentPointer(ev: PointerEvent): void {
    if (!open) return;
    const target = ev.target as Node;
    if (input?.parentElement?.contains(target) || listbox?.contains(target)) return;
    closePopover();
  }
  function reposition(): void { if (open) placeListbox(); }

  onMount(() => {
    document.addEventListener("pointerdown", onDocumentPointer, true);
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    window.visualViewport?.addEventListener("resize", reposition);
    window.visualViewport?.addEventListener("scroll", reposition);
    return () => {
      document.removeEventListener("pointerdown", onDocumentPointer, true);
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
      window.visualViewport?.removeEventListener("resize", reposition);
      window.visualViewport?.removeEventListener("scroll", reposition);
      presence.destroy();
    };
  });
</script>

<div class="sk-combobox" class:is-open={open} class:is-invalid={invalid}>
  <input
    bind:this={input} bind:value {placeholder} {disabled} {...{ "aria-description": (title) }} {spellcheck}
    role="combobox" aria-label={ariaLabel} aria-autocomplete="list"
    aria-describedby={ariaDescribedby}
    aria-expanded={open} aria-controls={open ? `${uid}-listbox` : undefined}
    aria-activedescendant={open && highlight >= 0 ? `${uid}-option-${highlight}` : undefined}
    aria-invalid={invalid}
    oninput={onInput} onkeydown={onKeydown}
  />
  <button
    type="button" class="caret" aria-label={`Choose ${ariaLabel ?? "value"}`}
    tabindex="-1" {disabled}
    onclick={() => (open ? closePopover() : void openPopover(false))}
  ><Icon name="down" /></button>

  {#if presence.mounted}
    <ul bind:this={listbox} id={`${uid}-listbox`} class="popover t-dropdown"
      popover="manual" style={popoverStyle} role="listbox" aria-label={ariaLabel}>
      {#each visible as opt, i (opt.value)}
        <li id={`${uid}-option-${i}`} role="option"
          class:highlight={i === highlight} class:active={opt.value === value}
          aria-selected={opt.value === value}
          tabindex="-1"
          onpointerenter={() => (highlight = i)}
          onpointerdown={(ev) => ev.preventDefault()}
          onkeydown={(ev) => {
            if (ev.key === "Enter" || ev.key === " ") void commit(i);
          }}
          onclick={() => void commit(i)}><span class="text">{opt.label}</span>{#if opt.value === value}<span class="check" aria-hidden="true"><Icon name="check" /></span>{/if}</li>
      {:else}
        <li class="empty" role="option" aria-selected="false" aria-disabled="true">
          Custom role: “{value.trim()}”
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .sk-combobox {
    position: relative;
    display: flex;
    align-items: center;
    width: 100%;
    min-width: 0;
    min-height: 36px;
    border-radius: var(--r-col);
    background: var(--field);
    transition: background-color var(--fast) var(--ease);
  }
  .sk-combobox:hover, .sk-combobox.is-open { background: var(--field-hover); }
  /* Ledger has no red: an invalid field darkens toward ink. */
  .sk-combobox.is-invalid { background: color-mix(in oklab, var(--ink) 14%, var(--field)); }
  input {
    width: 100%;
    min-width: 0;
    min-height: 36px;
    padding: 0 40px 0 var(--s3);
    border: 0;
    border-radius: inherit;
    background: transparent;
    color: var(--ink);
    font: var(--type-small);
  }
  input:disabled, input:disabled + .caret { opacity: 0.4; cursor: default; }
  .caret {
    position: absolute;
    inset-inline-end: 0;
    display: grid;
    place-items: center;
    width: 36px;
    height: 100%;
    padding: 0;
    border: 0;
    border-radius: inherit;
    background: transparent;
    color: var(--ink3);
    cursor: pointer;
    transition: color var(--fast) var(--ease), rotate var(--glide) var(--ease);
  }
  .caret:hover:not(:disabled) { color: var(--ink); }
  .is-open .caret { rotate: 180deg; color: var(--ink); }

  /* The list: Select's flat wash2 surface, rising 6 px as it fades in. */
  .popover {
    --dropdown-close-dur: var(--out);
    position: fixed;
    inset: auto;
    z-index: 60;
    box-sizing: border-box;
    margin: 0;
    padding: var(--s1);
    overflow-y: auto;
    list-style: none;
    border: 0;
    border-radius: var(--r-box);
    background: var(--wash2);
    color: var(--ink2);
    opacity: 0;
    translate: 0 6px;
    transition: opacity var(--in) var(--ease), translate var(--in) var(--ease);
  }
  .popover[data-origin="bottom-left"] { translate: 0 -6px; }
  .popover.is-open { opacity: 1; translate: 0 0; }
  .popover.is-closing { opacity: 0; translate: 0 0; transition-duration: var(--out); }
  li {
    display: flex;
    align-items: center;
    gap: var(--s2);
    min-height: 40px;
    padding: var(--s2) var(--s3);
    border-radius: var(--r-col);
    color: var(--ink2);
    font: var(--type-control);
    overflow-wrap: anywhere;
    cursor: pointer;
    transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease);
  }
  li .text { flex: 1 1 auto; min-width: 0; }
  li .check { display: grid; flex: none; }
  li.active { color: var(--ink); }
  li.highlight { background: var(--paper); color: var(--ink); }
  li.empty { color: var(--ink3); cursor: default; }

  @media (max-width: 720px), (pointer: coarse) {
    .sk-combobox, input, li { min-height: 44px; }
    .caret { width: 44px; }
  }
  @media (prefers-reduced-motion: reduce) {
    .popover, .popover[data-origin="bottom-left"] { translate: none; }
  }
</style>
