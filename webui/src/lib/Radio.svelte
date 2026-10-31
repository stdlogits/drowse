<script lang="ts" generics="T extends string | number">
  // One radio of a group: one ``<Radio />`` per option, sharing a ``group`` binding inside a
  // ``role="radiogroup"`` (arrow keys move through it). Generic over the value type so callers
  // keep their enums.

  interface Props {
    /** This radio's value — committed into ``group`` on click. */
    value: T;
    /** The shared selection — bind to the same variable across the
     *  group of radios. */
    group: T;
    disabled?: boolean;
    ariaLabel?: string;
    title?: string;
    label?: string;
    name?: string;
    onchange?: (value: T) => void;
  }

  let {
    value,
    group = $bindable(),
    disabled = false,
    ariaLabel,
    title,
    label,
    name,
    onchange,
  }: Props = $props();

  const selected = $derived(group === value);

  function pick(): void {
    if (disabled || selected) return;
    group = value;
    onchange?.(value);
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (disabled) return;
    if (ev.key === " " || ev.key === "Enter") {
      ev.preventDefault();
      pick();
      return;
    }
    const groupEl = (ev.currentTarget as HTMLElement).closest('[role="radiogroup"]');
    if (!groupEl) return;
    const radios = [...groupEl.querySelectorAll<HTMLButtonElement>('[role="radio"]')].filter(
      (radio) => !radio.disabled,
    );
    const current = radios.indexOf(ev.currentTarget as HTMLButtonElement);
    if (current < 0 || radios.length === 0) return;
    let next = current;
    const rtl = getComputedStyle(groupEl).direction === "rtl";
    if (ev.key === "ArrowDown" || ev.key === "ArrowRight") {
      const delta = ev.key === "ArrowRight" && rtl ? -1 : 1;
      next = (current + delta + radios.length) % radios.length;
    } else if (ev.key === "ArrowUp" || ev.key === "ArrowLeft") {
      const delta = ev.key === "ArrowLeft" && rtl ? 1 : -1;
      next = (current + delta + radios.length) % radios.length;
    }
    else if (ev.key === "Home") next = 0;
    else if (ev.key === "End") next = radios.length - 1;
    else return;
    ev.preventDefault();
    radios[next].focus();
    radios[next].click();
  }
</script>

<button
  type="button"
  role="radio"
  class="sk-radio"
  class:is-selected={selected}
  class:is-disabled={disabled}
  aria-checked={selected}
  aria-label={ariaLabel}
  tabindex={selected && !disabled ? 0 : -1}
  data-name={name}
  {disabled}
  {...{ "aria-description": (title) }}
  onclick={pick}
  onkeydown={onKeydown}
>
  <span class="sk-radio-control" aria-hidden="true">
    <span class="sk-radio-box" aria-hidden="true">
      <span class="sk-radio-dot" class:is-visible={selected}></span>
    </span>
  </span>
  {#if label}
    <span class="sk-radio-label">{label}</span>
  {/if}
</button>

<style>
  /* Drawn like Ledger's Switch: a wash2 disc that fills with ink when chosen, a paper dot swelling
   * in at its centre (240 ms); the label steps from ink2 to ink. */
  .sk-radio {
    display: inline-flex;
    align-items: center;
    gap: var(--s2);
    min-height: 36px;
    padding: 0 var(--s2) 0 0;
    border: 0;
    border-radius: var(--r-col);
    background: transparent;
    color: var(--ink2);
    font: var(--type-control);
    text-align: start;
    cursor: pointer;
    transition: color var(--fast) var(--ease), scale var(--fast) var(--ease);
  }
  .sk-radio:active:not(:disabled) { scale: 0.97; }
  .sk-radio.is-selected { color: var(--ink); }
  .sk-radio.is-disabled { opacity: 0.4; cursor: default; }
  @media (hover: hover) { .sk-radio:hover:not(.is-disabled) { color: var(--ink); } }
  .sk-radio-control {
    display: inline-grid;
    flex: none;
    place-items: center;
    width: 20px;
    height: 20px;
  }
  .sk-radio-box {
    display: grid;
    place-items: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--wash2);
    transition: background-color var(--glide) var(--ease);
  }
  .sk-radio.is-selected .sk-radio-box { background: var(--ink); }
  .sk-radio-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--paper);
    opacity: 0;
    scale: 0.25;
    pointer-events: none;
    transition: opacity var(--glide) var(--ease), scale var(--glide) var(--ease);
  }
  .sk-radio-dot.is-visible { opacity: 1; scale: 1; }
  .sk-radio-label { min-width: 0; }

  @media (pointer: coarse) {
    .sk-radio { min-height: 44px; }
  }
</style>
