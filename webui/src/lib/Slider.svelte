<script lang="ts">
  import { onMount } from "svelte";
  import { hideTip, showTip } from "@stdlogits/ledger";
  // The one range slider (sampling, steering strengths, the steering picker): a 4 px wash2 groove
  // and a 14 px ink thumb, as the Instruments canvas draws it. While it is dragged or stepped from
  // the keyboard, Ledger's tip shows the value above the thumb.
  //
  // Deliberately dumb: it reports the raw value through ``oninput`` and ``bind:value``; callers own
  // any snapping (the steering strip's 0-detent, say).

  interface Props {
    /** Current value — bindable. */
    value: number;
    min?: number;
    max?: number;
    step?: number;
    disabled?: boolean;
    ariaLabel?: string;
    title?: string;
    displayValue?: string;
    /** Fired on every drag tick with the raw (un-snapped) value. */
    oninput?: (value: number) => void;
  }

  let {
    value = $bindable(),
    min = 0,
    max = 1,
    step = 0.01,
    disabled = false,
    ariaLabel,
    title,
    displayValue,
    oninput,
  }: Props = $props();

  /** The thumb's diameter in px (the CSS below draws it). */
  const THUMB = 14;

  let inputEl: HTMLInputElement;
  let active = $state(false);
  let tipShown = false;
  const formattedValue = $derived(displayValue ?? value.toFixed(Math.min(6, (String(step).split(".")[1] ?? "").length)));

  function handle(ev: Event): void {
    const v = parseFloat((ev.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(v)) return;
    active = true;
    value = v;
    oninput?.(v);
  }

  function fractionOf(v: number): number {
    return max === min ? 0 : Math.max(0, Math.min(1, (v - min) / (max - min)));
  }

  /** The tip over the thumb, or none when the slider is idle or scrolled away. */
  function placeTip(): void {
    if (!inputEl) return;
    if (!active || disabled) { hideValue(); return; }
    const rect = inputEl.getBoundingClientRect();
    if (rect.bottom <= 0 || rect.top >= window.innerHeight) { hideValue(); return; }
    const rtl = getComputedStyle(inputEl).direction === "rtl";
    const fraction = fractionOf(value);
    const x = rect.left + THUMB / 2 + (rtl ? 1 - fraction : fraction) * (rect.width - THUMB);
    const y = rect.top + rect.height / 2;
    showTip(new DOMRect(x - THUMB / 2, y - THUMB / 2, THUMB, THUMB), { title: formattedValue });
    tipShown = true;
  }

  function hideValue(): void {
    if (!tipShown) return;
    tipShown = false;
    hideTip();
  }

  onMount(() => {
    window.addEventListener("resize", placeTip);
    document.addEventListener("scroll", placeTip, true);
    return () => {
      window.removeEventListener("resize", placeTip);
      document.removeEventListener("scroll", placeTip, true);
      hideValue();
    };
  });

  $effect(() => {
    void value;
    void formattedValue;
    void active;
    void disabled;
    placeTip();
  });

  let drag: { id: number; offset: number } | null = null;

  function move(event: PointerEvent): void {
    if (!drag || event.pointerId !== drag.id || disabled) return;
    const input = event.currentTarget as HTMLInputElement;
    const rect = input.getBoundingClientRect();
    const width = Math.max(1, rect.width - THUMB);
    const rtl = getComputedStyle(input).direction === "rtl";
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - THUMB / 2 - drag.offset) / width));
    const raw = min + (rtl ? 1 - fraction : fraction) * (max - min);
    input.value = String(Math.max(min, Math.min(max, min + Math.round((raw - min) / step) * step)));
    value = input.valueAsNumber;
    oninput?.(value);
  }

  function start(event: PointerEvent): void {
    if (disabled || event.button !== 0 || !event.isPrimary) return;
    event.preventDefault();
    const input = event.currentTarget as HTMLInputElement;
    const rect = input.getBoundingClientRect();
    const fraction = fractionOf(value);
    const center = rect.left + THUMB / 2 + (getComputedStyle(input).direction === "rtl" ? 1 - fraction : fraction) * (rect.width - THUMB);
    const offset = event.clientX - center;
    active = true;
    drag = { id: event.pointerId, offset: Math.abs(offset) <= 12 ? offset : 0 };
    input.focus({ preventScroll: true });
    input.setPointerCapture(event.pointerId);
    move(event);
  }

  function end(event: PointerEvent): void {
    if (drag?.id !== event.pointerId) return;
    drag = null;
    active = false;
    const input = event.currentTarget as HTMLInputElement;
    if (input.hasPointerCapture(event.pointerId)) input.releasePointerCapture(event.pointerId);
  }
</script>

<input
  bind:this={inputEl}
  onfocus={() => active = true}
  onblur={() => active = false}
  class="sk-slider"
  type="range"
  {min}
  {max}
  {step}
  value={value}
  {disabled}
  {...{ "aria-description": (title) }}
  aria-label={ariaLabel}
  aria-valuetext={formattedValue}
  oninput={handle}
  onpointerdown={start}
  onpointermove={move}
  onpointerup={end}
  onpointercancel={end}
  onlostpointercapture={end}
/>

<style>
  .sk-slider {
    -webkit-appearance: none;
    appearance: none;
    display: block;
    width: 100%;
    height: 32px;
    margin: 0;
    padding: 0;
    border: 0;
    border-radius: var(--r-chip);
    background: transparent;
    touch-action: none;
    -webkit-user-select: none;
    user-select: none;
    cursor: ew-resize;
  }
  .sk-slider:disabled { opacity: 0.4; cursor: default; }
  .sk-slider::-webkit-slider-runnable-track {
    height: 4px;
    border: 0;
    border-radius: 2px;
    background: var(--wash2);
  }
  .sk-slider::-moz-range-track {
    height: 4px;
    border: 0;
    border-radius: 2px;
    background: var(--wash2);
  }
  .sk-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    margin-top: -5px;
    border: 0;
    border-radius: 50%;
    background: var(--ink);
    cursor: inherit;
    transition: scale var(--fast) var(--ease), background-color var(--fast) var(--ease);
  }
  .sk-slider::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border: 0;
    border-radius: 50%;
    background: var(--ink);
    cursor: inherit;
    transition: scale var(--fast) var(--ease), background-color var(--fast) var(--ease);
  }
  .sk-slider:hover:not(:disabled)::-webkit-slider-thumb,
  .sk-slider:active:not(:disabled)::-webkit-slider-thumb { scale: 1.15; }
  .sk-slider:hover:not(:disabled)::-moz-range-thumb,
  .sk-slider:active:not(:disabled)::-moz-range-thumb { scale: 1.15; }
  .sk-slider:disabled::-webkit-slider-thumb { background: var(--ink3); }
  .sk-slider:disabled::-moz-range-thumb { background: var(--ink3); }

  @media (max-width: 720px), (pointer: coarse) {
    .sk-slider { height: 44px; }
  }
</style>
