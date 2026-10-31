<script lang="ts">
  import { Icon, MorphText } from "@stdlogits/ledger";
  // The numeric field: a wash field without the OS spinners, with a quiet ▴/▾ pair at the right
  // edge on hover. Keyboard ↑/↓ still work natively (the ``<input type="number">`` is unchanged);
  // the steppers give a pointer the same steps.
  //
  // Drop-in for ``<input type="number">``: same ``min`` / ``max`` /
  // ``step`` / ``placeholder`` props, ``bind:value`` to a number (with
  // ``allowEmpty`` if the caller treats blank as ``null``).

  interface Props {
    /** Bindable numeric value.  When ``allowEmpty`` is true and the
     *  field is blank the binding receives ``null``. */
    value: number | null;
    min?: number;
    max?: number;
    step?: number;
    placeholder?: string;
    disabled?: boolean;
    invalid?: boolean;
    ariaLabel?: string;
    ariaDescribedby?: string;
    title?: string;
    /** Allow a blank state — emits ``null`` on empty.  Default false. */
    allowEmpty?: boolean;
    /** Fires on every keystroke (after numeric coercion).  Bind:value
     *  is updated regardless. */
    oninput?: (value: number | null) => void;
    /** Fires only when the input commits (blur or Enter), mirroring the
     *  native ``change`` event.  Use this when the parent should defer
     *  expensive side-effects (network PATCH, etc.) until the user
     *  finishes typing. */
    onchange?: (value: number | null) => void;
    /** Forwarded to the inner ``<input>`` — modal-style Enter-to-commit
     *  hosts can hook this without an extra wrapper. */
    onkeydown?: (ev: KeyboardEvent) => void;
  }

  let {
    value = $bindable(),
    min,
    max,
    step = 1,
    placeholder,
    disabled = false,
    invalid = false,
    ariaLabel,
    ariaDescribedby,
    title,
    allowEmpty = false,
    oninput,
    onchange,
    onkeydown,
  }: Props = $props();

  let editing = $state(false);
  let inputEl: HTMLInputElement | null = $state(null);

  /** Imperative focus — bound parents can call ``ref.focus()`` to drop
   *  the cursor into the input, same as a native ``HTMLInputElement``. */
  export function focus(): void {
    inputEl?.focus();
  }
  /** Imperative select — mirrors ``HTMLInputElement.select`` so the
   *  modal-style "focus + select existing value for fast retype" pattern
   *  still works after the chrome swap. */
  export function select(): void {
    inputEl?.select();
  }

  function clamp(v: number | null): number | null {
    if (v === null) return null;
    let next = v;
    if (min !== undefined && next < min) next = min;
    if (max !== undefined && next > max) next = max;
    return next;
  }

  function emitInput(v: number | null): void {
    const next = clamp(v);
    if (inputEl && next !== v) inputEl.value = next === null ? "" : String(next);
    value = next;
    oninput?.(next);
  }

  function emitChange(v: number | null): void {
    const next = clamp(v);
    if (inputEl && next !== v) inputEl.value = next === null ? "" : String(next);
    value = next;
    onchange?.(next);
  }

  function onInput(ev: Event): void {
    const raw = (ev.currentTarget as HTMLInputElement).value;
    if (raw === "") {
      emitInput(allowEmpty ? null : 0);
      return;
    }
    const v = Number(raw);
    if (!Number.isFinite(v)) return;
    emitInput(v);
  }

  function onChange(ev: Event): void {
    const raw = (ev.currentTarget as HTMLInputElement).value;
    if (raw === "") {
      emitChange(allowEmpty ? null : 0);
      return;
    }
    const v = Number(raw);
    if (!Number.isFinite(v)) return;
    emitChange(v);
  }

  function handleKeydown(ev: KeyboardEvent): void {
    onkeydown?.(ev);
    if (ev.defaultPrevented || ev.key !== "Enter") return;
    // Own Enter completely.  Letting the native number input / surrounding
    // form continue after our explicit commit can dispatch a second change or
    // activate a neighbouring default button; in the sampling strip that
    // turned a just-entered max-token value into the field minimum and could
    // start a generation with the wrong cap.
    ev.preventDefault();
    ev.stopPropagation();
    // Native number inputs do not consistently emit ``change`` on Enter
    // (Chrome waits for blur), despite Enter being the explicit commit
    // gesture promised by this component. Commit the current draft here so
    // keyboard-only users and form-adjacent controls apply immediately.
    const raw = inputEl?.value ?? "";
    if (raw === "") {
      emitChange(allowEmpty ? null : 0);
      return;
    }
    const v = Number(raw);
    if (Number.isFinite(v)) emitChange(v);
  }

  function nudge(direction: 1 | -1): void {
    if (disabled) return;
    const base = value ?? 0;
    let next = base + direction * step;
    // Round to ``step``'s decimal precision so 0.1 nudges stay clean.
    const decimals = (step.toString().split(".")[1] ?? "").length;
    next = Number(next.toFixed(decimals));
    // Stepper button is treated as a commit — fires both oninput and
    // onchange so hosts that only listen on commit also see it.
    emitInput(next);
    onchange?.(clamp(next));
    inputEl?.focus();
  }
</script>

<span class="sk-number" class:is-disabled={disabled} class:is-invalid={invalid} class:resting={!editing && value !== null}>
  <input
    bind:this={inputEl}
    onfocus={() => editing = true}
    onblur={() => editing = false}
    type="number"
    class="sk-number-input"
    value={value === null ? "" : value}
    {min}
    {max}
    {step}
    {placeholder}
    {disabled}
    {...{ "aria-description": (title) }}
    aria-label={ariaLabel}
    aria-invalid={invalid}
    aria-describedby={ariaDescribedby}
    oninput={onInput}
    onchange={onChange}
    onkeydown={handleKeydown}
  />
  {#if !editing && value !== null}
    <span class="number-rest" aria-hidden="true"><MorphText text={String(value)} /></span>
  {/if}
  {#if !disabled}
    <span class="sk-number-steppers" aria-hidden="true">
      <button
        type="button"
        class="sk-number-step"
        tabindex="-1"
        onclick={() => nudge(1)}
      ><Icon name="up" size={12} /></button>
      <button
        type="button"
        class="sk-number-step"
        tabindex="-1"
        onclick={() => nudge(-1)}
      ><Icon name="down" size={12} /></button>
    </span>
  {/if}
</span>

<style>
  /* A wash field like every text field. The value at rest is drawn over the input with MorphText,
   * so a change rolls rather than jumps; editing shows the input itself. */
  .sk-number {
    position: relative;
    display: inline-flex;
    align-items: stretch;
    width: 100%;
    min-width: 0;
    border-radius: var(--r-col);
  }
  .sk-number-input {
    flex: 1 1 0;
    width: 100%;
    min-width: 0;
    min-height: 36px;
    padding: 0 28px 0 var(--s3);
    border: 0;
    border-radius: inherit;
    background: var(--field);
    color: var(--ink);
    font: var(--type-small);
    font-variant-numeric: tabular-nums;
    -moz-appearance: textfield;
    appearance: textfield;
    transition: background-color var(--fast) var(--ease);
  }
  .sk-number-input::-webkit-outer-spin-button,
  .sk-number-input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  .sk-number:hover .sk-number-input:not(:disabled) { background: var(--field-hover); }
  /* Ledger has no red: an invalid field darkens toward ink. */
  .sk-number.is-invalid .sk-number-input { background: color-mix(in oklab, var(--ink) 14%, var(--field)); }
  .sk-number-input:disabled { cursor: default; }
  .resting .sk-number-input { color: transparent; }
  .number-rest {
    position: absolute;
    inset-inline: var(--s3) 28px;
    top: 50%;
    overflow: hidden;
    color: var(--ink);
    font: var(--type-small);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    transform: translateY(-50%);
    pointer-events: none;
  }

  /* The steppers: two quiet glyphs at the right edge, shown on hover and focus. */
  .sk-number-steppers {
    position: absolute;
    top: 2px;
    bottom: 2px;
    inset-inline-end: 2px;
    display: flex;
    flex-direction: column;
    width: 24px;
    opacity: 0;
    pointer-events: none;
    transition: opacity var(--fast) var(--ease);
  }
  .sk-number:hover .sk-number-steppers,
  .sk-number:focus-within .sk-number-steppers {
    opacity: 1;
    pointer-events: auto;
  }
  .sk-number-step {
    display: grid;
    flex: 1 1 0;
    place-items: center;
    padding: 0;
    border: 0;
    border-radius: var(--r-chip);
    background: transparent;
    color: var(--ink3);
    cursor: pointer;
    transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease);
  }
  .sk-number-step:hover { background: var(--paper); color: var(--ink); }

  .sk-number.is-disabled { opacity: 0.4; }

  @media (max-width: 720px) {
    .sk-number-input { min-height: 44px; padding-inline-end: var(--s3); }
    .number-rest { inset-inline-end: var(--s3); }
    .sk-number-steppers { display: none; }
  }
</style>
