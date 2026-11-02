// The Slider's pointer drag, store-free so a script test can drive it. The Slider runs a drag itself:
// pointerdown is prevented (the thumb follows the pointer from where it was grabbed, and a touch
// does not scroll the page), so the browser's own range drag never starts and never fires change.
// The drag sets the input's value as the pointer moves and reports each tick; when it ends (released,
// cancelled or its capture lost) on a value other than the one it started at, it dispatches one
// change on the input, as a keyboard step does, so a listener for change hears a drag once, at its end.

/** The thumb's diameter in px (the Slider's CSS draws it). */
export const THUMB = 14;

/** Grabbing within this many px of the thumb's centre keeps that offset while dragging. */
const GRAB = 12;

export interface SliderRange {
  min: number;
  max: number;
  step: number;
  disabled: boolean;
}

/** Where ``value`` sits along a ``min``..``max`` track, from 0 to 1. */
export function fractionOf(value: number, min: number, max: number): number {
  return max === min ? 0 : Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function rtl(input: HTMLInputElement): boolean {
  return getComputedStyle(input).direction === "rtl";
}

/**
 * The handlers for one Slider's input: ``start`` on pointerdown, ``move`` on pointermove and ``end``
 * on pointerup, pointercancel and lostpointercapture. ``range`` reads the props at each event;
 * ``report`` gets every value the drag sets, ``activate`` whether a drag is under way.
 */
export function sliderDrag(
  range: () => SliderRange,
  report: (value: number) => void,
  activate: (active: boolean) => void,
) {
  let drag: { id: number; offset: number; from: string } | null = null;

  function move(event: PointerEvent): void {
    const { min, max, step, disabled } = range();
    if (!drag || event.pointerId !== drag.id || disabled) return;
    const input = event.currentTarget as HTMLInputElement;
    const rect = input.getBoundingClientRect();
    const width = Math.max(1, rect.width - THUMB);
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - THUMB / 2 - drag.offset) / width));
    const raw = min + (rtl(input) ? 1 - fraction : fraction) * (max - min);
    input.value = String(Math.max(min, Math.min(max, min + Math.round((raw - min) / step) * step)));
    report(input.valueAsNumber);
  }

  function start(event: PointerEvent): void {
    const { min, max, disabled } = range();
    if (disabled || event.button !== 0 || !event.isPrimary) return;
    event.preventDefault();
    const input = event.currentTarget as HTMLInputElement;
    const rect = input.getBoundingClientRect();
    const fraction = fractionOf(input.valueAsNumber, min, max);
    const center = rect.left + THUMB / 2 + (rtl(input) ? 1 - fraction : fraction) * (rect.width - THUMB);
    const offset = event.clientX - center;
    activate(true);
    drag = { id: event.pointerId, offset: Math.abs(offset) <= GRAB ? offset : 0, from: input.value };
    input.focus({ preventScroll: true });
    input.setPointerCapture(event.pointerId);
    move(event);
  }

  function end(event: PointerEvent): void {
    if (drag?.id !== event.pointerId) return;
    const { from } = drag;
    drag = null;
    activate(false);
    const input = event.currentTarget as HTMLInputElement;
    if (input.hasPointerCapture(event.pointerId)) input.releasePointerCapture(event.pointerId);
    if (input.value !== from) input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  return { start, move, end };
}
