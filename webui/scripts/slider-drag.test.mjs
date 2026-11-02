import assert from "node:assert/strict";
import { createServer } from "vite";

// The Slider's pointer drag (lib/sliderDrag.ts) over a stand-in input: the value follows the pointer
// in steps, and the end of a drag that moved it fires one change, as a keyboard step does, so a
// caller listening for change (drowse.ai's recorded demo announces there) hears drags too.
const server = await createServer({
  configFile: false, appType: "custom", logLevel: "error",
  optimizeDeps: { noDiscovery: true, include: [] },
  server: { middlewareMode: true, watch: null, hmr: false, ws: false },
});
const previousStyle = globalThis.getComputedStyle;
try {
  const { THUMB, sliderDrag } = await server.ssrLoadModule("/src/lib/sliderDrag.ts");
  let direction = "ltr";
  globalThis.getComputedStyle = () => ({ direction });

  // A range input 114 px wide at x = 0: the thumb's centre runs from 7 to 107, 100 px for 0..10.
  class Input extends EventTarget {
    value = "0";
    captured = null;
    focused = false;
    get valueAsNumber() { return Number(this.value); }
    getBoundingClientRect() { return { left: 0, top: 0, width: 100 + THUMB, height: 32 }; }
    focus() { this.focused = true; }
    setPointerCapture(id) { this.captured = id; }
    hasPointerCapture(id) { return this.captured === id; }
    releasePointerCapture(id) { if (this.captured === id) this.captured = null; }
  }
  const at = (fraction) => THUMB / 2 + fraction * 100;
  const pointer = (input, clientX, props = {}) => ({
    pointerId: 1, button: 0, isPrimary: true, clientX, currentTarget: input,
    prevented: false, preventDefault() { this.prevented = true; }, ...props,
  });
  function rig(range = {}) {
    const input = new Input();
    const changes = [];
    const ticks = [];
    const active = [];
    input.addEventListener("change", (event) => changes.push({ value: input.value, bubbles: event.bubbles }));
    const props = { min: 0, max: 10, step: 1, disabled: false, ...range };
    const drag = sliderDrag(() => props, (value) => ticks.push(value), (on) => active.push(on));
    return { input, changes, ticks, active, drag, props };
  }

  {
    // A drag across the track: every tick is reported, one change when it ends.
    const { input, changes, ticks, active, drag } = rig();
    const down = pointer(input, at(0.31));
    drag.start(down);
    assert.equal(down.prevented, true, "the slider runs the drag, not the browser");
    assert.equal(input.captured, 1);
    assert.equal(input.focused, true);
    drag.move(pointer(input, at(0.58)));
    drag.move(pointer(input, at(1.4)));
    assert.deepEqual(ticks, [3, 6, 10], "the value follows the pointer in steps and stops at max");
    assert.deepEqual(changes, [], "no change while the drag runs");
    drag.end(pointer(input, at(1.4)));
    assert.deepEqual(changes, [{ value: "10", bubbles: true }], "the drag's end commits once, and the event bubbles");
    assert.deepEqual(active, [true, false]);
    assert.equal(input.captured, null);
    // Releasing capture fires lostpointercapture after pointerup: the drag is over, so no second change.
    drag.end(pointer(input, at(1.4)));
    assert.equal(changes.length, 1);
  }
  {
    // Pressing the thumb where it sits and letting go changes nothing, so nothing is committed.
    const { input, changes, ticks, drag } = rig();
    input.value = "5";
    drag.start(pointer(input, at(0.5) + 4));
    drag.end(pointer(input, at(0.5) + 4));
    assert.deepEqual(ticks, [5], "a grab near the centre keeps its offset");
    assert.deepEqual(changes, []);
  }
  {
    // A drag that comes back to where it began commits nothing.
    const { input, changes, drag } = rig();
    input.value = "2";
    drag.start(pointer(input, at(0.2)));
    drag.move(pointer(input, at(0.7)));
    drag.move(pointer(input, at(0.2)));
    drag.end(pointer(input, at(0.2)));
    assert.deepEqual(changes, []);
  }
  {
    // A press on the track jumps the thumb there: a commit, as a click on a native range is.
    const { input, changes, drag } = rig();
    drag.start(pointer(input, at(0.8)));
    drag.end(pointer(input, at(0.8)));
    assert.deepEqual(changes, [{ value: "8", bubbles: true }]);
  }
  {
    // A cancelled drag keeps the value it reached, so it commits it too.
    const { input, changes, drag } = rig();
    drag.start(pointer(input, at(0.4)));
    drag.end(pointer(input, at(0.4), { type: "pointercancel" }));
    assert.deepEqual(changes, [{ value: "4", bubbles: true }]);
  }
  {
    // Another pointer, a secondary button or a disabled slider starts or ends nothing.
    const { input, changes, ticks, drag, props } = rig();
    drag.start(pointer(input, at(0.9), { button: 2 }));
    drag.start(pointer(input, at(0.9), { isPrimary: false }));
    assert.deepEqual(ticks, []);
    drag.start(pointer(input, at(0.3)));
    drag.move(pointer(input, at(0.6), { pointerId: 2 }));
    drag.end(pointer(input, at(0.6), { pointerId: 2 }));
    assert.deepEqual(ticks, [3]);
    assert.deepEqual(changes, [], "a second pointer's release does not end the drag");
    drag.end(pointer(input, at(0.3)));
    props.disabled = true;
    drag.start(pointer(input, at(0.9)));
    assert.deepEqual(ticks, [3]);
    assert.deepEqual(changes, [{ value: "3", bubbles: true }]);
  }
  {
    // Right to left, the track runs from max at the left edge.
    direction = "rtl";
    const { input, changes, drag } = rig({ min: -1, max: 1, step: 0.5 });
    drag.start(pointer(input, at(0.25)));
    drag.end(pointer(input, at(0.25)));
    assert.deepEqual(changes, [{ value: "0.5", bubbles: true }]);
    direction = "ltr";
  }
  console.log("Slider drags follow the pointer in steps and fire one change when they end on a new value");
} finally {
  globalThis.getComputedStyle = previousStyle;
  await server.close();
}
