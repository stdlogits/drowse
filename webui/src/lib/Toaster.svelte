<script lang="ts">
  // Toast host — renders ``toasts.entries`` in the bottom-right corner.
  // Each toast auto-dismisses after its ``ttlMs`` fires (paused under the
  // pointer or focus); the ✕ dismisses early.  Toasts with ``ttlMs === null``
  // are sticky — long-running async work (extract / clone) drives a single
  // toast from kickoff to completion via ``updateToast``.

  import { onDestroy } from "svelte";
  import { Icon, fadeOut, rise } from "@stdlogits/ledger";
  import { dismissToast, toasts } from "./stores.svelte";

  // Track which toast ids have an active timer so we don't re-schedule
  // dismissal every time the entries array reshuffles.  Sticky entries
  // (``ttlMs === null``) never enter the set and never get scheduled.
  interface ToastTimer {
    handle: ReturnType<typeof setTimeout> | null;
    remaining: number;
    startedAt: number;
    paused: boolean;
  }

  const timers = new Map<number, ToastTimer>();

  function finish(id: number): void {
    const timer = timers.get(id);
    if (timer && timer.handle !== null) clearTimeout(timer.handle);
    timers.delete(id);
    dismissToast(id);
  }

  function arm(id: number, timer: ToastTimer): void {
    if (timer.remaining <= 0) {
      finish(id);
      return;
    }
    timer.startedAt = performance.now();
    timer.paused = false;
    timer.handle = setTimeout(() => finish(id), timer.remaining);
  }

  function pause(id: number): void {
    const timer = timers.get(id);
    if (!timer || timer.paused || timer.handle === null) return;
    clearTimeout(timer.handle);
    timer.handle = null;
    timer.remaining = Math.max(0, timer.remaining - (performance.now() - timer.startedAt));
    timer.paused = true;
  }

  function resume(id: number): void {
    const timer = timers.get(id);
    if (!timer || !timer.paused) return;
    arm(id, timer);
  }

  $effect(() => {
    for (const t of toasts.entries) {
      if (t.ttlMs === null || timers.has(t.id)) continue;
      const timer: ToastTimer = {
        handle: null,
        remaining: t.ttlMs,
        startedAt: 0,
        paused: false,
      };
      timers.set(t.id, timer);
      arm(t.id, timer);
    }
    const visibleIds = new Set(toasts.entries.map((toast) => toast.id));
    for (const [id, timer] of timers) {
      if (visibleIds.has(id)) continue;
      if (timer.handle !== null) clearTimeout(timer.handle);
      timers.delete(id);
    }
  });

  onDestroy(() => {
    for (const timer of timers.values()) {
      if (timer.handle !== null) clearTimeout(timer.handle);
    }
    timers.clear();
  });
</script>

{#if toasts.entries.length > 0}
  <div class="toaster" role="region" aria-label="Notifications">
    {#each toasts.entries as t (t.id)}
      <div
        class="toast"
        class:warning={t.kind === "warning"}
        class:error={t.kind === "error"}
        role={t.kind === "error" ? "alert" : "status"}
        aria-live={t.kind === "error" ? "assertive" : "polite"}
        aria-atomic="true"
        in:rise
        out:fadeOut
        onpointerenter={() => pause(t.id)}
        onpointerleave={() => resume(t.id)}
        onfocusin={() => pause(t.id)}
        onfocusout={(event) => {
          if (!(event.currentTarget as HTMLElement).contains(event.relatedTarget as Node | null)) {
            resume(t.id);
          }
        }}
      >
        <div class="body">
          <span class="msg">{t.message}</span>
          {#if t.detail}
            <span class="detail">{t.detail}</span>
          {/if}
        </div>
        <button
          type="button"
          class="dismiss"
          aria-label={`Dismiss notification: ${t.message}`}
          onclick={() => finish(t.id)}
        ><Icon name="close" /></button>
      </div>
    {/each}
  </div>
{/if}

<style>
  /* Toasts: flat wash2 surfaces stacked at the bottom right, no line and no shadow. An error is told
   * by its role and its words, not a colour (Ledger has none for it). */
  .toaster {
    position: fixed;
    inset-inline-end: calc(var(--s5) + env(safe-area-inset-right, 0px));
    bottom: calc(var(--s5) + env(safe-area-inset-bottom, 0px));
    z-index: 70;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: var(--s2);
    width: min(400px, calc(100vw - var(--s5) * 2));
    pointer-events: none;
  }
  .toast {
    display: flex;
    align-items: flex-start;
    gap: var(--s3);
    max-width: 100%;
    padding: var(--s3) var(--s2) var(--s3) var(--s4);
    border: 0;
    border-radius: var(--r-box);
    background: var(--wash2);
    color: var(--ink);
    font: var(--type-small);
    pointer-events: auto;
  }
  .body {
    display: grid;
    flex: 1 1 auto;
    gap: 2px;
    min-width: 0;
    padding-block: var(--s1);
  }
  .msg { color: var(--ink); font: var(--type-control); overflow-wrap: anywhere; }
  .toast.warning .msg, .toast.error .msg { font-weight: 600; }
  .detail { color: var(--ink2); font: var(--type-meta); overflow-wrap: anywhere; }
  .dismiss {
    display: grid;
    flex: none;
    place-items: center;
    width: 28px;
    height: 28px;
    padding: 0;
    border: 0;
    border-radius: var(--r-pill);
    background: transparent;
    color: var(--ink3);
    cursor: pointer;
    transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease), scale var(--fast) var(--ease);
  }
  .dismiss:hover { background: var(--paper); color: var(--ink); }
  .dismiss:active { scale: 0.97; }
  @media (forced-colors: active) {
    .toast { border: 1px solid CanvasText; }
  }
  @media (max-width: 720px) {
    .toaster { inset-inline: var(--s4); bottom: calc(var(--s4) + env(safe-area-inset-bottom, 0px)); width: auto; align-items: stretch; }
  }
  @media (max-width: 720px), (pointer: coarse) {
    .dismiss { width: 44px; height: 44px; margin-block: calc(var(--s2) * -1); }
  }
</style>
