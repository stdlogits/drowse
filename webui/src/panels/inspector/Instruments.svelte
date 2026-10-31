<script lang="ts">
  // The Inspector with nothing selected: the instruments (the Instruments canvas). Probes as rows,
  // the lens live, the SAE's features, sampling, and the sources the lens and SAE read through,
  // one line each (they are chosen in the Lens room).
  //
  // Sampling is the canvas's three range rows (temperature, top-p, max tokens; each moves the local
  // value as it slides and patches the session default when it settles), the thinking switch when
  // the model has a choice, and one quiet line to the rest: the full sampling settings, the system
  // prompt and a reset to the model's own defaults.
  import { MorphText, Switch, tokenParts } from "@stdlogits/ledger";
  import ProbeRack from "../ProbeRack.svelte";
  import JLensPanel from "../JLensPanel.svelte";
  import SaePanel from "../SaePanel.svelte";
  import { onMount } from "svelte";
  import { saeLoaded, sessionState } from "../../lib/stores/session.svelte";
  import { lensSourceState, refreshLensSources } from "../../lib/stores/instruments.svelte";
  import { sitesText, unloaded } from "../../lib/lensRoom";
  import {
    genStatus,
    openDrawer,
    patchSessionDefaults,
    samplingState,
    setSampling,
    tokenHoverState,
  } from "../../lib/stores.svelte";
  import { modelDefaultsState } from "../../lib/stores/sampling.svelte";
  import { resetSettings, settingsResetState } from "../../lib/stores/settingsReset.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import { goto } from "../../lib/stores/rooms.svelte";
  import { getHostedController, getRuntimeCapabilities } from "../../lib/runtime/registry";
  import { outputTokenLimitForSignals } from "../../lib/runtime/outputTokenPolicy";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { SAMPLING_HELP } from "../../lib/parameterHelp";

  const saeOn = $derived(saeLoaded());
  const hovered = $derived(tokenHoverState.active ? tokenParts(tokenHoverState.tokenText) : null);

  // The sources, one line each: the R family's from the session (the source every lens reading is
  // stamped with), the S family's from the source list, the SAE's from the session.
  const sources = $derived.by(() => {
    const blocks = sessionState.info?.instruments ?? [];
    const of = (family: string) => blocks.find(block => block.family === family)?.source ?? null;
    const lensRow = (family: "R" | "S") => lensSourceState.sources.find(row => row.family === family && row.active === true) ?? null;
    const r = of("lens");
    const rRow = lensSourceState.sources.find(row => row.source === r) ?? null;
    const s = lensRow("S");
    // A selected source that does not load for these weights says so (the R family's from the
    // session's jlens_fitted, which every runtime sends; the S family's from its row).
    const rLoads = !(r && sessionState.info?.jlens_fitted === false) && !unloaded(rRow);
    return [
      { key: "R-lens", value: r, meta: rLoads ? sitesText(rRow?.sites) : "not loaded for this model" },
      { key: "S-lens", value: s?.source ?? null, meta: unloaded(s) ? "not loaded for this model" : sitesText(s?.sites) },
      { key: "SAE", value: of("sae"), meta: "" },
    ];
  });
  onMount(() => {
    if (lensSourceState.sources.length === 0) void refreshLensSources();
  });

  // ---------- sampling ----------
  /** Controls wait for the session (and for a reset in flight); nothing here reaches the server before. */
  const ready = $derived(sessionState.info !== null && !settingsResetState.busy);
  const maxTokenLimit = $derived(
    sessionState.info
      ? outputTokenLimitForSignals(getRuntimeCapabilities()?.signals, getHostedController()?.snapshot.contextTokens ?? undefined)
      : 1,
  );
  const temperature = $derived(samplingState.temperature ?? 1);
  const topP = $derived(samplingState.top_p ?? 1);
  const maxTokens = $derived(Math.min(samplingState.max_tokens || 512, maxTokenLimit));

  type Key = "temperature" | "top_p" | "max_tokens";
  interface Range { key: Key; id: string; label: string; help: string; min: number; max: number; step: number; value: number; shown: string }
  const ranges = $derived<Range[]>([
    { key: "temperature", id: "sampling-temperature", label: "Temperature", help: SAMPLING_HELP.temperature,
      min: 0, max: Math.max(2, temperature), step: 0.05, value: temperature, shown: temperature.toFixed(2) },
    { key: "top_p", id: "sampling-top-p", label: "Top P", help: SAMPLING_HELP.topP,
      min: 0, max: 1, step: 0.01, value: topP, shown: topP.toFixed(2) },
    { key: "max_tokens", id: "sampling-max-tokens", label: "Max tokens", help: SAMPLING_HELP.maxTokens,
      min: 16, max: Math.max(16, Math.min(maxTokenLimit, Math.max(4096, maxTokens))), step: 16, value: maxTokens, shown: String(maxTokens) },
  ]);

  function valueOf(key: Key, input: HTMLInputElement): number | null {
    const v = input.valueAsNumber;
    if (!Number.isFinite(v)) return null;
    return key === "max_tokens" ? Math.max(1, Math.min(maxTokenLimit, Math.floor(v))) : v;
  }
  /** Sliding moves the value the next generation uses; settling patches the session default. */
  function slide(key: Key, event: Event): void {
    const v = valueOf(key, event.currentTarget as HTMLInputElement);
    if (v !== null) setSampling(key, v);
  }
  async function settle(key: Key, event: Event): Promise<void> {
    const v = valueOf(key, event.currentTarget as HTMLInputElement);
    if (v === null) return;
    setSampling(key, v);
    try {
      await patchSessionDefaults({ [key]: v });
    } catch (e) {
      console.warn("[sampling] patch failed", e);
    }
  }

  const thinkingSupported = $derived(sessionState.info?.supports_thinking ?? false);
  const thinkingOptional = $derived(sessionState.info?.thinking_is_optional === true);
  const thinkingForced = $derived(thinkingSupported && !thinkingOptional);
  function setThinking(on: boolean): void {
    setSampling("thinking", on);
    void patchSessionDefaults({ thinking: on }).catch((e) => console.warn("[sampling] patch failed", e));
  }

  let resetDone = $state(false);
  $effect(() => {
    if (!resetDone) return;
    const timer = setTimeout(() => (resetDone = false), 1800);
    return () => clearTimeout(timer);
  });
  async function reset(): Promise<void> {
    try {
      await resetSettings(false);
      resetDone = true;
      pushToast("Generation settings reset.", { kind: "info" });
    } catch (cause) {
      pushToast(userFacingError(cause, "Settings could not be fully reset. Try again."), { kind: "error" });
    }
  }
</script>

{#snippet at()}
  {#if hovered}<span class="meta">at <span class="dw-ws">{hovered.lead ? "·" : ""}</span>{hovered.text}</span>{/if}
{/snippet}

<div class="instruments">
  <section class="dw-sec" id="instrument-probes" aria-labelledby="instrument-probes-head">
    <div class="dw-sec-head"><span class="head" id="instrument-probes-head">Probes</span>{@render at()}</div>
    <ProbeRack />
  </section>

  <section class="dw-sec" id="instrument-lens" aria-labelledby="instrument-lens-head">
    <div class="dw-sec-head"><span class="head" id="instrument-lens-head">Live</span>{@render at()}</div>
    <JLensPanel />
  </section>

  <section class="dw-sec" id="instrument-sae" aria-labelledby="instrument-sae-head">
    <div class="dw-sec-head"><span class="head" id="instrument-sae-head">Features</span>{#if !saeOn}<span class="meta">Off</span>{:else}{@render at()}{/if}</div>
    {#if saeOn}
      <SaePanel sources={false} />
    {:else}
      <div class="dw-row"><button type="button" class="dw-quiet" onclick={() => goto("lens")}>Choose an SAE</button></div>
    {/if}
  </section>

  <section class="dw-sec" id="instrument-sampling" aria-labelledby="instrument-sampling-head">
    <div class="dw-sec-head"><span class="head" id="instrument-sampling-head">Sampling</span></div>
    <div class="ranges">
      {#each ranges as r (r.key)}
        <div class="range">
          <label for={r.id} data-tip={r.help}>{r.label}</label>
          <input type="range" id={r.id} min={r.min} max={r.max} step={r.step} value={r.value} disabled={!ready}
            aria-valuetext={r.shown} oninput={(e) => slide(r.key, e)} onchange={(e) => void settle(r.key, e)} />
          <span class="rv"><MorphText text={r.shown} /></span>
        </div>
      {/each}
    </div>
    {#if thinkingSupported}
      <div class="dw-row">
        <span class="switch-row" inert={!ready || thinkingForced}>
          <Switch checked={thinkingForced ? true : (samplingState.thinking ?? false)} onchange={setThinking}
            label={thinkingForced ? "Thinking, always on" : "Thinking"}
            hint={thinkingForced ? "This model always uses its reasoning phase." : SAMPLING_HELP.thinking} />
        </span>
      </div>
    {/if}
    <div class="more">
      <button type="button" class="dw-text" disabled={!ready} onclick={() => openDrawer("advanced_sampling")}>Sampling settings</button>
      {#if !sessionState.info?.is_base_model}
        <button type="button" class="dw-text" disabled={!ready} onclick={() => openDrawer("system_prompt")}>System prompt</button>
      {/if}
      <button type="button" class="dw-text reset" disabled={!modelDefaultsState.info || settingsResetState.busy || genStatus.active}
        onclick={() => void reset()}><MorphText text={settingsResetState.busy ? "Resetting…" : resetDone ? "Settings reset" : "Reset to original"} /></button>
    </div>
  </section>

  <section class="dw-sec" aria-labelledby="instrument-sources-head">
    <span class="head" id="instrument-sources-head">Sources</span>
    <div class="srcs">
      {#each sources as row (row.key)}
        <div class="src">
          <span class="sk">{row.key}</span>
          <span class="sv" class:none={!row.value}>{row.value ?? "none"}</span>
          <span class="meta">{row.meta}</span>
        </div>
      {/each}
    </div>
    <button type="button" class="dw-text room" onclick={() => goto("lens")}>Lens room</button>
  </section>
</div>

<style>
  .instruments { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s6); min-width: 0; }

  /* Sampling: one row per control (label, range, value), as the canvas draws it. */
  .ranges { display: grid; gap: var(--s1); }
  .range { display: grid; grid-template-columns: 96px minmax(0, 1fr) 40px; gap: var(--s4); align-items: center; min-height: 32px; }
  .range label { color: var(--ink2); font: var(--type-small); }
  .range input {
    -webkit-appearance: none;
    appearance: none;
    display: block;
    width: 100%;
    height: 20px;
    margin: 0;
    background: transparent;
    cursor: pointer;
  }
  .range input:disabled { opacity: 0.4; cursor: default; }
  .range input:focus { outline: none; }
  .range input:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; border-radius: var(--r-chip); }
  .range input::-webkit-slider-runnable-track { height: 4px; border-radius: 2px; background: var(--wash2); }
  .range input::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 14px; height: 14px; margin-top: -5px; border: 0; border-radius: 50%; background: var(--ink); }
  .range input::-moz-range-track { height: 4px; border: 0; border-radius: 2px; background: var(--wash2); }
  .range input::-moz-range-thumb { width: 14px; height: 14px; border: 0; border-radius: 50%; background: var(--ink); }
  .rv { color: var(--ink2); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
  .switch-row[inert] { opacity: 0.6; }
  .more { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s4); }
  .more .dw-text { font: var(--type-meta); }

  .srcs { display: grid; gap: var(--s2); }
  .src { display: grid; grid-template-columns: 96px auto minmax(0, 1fr); gap: var(--s4); align-items: baseline; }
  .sk { color: var(--ink2); font: var(--type-small); }
  .sv { color: var(--ink); font: 600 14px/20px var(--display); letter-spacing: -0.01em; overflow-wrap: anywhere; }
  .sv.none { color: var(--ink3); font: var(--type-small); }
  .room { justify-self: start; }

  /* Phone sizes: every target 44 px (a range's thumb, the text buttons, the switch). */
  @media (max-width: 720px), (pointer: coarse) {
    .range { grid-template-columns: 88px minmax(0, 1fr) 40px; gap: var(--s3); min-height: 44px; }
    .range input { height: 44px; }
    .more .dw-text, .room { min-height: 44px; }
    .switch-row :global(.switch) { height: 44px; }
  }
  @media (max-width: 720px) {
    .src { grid-template-columns: 64px auto minmax(0, 1fr); gap: var(--s3); }
  }
</style>
