<script lang="ts">
  // Sampling settings (a dialog, from the sampling section's "More settings"): the filters and
  // penalties, capture and seed as label-over-field pairs, and the rarer knobs folded under More.
  import { Expander, IconButton, Segmented } from "@stdlogits/ledger";
  import {
    closeDrawer,
    samplingState,
    setSampling,
    patchSessionDefaults,
    genUiMode,
    setGenUiMode,
    sessionState,
  } from "../lib/stores.svelte";
  import NumberInput from "../lib/NumberInput.svelte";
  import { getRuntimeClient } from "../lib/runtime/registry";
  import { SAMPLING_HELP } from "../lib/parameterHelp";
  import {
    BROWSER_SAMPLING_TOP_K_MAX,
    SAMPLING_TEMPERATURE_MAX,
    SAMPLING_SEED_MAX,
    samplingSeedMinimum,
    clampTokenAlternativeCount,
    tokenAlternativeLimit,
  } from "../lib/runtime/samplingCapabilities";

  // Render mode — two-state: chat renders turns and roles, raw renders a
  // single flat completion buffer.  The mode is seeded from the model's
  // ``is_base_model`` flag the first time the model is seen, then it's a
  // plain user toggle (a base model stays raw).  Toggling never mutates the
  // loom tree.
  const RENDER_MODES: { value: "chat" | "raw"; label: string }[] = [
    { value: "chat", label: "Chat template" },
    { value: "raw", label: "Raw completion" },
  ];
  const isBaseModel = $derived(sessionState.info?.is_base_model === true);
  const runtimeMode = getRuntimeClient().mode;
  const alternativesMax = tokenAlternativeLimit(runtimeMode);
  const alternativesAvailable = alternativesMax > 0;
  const TOP_K_MIN = 0;
  const TOP_K_MAX = BROWSER_SAMPLING_TOP_K_MAX;
  const PENALTY_MIN = -2;
  const PENALTY_MAX = 2;
  let technicalOpen = $state(false);

  const HELP = { ...SAMPLING_HELP, temperature: SAMPLING_HELP.advancedTemperature };

  function onTopK(raw: number | null): void {
    if (raw === null) {
      setSampling("top_k", null);
      void patchSessionDefaults({ top_k: null });
      return;
    }
    const value = Math.max(TOP_K_MIN, Math.min(TOP_K_MAX, Math.floor(raw)));
    setSampling("top_k", value);
    void patchSessionDefaults({ top_k: value });
  }

  function onPenalty(
    key: "presence_penalty" | "frequency_penalty",
    raw: number | null,
  ): void {
    if (raw === null) return;
    setSampling(key, Math.max(PENALTY_MIN, Math.min(PENALTY_MAX, raw)));
  }

  function onAlternatives(raw: number | null): void {
    if (raw === null) return;
    setSampling("return_top_k", clampTokenAlternativeCount(raw, runtimeMode));
  }

  function onSeed(raw: number | null): void {
    setSampling("seed", raw === null ? null : Math.floor(raw));
  }

  const logitBiasValid = $derived.by(() => {
    const raw = samplingState.logit_bias_text.trim();
    if (!raw) return true;
    try {
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed);
    } catch {
      return raw
        .split(/\r?\n/)
        .filter(Boolean)
        .every((line) =>
          /^\s*-?\d+\s*[:=,\s]\s*-?\d+(?:\.\d+)?\s*$/.test(line),
        );
    }
  });
</script>

<section class="sampling-settings" aria-label="Sampling settings drawer">
  <header class="dlg-head">
    <h2>Sampling settings</h2>
    <IconButton icon="close" label="Close" onclick={closeDrawer} />
  </header>

  <div class="body">
    <section class="dw-sec" aria-labelledby="sampling-filters-head">
      <span class="head" id="sampling-filters-head">Filters and penalties</span>
      <div class="fields">
        <div class="field">
          <span class="label" data-tip={HELP.temperature}>Temperature</span>
          <NumberInput
            value={samplingState.temperature}
            min={0}
            max={SAMPLING_TEMPERATURE_MAX}
            step={0.05}
            onchange={(value) => {
              if (value === null) return;
              setSampling("temperature", value);
              void patchSessionDefaults({ temperature: value });
            }}
            ariaLabel="Temperature value"
          />
        </div>
        <div class="field">
          <span class="label" data-tip={HELP.topK}>Top-k</span>
          <NumberInput
            value={samplingState.top_k}
            min={TOP_K_MIN}
            max={TOP_K_MAX}
            step={1}
            placeholder="1024 by default"
            allowEmpty
            onchange={onTopK}
            ariaLabel="Top-k"
          />
        </div>
        <div class="field">
          <span class="label" data-tip={HELP.frequencyPenalty}>Frequency penalty</span>
          <NumberInput
            value={samplingState.frequency_penalty}
            min={PENALTY_MIN}
            max={PENALTY_MAX}
            step={0.05}
            onchange={(value) => onPenalty("frequency_penalty", value)}
            ariaLabel="Frequency penalty"
          />
        </div>
        <div class="field">
          <span class="label" data-tip={HELP.presencePenalty}>Presence penalty</span>
          <NumberInput
            value={samplingState.presence_penalty}
            min={PENALTY_MIN}
            max={PENALTY_MAX}
            step={0.05}
            onchange={(value) => onPenalty("presence_penalty", value)}
            ariaLabel="Presence penalty"
          />
        </div>
      </div>
    </section>

    <section class="dw-sec" aria-labelledby="sampling-capture-head">
      <span class="head" id="sampling-capture-head">Capture and seed</span>
      <div class="fields">
        <div class="field">
          <span class="label" data-tip={HELP.returnTopK}>Alternatives per token</span>
          <NumberInput
            value={samplingState.return_top_k}
            min={0}
            max={alternativesMax}
            step={1}
            disabled={!alternativesAvailable}
            onchange={onAlternatives}
            ariaLabel="Alternatives per token"
          />
        </div>
        <div class="field">
          <span class="label" data-tip={HELP.seed}>Seed</span>
          <NumberInput
            value={samplingState.seed}
            min={samplingSeedMinimum(runtimeMode)}
            max={SAMPLING_SEED_MAX}
            step={1}
            placeholder="Not fixed"
            allowEmpty
            onchange={onSeed}
            ariaLabel="Seed"
          />
        </div>
      </div>
    </section>

    <!-- Outside the fold: Segmented's gliding thumb is placed only while no ancestor is inert,
         and a closed Expander's body is. -->
    <section class="dw-sec" aria-labelledby="sampling-format-head">
      <span class="head" id="sampling-format-head">Format</span>
      {#if isBaseModel}
        <span class="fixed" data-tip="A base model continues the text as written, without chat roles or a system prompt">Raw completion</span>
      {:else}
        <div class="dw-row">
          <Segmented label="Conversation format" value={genUiMode.mode} onchange={setGenUiMode} options={RENDER_MODES} />
        </div>
      {/if}
    </section>

    <Expander bind:open={technicalOpen} label="More">
      <div class="more">

        <label class="dw-sec">
          <span class="head">Stop sequences</span>
          <textarea
            rows="4"
            value={samplingState.stop_sequences}
            oninput={(ev) => setSampling("stop_sequences", (ev.currentTarget as HTMLTextAreaElement).value)}
            placeholder={"One per line\n###\n<|eot_id|>"}
            aria-label="Stop sequences"
          ></textarea>
        </label>

        <label class="dw-sec">
          <span class="head">Logit bias</span>
          <textarea
            rows="5"
            class:invalid={!logitBiasValid}
            value={samplingState.logit_bias_text}
            oninput={(ev) => setSampling("logit_bias_text", (ev.currentTarget as HTMLTextAreaElement).value)}
            placeholder={'{"198": -4, "220": 1.5}'}
            aria-label="Logit bias"
            aria-invalid={!logitBiasValid}
            aria-describedby="logit-bias-help"
          ></textarea>
          <span id="logit-bias-help" class="meta" class:error={!logitBiasValid}>
            {logitBiasValid ? "A JSON object, or one token ID and value per line" : "Use JSON, or one token ID and value per line"}
          </span>
        </label>
      </div>
    </Expander>
  </div>
</section>

<style>
  .sampling-settings {
    container: sampling-settings / inline-size;
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    color: var(--ink2);
  }
  .dlg-head { display: flex; flex: 0 0 auto; align-items: center; justify-content: space-between; gap: var(--s4); padding: var(--s5) var(--s5) 0; }
  .dlg-head h2 { margin: 0; color: var(--ink); font: var(--type-mark); }
  .body {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: var(--s6);
    align-content: start;
    min-height: 0;
    padding: var(--s5);
    overflow: auto;
    overscroll-behavior: contain;
  }

  /* Two columns of label-over-field, one when the dialog is narrow. */
  .fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s4) var(--s5); }
  .field { display: grid; gap: var(--s2); min-width: 0; }
  .label { justify-self: start; color: var(--ink2); font: var(--type-control); }
  .more { display: grid; gap: var(--s5); }
  .fixed { justify-self: start; color: var(--ink); font: var(--type-control); }

  textarea {
    width: 100%;
    min-height: 96px;
    padding: var(--s2) var(--s3);
    border: 0;
    border-radius: var(--r-col);
    background: var(--field);
    color: var(--ink);
    font: var(--type-small);
    resize: vertical;
    transition: background-color var(--fast) var(--ease);
  }
  textarea:hover { background: var(--field-hover); }
  /* Ledger has no red: an invalid field darkens toward ink, and the line under it says why. */
  textarea.invalid { background: color-mix(in oklab, var(--ink) 14%, var(--field)); }
  .meta.error { color: var(--ink); }

  @container sampling-settings (max-width: 560px) {
    .fields { grid-template-columns: minmax(0, 1fr); }
  }
  @media (max-width: 720px) {
    .dlg-head { padding: var(--s4) var(--s4) 0; }
    .body { padding: var(--s4); }
  }
</style>
