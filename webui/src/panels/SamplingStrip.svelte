<script lang="ts">
  import { Switch } from "@stdlogits/ledger";
  import RollingNumber from "../lib/ui/RollingNumber.svelte";
  import ResetSettings from "../lib/ui/ResetSettings.svelte";
  import { settingsResetState } from "../lib/stores/settingsReset.svelte";
  // The Inspector's Sampling section (the Instruments canvas): the everyday controls as rows.
  // Secondary generation and inspection controls live in AdvancedSamplingDrawer.
  //
  // Every edit applies immediately.  temperature / top-p / top-k /
  // max-tokens / thinking PATCH the session defaults as the user moves
  // them; seed and the advanced extras (penalties, stop strings, logit
  // bias, return_top_k) have no PATCH path, so ``sendGenerate`` packs
  // them onto each call's ``SamplingConfig``.  Either way the value the
  // strip shows is the value the next generation uses.
  //

  import {
    samplingState,
    sessionState,
    setSampling,
    patchSessionDefaults,
    openDrawer,
  } from "../lib/stores.svelte";
  import Slider from "../lib/Slider.svelte";
  import NumberInput from "../lib/NumberInput.svelte";
  import { getRuntimeCapabilities, getHostedController } from "../lib/runtime/registry";
  import { outputTokenLimitForSignals } from "../lib/runtime/outputTokenPolicy";
  import { SAMPLING_HELP } from "../lib/parameterHelp";

  // ------------------------------------------------------------------- consts

  // Placeholder defaults shown while the session info hasn't landed yet.
  // These never reach the server — the strip is disabled in this state.
  const PLACEHOLDER = {
    temperature: 1.0,
    top_p: 1.0,
    max_tokens: 512,
  };

  const TEMP_MIN = 0;
  const TEMP_MAX = 2;
  const TEMP_STEP = 0.05;
  const TOP_P_MIN = 0;
  const TOP_P_MAX = 1;
  const TOP_P_STEP = 0.01;
  const MAX_TOK_MIN = 1;

  const HELP = SAMPLING_HELP;

  function describedBy(node: HTMLElement, id: string) {
    let descriptionId = id;
    let input: HTMLInputElement | null = null;
    const connect = () => {
      input = node.querySelector("input");
      input?.setAttribute("aria-describedby", descriptionId);
    };
    queueMicrotask(connect);
    return {
      update(next: string) {
        if (input?.getAttribute("aria-describedby") === descriptionId) {
          input.removeAttribute("aria-describedby");
        }
        descriptionId = next;
        connect();
      },
      destroy() {
        if (input?.getAttribute("aria-describedby") === descriptionId) {
          input.removeAttribute("aria-describedby");
        }
      },
    };
  }

  // ------------------------------------------------------------------- ready

  /** True once session info has loaded — gates control enable state. */
  const ready = $derived(sessionState.info !== null && !settingsResetState.busy);

  /** True iff thinking is supported for this model.  ``supports_thinking``
   * comes off the session info and may flip once the model loads. */
  const thinkingSupported = $derived(
    sessionState.info?.supports_thinking ?? false,
  );

  /** True iff the user can actually turn thinking off (the chat template
   *  has an ``enable_thinking`` switch).  Forced-thinking models leave
   *  the toggle locked and read-only so the user knows clicking it
   *  is a no-op.  The field is required by the current session contract
   *  so we don't lock controls against backends that pre-date the
   *  field. */
  const thinkingOptional = $derived(
    sessionState.info?.thinking_is_optional === true,
  );

  // (The per-message role boxes moved to the composer's cast row —
  // Chat.svelte ``speaking as`` / ``reply as`` chips, same client state.)

  /** Tri-state for the title attribute and disabled gate. */
  const thinkingForced = $derived(
    thinkingSupported && !thinkingOptional,
  );

  // ------------------------------------------------------------------- views
  //
  // Each control's *display* value reads ``samplingState`` first (which the
  // store's bootstrap populates from session config) and falls back to a
  // placeholder when it's still null.

  const tempView = $derived(samplingState.temperature ?? PLACEHOLDER.temperature);
  const topPView = $derived(samplingState.top_p ?? PLACEHOLDER.top_p);
  const maxTokenLimit = $derived(
    sessionState.info ? outputTokenLimitForSignals(
      getRuntimeCapabilities()?.signals,
      getHostedController()?.snapshot.contextTokens ?? undefined,
    ) : MAX_TOK_MIN,
  );
  const maxView = $derived(
    Math.min(samplingState.max_tokens || PLACEHOLDER.max_tokens, maxTokenLimit),
  );
  const thinkingView = $derived(samplingState.thinking ?? false);

  // ------------------------------------------------------------------- writes

  /** PATCH the server with a single field.  Errors surface as a
   * console.warn — the strip itself stays usable (local state already
   * updated; user can retry). */
  async function persistDefault(
    body: Partial<{
      temperature: number;
      top_p: number;
      top_k: number | null;
      max_tokens: number;
      thinking: boolean;
    }>,
  ): Promise<void> {
    try {
      await patchSessionDefaults(body);
    } catch (e) {
      console.warn("[sampling] patch failed", e);
    }
  }

  function onTemp(v: number): void {
    setSampling("temperature", v);
    void persistDefault({ temperature: v });
  }

  function onTopP(v: number): void {
    setSampling("top_p", v);
    void persistDefault({ top_p: v });
  }

  function onMax(raw: number | null): void {
    if (raw === null) return;
    const v = Math.max(MAX_TOK_MIN, Math.min(maxTokenLimit, Math.floor(raw)));
    setSampling("max_tokens", v);
    void persistDefault({ max_tokens: v });
  }

  function onThinking(v: boolean): void {
    setSampling("thinking", v);
    void persistDefault({ thinking: v });
  }

  function openSystemPrompt(): void {
    openDrawer("system_prompt");
  }

  function openAdvanced(): void {
    openDrawer("advanced_sampling");
  }
</script>

<section class="sampling-strip" aria-label="sampling controls">
  <div class="ranges">
    <div class="range" use:describedBy={"sampling-temperature-help"}>
      <span class="label" data-tip={HELP.temperature}>Temperature</span>
      <Slider
        value={tempView}
        min={TEMP_MIN}
        max={Math.max(TEMP_MAX, tempView)}
        step={TEMP_STEP}
        disabled={!ready}
        oninput={onTemp}
        ariaLabel="Temperature"
      />
      <span class="value"><RollingNumber value={tempView} digits={2} /></span>
      <span id="sampling-temperature-help" class="sr-only">{HELP.temperature}</span>
    </div>

    <div class="range" use:describedBy={"sampling-top-p-help"}>
      <span class="label" data-tip={HELP.topP}>Top-p</span>
      <Slider
        value={topPView}
        min={TOP_P_MIN}
        max={TOP_P_MAX}
        step={TOP_P_STEP}
        disabled={!ready}
        oninput={onTopP}
        ariaLabel="Top P"
      />
      <span class="value"><RollingNumber value={topPView} digits={2} /></span>
      <span id="sampling-top-p-help" class="sr-only">{HELP.topP}</span>
    </div>

    <div class="range" use:describedBy={"sampling-max-tokens-help"}>
      <span class="label" data-tip={HELP.maxTokens}>Max tokens</span>
      <span class="field">
        <NumberInput
          value={maxView}
          min={MAX_TOK_MIN}
          max={maxTokenLimit}
          step={1}
          disabled={!ready}
          onchange={onMax}
          ariaLabel="Max tokens"
        />
      </span>
      <span id="sampling-max-tokens-help" class="sr-only">{HELP.maxTokens}</span>
    </div>
  </div>

  {#if thinkingSupported}
    <div class="thinking" class:forced={thinkingForced} use:describedBy={"sampling-thinking-help"}>
      <span inert={!ready || thinkingForced}>
        <Switch
          checked={thinkingForced ? true : thinkingView}
          onchange={onThinking}
          label={`Thinking${thinkingForced ? ", always on" : ""}`}
          hint={!thinkingOptional
            ? "This model always uses its reasoning phase, so this setting cannot be changed."
            : HELP.thinking}
        />
      </span>
      <span id="sampling-thinking-help" class="sr-only">
        {!thinkingOptional
          ? "This model always uses its reasoning phase, so this setting cannot be changed."
          : HELP.thinking}
      </span>
    </div>
  {/if}

  <div class="actions">
    <button type="button" class="dw-text" disabled={!ready} onclick={openAdvanced}>More settings</button>
    {#if !sessionState.info?.is_base_model}
      <button type="button" class="dw-text" disabled={!ready} onclick={openSystemPrompt}>System prompt</button>
    {/if}
    <ResetSettings />
  </div>
</section>

<style>
  /* The Instruments canvas's sampling: one row per setting (a label, the control, its value), a
   * switch for thinking, then the quiet text actions. */
  .sampling-strip { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s3); min-width: 0; color: var(--ink2); }
  .ranges { display: grid; gap: var(--s1); }
  .range { display: grid; grid-template-columns: 96px minmax(0, 1fr) 40px; gap: var(--s4); align-items: center; min-height: 32px; }
  .label { justify-self: start; color: var(--ink2); font: var(--type-small); }
  .value { color: var(--ink2); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; }
  .field { display: flex; grid-column: 2 / -1; min-width: 0; }
  .thinking { display: flex; align-items: center; }
  .thinking.forced { opacity: 0.6; }
  .actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s4); }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }

  @media (max-width: 720px) {
    .range { grid-template-columns: 88px minmax(0, 1fr) 40px; gap: var(--s3); min-height: 44px; }
  }
  @media (max-width: 720px), (pointer: coarse) {
    .actions .dw-text { min-height: 44px; }
  }
</style>
