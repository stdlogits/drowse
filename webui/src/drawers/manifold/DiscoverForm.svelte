<script lang="ts">
  import { Segmented, Switch } from "@stdlogits/ledger";
  import { onMount as onInterfaceMount } from "svelte";
  import { registerInterfaceController } from "../../lib/workspaceController";
  import { discoverDraftSchema } from "../../lib/interfaceSchemas";
  import { ToolError } from "../../lib/webmcp/types";

  import { MorphText } from "@stdlogits/ledger";
  // Auto-generated authoring: hand the model a flat concept list, the
  // K-tuple generator produces per-concept corpora against the shared
  // baseline prompts, then the fitter derives coords per-model via PCA or
  // spectral embedding.  No coords to author; no domain to pick.
  //
  // Generate and fit are deliberately two server calls — a flaky
  // generation leaves inspectable corpora — but "fit now" chains them so
  // the common case is one gesture.

  import { onMount, tick } from "svelte";

  import {
    apiManifoldFitStream,
    apiManifoldGenerateStream,
    describeError,
  } from "../../lib/runtime/services";
  import { openDrawer, refreshManifoldList, attachProbe } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import { manifoldJobs } from "../../lib/stores/manifoldJobs.svelte";
  import { EXAMPLES_PER_ROUND, EXAMPLE_BUDGETS } from "../../lib/exampleBudget";
  import { isFittingCancellation } from "../../lib/runtime/fittingCancellation";
  import type { FitManifoldRequest, GenerateManifoldRequest } from "../../lib/types";
  import NumberInput from "../../lib/NumberInput.svelte";
  import Select from "../../lib/Select.svelte";
  import AdvancedSection from "../../lib/builder/AdvancedSection.svelte";
  import ValidationBlock from "../../lib/builder/ValidationBlock.svelte";
  import DiscoverTuningFields from "./DiscoverTuningFields.svelte";
  import FitMethodPicker from "./FitMethodPicker.svelte";
  import {
    defaultTuning,
    identitySlugs,
    parseTokens,
    slug,
    tuningHyperparams,
    tuningMessages,
    type ManifoldIdentity,
  } from "./shared";
  import { getRuntimeManifoldFitMaxIntrinsicDim } from "../../lib/runtime/registry";
  import { runtimeClient } from "../../lib/runtime/client";
  import { PER_NODE_ROLE_HELP } from "../../lib/manifolds/selectors";
  import {
    refreshSaeSources,
    saeSourceState,
  } from "../../lib/stores/instruments.svelte";

  let { identity, oncomplete }: { identity: ManifoldIdentity; oncomplete?: (result: { namespace: string; name: string }) => void } = $props();

  type DiscoverKind = "abstract" | "concrete" | "custom";

  const maxDimLimit = getRuntimeManifoldFitMaxIntrinsicDim();
  const browserMode = runtimeClient.mode !== "http";
  const tuning = $state(defaultTuning(maxDimLimit, browserMode));
  let conceptsText = $state("");
  // Conversational corpus knobs: ``kind`` frames each concept's system
  // prompt (abstract → "someone {c}", concrete → "{article} {c}");
  // ``samplesPerPrompt`` is the in-character responses generated per
  // shared baseline prompt.
  let kind: DiscoverKind = $state("concrete");
  let customSystem = $state("You are {c}.");
  let samplesPerPrompt = $state(1);
  let budget = $state("1");
  // Persona-manifold opt-in: when set, each concept slug doubles as the
  // matching node's assistant-role substitution at fit time, producing a
  // role-paired manifold (steering through it implies the nearest node's
  // role at decode time).  The slug regex matches the engine's role
  // validation — concepts that pass ``slug()`` are already in
  // ``[a-z0-9._-]+``.
  let rolePerNode = $state(false);
  let force = $state(false);
  let saeRelease = $state("");
  let alsoFit = $state(true);
  let advancedOpen = $state(false);
  let submitting = $state(false);
  let result = $state<{ namespace: string; name: string; fitted: boolean } | null>(null);
  let lastFit: FitManifoldRequest = {};
  let error = $state<string | null>(null);
  let attaching = $state(false);
  let attached = $state(false);
  const anotherJob = $derived(manifoldJobs.current !== null && ["running", "waiting", "cancelling"].includes(manifoldJobs.current.status));
  let validationAttempted = $state(false);
  let formRegion: HTMLDivElement | null = $state(null);
  let conceptsInput: HTMLTextAreaElement | null = $state(null);
  let samplesField: HTMLLabelElement | null = $state(null);
  let customSystemInput: HTMLTextAreaElement | null = $state(null);

  const concepts = $derived(parseTokens(conceptsText));
  const examplesPerConcept = $derived(samplesPerPrompt * EXAMPLES_PER_ROUND);
  const totalExamples = $derived(examplesPerConcept * concepts.length);
  const browserSaeOptions = $derived([
    { value: "", label: "Standard fit (no feature pack)" },
    ...saeSourceState.sources.map((source) => ({
      value: source.source,
      label: source.name?.trim() || source.source,
    })),
  ]);

  onMount(() => {
    if (browserMode) void refreshSaeSources();
  });

  const validation = $derived.by<{ ok: boolean; messages: string[] }>(() => {
    const messages: string[] = [];
    if (!slug(identity.name)) {
      messages.push("name required");
    }
    if (concepts.length < 2) {
      messages.push(`concepts: ${concepts.length} / 2`);
    }
    const seen = new Set<string>();
    for (const c of concepts) {
      const s = slug(c);
      if (!s) {
        messages.push(`invalid concept "${c}"`);
      } else if (seen.has(s)) {
        messages.push(`duplicate concept "${s}"`);
      } else {
        seen.add(s);
      }
    }
    if (!Number.isSafeInteger(samplesPerPrompt) || samplesPerPrompt < 1) {
      messages.push("Choose a whole number of responses per prompt, at least 1.");
    }
    if (kind === "custom") {
      const template = customSystem.trim();
      if (!template) {
        messages.push("system template required");
      } else if (!template.includes("{c}")) {
        messages.push('system template needs "{c}"');
      }
    }
    messages.push(...tuningMessages(tuning, maxDimLimit));
    return { ok: messages.length === 0, messages };
  });

  const conceptsError = $derived.by<string | null>(() => {
    if (!validationAttempted) return null;
    if (concepts.length < 2) return "Enter at least two concepts.";
    const seen = new Set<string>();
    for (const concept of concepts) {
      const formatted = slug(concept);
      if (!formatted) return `Concept “${concept}” needs a letter or number.`;
      if (seen.has(formatted)) return "Each concept must be unique after formatting.";
      seen.add(formatted);
    }
    return null;
  });

  const samplesError = $derived(
    validationAttempted && (!Number.isSafeInteger(samplesPerPrompt) || samplesPerPrompt < 1)
      ? "Use a whole number, at least 1."
      : null,
  );

  const customSystemError = $derived.by<string | null>(() => {
    if (!validationAttempted || kind !== "custom") return null;
    if (!customSystem.trim()) return "Enter a system template.";
    if (!customSystem.includes("{c}")) return 'Include the "{c}" placeholder.';
    return null;
  });

  $effect(() => {
    const input = samplesField?.querySelector("input");
    if (!input) return;
    if (samplesError) {
      input.setAttribute("aria-invalid", "true");
      input.setAttribute("aria-describedby", "discover-samples-error");
    } else {
      input.removeAttribute("aria-invalid");
      input.removeAttribute("aria-describedby");
    }
  });

  function sharedNameInput(): HTMLInputElement | null {
    const body = formRegion?.closest(".mb-form");
    return body?.querySelector<HTMLInputElement>('input[data-identity="name"]') ?? null;
  }

  function tuningInput(label: string): HTMLInputElement | null {
    const fields = formRegion?.querySelectorAll<HTMLElement>(".field") ?? [];
    for (const field of fields) {
      const text = field.querySelector<HTMLElement>(".label")?.textContent?.trim();
      if (text === label) return field.querySelector<HTMLInputElement>("input");
    }
    return null;
  }

  async function focusFirstInvalid(): Promise<void> {
    await tick();
    if (!slug(identity.name)) {
      sharedNameInput()?.focus();
      return;
    }
    if (conceptsError) {
      conceptsInput?.focus();
      return;
    }
    if (samplesError) {
      samplesField?.querySelector<HTMLInputElement>("input")?.focus();
      return;
    }
    if (customSystemError) {
      customSystemInput?.focus();
      return;
    }
    const tuningErrors = tuningMessages(tuning, maxDimLimit);
    if (tuningErrors.length === 0) return;
    advancedOpen = true;
    await tick();
    const label = tuning.maxDim < 1 || maxDimLimit !== null && tuning.maxDim > maxDimLimit
      ? "Dimensions at most"
      : "Variance kept";
    tuningInput(label)?.focus();
  }

  async function save(): Promise<void> {
    if (submitting || anotherJob) return;
    validationAttempted = true;
    if (!validation.ok) {
      await focusFirstInvalid();
      return;
    }
    submitting = true;
    error = null;
    result = null;
    attached = false;
    const { namespace, name, description } = identitySlugs(identity);
    const hyperparams = tuningHyperparams(tuning);
    const req: GenerateManifoldRequest = {
      namespace,
      name,
      description,
      concepts: concepts.map((c) => slug(c)),
      kind,
      custom_system: kind === "custom" ? customSystem.trim() : undefined,
      samples_per_prompt: samplesPerPrompt,
      fit_mode: tuning.fitMode,
      hyperparams,
      force,
      role_per_node: rolePerNode,
    };
    lastFit = { sae: saeRelease.trim() || null, fit_mode: tuning.fitMode, hyperparams };
    const fitRequested = alsoFit;
    try {
      await apiManifoldGenerateStream(req, () => {}, fitRequested);
      result = { namespace, name, fitted: false };
      if (fitRequested) {
        await apiManifoldFitStream(namespace, name, lastFit, () => {});
        result = { namespace, name, fitted: true };
      }
      await refreshManifoldList();
    } catch (e) {
      error = isFittingCancellation(e)
        ? result ? "Fitting cancelled. Your examples are saved; you can fit them below." : "Generation cancelled. Completed concept groups are kept; run again to fill any missing groups."
        : `${result ? "Examples were saved, but fitting failed" : "Could not generate the examples"}: ${describeError(e)}`;
      await refreshManifoldList();
    } finally {
      submitting = false;
    }
  }

  async function fitSaved(): Promise<void> {
    if (!result || submitting || anotherJob) return;
    const saved = result;
    submitting = true;
    error = null;
    try {
      await apiManifoldFitStream(saved.namespace, saved.name, lastFit, () => {});
      result = { ...saved, fitted: true };
      await refreshManifoldList();
    } catch (e) {
      error = isFittingCancellation(e) ? "Fitting cancelled. Your saved examples are ready to try again." : `Could not fit the saved examples: ${describeError(e)}`;
    } finally { submitting = false; }
  }

  async function addProbe(): Promise<void> {
    if (!result?.fitted || attaching) return;
    attaching = true;
    try {
      await attachProbe(`${result.namespace}/${result.name}`);
      attached = true;
      pushToast("Probe added. Generate a reply to see its readings.");
    } catch (e) { error = `Could not add the probe: ${describeError(e)}`; }
    finally { attaching = false; }
  }

  onInterfaceMount(() => registerInterfaceController("manifold_discover", {
    schema: discoverDraftSchema,
    read: () => ({ busy: submitting || attaching || anotherJob, values: { concepts_text: conceptsText, kind, custom_system: customSystem, samples_per_prompt: samplesPerPrompt, role_per_node: rolePerNode, force, sae_release: saeRelease, also_fit: alsoFit, advanced_open: advancedOpen, tuning }, validation, examples_per_concept: examplesPerConcept, total_examples: totalExamples }),
    update: async (change) => {
      if (submitting || attaching || anotherJob) throw new ToolError("BUSY", "Wait for this interface operation to finish.");
      if (change.concepts_text !== undefined) conceptsText = change.concepts_text as string;
      if (change.kind !== undefined) kind = change.kind as DiscoverKind;
      if (change.custom_system !== undefined) customSystem = change.custom_system as string;
      if (change.samples_per_prompt !== undefined) { samplesPerPrompt = change.samples_per_prompt as number; budget = String(samplesPerPrompt); }
      if (change.role_per_node !== undefined) rolePerNode = change.role_per_node as boolean;
      if (change.force !== undefined) force = change.force as boolean;
      if (change.sae_release !== undefined) saeRelease = change.sae_release as string;
      if (change.also_fit !== undefined) alsoFit = change.also_fit as boolean;
      if (change.advanced_open !== undefined) advancedOpen = change.advanced_open as boolean;
      if (change.tuning) Object.assign(tuning, change.tuning);
    },
  }));
</script>

<div
  bind:this={formRegion}
  class="form-stack"
  role="form"
  aria-label="Generate manifold"
  aria-busy={submitting}
>
  <fieldset class="creation-fields" disabled={submitting || anotherJob}>
  <section class="step" aria-labelledby="discover-concepts-title">
    <h3 id="discover-concepts-title" class="step-title"><span class="n">2</span>Concepts</h3>
    <label class="field">
      <span class="sr-only">Concepts (at least 2)</span>
      <textarea
        bind:this={conceptsInput}
        class="input"
        rows="4"
        placeholder="pirate, assistant"
        bind:value={conceptsText}
        spellcheck="false"
        aria-invalid={!!conceptsError}
        aria-describedby={conceptsError ? "discover-concepts-error" : undefined}
      ></textarea>
      {#if conceptsError}
        <span id="discover-concepts-error" class="field-error">{conceptsError}</span>
      {/if}
      <span class="dim-note" data-tip="Separate with commas or new lines; join words with underscores">
        <strong><MorphText text={concepts.length} /></strong> concepts
      </span>
    </label>
    <div class="kind">
      <Segmented
        label="Concept kind"
        value={kind}
        onchange={(next) => (kind = next)}
        options={[
          { value: "concrete", label: "Roles", hint: "Roles or characters: “a pirate”" },
          { value: "abstract", label: "Traits", hint: "Traits or feelings: “someone patient”" },
          { value: "custom", label: "Custom", hint: "Your own system template, with {c} for the concept" },
        ]}
      />
    </div>
    {#if kind === "custom"}
      <label class="field">
        <span class="label">System template</span>
        <textarea
          bind:this={customSystemInput}
          class="input"
          rows="3"
          bind:value={customSystem}
          spellcheck="false"
          aria-invalid={!!customSystemError}
          aria-describedby={customSystemError
            ? "discover-system-error"
            : undefined}
        ></textarea>
        {#if customSystemError}
          <span id="discover-system-error" class="field-error">
            {customSystemError}
          </span>
        {/if}
      </label>
    {/if}
  </section>

  <section class="step" aria-labelledby="example-budget-title">
    <h3 id="example-budget-title" class="step-title"><span class="n">3</span>Examples</h3>
    <div class="choice-rows" role="radiogroup" aria-label="Example budget">
      {#each EXAMPLE_BUDGETS as option}
        <label class="choice" data-tip={option.description}>
          <input class="sr-only" type="radio" name="example-budget" value={String(option.rounds)} bind:group={budget} onchange={() => (samplesPerPrompt = option.rounds)} />
          <b>{option.label}</b><span class="meta">{option.rounds * EXAMPLES_PER_ROUND} per concept</span>
        </label>
      {/each}
      <label class="choice">
        <input class="sr-only" type="radio" name="example-budget" value="custom" bind:group={budget} />
        <b>Custom</b><span class="meta">responses per prompt</span>
      </label>
    </div>
    {#if budget === "custom"}
      <label bind:this={samplesField} class="field samples">
        <span class="label">Responses per prompt</span>
        <NumberInput value={samplesPerPrompt} min={1} step={1} oninput={(v) => { if (v !== null) samplesPerPrompt = v; }} />
        {#if samplesError}<span id="discover-samples-error" class="field-error">{samplesError}</span>{/if}
      </label>
    {/if}
    <p class="dim-note" data-tip={`Each round covers all ${EXAMPLES_PER_ROUND} prompts. More examples can reduce sampling noise, but do not guarantee a better probe.`}>{#if Number.isSafeInteger(totalExamples) && totalExamples > 0}<strong>{`${totalExamples.toLocaleString()} examples total`}</strong> · {concepts.length} × {examplesPerConcept}{:else}Choose concepts and a budget{/if}</p>
  </section>

  <AdvancedSection bind:expanded={advancedOpen}>
    <FitMethodPicker {tuning} linearOnly={browserMode} spectralNote="curved · best with ≥50 nodes" onchange={(fitMode) => (tuning.fitMode = fitMode)} />
    {#if browserMode}
      <div class="field">
        <span class="label">Feature space</span>
        <Select
          bind:value={saeRelease}
          options={browserSaeOptions}
          disabled={saeSourceState.loading || saeSourceState.busy}
          ariaLabel="Feature space used for manifold fitting"
        />
        {#if saeSourceState.error}
          <span class="field-error">{saeSourceState.error}</span>
        {:else if !saeSourceState.loading && saeSourceState.sources.length === 0}
          <span class="dim-note">No compatible feature pack installed</span>
        {/if}
      </div>
    {:else}
      <label class="field">
        <span class="label">SAE release</span>
        <input
          class="input"
          bind:value={saeRelease}
          placeholder="gemma-scope-2-4b-it-res"
          spellcheck="false"
        />
      </label>
    {/if}
    <DiscoverTuningFields {tuning} {maxDimLimit} />
    <div class="check-stack">
      <Switch checked={alsoFit} onchange={(on) => (alsoFit = on)} label="Fit after generating examples" />
      <Switch
        checked={rolePerNode}
        onchange={(on) => (rolePerNode = on)}
        label="Use a separate role for each concept"
        hint={PER_NODE_ROLE_HELP}
      />
      {#if rolePerNode}
        <p class="role-hint">Each concept becomes its assistant voice; a chat template without roles fails before fitting.</p>
      {/if}
      <Switch checked={force} onchange={(on) => (force = on)} label="Replace existing examples with this name"
        hint="Without replacement, existing concept groups are reused. To change their budget, use a new name or replace them." />
    </div>
  </AdvancedSection>
  </fieldset>
  {#if error}<p class="mf-error" role="alert">{error}</p>{/if}
  {#if result && !submitting}
    <section class="mf-wash" aria-label="Creation result">
      <div class="mf-sec-head"><b class="result-name">{result.namespace}/{result.name}</b><span class="meta">{result.fitted ? "Ready to use" : "Examples saved"}</span></div>
      <div class="mf-actions">
        {#if result.fitted}<button type="button" class="dw-button" disabled={attached || attaching} onclick={() => void addProbe()}>{attached ? "Probe added" : "Add probe"}</button>
        {:else}<button type="button" class="dw-button" disabled={anotherJob} onclick={() => void fitSaved()}>Fit saved examples</button>{/if}
        <button type="button" class="dw-quiet" onclick={() => oncomplete ? oncomplete(result!) : openDrawer("subspace")}>Open library</button>
      </div>
    </section>
  {/if}

  <ValidationBlock
    verb="generate examples"
    messages={validationAttempted ? validation.messages : []}
  />

  <div class="submit">
    <button
      type="button"
      class="dw-button"
      disabled={submitting || anotherJob}
      onclick={save}
    >
      <MorphText text={alsoFit ? "Generate and fit" : "Generate examples"} />
    </button>
  </div>
</div>

<style>
  .kind { min-width: 0; max-width: 100%; overflow-x: auto; scrollbar-width: none; }
  .kind::-webkit-scrollbar { display: none; }
  .samples { max-width: 200px; }
  .result-name { min-width: 0; color: var(--ink); font: var(--type-control); overflow-wrap: anywhere; }
</style>
