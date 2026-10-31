<script lang="ts">
  import { onMount as onInterfaceMount } from "svelte";
  import { registerInterfaceController } from "../lib/workspaceController";
  import { templateDraftSchema } from "../lib/interfaceSchemas";
  import { ToolError } from "../lib/webmcp/types";

  import TemplatePreview from "../lib/ui/TemplatePreview.svelte";
  import { Icon, IconButton, MorphText } from "@stdlogits/ledger";
  import Select from "../lib/Select.svelte";
  import "./manifold/form.css";
  // TemplateLabDrawer — the standalone templated-completion artifact.
  //
  // Two tabs:
  //   * score — pick a template, optionally steer, run; render the per-context
  //     restricted-choice value distribution (the logit read). With a steering
  //     expression this is the distributional before/after.
  //   * build — author a template: a slot token, candidate values, and one or
  //     more multi-turn contexts (history turns + the slotted final assistant
  //     turn). The slot lives only in the assistant turn.
  //
  // Reached from the command palette, or deep-linked to the build tab
  // (``params: { tab: "build" }``) from the manifold builder's template
  // path — this is the ONE template editor, and the builder derives a
  // manifold from what it authors (`drowse manifold from-template`).

  import { onMount, tick, untrack } from "svelte";
  import { isFittingCancellation } from "../lib/runtime/fittingCancellation";
  import { getHostedController } from "../lib/runtime/registry";
  import { apiTemplates, describeError } from "../lib/runtime/services";
  import { validateTemplateDraft } from "../lib/templates";
  import { onArtifactUpdate } from "../lib/artifactUpdates";
  import { closeDrawer } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import { Segmented } from "@stdlogits/ledger";
  import type {
    ChoiceScores,
    TemplateContextSpec,
    TemplateSummary,
    TemplateDetail,
    TemplateTurn,
  } from "../lib/types";

  let { params }: { params?: unknown } = $props();
  const embedded = $derived((params as { embedded?: unknown } | null)?.embedded === true);

  type Tab = "score" | "build";
  // Opening tab: ``score`` unless a caller deep-linked the editor.  Read
  // once at mount — the drawer host remounts this component on every
  // open, so a later params change isn't a case that exists.
  let tab: Tab = $state(
    untrack(
      () =>
        (params as { tab?: unknown } | null | undefined)?.tab === "build"
          ? "build"
          : "score",
    ),
  );

  const TAB_ITEMS: Array<{ value: Tab; label: string; title: string }> = [
    { value: "score", label: "Score", title: "The model's preference among a fixed set of answers" },
    { value: "build", label: "Build", title: "A new template" },
  ];

  // ----- shared: template catalog --------------------------------------
  let templates: TemplateSummary[] = $state([]);
  let loading = $state(false);
  let deleteCandidate = $state<string | null>(null);
  let deleting = $state(false);

  async function loadTemplates(): Promise<void> {
    loading = true;
    try {
      templates = (await apiTemplates.list()).templates;
    } catch (e) {
      pushToast(`couldn't load templates: ${describeError(e)}`, { kind: "error" });
    } finally {
      loading = false;
    }
  }
  onMount(loadTemplates);
  onMount(() => onArtifactUpdate(async (area) => {
    if (area === "templates") await loadTemplates();
  }));

  // ----- score tab -----------------------------------------------------
  let selectedKey = $state("");
  let steerExpr = $state("");
  let scoring = $state(false);
  let scoreBy: "sum" | "mean" = $state("sum");
  let baseline: ChoiceScores[] | null = $state(null);
  let steered: ChoiceScores[] | null = $state(null);
  let scoredKey = $state("");
  let cancellingScore = $state(false);
  const hostedController = getHostedController();

  const selectedTemplate = $derived(
    templates.find((t) => `${t.namespace}/${t.name}` === selectedKey) ?? null,
  );

  let previewTemplate = $state<TemplateDetail | null>(null);
  let previewError = $state("");
  $effect(() => {
    const template = selectedTemplate;
    previewTemplate = null;
    previewError = "";
    if (!template) return;
    let current = true;
    void apiTemplates.get(template.namespace, template.name).then(detail => {
      if (current) previewTemplate = detail;
    }).catch(() => { if (current) previewError = "Preview unavailable. Scoring is still available."; });
    return () => { current = false; };
  });

  function probOf(c: { prob_sum: number; prob_mean: number }): number {
    return scoreBy === "sum" ? c.prob_sum : c.prob_mean;
  }

  async function runScore(): Promise<void> {
    if (!selectedTemplate || scoring) return;
    scoring = true;
    cancellingScore = false;
    baseline = null;
    steered = null;
    const { namespace, name } = selectedTemplate;
    try {
      baseline = (await apiTemplates.score(namespace, name, null)).contexts;
      const expr = steerExpr.trim();
      if (expr) {
        steered = (await apiTemplates.score(namespace, name, expr)).contexts;
      }
      scoredKey = selectedKey;
    } catch (e) {
      baseline = null;
      steered = null;
      scoredKey = "";
      if (isFittingCancellation(e)) {
        pushToast("scoring cancelled", { kind: "info" });
      } else {
        pushToast(`scoring failed: ${describeError(e)}`, { kind: "error" });
      }
    } finally {
      scoring = false;
      cancellingScore = false;
    }
  }

  async function cancelScore(): Promise<void> {
    if (!hostedController || !scoring || cancellingScore) return;
    cancellingScore = true;
    try {
      await hostedController.cancelFitting();
    } catch (e) {
      pushToast(`couldn't cancel scoring: ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      cancellingScore = false;
    }
  }

  /** Rows for one context: label + baseline prob + optional steered prob,
   *  sorted by the active baseline probability descending. */
  function rows(ctxIdx: number): {
    label: string;
    base: number;
    steer: number | null;
  }[] {
    const b = baseline?.[ctxIdx];
    if (!b) return [];
    const s = steered?.[ctxIdx] ?? null;
    const out = b.choices.map((c, i) => ({
      label: c.label,
      base: probOf(c),
      steer: s ? probOf(s.choices[i]) : null,
    }));
    out.sort((x, y) => Math.max(y.base, y.steer ?? -1) - Math.max(x.base, x.steer ?? -1));
    return out;
  }

  // ----- build tab -----------------------------------------------------
  let bName = $state("");
  let bSlot = $state("[DAY]");
  let bValuesText = $state("");
  let bContexts = $state<TemplateContextSpec[]>([
    { turns: [{ role: "user", content: "" }], assistant: "" },
  ]);
  let building = $state(false);
  let buildSubmitted = $state(false);
  let buildForm: HTMLFormElement | null = $state(null);

  const bValues = $derived(
    bValuesText
      .split(/[\n,]+/)
      .map((v) => v.trim())
      .filter(Boolean),
  );

  function addContext(): void {
    bContexts = [
      ...bContexts,
      { turns: [{ role: "user", content: "" }], assistant: "" },
    ];
  }
  function removeContext(i: number): void {
    bContexts = bContexts.filter((_, idx) => idx !== i);
  }
  function addTurn(ci: number): void {
    const ctx = bContexts[ci];
    const lastRole = ctx.turns[ctx.turns.length - 1]?.role;
    const role: TemplateTurn["role"] = lastRole === "user" ? "assistant" : "user";
    ctx.turns = [...ctx.turns, { role, content: "" }];
    bContexts = [...bContexts];
  }
  function removeTurn(ci: number, ti: number): void {
    bContexts[ci].turns = bContexts[ci].turns.filter((_, idx) => idx !== ti);
    bContexts = [...bContexts];
  }

  // Blank turn rows are the editor's "add a turn" affordance, not part of
  // the draft — they're dropped here exactly as ``submitBuild`` drops them.
  const buildDraft = $derived({
    slot: bSlot,
    values: bValues,
    contexts: bContexts.map((c) => ({
      turns: c.turns.filter((t) => t.content.trim()),
      assistant: c.assistant,
    })),
  });

  const buildValidation = $derived([
    ...(bName.trim() ? [] : ["name required"]),
    ...validateTemplateDraft(buildDraft),
  ]);
  const nameInvalid = $derived(!bName.trim());
  const slotInvalid = $derived(buildValidation.includes("slot required"));
  const valuesInvalid = $derived(
    buildValidation.some((error) => error === "≥ 2 values" || error.startsWith('value "')),
  );
  const contextInvalid = (index: number) =>
    buildValidation.some((error) => error.startsWith(`context ${index + 1}:`));
  const validationDescription = (invalid: boolean) =>
    buildSubmitted && invalid ? "template-build-errors" : undefined;

  async function submitBuild(ev: Event): Promise<void> {
    ev.preventDefault();
    if (buildValidation.length) {
      buildSubmitted = true;
      await tick();
      buildForm?.querySelector<HTMLElement>('[aria-invalid="true"]')?.focus();
      return;
    }
    building = true;
    try {
      await apiTemplates.create({
        namespace: "local",
        name: bName.trim(),
        slot: bSlot.trim(),
        values: bValues,
        contexts: buildDraft.contexts,
      });
      pushToast(`template ${bName.trim()} created`, { kind: "info" });
      await loadTemplates();
      selectedKey = `local/${bName.trim()}`;
      tab = "score";
    } catch (e) {
      pushToast(`create failed: ${describeError(e)}`, { kind: "error" });
    } finally {
      building = false;
    }
  }

  async function deleteTemplate(t: TemplateSummary): Promise<void> {
    if (deleting) return;
    deleting = true;
    try {
      await apiTemplates.delete(t.namespace, t.name);
      pushToast(`removed ${t.namespace}/${t.name}`, { kind: "info" });
      if (selectedKey === `${t.namespace}/${t.name}`) selectedKey = "";
      await loadTemplates();
      deleteCandidate = null;
    } catch (e) {
      pushToast(`delete failed: ${describeError(e)}`, { kind: "error" });
    } finally {
      deleting = false;
    }
  }

  onInterfaceMount(() => registerInterfaceController("template_lab", {
    schema: templateDraftSchema,
    read: () => ({ busy: building || scoring || deleting || cancellingScore, dirty: !!bName || bSlot !== "[DAY]" || !!bValuesText || JSON.stringify(bContexts) !== JSON.stringify([{ turns: [{ role: "user", content: "" }], assistant: "" }]), values: { tab, selected_key: selectedKey, steering: steerExpr, score_by: scoreBy, name: bName, slot: bSlot, values_text: bValuesText, contexts: bContexts }, validation: buildValidation, available_templates: templates.map(row => `${row.namespace}/${row.name}`), loading }),
    update: async (change) => {
      if (building || scoring || deleting || cancellingScore) throw new ToolError("BUSY", "Wait for this interface operation to finish.");
      if (change.selected_key && !templates.some(template => `${template.namespace}/${template.name}` === change.selected_key)) throw new ToolError("NOT_FOUND", "Choose a template from available_templates.");
      if (change.tab !== undefined) tab = change.tab as Tab;
      if (change.selected_key !== undefined) selectedKey = change.selected_key as string;
      if (change.steering !== undefined) steerExpr = change.steering as string;
      if (change.score_by !== undefined) scoreBy = change.score_by as "sum" | "mean";
      if (change.name !== undefined) bName = change.name as string;
      if (change.slot !== undefined) bSlot = change.slot as string;
      if (change.values_text !== undefined) bValuesText = change.values_text as string;
      if (change.contexts !== undefined) bContexts = structuredClone(change.contexts) as TemplateContextSpec[];
    },
  }));
</script>

<section class="mf-tool" class:dialog={!embedded} aria-label="Template lab">
  {#if !embedded}
    <header class="mf-head">
      <h2 class="dw-title">Templates</h2>
      <IconButton icon="close" label="Close" onclick={closeDrawer} />
    </header>
  {/if}

  <div class="views">
    <Segmented label="Template lab view" value={tab} onchange={(next) => (tab = next)} options={TAB_ITEMS.map(item => ({ value: item.value, label: item.label, hint: item.title }))} />
  </div>

  <div class="mb-form lab">
    {#if tab === "score"}
      {#if loading}
        <div class="dw-blocks loading-placeholder" role="status" aria-label="Loading templates"><i></i></div>
      {:else if templates.length === 0}
        <div class="dw-empty">
          <p>No templates yet.</p>
          <button type="button" class="dw-quiet" onclick={() => (tab = "build")}>Write one</button>
        </div>
      {:else}
        <section class="step" aria-label="Score a template">
          <div class="grid2">
            <label class="field">
              <span class="label">Template</span>
              <Select bind:value={selectedKey} disabled={scoring} ariaLabel="Template"
                options={[{ value: "", label: "Choose a template" }, ...templates.map(t => ({
                  value: `${t.namespace}/${t.name}`, label: `${t.namespace}/${t.name} · ${t.n_values} × ${t.n_contexts}`,
                }))]} />
            </label>
            <label class="field">
              <span class="label">Steering</span>
              <input type="text" class="input" placeholder="0.5 patient.hurried" bind:value={steerExpr}
                disabled={scoring} autocomplete="off" spellcheck="false" />
            </label>
          </div>

          {#if previewTemplate}
            {#each previewTemplate.contexts as context, index (index)}
              <TemplatePreview sentence={context.assistant} slot={previewTemplate.slot} values={previewTemplate.values} />
            {/each}
          {:else if previewError}<p class="dim-note">{previewError}</p>{/if}

          <div class="submit">
            <button type="button" class="dw-button" aria-busy={scoring} disabled={!selectedTemplate || scoring} onclick={runScore}>
              <MorphText text={scoring ? "Scoring…" : "Score"} />
            </button>
            {#if hostedController && scoring}
              <button type="button" class="dw-text" disabled={cancellingScore} onclick={cancelScore}>
                <MorphText text={cancellingScore ? "cancelling…" : "cancel"} />
              </button>
            {/if}
            <span class="rank">
              <Segmented label="Rank by" value={scoreBy} onchange={(next) => (scoreBy = next)}
                options={[{ value: "sum", label: "Sum", hint: "Rank by the summed probability of each answer's tokens" }, { value: "mean", label: "Mean", hint: "Rank by the mean probability of each answer's tokens" }]} />
            </span>
          </div>
        </section>

        {#if baseline && scoredKey === selectedKey && !scoring}
          {#each baseline as _ctx, ci (ci)}
            <section class="mf-sec result rise" style:--i={ci} aria-label={`Context ${ci + 1}`}>
              <div class="mf-sec-head"><h3 class="head">Context {ci + 1}</h3>{#if steered}<span class="meta">base → steered</span>{/if}</div>
              <div class="mass-row"><span class="meta">{steered ? "Base" : "Share"}</span><div class="mass-bar" class:is-base={!!steered} role="img" aria-label={`Baseline mass: ${rows(ci).map(r => `${r.label}: ${(r.base * 100).toFixed(1)}%`).join(", ")}`}>{#each rows(ci) as r, index (r.label)}<span style:width={`${r.base * 100}%`} style:opacity={.35 + (index % 4) * .2} data-tip={`${r.label}: ${(r.base * 100).toFixed(1)}%`}></span>{/each}</div></div>
              {#if steered}<div class="mass-row"><span class="meta">Steered</span><div class="mass-bar" role="img" aria-label={`Steered mass: ${rows(ci).map(r => `${r.label}: ${((r.steer ?? 0) * 100).toFixed(1)}%`).join(", ")}`}>{#each rows(ci) as r, index (r.label)}<span style:width={`${(r.steer ?? 0) * 100}%`} style:opacity={.35 + (index % 4) * .2} data-tip={`${r.label}: ${((r.steer ?? 0) * 100).toFixed(1)}%`}></span>{/each}</div></div>{/if}
              <div class="bar-rows">
                {#each rows(ci) as r (r.label)}
                  <div class="bar-row">
                    <span class="bar-label" {...{ "aria-description": (r.label) }}>{r.label}</span>
                    <div class="bars" class:is-pair={r.steer !== null}>
                      <span class="bar base"><i style:transform={`scaleX(${Math.max(0, Math.min(1, r.base))})`}></i></span>
                      {#if r.steer !== null}
                        <span class="bar steer"><i style:transform={`scaleX(${Math.max(0, Math.min(1, r.steer))})`}></i></span>
                      {/if}
                    </div>
                    <span class="bar-num">
                      <MorphText text={`${(r.base * 100).toFixed(0)}%`} />{#if r.steer !== null}<span class="arrow">→</span><MorphText text={`${(r.steer * 100).toFixed(0)}%`} />{/if}
                    </span>
                  </div>
                {/each}
              </div>
            </section>
          {/each}
        {/if}
      {/if}

    {:else}
      <form bind:this={buildForm} class="form-stack" aria-busy={building} onsubmit={submitBuild}>
        <section class="step" aria-label="Template">
          <h3 class="step-title"><span class="n">1</span>Answers</h3>
          <div class="grid2">
            <label class="field">
              <span class="label">Name</span>
              <input type="text" class="input" placeholder="weekday" bind:value={bName} disabled={building}
                autocomplete="off" spellcheck="false"
                aria-invalid={buildSubmitted && nameInvalid}
                aria-describedby={validationDescription(nameInvalid)} />
            </label>
            <label class="field">
              <span class="label">Slot</span>
              <input type="text" class="input" placeholder="[DAY]" bind:value={bSlot} disabled={building}
                autocomplete="off" spellcheck="false"
                aria-invalid={buildSubmitted && slotInvalid}
                aria-describedby={validationDescription(slotInvalid)} />
            </label>
          </div>
          <label class="field">
            <span class="label">Values, one per line</span>
            <textarea class="input" rows="3" placeholder={"Monday\nTuesday\nWednesday"}
              bind:value={bValuesText} disabled={building}
              aria-invalid={buildSubmitted && valuesInvalid}
              aria-describedby={validationDescription(valuesInvalid)}></textarea>
          </label>
        </section>

        <section class="step" aria-labelledby="template-contexts-title">
          <div class="mf-sec-head">
            <h3 id="template-contexts-title" class="step-title"><span class="n">2</span>Contexts</h3>
            <span class="meta"><MorphText text={bContexts.length} /></span>
          </div>
          {#each bContexts as ctx, ci (ci)}
            <div class="mf-wash ctx-build">
              <div class="mf-sec-head">
                <span class="head">Context {ci + 1}</span>
                {#if bContexts.length > 1}
                  <button
                    type="button"
                    class="dw-text"
                    aria-label={`Remove context ${ci + 1}`}
                    onclick={() => removeContext(ci)}
                  >Remove</button>
                {/if}
              </div>
              {#each ctx.turns as turn, ti (ti)}
                <div class="turn-row">
                  <Select
                    bind:value={turn.role}
                    disabled={building}
                    ariaLabel={`Context ${ci + 1}, turn ${ti + 1} role`}
                    invalid={buildSubmitted && contextInvalid(ci)}
                    ariaDescribedby={validationDescription(contextInvalid(ci))}
                    options={[{ value: "user", label: "User" }, { value: "assistant", label: "Assistant" }, { value: "system", label: "System" }]}
                  />
                  <input type="text" class="input" placeholder="Turn" bind:value={turn.content}
                    disabled={building} autocomplete="off"
                    aria-invalid={buildSubmitted && contextInvalid(ci)}
                    aria-describedby={validationDescription(contextInvalid(ci))}
                    aria-label={`Context ${ci + 1}, turn ${ti + 1} content`} />
                  {#if ctx.turns.length > 1}
                    <button
                      type="button"
                      class="node-remove"
                      aria-label={`Remove turn ${ti + 1} from context ${ci + 1}`}
                      onclick={() => removeTurn(ci, ti)}
                    ><Icon name="close" /></button>
                  {/if}
                </div>
              {/each}
              <button
                type="button"
                class="add-node"
                aria-label={`Add turn to context ${ci + 1}`}
                onclick={() => addTurn(ci)}
              >Add turn</button>
              <label class="field">
                <span class="label">Assistant, with the slot</span>
                <input type="text" class="input" placeholder={`Today is ${bSlot}`} bind:value={ctx.assistant}
                  disabled={building} autocomplete="off"
                  aria-invalid={buildSubmitted && contextInvalid(ci)}
                  aria-describedby={validationDescription(contextInvalid(ci))} />
              </label>
              <TemplatePreview sentence={ctx.assistant} slot={bSlot} values={bValuesText.split("\n").map(value => value.trim()).filter(Boolean)} />
            </div>
          {/each}
          <button type="button" class="add-node" onclick={addContext}>Add context</button>
        </section>

        {#if buildSubmitted && buildValidation.length}
          <ul id="template-build-errors" class="sk-errs" role="status" aria-live="polite">
            {#each buildValidation as e (e)}<li>{e}</li>{/each}
          </ul>
        {/if}

        <div class="submit">
          <button type="submit" class="dw-button" aria-busy={building} disabled={building}>
            <MorphText text={building ? "Creating…" : "Create template"} />
          </button>
          <button type="button" class="dw-text" onclick={closeDrawer}>Cancel</button>
        </div>
      </form>
    {/if}

    {#if templates.length > 0}
      <section class="mf-sec" aria-labelledby="template-catalog-title">
        <div class="mf-sec-head"><h3 id="template-catalog-title" class="head">Installed</h3><span class="meta">{templates.length}</span></div>
        <ul class="mf-rows" role="list">
          {#each templates as t, i (`${t.namespace}/${t.name}`)}
            <li class="mf-row stack rise" style:--i={i}>
              <span class="mf-id">
                <span class="mf-name">{t.namespace}/{t.name}</span>
                <span class="meta">{t.slot} · {t.n_values} values × {t.n_contexts} contexts</span>
              </span>
              <span class="mf-end">
                {#if deleteCandidate === `${t.namespace}/${t.name}`}
                  <span class="meta">Delete for good?</span>
                  <button type="button" class="dw-quiet" disabled={deleting} onclick={() => deleteTemplate(t)}><MorphText text={deleting ? "Deleting…" : "Delete template"} /></button>
                  <button type="button" class="dw-text" disabled={deleting} onclick={() => (deleteCandidate = null)}>Cancel</button>
                {:else}
                  <button
                    type="button"
                    class="dw-text"
                    disabled={deleting}
                    aria-label={`Delete template ${t.namespace}/${t.name}`}
                    onclick={() => (deleteCandidate = `${t.namespace}/${t.name}`)}
                  >Delete</button>
                {/if}
              </span>
            </li>
          {/each}
        </ul>
      </section>
    {/if}
  </div>
</section>

<style>
  .views { display: flex; min-width: 0; }
  .lab { max-width: 760px; }
  .rank { margin-left: auto; }

  /* The restricted-choice read: each answer's share before and after steering, as meters on the
   * ramp. Without steering the share is the data, so it takes the tint; with it, the base steps
   * back to faint ink and the steered share takes the tint. */
  .result { gap: var(--s2); }
  .mass-row { display: grid; grid-template-columns: 64px minmax(0, 1fr); align-items: center; gap: var(--s3); }
  .mass-bar { display: flex; gap: 2px; height: 8px; overflow: hidden; border-radius: 2px; }
  .mass-bar span { flex: none; min-width: 0; border-radius: 2px; background: var(--tint); transition: width var(--glide) var(--ease); }
  .mass-bar.is-base span { background: var(--ink3); }
  .bar-rows { display: grid; gap: 2px; margin-top: var(--s2); }
  .bar-row { display: grid; grid-template-columns: minmax(64px, 160px) minmax(0, 1fr) 88px; align-items: center; gap: var(--s3); min-height: 32px; padding: 0 var(--s3); border-radius: var(--r-col); transition: background-color var(--fast) var(--ease); }
  @media (hover: hover) { .bar-row:hover { background: var(--wash); } }
  .bar-label { overflow: hidden; color: var(--ink); font: var(--type-small); white-space: nowrap; text-overflow: ellipsis; }
  .bars { display: grid; gap: 2px; }
  .bar { position: relative; display: block; height: 4px; border-radius: 2px; background: var(--wash2); overflow: hidden; }
  .bar i { position: absolute; inset: 0; border-radius: 2px; background: var(--tint); transform-origin: left; transition: transform var(--glide) var(--ease); }
  .is-pair .bar.base i { background: var(--ink3); opacity: 0.6; }
  .bar-num { color: var(--ink2); font: var(--type-meta); text-align: end; white-space: nowrap; }
  .arrow { margin: 0 var(--s1); color: var(--ink3); }

  .ctx-build { gap: var(--s2); }
  .turn-row { display: grid; grid-template-columns: 128px minmax(0, 1fr) auto; align-items: center; gap: var(--s2); }
  .sk-errs { display: grid; gap: var(--s1); margin: 0; padding: var(--s4); border-radius: var(--r-box); background: var(--wash); color: var(--ink); font: var(--type-small); list-style: none; }

  @media (max-width: 520px) {
    .mf-row.stack { grid-template-columns: minmax(0, 1fr); }
    .stack .mf-end { justify-content: flex-start; }
    .rank { margin-left: 0; }
    .bar-row { grid-template-columns: 88px minmax(0, 1fr) 76px; padding: 0 var(--s2); }
    .turn-row { grid-template-columns: minmax(0, 1fr) auto; }
    .turn-row > :global(:first-child) { grid-column: 1 / -1; }
  }
</style>
