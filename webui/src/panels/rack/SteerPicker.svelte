<script lang="ts">
  // "+ Steer": what a term can steer with, in priority order, each with a picker. A neuron or a head
  // (by its ID, by the word it writes, or among the units most active in the latest reply; a neuron
  // is pushed or silenced, a head only silenced), a lens token (what a layer says, ``jlens/``, or
  // what it reads, ``slens/``: a word, or one the lens reads now), a concept (a flat fit's node: a
  // pole, a persona), a manifold position (a curved fit's node) and an SAE feature (its number, or a
  // label among the features seen). A pick adds the term through lib/steerActions.ts, which selects
  // its chip, so its editor opens for the strength. A kind this runtime cannot read is left out.
  import { onMount, tick } from "svelte";
  import { Segmented, Tabs, tokenParts } from "@stdlogits/ledger";
  import { apiInstruments, describeError } from "../../lib/runtime/services";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { genStatus } from "../../lib/stores/chat.svelte";
  import { openDrawer } from "../../lib/stores/drawers.svelte";
  import {
    LENS_NAMESPACES,
    lensAggregateForDisplay,
    lensNamespaceInUse,
    lensNamespaceReady,
    lensPairState,
    saeReadoutForDisplay,
    saeState,
    type LensNamespace,
  } from "../../lib/stores/instruments.svelte";
  import { goto } from "../../lib/stores/rooms.svelte";
  import { saeLoaded, sessionState } from "../../lib/stores/session.svelte";
  import { refreshManifoldList, steerRack } from "../../lib/stores/steering.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import {
    latestReply,
    nodeActivity,
    requestNodeActivity,
    requestUnitsForWord,
    retryUnits,
    unitActivityAvailable,
    unitLabelFor,
    unitsAvailable,
    unitsForWord,
  } from "../../lib/stores/units.svelte";
  import { fitGeometry } from "../../lib/manifolds/fitKind";
  import { manifoldSelectorOptions, selectedManifoldOption } from "../../lib/manifolds/selectors";
  import { pushConcept, pushFeature, pushLensToken, pushPosition, pushUnit, silenceUnit } from "../../lib/steerActions";
  import { labelText, parseQuery, signedValue, unitTitle, type UnitRef } from "../../lib/units";
  import { UnitLabels } from "../../drawers/token/labels.svelte";
  import type { UnitLabelJSON, Variant } from "../../lib/types";

  let { onpicked }: { onpicked: () => void } = $props();

  type Kind = "unit" | "lens" | "concept" | "position" | "feature";
  const KINDS: readonly { value: Kind; label: string; placeholder: string }[] = [
    { value: "unit", label: "Neuron or head", placeholder: "An ID like L20 n512, or a word" },
    { value: "lens", label: "Lens token", placeholder: "A word" },
    { value: "concept", label: "Concept", placeholder: "Find a concept" },
    { value: "position", label: "Manifold position", placeholder: "Find a position" },
    { value: "feature", label: "SAE feature", placeholder: "A feature number or label" },
  ];

  const lensReady = $derived(lensNamespaceReady("jlens") || lensNamespaceReady("slens"));
  const kinds = $derived(KINDS.filter((k) => {
    if (k.value === "unit") return unitsAvailable();
    if (k.value === "lens") return lensReady;
    if (k.value === "feature") return saeLoaded();
    return true;
  }));
  let chosen = $state<Kind>(lastKind);
  const kind = $derived<Kind>(kinds.some((k) => k.value === chosen) ? chosen : kinds[0]?.value ?? "concept");
  const placeholder = $derived(KINDS.find((k) => k.value === kind)?.placeholder ?? "");

  let field: HTMLInputElement | null = $state(null);
  let query = $state("");
  /** The query a search runs on, a moment after typing stops (the unit search asks the server). */
  let settled = $state("");
  let timer: ReturnType<typeof setTimeout> | null = null;
  function typed(value: string): void {
    query = value;
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => (settled = query), 250);
  }
  function choose(next: Kind): void {
    chosen = next;
    lastKind = next;
    query = "";
    settled = "";
    void tick().then(() => field?.focus());
  }
  onMount(() => {
    field?.focus();
    if (steerRack.catalog.length === 0 && !steerRack.loading) void refreshManifoldList();
    return () => { if (timer) clearTimeout(timer); };
  });

  let busy = $state(false);
  function picked(name: string | null): void {
    if (name !== null) onpicked();
  }

  // ---------- a neuron or a head
  interface UnitPick { id: string; kind: "neuron" | "head"; pole: "+" | "-"; label: string | null; value: string | null }
  const uq = $derived(kind === "unit" ? parseQuery(settled) : null);
  const architecture = $derived(sessionState.info?.architecture ?? null);
  /** Whether the model has the unit (when the runtime describes its shape). */
  function exists(ref: UnitRef): boolean {
    if (!architecture) return true;
    const per = ref.kind === "neuron" ? architecture.neurons_per_layer : architecture.heads_per_layer;
    return ref.layer < architecture.layers && (per === null || ref.index < per);
  }
  const reply = $derived(uq?.mode === "active" && unitActivityAvailable() ? latestReply() : null);
  // The replay holds the model, so it waits for the picker to rest on an empty search: a unit named
  // at once (its chip asks where it acts) never finds the model busy.
  $effect(() => {
    if (!reply || genStatus.active) return;
    const nodeId = reply.nodeId;
    const timer = setTimeout(() => requestNodeActivity(nodeId, { top: 32 }), 600);
    return () => clearTimeout(timer);
  });
  const active = $derived(reply ? nodeActivity(reply.nodeId, { top: 32 }) : null);
  const word = $derived(uq?.mode === "word" ? uq.word : null);
  $effect(() => {
    if (word && !genStatus.active) requestUnitsForWord(word, "writes", 48);
  });
  const writers = $derived(word ? unitsForWord(word, "writes", 48) : null);
  const labelOf = (id: string, pole: "+" | "-", label?: UnitLabelJSON | null) => labelText(label) ?? unitLabelFor(id, pole);
  /** The catalog's label for a unit named by its ID (a search hit carries its own). */
  const catalog = new UnitLabels();
  const named = $derived(uq?.mode === "id" && uq.ref && exists(uq.ref) ? uq.ref : null);
  $effect(() => {
    if (named) catalog.want([{ unit: named.id }]);
  });

  const unitPicks = $derived.by((): UnitPick[] => {
    if (!uq) return [];
    if (uq.mode === "id") {
      if (!uq.ref || !exists(uq.ref)) return [];
      const id = uq.ref.id;
      return [{ id, kind: uq.ref.kind, pole: "+", label: catalog.label(id, uq.ref.kind, "+")?.text ?? unitLabelFor(id), value: null }];
    }
    if (uq.mode === "word") {
      return (writers?.value?.hits ?? []).slice(0, 8).map((h) => {
        const pole = h.pole === "-" ? "-" : "+";
        return { id: h.unit, kind: "neuron", pole, label: labelOf(h.unit, pole, h.label), value: signedValue(h.score) };
      });
    }
    return (active?.value?.top ?? []).slice(0, 8).map((t) => {
      const pole = t.kind === "neuron" && t.peak < 0 ? "-" : "+";
      return { id: t.unit, kind: t.kind, pole, label: labelOf(t.unit, pole, t.label), value: signedValue(t.peak) };
    });
  });
  const unitHead = $derived.by((): string | null => {
    if (!uq || unitPicks.length === 0) return null;
    if (uq.mode === "word") return `Writers of ${tokenParts(word ?? "").text.trim() || word}`;
    if (uq.mode === "active") return "Most active in the latest reply";
    return null;
  });
  const unitStatus = $derived.by((): { text: string; retry?: boolean } | null => {
    if (!uq || unitPicks.length > 0) return null;
    if (uq.mode === "id") return uq.ref && query.trim() === settled.trim() ? { text: "This model has no such unit." } : null;
    const loaded = uq.mode === "word" ? writers : active;
    if (uq.mode === "active" && !reply) return null;
    if (genStatus.active && !loaded?.value) return { text: "Waiting for the reply to finish." };
    if (loaded?.status === "loading") return { text: "Reading…" };
    if (loaded?.status === "error") return { text: loaded.error ?? "This could not be read.", retry: true };
    if (uq.mode === "word" && loaded?.status === "ready") return { text: "No neuron writes it." };
    return null;
  });

  // ---------- a lens token
  const lensBoth = $derived(lensNamespaceReady("jlens") && lensNamespaceReady("slens"));
  const lensAs = $derived(lensNamespaceInUse());
  const pair = $derived(LENS_NAMESPACES.find((row) => row.value === lensAs) ?? LENS_NAMESPACES[0]);
  /** Words the lens reads now (live, or at the hovered token), by vocabulary id. */
  const lensNow = $derived(kind === "lens" ? (lensAggregateForDisplay() ?? []).slice(0, 8) : []);

  /** A word is checked against the vocabulary (one token); the family resolves, since only a ready
   *  one is offered, and the chip's editor says where the term acts. */
  async function addLensWord(raw: string): Promise<void> {
    const bare = raw.trim().replace(/^[js]lens\//u, "");
    const namespace = lensAs;
    if (!bare || busy) return;
    busy = true;
    try {
      const validated = await apiInstruments.validateLensToken(bare);
      picked(pushLensToken({ word: validated.word }, namespace));
    } catch (e) {
      pushToast(`Couldn't steer by ${namespace}/${bare}: ${describeError(e)}`, { kind: "error" });
    } finally {
      busy = false;
    }
  }

  // ---------- a concept or a manifold position: a fit's nodes, grouped by fit
  interface FitGroup { key: string; title: string; variant: Variant; nodes: string[]; more: number }
  const model = $derived(sessionState.info?.model_id ?? null);
  let opened = $state<string | null>(null);
  const fitGroups = $derived.by((): FitGroup[] => {
    if (kind !== "concept" && kind !== "position") return [];
    const geometry = kind === "concept" ? "flat" : "curved";
    const q = settled.trim().toLowerCase();
    const out: FitGroup[] = [];
    for (const m of steerRack.catalog) {
      if (!m.fitted_for_session || fitGeometry(m) !== geometry) continue;
      const choice = selectedManifoldOption(manifoldSelectorOptions(m, model), undefined);
      if (!choice?.available) continue;
      const key = `${m.namespace}/${m.name}`;
      const named = !q || key.toLowerCase().includes(q) || (m.description ?? "").toLowerCase().includes(q);
      const nodes = m.node_labels.filter((label) => named || label.toLowerCase().includes(q));
      if (nodes.length === 0) continue;
      const cap = opened === key ? nodes.length : 16;
      out.push({ key, title: m.name, variant: choice.variant, nodes: nodes.slice(0, cap), more: Math.max(0, nodes.length - cap) });
    }
    return out;
  });
  function pickNode(group: FitGroup, label: string): void {
    picked(kind === "concept" ? pushConcept(group.key, group.variant, label) : pushPosition(group.key, group.variant, label));
  }
  const fitStatus = $derived.by((): string | null => {
    if (kind !== "concept" && kind !== "position" || fitGroups.length > 0) return null;
    if (steerRack.loading) return "Reading the catalog…";
    if (settled.trim()) return "Nothing fitted matches.";
    return kind === "concept" ? "No concept is fitted for this model." : "No manifold is fitted for this model.";
  });

  // ---------- an SAE feature
  interface FeaturePick { id: number; label: string | null }
  const featurePicks = $derived.by((): FeaturePick[] => {
    if (kind !== "feature") return [];
    const q = settled.trim().toLowerCase().replace(/^sae\//u, "");
    const firing = saeReadoutForDisplay().map((row) => ({ id: row.id, label: row.label ?? saeState.meta.get(row.id)?.label ?? null }));
    if (!q) return firing.slice(0, 8);
    if (/^\d+$/u.test(q)) return [{ id: Number(q), label: saeState.meta.get(Number(q))?.label ?? null }];
    const seen = new Map<number, FeaturePick>();
    for (const row of firing) seen.set(row.id, row);
    for (const [id, meta] of saeState.meta) if (!seen.has(id)) seen.set(id, { id, label: meta.label });
    return [...seen.values()].filter((row) => row.label?.toLowerCase().includes(q)).slice(0, 8);
  });
  async function addFeature(id: number): Promise<void> {
    if (busy) return;
    busy = true;
    try {
      const validated = await apiInstruments.validateSaeFeature(id);
      picked(pushFeature(validated.id));
    } catch (error) {
      pushToast(userFacingError(error, `Feature ${id} could not be added. Try again.`), { kind: "error" });
    } finally {
      busy = false;
    }
  }

  // ---------- Enter takes the first pick
  function submit(event: SubmitEvent): void {
    event.preventDefault();
    settled = query;
    const text = query.trim();
    if (kind === "lens") {
      void addLensWord(text);
    } else if (kind === "feature") {
      const id = Number(text.replace(/^sae\//u, ""));
      if (Number.isInteger(id) && id >= 0 && text !== "") void addFeature(id);
      else if (featurePicks[0]) void addFeature(featurePicks[0].id);
    } else if (kind === "concept" || kind === "position") {
      const group = fitGroups[0];
      if (group && group.nodes[0] !== undefined) pickNode(group, group.nodes[0]);
    } else if (kind === "unit") {
      const ref = parseQuery(text);
      if (ref.mode === "id" && ref.ref && exists(ref.ref)) picked(ref.ref.kind === "neuron" ? pushUnit(ref.ref.id) : silenceUnit(ref.ref.id));
      else if (ref.mode !== "id" && unitPicks[0]) {
        const first = unitPicks[0];
        picked(first.kind === "neuron" ? pushUnit(first.id, first.pole) : silenceUnit(first.id));
      }
    }
  }
</script>

<script lang="ts" module>
  /** The kind "+ Steer" opens on: the last one chosen (this page's memory). */
  let lastKind: "unit" | "lens" | "concept" | "position" | "feature" = "unit";
</script>

<section class="picker" aria-label="Add a steering term">
  <Tabs label="Steer with" value={kind} options={kinds.map((k) => ({ value: k.value, label: k.label }))} onchange={choose} scroll />

  <form class="find" onsubmit={submit}>
    <label class="sr-only" for="steer-find">{KINDS.find((k) => k.value === kind)?.label}</label>
    <input id="steer-find" type="search" autocomplete="off" spellcheck="false" bind:this={field} value={query} {placeholder}
      inputmode={kind === "feature" && /^\d*$/u.test(query) ? "numeric" : undefined}
      oninput={(e) => typed((e.currentTarget as HTMLInputElement).value)} />
    {#if kind === "lens" && lensBoth}
      <Segmented label="Toward the word" value={lensAs} options={[...LENS_NAMESPACES]}
        onchange={(value: LensNamespace) => (lensPairState.namespace = value)} />
    {/if}
    {#if kind === "lens" || kind === "feature"}
      <button type="submit" class="dw-quiet" disabled={busy || !query.trim()} data-tip={kind === "lens" ? pair.hint : undefined}>Push</button>
    {/if}
  </form>

  {#if kind === "unit"}
    {#if unitHead}<span class="head">{unitHead}</span>{/if}
    {#if unitPicks.length > 0}
      <div class="picks" role="list">
        {#each unitPicks as u (u.id)}
          <div class="pick" role="listitem">
            <span class="pid">{unitTitle(u.id)}</span>
            <span class="plabel">{u.label ?? (u.kind === "head" ? "attention head" : "MLP neuron")}</span>
            {#if u.value !== null}<span class="pval">{u.value}</span>{/if}
            <span class="acts">
              {#if u.kind === "neuron"}
                <button type="button" class="dw-text" onclick={() => picked(pushUnit(u.id, u.pole))}>Push{u.pole === "-" ? " −" : ""}</button>
              {/if}
              <button type="button" class="dw-text" onclick={() => picked(silenceUnit(u.id))}>Silence</button>
            </span>
          </div>
        {/each}
      </div>
    {/if}
    {#if unitStatus}
      <div class="dw-row"><span class="meta" role="status">{unitStatus.text}</span>{#if unitStatus.retry}<button type="button" class="dw-text" onclick={retryUnits}>Try again</button>{/if}</div>
    {/if}
    <div class="foot"><button type="button" class="dw-text" onclick={() => { onpicked(); goto("units"); }}>Units room</button></div>
  {:else if kind === "lens"}
    {#if lensNow.length > 0}
      <span class="head">The lens reads</span>
      <div class="chips">
        {#each lensNow as chip (chip[4])}
          {@const parts = tokenParts(chip[0])}
          <button type="button" class="node" disabled={busy} data-tip={pair.hint}
            onclick={() => picked(pushLensToken({ id: chip[4], text: chip[0] }, lensAs))}><span class="ws">{parts.lead ? "·" : ""}</span>{parts.empty ? "∅" : parts.text}</button>
        {/each}
      </div>
    {/if}
  {:else if kind === "concept" || kind === "position"}
    {#if fitGroups.length > 0}
      <div class="groups">
        {#each fitGroups as group (group.key)}
          <div class="group">
            <span class="head" data-tip={group.key}>{group.title}</span>
            <div class="chips">
              {#each group.nodes as label (label)}
                <button type="button" class="node" onclick={() => pickNode(group, label)}>{label}</button>
              {/each}
              {#if group.more > 0}
                <button type="button" class="dw-text more" onclick={() => (opened = group.key)}>{group.more} more</button>
              {/if}
            </div>
          </div>
        {/each}
      </div>
    {/if}
    {#if fitStatus}<span class="meta" role="status">{fitStatus}</span>{/if}
    <div class="foot">
      <button type="button" class="dw-text" onclick={() => { onpicked(); openDrawer(kind === "concept" ? "subspace" : "manifolds"); }}>{kind === "concept" ? "All concepts" : "All manifolds"}</button>
      <button type="button" class="dw-text" onclick={() => { onpicked(); goto("artifacts", "build"); }}>Build one</button>
    </div>
  {:else if kind === "feature"}
    {#if featurePicks.length > 0}
      <div class="picks" role="list">
        {#each featurePicks as f (f.id)}
          <div class="pick" role="listitem">
            <span class="pid">{f.id}</span>
            <span class="plabel">{f.label ?? "SAE feature"}</span>
            <span class="acts"><button type="button" class="dw-text" disabled={busy} onclick={() => void addFeature(f.id)}>Push</button></span>
          </div>
        {/each}
      </div>
    {/if}
  {/if}
</section>

<style>
  /* The term editor's surface: a wash panel under the chips. */
  .picker { display: grid; gap: var(--s3); min-width: 0; padding: var(--s4) var(--s5) var(--s5); border-radius: var(--r-row); background: var(--wash); }
  .find { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s3); min-width: 0; }
  .find input {
    flex: 1 1 200px;
    min-width: 0;
    height: 40px;
    padding: 0 var(--s4);
    border: 0;
    border-radius: var(--r-pill);
    background: var(--paper);
    color: var(--ink);
    font: var(--type-body);
  }
  .find input::placeholder { color: var(--ink3); }
  .find input::-webkit-search-cancel-button { display: none; }
  .find input:focus { outline: none; }
  .find input:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
  .find :global(.seg) { background: var(--wash2); }
  .find .dw-quiet { background: var(--paper); }
  .find .dw-quiet:hover:not(:disabled) { background: var(--wash2); }

  .picks { display: grid; gap: 2px; }
  .pick { display: grid; grid-template-columns: 88px minmax(0, 1fr) auto auto; gap: var(--s3); align-items: center; min-height: 40px; padding: 0 var(--s2) 0 var(--s3); border-radius: var(--r-box); transition: background-color var(--fast) var(--ease); }
  .pick:hover { background: var(--wash2); }
  .pid { color: var(--ink); font: 600 14px/20px var(--display); letter-spacing: -0.01em; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .plabel { min-width: 0; overflow: hidden; color: var(--ink2); font: var(--type-small); white-space: nowrap; text-overflow: ellipsis; }
  .pval { color: var(--ink3); font: var(--type-meta); font-variant-numeric: tabular-nums; }
  .acts { display: flex; gap: var(--s4); padding-left: var(--s2); }
  .acts .dw-text { height: 32px; color: var(--ink2); }
  .acts .dw-text:hover:not(:disabled) { color: var(--ink); }

  .groups { display: grid; gap: var(--s4); max-height: 320px; overflow-y: auto; overscroll-behavior: contain; }
  .group { display: grid; gap: var(--s2); }
  .chips { display: flex; flex-wrap: wrap; gap: var(--s1); min-width: 0; }
  .node { height: 32px; padding: 0 var(--s3); border: 0; border-radius: var(--r-pill); background: var(--paper); color: var(--ink); font: var(--type-small); white-space: pre; cursor: pointer; transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .node:hover:not(:disabled) { background: var(--wash2); }
  .node:active:not(:disabled) { scale: 0.97; }
  .node:disabled { opacity: 0.4; cursor: default; }
  .ws { color: var(--ink3); }
  .more { height: 32px; font: var(--type-meta); }
  .foot { display: flex; flex-wrap: wrap; gap: var(--s1) var(--s4); }
  .foot .dw-text { font: var(--type-meta); }

  @media (max-width: 720px) {
    .picker { padding: var(--s3) var(--s4) var(--s4); max-height: 60dvh; overflow-y: auto; overscroll-behavior: contain; }
    .find input { height: 44px; }
    .pick { grid-template-columns: minmax(0, 1fr) auto; grid-template-areas: "id acts" "label label"; gap: 0 var(--s3); padding: var(--s1) var(--s2); }
    .pick .pid { grid-area: id; }
    .pick .plabel { grid-area: label; }
    .pick .pval { display: none; }
    .pick .acts { grid-area: acts; }
    .acts .dw-text, .node, .foot .dw-text, .more { min-height: 44px; }
    .groups { max-height: none; overflow: visible; }
  }
</style>
