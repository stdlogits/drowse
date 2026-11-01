<script lang="ts">
  // The open term, in place under the recipe's chips (the Steering canvas): a wash surface with the
  // term's name and what it is, its strength (a big readout that scrubs sideways, over a signed
  // range), when it acts, its position on its geometry, where it acts by layer (the depth strip),
  // and the term as the expression writes it. A unit (a neuron or a head) follows lib/unitTerms.ts:
  // its push and its ablation each take their own strength, a head is only ablated, and its strip
  // shows its own layer, fixed; its catalog label joins the facts, and it opens its unit view. Every change applies at once (through the rack's
  // pending queue while a reply generates); Done closes it.
  import { MorphText, Segmented, Switch } from "@stdlogits/ledger";
  import Select from "../../lib/Select.svelte";
  import Slider from "../../lib/Slider.svelte";
  import XYPad from "../manifold/XYPad.svelte";
  import LayerStrip from "./LayerStrip.svelte";
  import {
    ATOM_DEFAULT_ALPHA,
    manifoldByName,
    patchSteerEntry,
    setSteerCoords,
    setSteerLabel,
    setSteerLayers,
    setSubspaceAlong,
    steerRack,
  } from "../../lib/stores/steering.svelte";
  import { sessionState } from "../../lib/stores/session.svelte";
  import { select } from "../../lib/stores/selection.svelte";
  import { setUnitOperation, unitTermLabel, unitsAvailable } from "../../lib/stores/units.svelte";
  import { formatTerm } from "../../lib/expression";
  import { layerClause, layerRunsText, layerTint, peakLayer } from "../../lib/steering/layers";
  import { placeOf } from "../../lib/steering/termLayers.svelte";
  import {
    WHEN,
    canRemove,
    coefficientText,
    removes,
    strengthDial,
    strengthSays,
    termKind,
    termSource,
    termTitle,
    unitLayer,
    wrapPieces,
  } from "../../lib/steering/terms";
  import type { LayerScope, SteerEntry, Trigger } from "../../lib/types";

  let { name, entry, onclose, onremove }: {
    name: string;
    entry: SteerEntry;
    onclose: () => void;
    onremove: () => void;
  } = $props();

  const uid = $props.id();
  const info = $derived(entry.mode === "subspace" || entry.mode === "manifold" ? manifoldByName(name) : null);
  const title = $derived(termTitle(name, entry, entry.mode === "manifold" ? manifoldByName(name) : null));
  const removal = $derived(removes(name, entry));
  const place = $derived(placeOf(name, entry, sessionState.info?.model_id));
  const where = $derived(place.where);
  const unitAt = $derived(entry.mode === "unit" ? unitLayer(name) : null);

  // A unit's catalog label, for the pole its term pushes (a head's under +), once one is known.
  const unitLabel = $derived(entry.mode === "unit" ? unitTermLabel(name, entry) : null);

  const facts = $derived.by(() => {
    const at = unitAt !== null
      ? `L${unitAt}`
      : where && where.fitted.length ? `fitted at L${layerRunsText(where.fitted)}` : "";
    return [termSource(name, entry), termKind(entry, info), at, unitLabel].filter(Boolean).join(" · ");
  });

  // The term's colour: the depth where its share peaks among the layers it acts on.
  const tint = $derived.by(() => {
    if (!where) return null;
    const peak = unitAt ?? peakLayer(where.fitted, where.share, place.selected.length ? place.selected : undefined);
    return peak === null ? null : layerTint(peak, where.nLayers);
  });

  // ---------- strength

  const dial = $derived(strengthDial(name, entry, steerRack.subspaceAlong));
  const says = $derived(strengthSays(name, entry, dial.value));
  const label = $derived(dial.shared ? "Shared strength" : removal ? "How much" : "Strength");

  function setStrength(raw: number): void {
    if (!Number.isFinite(raw)) return;
    const value = Math.round(Math.min(dial.max, Math.max(dial.min, raw)) * 100) / 100;
    if (entry.mode === "subspace") setSubspaceAlong(value);
    else if (entry.mode === "manifold") patchSteerEntry(name, { blend: value }, ["manifold"]);
    else patchSteerEntry(name, { alpha: value }, ["jlens", "slens", "sae", "unit"]);
  }

  /** The range's track: filled from zero to the thumb, in the term's colour (ink, faint, below
   *  zero), measured in the thumb's own coordinates. */
  const fill = $derived.by(() => {
    const span = dial.max - dial.min || 1;
    const at = (v: number) => `calc(8px + (100% - 16px) * ${((v - dial.min) / span).toFixed(4)})`;
    const zero = at(Math.min(dial.max, Math.max(dial.min, 0)));
    const thumb = at(dial.value);
    return dial.value >= 0
      ? `linear-gradient(90deg, var(--wash2) 0 ${zero}, var(--tint) ${zero} ${thumb}, var(--wash2) ${thumb} 100%)`
      : `linear-gradient(90deg, var(--wash2) 0 ${thumb}, color-mix(in oklab, var(--ink3) 45%, var(--wash2)) ${thumb} ${zero}, var(--wash2) ${zero} 100%)`;
  });

  /** The readout scrubs: drag it sideways and the whole range passes under 240 px. (The range
   *  beside it is the keyboard's way in.) */
  function scrubbable(node: HTMLElement) {
    let scrub: { x: number; from: number } | null = null;
    const down = (event: PointerEvent) => {
      if (event.button !== 0) return;
      node.setPointerCapture(event.pointerId);
      scrub = { x: event.clientX, from: dial.value };
    };
    const move = (event: PointerEvent) => {
      if (!scrub) return;
      const raw = scrub.from + ((event.clientX - scrub.x) / 240) * (dial.max - dial.min);
      setStrength(Math.round(raw / dial.step) * dial.step);
    };
    const up = () => { scrub = null; };
    node.addEventListener("pointerdown", down);
    node.addEventListener("pointermove", move);
    node.addEventListener("pointerup", up);
    node.addEventListener("pointercancel", up);
    return {
      destroy() {
        node.removeEventListener("pointerdown", down);
        node.removeEventListener("pointermove", move);
        node.removeEventListener("pointerup", up);
        node.removeEventListener("pointercancel", up);
      },
    };
  }

  // ---------- push or remove

  const removable = $derived(canRemove(name, entry, info));
  function setRemoval(remove: boolean): void {
    if (remove === removal) return;
    if (entry.mode === "subspace") patchSteerEntry(name, { ablate: remove }, ["subspace"]);
    else if (entry.mode === "unit") setUnitOperation(name, remove);
    else patchSteerEntry(name, { ablate: remove, alpha: remove ? 1 : ATOM_DEFAULT_ALPHA }, ["jlens", "slens", "sae"]);
  }

  // ---------- when

  const thinks = $derived(sessionState.info?.supports_thinking === true);
  const when = $derived(WHEN
    .filter(option => !option.thinking || thinks || option.value === entry.trigger)
    .map(({ value, label, hint }) => ({ value, label, hint })));

  // ---------- position (a node of a fit, or a free point on it)

  const positioned = $derived((entry.mode === "subspace" && !entry.ablate) || entry.mode === "manifold");
  const nodeLabel = $derived(entry.mode === "subspace" || entry.mode === "manifold" ? entry.label : null);
  const coords = $derived(entry.mode === "subspace" || entry.mode === "manifold" ? entry.coords : []);
  const nodeOptions = $derived([
    { value: "", label: "Free position" },
    ...(info?.node_labels ?? []).map((node, i) => {
      const role = info?.node_roles?.[i];
      return { value: node, label: role ? `${node} (as ${role})` : node };
    }),
  ]);
  const orderedNodes = $derived.by(() => {
    if (!info || info.intrinsic_dim !== 1 || info.is_discover || info.node_coords.length !== info.node_labels.length) return [];
    return info.node_labels
      .map((node, i) => ({ node, at: info.node_coords[i]?.[0] }))
      .filter(row => Number.isFinite(row.at))
      .sort((a, b) => (a.at ?? 0) - (b.at ?? 0));
  });

  /** The node the slider sits at: the term's own node, else the one nearest its free position (a
   *  free point at 6.0 on months sits by its sixth node, not at the first). */
  const nodeIndex = $derived.by(() => {
    const named = orderedNodes.findIndex(row => row.node === nodeLabel);
    if (named >= 0) return named;
    const at = coords[0];
    if (orderedNodes.length === 0 || !Number.isFinite(at)) return 0;
    let best = 0;
    orderedNodes.forEach((row, i) => {
      if (Math.abs((row.at ?? 0) - at) < Math.abs((orderedNodes[best].at ?? 0) - at)) best = i;
    });
    return best;
  });

  // ---------- where

  const scope = $derived<LayerScope>(entry.mode === "unit" ? null : entry.layers ?? null);
  const noLayers = $derived(Array.isArray(scope) && scope.length === 0);

  // ---------- the term as the expression writes it

  const text = $derived(formatTerm(name, entry, steerRack.subspaceAlong));
  const clause = $derived(entry.mode === "unit" ? "" : layerClause(entry.layers));
  // Split where the line may wrap (lib/steering/terms.ts::wrapPieces), so it never breaks inside a
  // name or a clause's braces.
  const parts = $derived.by(() => {
    const at = clause ? text.lastIndexOf(clause) : -1;
    const [head, own, tail] = at < 0 ? [text, "", ""] : [text.slice(0, at), clause, text.slice(at + clause.length)];
    return { head: wrapPieces(head), clause: wrapPieces(own), tail: wrapPieces(tail) };
  });
  let copied = $state(false);
  async function copy(): Promise<void> {
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
    } catch {
      copied = false;
    }
  }
  $effect(() => {
    void text;
    copied = false;
  });
</script>

<section class="term rise" aria-label={title} style:--tint={tint}>
  <div class="top">
    <div class="name">
      <h2 class="dw-title">{title}</h2>
      {#if facts}<span class="meta">{facts}</span>{/if}
    </div>
    <div class="dw-row toggles">
      {#if removable}
        <Segmented label="Push or remove {title}" value={removal ? "remove" : "push"}
          options={[{ value: "push", label: "Push" }, { value: "remove", label: "Without" }]}
          onchange={(value) => setRemoval(value === "remove")} />
      {/if}
      <Switch label="On" checked={entry.enabled} onchange={(on) => patchSteerEntry(name, { enabled: on })} />
    </div>
  </div>

  <div class="controls">
    <div class="col strength">
      <div class="dw-sec-head"><label class="head" for="{uid}-alpha">{label}</label><span class="meta"><MorphText text={says} numbers={false} /></span></div>
      <div class="dial">
        <output class="readout" for="{uid}-alpha" aria-hidden="true" use:scrubbable>{coefficientText(dial.value, { fixed: true })}</output>
        <div class="range">
          <input id="{uid}-alpha" type="range" min={dial.min} max={dial.max} step={dial.step} value={dial.value}
            style:--fill={fill} aria-valuetext="{coefficientText(dial.value)}, {says}"
            oninput={(event) => setStrength(Number(event.currentTarget.value))} />
          <div class="ticks meta" aria-hidden="true">
            <span>{coefficientText(dial.min)}</span><span>{dial.min < 0 ? "0" : ""}</span><span>{coefficientText(dial.max)}</span>
          </div>
        </div>
      </div>
    </div>
    <div class="col">
      <span class="head" id="{uid}-when">When</span>
      <Segmented label="When {title} acts" value={entry.trigger} options={when}
        onchange={(value: Trigger) => patchSteerEntry(name, { trigger: value })} />
    </div>
  </div>

  {#if positioned && info}
    <div class="position">
      <div class="dw-sec-head"><span class="head">Position</span>{#if info.node_labels.length > 0}<span class="pick">
        <Select value={nodeLabel ?? ""} options={nodeOptions} ariaLabel="Node of {info.name}"
          onchange={(value) => setSteerLabel(name, value === "" ? null : value)} /></span>{/if}</div>
      {#if orderedNodes.length > 2}
        <Slider value={nodeIndex} min={0} max={orderedNodes.length - 1} step={1}
          displayValue={nodeLabel ?? `Free, near ${orderedNodes[nodeIndex]?.node ?? "a node"}`} ariaLabel="Nodes of {info.name} in order"
          oninput={(index) => setSteerLabel(name, orderedNodes[index].node)} />
      {/if}
      {#if nodeLabel === null || info.node_count > 2}
        <!-- A concept's pole needs no coordinates; a free point or a larger fit shows where it sits. -->
        <XYPad manifold={info} {coords} locked={nodeLabel !== null} onchange={(next) => setSteerCoords(name, next)} />
      {/if}
      {#if entry.mode === "manifold"}
        <label class="onto">
          <span class="head">Onto</span>
          <Slider value={entry.onto} min={0} max={1} step={0.05} ariaLabel="Onto, for {title}"
            oninput={(value) => { if (Number.isFinite(value)) patchSteerEntry(name, { onto: value }, ["manifold"]); }} />
          <span class="meta">{coefficientText(entry.onto, { fixed: true })}</span>
        </label>
      {/if}
    </div>
  {/if}

  {#if where && entry.mode === "unit"}
    <LayerStrip nLayers={where.nLayers} fitted={where.fitted} share={where.share} selected={place.selected}
      scope={null} note="its own layer" />
  {:else if where}
    <LayerStrip nLayers={where.nLayers} fitted={where.fitted} share={where.share} selected={place.selected}
      {scope} onchange={(next) => setSteerLayers(name, next)} />
  {:else if place.status === "loading"}
    <div class="waiting" aria-hidden="true"></div>
  {/if}
  {#if place.error}<p class="error">{place.error}</p>{/if}

  <div class="expr">
    <p class="text" class:is-empty={noLayers} aria-live="polite">
      {#if noLayers}No layers selected{:else}{#each parts.head as piece, i}{#if i}<wbr>{/if}{piece}{/each}{#if parts.clause.length}<wbr><span class="clause">{#each parts.clause as piece, i}{#if i}<wbr>{/if}{piece}{/each}</span>{/if}{#each parts.tail as piece}<wbr>{piece}{/each}{/if}
    </p>
    <button type="button" class="dw-text" disabled={noLayers} onclick={() => void copy()}><MorphText text={copied ? "Copied" : "Copy"} numbers={false} /></button>
  </div>

  <div class="foot">
    <span class="dw-row">
      <button type="button" class="dw-text" onclick={onremove}>Remove</button>
      {#if entry.mode === "unit" && unitsAvailable()}
        <button type="button" class="dw-text" onclick={() => select({ kind: "unit", id: name })}>Open {title}</button>
      {/if}
    </span>
    <button type="button" class="dw-button" disabled={noLayers} onclick={onclose}>Done</button>
  </div>
</section>

<style>
  /* The open term: a wash surface on paper. Fields on it fill with wash2, and step up to paper. */
  .term {
    --field: var(--wash2);
    --field-hover: var(--paper);
    display: flex;
    flex-direction: column;
    gap: var(--s5);
    min-width: 0;
    padding: var(--s6);
    border-radius: var(--r-row);
    background: var(--wash);
  }
  .top { display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between; gap: var(--s3) var(--s5); }
  .name { display: grid; gap: var(--s1); min-width: 0; }
  .toggles { gap: var(--s4); }
  .toggles :global(.seg) { background: var(--wash2); }
  .controls { display: flex; flex-wrap: wrap; align-items: flex-start; gap: var(--s5) var(--s7); }
  .col { display: grid; gap: var(--s2); align-content: start; justify-items: start; min-width: 0; }
  .col :global(.seg) { background: var(--wash2); max-width: 100%; overflow-x: auto; scrollbar-width: none; }
  .strength { flex: 1 1 280px; justify-items: stretch; }

  /* Strength: a big tabular readout that scrubs, and a signed range filled from zero to the thumb. */
  .dial { display: flex; align-items: center; gap: var(--s4); min-width: 0; }
  .readout {
    flex: none;
    width: 124px;
    color: var(--ink);
    font: 600 40px/48px var(--display);
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
    cursor: ew-resize;
    touch-action: none;
    user-select: none;
    -webkit-user-select: none;
  }
  .range { flex: 1 1 0; min-width: 0; display: grid; gap: 2px; }
  .range input { -webkit-appearance: none; appearance: none; display: block; width: 100%; height: 24px; margin: 0; padding: 0; border: 0; background: transparent; cursor: pointer; }
  .range input::-webkit-slider-runnable-track { height: 4px; border-radius: 2px; background: var(--fill); }
  .range input::-moz-range-track { height: 4px; border-radius: 2px; background: var(--fill); }
  .range input::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 16px; height: 16px; margin-top: -6px; border: 0; border-radius: 50%; background: var(--ink); transition: scale var(--fast) var(--ease); }
  .range input::-moz-range-thumb { width: 16px; height: 16px; border: 0; border-radius: 50%; background: var(--ink); }
  .range input:active::-webkit-slider-thumb { scale: 1.12; }
  .ticks { display: grid; grid-template-columns: 1fr auto 1fr; padding: 0 2px; }
  .ticks > span:last-child { text-align: right; }

  .position { display: grid; gap: var(--s3); min-width: 0; }
  .position .dw-sec-head { align-items: center; }
  .pick { display: inline-flex; min-width: 0; max-width: 60%; }
  .onto { display: grid; grid-template-columns: auto minmax(0, 1fr) 40px; align-items: center; gap: var(--s3); }
  .onto .meta { text-align: right; }

  .waiting { height: 156px; border-radius: var(--r-box); background: var(--wash2); }
  .error { margin: 0; color: var(--ink2); font: var(--type-small); overflow-wrap: anywhere; }

  /* The term as the expression writes it, live, on paper; the clause in the term's colour. */
  .expr { display: flex; align-items: center; gap: var(--s4); min-width: 0; padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--paper); }
  .text { flex: 1 1 0; min-width: 0; margin: 0; color: var(--ink); font: var(--type-body); font-variant-numeric: tabular-nums; overflow-wrap: break-word; }
  .text.is-empty { color: var(--ink3); }
  .clause { padding: 0 2px; border-radius: 4px; background: color-mix(in oklch, var(--tint) 18%, transparent); transition: background-color var(--fast) var(--ease); }
  .foot { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); }

  @media (max-width: 720px) {
    .term { padding: var(--s4); gap: var(--s4); }
    .readout { width: 96px; font-size: 32px; line-height: 40px; }
    .controls { gap: var(--s4); }
    .pick { max-width: 100%; }
    /* 44 px to a finger: each choice reaches its pill's edge (inside the When pill's scroll box, which
       clips at that edge), the switch and the range are that tall. */
    .term :global(.seg > button) { position: relative; height: 36px; }
    .term :global(.seg > button::after) { content: ""; position: absolute; inset: -4px 0; }
    .toggles :global(.switch) { height: 44px; }
    .range input { height: 44px; }
    .foot .dw-text, .expr .dw-text { position: relative; }
    .foot .dw-text::after, .expr .dw-text::after { content: ""; position: absolute; inset: -14px -8px; }
  }
</style>
