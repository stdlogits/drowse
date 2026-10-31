<script lang="ts">
  // The resident SAE's features (the Instruments canvas's Features section): a switch turns the
  // live readout on; the features it finds most active and the pinned ones (persistent ``sae/<id>``
  // probes, gate-able) are one roster of unit rows in one sort. One field takes a feature number to
  // watch (pin) or to steer with (an ``α sae/<id>`` term in the recipe, edited from its chip). While
  // a token is hovered, the rows read that token.
  import { flip } from "svelte/animate";
  import { GLIDE, MorphText, Switch, ease, reducedMotion } from "@stdlogits/ledger";
  import RuntimeSaeSourceSection from "@runtime-sae-source";
  import SaeProbeCard from "./rack/SaeProbeCard.svelte";
  import { mergeInstrumentProbeRows } from "./rack/probeRows";
  import type { InstrumentProbeRow } from "./rack/probeRows";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import { apiInstruments } from "../lib/runtime/services";
  import {
    activeProbeNames,
    addSaeToRack,
    attachProbe,
    probeRack,
    probeEntryForDisplay,
    saeState,
    saeRawFallbackScale,
    saeReadoutForDisplay,
    setLiveSae,
    setSaeSortMode,
    tokenHoverState,
    instrumentFamily,
    saeLoaded,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { SaeProbeInfo, ScalarReadingJSON } from "../lib/types";
  import type { SaeSortMode } from "../lib/stores.svelte";

  /** ``sources={false}`` leaves the source picker to the Lens room (the Inspector's Features). */
  let { sources = true }: { sources?: boolean } = $props();
  const loaded = $derived(saeLoaded());
  const displayReadout = $derived(saeReadoutForDisplay());
  /** The resident SAE's hook layer, off the family block. */
  const residentLayer = $derived.by((): number | null => {
    const live = instrumentFamily("sae")?.live;
    return live && "layer" in live ? live.layer : null;
  });

  // ---------- the roster: pinned probes and the live discovery top-k ----------
  const pinnedBase = $derived.by(() => activeProbeNames()
    .map((name) => ({ name, entry: probeEntryForDisplay(name) }))
    .filter((row) => row.entry?.info.family === "sae"));

  const discoveryBase = $derived.by(() => displayReadout
    .filter((row) => !pinnedBase.some(({ entry }) =>
      entry?.info.family === "sae" && entry.info.feature_id === row.id))
    .map((row) => {
      // Metadata merges the row's server-cached values and the between-generation backfill.
      const meta = tokenHoverState.active ? undefined : saeState.meta.get(row.id);
      return { ...row, label: row.label ?? meta?.label ?? null, max_act: row.max_act ?? meta?.max_act ?? null };
    }));

  /** The shared raw scale for rows without a maxActApprox unit, so they rank alike. */
  const fallbackScale = $derived(saeRawFallbackScale());

  function visibleStrength(value: number, maxAct: number | null): number {
    return maxAct != null && maxAct > 0 ? value / maxAct : value / fallbackScale;
  }

  type Row =
    | ({ kind: "pinned"; name: string; entry: NonNullable<(typeof pinnedBase)[number]["entry"]> } & InstrumentProbeRow)
    | ({ kind: "discovery"; feature: (typeof discoveryBase)[number] } & InstrumentProbeRow);

  const pinnedRows = $derived.by((): Row[] =>
    pinnedBase.map((row) => {
      const entry = row.entry!;
      const info = entry.info as SaeProbeInfo;
      const latest = (entry.aggregate ?? entry.reading) as ScalarReadingJSON | null;
      const value = latest?.value ?? entry.current;
      return {
        kind: "pinned",
        key: row.name,
        name: row.name,
        entry,
        sortName: info.label || (!tokenHoverState.active && saeState.meta.get(info.feature_id)?.label) || String(info.feature_id),
        // A pinned value with max_act is normalized server-side; the reading's unit says which.
        strength: latest?.unit === "activation_over_max" ? value : value / fallbackScale,
      };
    }));

  const discoveryRows = $derived.by((): Row[] =>
    discoveryBase.map((feature) => ({
      kind: "discovery",
      key: `sae/${feature.id}`,
      feature,
      sortName: feature.label || String(feature.id),
      strength: visibleStrength(feature.activation, feature.max_act ?? null),
    })));

  /** One roster, one sort: pinning changes persistence and actions, never position. */
  const rows = $derived(
    mergeInstrumentProbeRows(pinnedRows, saeState.live || tokenHoverState.active ? discoveryRows : [], saeState.sortMode, { numericNames: true }),
  );

  const SORTS: SaeSortMode[] = ["strength", "name"];
  const nextSort = $derived(SORTS[(SORTS.indexOf(saeState.sortMode) + 1) % SORTS.length]);

  // ---------- one field: watch a feature, or steer with it ----------
  let field = $state("");
  let busy = $state(false);

  async function validated(raw: string): Promise<number | null> {
    const id = Number(raw.trim().replace(/^sae\//, ""));
    if (!Number.isInteger(id) || id < 0) {
      pushToast("Enter a feature number of 0 or greater.", { kind: "error" });
      return null;
    }
    return (await apiInstruments.validateSaeFeature(id)).id;
  }

  async function pin(id: number): Promise<void> {
    if (busy || probeRack.active.includes(`sae/${id}`)) return;
    busy = true;
    try {
      await apiInstruments.validateSaeFeature(id);
      await attachProbe(`sae/${id}`);
    } catch (error) {
      pushToast(userFacingError(error, "Unable to pin that feature. Try again."), { kind: "error" });
    } finally {
      busy = false;
    }
  }

  async function watch(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    if (busy) return;
    busy = true;
    try {
      const id = await validated(field);
      if (id !== null) {
        await attachProbe(`sae/${id}`);
        field = "";
      }
    } catch (error) {
      pushToast(userFacingError(error, "Unable to add that feature reading. Try again."), { kind: "error" });
    } finally {
      busy = false;
    }
  }

  async function steer(): Promise<void> {
    if (busy) return;
    busy = true;
    try {
      const id = await validated(field);
      if (id !== null) {
        addSaeToRack(id);
        field = "";
      }
    } catch (error) {
      pushToast(userFacingError(error, "Unable to add that feature control. Try again."), { kind: "error" });
    } finally {
      busy = false;
    }
  }
</script>

<div class="sae" aria-label="Features">
  {#if sources}<RuntimeSaeSourceSection />{/if}

  {#if loaded}
    <div class="dw-row">
      <span class="live" inert={saeState.busy}>
        <Switch checked={saeState.live} onchange={() => void setLiveSae(!saeState.live)} label="Feature readout"
          hint="Read which features fire while the model writes. Pinned features stay when it finishes." />
      </span>
    </div>

    {#if tokenHoverState.active && (tokenHoverState.saeLoading || tokenHoverState.saeError || rows.length === 0)}
      <p class="meta" role={tokenHoverState.saeError ? "alert" : undefined}>
        {tokenHoverState.saeLoading ? "Reading the hovered token…" : tokenHoverState.saeError ?? "No feature fired on this token."}
      </p>
    {/if}

    {#if rows.length > 0}
      <div class="rows" role="list" aria-label="SAE feature probes">
        {#each rows as row (row.key)}
          <div role="listitem" animate:flip={{ duration: reducedMotion() ? 0 : GLIDE, easing: ease }}>
            {#if row.kind === "pinned"}
              {@const reading = (row.entry.aggregate ?? row.entry.reading) as ScalarReadingJSON | null}
              {@const probe = row.entry.info as SaeProbeInfo}
              <SaeProbeCard
                id={probe.feature_id}
                probeName={row.name}
                label={probe.label || (!tokenHoverState.active ? saeState.meta.get(probe.feature_id)?.label : null)}
                layer={residentLayer}
                value={reading?.value ?? row.entry.current ?? 0}
                measured={reading !== null || row.entry.sparkline.length > 0}
                maxAct={reading?.unit === "raw_activation" ? null : probe.max_act}
                valueIsStrength={reading?.unit === "activation_over_max"}
                {fallbackScale}
                pinned={true}
              />
            {:else}
              <SaeProbeCard
                id={row.feature.id}
                label={row.feature.label}
                layer={residentLayer}
                value={row.feature.activation}
                maxAct={row.feature.max_act}
                {fallbackScale}
                pinned={false}
                {busy}
                onpin={(id) => void pin(id)}
              />
            {/if}
          </div>
        {/each}
      </div>
    {/if}

    <form class="ask" onsubmit={watch}>
      <label class="sr-only" for="sae-feature">A feature number</label>
      <input id="sae-feature" type="text" inputmode="numeric" placeholder="A feature number" autocomplete="off" bind:value={field} />
      <button type="submit" class="dw-text" disabled={busy || !field.trim()}>Watch</button>
      <button type="button" class="dw-text" disabled={busy || !field.trim()} onclick={() => void steer()}>Steer</button>
      {#if rows.length > 1}
        <button type="button" class="dw-text sort" onclick={() => setSaeSortMode(nextSort)}
          aria-label={`Sorted by ${saeState.sortMode}. Sort by ${nextSort}`}>by <MorphText text={saeState.sortMode} /></button>
      {/if}
    </form>
  {/if}
</div>

<style>
  .sae { display: grid; gap: var(--s3); min-width: 0; }
  .live[inert] { opacity: 0.4; }
  .rows { display: grid; gap: 2px; }

  /* The field: a wash well, then quiet text actions. */
  .ask { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s4); }
  .ask input {
    flex: 1 1 160px;
    min-width: 0;
    height: 36px;
    padding: 0 var(--s4);
    border: 0;
    border-radius: var(--r-pill);
    background: var(--wash);
    color: var(--ink);
    font: var(--type-small);
    transition: background-color var(--fast) var(--ease);
  }
  .ask input::placeholder { color: var(--ink3); }
  .ask input:hover { background: var(--wash2); }
  .ask input:focus { outline: none; }
  .ask input:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
  /* On the phone's sheet (itself a wash) the well steps up to wash2, and hover steps one further. */
  :global(.sheet) .ask input { background: var(--wash2); }
  :global(.sheet) .ask input:hover { background: color-mix(in oklab, var(--wash2), var(--ink) 6%); }
  .sort { margin-left: auto; font: var(--type-meta); }

  /* Phone sizes: every target 44 px (the field, its text buttons, the switch). */
  @media (max-width: 720px), (pointer: coarse) {
    .ask .dw-text { min-height: 44px; padding: 0 var(--s1); }
    .live :global(.switch) { height: 44px; }
  }
  @media (max-width: 720px) {
    .ask input { height: 44px; font: var(--type-body); }
  }
</style>
