<script lang="ts">
  // The geometry probes, as rows (the Instruments canvas): flat ones (a concept axis, a flat fan
  // like personas) and curved ones (a fitted surface like emotions), sorted by name, value or
  // change, then the unit probes (a neuron or a head; UnitProbeCard). Beneath them: + Probe (a
  // concept, a manifold or a unit, found in the Units room), the sort, and Live, the one switch for
  // per-token scoring (off, readings land when the reply ends). With ``family`` it shows one
  // geometry kind. Lens and SAE probes have their own rows in their own sections.
  import { Switch, expand } from "@stdlogits/ledger";
  import { MorphText } from "@stdlogits/ledger";
  import ProbeCard from "./rack/ProbeCard.svelte";
  import UnitProbeCard from "./rack/UnitProbeCard.svelte";
  import { goto } from "../lib/stores/rooms.svelte";
  import { unitsAvailable } from "../lib/stores/units.svelte";
  import type { ProbeInfo, ProbeSortMode } from "../lib/types";
  import {
    activeProbeNames,
    openDrawer,
    probeRack,
    probeEntryForDisplay,
    probesLiveState,
    setLiveProbes,
    setProbeSortMode,
    sessionState,
  } from "../lib/stores.svelte";

  let { family }: { family?: "subspace" | "manifold" } = $props();

  const liveOn = $derived(probesLiveState.enabled);
  const outputNoun = $derived(sessionState.info?.is_base_model ? "completion" : "reply");

  /** Flat (affine) geometry probes are the subspace family, curved ones the manifold family. */
  function kindOf(info: ProbeInfo | undefined): "subspace" | "manifold" | null {
    if (info?.family !== "geometry") return null;
    return info.is_affine ? "subspace" : "manifold";
  }

  // activeProbeNames() reads probeRack.active, entries and sortMode, so this re-runs on each.
  const probes = $derived.by(() =>
    activeProbeNames().filter((n) => {
      const kind = kindOf(probeRack.entries.get(n)?.info);
      return kind !== null && (family === undefined || kind === family);
    }),
  );

  /** Unit probes, shown with the full rack only. */
  const units = $derived(family === undefined
    ? activeProbeNames().filter((n) => probeRack.entries.get(n)?.info.family === "units")
    : []);

  const SORTS: ProbeSortMode[] = ["name", "value", "change"];
  function cycleSort(): void {
    setProbeSortMode(SORTS[(SORTS.indexOf(probeRack.sortMode) + 1) % SORTS.length]);
  }

  let adding = $state(false);
  function add(kind: "subspace" | "manifolds" | "units"): void {
    adding = false;
    if (kind === "units") goto("units");
    else openDrawer(kind);
  }
  function onAdd(): void {
    if (family === "subspace") add("subspace");
    else if (family === "manifold") add("manifolds");
    else adding = !adding;
  }
</script>

<div class="rack" aria-label="Probes">
  {#if probes.length > 0 || units.length > 0}
    <div class="rows" role="list">
      {#each probes as name (name)}
        {@const entry = probeEntryForDisplay(name)}
        {#if entry}
          <div role="listitem"><ProbeCard {name} {entry} /></div>
        {/if}
      {/each}
      {#each units as name (name)}
        {@const entry = probeEntryForDisplay(name)}
        {#if entry}
          <div role="listitem"><UnitProbeCard {name} {entry} /></div>
        {/if}
      {/each}
    </div>
  {/if}

  <div class="foot">
    <button type="button" class="dw-add" aria-expanded={family === undefined ? adding : undefined} onclick={onAdd}>
      + {family === "manifold" ? "Manifold probe" : family === "subspace" ? "Concept probe" : "Probe"}
    </button>
    <span class="end">
      {#if probes.length + units.length > 1}
        <button type="button" class="dw-text sort" onclick={cycleSort} aria-label={`Sorted by ${probeRack.sortMode}. Sort by ${SORTS[(SORTS.indexOf(probeRack.sortMode) + 1) % SORTS.length]}`}>
          by <MorphText text={probeRack.sortMode} />
        </button>
      {/if}
      <span class="live" inert={probesLiveState.busy}>
        <Switch checked={liveOn} onchange={() => void setLiveProbes(!liveOn)} label="Live"
          hint={liveOn ? `Reading every word as it is written. Off, readings land when the ${outputNoun} ends.` : `Read every word as it is written`} />
      </span>
    </span>
  </div>

  {#if adding}
    <div class="dw-row" transition:expand>
      <button type="button" class="dw-quiet" onclick={() => add("subspace")}>A concept</button>
      <button type="button" class="dw-quiet" onclick={() => add("manifolds")}>A manifold</button>
      {#if unitsAvailable()}<button type="button" class="dw-quiet" onclick={() => add("units")}>A unit</button>{/if}
    </div>
  {/if}
</div>

<style>
  .rack { display: grid; gap: var(--s2); min-width: 0; }
  .rows { display: grid; gap: 2px; }
  .foot { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s4); }
  .end { display: flex; align-items: center; gap: var(--s4); margin-left: auto; }
  .sort { font: var(--type-meta); }
  .live[inert] { opacity: 0.4; }

  /* Phone sizes: every target 44 px (the sort, the switch). */
  @media (max-width: 720px), (pointer: coarse) {
    .sort { min-height: 44px; padding: 0 var(--s1); }
    .live :global(.switch) { height: 44px; }
  }
</style>
