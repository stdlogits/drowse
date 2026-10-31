<script lang="ts">
  // A unit probe (``neuron/L31n4410``, ``head/L28h12``) as one row of the Instruments: its ID, its
  // catalog label when the model has one, its reading token by token over the latest reply (bars in
  // its layer's colour) and the latest value, in units of its typical high firing (1 is a firing
  // as strong as its neutral top; below zero, a neuron's − pole). Choosing the row tints the reply
  // by it and shows the unit in the Inspector, whose Probing button detaches it (as a geometry
  // probe's row leads to its inspector), so the row lines up with every other probe's.
  import { MorphText } from "@stdlogits/ledger";
  import type { ProbeRackEntry, ScalarReadingJSON, UnitsProbeInfo } from "../../lib/types";
  import Bar from "../../lib/charts/Bar.svelte";
  import Trace from "../../lib/charts/Trace.svelte";
  import { depthTint, readingText } from "../../lib/charts/depth";
  import { highlightState, setHighlightTarget } from "../../lib/stores.svelte";
  import { select } from "../../lib/stores/selection.svelte";
  import { unitLabelFor } from "../../lib/stores/units.svelte";
  import { parseProbeTarget } from "../../lib/tokens";
  import { unitTitle } from "../../lib/units";
  import InstrumentRow from "./InstrumentRow.svelte";
  import { replyTrace } from "./probeTrace";

  let { name, entry }: { name: string; entry: ProbeRackEntry } = $props();

  const info = $derived(entry.info as UnitsProbeInfo);
  const latest = $derived((entry.aggregate ?? entry.reading) as ScalarReadingJSON | null);
  const sparkline = $derived(entry.sparkline ?? []);
  const trace = $derived(replyTrace(name) ?? sparkline);
  const reading = $derived(latest !== null || sparkline.length > 0 ? entry.current : entry.savedAggregate ?? null);
  const tint = $derived(depthTint(latest?.depth?.center?.[0]));
  const label = $derived(unitLabelFor(info.unit));
  const isHighlight = $derived(highlightState.target !== null && parseProbeTarget(highlightState.target).base === name);

  function choose(): void {
    setHighlightTarget(name);
    select({ kind: "unit", id: info.unit });
  }
</script>

<InstrumentRow pressed={isHighlight} onselect={choose} {tint} title={info.unit}>
  {#snippet lead()}
    <span class="n is-id">{unitTitle(info.unit)}</span>{#if label}<span class="meta label">{label}</span>{/if}
  {/snippet}
  {#snippet mark()}
    {#if trace.length > 1}
      <Trace values={trace} limit={20} scale={1} label={`${unitTitle(info.unit)} by token`} />
    {:else if reading !== null}
      <Bar value={reading} max={1} width={140} height={4} bipolar={info.kind === "neuron"} title={`${unitTitle(info.unit)} ${readingText(reading, { signed: true })}`} />
    {/if}
  {/snippet}
  {#snippet value()}
    {#if reading !== null}<MorphText text={readingText(reading, { signed: info.kind === "neuron" })} />{:else}–{/if}
  {/snippet}
</InstrumentRow>

<style>
  .label { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
</style>
