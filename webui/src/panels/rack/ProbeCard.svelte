<script lang="ts">
  // A geometry probe as one row of the Instruments: its name and the layer it reads at, its
  // reading token by token (bars on its depth's place on the ramp), and the latest value. A flat
  // probe reads its signed axis-0 coordinate, a curved one the share of the activation on its
  // surface (0 to 1), and the trace, the bar's scale and the value all read that one quantity.
  // Choosing the row tints the reply by it and shows the probe in the Inspector, where its trace,
  // layers, coordinates and geometry live (drawers/ProbeInspectorDrawer.svelte).
  import type { GeometryProbeInfo, ProbeReadingJSON, ProbeRackEntry } from "../../lib/types";
  import Bar from "../../lib/charts/Bar.svelte";
  import Trace from "../../lib/charts/Trace.svelte";
  import { depthLayer, depthTint, readingText } from "../../lib/charts/depth";
  import { sessionState } from "../../lib/stores/session.svelte";
  import { MorphText } from "@stdlogits/ledger";
  import { nodeCoordExtent, parseProbeTarget } from "../../lib/tokens";
  import { highlightState, openDrawer, setHighlightTarget } from "../../lib/stores.svelte";
  import InstrumentRow from "./InstrumentRow.svelte";
  import { probeTitle } from "./probeRows";
  import { replyTrace } from "./probeTrace";

  let { name, entry }: { name: string; entry: ProbeRackEntry } = $props();

  const info = $derived(entry.info as GeometryProbeInfo);
  const affine = $derived(info.is_affine);
  /** Where a full bar sits: the most extreme node on axis 0 for a flat probe, 1 for a share. */
  const scale = $derived(affine ? nodeCoordExtent(info.node_coords, 0) : 1);
  const displayName = $derived(probeTitle(info, name).text);

  const latest = $derived((entry.aggregate ?? entry.reading) as ProbeReadingJSON | null);
  /** The live reading, else the saved reply's (its last measured row, after a reload). */
  const settled = $derived((latest ?? entry.savedReading ?? null) as ProbeReadingJSON | null);
  const sparkline = $derived(entry.sparkline ?? []);
  /** The latest reply token by token (a curved probe's share, from each token's reading), else the
   *  live buffer (the same primary reading). */
  const trace = $derived(replyTrace(name, { share: !affine }) ?? sparkline);
  const reading = $derived.by((): number | null => {
    if (latest !== null || sparkline.length > 0) return entry.current;
    if (affine) return entry.savedCoordinates?.[0] ?? entry.savedAggregate ?? null;
    return entry.savedFraction ?? entry.savedAggregate ?? null;
  });

  /** The layer its reading's depth centre of mass falls on, over the model's layers (a probe DLS
   *  fitted on a subset has one too). */
  const atLayer = $derived(depthLayer(settled?.depth_com?.[0], sessionState.info?.architecture?.layers, info.layers ?? []));
  /** The probe's place on the ramp: its measured depth (layer / (layers - 1)), when it has one. */
  const tint = $derived(depthTint(settled?.depth_com?.[0]));

  const isHighlight = $derived(
    highlightState.target !== null && parseProbeTarget(highlightState.target).base === name,
  );

  function choose(): void {
    setHighlightTarget(name);
    openDrawer("probe_inspector", { name });
  }
</script>

<InstrumentRow pressed={isHighlight} onselect={choose} {tint} title={name}>
  {#snippet lead()}
    <span class="n">{displayName}</span>{#if atLayer !== null}<span class="meta">L{atLayer}</span>{/if}
  {/snippet}
  {#snippet mark()}
    {#if trace.length > 1}
      <Trace values={trace} limit={20} {scale} label={`${displayName} by token`} />
    {:else if reading !== null}
      <Bar value={reading} max={scale} width={140} height={4} bipolar={affine} title={`${displayName} ${readingText(reading, { signed: affine })}`} />
    {/if}
  {/snippet}
  {#snippet value()}
    {#if reading !== null}<MorphText text={readingText(reading, { signed: affine })} />{:else}–{/if}
  {/snippet}
</InstrumentRow>
