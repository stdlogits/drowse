<script lang="ts">
  // A geometry probe as one row of the Instruments: its name and the layer it peaks at, its
  // reading token by token (bars on its depth's place on the ramp), and the latest value. A flat
  // probe reads its signed axis-0 coordinate, a curved one the share of the activation on its
  // surface. Choosing the row tints the reply by it and shows the probe in the Inspector, where its
  // trace, layers, coordinates and geometry live (drawers/ProbeInspectorDrawer.svelte).
  import type { GeometryProbeInfo, ProbeReadingJSON, ProbeRackEntry } from "../../lib/types";
  import Bar from "../../lib/charts/Bar.svelte";
  import Trace from "../../lib/charts/Trace.svelte";
  import { depthTint, layerDepth, peakLayer, readingText } from "../../lib/charts/depth";
  import { nodeCoordExtent, parseProbeTarget } from "../../lib/tokens";
  import { highlightState, openDrawer, setHighlightTarget } from "../../lib/stores.svelte";
  import RollingNumber from "../../lib/ui/RollingNumber.svelte";
  import InstrumentRow from "./InstrumentRow.svelte";

  let { name, entry }: { name: string; entry: ProbeRackEntry } = $props();

  const info = $derived(entry.info as GeometryProbeInfo);
  const affine = $derived(info.is_affine);
  /** Where a full bar sits: the most extreme node on axis 0 for a flat probe, 1 for a share. */
  const scale = $derived(affine ? nodeCoordExtent(info.node_coords, 0) : 1);
  const displayName = $derived(name.split("/").pop() ?? name);

  const latest = $derived((entry.aggregate ?? entry.reading) as ProbeReadingJSON | null);
  const sparkline = $derived(entry.sparkline ?? []);
  const reading = $derived.by((): number | null => {
    if (latest !== null || sparkline.length > 0) return entry.current;
    if (affine) return entry.savedCoordinates?.[0] ?? entry.savedAggregate ?? null;
    return entry.savedFraction ?? entry.savedAggregate ?? null;
  });

  const peak = $derived(peakLayer(entry.perLayer));
  const deepest = $derived(Math.max(0, ...(info.layers ?? []), ...Object.keys(entry.perLayer ?? {}).map(Number)));
  const tint = $derived(depthTint(latest?.depth_com?.[0] ?? (peak === null ? null : layerDepth(peak, deepest))));

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
    <span class="n">{displayName}</span>{#if peak !== null}<span class="meta">L{peak}</span>{/if}
  {/snippet}
  {#snippet mark()}
    {#if sparkline.length > 1}
      <Trace values={sparkline} limit={32} {scale} label={`${displayName} by token`} />
    {:else if reading !== null}
      <Bar value={reading} max={scale} width={140} height={4} bipolar={affine} title={`${displayName} ${readingText(reading, { signed: affine })}`} />
    {/if}
  {/snippet}
  {#snippet value()}
    {#if reading !== null}<RollingNumber value={reading} digits={2} signed={affine} />{:else}–{/if}
  {/snippet}
</InstrumentRow>
