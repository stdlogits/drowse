<script lang="ts">
  // One feature of the resident SAE as a unit row: its ID, its label, a meter and the value. The
  // meter reads strength, ``activation / maxActApprox`` (the 0..1 unit every SAE surface and the
  // ``@when:sae/<id>`` gate read); a feature without Neuronpedia metadata reads its raw activation
  // on the panel's shared scale, so meters and numbers rank alike. Pinned (a persistent ``sae/<id>``
  // probe), the row can be chosen: the reply is tinted by it and the Inspector shows its trace.
  // Unpinned (a feature the live readout found), its marker pins it. Push (``onpush``) adds a push
  // along the feature to the recipe's strip.
  import { Meter, MorphText } from "@stdlogits/ledger";
  import { readingText } from "../../lib/charts/depth";
  import { detachProbe, highlightState, openDrawer, setHighlightTarget } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import InstrumentRow from "./InstrumentRow.svelte";
  import ProbePinButton from "./ProbePinButton.svelte";
  import SteerButton from "./SteerButton.svelte";
  import { probeTitle } from "./probeRows";

  interface Props {
    /** Feature index into the resident SAE's dictionary. */
    id: number;
    probeName?: string;
    measured?: boolean;
    /** The feature's label (Neuronpedia's, when it has one). */
    label?: string | null;
    /** The resident SAE's hook layer. */
    layer: number | null;
    /** Latest reading: raw activation for a discovery row; a pinned probe's channel, already
     *  strength when the probe has ``max_act`` (``valueIsStrength``). */
    value: number;
    /** Neuronpedia ``maxActApprox``, the strength unit; null without metadata. */
    maxAct?: number | null;
    valueIsStrength?: boolean;
    /** The panel's shared raw scale for rows without metadata. */
    fallbackScale?: number;
    pinned: boolean;
    /** Disables the marker while the panel validates another feature. */
    busy?: boolean;
    /** Unpinned rows only: the panel owns validation and attach. */
    onpin?: (id: number) => void;
    /** Push along the feature: its chip lands in the recipe's strip (lib/steerActions.ts). */
    onpush?: () => void;
  }

  let {
    id, probeName, measured = true, label = null, layer, value, maxAct = null,
    valueIsStrength = false, fallbackScale = 1, pinned, busy = false, onpin, onpush,
  }: Props = $props();

  const name = $derived(probeName ?? `sae/${id}`);
  const title = $derived(probeTitle({ family: "sae", feature_id: id }, name, layer).text);
  const isHighlight = $derived(highlightState.target === name);
  /** Normalized 0..1 strength when the unit is known; null keeps raw. */
  const strength = $derived(
    valueIsStrength ? value : maxAct != null && maxAct > 0 ? value / maxAct : null,
  );
  /** The raw activation (reconstructed for a normalized pinned row). */
  const rawValue = $derived(valueIsStrength && maxAct != null ? value * maxAct : value);
  const share = $derived(strength ?? rawValue / Math.max(fallbackScale, 1));

  let unpinBusy = $state(false);

  async function onUnpin(): Promise<void> {
    if (unpinBusy) return;
    unpinBusy = true;
    // Unpinning destroys this keyed row: capture its identity before the request so the
    // completion path never reads a derived owned by the destroyed component.
    const featureName = name;
    try {
      await detachProbe(featureName);
      pushToast(`unpinned ${featureName}`, { kind: "info" });
    } catch (e) {
      pushToast(userFacingError(e, `Unable to unpin ${featureName}. Try again.`), { kind: "error", ttlMs: null });
    } finally {
      unpinBusy = false;
    }
  }

  function choose(): void {
    setHighlightTarget(name);
    openDrawer("probe_inspector", { name });
  }
</script>

<InstrumentRow pressed={pinned && isHighlight} onselect={pinned ? choose : undefined} title={name}>
  {#snippet lead()}
    <span class="n is-id">{title}</span>
  {/snippet}
  {#snippet sub()}{label ?? ""}{/snippet}
  {#snippet mark()}
    {#if measured}
      <Meter value={Math.max(0, share)} width={56}
        label={strength !== null ? `Strength ${strength.toFixed(2)}` : `Activation ${rawValue.toFixed(2)}`} />
    {/if}
  {/snippet}
  {#snippet value()}
    {#if !measured}<span aria-hidden="true">–</span><span class="sr-only">Not measured</span>
    {:else}<MorphText text={readingText(strength ?? rawValue)} />{/if}
  {/snippet}
  {#snippet action()}
    {#if onpush}<SteerButton label="Push" aria={`Push along feature ${id}`} onclick={onpush} />{/if}
    <ProbePinButton
      shape="triangle"
      {pinned}
      disabled={pinned ? unpinBusy : busy}
      onclick={() => pinned ? void onUnpin() : onpin?.(id)}
      title={pinned ? "unpin" : "pin"}
      ariaLabel={`${pinned ? "Unpin" : "Pin"} probe ${name}`}
    />
  {/snippet}
</InstrumentRow>
