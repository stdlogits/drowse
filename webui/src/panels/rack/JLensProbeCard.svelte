<script lang="ts">
  // A word the lens reads, as one row: the word, its reading at each fitted layer (a bar per layer
  // in that layer's colour; an empty slot is a layer where it fell below the top-k), and its
  // strength (the one channel a ``@when:`` gate on it reads). A ``jlens/`` row reads the word as
  // said: its probability, the mean over the fitted layers. An ``slens/`` row reads it as read: its
  // footprint z, signed, weighted toward the early layers where the S lens is faithful. Pinned (a
  // persistent probe), the row can be chosen: the reply is tinted by it and the Inspector shows its
  // trace. Unpinned (a word the live readout found), its marker pins it. Push (``onpush``) adds a
  // push toward the word to the recipe's strip.
  import { tokenParts } from "@stdlogits/ledger";
  import { depthTint, layerDepth, readingText } from "../../lib/charts/depth";
  import Trace from "../../lib/charts/Trace.svelte";
  import { detachProbe, highlightState, openDrawer, setHighlightTarget } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import InstrumentRow from "./InstrumentRow.svelte";
  import ProbePinButton from "./ProbePinButton.svelte";
  import SteerButton from "./SteerButton.svelte";
  import type { LensNamespace } from "../../lib/stores/instruments.svelte";

  interface Props {
    /** Raw vocabulary token text (untrimmed on a discovery row; a pinned probe's word is bare). */
    token: string;
    /** ``jlens``: the mean fitted-layer probability, 0..1; ``slens``: the mean footprint z. Null
     *  when not measured. */
    strength: number | null;
    /** Which lens reads the word: as said (``jlens``) or as read (``slens``). */
    namespace?: LensNamespace;
    /** The pair's word for the row (``said`` / ``read``), shown while both lenses are offered. */
    pairWord?: string | null;
    probeName?: string;
    /** Probability-mass-weighted depth centre of mass, 0..1. */
    com: number | null;
    /** Depth spread around the centre of mass. */
    spread: number | null;
    /** Per-layer readings in layer order; ``p === null`` = below that layer's top-k. */
    cells: { layer: number; p: number | null }[];
    pinned: boolean;
    /** Pin in flight: disables the action. */
    busy?: boolean;
    /** Unpinned rows only: which lens a pin reads the word with (the pair's choice). */
    pinAs?: LensNamespace;
    /** Unpinned rows only: the panel owns validation and attach. */
    onpin?: (word: string) => void;
    /** Push toward the word: its chip lands in the recipe's strip (lib/steerActions.ts). */
    onpush?: () => void;
  }

  let { token, strength, namespace = "jlens", pinAs = "jlens", pairWord = null, com, spread, cells, pinned, probeName, busy = false, onpin, onpush }: Props = $props();

  const parts = $derived(tokenParts(token));
  const wordStart = $derived(parts.lead);
  const display = $derived(parts.empty ? JSON.stringify(token) : parts.text);
  const pinnable = $derived(token.length > 0 && token === token.trim());
  const name = $derived(probeName ?? `${namespace}/${display}`);
  const percent = $derived(namespace === "jlens");
  const isHighlight = $derived(highlightState.target === name);

  /** Bars read against the row's own largest layer: absolute p spans orders of magnitude (and a z
   *  runs either way). */
  const cellScale = $derived(Math.max(...cells.map((c) => Math.abs(c.p ?? 0)), 1e-12));
  const deepest = $derived(Math.max(0, ...cells.map((c) => c.layer)));
  const colors = $derived(cells.map((c) => depthTint(layerDepth(c.layer, deepest))));
  const depthText = $derived(com === null ? "" : ` · depth ${com.toFixed(2)}${spread === null ? "" : ` ± ${spread.toFixed(2)}`}`);

  let unpinBusy = $state(false);

  async function onUnpin(): Promise<void> {
    if (unpinBusy) return;
    unpinBusy = true;
    // Unpinning destroys this keyed row: capture its identity before the request so the
    // completion path never reads a derived owned by the destroyed component.
    const probe = name;
    try {
      await detachProbe(probe);
      pushToast(`unpinned ${probe}`, { kind: "info" });
    } catch (e) {
      pushToast(userFacingError(e, `Unable to unpin ${probe}. Try again.`), { kind: "error", ttlMs: null });
    } finally {
      unpinBusy = false;
    }
  }

  function choose(): void {
    setHighlightTarget(name);
    openDrawer("probe_inspector", { name });
  }
</script>

<InstrumentRow pressed={pinned && isHighlight} onselect={pinned ? choose : undefined} tint={depthTint(com)}
  title={`${pinned ? name : `"${token}", averaged across fitted layers`}${depthText}`}>
  {#snippet lead()}
    <span class="n">{#if wordStart}<span class="ws">·</span>{/if}{display}{#if pairWord}<span class="pair">{` ${pairWord}`}</span>{/if}</span>
  {/snippet}
  {#snippet mark()}
    {#if cells.length > 0}
      <Trace values={cells.map((c) => c.p)} limit={0} scale={cellScale} {colors} label={`${display} by layer`} />
    {/if}
  {/snippet}
  {#snippet value()}
    {#if strength === null}<span aria-hidden="true">–</span><span class="sr-only">Not measured</span>{:else}{readingText(strength, { percent })}{/if}
  {/snippet}
  {#snippet action()}
    {#if onpush}
      <SteerButton label="Push" aria={`Push toward ${display}, ${(pinned ? namespace : pinAs) === "slens" ? "as read" : "as said"}`}
        tip={(pinned ? namespace : pinAs) === "slens" ? "Toward having read it" : "Toward saying it"} onclick={onpush} />
    {/if}
    <ProbePinButton
      shape="square"
      {pinned}
      disabled={pinned ? unpinBusy : busy || !pinnable}
      onclick={() => pinned ? void onUnpin() : onpin?.(token)}
      title={pinned ? "unpin" : pinnable ? (pinAs === "slens" ? "pin as read" : "pin as said") : "This token includes whitespace. Watch the bare word to follow it."}
      ariaLabel={pinned ? `Unpin probe ${name}` : `Pin probe ${pinAs}/${display}`}
    />
  {/snippet}
</InstrumentRow>

<style>
  .pair { color: var(--ink3); font: var(--type-meta); white-space: pre; }
</style>
