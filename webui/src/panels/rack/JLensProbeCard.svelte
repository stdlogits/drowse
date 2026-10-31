<script lang="ts">
  // A word the lens reads, as one row: the word, its probability at each fitted layer (a bar per
  // layer in that layer's colour; an empty slot is a layer where it fell below the top-k), and its
  // strength, the mean fitted-layer probability (the one channel a ``@when:jlens/<word>`` gate
  // reads). Pinned (a persistent ``jlens/<word>`` probe), the row can be chosen: the reply is
  // tinted by it and the Inspector shows its trace. Unpinned (a word the live readout found), its
  // marker pins it.
  import { tokenParts } from "@stdlogits/ledger";
  import { chartValue } from "../../lib/charts/chartValues";
  import { depthTint, layerDepth } from "../../lib/charts/depth";
  import Trace from "../../lib/charts/Trace.svelte";
  import { detachProbe, highlightState, openDrawer, setHighlightTarget } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import InstrumentRow from "./InstrumentRow.svelte";
  import ProbePinButton from "./ProbePinButton.svelte";

  interface Props {
    /** Raw vocabulary token text (untrimmed on a discovery row; a pinned probe's word is bare). */
    token: string;
    /** Mean fitted-layer probability, 0..1; null when not measured. */
    strength: number | null;
    probeName?: string;
    /** Probability-mass-weighted depth centre of mass, 0..1. */
    com: number | null;
    /** Depth spread around the centre of mass. */
    spread: number | null;
    /** Per-layer probabilities in layer order; ``p === null`` = below that layer's top-k. */
    cells: { layer: number; p: number | null }[];
    pinned: boolean;
    /** Pin in flight: disables the action. */
    busy?: boolean;
    /** Unpinned rows only: the panel owns validation and attach. */
    onpin?: (word: string) => void;
  }

  let { token, strength, com, spread, cells, pinned, probeName, busy = false, onpin }: Props = $props();

  const parts = $derived(tokenParts(token));
  const wordStart = $derived(parts.lead);
  const display = $derived(parts.empty ? JSON.stringify(token) : parts.text);
  const pinnable = $derived(token.length > 0 && token === token.trim());
  const name = $derived(probeName ?? `jlens/${display}`);
  const isHighlight = $derived(highlightState.target === name);

  /** Bars read against the row's own largest layer: absolute p spans orders of magnitude. */
  const cellScale = $derived(Math.max(...cells.map((c) => c.p ?? 0), 1e-12));
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
    <span class="n">{#if wordStart}<span class="ws">·</span>{/if}{display}</span>
  {/snippet}
  {#snippet mark()}
    {#if cells.length > 0}
      <Trace values={cells.map((c) => c.p)} limit={0} scale={cellScale} {colors} label={`${display} by layer`} />
    {/if}
  {/snippet}
  {#snippet value()}
    {#if strength === null}<span aria-hidden="true">–</span><span class="sr-only">Not measured</span>{:else}{chartValue(strength, true)}{/if}
  {/snippet}
  {#snippet action()}
    <ProbePinButton
      shape="square"
      {pinned}
      disabled={pinned ? unpinBusy : busy || !pinnable}
      onclick={() => pinned ? void onUnpin() : onpin?.(token)}
      title={pinned ? "unpin" : pinnable ? "pin" : "This token includes whitespace. Watch the bare word to follow it."}
      ariaLabel={`${pinned ? "Unpin" : "Pin"} probe ${name}`}
    />
  {/snippet}
</InstrumentRow>
