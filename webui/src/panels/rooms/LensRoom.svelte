<script lang="ts">
  // The Lens room (#/lens): what the lenses read, at the token cursor (TokenStrip). At a token, each
  // layer through both lenses (LensLayers: the S lens's read, input, and the R lens's say, output,
  // each by its weight in the depth rule, the crossover band on a wash) and the lens probes' values
  // there, from one replay of the token (the token anatomy, as the workbench's token view replays
  // it, steered as the token was or, asked, without the steering). Whole, the lens probes over the
  // reply (its node's aggregate envelope; the live readout's words are per token and the node keeps
  // no reply-level words), or, with none, the way to watch a word. Fitting and fetching lenses live
  // in the Artifacts room; without a lens the room links there.
  import { onMount } from "svelte";
  import { Icon, MorphText, Segmented, Token } from "@stdlogits/ledger";
  import TokenStrip from "./TokenStrip.svelte";
  import LensLayers from "./lens/LensLayers.svelte";
  import { cachedTokenAnatomy } from "../../lib/runtime/tokenReadoutCache";
  import { replayCapabilities } from "../../lib/runtime/contracts";
  import { getRuntimeClient } from "../../lib/runtime/registry";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { genStatus, effectiveRawMode } from "../../lib/stores/chat.svelte";
  import { loomTree } from "../../lib/stores/loom.svelte";
  import { goto } from "../../lib/stores/rooms.svelte";
  import { showInstrument } from "../../lib/commands";
  import { lensSourceState, refreshLensSources } from "../../lib/stores/instruments.svelte";
  import { probeRack } from "../../lib/stores/probes.svelte";
  import { sessionState } from "../../lib/stores/session.svelte";
  import { settled, tokenCursor } from "../../lib/stores/tokenCursor.svelte";
  import { depthTint, readingText } from "../../lib/charts/depth";
  import { replayAnchor } from "../../drawers/token/readings";
  import { probeTitle } from "../rack/probeRows";
  import type { MeasurementsEnvelopeJSON, ScalarReadingJSON } from "../../lib/types";

  const anatomyAvailable = replayCapabilities(getRuntimeClient().instruments).tokenAnatomy;
  onMount(() => {
    if (lensSourceState.sources.length === 0) void refreshLensSources();
  });
  /** A lens to read with: the R lens fitted, or any active lens (the anatomy reads through either). */
  const lensReady = $derived(sessionState.info?.jlens_fitted === true || lensSourceState.sources.some((source) => source.active));

  const here = settled(() => tokenCursor.token);
  const cursorToken = $derived(tokenCursor.token);
  const whole = $derived(cursorToken === null);
  const token = $derived(here.current);
  const node = $derived(token?.nodeId ? (loomTree.nodes.get(token.nodeId) ?? null) : null);
  const steering = $derived(token ? replayAnchor(node, token.token, token.turn.appliedSteering).steering : null);
  let steered = $state(true);

  // ---------- one token: its anatomy
  let read = $state.raw<{ key: string; status: "loading" | "ready" | "error"; value: MeasurementsEnvelopeJSON | null; error: string | null }>({
    key: "", status: "loading", value: null, error: null,
  });
  let attempt = $state(0);
  const want = $derived.by(() => {
    if (!anatomyAvailable || !lensReady || !token || token.nodeId === null || token.index === null) return null;
    const asSteered = steered || steering === null;
    return { nodeId: token.nodeId, index: token.index, steered: asSteered, raw: effectiveRawMode(), key: `${token.nodeId}:${token.index}:${asSteered}:${effectiveRawMode()}:${attempt}` };
  });
  $effect(() => {
    const w = want;
    if (!w || genStatus.active || read.key === w.key) return;
    const key = w.key;
    read = { key, status: "loading", value: read.value, error: null };
    cachedTokenAnatomy(w.nodeId, w.index, { k: 8, units: 1, patterns: 0, steered: w.steered, raw: w.raw }).then(
      (value) => { if (read.key === key) read = { key, status: "ready", value: value.measurements, error: null }; },
      (error: unknown) => { if (read.key === key) read = { key, status: "error", value: read.value, error: userFacingError(error, "This token's lenses could not be replayed.") }; },
    );
  });
  const anatomy = $derived(read.value?.anatomy ?? null);
  const stale = $derived(read.status === "loading" || (anatomy !== null && token !== null && anatomy.token.raw_index !== token.index));
  const tokenPins = $derived(read.value?.instruments.lens?.readings ?? {});

  // ---------- the whole reply: its lens probes
  const reply = $derived(tokenCursor.reply);
  const aggregate = $derived(reply?.nodeId ? (loomTree.nodes.get(reply.nodeId)?.aggregate_measurements?.instruments.lens ?? null) : null);

  /** The lens probes a view lists: what it read, by the rack's names. */
  const pins = $derived(Object.entries(whole ? (aggregate?.readings ?? {}) : tokenPins) as [string, ScalarReadingJSON][]);
  const pinMax = $derived(Math.max(1e-9, ...pins.map(([, r]) => Math.abs(r.value))));
  function pinTitle(name: string): { lead: boolean; text: string } {
    return probeTitle(probeRack.entries.get(name)?.info ?? null, name);
  }
  const percent = (reading: ScalarReadingJSON) => reading.unit.includes("probability");
</script>

<div class="dw-room lens">
  <TokenStrip />

  {#if !lensReady}
    <div class="dw-empty">
      <span class="head">No lens</span>
      <button type="button" class="dw-quiet" onclick={() => goto("artifacts", "lenses")}>Lenses<Icon name="right" /></button>
    </div>
  {:else if whole}
    {#if !reply}
      <p class="meta none">The lenses read here once the model has written a reply.</p>
    {:else if pins.length === 0}
      <div class="dw-empty">
        <span class="head">No lens probes</span>
        <button type="button" class="dw-quiet" onclick={() => showInstrument("lens")}>Watch a word<Icon name="right" /></button>
      </div>
    {/if}
  {:else if !anatomyAvailable}
    <div class="dw-empty"><span class="head">Lens</span><p>This runtime does not replay tokens.</p></div>
  {:else if token && (token.nodeId === null || token.index === null)}
    <div class="dw-empty"><span class="head">Lens</span><p>This token has no replay record.</p></div>
  {:else}
    <section class="dw-sec" aria-labelledby="lens-token-head">
      <div class="dw-sec-head">
        <span class="head" id="lens-token-head">{#if token}At <Token token={token.token.text} />{/if}</span>
        {#if steering !== null}
          <Segmented label="Readings" value={steered ? "steered" : "unsteered"} onchange={(v) => (steered = v === "steered")}
            options={[{ value: "steered", label: "Steered", hint: steering }, { value: "unsteered", label: "Unsteered", hint: "Replayed without the steering" }]} />
        {/if}
      </div>
      {#if read.status === "error" && read.key === want?.key}
        <div class="dw-row"><span class="meta">{read.error}</span><button type="button" class="dw-text" onclick={() => (attempt += 1)}>Try again</button></div>
      {/if}
      {#if anatomy}
        <div class="read" class:stale={stale || cursorToken?.ordinal !== token?.ordinal}>
          <LensLayers layers={anatomy.layers} tokenId={anatomy.token.token_id} />
        </div>
      {:else if read.status !== "error"}
        <div class="ghost" aria-hidden="true">{#each [0, 1, 2, 3, 4, 5] as k (k)}<i></i>{/each}</div>
      {/if}
    </section>
  {/if}

  {#if lensReady && pins.length > 0}
    <section class="dw-sec probes" aria-labelledby="lens-probes-head">
      <div class="dw-sec-head"><span class="head" id="lens-probes-head">Lens probes</span><span class="meta"><MorphText text={whole ? "over this reply" : "at this token"} /></span></div>
      <div class="rows">
        {#each pins as [name, reading] (name)}
          {@const title = pinTitle(name)}
          <div class="prow" style:--tint={depthTint(reading.depth?.center?.[0] ?? null) ?? undefined}>
            <span class="pname"><span class="ns">{name.startsWith("slens/") ? "reads" : "says"}</span><Token token={`${title.lead ? " " : ""}${title.text}`} /></span>
            <span class="meter"><i style:width="{Math.round((Math.abs(reading.value) / pinMax) * 100)}%"></i></span>
            <span class="pval">{readingText(reading.value, { percent: percent(reading) })}</span>
          </div>
        {/each}
      </div>
    </section>
  {/if}
</div>

<style>
  .lens { max-width: 960px; }
  .read { transition: opacity var(--fast) var(--ease); }
  .read.stale { opacity: 0.55; }
  .ghost { display: grid; gap: 2px; }
  .ghost > i { display: block; height: 36px; border-radius: var(--r-box); background: var(--wash); }
  .none { margin: 0; }

  .meter { position: relative; display: block; width: 56px; height: 4px; border-radius: 2px; background: var(--wash2); }
  .meter > i { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 2px; background: var(--tint, var(--deep)); transition: width var(--glide) var(--ease); }

  .rows { display: grid; gap: 2px; max-width: 640px; }
  .prow { display: grid; grid-template-columns: minmax(0, 1fr) 72px 56px; gap: var(--s3); align-items: center; min-height: 36px; padding: 0 var(--s3); border-radius: var(--r-box); }
  .prow:hover { background: var(--wash); }
  .pname { display: flex; align-items: center; gap: var(--s2); min-width: 0; }
  .ns { color: var(--ink3); font: var(--type-meta); }
  .pval { color: var(--ink2); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; }
</style>
