<script lang="ts">
  // The Lens room (#/lens): what the lenses read, at the token cursor (TokenStrip). At a token, each
  // layer through both lenses (LensLayers: the S lens's read, input, and the R lens's say, output,
  // each by its weight in the depth rule, the crossover band on a wash) and the lens probes' values
  // there, from one replay of the token (the token anatomy, as the workbench's token view replays
  // it, steered as the token was or, asked, without the steering; a runtime without the anatomy
  // replays the lens readout, the R lens alone). Whole, the lens probes over the reply (its node's
  // aggregate envelope; the live readout's words are per token and the node keeps no reply-level
  // words), or, with none, the reply's last token read as above, beside the way to watch a word
  // (the cursor stays whole). A reading shows as the token's only while it
  // answers that token's request: a new one dims it, a failed one takes it away. Fitting and
  // fetching lenses live in the Artifacts room; without a lens the room links there.
  import { onMount } from "svelte";
  import { Icon, MorphText, Segmented, Token } from "@stdlogits/ledger";
  import TokenStrip from "./TokenStrip.svelte";
  import LensLayers from "./lens/LensLayers.svelte";
  import { cachedTokenAnatomy, cachedTokenReadout } from "../../lib/runtime/tokenReadoutCache";
  import { replayCapabilities } from "../../lib/runtime/contracts";
  import { getRuntimeClient } from "../../lib/runtime/registry";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { genStatus, effectiveRawMode } from "../../lib/stores/chat.svelte";
  import { loomTree } from "../../lib/stores/loom.svelte";
  import { goto } from "../../lib/stores/rooms.svelte";
  import { showInstrument } from "../../lib/commands";
  import { lensSourceState, refreshLensSources } from "../../lib/stores/instruments.svelte";
  import { instrumentFamily } from "../../lib/stores/session.svelte";
  import { probeRack } from "../../lib/stores/probes.svelte";
  import { sessionState } from "../../lib/stores/session.svelte";
  import { settled, tokenCursor } from "../../lib/stores/tokenCursor.svelte";
  import { depthTint, readingText } from "../../lib/charts/depth";
  import { lensReplay, replayAnchor } from "../../drawers/token/readings";
  import { readoutLayers, type LensLayerRead } from "../../lib/lensRoom";
  import { probeTitle } from "../rack/probeRows";
  import type { ScalarReadingJSON } from "../../lib/types";

  const anatomyAvailable = replayCapabilities(getRuntimeClient().instruments).tokenAnatomy;
  /** A runtime without the anatomy (the browser's) replays the lens readout instead. */
  const readoutAvailable = $derived(instrumentFamily("lens")?.capabilities.token_readout === true);
  onMount(() => {
    if (lensSourceState.sources.length === 0) void refreshLensSources();
  });
  /** A lens to read with: the anatomy reads through the R lens or any active lens, the readout
   *  through the R lens (the workbench's token view gates on the same answer). */
  const gate = $derived(lensReplay({
    anatomy: anatomyAvailable,
    tokenReadout: readoutAvailable,
    rLens: sessionState.info?.jlens_fitted === true,
    activeLens: lensSourceState.sources.some((source) => source.active),
    replayContext: true,
  }));
  const lensReady = $derived(gate.ready);

  const cursorToken = $derived(tokenCursor.token);
  const whole = $derived(cursorToken === null);
  // ---------- the whole reply: its lens probes, else its last token
  const reply = $derived(tokenCursor.reply);
  const aggregate = $derived(reply?.nodeId ? (loomTree.nodes.get(reply.nodeId)?.aggregate_measurements?.instruments.lens ?? null) : null);
  const replyPins = $derived(Object.entries(aggregate?.readings ?? {}) as [string, ScalarReadingJSON][]);
  /** Whole with no lens probes over the reply: the room reads the reply's last token instead. */
  const lastStands = $derived(whole && replyPins.length === 0);
  const wholeView = $derived(whole && !lastStands);
  /** The token the room reads: the cursor's, or, standing in for the whole reply, its last. */
  const target = $derived(cursorToken ?? (lastStands ? tokenCursor.replyLast : null));
  const here = settled(() => target);
  const token = $derived(here.current);
  const node = $derived(token?.nodeId ? (loomTree.nodes.get(token.nodeId) ?? null) : null);
  const steering = $derived(token ? replayAnchor(node, token.token, token.turn.appliedSteering).steering : null);
  let steered = $state(true);

  // ---------- one token: its anatomy, or the lens readout where the runtime has no anatomy
  /** One token's read: each layer through the lenses, and the lens probes' values there. */
  interface TokenRead { layers: LensLayerRead[]; tokenId: number; pins: Record<string, ScalarReadingJSON> }
  /** The request in flight or last answered (``key``), and how it went. */
  let read = $state.raw<{ key: string; status: "loading" | "ready" | "error"; error: string | null }>({
    key: "", status: "loading", error: null,
  });
  /** The last reading that came back, with the key of the request it answers. */
  let held = $state.raw<{ key: string; value: TokenRead } | null>(null);
  let attempt = $state(0);
  const want = $derived.by(() => {
    if ((!anatomyAvailable && !readoutAvailable) || !lensReady || !token || token.nodeId === null || token.index === null) return null;
    const asSteered = steered || steering === null;
    const kind = anatomyAvailable ? "anatomy" : "readout";
    return {
      kind, nodeId: token.nodeId, index: token.index, tokenId: token.token.tokenId ?? -1, steered: asSteered, raw: effectiveRawMode(),
      key: `${kind}:${token.nodeId}:${token.index}:${asSteered}:${effectiveRawMode()}:${attempt}`,
    };
  });
  /** One token's read, through whichever replay this runtime has. */
  async function readToken(w: NonNullable<typeof want>): Promise<TokenRead> {
    if (w.kind === "anatomy") {
      const { measurements } = await cachedTokenAnatomy(w.nodeId, w.index, { k: 8, units: 1, patterns: 0, steered: w.steered, raw: w.raw });
      if (!measurements.anatomy) throw new Error("No anatomy came back for this token.");
      return { layers: measurements.anatomy.layers, tokenId: measurements.anatomy.token.token_id, pins: measurements.instruments.lens?.readings ?? {} };
    }
    const { measurements } = await cachedTokenReadout("lens", w.nodeId, w.index, { topK: 8, steered: w.steered, raw: w.raw, layers: "all" });
    const lens = measurements.instruments.lens;
    if (!lens?.readout) throw new Error("No lens reading came back. Check the active lens source.");
    return { layers: readoutLayers({ layers: lens.readout.layers ?? [] }, lens.binding.sources.R ?? null), tokenId: w.tokenId, pins: lens.readings ?? {} };
  }
  $effect(() => {
    const w = want;
    if (!w || genStatus.active || read.key === w.key) return;
    const key = w.key;
    read = { key, status: "loading", error: null };
    readToken(w).then(
      (value) => { if (read.key === key) { read = { key, status: "ready", error: null }; held = { key, value }; } },
      (error: unknown) => { if (read.key === key) read = { key, status: "error", error: userFacingError(error, "This token's lenses could not be replayed.") }; },
    );
  });
  /** The reading answers the token, setting and attempt asked for now. */
  const fresh = $derived(held !== null && want !== null && held.key === want.key);
  /** What the token view draws: the fresh reading, or the last one dimmed while the next loads;
   *  never the last one beside a failed read. */
  const shownRead = $derived(held === null || (!fresh && read.status === "error") ? null : held.value);
  const stale = $derived(!fresh || target?.ordinal !== token?.ordinal);
  const tokenPins = $derived(shownRead?.pins ?? {});

  /** The lens probes a view lists: what it read, by the rack's names. */
  const pins = $derived(wholeView ? replyPins : Object.entries(tokenPins) as [string, ScalarReadingJSON][]);
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
  {:else if whole && !token}
    {#if !reply}
      <div class="dw-empty">
        <span class="head">No reply yet</span>
        <button type="button" class="dw-quiet" onclick={() => goto("workbench")}>Workbench<Icon name="right" /></button>
      </div>
    {/if}
  {:else if wholeView}
    <!-- the lens probes over the reply, below -->
  {:else if !anatomyAvailable && !readoutAvailable}
    <div class="dw-empty"><span class="head">Lens</span><p>This runtime does not replay tokens.</p></div>
  {:else if token && (token.nodeId === null || token.index === null)}
    <div class="dw-empty"><span class="head">Lens</span><p>This token has no replay record.</p></div>
  {:else}
    <section class="dw-sec" aria-labelledby="lens-token-head">
      <div class="dw-sec-head">
        <span class="head" id="lens-token-head">{#if token}{lastStands ? "Last token" : "At"} <Token token={token.token.text} />{/if}</span>
        {#if lastStands}
          <button type="button" class="dw-quiet" onclick={() => showInstrument("lens")}>Watch a word<Icon name="right" /></button>
        {/if}
        {#if steering !== null}
          <Segmented label="Readings" value={steered ? "steered" : "unsteered"} onchange={(v) => (steered = v === "steered")}
            options={[{ value: "steered", label: "Steered", hint: steering }, { value: "unsteered", label: "Unsteered", hint: "Replayed without the steering" }]} />
        {/if}
      </div>
      {#if read.status === "error" && read.key === want?.key}
        <div class="dw-row"><span class="meta">{read.error}</span><button type="button" class="dw-text" onclick={() => (attempt += 1)}>Try again</button></div>
      {/if}
      {#if shownRead}
        <div class="read" class:stale>
          <LensLayers layers={shownRead.layers} tokenId={shownRead.tokenId} />
        </div>
      {:else if read.status !== "error"}
        <div class="ghost" aria-hidden="true">{#each [0, 1, 2, 3, 4, 5] as k (k)}<i></i>{/each}</div>
      {/if}
    </section>
  {/if}

  {#if lensReady && pins.length > 0}
    <section class="dw-sec probes" class:stale={!wholeView && stale} aria-labelledby="lens-probes-head">
      <div class="dw-sec-head"><span class="head" id="lens-probes-head">Lens probes</span><span class="meta"><MorphText text={wholeView ? "over this reply" : "at this token"} /></span></div>
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
  .read, .probes { transition: opacity var(--fast) var(--ease); }
  .read.stale, .probes.stale { opacity: 0.55; }
  .ghost { display: grid; gap: 2px; }
  .ghost > i { display: block; height: 36px; border-radius: var(--r-box); background: var(--wash); }

  .meter { position: relative; display: block; width: 56px; height: 4px; border-radius: 2px; background: var(--wash2); }
  .meter > i { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 2px; background: var(--tint, var(--deep)); transition: width var(--glide) var(--ease); }

  .rows { display: grid; gap: 2px; max-width: 640px; }
  .prow { display: grid; grid-template-columns: minmax(0, 1fr) 72px 56px; gap: var(--s3); align-items: center; min-height: 36px; padding: 0 var(--s3); border-radius: var(--r-box); }
  .prow:hover { background: var(--wash); }
  .pname { display: flex; align-items: center; gap: var(--s2); min-width: 0; }
  .ns { color: var(--ink3); font: var(--type-meta); }
  .pval { color: var(--ink2); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: right; }
</style>
