<script lang="ts">
  // One lens family as a column of the Lens canvas's 2x2 (S early, R late): its selected source at
  // each site, then its sources (choose one; fetch a provider not yet here; a local fit's progress
  // while it runs) and a local fit (documents, kind, sites). The column's rows sit in the parent's
  // grid (JLensSourceSection) so the two families' out and mid cards line up beside the row labels.
  import { IconButton, MorphText, Segmented } from "@stdlogits/ledger";
  import LensCard from "./LensCard.svelte";
  import {
    FAMILY_KINDS,
    documentLadder,
    documentsText,
    familyItems,
    familySpan,
    familyWeights,
    fitProgressText,
    foreignKind,
    rangeText,
    replaceTarget,
    siteSeries,
    sitesText,
    sourceFamily,
    stepDocuments,
    unloaded,
    type FitBody,
    type LensFamily,
    replaceMessage,
  } from "../../../lib/lensRoom";
  import { etaLabel, remainingWorkMs, type WorkSample } from "../../../lib/manifoldProgress";
  import { runtimeOperationAvailability } from "../../../lib/runtime/ui-capabilities";
  import { instrumentFamily, lensFetch, lensSourceState, useLensSource } from "../../../lib/stores.svelte";
  import { lensFit } from "../../../lib/stores/instrumentAuthoring.svelte";

  let { family, n, scale }: { family: LensFamily; n: number; scale: number | null } = $props();

  const allowFit = runtimeOperationAvailability("jlens_fitting").available;
  const allowFetch = $derived(instrumentFamily("lens")?.capabilities.preparations.includes("fetch") === true);
  const allowSwitch = $derived(instrumentFamily("lens")?.capabilities.source_switch === true);

  const weights = $derived(familyWeights(lensSourceState.band, family));
  const span = $derived(familySpan(lensSourceState.band, family));
  const items = $derived(familyItems(lensSourceState.sources, lensSourceState.providers, family)
    .filter(item => allowSwitch || item.row?.active === true || (item.row === null && allowFetch)));
  const active = $derived(lensSourceState.sources.find(row => row.family === family && row.active === true) ?? null);
  const activeProvider = $derived(lensSourceState.providers.find(row => row.source === active?.source) ?? null);

  // One lens preparation runs at a time (the server's lens group), whichever family it is for.
  const busy = $derived(lensSourceState.busy || lensFetch.state.running || lensFit.state.running);
  const ofFamily = (source: string | null | undefined) =>
    sourceFamily(source, lensSourceState.sources, lensSourceState.providers) === family;
  const fetching = $derived(lensFetch.state.running ? (lensFetch.state.latest?.source ?? null) : null);
  const fitSource = $derived(lensFit.state.latest?.source ?? null);
  const fitting = $derived(allowFit && lensFit.state.running && ofFamily(fitSource));
  const fitError = $derived(allowFit && ofFamily(fitSource) ? lensFit.state.error : null);
  const fetchError = $derived(ofFamily(lensFetch.state.latest?.source) ? lensFetch.state.error : null);

  // ---------- the fit's progress and time left (from the documents it has finished so far) ----------
  let samples = $state<WorkSample[]>([]);
  $effect(() => {
    const current = lensFit.state.current;
    if (!fitting) {
      samples = [];
      return;
    }
    if (current > 0 && samples.at(-1)?.completed !== current) samples = [...samples, { at: Date.now(), completed: current }];
  });
  const left = $derived.by(() => {
    if (!fitting || lensFit.state.cancelling) return lensFit.state.cancelling ? "Cancelling" : "";
    const ms = remainingWorkMs(samples, lensFit.state.total || null, samples.at(-1)?.at ?? 0);
    return ms === null ? "" : etaLabel(ms);
  });
  const fitShare = $derived(lensFit.state.total > 0 ? Math.min(1, lensFit.state.current / lensFit.state.total) : 0);
  /** The stage until the first document is in ("Streaming 64 documents from …"), then the count. */
  const progressText = $derived(fitProgressText(lensFit.state.current, lensFit.state.total, lensFit.state.message));

  // ---------- a local fit: documents (the server's default, doubled or halved), kind, sites ----------
  const ladder = $derived(documentLadder(lensSourceState.fitDocuments ?? 1));
  let picked = $state<number | null>(null);
  const documents = $derived(picked ?? lensSourceState.fitDocuments);
  const [ownKind, plainKind] = $derived(FAMILY_KINDS[family]);
  let kind = $state<string | null>(null);
  const fitKind = $derived(kind ?? ownKind);
  let both = $state(true);
  const kindOptions = $derived([
    { value: ownKind, label: ownKind.toUpperCase(), hint: `${ownKind.toUpperCase()}-lens, under the RelP rules` },
    { value: plainKind, label: plainKind.toUpperCase(), hint: `${plainKind.toUpperCase()}-lens, plain gradients` },
  ]);

  // A fit that would shrink an artifact already here (fewer documents, another corpus, fewer sites)
  // fails rather than replace it, and the server marks that refusal (``replaceable``); only then does
  // the failure offer the same fit again, replacing it. Any other failure offers nothing.
  let lastFit = $state<FitBody | null>(null);
  const replaces = $derived(fitError ? replaceTarget(lensFit.state.latest, lastFit) : null);
  const replaceText = $derived(replaces && lastFit
    ? replaceMessage(replaces, items.find((item) => item.source === replaces)?.row?.n_prompts, lastFit.prompts)
    : null);

  function startFit(force = false): void {
    if (!allowFit || busy || documents === null) return;
    const body = force && lastFit ? lastFit : { prompts: documents, kind: fitKind, sites: both ? ["out", "mid"] : ["out"] };
    lastFit = body;
    void lensFit.start(force ? { ...body, force: true } : body);
  }

  function itemMeta(item: (typeof items)[number]): string {
    if (unloaded(item.row)) return "not loaded for this model";
    if (item.row) return [foreignKind(item.row.lens, family), documentsText(item.row.n_prompts), sitesText(item.row.sites)].filter(Boolean).join(" · ");
    const provider = item.provider;
    return provider ? [foreignKind(provider.lens, family), sitesText(provider.sites), provider.repo_id].filter(Boolean).join(" · ") : "";
  }
</script>

<div class="fam">
  <div class="colhead"><h2 class="h">{family}-lens</h2><span class="meta">{family === "S" ? "early layers" : "late layers"}</span></div>

  <LensCard {family} site="out" row={active} provider={activeProvider} {n} {weights} {span} {scale} />
  <LensCard {family} site="mid" row={active} provider={activeProvider} {n} {weights} {span} {scale} />

  <div class="foot">
    <div class="dw-sec tight">
      <span class="head">Sources</span>
      <div class="items">
        {#each items as item, i (item.source)}
          {#if item.row && !unloaded(item.row)}
            <button type="button" class="item rise" style:--i={i} aria-pressed={item.row.active === true}
              disabled={busy && item.row.active !== true} onclick={() => { if (!item.row?.active) void useLensSource(item.source); }}>
              <span class="l1">{item.source}</span>
              <span class="meta im">{itemMeta(item)}</span>
              <span class="meta">{rangeText(siteSeries(item.row, "out", n).split)}</span>
            </button>
          {:else}
            <!-- A provider still to fetch, or the selected source when it does not load for these
                 weights (a provider's offers to fetch it again; a local fit's is fitted again below). -->
            <div class="item rise" class:sel={item.row !== null} style:--i={i} aria-current={item.row ? "true" : undefined}>
              <span class="l1" class:todo={!item.row}>{item.source}</span>
              <span class="meta im">{#if fetching === item.source}<MorphText text={lensFetch.state.message ?? "Fetching"} />{:else}{itemMeta(item)}{/if}</span>
              {#if allowFetch && item.provider}
                <button type="button" class="dw-text act" disabled={busy} onclick={() => void lensFetch.start({ source: item.source })}>
                  {fetching === item.source ? "Fetching" : item.row ? "Fetch again" : "Fetch"}
                </button>
              {/if}
            </div>
          {/if}
        {/each}
        {#if fitting}
          <div class="item rise" role="status" aria-label="Lens fit progress">
            <span class="l1">{fitSource}</span>
            <span class="prog">
              <span class="meter" role="progressbar" aria-label="Documents fitted" data-tip="Generation pauses while documents are fitted" aria-valuemin={0}
                aria-valuemax={Math.max(lensFit.state.total, 1)} aria-valuenow={lensFit.state.current}>
                <i style:--lens-fill={family === "S" ? "var(--deep)" : "var(--late)"} style:width={`${(fitShare * 100).toFixed(1)}%`}></i>
              </span>
              <span class="meta stage" data-tip={progressText}><MorphText text={progressText} /></span>
              {#if left}<span class="meta"><MorphText text={left} /></span>{/if}
            </span>
            <button type="button" class="dw-text act" disabled={lensFit.state.cancelling} onclick={() => void lensFit.cancel()}>Cancel</button>
          </div>
        {/if}
      </div>
      {#if fitError}
        <p class="meta err" role="alert">{replaceText ?? fitError}</p>
        {#if replaces}<button type="button" class="dw-text act replace" disabled={busy} onclick={() => startFit(true)}>Replace {replaces}</button>{/if}
      {/if}
      {#if fetchError}<p class="meta err" role="alert">{fetchError}</p>{/if}
    </div>

    {#if allowFit && !fitting && documents !== null}
      <div class="dw-sec tight">
        <span class="head">Fit locally</span>
        <div class="fitctl">
          <div class="step" role="group" aria-label="Documents">
            <IconButton icon="left" label="Fewer documents" disabled={documents <= ladder[0]}
              onclick={() => (picked = stepDocuments(ladder, documents, -1))} />
            <span class="stepv"><span class="num"><MorphText text={String(documents)} /></span><span class="meta">documents</span></span>
            <IconButton icon="right" label="More documents" disabled={documents >= ladder[ladder.length - 1]}
              onclick={() => (picked = stepDocuments(ladder, documents, 1))} />
          </div>
          <Segmented label="Kind" value={fitKind} options={kindOptions} onchange={(value) => (kind = value)} />
          <Segmented label="Sites to fit" value={both ? "both" : "out"} onchange={(value) => (both = value === "both")}
            options={[{ value: "out", label: "Out" }, { value: "both", label: "Out and mid" }]} />
          <button type="button" class="dw-button go" disabled={busy} onclick={() => startFit()}
            data-tip="Generation pauses while documents are fitted; a large fit can take hours">Fit</button>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  /* The column's four rows (head, out, mid, foot) are the parent quad's, so cards align across. */
  .fam { grid-row: 1 / span 4; display: grid; grid-template-rows: subgrid; gap: var(--s3); min-width: 0; }
  .colhead { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s1) var(--s3); }
  .h { margin: 0; color: var(--ink); font: var(--type-mark); letter-spacing: -0.01em; }

  .foot { display: grid; gap: var(--s4); align-content: start; min-width: 0; padding-top: var(--s2); }
  .tight { gap: var(--s2); }
  .items { display: grid; gap: 2px; }
  .item {
    display: grid;
    grid-template-columns: 104px minmax(0, 1fr) auto;
    gap: var(--s3);
    align-items: center;
    min-height: 40px;
    padding: var(--s1) var(--s3);
    border: 0;
    border-radius: var(--r-box);
    background: transparent;
    color: inherit;
    font: inherit;
    text-align: left;
    transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease);
  }
  button.item { cursor: pointer; }
  button.item:hover:not(:disabled) { background: var(--wash); }
  .item[aria-pressed="true"], .item[aria-pressed="true"]:hover, .item.sel { background: var(--wash2); cursor: default; }
  button.item:active:not(:disabled) { scale: 0.99; }
  button.item:disabled { cursor: default; }
  button.item:disabled:not([aria-pressed="true"]) { opacity: 0.5; }
  .l1 { overflow: hidden; color: var(--ink); font: var(--type-control); white-space: nowrap; text-overflow: ellipsis; }
  .l1.todo { color: var(--ink2); }
  .im { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  /* A morphing fetch message ellipsizes inside the morph, as plain metadata does. */
  .im :global(.morph-text) { display: block; overflow: hidden; text-overflow: ellipsis; }
  .act { font: var(--type-meta); }
  .err { color: var(--ink2); }
  .replace { justify-self: start; }

  .prog { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s3); min-width: 0; }
  /* One line beside the meter; a long stage (the corpus and its commit) ends in an ellipsis and
   * shows whole in its tip. */
  .stage { flex: 1 1 160px; min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .stage :global(.morph-text) { display: block; overflow: hidden; text-overflow: ellipsis; }
  .meter { position: relative; width: 96px; height: 4px; border-radius: 2px; background: var(--wash2); overflow: hidden; }
  .meter > i { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 2px; background: var(--lens-fill); transition: width var(--glide) var(--ease); }

  .fitctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s3); }
  .step { display: inline-flex; align-items: center; gap: var(--s2); }
  .stepv { display: inline-flex; align-items: baseline; justify-content: center; gap: 6px; min-width: 104px; }
  .num { color: var(--ink); font: var(--type-label); font-variant-numeric: tabular-nums; }
  .go { margin-left: auto; }

  /* Phone sizes (the Phone canvas): every target 44 px. The stepper's buttons grow; a segment keeps
   * its 36 px pill inside a 44 px group, with a hit area to the group's edges. */
  @media (max-width: 720px) {
    .fam { grid-row: auto; grid-template-rows: none; }
    .item { grid-template-columns: 88px minmax(0, 1fr) auto; min-height: 44px; }
    .act { min-width: 44px; min-height: 44px; }
    .step :global(button.icon) { width: 44px; height: 44px; }
    .fitctl :global(.seg > button) { position: relative; height: 36px; }
    .fitctl :global(.seg > button::after) { content: ""; position: absolute; inset: -4px 0; }
  }
</style>
