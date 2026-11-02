<script lang="ts">
  // A lens family's column with drowse's local fit (the column panels/JLensSourceSection.svelte gives
  // LensQuad): the family's LensFamily, told when a fit runs, with the fit's parts placed in it: the
  // running fit's progress among the sources, a failed fit's error (and Replace) under them, and a
  // local fit (documents, kind, sites) after them. Only that section renders it; drowse.ai swaps the
  // section through the @runtime-jlens-source alias, so its hosted build carries none of this.
  import { IconButton, MorphText, Segmented } from "@stdlogits/ledger";
  import LensFamily from "./LensFamily.svelte";
  import {
    FAMILY_KINDS,
    documentLadder,
    fitProgressText,
    replaceMessage,
    replaceTarget,
    sourceFamily,
    stepDocuments,
    type FitBody,
    type FitPart,
    type LensFamily as Family,
  } from "../../../lib/lensRoom";
  import { etaLabel, remainingWorkMs, type WorkSample } from "../../../lib/manifoldProgress";
  import { runtimeOperationAvailability } from "../../../lib/runtime/ui-capabilities";
  import { lensFetch, lensSourceState } from "../../../lib/stores.svelte";
  import { lensFit } from "../../../lib/stores/instrumentAuthoring.svelte";

  let { family, n, scale }: { family: Family; n: number; scale: number | null } = $props();

  const allowFit = runtimeOperationAvailability("jlens_fitting").available;

  // One lens preparation runs at a time (the server's lens group), whichever family it is for.
  const busy = $derived(lensSourceState.busy || lensFetch.state.running || lensFit.state.running);
  const fitSource = $derived(lensFit.state.latest?.source ?? null);
  const ofFamily = $derived(sourceFamily(fitSource, lensSourceState.sources, lensSourceState.providers) === family);
  const fitting = $derived(allowFit && lensFit.state.running && ofFamily);
  const fitError = $derived(allowFit && ofFamily ? lensFit.state.error : null);

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
    ? replaceMessage(replaces, lensSourceState.sources.find((row) => row.source === replaces)?.n_prompts, lastFit.prompts)
    : null);

  function startFit(force = false): void {
    if (!allowFit || busy || documents === null) return;
    const body = force && lastFit ? lastFit : { prompts: documents, kind: fitKind, sites: both ? ["out", "mid"] : ["out"] };
    lastFit = body;
    void lensFit.start(force ? { ...body, force: true } : body);
  }
</script>

<LensFamily {family} {n} {scale} fitRunning={lensFit.state.running}>
  {#snippet fit(part: FitPart)}
    {#if part === "status"}
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
    {:else if part === "error"}
      {#if fitError}
        <p class="meta err" role="alert">{replaceText ?? fitError}</p>
        {#if replaces}<button type="button" class="dw-text act replace" disabled={busy} onclick={() => startFit(true)}>Replace {replaces}</button>{/if}
      {/if}
    {:else if part === "form" && allowFit && !fitting && documents !== null}
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
  {/snippet}
</LensFamily>

<style>
  /* The fit's markup sits in LensFamily's column (the snippet keeps this component's scope), so its
   * rows match the Sources rows there: the status is one of them, the form a section beside it. */
  .tight { gap: var(--s2); }
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
  .l1 { overflow: hidden; color: var(--ink); font: var(--type-control); white-space: nowrap; text-overflow: ellipsis; }
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
    .item { grid-template-columns: 88px minmax(0, 1fr) auto; min-height: 44px; }
    .act { min-width: 44px; min-height: 44px; }
    .step :global(button.icon) { width: 44px; height: 44px; }
    .fitctl :global(.seg > button) { position: relative; height: 36px; }
    .fitctl :global(.seg > button::after) { content: ""; position: absolute; inset: -4px 0; }
  }
</style>
