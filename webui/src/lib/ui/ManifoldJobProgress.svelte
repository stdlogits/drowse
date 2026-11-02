<script lang="ts">
  import { elapsedLabel, etaLabel, remainingWorkMs } from "../manifoldProgress";
  import { manifoldJobs, dismissManifoldJob, cancelManifoldJob } from "../stores/manifoldJobs.svelte";
  import { openDrawer } from "../stores/drawers.svelte";
  import { pushToast } from "../stores/toasts.svelte";
  import { describeError } from "../runtime/errors";
  import { steerRack } from "../stores/steering.svelte";
  import { steeringDrawerFor } from "../manifolds/fitKind";
  import { runtimeClient } from "../runtime/client";
  import { jobNote } from "../manifolds/jobNote";

  let { compact = false }: { compact?: boolean } = $props();
  let now = $state(Date.now());
  const job = $derived(manifoldJobs.current);
  const busy = $derived(job !== null && ["running", "waiting", "cancelling"].includes(job.status));
  const fraction = $derived(job?.progress.completed !== null && job?.progress.total
    ? job.progress.completed! / job.progress.total : null);
  const remaining = $derived(job && busy ? remainingWorkMs(job.samples, job.progress.total, now) : null);
  const title = $derived(job?.status === "failed" ? "Creation needs attention"
    : job?.status === "cancelled" ? "Creation cancelled"
    : job?.status === "cancelling" ? "Cancelling…" : job?.progress.label ?? "");
  const timing = $derived(job?.status === "waiting" ? "Fitting starts next"
    : job?.status === "cancelling" ? "Waiting for the current step to stop"
    : !busy ? "" : remaining === null ? "Estimating this step…" : `${etaLabel(remaining)} in this step`);
  // The browser runtime does the work in this tab; the server does not need it open.
  const keepOpen = runtimeClient.mode !== "http";
  const next = $derived(job ? jobNote(job, keepOpen) : { note: "", tip: "" });
  const announcement = $derived(job ? `${job.namespace}/${job.name}: ${title}${busy && fraction !== null ? `, ${Math.floor(fraction * 10) * 10}% of this step` : ""}` : "");

  $effect(() => {
    if (!busy) return;
    now = Date.now();
    const timer = setInterval(() => { now = Date.now(); }, 1000);
    return () => clearInterval(timer);
  });

  async function cancel(): Promise<void> {
    try { await cancelManifoldJob(); }
    catch (error) {
      pushToast(`Could not cancel: ${describeError(error)}`, { kind: "error" });
    }
  }

  function openResult(): void {
    if (!job) return;
    if (job.phase === "scoring") { openDrawer("template_lab"); return; }
    openDrawer(steeringDrawerFor(steerRack.catalog.find(row => row.namespace === job.namespace && row.name === job.name)));
  }
</script>

<p class="sr-only" role="status" aria-live="polite" aria-atomic="true">{announcement}</p>
{#if job}
  <section class="job" class:compact aria-label={`Creation progress for ${job.namespace}/${job.name}`}>
    <div class="row">
      <span class="id"><span class="title">{title}</span><span class="meta name">{job.namespace}/{job.name}</span></span>
      {#if busy || job.status === "complete"}
        <span class="meter" role="progressbar" aria-label={job.phase === "generating" ? "Examples generated" : job.progress.label}
          aria-valuemin="0" aria-valuemax="100" aria-valuenow={fraction === null ? undefined : Math.floor(fraction * 100)}
          aria-valuetext={fraction === null ? title : `${job.progress.completed} of ${job.progress.total} completed in this step`}>
          <i style:transform={`scaleX(${fraction ?? 0})`}></i>
        </span>
      {/if}
      <span class="meta">{job.progress.completed !== null && job.progress.total !== null ? `${job.progress.completed} / ${job.progress.total} ${job.phase === "generating" ? "examples" : "completed"}` : job.progress.detail}</span>
      {#if timing}<span class="meta">{timing}</span>{/if}
      <span class="meta">{elapsedLabel((job.finishedAt ?? now) - job.startedAt)} elapsed</span>
      <span class="actions">
        {#if busy && job.cancellable}
          <button type="button" class="dw-text" disabled={job.status !== "running"} onclick={() => void cancel()}>Cancel</button>
        {:else if !busy}
          <button type="button" class="dw-quiet" data-tip={next.tip || undefined} onclick={openResult}>Open library</button>
          <button type="button" class="dw-text" onclick={dismissManifoldJob}>Dismiss</button>
        {/if}
      </span>
    </div>
    {#if job.error}<p class="error">{job.error}</p>{/if}
    {#if next.note}<p class="meta">{next.note}</p>{/if}
  </section>
{/if}

<style>
  /* A running job, drawn as the Lens room draws a fit: its name, a meter on the ramp, the count and
   * the time left, on a wash band. */
  .job { display: grid; gap: var(--s1); min-width: 0; padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); color: var(--ink2); }
  .compact { padding: var(--s3) var(--s6); border-radius: 0; }
  .row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s4); min-width: 0; }
  .id { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s1) var(--s3); min-width: 0; }
  .title { color: var(--ink); font: var(--type-control); }
  .name { overflow-wrap: anywhere; }
  .meter { position: relative; flex: none; width: 96px; height: 4px; border-radius: 2px; background: var(--wash2); overflow: hidden; }
  .meter i { position: absolute; inset: 0; border-radius: 2px; background: var(--tint); transform-origin: left; transition: transform var(--glide) var(--ease); }
  .actions { display: flex; align-items: center; gap: var(--s3); margin-left: auto; }
  .actions .dw-quiet { background: var(--wash2); }
  p { margin: 0; }
  .error { color: var(--ink); font: var(--type-small); overflow-wrap: anywhere; }
  @media (max-width: 720px) {
    .compact { padding: var(--s3) var(--s4); }
    /* A 44 px target for the text button, without growing the band. */
    .actions .dw-text { position: relative; }
    .actions .dw-text::after { content: ""; position: absolute; inset: -12px -6px; }
  }
  @media (forced-colors: active) { .meter { outline: 1px solid CanvasText; } .meter i { background: Highlight; } }
</style>
