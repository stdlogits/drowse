<script lang="ts">
  // The Model room's Health section (and, opened anywhere else, a dialog): what the session holds
  // now as a row of figures, the checks and warnings as rows, and the full settings reset last.
  import { Icon, IconButton, MorphText } from "@stdlogits/ledger";
  import ResetSettings from "../lib/ui/ResetSettings.svelte";
  import {
    closeDrawer,
    genStatus,
    geometricMeanPpl,
    loomTree,
    probeRack,
    sessionState,
    steerRack,
    vectorsState,
  } from "../lib/stores.svelte";
  import Button from "../lib/ui/Button.svelte";
  import { healthState, healthWarnings, refreshHealth } from "../lib/stores/health.svelte";

  let { params = null }: { params?: unknown } = $props();
  const embedded = $derived(
    (params as { embedded?: boolean } | null)?.embedded === true,
  );
  const uid = $props.id();

  const busy = $derived(healthState.busy);
  const lastAudit = $derived(healthState.lastAudit ? new Date(healthState.lastAudit).toLocaleTimeString() : null);
  const errorMsg = $derived(healthState.error);

  const ppl = $derived(geometricMeanPpl(genStatus));
  const warnings = $derived(healthWarnings());

  const figures = $derived([
    {
      key: "Generation",
      value: genStatus.active ? "active" : genStatus.finishReason ?? "idle",
      meta: `${genStatus.tokensSoFar}/${genStatus.maxTokens || "–"} tokens · ${genStatus.tokPerSec.toFixed(1)} tok/s`,
    },
    // The footer's figure (the geometric mean of each step's entropy perplexity), named as the
    // footer names it: a reply's own perplexity (from its chosen tokens) is another number.
    { key: "Uncertainty", value: ppl === null ? "–" : ppl.toFixed(2), meta: `entropy perplexity · ${genStatus.ppl.count} steps` },
    {
      key: "Loom",
      value: loomTree.nodes.size ? String(loomTree.nodes.size) : "–",
      meta: `rev ${loomTree.loaded ? loomTree.rev : "–"} · depth ${loomTree.activePath.length || "–"}`,
    },
    {
      key: "Manifolds",
      value: String(steerRack.catalog.length),
      meta: `${steerRack.entries.size} racked · ${vectorsState.names.length} resident`,
    },
    {
      key: "Probes",
      value: String(probeRack.active.length),
      meta: `${probeRack.entries.size} rows · ${steerRack.correlation ? "correlation cached" : "no correlation"}`,
    },
  ]);

  const checks = $derived([
    { key: "Session", ok: !!sessionState.info, state: sessionState.info ? "loaded" : "unavailable" },
    { key: "Loom", ok: loomTree.loaded && !loomTree.error, state: loomTree.loaded && !loomTree.error ? "loaded" : "unavailable" },
    { key: "Manifolds", ok: steerRack.catalog.length > 0, state: steerRack.catalog.length > 0 ? "available" : "none installed" },
    { key: "Probes", ok: probeRack.active.length > 0, state: probeRack.active.length > 0 ? "active" : "none active" },
    { key: "Correlation", ok: steerRack.correlation !== null, state: steerRack.correlation !== null ? "measured" : "not measured" },
  ]);
  const passing = $derived(checks.filter(check => check.ok).length);

  async function audit(): Promise<void> {
    if (!healthState.busy) await refreshHealth();
  }
</script>

<section
  class="health"
  class:embedded
  role={embedded ? "region" : undefined}
  aria-label={embedded ? "Model health" : "Health drawer"}
>
  {#if !embedded}
    <header class="dlg-head">
      <h2>Model health</h2>
      <IconButton icon="close" label="Close" onclick={closeDrawer} />
    </header>
  {/if}

  <div class="body" aria-busy={busy}>
    {#if !embedded}
      <div class="ident">
        <span class="name">{sessionState.info?.model_id ?? "No model"}</span>
        <span class="meta">{sessionState.info ? `${sessionState.info.device}/${sessionState.info.dtype}` : "offline"}</span>
      </div>
    {/if}

    <div class="dw-row">
      <Button {busy} disabled={busy} onclick={audit}><MorphText text={busy ? "Checking…" : "Check again"} numbers={false} /></Button>
      {#if lastAudit}<span class="meta">Checked {lastAudit}</span>{/if}
    </div>
    {#if errorMsg}<p class="error" role="alert">Health check failed: {errorMsg}</p>{/if}

    <section class="dw-sec" aria-labelledby={`${uid}-now`}>
      <span class="head" id={`${uid}-now`}>Now</span>
      <dl class="figures">
        {#each figures as figure (figure.key)}
          <div class="figure">
            <dt class="meta">{figure.key}</dt>
            <dd class="value"><MorphText text={figure.value} /></dd>
            <dd class="meta">{figure.meta}</dd>
          </div>
        {/each}
      </dl>
    </section>

    <section class="dw-sec narrow" aria-labelledby={`${uid}-checks`}>
      <div class="dw-sec-head"><span class="head" id={`${uid}-checks`}>Checks</span><span class="meta"><MorphText text={`${passing} of ${checks.length}`} /></span></div>
      <ul class="rows">
        {#each checks as check (check.key)}
          <li class="row" class:ok={check.ok}>
            <span class="mark" aria-hidden="true">{#if check.ok}<Icon name="check" />{:else}<i></i>{/if}</span>
            <span class="key">{check.key}</span>
            <span class="meta">{check.state}</span>
          </li>
        {/each}
      </ul>
    </section>

    <section class="dw-sec narrow" aria-labelledby={`${uid}-warnings`}>
      <div class="dw-sec-head"><span class="head" id={`${uid}-warnings`}>Warnings</span><span class="meta"><MorphText text={warnings.length ? String(warnings.length) : "None"} /></span></div>
      {#if warnings.length}
        <ul class="warnings">
          {#each warnings as warning (warning)}
            <li>{warning}</li>
          {/each}
        </ul>
      {/if}
    </section>

    <section class="dw-sec"><ResetSettings full /></section>
  </div>
</section>

<style>
  .health { display: flex; flex-direction: column; min-height: 0; color: var(--ink2); }
  .dlg-head { display: flex; flex: none; align-items: center; justify-content: space-between; gap: var(--s4); padding: var(--s5) var(--s5) 0; }
  .dlg-head h2 { margin: 0; color: var(--ink); font: var(--type-mark); }
  .body { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s6); align-content: start; max-width: 760px; padding: var(--s5); overflow: auto; }
  .embedded .body { padding: 0; overflow: visible; }
  .ident { display: grid; gap: 2px; }
  .ident .name { color: var(--ink); font: 600 16px/24px var(--display); letter-spacing: -0.01em; overflow-wrap: anywhere; }
  .error { color: var(--ink); font: var(--type-small); overflow-wrap: anywhere; }

  /* What the session holds now: figures on paper, no boxes. */
  .figures { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: var(--s5); margin: 0; }
  .figure { display: grid; gap: 2px; align-content: start; min-width: 0; }
  .figure dd { margin: 0; overflow-wrap: anywhere; }
  .figure .value { color: var(--ink); font: var(--type-lead); font-variant-numeric: tabular-nums; }

  .rows { display: grid; gap: 2px; }
  /* No fill, so no inset: a row's state ends under its head's count. */
  .row { display: grid; grid-template-columns: 20px minmax(0, 1fr) auto; gap: var(--s3); align-items: center; min-height: 36px; }
  .row .key { color: var(--ink2); font: var(--type-small); }
  .row.ok .key { color: var(--ink); }
  .mark { display: grid; place-items: center; color: var(--ink); }
  .mark i { width: 6px; height: 6px; border-radius: 50%; background: var(--ink3); opacity: 0.4; }
  /* Rows read across a short measure, not the room's width. */
  .narrow { max-width: 520px; }

  .warnings { display: grid; gap: var(--s2); padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .warnings li { color: var(--ink); font: var(--type-small); overflow-wrap: anywhere; }

  @media (max-width: 720px) {
    .dlg-head { padding: var(--s4) var(--s4) 0; }
    .body { padding: var(--s4); }
    .figures { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s4); }
  }
  @media (max-width: 720px), (pointer: coarse) {
    .dlg-head :global(.icon) { width: 44px; height: 44px; }
  }
</style>
