<script lang="ts">
  import Slider from "../Slider.svelte";
  import { MorphText } from "@stdlogits/ledger";
  // Render a discover-mode manifold's PCA variance bars or spectral
  // eigenvalue spectrum, with the picked-k cut highlighted.  Used by
  // ManifoldDrawer's per-row inspector; safe to drop into any other
  // surface that has a ManifoldInfo on hand (the panel skips itself
  // gracefully on authored fits / fits with no diagnostics).

  import type { ManifoldInfo } from "../types";
  import { chartValue } from "../charts/chartValues";
  import { quotientDescription } from "./surfaceGeometry";
  import { isDiscoverFit } from "./fitKind";
  import {
    classifyDiagnostics,
    diagnosticsSummary,
    pcaBars,
    pickDiscoverFit,
    spectralBars,
    type DiagnosticsBar,
  } from "./diagnostics";

  let { manifold }: { manifold: ManifoldInfo } = $props();

  let selectedComponent = $state(0);
  const fit = $derived(pickDiscoverFit(manifold.fitted));
  const diag = $derived(fit ? classifyDiagnostics(fit) : null);
  const bars: DiagnosticsBar[] = $derived(
    diag === null
      ? []
      : diag.kind === "pca"
        ? pcaBars(diag)
        : spectralBars(diag),
  );
  const selectedBar = $derived(bars[Math.min(selectedComponent, bars.length - 1)]);
</script>

{#if quotientDescription(manifold.domain)}
  <div class="diag">
    <p class="summary">{quotientDescription(manifold.domain)}</p>
    <p class="muted">Authored geometry, not a detected topology</p>
  </div>
{/if}
{#if diag !== null}
  <div class="diag" data-kind={diag.kind}>
    <p class="summary">{diagnosticsSummary(diag)}</p>
    {#if selectedBar}
      <p class="summary"><MorphText text={`Component ${selectedBar.index} · ${diag.kind === "pca" ? `${chartValue(selectedBar.value, true)} variance` : `eigenvalue ${chartValue(selectedBar.value)}`} · ${selectedBar.picked ? "kept" : "not kept"}`} identity={diag.kind} /></p>
      <Slider value={Math.min(selectedComponent, bars.length - 1)} min={0} max={Math.max(0, bars.length - 1)} step={1} ariaLabel="Diagnostic component" displayValue={`Component ${selectedBar.index}`} oninput={value => selectedComponent = value} />
    {/if}
    <ol class="bars" aria-label={`${diag.kind} diagnostics`}>
      {#each bars as bar (bar.index)}
        <li
          class="bar"
          class:picked={bar.picked}
          {...{ "aria-description": (`#${bar.index} · ${diag.kind === "pca" ? `${chartValue(bar.value, true)} variance` : `eigenvalue ${chartValue(bar.value)}`} · ${chartValue(bar.frac, true)} of largest component${bar.picked ? " (kept)" : ""}`) }}
        >
          <span class="bar-fill" style="height: {Math.max(2, bar.frac * 100)}%"
          ></span>
          <span class="bar-label">{bar.index}</span>
        </li>
      {/each}
    </ol>
    {#if diag.kind === "spectral" && diag.component_count > 1}
      <p class="warn">
        {diag.component_count} components · raise k_nn or use pca
      </p>
    {/if}
    {#if fit && fit.hyperparams}
      <p class="hyperparams">
        {#each Object.entries(fit.hyperparams) as [k, v] (k)}
          <span>{k} <b>{typeof v === "number" ? v.toString() : v}</b></span>
        {/each}
      </p>
    {/if}
  </div>
{:else if isDiscoverFit(manifold)}
  <p class="muted">Run a fit for diagnostics</p>
{/if}

<style>
  /* A fit's components as bars on a wash: the kept ones in the depth tint, the rest in faint ink. */
  .diag { display: grid; gap: var(--s2); min-width: 0; padding: var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .summary { margin: 0; color: var(--ink); font: var(--type-small); font-variant-numeric: tabular-nums; }
  .bars { display: flex; align-items: stretch; gap: 2px; height: 80px; margin: 0; padding: 0; list-style: none; }
  .bar { position: relative; display: grid; grid-template-rows: minmax(0, 1fr) 16px; flex: 1 1 0; min-width: 6px; }
  .bar-fill { align-self: end; width: 100%; min-height: 2px; border-radius: 2px; background: var(--ink3); opacity: 0.45; transition: background-color var(--fast) var(--ease), opacity var(--fast) var(--ease); }
  .bar.picked .bar-fill { background: var(--tint); opacity: 1; }
  .bar-label { justify-self: center; color: var(--ink3); font: var(--type-meta); }
  .hyperparams { display: flex; flex-wrap: wrap; gap: var(--s1) var(--s4); margin: 0; color: var(--ink3); font: var(--type-meta); }
  .hyperparams b { color: var(--ink); font-weight: 500; }
  .warn { margin: 0; color: var(--ink); font: var(--type-meta); }
  .muted { margin: 0; color: var(--ink3); font: var(--type-meta); }
</style>
