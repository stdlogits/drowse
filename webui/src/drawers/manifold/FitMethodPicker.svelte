<script lang="ts">
  // The flat / curved / auto choice, shared by every tab that routes
  // through a discover fit.  ``tuning`` is a $state object owned by the
  // form. The owner supplies the mutation callback so this child never
  // writes through an unbound state prop.

  import { Segmented } from "@stdlogits/ledger";
  import type { DiscoverFitMode, DiscoverTuning } from "./shared";

  let {
    tuning,
    /** ``spectral``'s note differs where node counts are the practical
     *  constraint (the generated tab) from where they aren't. */
    spectralNote = "curved",
    linearOnly = false,
    step,
    onchange,
  }: {
    tuning: DiscoverTuning;
    spectralNote?: string;
    linearOnly?: boolean;
    /** Its number when it is a step of the form (the template and authored paths). */
    step?: number;
    onchange: (value: DiscoverFitMode) => void;
  } = $props();

  $effect(() => {
    if (linearOnly && tuning.fitMode !== "pca") onchange("pca");
  });
</script>

<section class="step" aria-label="Fit method">
  <h3 class="step-title">{#if step}<span class="n">{step}</span>{/if}Fit</h3>
  {#if linearOnly}
    <p class="dim-note">Linear (PCA) · curved layouts are not in the hosted release</p>
  {:else}
    <div class="row">
      <Segmented
        label="Fit method"
        value={tuning.fitMode}
        {onchange}
        options={[
          { value: "auto", label: "Auto", hint: "Suggest a flat or periodic layout per model. This does not recover Klein-bottle or RP² coordinates." },
          { value: "pca", label: "PCA", hint: "Flat" },
          { value: "spectral", label: "Spectral", hint: spectralNote },
        ]}
      />
      <span class="dim-note">{tuning.fitMode === "auto" ? "flat or periodic" : tuning.fitMode === "pca" ? "flat" : spectralNote}</span>
    </div>
  {/if}
</section>

<style>
  .row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s4); }
</style>
