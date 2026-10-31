<script lang="ts">
  import Select from "../Select.svelte";
  import { MorphText } from "@stdlogits/ledger";
  let { sentence, slot, values }: { sentence: string; slot: string; values: string[] } = $props();
  let chosen = $state("");
  const candidate = $derived(values.includes(chosen) ? chosen : values[0] ?? "");
  const parts = $derived(slot ? sentence.split(slot) : []);
</script>
{#if parts.length === 2 && values.length > 0}
  <section class="template-preview" aria-label="Template slot preview">
    <label><span class="sr-only">Preview candidate</span><Select value={candidate} options={values.map(value => ({value, label: value}))} onchange={value => chosen = value} ariaLabel="Preview candidate" /></label>
    <p>{parts[0]}<mark><MorphText text={candidate} numbers={false} /></mark>{parts[1]}</p>
  </section>
{/if}
<style>
  /* The sentence with the slot filled, the candidate marked as a token a unit fired on: the depth
   * tint at 22% with its 2px spread, ink on top. */
  .template-preview { display: grid; justify-items: start; gap: var(--s2); min-width: 0; }
  p { margin: 0; color: var(--ink); font: var(--type-body); line-height: 32px; white-space: pre-wrap; overflow-wrap: anywhere; }
  mark { --mark: color-mix(in oklch, var(--tint) 22%, transparent); padding: 0; border-radius: 4px; background: var(--mark); box-shadow: 0 0 0 2px var(--mark); color: var(--ink); }
</style>
