<script lang="ts">
  // Builder-drawer validation summary.  Callers reveal it after a submit
  // attempt so editing a field does not repeatedly interrupt screen-reader
  // users with a changing global alert.
  //
  // Renders nothing when ``messages`` is empty so callers can drop it in
  // unconditionally.

  interface Props {
    /** What action would run if validation passed — completes the
     *  "not ready to ____:" header sentence.  Examples: "extract",
     *  "build", "discover". */
    verb: string;
    messages: string[];
  }

  let { verb, messages }: Props = $props();
  const uid = $props.id();
</script>

{#if messages.length > 0}
  <section class="sk-validation" aria-labelledby={`${uid}-heading`}>
    <p id={`${uid}-heading`} class="sk-validation-head">Check these before you {verb}</p>
    <ul>
      {#each messages as m (m)}
        <li>{m}</li>
      {/each}
    </ul>
  </section>
{/if}

<style>
  /* Ink on a wash: Ledger has no warning colour. */
  .sk-validation { display: grid; gap: var(--s2); padding: var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .sk-validation-head { margin: 0; color: var(--ink); font: var(--type-control); }
  .sk-validation ul { display: grid; gap: var(--s1); margin: 0; padding: 0; list-style: none; color: var(--ink2); font: var(--type-small); }
</style>
