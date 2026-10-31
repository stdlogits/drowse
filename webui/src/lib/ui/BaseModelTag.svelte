<script lang="ts">
  // Marks a base model (one that continues text rather than following chat turns): a small wash
  // chip, or, ``plain``, the bare word beside a name. Path and props are shared with drowse.ai.
  let { plain = false }: { plain?: boolean } = $props();
</script>

<span class="base-tag" class:plain {...{ "aria-description": "Base model · continues text without chat instructions" }}>base</span>

<style>
  .base-tag {
    display: inline-block;
    margin-inline-start: var(--s2);
    padding: 0 var(--s2);
    border-radius: var(--r-chip);
    background: var(--wash);
    color: var(--ink2);
    font: var(--type-meta);
    white-space: nowrap;
    vertical-align: middle;
  }
  .base-tag.plain {
    flex-shrink: 0;
    margin: 0;
    padding: 0;
    border-radius: 0;
    background: none;
    color: var(--ink3);
  }
</style>
