<script lang="ts">
  // The Units room: the model's neurons and heads as an atlas, searchable by what they write. It
  // fills in when the engine has units; until then it is an empty state.
  import { goto } from "../../lib/stores/rooms.svelte";
</script>

<div class="dw-room">
  <div class="head-block">
    <div class="dw-crumbs">
      <button type="button" onclick={() => goto("workbench")}>Workbench</button><span aria-hidden="true">›</span>
      <span class="dw-here">Units</span>
    </div>
    <h2 class="dw-title">Units</h2>
  </div>
  <div class="dw-empty">
    <span class="head">Neurons and heads</span>
    <p>Every neuron and head of the model, by layer, searchable by the words it writes. They arrive with the engine.</p>
    <div class="dw-blocks atlas" aria-hidden="true"><i></i><i></i><i></i></div>
    <button type="button" class="dw-quiet" onclick={() => goto("workbench")}>Back to the workbench</button>
  </div>
</div>

<style>
  .head-block { display: grid; gap: var(--s1); }
  .atlas { width: 100%; }
  .atlas > i { background: var(--wash2); }
</style>
