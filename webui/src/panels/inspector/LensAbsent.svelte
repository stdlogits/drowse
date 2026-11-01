<script lang="ts">
  // A unit read's half with nothing to read it through: no lens of its family selected ("No R
  // lens"), or one that holds no matrix there ("No R read"), and the one way out, the Artifacts
  // room's lenses. The server's reason (on HTTP it names CLI commands) rides the tip.
  import { Icon } from "@stdlogits/ledger";
  import { goto } from "../../lib/stores/rooms.svelte";

  let { family, reason }: { family: "R" | "S"; reason: string } = $props();
  const missing = $derived(/lens is selected/u.test(reason));
</script>

<span class="absent">
  <span class="meta" data-tip={reason}>{missing ? `No ${family} lens` : `No ${family} read`}</span>
  <button type="button" class="dw-text" onclick={() => goto("artifacts", "lenses")}>Lenses<Icon name="right" /></button>
</span>

<style>
  .absent { display: inline-flex; align-items: center; gap: var(--s2); }
</style>
