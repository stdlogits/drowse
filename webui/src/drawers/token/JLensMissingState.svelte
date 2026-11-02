<script lang="ts">
  // No lens for this model yet: the Artifacts room fits or fetches one (#/artifacts/lenses).
  // drowse.ai swaps in its own (@runtime-jlens-missing), with the same prop.
  import { goto } from "../../lib/stores/rooms.svelte";
  import EmptyState from "./EmptyState.svelte";

  let { modelId }: { modelId: string | null } = $props();
</script>

<EmptyState title={modelId ? `No lens for ${modelId.split("/").at(-1)}` : "No lens yet"}>
  <button type="button" class="dw-quiet" onclick={() => goto("artifacts", "lenses")}>Choose a lens</button>
</EmptyState>
