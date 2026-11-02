<script lang="ts">
  // The Lens room's four lenses with drowse's local fit: LensQuad (the 2x2, which drowse.ai's hosted
  // section renders too) with LensFit columns, each family with its fit's progress, error and form.
  // drowse.ai swaps this module for its hosted section through the @runtime-jlens-source alias, so
  // LensFit and the lens fit job (lib/stores/instrumentAuthoring) reach the Lens room only here.
  import LensFit from "./rooms/lens/LensFit.svelte";
  import LensQuad from "./rooms/lens/LensQuad.svelte";
  import { runtimeOperationAvailability } from "../lib/runtime/ui-capabilities";
  import { lensFit } from "../lib/stores/instrumentAuthoring.svelte";

  /** The fit's status, which LensQuad checks on mount before the fetch's and the sources. */
  function checkFit(): void {
    if (runtimeOperationAvailability("jlens_fitting").available) void lensFit.check();
  }
</script>

<LensQuad column={LensFit} check={checkFit} />
