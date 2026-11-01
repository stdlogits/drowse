<script lang="ts">
  // The Artifacts room (#/artifacts): where what the dashboard reads and steers with is fitted,
  // fetched and installed. Its sections (ROOM_SECTIONS.artifacts) are the neuron-lens unit catalog,
  // the lenses, the manifold tools (build, packs, combine, templates, roles) and the SAEs. The
  // manifold tools are hosted drawers; units, lenses and SAEs are this room's panels. The unit
  // catalog shows only on a runtime that reads units, as the Units room does.
  import type { Component } from "svelte";
  import RoomSections from "./RoomSections.svelte";
  import UnitCatalog from "./artifacts/UnitCatalog.svelte";
  import LensSources from "./artifacts/LensSources.svelte";
  import SaeSources from "./artifacts/SaeSources.svelte";
  import { runtimeClient } from "../../lib/runtime/client";
  import { unitReadsAvailable } from "../../lib/runtime/contracts";

  const panels: Readonly<Record<string, Component>> = {
    ...(unitReadsAvailable(runtimeClient) ? { units: UnitCatalog } : {}),
    lenses: LensSources,
    saes: SaeSources,
  };
</script>

<RoomSections room="artifacts" {panels} />
