<script lang="ts">
  import { onMount as onInterfaceMount } from "svelte";
  import { registerInterfaceController } from "../lib/workspaceController";
  import { manifoldIdentitySchema } from "../lib/interfaceSchemas";
  import { ToolError } from "../lib/webmcp/types";

  // Manifold authoring shell — reached from "+ build manifold" in the
  // rack drawer.
  //
  // Three disjoint authoring paths share this drawer, and the only thing
  // they genuinely share is the identity (namespace / name / description)
  // and the mode tabs; everything below is per-path, so each is its own
  // sibling under ``drawers/manifold/``:
  //
  //   * auto      — DiscoverForm: hand the model a flat concept list, the
  //                 generator writes per-concept corpora, the fitter
  //                 derives coords per-model.  No coords, no domain.
  //   * template  — TemplatedForm: derive a manifold from a standalone
  //                 template (slot + values + multi-turn contexts).  The
  //                 tool for categories one references rather than
  //                 embodies (days, months, …).  Deterministic, no model.
  //   * custom    — AuthoredForm: user-supplied corpora, with coordinates
  //                 either hand-placed on a picked domain or derived by
  //                 the fitter (the ``auto-domain`` switch).
  //
  // All three build the same on-disk manifold artifact; the path shows up
  // as ``manifold.json::fit_mode``.  Inspector + steering are unchanged
  // from there on.
  //
  // The shared field/step/node styling lives in ``manifold/form.css``,
  // scoped under the ``.mb-form`` class on the body below, so the three
  // forms read as one surface without each carrying a copy. In the
  // Manifolds room the room draws the title and the way back; as a
  // dialog (drowse.ai's shell) the tool draws its own head.

  import { IconButton } from "@stdlogits/ledger";
  import { untrack } from "svelte";
  import { Segmented } from "@stdlogits/ledger";
  import { closeDrawer, openDrawer, steerRack, drawerState } from "../lib/stores.svelte";
  import { manifoldJobs } from "../lib/stores/manifoldJobs.svelte";
  import { steeringDrawerFor } from "../lib/manifolds/fitKind";
  import AuthoredForm from "./manifold/AuthoredForm.svelte";
  import DiscoverForm from "./manifold/DiscoverForm.svelte";
  import TemplatedForm from "./manifold/TemplatedForm.svelte";
  import { identitySlugs, type ManifoldIdentity } from "./manifold/shared";
  import "./manifold/form.css";

  let { params }: { params?: unknown } = $props();
  const embedded = $derived((params as { embedded?: unknown } | null)?.embedded === true);
  const returnToToken = $derived((params as { returnToToken?: unknown } | null)?.returnToToken);

  type AuthoringMode = "authored" | "discover" | "templated";
  let authoringMode: AuthoringMode = $state(untrack(() => ["authored", "templated"].includes((params as { mode?: string } | null)?.mode ?? "") ? (params as { mode: AuthoringMode }).mode : "discover"));
  let visited = $state<AuthoringMode[]>([]);
  $effect(() => { if (!visited.includes(authoringMode)) visited = [...visited, authoringMode]; });
  const busy = $derived(manifoldJobs.current !== null && ["running", "waiting", "cancelling"].includes(manifoldJobs.current.status));

  const identity: ManifoldIdentity = $state({
    namespace: "local",
    name: "",
    description: "",
  });

  function cancel(): void {
    if (returnToToken) {
      openDrawer("token_drilldown", returnToToken);
      return;
    }
    closeDrawer();
  }

  function complete(result?: { namespace: string; name: string }): void {
    if (drawerState.open !== "manifold_builder") return;
    const { namespace, name } = result ?? identitySlugs(identity);
    const created = steerRack.catalog.find(row => row.namespace === namespace && row.name === name);
    openDrawer(steeringDrawerFor(created), { returnToToken });
  }

  onInterfaceMount(() => registerInterfaceController("manifold_builder", {
    schema: manifoldIdentitySchema,
    read: () => ({ busy: busy, values: { mode: authoringMode, ...identity } }),
    update: async (change) => {
      if (busy) throw new ToolError("BUSY", "Wait for this interface operation to finish.");
      if (change.mode !== undefined) authoringMode = change.mode as AuthoringMode;
      for (const key of ["namespace", "name", "description"] as const) if (change[key] !== undefined) identity[key] = change[key] as string;
    },
  }));
</script>

<section class="mf-tool" class:dialog={!embedded} aria-label="Build manifold">
  {#if !embedded}
    <header class="mf-head">
      <h2 class="dw-title">Create a concept or scale</h2>
      <IconButton icon="close" label="Close" onclick={cancel} />
    </header>
  {/if}

  <div class="mb-form">
    <!-- Held while a job runs: the fieldset disables its buttons, and form.css dims it. -->
    <fieldset class="mode" disabled={busy}><Segmented
      value={authoringMode}
      onchange={(mode) => { if (!busy) authoringMode = mode; }}
      options={[
        { value: "discover", label: "Generate", hint: "Name two or more contrasting concepts; drowse writes the examples and fits a direction" },
        { value: "templated", label: "Template", hint: "A template's possible answers (days, months, directions) become the examples" },
        { value: "authored", label: "Your examples", hint: "Your own labelled statements, on coordinates you place or a layout drowse finds" },
      ]}
      label="Authoring mode"
    /></fieldset>

    <section class="step" aria-label="Name">
      <h3 class="step-title"><span class="n">1</span>Name</h3>
      <div class="grid2">
        <label class="field">
          <span class="label">Folder</span>
          <input
            type="text"
            class="input"
            bind:value={identity.namespace}
            placeholder="local"
            autocomplete="off"
            spellcheck="false"
            disabled={busy}
          />
        </label>
        <label class="field">
          <span class="label">Name</span>
          <input
            type="text"
            class="input"
            data-identity="name"
            bind:value={identity.name}
            required
            placeholder="pirate"
            autocomplete="off"
            spellcheck="false"
            disabled={busy}
          />
        </label>
      </div>
      <label class="field">
        <span class="label">Description</span>
        <input
          type="text"
          class="input"
          bind:value={identity.description}
          placeholder="What it measures"
          autocomplete="off"
          disabled={busy}
        />
      </label>
    </section>

    {#each visited as mode (mode)}
      <div class="authoring-form" hidden={authoringMode !== mode} inert={authoringMode !== mode}>
        {#if mode === "authored"}<AuthoredForm {identity} oncomplete={complete} />
        {:else if mode === "discover"}<DiscoverForm {identity} oncomplete={complete} />
        {:else}<TemplatedForm {identity} oncomplete={complete} />{/if}
      </div>
    {/each}
  </div>
</section>

<style>
  .mode { display: block; min-width: 0; }
  .authoring-form[hidden] { display: none; }
</style>
