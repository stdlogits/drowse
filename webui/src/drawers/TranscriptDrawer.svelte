<script lang="ts">
  import { Switch } from "@stdlogits/ledger";
  import { MorphText } from "@stdlogits/ledger";
  import { IconButton } from "@stdlogits/ledger";
  // Transcript export / import drawer — phase 5.  Two tabs:
  //
  //   * export — render the path ending at the chosen node as
  //     transcript YAML and offer it as a .yaml download.  Defaults
  //     to the active node; the user can pick any node id via a
  //     short id-prefix.
  //   * import — paste YAML or upload a file, pick a mode
  //     (default / here / merge), tick strict, fire the load.
  //     Guard warnings (model / system-prompt / probe drift) surface
  //     in a banner with the diff list.

  import { apiTree, describeError } from "../lib/runtime/services";
  import {
    closeDrawer,
    loomTree,
    refreshLoomTree,
  } from "../lib/stores.svelte";
  import Radio from "../lib/Radio.svelte";

  type Tab = "export" | "import";
  let tab: Tab = $state("export");

  function selectTab(next: Tab): void {
    tab = next;
  }

  function onTabKeydown(ev: KeyboardEvent): void {
    const tabs = ["export", "import"] as const;
    const current = tabs.indexOf(tab);
    let next = current;
    if (ev.key === "ArrowRight") next = (current + 1) % tabs.length;
    else if (ev.key === "ArrowLeft") next = (current - 1 + tabs.length) % tabs.length;
    else if (ev.key === "Home") next = 0;
    else if (ev.key === "End") next = tabs.length - 1;
    else return;
    ev.preventDefault();
    selectTab(tabs[next]);
    document.getElementById(`transcript-${tabs[next]}-tab`)?.focus();
  }

  // -------------------------------------------------- export state ---

  let exportTargetId = $state(""); // empty = active node
  let exportYaml = $state("");
  let exportError: string | null = $state(null);
  let exportBusy = $state(false);
  let exportLeafId: string | null = $state(null);

  async function runExport(): Promise<void> {
    if (exportBusy) return;
    exportBusy = true;
    exportError = null;
    exportYaml = "";
    exportLeafId = null;
    try {
      const id = exportTargetId.trim();
      let resolved: string | null = null;
      if (id) {
        const r = resolveByPrefix(id);
        if (r.id) {
          resolved = r.id;
        } else if (r.matches.length === 0) {
          exportError = `no node matches prefix "${id}"`;
          return;
        } else {
          const preview = r.matches.slice(0, 6).map((s) => s.slice(0, 8)).join(", ");
          exportError =
            `ambiguous: ${r.matches.length} matches (${preview}` +
            (r.matches.length > 6 ? ", …" : "") + ")";
          return;
        }
      }
      const r = await apiTree.transcriptExport(resolved);
      exportYaml = r.yaml;
      exportLeafId = r.node_id;
    } catch (e) {
      exportError = describeError(e);
    } finally {
      exportBusy = false;
    }
  }

  function downloadYaml(): void {
    if (!exportYaml) return;
    const blob = new Blob([exportYaml], {
      type: "application/yaml;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const tsId = exportLeafId ? exportLeafId.slice(0, 8) : "active";
    a.download = `drowse-transcript-${tsId}.yaml`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  let copied = $state(false);
  $effect(() => { if (!copied) return; const timer = setTimeout(() => copied = false, 1800); return () => clearTimeout(timer); });
  async function copyYaml(): Promise<void> {
    if (!exportYaml) return;
    try {
      await navigator.clipboard.writeText(exportYaml);
      copied = true;
    } catch {
      exportError = "Could not copy. Select the YAML above and copy it manually.";
    }
  }

  interface PrefixResolution {
    /** Single unambiguous match — caller can use immediately. */
    id: string | null;
    /** All node ids whose ulid starts with the prefix; ``length > 1``
     *  signals ambiguity. */
    matches: string[];
  }

  function resolveByPrefix(prefix: string): PrefixResolution {
    const p = prefix.trim();
    if (!p) return { id: null, matches: [] };
    if (p === "root") {
      const root = loomTree.root_id;
      return root ? { id: root, matches: [root] } : { id: null, matches: [] };
    }
    const matches: string[] = [];
    for (const id of loomTree.nodes.keys()) {
      if (id === p) return { id, matches: [id] };
      if (id.startsWith(p)) matches.push(id);
    }
    if (matches.length === 1) return { id: matches[0], matches };
    return { id: null, matches };
  }

  // -------------------------------------------------- import state ---

  let importYaml = $state("");
  let importMode: "default" | "here" | "merge" = $state("default");
  let importStrict = $state(false);
  let importError: string | null = $state(null);
  let importBusy = $state(false);
  let importGuards: string[] = $state([]);
  let importLeafId: string | null = $state(null);

  let fileInputRef: HTMLInputElement | null = $state(null);

  async function onFileChange(ev: Event): Promise<void> {
    const target = ev.target as HTMLInputElement;
    const file = target.files?.[0];
    if (!file) return;
    importYaml = await file.text();
  }

  async function runImport(): Promise<void> {
    if (importBusy) return;
    const yaml = importYaml.trim();
    if (!yaml) {
      importError = "paste a transcript YAML or upload a file";
      return;
    }
    importBusy = true;
    importError = null;
    importGuards = [];
    importLeafId = null;
    try {
      const r = await apiTree.transcriptLoad(yaml, importMode, importStrict);
      importGuards = r.guards;
      importLeafId = r.leaf_id;
      // Refresh the tree so the new branch shows up in the sidebar.
      await refreshLoomTree();
    } catch (e) {
      importError = describeError(e);
    } finally {
      importBusy = false;
    }
  }
</script>

<section class="drawer-shell" aria-label="Transcript drawer">
  <header class="header">
    <h2 class="dw-title">Transcript</h2>
    <IconButton icon="close" label="Close" onclick={closeDrawer} />
  </header>
  <div class="tabs" role="tablist" aria-label="Transcript action">
    <button
      id="transcript-export-tab"
      type="button"
      class="tab"
      role="tab"
      aria-selected={tab === "export"}
      aria-controls="transcript-export-panel"
      tabindex={tab === "export" ? 0 : -1}
      onclick={() => selectTab("export")}
      onkeydown={onTabKeydown}
    >Export</button>
    <button
      id="transcript-import-tab"
      type="button"
      class="tab"
      role="tab"
      aria-selected={tab === "import"}
      aria-controls="transcript-import-panel"
      tabindex={tab === "import" ? 0 : -1}
      onclick={() => selectTab("import")}
      onkeydown={onTabKeydown}
    >Import</button>
  </div>

  <div class="body">
    {#if tab === "export"}
      <div
        id="transcript-export-panel"
        class="tab-panel"
        role="tabpanel"
        aria-labelledby="transcript-export-tab"
        tabindex="0"
      >
      <label class="field">
        <span class="meta">End at</span>
        <input
          type="text"
          bind:value={exportTargetId}
          placeholder={`the active turn, ${loomTree.active_node_id?.slice(0, 12) ?? "none"}`}
          autocomplete="off"
          spellcheck="false"
        />
      </label>

      <div class="dw-row">
        <button type="button" class="dw-button" onclick={runExport} disabled={exportBusy}
        ><MorphText text={exportBusy ? "Preparing…" : "Preview YAML"} /></button>
      </div>

      {#if exportError}
        <p class="error" role="alert">{exportError}</p>
      {/if}

      {#if exportYaml}
        <textarea
          class="yaml"
          readonly
          rows="20"
          value={exportYaml}
          aria-label="Rendered transcript YAML"
        ></textarea>
        <div class="dw-row">
          <button type="button" class="dw-quiet" data-cursor="copy" onclick={copyYaml}><span role="status"><MorphText text={copied ? "Copied" : "Copy"} numbers={false} /></span></button>
          <button type="button" class="dw-quiet" onclick={downloadYaml}>Download .yaml</button>
        </div>
      {/if}
      </div>
    {:else}
      <div
        id="transcript-import-panel"
        class="tab-panel"
        role="tabpanel"
        aria-labelledby="transcript-import-tab"
        tabindex="0"
      >
      <div class="field" role="radiogroup" aria-labelledby="import-mode-label">
        <span class="meta" id="import-mode-label">Place the turns</span>
        <Radio bind:group={importMode} value="default" label="On a new path from the root" />
        <Radio bind:group={importMode} value="here" label="After the active turn" />
        <Radio bind:group={importMode} value="merge" label="After the deepest matching turn" />
      </div>

      <Switch checked={importStrict} onchange={(on) => (importStrict = on)} label="Require matching reading settings" />

      <label class="field">
        <span class="meta">File</span>
        <input
          type="file"
          class="file"
          accept=".yaml,.yml,application/x-yaml,text/yaml"
          bind:this={fileInputRef}
          onchange={onFileChange}
        />
      </label>

      <textarea
        class="yaml"
        rows="14"
        bind:value={importYaml}
        placeholder="Or paste transcript YAML"
        spellcheck="false"
        aria-label="Transcript YAML to import"
      ></textarea>

      <div class="dw-row">
        <button type="button" class="dw-button" onclick={runImport} disabled={importBusy}
        ><MorphText text={importBusy ? "Importing…" : "Import"} /></button>
      </div>

      {#if importError}
        <p class="error" role="alert">{importError}</p>
      {/if}

      {#if importGuards.length > 0}
        <div class="banner">
          <span class="head">Guards, kept as notes on the root</span>
          <ul>
            {#each importGuards as g (g)}
              <li>{g}</li>
            {/each}
          </ul>
        </div>
      {/if}
      {#if importLeafId && !importError}
        <p class="ok">Imported, ending at {importLeafId.slice(0, 12)}</p>
      {/if}
      </div>
    {/if}
  </div>
</section>

<style>
  .drawer-shell { display: flex; flex-direction: column; gap: var(--s3); height: 100%; min-height: 0; padding: var(--s5) var(--s6) 0; color: var(--ink2); font: var(--type-small); }
  .header { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); }
  /* Export and import as Ledger's tabs: text only, the chosen one in ink. */
  .tabs { display: flex; gap: var(--s1) 20px; }
  .tab { height: 32px; padding: 0; border: 0; background: transparent; color: var(--ink3); font: var(--type-control); cursor: pointer; transition: color var(--fast) var(--ease); }
  .tab[aria-selected="true"] { color: var(--ink); }
  @media (hover: hover) { .tab:hover { color: var(--ink); } }
  .body { display: flex; flex: 1 1 auto; flex-direction: column; min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding-bottom: var(--s6); }
  .tab-panel { display: flex; flex-direction: column; gap: var(--s4); }
  .tab-panel:focus { outline: none; }
  .field { display: grid; gap: var(--s1); }
  .field input[type="text"] { width: 100%; height: 40px; padding: 0 var(--s3); border: 0; border-radius: var(--r-col); background: var(--wash); color: var(--ink); font: var(--type-small); }
  .file { color: var(--ink2); font: var(--type-small); }
  .file::file-selector-button { height: 36px; margin-right: var(--s3); padding: 0 var(--s4); border: 0; border-radius: var(--r-pill); background: var(--wash); color: var(--ink); font: var(--type-control); cursor: pointer; transition: background-color var(--fast) var(--ease); }
  .file::file-selector-button:hover { background: var(--wash2); }
  .yaml { width: 100%; min-height: 200px; padding: var(--s3) var(--s4); border: 0; border-radius: var(--r-box); background: var(--wash); color: var(--ink); font: var(--type-small); line-height: 22px; resize: vertical; white-space: pre; overflow-wrap: normal; overflow-x: auto; tab-size: 2; }
  .error, .ok { color: var(--ink); }
  .banner { display: grid; gap: var(--s2); padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .banner ul { display: grid; gap: 2px; }
  @media (max-width: 720px) {
    .drawer-shell { padding: var(--s4) var(--s4) 0; }
    .field input[type="text"], .yaml { font-size: 16px; }
  }
</style>
