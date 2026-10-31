<script lang="ts">
  import { Icon, IconButton, MorphText, Segmented } from "@stdlogits/ledger";
  // ManifoldPacksDrawer — local manifold catalog and HF search/install.
  //
  // Hosted search accepts only repositories with one immutable, provenance-
  // bound .drowse; the Python runtime also supports legacy folder repos.
  //
  // The rack browser is the active-session surface (steer, probe, fit,
  // delete); this drawer is the catalog surface (list local, browse HF,
  // install). It is reachable from the command palette.

  import { onMount } from "svelte";
  import { onArtifactUpdate } from "../lib/artifactUpdates";
  import { ApiError, apiManifoldInstallStream, apiManifolds } from "../lib/runtime/services";
  import { runtimeClient } from "../lib/runtime/client";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import { runtimeOperationAvailability } from "../lib/runtime/ui-capabilities";
  import {
    closeDrawer,
    probeRack,
    steerRack,
    refreshManifoldList,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { HostedManifoldPackInfo } from "../lib/runtime/contracts";
  import type { ManifoldInfo, RemoteManifoldInfo } from "../lib/types";
  import { manifoldUsage, manifoldUsageMessage } from "../lib/manifolds/selectors";
  import "./manifold/form.css";

  type Tab = "installed" | "search";

  let { params }: { params?: unknown } = $props();
  const embedded = $derived((params as { embedded?: unknown } | null)?.embedded === true);
  let tab: Tab = $state("installed");

  // ----- search-tab state ----------------------------------------------
  let query = $state("");
  let searchResults: RemoteManifoldInfo[] = $state([]);
  let searchLoading = $state(false);
  let searchError: string | null = $state(null);
  let installing: string | null = $state(null);
  // Latest ``progress`` frame of the in-flight install, shown in place of a
  // bare spinner — an HF manifold repo can be hundreds of megabytes.
  let installStage: string | null = $state(null);
  const browserMode = runtimeClient.mode !== "http";
  const artifactAvailability = runtimeOperationAvailability("manifold_artifacts");
  const hfDiscoveryAvailable = !browserMode || artifactAvailability.available;
  let localPacks: HostedManifoldPackInfo[] = $state([]);
  let localPacksLoading = $state(false);
  let archiveBusy: string | null = $state(null);
  let archiveError: string | null = $state(null);
  let archiveProgress: string | null = $state(null);
  let pendingReplacement: File | null = $state(null);
  let confirmLocalDelete: string | null = $state(null);

  // Redo searches 300ms after the user stops typing.
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  function scheduleSearch(): void {
    if (debounceTimer) clearTimeout(debounceTimer);
    const q = query.trim();
    if (!q) {
      searchResults = [];
      searchError = null;
      searchLoading = false;
      return;
    }
    searchLoading = true;
    debounceTimer = setTimeout(() => {
      void runSearch(q);
    }, 300);
  }

  async function runSearch(q: string): Promise<void> {
    try {
      const r = await apiManifolds.search(q, 20);
      searchResults = r.results ?? [];
      searchError = null;
    } catch (e) {
      searchResults = [];
      if (e instanceof ApiError) {
        if (e.status === 503) {
          searchError = browserMode
            ? "Hugging Face search is unavailable. Check your connection and try again."
            : "huggingface_hub isn't installed on the server. Run `pip install -e \".[serve]\"` and restart.";
        } else if (e.status === 502) {
          searchError = userFacingError(
            e,
            "Hugging Face search was interrupted. Check your connection and try again.",
          );
        } else {
          searchError = userFacingError(e, "Unable to search Hugging Face. Try again.");
        }
      } else {
        searchError = userFacingError(e, "Unable to search Hugging Face. Try again.");
      }
    } finally {
      searchLoading = false;
    }
  }

  async function installRow(row: RemoteManifoldInfo): Promise<void> {
    const repository = typeof row.repository === "string"
      ? row.repository
      : `${row.namespace}/${row.name}`;
    const displayTarget = selectorOf(row);
    const requestTarget = typeof row.revision === "string"
      ? `${repository}@${row.revision}`
      : repository;
    installing = displayTarget;
    installStage = null;
    try {
      // Streaming client, like the fit / generate flows: the terminal
      // ``error`` frame surfaces as a plain Error (no HTTP status), so the
      // ApiError branches below only fire for a pre-stream rejection.
      const installed = await apiManifoldInstallStream({ target: requestTarget }, (ev) => {
        if (ev.event === "progress") {
          const progress = ev.data as {
            message?: string;
            phase?: string;
            downloadedBytes?: number;
            totalBytes?: number | null;
          } | null;
          if (progress?.message) {
            installStage = progress.message;
          } else if (progress?.phase === "downloading" && progress.downloadedBytes !== undefined) {
            const downloaded = `${(progress.downloadedBytes / 1_000_000).toFixed(1)} MB`;
            const total = progress.totalBytes
              ? ` of ${(progress.totalBytes / 1_000_000).toFixed(1)} MB`
              : "";
            installStage = `Downloading ${downloaded}${total}`;
          } else if (progress?.phase) {
            installStage = `${progress.phase[0].toUpperCase()}${progress.phase.slice(1)} archive`;
          }
        }
      });
      await refreshManifoldList();
      tab = "installed";
      pushToast(`installed ${installed.namespace}/${installed.name}`, { kind: "info" });
    } catch (e) {
      if (e instanceof ApiError) {
        if (e.status === 503) {
          pushToast(
            browserMode
              ? "Hugging Face install is unavailable. Check your connection and try again."
              : "huggingface_hub isn't installed on the server",
            { kind: "error", ttlMs: null },
          );
        } else if (e.status === 502) {
          pushToast(userFacingError(
            e,
            "The Hugging Face download was interrupted. Check your connection and try again.",
          ), {
            kind: "error", ttlMs: null,
          });
        } else if (e.status === 409) {
          pushToast(`${displayTarget} is already installed`, {
            kind: "error", ttlMs: null,
          });
        } else {
          pushToast(userFacingError(
            e,
            `Unable to install ${displayTarget}. Check the pack and try again.`,
          ), {
            kind: "error", ttlMs: null,
          });
        }
      } else {
        pushToast(userFacingError(
          e,
          `Unable to install ${displayTarget}. Check the pack and try again.`,
        ), {
          kind: "error", ttlMs: null,
        });
      }
    } finally {
      installing = null;
      installStage = null;
    }
  }

  async function refreshLocalPacks(): Promise<void> {
    if (!browserMode || !artifactAvailability.available) return;
    localPacksLoading = true;
    try {
      localPacks = (await apiManifolds.drowseArchiveList()).packs;
      archiveError = null;
    } catch (error) {
      archiveError = describe(error);
    } finally {
      localPacksLoading = false;
    }
  }

  async function importArchive(file: File, force = false): Promise<void> {
    if (!file.name.toLowerCase().endsWith(".drowse")) {
      archiveError = "Choose a .drowse file.";
      return;
    }
    archiveBusy = "import";
    archiveError = null;
    archiveProgress = "Inspecting archive";
    try {
      const installed = await apiManifolds.drowseArchiveInstall(
        file,
        { force },
        (event) => {
          if (!event.data || typeof event.data !== "object") return;
          const progress = event.data as { phase?: string; path?: string | null };
          archiveProgress = progress.path
            ? `${progress.phase ?? "Verifying"}: ${progress.path}`
            : progress.phase ?? "Verifying archive";
        },
      );
      pendingReplacement = null;
      await refreshLocalPacks();
      await refreshManifoldList();
      pushToast(`installed ${installed.namespace}/${installed.name}`, { kind: "info" });
    } catch (error) {
      const message = describe(error);
      archiveError = message;
      pendingReplacement = /already installed/i.test(message) && !force ? file : null;
    } finally {
      archiveBusy = null;
      archiveProgress = null;
    }
  }

  let exportedPack = $state<string | null>(null);
  $effect(() => { if (!exportedPack) return; const timer = setTimeout(() => exportedPack = null, 1800); return () => clearTimeout(timer); });
  async function exportArchive(pack: HostedManifoldPackInfo): Promise<void> {
    archiveBusy = `export:${pack.id}`;
    archiveError = null;
    try {
      const archive = await apiManifolds.drowseArchiveExport(pack.id);
      const url = URL.createObjectURL(archive);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${pack.name}.drowse`;
      link.click();
      exportedPack = pack.id;
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (error) {
      archiveError = describe(error);
    } finally {
      archiveBusy = null;
    }
  }

  async function deleteArchive(pack: HostedManifoldPackInfo): Promise<void> {
    const guard = manifoldUsageMessage(
      pack,
      manifoldUsage(pack, steerRack.entries, probeRack.entries),
    );
    if (guard) {
      archiveError = guard;
      return;
    }
    archiveBusy = `delete:${pack.id}`;
    archiveError = null;
    try {
      await apiManifolds.drowseArchiveDelete(pack.id);
      confirmLocalDelete = null;
      await refreshLocalPacks();
      await refreshManifoldList();
    } catch (error) {
      archiveError = describe(error);
    } finally {
      archiveBusy = null;
    }
  }

  function describe(error: unknown): string {
    return userFacingError(
      error,
      "Unable to update local direction packs. Check the file and try again.",
    );
  }

  function selectTab(next: Tab): void {
    tab = next;
  }

  // ----- installed-tab state -------------------------------------------
  function selectorOf(row: ManifoldInfo | RemoteManifoldInfo): string {
    if ("repository" in row && typeof row.repository === "string") return row.repository;
    return `${row.namespace}/${row.name}`;
  }
  function fitBadge(row: ManifoldInfo | RemoteManifoldInfo): string | null {
    if (row.fit_mode && row.fit_mode !== "authored") return row.fit_mode;
    return null;
  }

  // A row's depth strip: twelve cells down the ramp, each its depth's colour, filled when the pack
  // is fitted for this model (the catalog says whether, not at which layers).
  const DEPTH = Array.from({ length: 12 }, (_, i) => Math.round(100 - (i / 11) * 100));
  function fitState(row: ManifoldInfo): { fitted: boolean; word: string; tip: string } {
    const others = row.fitted_models.length - (row.fitted_for_session ? 1 : 0);
    const elsewhere = others > 0 ? ` · fitted for ${others} other model${others === 1 ? "" : "s"}` : "";
    if (row.fitted_for_session && row.stale) return { fitted: true, word: "stale", tip: `Fitted for this model, but stale: refit it${elsewhere}` };
    if (row.fitted_for_session) return { fitted: true, word: "", tip: `Fitted for this model${elsewhere}` };
    if (row.stale) return { fitted: false, word: "stale", tip: `Stale: refit it${elsewhere}` };
    return { fitted: false, word: "not fitted", tip: `Not fitted for this model${elsewhere}` };
  }

  // The store keeps the local catalog hot — refresh on mount in case
  // we landed here without having visited ManifoldDrawer first.
  onMount(() => onArtifactUpdate(async (area) => {
    if (area === "manifolds") await refreshLocalPacks();
  }));

  onMount(() => {
    void refreshManifoldList();
    void refreshLocalPacks();
  });
</script>

<section class="mf-tool" class:dialog={!embedded} aria-label="Downloaded response controls">
  {#if !embedded}
    <header class="mf-head">
      <h2 class="dw-title">Packs</h2>
      <IconButton icon="close" label="Close" onclick={closeDrawer} />
    </header>
  {/if}

  <div class="mf-sec-head views">
    {#if hfDiscoveryAvailable}
      <Segmented label="Manifold pack views" value={tab} onchange={(next) => selectTab(next)}
        options={[{ value: "installed", label: "Installed" }, { value: "search", label: "Hugging Face" }]} />
    {/if}
    <span class="meta catalog-count"><MorphText text={tab === "installed" ? `${steerRack.catalog.length} installed` : !query.trim() ? "" : `${searchResults.length} found`} /></span>
  </div>

  <div class="panel" id="packs-panel">
    {#if tab === "installed"}
      {#if browserMode}
        <section class="mf-sec" aria-labelledby="archive-tools-title">
          <div class="mf-sec-head">
            <h3 id="archive-tools-title" class="head" data-tip="Checksums verify archive integrity, not the publisher’s identity">Portable files</h3>
            {#if artifactAvailability.available}
              <label class="dw-quiet file-action" class:disabled={archiveBusy !== null}>
                <MorphText text={archiveBusy === "import" ? "Verifying…" : "Import pack"} />
                <input
                  type="file"
                  accept=".drowse,application/zip"
                  disabled={archiveBusy !== null}
                  onchange={(event) => {
                    const input = event.currentTarget;
                    const file = input.files?.[0];
                    if (file) void importArchive(file);
                    input.value = "";
                  }}
                />
              </label>
            {/if}
          </div>
          {#if !artifactAvailability.available}
            <p class="meta">{artifactAvailability.reason}</p>
          {:else if archiveProgress}
            <p class="meta stage" aria-live="polite"><MorphText text={archiveProgress.split(": ")[0]} numbers={false} />{#if archiveProgress.includes(": ")} · {archiveProgress.slice(archiveProgress.indexOf(": ") + 2)}{/if}</p>
          {/if}
          {#if artifactAvailability.available && archiveError}<p class="mf-error" role="alert">{archiveError}</p>{/if}
          {#if artifactAvailability.available && pendingReplacement}
            <div class="mf-wash" role="alert">
              <p>This replaces the installed pack and its templates. The current files stay until the replacement passes verification.</p>
              <div class="mf-actions">
                <button type="button" class="dw-button" disabled={archiveBusy !== null} onclick={() => void importArchive(pendingReplacement!, true)}>Replace installed pack</button>
                <button type="button" class="dw-text" disabled={archiveBusy !== null} onclick={() => (pendingReplacement = null)}>Keep current pack</button>
              </div>
            </div>
          {/if}
          {#if artifactAvailability.available && localPacksLoading}
            <div class="dw-blocks" role="status" aria-label="Loading portable packs"><i></i></div>
          {:else if artifactAvailability.available && localPacks.length > 0}
            <ul class="mf-rows" role="list">
              {#each localPacks as pack, i (pack.id)}
                <li class="mf-row stack rise" style:--i={i}>
                  <span class="mf-id">
                    <span class="mf-name">{pack.namespace}/{pack.name}</span>
                    <span class="meta">{pack.source.repository ? `${pack.source.repository}@${pack.source.revision}` : pack.source.uri} · publisher unverified</span>
                  </span>
                  <span class="mf-end">
                    {#if confirmLocalDelete === pack.id}
                      <span class="meta" data-tip="Export a backup first if you want to keep it">Delete from this device?</span>
                      <button type="button" class="dw-quiet" disabled={archiveBusy !== null} onclick={() => void deleteArchive(pack)}>Delete pack</button>
                      <button type="button" class="dw-text" disabled={archiveBusy !== null} onclick={() => (confirmLocalDelete = null)}>Cancel</button>
                    {:else}
                      <button type="button" class="dw-text" disabled={archiveBusy !== null} onclick={() => void exportArchive(pack)}><MorphText text={archiveBusy === `export:${pack.id}` ? "Preparing…" : exportedPack === pack.id ? "Download started" : "Export"} numbers={false} /></button>
                      <button type="button" class="dw-text" disabled={archiveBusy !== null} onclick={() => (confirmLocalDelete = pack.id)}>Delete</button>
                    {/if}
                  </span>
                </li>
              {/each}
            </ul>
          {/if}
        </section>
      {/if}
      <section class="mf-sec" aria-label="Installed manifolds">
        {#if browserMode}<h3 class="head">Installed</h3>{/if}
        {#if steerRack.loading && steerRack.catalog.length === 0}
          <div class="dw-blocks" role="status" aria-label="Loading manifolds"><i></i><i></i><i></i></div>
        {:else if steerRack.catalog.length === 0}
          <p class="meta">
            {artifactAvailability.available
              ? browserMode
                ? "Nothing installed yet. Import a .drowse or search Hugging Face."
                : "Nothing installed yet. Search Hugging Face for a pack."
              : "Nothing installed yet."}
          </p>
        {:else}
          <ul class="mf-rows" role="list">
            {#each steerRack.catalog as m, i (selectorOf(m))}
              {@const key = selectorOf(m)}
              {@const badge = fitBadge(m)}
              {@const fit = fitState(m)}
              <li class="mf-row rise" style:--i={i} {...{ "aria-description": (m.description || key) }}>
                <span class="mf-id">
                  <span class="mf-name">{key}</span>
                  <span class="meta">{m.domain_label} · {m.node_count} nodes{badge ? ` · ${badge}` : ""}</span>
                </span>
                <span class="mf-end">
                  <span class="meta">{fit.word}</span>
                  <span class="mf-depth" class:is-fitted={fit.fitted} class:is-stale={m.stale} role="img" aria-label={fit.tip} data-tip={fit.tip}>
                    {#each DEPTH as p (p)}<i style:--p="{p}%"></i>{/each}
                  </span>
                </span>
              </li>
            {/each}
          </ul>
        {/if}
      </section>
    {:else}
      <section class="mf-sec" aria-label="Search Hugging Face">
        <label class="mf-search">
          <Icon name="search" />
          <span class="sr-only">Search query</span>
          <input
            type="search"
            placeholder="Search Hugging Face"
            aria-label="Search HF for drowse-manifold repos"
            autocomplete="off"
            spellcheck="false"
            bind:value={query}
            oninput={scheduleSearch}
          />
        </label>
        {#if !query.trim()}
          <p class="meta">Public, browser-compatible drowse packs</p>
        {:else if searchLoading}
          <div class="dw-blocks" role="status" aria-label="Searching"><i></i><i></i></div>
        {:else if searchError}
          <p class="mf-error" role="alert">{searchError}</p>
        {:else if searchResults.length === 0}
          <p class="meta">Nothing matches “{query.trim()}”</p>
        {:else}
          <ul class="mf-rows" role="list">
            {#each searchResults as row, i (selectorOf(row))}
              {@const target = selectorOf(row)}
              {@const inFlight = installing === target}
              {@const badge = fitBadge(row)}
              <li class="mf-row stack rise" style:--i={i} {...{ "aria-description": (row.description || target) }}>
                <span class="mf-id">
                  <span class="mf-name">{target}</span>
                  <span class="meta">{row.domain_label} · {row.node_count} nodes{badge ? ` · ${badge}` : ""}{row.tensor_models.length > 0 ? ` · ${row.tensor_models.length} fit${row.tensor_models.length === 1 ? "" : "s"}` : ""}</span>
                  {#if inFlight && installStage}
                    <span class="meta stage" aria-live="polite"><MorphText text={installStage} /></span>
                  {/if}
                </span>
                <span class="mf-end">
                  <button
                    type="button"
                    class="dw-quiet"
                    disabled={inFlight}
                    onclick={() => void installRow(row)}
                    {...{ "aria-description": (`install ${target}`) }}
                  ><MorphText text={inFlight ? "Installing…" : "Install"} /></button>
                </span>
              </li>
            {/each}
          </ul>
        {/if}
      </section>
    {/if}
  </div>
</section>

<style>
  .views { align-items: center; min-height: 40px; }
  .catalog-count { margin-left: auto; }
  .panel { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s6); min-width: 0; max-width: 760px; }
  .stage { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .file-action { position: relative; overflow: hidden; }
  .file-action.disabled { opacity: 0.4; cursor: default; }
  .file-action input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .file-action.disabled input { cursor: default; }
  .file-action:focus-within { outline: 2px solid var(--ink); outline-offset: 2px; }
  @media (max-width: 520px) {
    .mf-row.stack { grid-template-columns: minmax(0, 1fr); }
    .stack .mf-end { justify-content: flex-start; }
  }
</style>
