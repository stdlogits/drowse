<script lang="ts">
  import { Icon } from "@stdlogits/ledger";
  import { Expander } from "@stdlogits/ledger";
  import { expand } from "@stdlogits/ledger";
  import { IconButton } from "@stdlogits/ledger";
  import Button from "../lib/ui/Button.svelte";
  // Shared rack drawer — one component, two reskins split by geometry
  // family.  Subsumes the former VectorsDrawer + ManifoldDrawer: same
  // layout, same actions, same chrome, differing only by accent colour,
  // header label, and which manifolds the catalog filter admits.
  //
  //   * subspace (white ``--accent``) — every flat affine fit
  //     (``fit_mode`` ``pca`` or ``baked``): 2-node concept axes AND
  //     higher-rank flats like ``personas``.
  //   * manifold (the depth tint) — curved fits only
  //     (``fit_mode`` ``spectral`` or ``authored``), e.g. ``emotions``.
  //
  // Within the family filter the existing fitted / unfitted split
  // (``fitted_for_session``) applies, plus optional category grouping and
  // a free-text search over manifolds *and* their node labels.  Two
  // sections:
  //
  //   * Fitted    — has a baked tensor for the loaded model.  Per row:
  //                 [ⓘ] [+steer] [+probe] [re-fit] [delete].
  //   * Unfitted  — node corpus on disk but no tensor for this model.
  //                 Per row: [fit] [delete].
  //
  // Nothing here is family-specific beyond the catalog filter and the
  // chrome: the top "+ …" launcher opens the manifold builder for both
  // families (see ``launcherLabel`` below).

  import { onMount } from "svelte";
  import { SvelteMap, SvelteSet } from "svelte/reactivity";
  import { apiManifolds, apiManifoldFitStream, describeError } from "../lib/runtime/services";
  import { isFittingCancellation } from "../lib/runtime/fittingCancellation";
  import {
    addManifoldToRack,
    addSubspaceToRack,
    attachProbe,
    closeDrawer,
    drawerState,
    probeRack,
    steerRack,
    openDrawer,
    refreshManifoldList,
    refreshProbeList,
    refreshVectorList,
    setSteerLabel,
  } from "../lib/stores.svelte";
  import {
    pushToast,
  } from "../lib/stores/toasts.svelte";
  import type { ManifoldInfo } from "../lib/types";
  import {
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    DEFAULT_EXPANDED,
    categoryOf,
    type Category,
  } from "../lib/concepts";
  import DiagnosticsPanel from "../lib/manifolds/DiagnosticsPanel.svelte";
  import { fitGeometry, isDiscoverFit } from "../lib/manifolds/fitKind";
  import { drawerAvailability, runtimeOperationAvailability } from "../lib/runtime/ui-capabilities";
  import { runtimeClient } from "../lib/runtime/client";
  import Select from "../lib/Select.svelte";
  import {
    manifoldSelectorOptions,
    manifoldUsage,
    manifoldUsageMessage,
    probeReferencesManifold,
    selectedManifoldOption,
    selectorReferencesManifold,
    type ManifoldSelectorOption,
  } from "../lib/manifolds/selectors";
  import { sessionState } from "../lib/stores/session.svelte";

  // ``params`` carries ``{ family, seed_a? }``; typed ``unknown`` so it
  // round-trips through ``drawerState.params`` (loosely typed so each
  // drawer owns its own shape).  ``family`` is derived defensively.  The
  // drawer is one steer+probe browser — every fitted row carries both
  // +steer and +probe, so there is no steer-vs-probe mode split.
  let { params }: { params?: unknown } = $props();
  const returnToToken = $derived((params as { returnToToken?: unknown } | null)?.returnToToken);

  function finishProbeSetup(target: unknown): void {
    if (target) openDrawer("token_drilldown", target);
    else closeDrawer();
  }

  const family = $derived.by<"subspace" | "manifold">(() => {
    if (
      params && typeof params === "object"
      && (params as { family?: unknown }).family === "manifold"
    ) {
      return "manifold";
    }
    return "subspace";
  });

  const title = $derived(returnToToken ? "Add a probe" : family === "manifold" ? "Manifolds" : "Concepts");
  // Authoring always routes to the manifold builder — a flat (2-node /
  // personas) fit is just a pca manifold, so there is no separate vector
  // extraction form.  Both families' launcher opens the same builder.
  const launcherLabel = $derived(returnToToken ? "Create a concept" : "Build a manifold");
  const launcherHint = $derived(returnToToken ? "train one for this model" : "a domain and its nodes");
  const authoringAvailable = drawerAvailability("manifold_builder").available;
  const fittingAvailability = runtimeOperationAvailability("fitting");
  const browserMode = runtimeClient.mode !== "http";

  let errorMsg: string | null = $state(null);
  let mounted = false;

  const busyKeys = new SvelteSet<string>();
  const confirmKeys = new SvelteSet<string>();
  const confirmTimers = new Map<string, number>();

  // Per-row inspector state.  The list endpoint omits per-tensor
  // diagnostics (the JSON would balloon on big heaps), so the inspector
  // lazily GETs the detail shape the first time the user expands a row.
  const inspectKeys = new SvelteSet<string>();
  const detailCache = new SvelteMap<string, ManifoldInfo>();
  const detailLoading = new SvelteSet<string>();
  const detailErrors = new SvelteMap<string, string>();
  const detailRevisions = new Map<string, number>();

  // Free-text filter over manifolds *and* their node labels.
  // ``searching`` auto-expands every node list + category so matches are
  // visible without a manual unfold.
  let query: string = $state("");
  let searchInputRef: HTMLInputElement | null = $state(null);

  // Per-row node-list disclosure.  Keep rows compact by default: opening a
  // category containing ``personas`` must not immediately render 107 disabled
  // controls and bury the next manifold several screens away.  Search still
  // auto-expands matching node lists so discovery remains one-step.
  const expandedNodes = new SvelteSet<string>();
  const selectedSelectors = new SvelteMap<string, string>();

  // Section-prefixed category expansion ("ft:register" / "un:register")
  // so the Fitted / Unfitted sections track open state independently.
  const expanded = new SvelteSet<string>(
    [...DEFAULT_EXPANDED].flatMap((c) => [`ft:${c}`, `un:${c}`]),
  );

  onMount(() => {
    mounted = true;
    void refreshManifoldList();
    // Pull the attached-probe list too — the +probe button disables
    // itself when a fitted manifold is already attached, so the drawer
    // needs this fresh on mount.
    void refreshProbeList();
    queueMicrotask(() => searchInputRef?.focus({ preventScroll: true }));
    return () => {
      mounted = false;
      for (const t of confirmTimers.values()) window.clearTimeout(t);
      confirmTimers.clear();
    };
  });

  function rowKey(m: ManifoldInfo): string {
    return `${m.namespace}/${m.name}`;
  }

  function selectorOptions(m: ManifoldInfo): ManifoldSelectorOption[] {
    return manifoldSelectorOptions(m, sessionState.info?.model_id ?? null);
  }

  function selectorChoice(m: ManifoldInfo): ManifoldSelectorOption | null {
    return selectedManifoldOption(selectorOptions(m), selectedSelectors.get(rowKey(m)));
  }

  function chooseSelector(m: ManifoldInfo, selector: string): void {
    selectedSelectors.set(rowKey(m), selector);
  }

  function unavailableSelectorMessage(m: ManifoldInfo): string | null {
    const choice = selectorChoice(m);
    return choice && !choice.available
      ? choice.unavailableReason ?? `The selected fit for ${rowKey(m)} is unavailable.`
      : null;
  }

  const searching = $derived(query.trim().length > 0);

  function nodeMatches(label: string, q: string): boolean {
    return label.toLowerCase().includes(q);
  }

  /** A manifold matches the query on its key/name/namespace/description/
   *  tags OR on any of its node labels — so searching a node name
   *  surfaces the card that owns it. */
  function rowMatches(m: ManifoldInfo, q: string): boolean {
    if (!q) return true;
    const n = q.toLowerCase();
    if (rowKey(m).toLowerCase().includes(n)) return true;
    if (m.name.toLowerCase().includes(n)) return true;
    if (m.namespace.toLowerCase().includes(n)) return true;
    if (m.description && m.description.toLowerCase().includes(n)) return true;
    if (Array.isArray(m.tags)) {
      for (const t of m.tags) {
        if (typeof t === "string" && t.toLowerCase().includes(n)) return true;
      }
    }
    return (m.node_labels ?? []).some((l) => nodeMatches(l, n));
  }

  /** Node labels to render for a row under the active query.  When
   *  searching, narrow to matching labels — but if the row matched only
   *  on its name/description (no node hit), keep the full list so the
   *  card never renders an empty node block. */
  function visibleNodes(m: ManifoldInfo): string[] {
    const labels = m.node_labels ?? [];
    const q = query.trim().toLowerCase();
    if (!q) return labels;
    const hits = labels.filter((l) => nodeMatches(l, q));
    return hits.length > 0 ? hits : labels;
  }

  /** Per-node role slug (role-baselined manifolds), aligned with
   *  ``node_labels``.  ``null`` when the manifold carries no roles. */
  function nodeRoleFor(m: ManifoldInfo, label: string): string | null {
    const roles = m.node_roles;
    if (!roles) return null;
    const idx = (m.node_labels ?? []).indexOf(label);
    return idx >= 0 ? (roles[idx] ?? null) : null;
  }

  function nodesOpen(key: string): boolean {
    return searching || expandedNodes.has(key);
  }
  function toggleNodes(key: string): void {
    if (expandedNodes.has(key)) expandedNodes.delete(key);
    else expandedNodes.add(key);
  }

  // ----- family filter + section split --------------------------------

  /** The family discriminator — subspace admits every flat affine fit
   *  (pca / baked), manifold admits curved fits only (spectral /
   *  authored).
   *
   *  An ``auto`` discover folder resolves its geometry per-model at fit
   *  time, so route it by ``resolved_fit_mode`` once fitted; while it's
   *  unfitted the geometry is unknown (``resolved_fit_mode`` null), so it
   *  belongs to *both* families until a fit pins it — otherwise an
   *  unfitted auto manifold (personas / emotions) shows in neither drawer. */
  function inFamily(m: ManifoldInfo): boolean {
    const geometry = fitGeometry(m);
    return geometry === "unresolved" || geometry === (family === "subspace" ? "flat" : "curved");
  }

  const fitted = $derived(
    steerRack.catalog.filter(
      (m) => inFamily(m) && m.fitted_for_session && rowMatches(m, query.trim()),
    ),
  );
  const unfitted = $derived(
    steerRack.catalog.filter(
      (m) => inFamily(m) && !m.fitted_for_session && rowMatches(m, query.trim()),
    ),
  );

  // Whether *any* row in the family exists at all (drives the empty
  // states vs the "no match" state).
  const familyTotal = $derived(
    steerRack.catalog.filter((m) => inFamily(m)).length,
  );

  // ----- category grouping --------------------------------------------

  function groupedByCategory(rows: ManifoldInfo[]): Map<Category, ManifoldInfo[]> {
    const m = new Map<Category, ManifoldInfo[]>();
    for (const r of rows) {
      // Tags ride the wire now (when the list serializer emits them);
      // missing → "other".
      const c = categoryOf(r.tags);
      const list = m.get(c);
      if (list) list.push(r);
      else m.set(c, [r]);
    }
    for (const list of m.values()) {
      list.sort((a, b) => a.name.localeCompare(b.name));
    }
    return m;
  }

  function sectionsOf(g: Map<Category, ManifoldInfo[]>) {
    const order: Category[] = [...CATEGORY_ORDER, "other"];
    return order
      .filter((c) => (g.get(c)?.length ?? 0) > 0)
      .map((c) => ({ cat: c, items: g.get(c) as ManifoldInfo[] }));
  }

  const ftSections = $derived(sectionsOf(groupedByCategory(fitted)));
  const unSections = $derived(sectionsOf(groupedByCategory(unfitted)));

  function toggleCat(key: string): void {
    if (expanded.has(key)) expanded.delete(key);
    else expanded.add(key);
  }
  function catOpen(key: string): boolean {
    return searching || expanded.has(key);
  }

  // ----- per-row membership -------------------------------------------

  function isRacked(m: ManifoldInfo): boolean {
    return [...steerRack.entries.keys()].some((selector) =>
      selectorReferencesManifold(selector, m)
    );
  }

  function isProbed(m: ManifoldInfo): boolean {
    const choice = selectorChoice(m);
    if (!choice) return false;
    return [...probeRack.entries.values()].some((entry) =>
      probeReferencesManifold(entry.info, m, entry.request.selector) &&
      (entry.info.family !== "geometry" ||
        entry.info.manifold === choice.selector || entry.request.selector === choice.selector)
    );
  }

  function deletionGuard(m: ManifoldInfo): string | null {
    return manifoldUsageMessage(
      m,
      manifoldUsage(m, steerRack.entries, probeRack.entries),
    );
  }

  // ----- steer / node-steer / probe -----------------------------------

  /** Rack the manifold (if not already) and pin it to ``label`` as a
   *  label-form term, then close — the one-click "steer to this node"
   *  affordance.  A second node click on an already-racked manifold just
   *  re-targets the existing term.  Routes by family: subspace (flat) lands
   *  a subspace term at the shared along; manifold (curved) a curved term. */
  function onSteerNode(m: ManifoldInfo, label: string): void {
    if (!m.fitted_for_session) return;
    const choice = selectorChoice(m);
    if (!choice || !choice.available) {
      pushToast(choice?.unavailableReason ?? `No compatible fit is available for ${rowKey(m)}.`, {
        kind: "error",
        ttlMs: null,
      });
      return;
    }
    const key = rowKey(m);
    if (family === "subspace") addSubspaceToRack(key, choice.variant);
    else addManifoldToRack(key, choice.variant);
    setSteerLabel(key, label);
    closeDrawer();
  }

  /** +steer dispatch — flat fits join as a subspace term (magnitude is the
   *  shared "subspace along" master); curved fits join as a manifold term
   *  with their own per-card along/onto.  The drawer ``family`` already
   *  filtered the rows, so it is the discriminator. */
  function onSteer(m: ManifoldInfo): void {
    if (isRacked(m)) return;
    const choice = selectorChoice(m);
    if (!choice || !choice.available) {
      pushToast(choice?.unavailableReason ?? `No compatible fit is available for ${rowKey(m)}.`, {
        kind: "error",
        ttlMs: null,
      });
      return;
    }
    if (family === "subspace") addSubspaceToRack(rowKey(m), choice.variant);
    else addManifoldToRack(rowKey(m), choice.variant);
    closeDrawer();
  }

  async function onProbe(m: ManifoldInfo): Promise<void> {
    if (isProbed(m) || busyKeys.has(rowKey(m))) return;
    const origin = drawerState.params;
    const originDrawer = drawerState.open;
    const returnTarget = returnToToken;
    const choice = selectorChoice(m);
    if (!choice || !choice.available) {
      pushToast(choice?.unavailableReason ?? `No compatible fit is available for ${rowKey(m)}.`, {
        kind: "error",
        ttlMs: null,
      });
      return;
    }
    const key = rowKey(m);
    busyKeys.add(key);
    try {
      const info = await attachProbe(choice.selector);
      pushToast(`probe ${info.name}`, { kind: "info" });
      if (mounted && drawerState.open === originDrawer && drawerState.params === origin) {
        finishProbeSetup(returnTarget);
      }
    } catch (e) {
      pushToast(`attach: ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      busyKeys.delete(key);
    }
  }

  // ----- custom-attach form (attach probe by selector) ----------------

  let customSelector: string = $state("");
  let customAlias: string = $state("");
  let customTopN: number = $state(3);
  let customAttaching: boolean = $state(false);

  async function onCustomAttachSubmit(ev: SubmitEvent): Promise<void> {
    ev.preventDefault();
    if (customAttaching) return;
    const origin = drawerState.params;
    const originDrawer = drawerState.open;
    const returnTarget = returnToToken;
    const sel = customSelector.trim();
    if (!sel) {
      pushToast("selector required", { kind: "error" });
      return;
    }
    if (browserMode) {
      const manifold = steerRack.catalog.find((row) =>
        selectorReferencesManifold(sel, row)
      );
      if (manifold) {
        const options = selectorOptions(manifold);
        const option = options.find((candidate) => candidate.selector === sel);
        const mixedRoleBlock = options.find((candidate) =>
          !candidate.available && candidate.unavailableReason !== null
        );
        if (option && !option.available || !option && mixedRoleBlock) {
          pushToast(option?.unavailableReason ?? mixedRoleBlock!.unavailableReason!, {
            kind: "error",
            ttlMs: null,
          });
          return;
        }
      }
    }
    customAttaching = true;
    try {
      const opts: { name?: string; top_n?: number } = {};
      if (customAlias.trim()) opts.name = customAlias.trim();
      if (customTopN && customTopN > 0) opts.top_n = customTopN;
      const info = await attachProbe(sel, opts);
      pushToast(`probe ${info.name}`, { kind: "info" });
      customSelector = "";
      customAlias = "";
      if (mounted && drawerState.open === originDrawer && drawerState.params === origin && returnTarget) {
        finishProbeSetup(returnTarget);
      }
    } catch (e) {
      pushToast(`attach: ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      customAttaching = false;
    }
  }

  // ----- fit / delete -------------------------------------------------

  function onFit(m: ManifoldInfo): void {
    if (!fittingAvailability.available) {
      pushToast(fittingAvailability.reason ?? "Manifold fitting is unavailable.", {
        kind: "error",
        ttlMs: null,
      });
      return;
    }
    const key = rowKey(m);
    if (busyKeys.has(key)) return;
    busyKeys.add(key);
    errorMsg = null;
    void (async () => {
      try {
        await apiManifoldFitStream(m.namespace, m.name, {}, () => {});
        detailCache.delete(key);
        detailErrors.delete(key);
        detailRevisions.set(key, (detailRevisions.get(key) ?? 0) + 1);
        detailLoading.delete(key);
        await Promise.all([refreshManifoldList(), refreshProbeList(), refreshVectorList()]);
        if (mounted && inspectKeys.has(key)) {
          inspectKeys.delete(key);
          await toggleInspect(m);
        }
        pushToast(`fitted ${key}`, { kind: "info" });
      } catch (e) {
        if (isFittingCancellation(e)) {
          pushToast(`Fitting ${key} cancelled.`, { kind: "info" });
          return;
        }
        pushToast(`Couldn't fit '${key}': ${describeError(e)}`, {
          kind: "error",
          ttlMs: null,
        });
      } finally {
        busyKeys.delete(key);
      }
    })();
  }

  function onDeleteClick(m: ManifoldInfo): void {
    const key = rowKey(m);
    const guard = deletionGuard(m);
    if (guard) {
      pushToast(guard, { kind: "error", ttlMs: null });
      return;
    }
    if (confirmKeys.has(key)) {
      const t = confirmTimers.get(key);
      if (t !== undefined) {
        window.clearTimeout(t);
        confirmTimers.delete(key);
      }
      confirmKeys.delete(key);
      void doDelete(m);
      return;
    }
    confirmKeys.add(key);
    const t = window.setTimeout(() => {
      confirmKeys.delete(key);
      confirmTimers.delete(key);
    }, 3000);
    confirmTimers.set(key, t);
  }

  async function doDelete(m: ManifoldInfo): Promise<void> {
    const key = rowKey(m);
    busyKeys.add(key);
    errorMsg = null;
    try {
      await apiManifolds.delete(m.namespace, m.name);
      await refreshManifoldList();
      // Bundled artifacts (the ``default/`` namespace) respawn on the
      // next session init, so flag that in the toast.
      const msg = m.namespace === "default"
        ? `Deleted ${key}. This bundled item will return on restart.`
        : `deleted ${key}`;
      pushToast(msg, { kind: "info" });
    } catch (e) {
      errorMsg = describeError(e);
    } finally {
      busyKeys.delete(key);
    }
  }

  // ----- launchers + badges -------------------------------------------

  function gotoLauncher(): void {
    openDrawer("manifold_builder", returnToToken ? { returnToToken, mode: "discover" } : null);
  }

  function fitModeBadge(m: ManifoldInfo): string | null {
    if (m.fit_mode && m.fit_mode !== "authored") return m.fit_mode;
    return null;
  }

  /** True iff at least one node carries a non-null role — a persona /
   *  role-paired manifold. */
  function isRoleAugmented(m: ManifoldInfo): boolean {
    return (m.node_roles ?? []).some((r) => r);
  }

  async function toggleInspect(m: ManifoldInfo): Promise<void> {
    const key = rowKey(m);
    if (inspectKeys.has(key)) {
      inspectKeys.delete(key);
      return;
    }
    inspectKeys.add(key);
    if (detailCache.has(key)) return;
    detailLoading.add(key);
    detailErrors.delete(key);
    const revision = (detailRevisions.get(key) ?? 0) + 1;
    detailRevisions.set(key, revision);
    try {
      const detail = await apiManifolds.get(m.namespace, m.name);
      if (!mounted || detailRevisions.get(key) !== revision) return;
      detailCache.set(key, detail);
    } catch (e) {
      if (!mounted || detailRevisions.get(key) !== revision) return;
      detailErrors.set(key, describeError(e));
    } finally {
      if (detailRevisions.get(key) === revision) detailLoading.delete(key);
    }
  }
</script>

<section
  class="drawer-shell"
  class:fam-manifold={family === "manifold"}
  aria-label={family === "manifold" ? "Manifolds" : "Subspaces"}
>
  <header class="header">
    <h2 class="title">{title}</h2>
    {#if returnToToken}<Button onclick={() => finishProbeSetup(returnToToken)}>Back to token</Button>{/if}
    <IconButton icon="close" label="Close" onclick={closeDrawer} />
  </header>

  <div class="body">
    {#snippet nodeSection(m: ManifoldInfo, steerable: boolean)}
      {@const key = rowKey(m)}
      {@const total = (m.node_labels ?? []).length}
      {@const nodes = visibleNodes(m)}
      {#if total > 0}
        <div class="nodes-block">
          <button
            type="button"
            class="nodes-toggle"
            aria-expanded={nodesOpen(key)}
            onclick={() => toggleNodes(key)}
            {...{ "aria-description": (nodesOpen(key) ? "collapse" : "expand") }}
          >
            <span class="caret" class:open={nodesOpen(key)} aria-hidden="true"><Icon name="down" /></span>
            <span class="nodes-label">nodes</span>
            <span class="nodes-count"
              >{nodes.length}{#if nodes.length !== total}/{total}{/if}</span>
          </button>
          {#if nodesOpen(key)}
            <ul class="node-chips" role="list" transition:expand>
              {#each nodes as label (label)}
                {@const role = nodeRoleFor(m, label)}
                <li>
                  <button
                    type="button"
                    class="node-chip"
                    disabled={!steerable}
                    onclick={() => onSteerNode(m, label)}
                    {...{ "aria-description": (steerable
                      ? `steer ${key} → ${label}`
                      : `fit ${key} first to steer to ${label}`) }}
                  >
                    <span class="node-name">{label}</span>
                    {#if role}<span class="node-role" {...{ "aria-description": "role baseline" }}>{role}</span>{/if}
                  </button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>
      {/if}
    {/snippet}

    {#snippet fittedRow(m: ManifoldInfo)}
      {@const key = rowKey(m)}
      {@const busy = busyKeys.has(key)}
      {@const confirming = confirmKeys.has(key)}
      {@const inspecting = inspectKeys.has(key)}
      {@const badge = fitModeBadge(m)}
      {@const options = selectorOptions(m)}
      {@const choice = selectorChoice(m)}
      {@const unavailable = unavailableSelectorMessage(m)}
      {@const deleteGuard = deletionGuard(m)}
      <li class="row" {...{ "aria-description": (m.description || key) }}>
        <div class="row-line">
          <div class="meta">
            <span class="row-name">{key}</span>
            <span class="row-sub">
              {m.domain_label} · {m.node_count} nodes
              {#if badge} · <span class="fit-badge fit-{badge}">{badge}</span>{/if}
              {#if isRoleAugmented(m)} · <span class="fit-badge fit-persona" {...{ "aria-description": "role-paired manifold" }}>persona</span>{/if}
              {#if m.stale} · <span class="stale">stale</span>{/if}
            </span>
          </div>
          <div class="actions">
            <button
              type="button"
              class="act inspect"
              aria-expanded={inspecting}
              aria-label={`${inspecting ? "Hide" : "Show"} details for ${key}`}
              onclick={() => void toggleInspect(m)}
              {...{ "aria-description": (inspecting ? "hide" : "inspect") }}
            ><Icon name={inspecting ? "down" : "right"} /></button>
            <button
              type="button"
              class="act steer"
              disabled={busy || isRacked(m) || choice === null || !choice.available}
              onclick={() => onSteer(m)}
              {...{ "aria-description": (unavailable ?? (isRacked(m)
                ? `${key} already racked`
                : `steer with ${choice?.selector ?? key}`)) }}
            >Steer</button>
            <button
                type="button"
                class="act probe"
                disabled={busy || isProbed(m) || choice === null || !choice.available}
                onclick={() => void onProbe(m)}
                {...{ "aria-description": (unavailable ?? (isProbed(m)
                  ? `${key} already attached`
                  : `probe with ${choice?.selector ?? key}`)) }}
            >Probe</button>
            <button
              type="button"
              class="act fit"
              disabled={busy || !fittingAvailability.available}
              onclick={() => onFit(m)}
              {...{ "aria-description": (fittingAvailability.available
                ? `re-fit ${key}`
                : fittingAvailability.reason ?? "Fitting unavailable") }}
            >{busy ? "…" : "Re-fit"}</button>
            <button
              type="button"
              class="act del"
              class:confirm={confirming}
              disabled={busy}
              onclick={() => onDeleteClick(m)}
              {...{ "aria-description": (deleteGuard ?? (confirming ? "click again to confirm" : `delete ${key}`)) }}
            >{confirming ? "Confirm?" : "Delete"}</button>
          </div>
        </div>
        {#if options.length > 0 && choice}
          <div class="selector-row">
            <span>fit</span>
            <Select
              value={choice.selector}
              options={options.map((option) => ({
                value: option.selector,
                label: option.label,
                disabled: !option.available,
              }))}
              ariaLabel={`Fit for ${key}`}
              title={choice.unavailableReason ?? `Use ${choice.selector}`}
              onchange={(selector) => chooseSelector(m, selector)}
            />
            <span class="sel">{choice.selector}</span>
          </div>
          {#if unavailable}<p class="selector-unavailable" role="status">{unavailable}</p>{/if}
        {/if}
        {#if inspecting}
          <div class="inspect-body" transition:expand>
            {#if detailLoading.has(key)}
              <p class="muted loading-pulse loading-placeholder" role="status">Loading details…</p>
            {:else if detailErrors.has(key)}
              <p class="error">{detailErrors.get(key)}</p>
            {:else if detailCache.has(key)}
              {@const detail = detailCache.get(key)!}
              {#if isDiscoverFit(detail)}
                <DiagnosticsPanel manifold={detail} />
              {:else}
                <p class="muted">{detail.fit_mode} · {detail.domain_label} · dim {detail.intrinsic_dim}</p>
              {/if}
            {/if}
          </div>
        {/if}
        {@render nodeSection(m, choice?.available === true)}
      </li>
    {/snippet}

    {#snippet unfittedRow(m: ManifoldInfo)}
      {@const key = rowKey(m)}
      {@const busy = busyKeys.has(key)}
      {@const confirming = confirmKeys.has(key)}
      {@const badge = fitModeBadge(m)}
      {@const deleteGuard = deletionGuard(m)}
      <li class="row" {...{ "aria-description": (m.description || key) }}>
        <div class="row-line">
          <div class="meta">
            <span class="row-name">{key}</span>
            <span class="row-sub">
              {m.domain_label} · {m.node_count} nodes
              {#if badge} · <span class="fit-badge fit-{badge}">{badge}</span>{/if}
            </span>
          </div>
          <div class="actions">
            <button
              type="button"
              class="act fit"
              disabled={busy || !fittingAvailability.available}
              onclick={() => onFit(m)}
              {...{ "aria-description": (fittingAvailability.available
                ? `fit ${key}`
                : fittingAvailability.reason ?? "Fitting unavailable") }}
            >{busy ? "…" : "Fit"}</button>
            <button
              type="button"
              class="act del"
              class:confirm={confirming}
              disabled={busy}
              onclick={() => onDeleteClick(m)}
              {...{ "aria-description": (deleteGuard ?? (confirming ? "click again to confirm" : `delete ${key}`)) }}
            >{confirming ? "Confirm?" : "Delete"}</button>
          </div>
        </div>
        {@render nodeSection(m, false)}
      </li>
    {/snippet}

    {#if errorMsg}
      <p class="error" role="alert">{errorMsg}</p>
    {/if}

    <div class="custom-attach"><Expander label="Attach a selector">
      <form class="attach-form" onsubmit={onCustomAttachSubmit}>
        <label class="row-label">
          <span>selector</span>
          <input
            type="text"
            class="text-input"
            placeholder="ns/name"
            aria-label="Selector"
            bind:value={customSelector}
            disabled={customAttaching}
          />
        </label>
        <label class="row-label">
          <span>name</span>
          <input
            type="text"
            class="text-input"
            placeholder="name"
            aria-label="Registered probe name (optional)"
            bind:value={customAlias}
            disabled={customAttaching}
          />
        </label>
        <label class="row-label">
          <span>top_n</span>
          <input
            type="number"
            class="text-input small"
            min="1"
            max="32"
            step="1"
            aria-label="Per-token nearest-node list length"
            bind:value={customTopN}
            disabled={customAttaching}
          />
        </label>
        <button
          type="submit"
          class="act probe"
          disabled={customAttaching || !customSelector.trim()}
        >Attach</button>
      </form>
    </Expander></div>

    {#if !returnToToken || authoringAvailable}
    <button type="button" class="build-btn" onclick={gotoLauncher}>
      <span class="build-label">+ {launcherLabel}</span>
      <span class="build-hint">{launcherHint}</span>
    </button>
    {/if}

    {#if familyTotal > 0}
      <div class="search-row">
        <input
          type="search"
          class="search"
          bind:this={searchInputRef}
          bind:value={query}
          placeholder={family === "manifold"
            ? "search manifolds & nodes…"
            : "search subspaces & nodes…"}
          aria-label={family === "manifold"
            ? "Search manifolds and nodes"
            : "Search subspaces and nodes"}
          autocomplete="off"
          spellcheck="false"
        />
        <button
          type="button"
          class="refresh"
          onclick={() => void refreshManifoldList()}
          disabled={steerRack.loading}
          {...{ "aria-description": "refresh" }}
          aria-label="Refresh"
        >Refresh</button>
      </div>
    {/if}

    {#if steerRack.error}
      <p class="error" role="alert">{steerRack.error}</p>
    {/if}

    {#if steerRack.loading && familyTotal === 0}
      <p class="muted loading-pulse loading-placeholder" role="status">Loading controls…</p>
    {:else if familyTotal === 0}
      <p class="muted">{returnToToken ? "No saved concepts in this category yet." : "none"}</p>
    {:else}
      {#if ftSections.length > 0}
        <h3 class="section-title head">Fitted</h3>
        <div class="catalog">
          {#each ftSections as { cat, items } (cat)}
            {@const ckey = `ft:${cat}`}
            {@const open = catOpen(ckey)}
            <section class="category">
              <button
                type="button"
                class="cat-header"
                class:open
                aria-expanded={open}
                onclick={() => toggleCat(ckey)}
                disabled={searching}
              >
                <span class="caret" class:open aria-hidden="true"><Icon name="down" /></span>
                <span class="cat-name">{CATEGORY_LABELS[cat]}</span>
                <span class="cat-count">{items.length}</span>
              </button>
              {#if open}
                <ul class="rows" role="list" aria-label={CATEGORY_LABELS[cat]} transition:expand>
                  {#each items as m (rowKey(m))}
                    {@render fittedRow(m)}
                  {/each}
                </ul>
              {/if}
            </section>
          {/each}
        </div>
      {/if}

      {#if unSections.length > 0}
        <h3 class="section-title head">Unfitted</h3>
        <div class="catalog">
          {#each unSections as { cat, items } (cat)}
            {@const ckey = `un:${cat}`}
            {@const open = catOpen(ckey)}
            <section class="category">
              <button
                type="button"
                class="cat-header"
                class:open
                aria-expanded={open}
                onclick={() => toggleCat(ckey)}
                disabled={searching}
              >
                <span class="caret" class:open aria-hidden="true"><Icon name="down" /></span>
                <span class="cat-name">{CATEGORY_LABELS[cat]}</span>
                <span class="cat-count">{items.length}</span>
              </button>
              {#if open}
                <ul class="rows" role="list" aria-label={CATEGORY_LABELS[cat]} transition:expand>
                  {#each items as m (rowKey(m))}
                    {@render unfittedRow(m)}
                  {/each}
                </ul>
              {/if}
            </section>
          {/each}
        </div>
      {/if}

      {#if fitted.length === 0 && unfitted.length === 0}
        <p class="muted">no matches</p>
      {/if}
    {/if}
  </div>
</section>

<style>
  /* The catalogue of concepts or manifolds, in a dialog: rows on washes, quiet text actions. */
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    color: var(--ink2);
    font: var(--type-small);
  }
  .header {
    display: flex;
    align-items: center;
    gap: var(--s3);
    padding: var(--s5) var(--s5) var(--s3);
  }
  .title {
    flex: 1 1 auto;
    margin: 0;
    color: var(--ink);
    font: var(--type-lead);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--s3) var(--s5) var(--s5);
    display: flex;
    flex-direction: column;
    gap: var(--s4);
    min-height: 0;
  }
  .error {
    margin: 0;
    color: var(--ink);
    font: var(--type-small);
    overflow-wrap: anywhere;
  }
  .muted {
    margin: 0;
    color: var(--ink3);
    font: var(--type-small);
  }

  /* ---- the authoring launcher: a quiet wide button on a wash ---- */
  .build-btn {
    display: flex;
    align-items: baseline;
    gap: var(--s3);
    width: 100%;
    min-height: 44px;
    padding: var(--s3) var(--s4);
    border: 0;
    border-radius: var(--r-box);
    background: var(--wash);
    color: var(--ink);
    font: var(--type-control);
    text-align: start;
    cursor: pointer;
    transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease);
  }
  .build-btn:hover { background: var(--wash2); }
  .build-btn:active:not(:disabled) { scale: 0.97; }
  .build-hint {
    flex: 1 1 auto;
    color: var(--ink3);
    font: var(--type-meta);
    text-align: end;
  }

  /* ---- section heads and the category groups ---- */
  .section-title { margin: var(--s3) 0 0; }
  .catalog {
    display: flex;
    flex-direction: column;
    gap: var(--s1);
  }
  .category {
    display: flex;
    flex-direction: column;
  }
  .cat-header {
    display: flex;
    align-items: center;
    gap: var(--s2);
    width: 100%;
    min-height: 36px;
    padding: 0 var(--s2);
    border: 0;
    border-radius: var(--r-col);
    background: transparent;
    color: var(--ink2);
    font: var(--type-control);
    text-align: start;
    cursor: pointer;
    transition: color var(--fast) var(--ease), background-color var(--fast) var(--ease);
  }
  .cat-header:hover:not(:disabled) { color: var(--ink); background: var(--wash); }
  .cat-header:disabled { cursor: default; }
  .cat-name { flex: 1 1 auto; }
  .cat-header.open .cat-name { color: var(--ink); }
  .cat-count {
    color: var(--ink3);
    font: var(--type-meta);
    font-variant-numeric: tabular-nums;
  }
  .caret {
    display: grid;
    color: var(--ink3);
    rotate: -90deg;
    transition: rotate var(--glide) var(--ease);
  }
  .caret.open { rotate: 0deg; }

  /* ---- rows ---- */
  .rows {
    list-style: none;
    margin: 0;
    padding: var(--s2) 0;
    display: flex;
    flex-direction: column;
    gap: var(--s2);
  }
  .row {
    display: flex;
    flex-direction: column;
    gap: var(--s2);
    padding: var(--s3) var(--s4);
    border-radius: var(--r-box);
    background: var(--wash);
    transition: background-color var(--fast) var(--ease);
  }
  .row-line {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--s3);
  }
  .meta {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }
  .row-name {
    overflow: hidden;
    color: var(--ink);
    font: 600 14px/20px var(--display);
    letter-spacing: -0.01em;
    white-space: nowrap;
    text-overflow: ellipsis;
  }
  .row-sub {
    color: var(--ink3);
    font: var(--type-meta);
    font-variant-numeric: tabular-nums;
  }
  .fit-badge,
  .stale { color: var(--ink2); }
  .inspect-body { padding-top: var(--s2); }

  .actions {
    display: inline-flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    align-items: center;
    gap: var(--s1) var(--s3);
  }
  .act {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 28px;
    padding: 0;
    border: 0;
    background: transparent;
    color: var(--ink3);
    font: var(--type-control);
    cursor: pointer;
    transition: color var(--fast) var(--ease), scale var(--fast) var(--ease);
  }
  .act:hover:not(:disabled) { color: var(--ink); }
  .act:active:not(:disabled) { scale: 0.97; }
  .act:disabled { opacity: 0.4; cursor: default; }
  .act.steer, .act.probe { color: var(--ink2); }
  .act.del.confirm { color: var(--ink); }
  .act.inspect { min-width: 28px; }

  /* ---- the fit a row reads through ---- */
  .selector-row {
    display: grid;
    grid-template-columns: auto minmax(128px, 0.7fr) minmax(0, 1fr);
    align-items: center;
    gap: var(--s3);
    color: var(--ink3);
    font: var(--type-meta);
  }
  .sel {
    overflow: hidden;
    color: var(--ink2);
    white-space: nowrap;
    text-overflow: ellipsis;
  }
  .selector-unavailable {
    margin: 0;
    color: var(--ink3);
    font: var(--type-meta);
  }

  /* ---- attach a selector by hand ---- */
  .custom-attach { display: grid; }
  .attach-form {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: var(--s2) var(--s3);
  }
  .attach-form .row-label {
    display: flex;
    flex-direction: column;
    gap: var(--s1);
    flex: 1 1 128px;
    min-width: 0;
  }
  .attach-form .row-label > span {
    color: var(--ink3);
    font: var(--type-meta);
  }
  .text-input,
  .search {
    box-sizing: border-box;
    width: 100%;
    min-width: 0;
    height: 36px;
    padding: 0 var(--s3);
    border: 0;
    border-radius: var(--r-col);
    background: var(--wash);
    color: var(--ink);
    font: var(--type-small);
    transition: background-color var(--fast) var(--ease);
  }
  .text-input::placeholder,
  .search::placeholder { color: var(--ink3); }
  .text-input:hover,
  .search:hover { background: var(--wash2); }
  .text-input.small { max-width: 96px; }
  .attach-form .act.probe { height: 36px; color: var(--ink); }

  /* ---- search ---- */
  .search-row {
    display: flex;
    align-items: center;
    gap: var(--s3);
  }
  .search { flex: 1 1 auto; border-radius: var(--r-pill); padding: 0 var(--s4); }
  .refresh {
    flex: none;
    padding: 0;
    border: 0;
    background: transparent;
    color: var(--ink3);
    font: var(--type-control);
    cursor: pointer;
    transition: color var(--fast) var(--ease);
  }
  .refresh:hover:not(:disabled) { color: var(--ink); }
  .refresh:disabled { opacity: 0.4; cursor: default; }

  /* ---- a row's nodes: chips that steer to one ---- */
  .nodes-block { display: flex; flex-direction: column; gap: var(--s2); }
  .nodes-toggle {
    display: inline-flex;
    align-items: center;
    gap: var(--s2);
    align-self: flex-start;
    min-height: 28px;
    padding: 0;
    border: 0;
    background: transparent;
    color: var(--ink3);
    font: var(--type-meta);
    cursor: pointer;
    transition: color var(--fast) var(--ease);
  }
  .nodes-toggle:hover { color: var(--ink); }
  .nodes-count { font-variant-numeric: tabular-nums; }
  .node-chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s1);
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .node-chip {
    display: inline-flex;
    align-items: baseline;
    gap: var(--s1);
    padding: 2px var(--s2);
    border: 0;
    border-radius: var(--r-chip);
    background: var(--wash2);
    color: var(--ink);
    font: var(--type-small);
    cursor: pointer;
    transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease);
  }
  .node-chip:hover:not(:disabled) { background: color-mix(in oklch, var(--tint) 14%, var(--wash2)); }
  .node-chip:active:not(:disabled) { scale: 0.97; }
  .node-chip:disabled { color: var(--ink3); cursor: default; }
  .node-role { color: var(--ink3); font: var(--type-meta); }

  @media (max-width: 720px) {
    .act, .nodes-toggle { min-height: 44px; }
    .text-input, .search { height: 44px; font: var(--type-body); }
    .selector-row { grid-template-columns: auto minmax(0, 1fr); }
    .selector-row .sel { grid-column: 1 / -1; }
  }
</style>
