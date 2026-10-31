<script lang="ts">
  import { onMount as onInterfaceMount } from "svelte";
  import { registerInterfaceController } from "../lib/workspaceController";
  import { nodeCompareViewSchema } from "../lib/interfaceSchemas";
  import { ToolError } from "../lib/webmcp/types";

  import { IconButton, MorphText, Segmented } from "@stdlogits/ledger";
  // Cross-branch diff drawer — phase 5.  Renders a side-by-side
  // word-level diff between two (or more) generated nodes, plus the
  // sorted readings-delta table and the per-token alignment for the
  // primary pair.
  //
  // Drawer params: ``{node_ids: string[], parent_id?: string}``.
  // Surfaces:
  //   * 2 ids — single side-by-side comparison.
  //   * 3+ ids — N-column rendering, one column per id, diffs computed
  //     against the first id as anchor.

  import { onMount } from "svelte";
  import { apiTree, ApiError } from "../lib/runtime/services";
  import type { RuntimeTreeReplayCapabilities } from "../lib/runtime/contracts";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import { diffSideTokens } from "../panels/loom/loomDiff";
  import {
    clearNodeSelection,
    closeDrawer,
    drawerState,
    loomTree,
  } from "../lib/stores.svelte";
  import type {
    DiffReadingDeltaJSON,
    DiffTextSpanJSON,
    DiffTokenSpanJSON,
    JointLogprobsJSON,
    JointLogprobRowJSON,
    NodeDiffJSON,
  } from "../lib/types";

  // --------------------------------------------------------- props ---

  interface Params {
    node_ids?: string[];
    parent_id?: string | null;
  }

  let { params }: { params: unknown } = $props();
  const ids = $derived.by<string[]>(() => {
    const p = (params ?? drawerState.params ?? {}) as Params;
    return Array.isArray(p.node_ids) ? p.node_ids.filter(Boolean) : [];
  });
  const anchorId = $derived(ids[0] ?? null);

  // --------------------------------------------------------- diffs ---

  // For each non-anchor id, fetch the diff against the anchor.  When
  // there are N>=2 ids, we fetch N-1 diffs.  Single fetch when N=2
  // (the common case).  Re-runs whenever ``ids`` changes.

  type DiffOrError =
    | { kind: "ok"; diff: NodeDiffJSON }
    | { kind: "err"; message: string };

  /** Logit-pass Phase 5: joint-logprobs cache state per (anchor, other)
   *  pair.  Loads lazily after the main diff lands so the drawer feels
   *  responsive — the forward passes for cross-evaluation are slower
   *  than the text diff. */
  type JointOrError =
    | { kind: "loading" }
    | { kind: "ok"; data: JointLogprobsJSON }
    | { kind: "unavailable"; message: string }
    | { kind: "err"; message: string };

  let diffs: DiffOrError[] = $state([]);
  let joints: JointOrError[] = $state([]);
  let loading = $state(false);
  let hoveredAnchorIdx: number | null = $state(null);

  /** Layout helper — ``"unified"`` stacks the diff vertically for
   * narrow screens; ``"side-by-side"`` is the default. */
  let layout: "side-by-side" | "unified" = $state("side-by-side");
  let sortBy: "magnitude" | "name" = $state("magnitude");

  async function fetchAll(): Promise<void> {
    const list = ids;
    if (list.length < 2 || !anchorId) {
      diffs = [];
      joints = [];
      return;
    }
    loading = true;
    const replayCapabilities = apiTree.replayCapabilities().catch((error) => ({
      jointLogprobs: {
        available: false,
        reason: userFacingError(
          error,
          "Token likelihood replay is unavailable in this runtime.",
        ),
      },
    } satisfies RuntimeTreeReplayCapabilities));
    const out: DiffOrError[] = [];
    try {
      for (let i = 1; i < list.length; i++) {
        try {
          const r = await apiTree.diff(anchorId, list[i]);
          out.push({ kind: "ok", diff: r });
        } catch (e) {
          if (e instanceof ApiError) {
            const detail =
              e.body && typeof e.body === "object" && "detail" in (e.body as object)
                ? String((e.body as { detail: unknown }).detail)
                : e.message;
            out.push({
              kind: "err",
              message: userFacingError(
                e,
                `Unable to compare this branch. ${detail}`,
              ),
            });
          } else {
            out.push({
              kind: "err",
              message: userFacingError(e, "Unable to compare this branch. Try again."),
            });
          }
        }
      }
      diffs = out;
    } finally {
      loading = false;
    }
    // Logit-pass Phase 5: fire the cross-evaluation forward passes in
    // parallel after the text diff is in place.  Loading state per
    // pair so the drawer can show "(crunching cross logprobs…)"
    // independently of the diff rendering.
    await fetchJoints((await replayCapabilities).jointLogprobs);
  }

  /** Lazy fetch of joint logprobs for each (anchor, other) pair.
   *  Hits the session-lifetime cache on the server side; second open
   *  of the same pair is a memory lookup. */
  async function fetchJoints(
    availability: RuntimeTreeReplayCapabilities["jointLogprobs"],
  ): Promise<void> {
    const list = ids;
    if (list.length < 2 || !anchorId) {
      joints = [];
      return;
    }
    if (!availability.available) {
      const message = availability.reason ??
        "This model runtime cannot replay exact token likelihoods for branch comparison.";
      joints = list.slice(1).map(() => ({ kind: "unavailable", message }));
      return;
    }
    joints = list.slice(1).map(() => ({ kind: "loading" }) as JointOrError);
    const out: JointOrError[] = [];
    for (let i = 1; i < list.length; i++) {
      try {
        const r = await apiTree.jointLogprobs(anchorId, list[i]);
        out.push({ kind: "ok", data: r });
      } catch (e) {
        if (e instanceof ApiError) {
          const detail =
            e.body && typeof e.body === "object" && "detail" in (e.body as object)
              ? String((e.body as { detail: unknown }).detail)
              : e.message;
          out.push({
            kind: "err",
            message: userFacingError(
              e,
              `Unable to compare token likelihoods. ${detail}`,
            ),
          });
        } else {
          out.push({
            kind: "err",
            message: userFacingError(
              e,
              "Token likelihood comparison is unavailable for these branches.",
            ),
          });
        }
      }
      // Update incrementally so each pair lands as soon as the server
      // returns — useful for N-way comparisons.
      joints = [...out, ...joints.slice(out.length)];
    }
    joints = out;
  }

  // Refetch whenever the id list changes.
  $effect(() => {
    void ids;
    void fetchAll();
  });

  onMount(() => {
    // Clear the multi-select once the drawer's data is in flight so the
    // sidebar's selection-bar disappears (user-feel: "ok, I've moved
    // them into the drawer").
    return () => {
      clearNodeSelection();
    };
  });

  // ----------------------------------------------- node previews ---

  function nodePreview(id: string): string {
    const n = loomTree.nodes.get(id);
    const t = (n?.text ?? "").replace(/\s+/g, " ").trim();
    if (!t) return "(empty)";
    return t.length > 80 ? t.slice(0, 80) + "…" : t;
  }

  // ---------------------------------------- readings sort + top-N --

  function sortedReadings(rs: DiffReadingDeltaJSON[]): DiffReadingDeltaJSON[] {
    const arr = [...rs];
    if (sortBy === "magnitude") {
      arr.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
    } else {
      arr.sort((a, b) => a.name.localeCompare(b.name));
    }
    return arr;
  }

  function topKSet(rs: DiffReadingDeltaJSON[], k = 5): Set<string> {
    const sorted = [...rs].sort(
      (a, b) => Math.abs(b.delta) - Math.abs(a.delta),
    );
    return new Set(sorted.slice(0, k).map((r) => r.name));
  }

  // ------------------------------------- per-token hover alignment --

  // Decision (option a): render the per-pane token spans directly from
  // ``diff.per_token`` rather than re-splitting ``a_text`` / ``b_text``
  // on whitespace.  The server's :func:`per_token_diff` walks the real
  // model-tokenizer byte offsets, so its ``a_index`` / ``b_index`` keys
  // only line up with the spans it emitted — never with a client-side
  // whitespace split.  Iterating ``per_token`` gives us correct
  // tokenization for free, plus the hover-cross-highlight is exact
  // because each rendered span carries its own counterpart index.
  //
  // ``per_token`` may be empty for imported transcripts that do not carry
  // token sequences; in that case we fall back to
  // the whitespace-split renderer so the diff stays visible — alignment
  // tooltips just won't fire on the fallback path.

  /** Anchor-side tokens, each once, for one diff's per-token array (``[]`` when it carries none). */
  function panelTokensA(spans: DiffTokenSpanJSON[]): DiffTokenSpanJSON[] {
    return diffSideTokens(spans, "a");
  }
  /** Other-side tokens, each once. */
  function panelTokensB(spans: DiffTokenSpanJSON[]): DiffTokenSpanJSON[] {
    return diffSideTokens(spans, "b");
  }

  function formatReadingDelta(r: DiffReadingDeltaJSON): string {
    const sign = r.delta >= 0 ? "+" : "";
    return `${sign}${r.delta.toFixed(3)}`;
  }

  function spanColor(state: DiffTextSpanJSON["state"]): string {
    if (state === "insert")
      return "color-mix(in oklch, var(--tint) 18%, transparent)";
    if (state === "delete")
      return "color-mix(in oklab, var(--ink3) 18%, transparent)";
    return "transparent";
  }

  function tokensFor(text: string): string[] {
    // Fallback splitter for the per-pane render when ``per_token`` is
    // empty (transcript-loaded nodes without token sequences).  Word-
    // split has no alignment metadata, so cross-pane hover stays off
    // on this path.
    return text.split(/(\s+)/);
  }

  function spanTooltip(sp: DiffTokenSpanJSON): string {
    if (!sp.reading_deltas || sp.reading_deltas.length === 0) return "";
    return sp.reading_deltas
      .slice(0, 3)
      .map((r) => `${r.name} ${formatReadingDelta(r)}`)
      .join(" · ");
  }

  // ----------------------------------------- joint logprobs (Phase 5) --

  /** Format a logprob.  Null / non-finite render as ``—``. */
  function fmtLp(v: number | null | undefined): string {
    if (v == null || !Number.isFinite(v)) return "-";
    return v.toFixed(2);
  }

  /** Format a delta logprob (always signed) — useful when reading
   *  "did B give this token less / more weight than A did?". */
  function fmtDeltaLp(a: number | null, b: number | null): string {
    if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b)) {
      return "-";
    }
    const d = a - b;
    const sign = d >= 0 ? "+" : "";
    return `${sign}${d.toFixed(2)}`;
  }

  /** Format the approx-KL value — small numbers get more precision so
   *  the leading digits remain readable. */
  function fmtKl(v: number | null): string {
    if (v == null || !Number.isFinite(v)) return "-";
    if (Math.abs(v) < 0.01) return v.toExponential(1);
    return v.toFixed(2);
  }

  /** Aligned-only filter — divergent rows have null cross-eval and are
   *  surfaced separately in the divergent-text view above.  Keeps the
   *  logprob table focused on the byte-aligned positions where the
   *  cross comparison is well-defined. */
  function alignedRows(rows: JointLogprobRowJSON[]): JointLogprobRowJSON[] {
    return rows.filter((r) => r.aligned);
  }

  // ------------------------------ siblings summary (Phase 6) ------------
  //
  // Per-sibling rollup for the loom "compare children" surface — answers
  // "did my α=0.5 shift the distribution at the contentious tokens, or
  // just the argmax?" without making the researcher scan the per-token
  // table.  Baseline = anchor (left-most node); one row per non-anchor
  // sibling shows ``mean_logprob`` (from the node), ``rank-1 %
  // unchanged`` (= 1 − n_rank1_changed / n_aligned), and the mean
  // ``approx_kl`` across aligned positions.  The anchor row carries
  // ``—`` for the comparison columns (it's not compared to itself).

  interface SiblingSummaryRow {
    nodeId: string;
    isAnchor: boolean;
    label: string;
    preview: string;
    meanLogprob: number | null;
    /** Fraction of aligned positions where the argmax did NOT change.
     *  Null for the anchor row. */
    rank1Unchanged: number | null;
    /** Mean approx-KL across aligned positions (the joint table's
     *  ``approx_kl`` column).  Null for the anchor row. */
    klMean: number | null;
    /** Joint-table loading state per row, so we can show a "(crunching)"
     *  shape next to the values instead of a blank row. */
    jointReady: boolean;
  }

  /** Build the summary table from currently-loaded state.  Pulls
   *  ``mean_logprob`` from the local tree cache and rolls up rank-1 /
   *  KL stats from each joint-logprobs response.  Re-runs reactively
   *  whenever ``ids`` or ``joints`` updates. */
  const siblingSummary = $derived.by<SiblingSummaryRow[]>(() => {
    if (ids.length < 2) return [];
    const out: SiblingSummaryRow[] = [];
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      const node = loomTree.nodes.get(id);
      const isAnchor = i === 0;
      const baseRow: SiblingSummaryRow = {
        nodeId: id,
        isAnchor,
        label: isAnchor
          ? "A"
          : ids.length === 2
            ? "B"
            : `B${i}`,
        preview: nodePreview(id),
        meanLogprob:
          node && typeof node.mean_logprob === "number"
            ? node.mean_logprob
            : null,
        rank1Unchanged: null,
        klMean: null,
        jointReady: isAnchor,
      };
      if (!isAnchor) {
        // Anchor takes index 0; ``joints`` is indexed by non-anchor
        // position so ``joints[i - 1]`` matches this row.
        const j = joints[i - 1];
        if (j) {
          if (j.kind === "ok") {
            const aligned = alignedRows(j.data.rows);
            if (aligned.length > 0) {
              baseRow.rank1Unchanged =
                1 - j.data.n_rank1_changed / aligned.length;
              let sum = 0;
              let n = 0;
              for (const r of aligned) {
                if (r.approx_kl != null && Number.isFinite(r.approx_kl)) {
                  sum += r.approx_kl;
                  n += 1;
                }
              }
              baseRow.klMean = n > 0 ? sum / n : null;
            }
            baseRow.jointReady = true;
          } else if (j.kind === "err" || j.kind === "unavailable") {
            baseRow.jointReady = true;
          }
        }
      }
      out.push(baseRow);
    }
    return out;
  });

  function fmtPct(v: number | null): string {
    if (v == null || !Number.isFinite(v)) return "-";
    return `${(v * 100).toFixed(0)}%`;
  }

  function fmtKlMean(v: number | null): string {
    if (v == null || !Number.isFinite(v)) return "-";
    if (Math.abs(v) < 0.01) return v.toExponential(1);
    return v.toFixed(3);
  }

  onInterfaceMount(() => registerInterfaceController("node_compare", {
    schema: nodeCompareViewSchema,
    read: () => ({ busy: false, values: { layout, sort_by: sortBy } }),
    update: async (change) => {
      if (false) throw new ToolError("BUSY", "Wait for this interface operation to finish.");
      if (change.layout !== undefined) layout = change.layout as typeof layout;
      if (change.sort_by !== undefined) sortBy = change.sort_by as typeof sortBy;
    },
  }));
</script>

<section class="drawer-shell" aria-label="Cross-branch diff drawer">
  <header class="header">
    <div class="title">
      <h2 class="dw-title">Compare branches</h2>
      <div class="dw-facts">
        {#if ids.length >= 2}
          <span><b><MorphText text={ids.length} /></b> branches</span>
        {:else}
          <span>Select two or more replies</span>
        {/if}
      </div>
    </div>
    <IconButton icon="close" label="Close" onclick={closeDrawer} />
  </header>

  <div class="toolbar">
    <Segmented label="Layout" value={layout} onchange={(value) => (layout = value)}
      options={[{ value: "side-by-side", label: "Side by side" }, { value: "unified", label: "Unified" }]} />
    <Segmented label="Sort readings by" value={sortBy} onchange={(value) => (sortBy = value)}
      options={[{ value: "magnitude", label: "Largest change" }, { value: "name", label: "Name" }]} />
  </div>

  <div class="body">
    {#if ids.length < 2}
      <p class="empty">Select two or more generated replies on the loom</p>
    {:else if loading && diffs.length === 0}
      <div class="dw-blocks" role="status" aria-label="Comparing replies"><i></i><i></i><i></i></div>
    {:else}
      <!-- Column heads: the anchor and each reply compared with it. -->
      <div class="columns" class:unified={layout === "unified"}>
        <div class="col anchor-col">
          <header class="col-header">
            <span class="col-tag">A</span>
            <span class="meta">{anchorId?.slice(0, 12) ?? ""}</span>
          </header>
          <p class="col-preview">{nodePreview(anchorId ?? "")}</p>
        </div>
        {#each ids.slice(1) as otherId, otherIdx (otherId)}
          <div class="col">
            <header class="col-header">
              <span class="col-tag">B{ids.length > 2 ? otherIdx + 1 : ""}</span>
              <span class="meta">{otherId.slice(0, 12)}</span>
            </header>
            <p class="col-preview">{nodePreview(otherId)}</p>
          </div>
        {/each}
      </div>

      <!-- Per-sibling rollup: KL and rank-1 columns fill in as the joint-logprob fetches land. -->
      {#if siblingSummary.length > 0}
        <section class="siblings-summary">
          <div class="dw-sec-head">
            <span class="head">Siblings</span>
            <span class="meta">against {siblingSummary[0]?.label ?? "A"}</span>
          </div>
          <table class="table">
            <thead>
              <tr>
                <th></th>
                <th>Reply</th>
                <th class="num" {...{ "aria-description": "mean logprob" }}>Mean log p</th>
                <th class="num" {...{ "aria-description": "rank-1 agreement" }}>Top token kept</th>
                <th class="num" {...{ "aria-description": "approximate KL" }}>Mean ≈KL</th>
              </tr>
            </thead>
            <tbody>
              {#each siblingSummary as row (row.nodeId)}
                <tr class:anchor={row.isAnchor}>
                  <td class="tag">{row.label}</td>
                  <td class="preview">{row.preview}</td>
                  <td class="num">{fmtLp(row.meanLogprob)}</td>
                  <td class="num">
                    <MorphText text={row.isAnchor
                      ? "-"
                      : row.jointReady
                        ? fmtPct(row.rank1Unchanged)
                        : "…"} />
                  </td>
                  <td class="num">
                    <MorphText text={row.isAnchor
                      ? "-"
                      : row.jointReady
                        ? fmtKlMean(row.klMean)
                        : "…"} />
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </section>
      {/if}

      {#each diffs as result, diffIdx (diffIdx)}
        {#if result.kind === "err"}
          <p class="error" role="alert">{result.message}</p>
        {:else}
          {@const d = result.diff}
          {@const topDeltas = topKSet(d.readings, 5)}
          {@const sortedRs = sortedReadings(d.readings)}
          {@const joint = joints[diffIdx]}

          <section class="diff-block">
            <header class="diff-header">
              <span class="diff-title">{ids.length === 2 ? "A and B" : `A and B${diffIdx + 1}`}</span>
              {#if d.parent_applied_steering !== null || d.steering_delta}
                <span class="meta" {...{ "aria-description": "steering delta A → B" }}>steering {d.steering_delta || "unchanged"}</span>
              {/if}
            </header>

            <!-- The text: side by side, or one unified run. -->
            {#if layout === "side-by-side"}
              {@const tokensA = panelTokensA(d.per_token)}
              {@const tokensB = panelTokensB(d.per_token)}
              <div class="text-grid">
                <div class="text-pane">
                  <span class="pane-label">A</span>
                  <div class="text-body">
                    {#if tokensA.length > 0}
                      {#each tokensA as sp (`a-${sp.a_index}`)}
                        <!-- svelte-ignore a11y_no_static_element_interactions -->
                        <span
                          class="tok"
                          class:highlight-anchor={hoveredAnchorIdx === sp.a_index}
                          {...{ "aria-description": (spanTooltip(sp)) }}
                          onmouseenter={() => (hoveredAnchorIdx = sp.a_index)}
                          onmouseleave={() => (hoveredAnchorIdx = null)}
                        >{sp.a_text}</span>
                      {/each}
                    {:else}
                      {#each tokensFor(d.a_text) as part, idx (idx)}
                        <span class="tok">{part}</span>
                      {/each}
                    {/if}
                  </div>
                </div>
                <div class="text-pane">
                  <span class="pane-label">B{ids.length > 2 ? diffIdx + 1 : ""}</span>
                  <div class="text-body">
                    {#if tokensB.length > 0}
                      {#each tokensB as sp (`b-${sp.b_index}`)}
                        {@const matched =
                          sp.a_index >= 0 &&
                          hoveredAnchorIdx !== null &&
                          sp.a_index === hoveredAnchorIdx}
                        <span
                          class="tok"
                          class:highlight-target={matched}
                          {...{ "aria-description": (spanTooltip(sp)) }}
                        >{sp.b_text}</span>
                      {/each}
                    {:else}
                      {#each tokensFor(d.b_text) as part, idx (idx)}
                        <span class="tok">{part}</span>
                      {/each}
                    {/if}
                  </div>
                </div>
              </div>
            {:else}
              <div class="unified-body">
                {#each d.text as span, idx (idx)}
                  <span
                    class="tok-span"
                    class:span-equal={span.state === "equal"}
                    class:span-insert={span.state === "insert"}
                    class:span-delete={span.state === "delete"}
                    style={`background-color: ${spanColor(span.state)}`}
                  >
                    {#if span.state === "insert"}<span class="span-sign">+</span
                      >{:else if span.state === "delete"}<span class="span-sign"
                        >−</span
                      >{/if}{span.text}{" "}
                  </span>
                {/each}
              </div>
            {/if}

            <!-- Each aligned token, evaluated under both branches; fetched beside the diff.
                 ``joint`` is declared at the top of the {:else} block (Svelte requires @const at
                 the start of a parent block). -->
            {#if joint}
              {#if joint.kind === "loading"}
                <p class="meta" role="status">Computing token probabilities…</p>
              {:else if joint.kind === "err"}
                <p class="error" role="alert">Log probabilities: {joint.message}</p>
              {:else if joint.kind === "unavailable"}
                <p class="meta replay-unavailable">{joint.message} The text and saved readings still compare.</p>
              {:else}
                {@const aligned = alignedRows(joint.data.rows)}
                {#if aligned.length > 0}
                  <div class="joint-table">
                    <div class="dw-sec-head">
                      <span class="head">Log probabilities</span>
                      <span class="meta"><b>{joint.data.n_rank1_changed}</b> of {aligned.length} top tokens change</span>
                    </div>
                    <div class="table-scroll">
                      <table class="table">
                        <thead>
                          <tr>
                            <th class="num">Pos</th>
                            <th>Token</th>
                            <th class="num">A</th>
                            <th class="num">B</th>
                            <th class="num" {...{ "aria-description": "log pB(A token) − log pA(A token)" }}>Δ A token</th>
                            <th class="num" {...{ "aria-description": "approx KL(A ∥ B) · top-K" }}>≈KL</th>
                            <th class="flag" {...{ "aria-description": "argmax differs" }}><span class="sr-only">Top token changed</span></th>
                          </tr>
                        </thead>
                        <tbody>
                          {#each aligned as row (`${row.a_index}-${row.b_index}`)}
                            <tr class:rank-flip={row.rank_changed}>
                              <td class="num pos">{row.a_index}</td>
                              <td class="token">{JSON.stringify(row.a_text)}</td>
                              <td class="num">{fmtLp(row.lp_a_in_a)}</td>
                              <td class="num">{fmtLp(row.lp_b_in_b)}</td>
                              <td class="num">{fmtDeltaLp(row.lp_a_in_b, row.lp_a_in_a)}</td>
                              <td class="num">{fmtKl(row.approx_kl)}</td>
                              <td class="flag">{#if row.rank_changed}<i aria-label="top token changed"></i>{/if}</td>
                            </tr>
                          {/each}
                        </tbody>
                      </table>
                    </div>
                  </div>
                {:else}
                  <p class="meta">No aligned tokens</p>
                {/if}
              {/if}
            {/if}

            <!-- How each reading moved from A to B. -->
            {#if sortedRs.length > 0}
              <div class="readings-table">
                <span class="head">Readings</span>
                <div class="readings-grid">
                  {#each sortedRs as r (r.name)}
                    {@const top = topDeltas.has(r.name)}
                    <div class="reading-row" class:top-delta={top}>
                      <span class="r-name">{r.name}</span>
                      <span class="r-vals">
                        <MorphText text={r.a_value.toFixed(3)} />
                        <span aria-hidden="true">→</span>
                        <MorphText text={r.b_value.toFixed(3)} />
                      </span>
                      <span class="r-bar-track">
                        <i class:neg={r.delta < 0} style={`width: ${Math.min(100, Math.abs(r.delta) * 100)}%`}></i>
                      </span>
                      <span class="r-delta"><MorphText text={formatReadingDelta(r)} /></span>
                    </div>
                  {/each}
                </div>
              </div>
            {:else}
              <p class="meta">No readings</p>
            {/if}
          </section>
        {/if}
      {/each}
    {/if}
  </div>
</section>

<style>
  /* Branches compared: the replies as washes side by side, then the numbers as rows. */
  .drawer-shell { display: flex; flex-direction: column; gap: var(--s4); height: 100%; min-height: 0; padding: var(--s5) var(--s6) 0; color: var(--ink2); font: var(--type-small); }
  .header { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--s5); }
  .title { display: grid; gap: var(--s1); min-width: 0; }
  .toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s4); }
  .body { display: flex; flex: 1 1 auto; flex-direction: column; gap: var(--s6); min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding-bottom: var(--s6); }
  .empty { color: var(--ink3); }
  .error { color: var(--ink); }
  .meta b { color: var(--ink); font-weight: 500; }

  .columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: var(--s3); }
  .columns.unified { grid-template-columns: minmax(0, 1fr); }
  .col { display: grid; gap: var(--s1); align-content: start; min-width: 0; padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .col.anchor-col { background: var(--wash2); }
  .col-header { display: flex; align-items: baseline; gap: var(--s2); }
  .col-tag, .pane-label, .tag { color: var(--ink); font: var(--type-control); }
  .col-preview { display: -webkit-box; -webkit-line-clamp: 3; line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; color: var(--ink2); overflow-wrap: anywhere; }

  .siblings-summary, .joint-table, .readings-table { display: grid; gap: var(--s2); min-width: 0; }
  .table-scroll { max-height: 360px; overflow: auto; }
  .table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  .table th { position: sticky; top: 0; padding: var(--s1) var(--s2); background: var(--paper); color: var(--ink3); font: var(--type-head); text-align: start; white-space: nowrap; }
  .table td { padding: var(--s1) var(--s2); color: var(--ink2); vertical-align: top; }
  .table tbody tr { transition: background-color var(--fast) var(--ease); }
  .table tbody tr:hover { background: var(--wash); }
  .table tbody tr.anchor { background: var(--wash); }
  .table td:first-child, .table th:first-child { border-radius: var(--r-col) 0 0 var(--r-col); }
  .table td:last-child, .table th:last-child { border-radius: 0 var(--r-col) var(--r-col) 0; }
  .table .num { text-align: end; white-space: nowrap; }
  .table .preview { max-width: 360px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .table .token { color: var(--ink); white-space: pre; }
  .table .pos { color: var(--ink3); }
  .table .flag { width: 24px; text-align: center; }
  .table .flag i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; background: var(--tint); }
  .table tr.rank-flip td { color: var(--ink); }

  .diff-block { display: grid; gap: var(--s4); min-width: 0; }
  .diff-header { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s1) var(--s3); }
  .diff-title { color: var(--ink); font: var(--type-lead); }
  .text-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s3); }
  .text-pane { display: grid; gap: var(--s2); align-content: start; min-width: 0; padding: var(--s3) var(--s4) var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .text-body, .unified-body { color: var(--ink); font: var(--type-body); line-height: 28px; white-space: pre-wrap; overflow-wrap: anywhere; }
  .unified-body { padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .tok { border-radius: 4px; transition: background-color var(--fast) var(--ease), box-shadow var(--fast) var(--ease); }
  .tok.highlight-anchor, .tok.highlight-target { --hl: color-mix(in oklch, var(--tint) 22%, transparent); background: var(--hl); box-shadow: 0 0 0 2px var(--hl); }
  .tok-span { border-radius: 4px; }
  .span-delete { color: var(--ink3); text-decoration: line-through; text-decoration-color: var(--ink3); }
  .span-sign { margin-right: 2px; color: var(--ink3); font: var(--type-meta); }

  .readings-grid { display: grid; gap: 2px; }
  .reading-row { display: grid; grid-template-columns: minmax(0, 1.4fr) auto 64px 64px; gap: var(--s3); align-items: center; min-height: 32px; padding: 0 var(--s2); border-radius: var(--r-col); transition: background-color var(--fast) var(--ease); }
  .reading-row:hover { background: var(--wash); }
  .r-name { overflow: hidden; color: var(--ink2); text-overflow: ellipsis; white-space: nowrap; }
  .top-delta .r-name { color: var(--ink); font-weight: 500; }
  .r-vals { display: inline-flex; gap: var(--s1); color: var(--ink3); font: var(--type-meta); font-variant-numeric: tabular-nums; }
  .r-bar-track { position: relative; height: 4px; overflow: hidden; border-radius: 2px; background: var(--wash2); }
  .r-bar-track > i { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 2px; background: var(--tint); }
  .r-bar-track > i.neg { background: var(--ink3); opacity: 0.6; }
  .r-delta { color: var(--ink2); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: end; }

  @media (max-width: 720px) {
    .drawer-shell { padding: var(--s4) var(--s4) 0; }
    .text-grid { grid-template-columns: minmax(0, 1fr); }
    .reading-row { grid-template-columns: minmax(0, 1fr) 48px 56px; }
    .r-vals { display: none; }
  }
</style>
