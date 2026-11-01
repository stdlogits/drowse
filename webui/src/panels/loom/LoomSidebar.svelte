<script lang="ts">
  import { loomViewSchema } from "../../lib/interfaceSchemas";
  import { registerInterfaceController } from "../../lib/workspaceController";
  import { ToolError } from "../../lib/webmcp/types";
  import { onMount as onInterfaceMount } from "svelte";

  import { Icon, IconButton, MorphText, Segmented, Tabs } from "@stdlogits/ledger";

  import RollingNumber from "../../lib/ui/RollingNumber.svelte";
  // Conversation loom. Turns advance from left to right and sibling
  // continuations separate vertically around their shared parent.
  //
  // Supports navigation, editing and regeneration, search, starring and notes,
  // compare pinning, fan-out, filtering, and cross-branch diff.

  import { onMount, tick, untrack } from "svelte";
  import { createLoomLiveNode, segmentHeightEstimate, segmentWidth } from "./loomMapModel";
  import { MediaQuery, SvelteMap, SvelteSet } from "svelte/reactivity";
  import { expand, fadeOut } from "@stdlogits/ledger";
  import {
    applyTreeFilter,
    chatLog,
    clearChat,
    clearNodeSelection,
    clearTreeFilter,
    edgeLabelCache,
    fetchEdgeLabel,
    filterState,
    highlightState,
    genStatus,
    loomContinueFromCommitted,
    loomTree,
    loomNodeIntersectsGeneration,
    LOOM_DELETE_DURING_GENERATION_MESSAGE,
    loomNavigate,
    loomSwapSeat,
    sessionState,
    loomStar,
    loomRegenerateActive,
    loomUiState,
    nodeSelection,
    openDrawer,
    pinNodeForComparison,
    pinnedComparison,
    refreshLoomTree,
    sendFork,
    pushToast,
    toggleNodeSelection,
  } from "../../lib/stores.svelte";
  import LoomNode from "./LoomNode.svelte";
  import { segmentLoom } from "./loomSegments";
  import LoomWeave from "./LoomWeave.svelte";
  import LoomDialog from "./LoomDialog.svelte";
  import {
    layoutLoom,
    loomBranchJunctions,
    type LoomLayoutEdge,
  } from "./loomLayout";
  import { leadingSpace, tokenSentences, tokenWords } from "./loomSentences";
  import {
    centerLoomRect,
    fitLoomCamera,
    LOOM_MAX_ZOOM,
    LOOM_MIN_ZOOM,
    resizeLoomCamera,
    settleLoomCenter,
    zoomLoomAt,
    type LoomCamera,
    type LoomRect,
  } from "./loomViewport";
  import type { LoomNodeJSON } from "../../lib/types";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { loomNodeSnippet, loomSearchExcerpt, loomTextMatches } from "../../lib/loomSearch";
  
  import {
    pinchMetrics,
    scaleFromPinch,
    type GesturePoint,
  } from "../../lib/pointerGesture";

  let { active = true, headersVisible = true }: { active?: boolean; headersVisible?: boolean } = $props();

  // ----------------------------------------- flat tree walk + depth --

  /** Depth-first walk of the tree starting at root.  Yields one row
   *  per node with its depth, so the renderer indents linearly. */
  interface Row {
    node: LoomNodeJSON;
    depth: number;
    isActivePath: boolean;
    isDead: boolean;
    /** Phase 5 filter dim: true iff the filter is on and this node
     *  doesn't match.  Rendered at 50% opacity — distinct visual
     *  channel from the 30% dead-branch dim. */
    filteredOut: boolean;
  }

  const activePathSet = $derived(new Set(loomTree.activePath));
  const selectionSet = $derived(new Set(nodeSelection.ids));
  const collapsedIds = new SvelteSet<string>();

  // --------------------------------------- compare bar (active node) --
  //
  // A single capability-aware "compare": a generated node compares its
  // siblings; a committed node compares its generated children.  Seat does
  // not decide which branch is a model result.

  /** The turn whose generated continuations the compare acts on. */
  const compareUserParentId = $derived.by<string | null>(() => {
    const id = loomTree.active_node_id;
    const n = id ? (loomTree.nodes.get(id) ?? null) : null;
    if (!n) return null;
    if (n.recipe !== null) return n.parent_id;
    if (n.role !== "system") return n.id;
    return null;
  });

  const comparableNodes = $derived.by<LoomNodeJSON[]>(() => {
    if (!compareUserParentId) return [];
    return (loomTree.children_of.get(compareUserParentId) ?? [])
      .map((id) => loomTree.nodes.get(id))
      .filter((n): n is LoomNodeJSON => n != null && n.recipe !== null);
  });

  function compareBranch(): void {
    if (!compareUserParentId || comparableNodes.length < 2) return;
    openDrawer("node_compare", {
      node_ids: comparableNodes.map((n) => n.id),
      parent_id: compareUserParentId,
    });
  }

  /** Logit-pass: order children by ``mean_logprob`` when the sibling-sort
   *  filter directive is active.  Returns the children list in the order
   *  to *visit* (DFS pushes in reverse, so this is "first-out" order).
   *  Nodes without ``mean_logprob`` sink to the end and preserve their
   *  insertion-order tiebreak. */
  function _orderedChildren(parentId: string): string[] {
    const children = loomTree.children_of.get(parentId) ?? [];
    const mode = loomUiState.siblingSort;
    if (mode === "default" || children.length < 2) return children;
    // ``surprise`` first ⇒ most-surprising (lowest logprob) first.
    // ``confidence`` first ⇒ highest logprob first.
    const sign = mode === "surprise" ? 1 : -1;
    const indexed = children.map((id, idx) => ({
      id,
      idx,
      lp: loomTree.nodes.get(id)?.mean_logprob ?? null,
    }));
    indexed.sort((a, b) => {
      // Stable: null sinks; equal logprobs preserve insertion order.
      if (a.lp === null && b.lp === null) return a.idx - b.idx;
      if (a.lp === null) return 1;
      if (b.lp === null) return -1;
      const delta = sign * (a.lp - b.lp);
      return delta !== 0 ? delta : a.idx - b.idx;
    });
    return indexed.map((e) => e.id);
  }

  /** The saved tree in visit order.  Structural only: it never reads the
   *  streaming turn, so a token frame does not re-walk the tree. */
  const treeRows = $derived.by<Row[]>(() => {
    const out: Row[] = [];
    if (!loomTree.root_id) return out;
    const matching = filterState.matchingIds;
    // Touch the sort key so $derived re-runs when it flips.  Reading
    // ``loomUiState.siblingSort`` happens inside ``_orderedChildren``
    // already; this is the Svelte 5 idiom for keeping the dependency
    // explicit when the helper is called inside a tight loop.
    void loomUiState.siblingSort;
    const stack: { id: string; depth: number; deadAncestor: boolean }[] = [
      { id: loomTree.root_id, depth: 0, deadAncestor: false },
    ];
    while (stack.length) {
      const { id, depth, deadAncestor } = stack.pop()!;
      const node = loomTree.nodes.get(id);
      if (!node) continue;
      const onActive = activePathSet.has(id);
      const isDead = deadAncestor || !onActive;
      // Skip the synthetic root system node from the visible list —
      // it has no text and is just an anchor.
      if (!(node.parent_id === null && node.role === "system" && !node.text)) {
        const filteredOut = matching !== null && !matching.has(id);
        out.push({
          node,
          depth,
          isActivePath: onActive,
          isDead: isDead && !onActive,
          filteredOut,
        });
      }
      // Push children in reverse so DFS visits them in order, then
      // apply the optional sibling-sort.
      const children = matching === null && collapsedIds.has(id) ? [] : _orderedChildren(id);
      for (let i = children.length - 1; i >= 0; i--) {
        stack.push({
          id: children[i],
          depth: depth + (node.parent_id === null && node.role === "system" && !node.text ? 0 : 1),
          deadAncestor: deadAncestor || (!onActive && node.parent_id !== null),
        });
      }
    }
    return out;
  });
  const treeRowById = $derived(new Map(treeRows.map((row) => [row.node.id, row])));

  // The streaming reply, overlaid on its saved node only while the loom is
  // on screen.  The overlay converts just the tokens that arrived since the
  // previous frame.
  const liveOverlay = createLoomLiveNode();
  const liveNode = $derived.by<LoomNodeJSON | null>(() => {
    if (!active || !genStatus.active || chatLog.pendingIndex === null || !loomTree.pendingNodeId) return null;
    const saved = loomTree.nodes.get(loomTree.pendingNodeId);
    const turn = chatLog.turns[chatLog.pendingIndex];
    if (!saved || !turn || !treeRowById.has(saved.id)) return null;
    const node = liveOverlay(saved, turn);
    return node === saved ? null : node;
  });
  /** Visit-order rows with the live reply's content.  Views that show node
   *  content read these; structure, focus, and search read ``treeRows``. */
  const rows = $derived(liveNode === null ? treeRows : treeRows.map((row) =>
    row.node.id === liveNode.id ? { ...row, node: liveNode } : row));
  const rowById = $derived(liveNode === null ? treeRowById : new Map(rows.map((row) => [row.node.id, row])));
  const coarsePointer = new MediaQuery("(pointer: coarse)");

  /** The map projection (segments, sizes, layout) runs only while the map
   *  is on screen; hidden pages keep their last measured card heights. */
  const mapActive = $derived(active && loomUiState.view === "map");
  const segments = $derived(mapActive ? segmentLoom(rows.map(row => row.node), activePathSet) : []);
  const measuredHeights = new SvelteMap<string, number>();
  $effect(() => {
    if (!mapActive) return;
    const ids = new Set(segments.map(segment => segment.id));
    for (const id of measuredHeights.keys()) {
      if (!ids.has(id)) measuredHeights.delete(id);
    }
  });
  const segmentById = $derived(new Map(segments.map(segment => [segment.id, segment])));
  const segmentWidths = $derived(new Map(segments.map(segment => [segment.id, segmentWidth(segment)])));
  const mapRowById = $derived(new Map(segments.map(segment => [segment.id, {
    ...rowById.get(segment.node.id)!,
    depth: segment.depth,
    isActivePath: segment.memberIds.some(id => activePathSet.has(id)),
    isDead: segment.memberIds.every(id => rowById.get(id)!.isDead),
    filteredOut: segment.memberIds.every(id => rowById.get(id)!.filteredOut),
  }])));
  const graph = $derived.by(() => layoutLoom(segments.map((segment) => ({
    id: segment.id,
    parentId: segment.parentId,
    depth: segment.depth,
    width: segmentWidths.get(segment.id),
    height: compactMap ? 168 : measuredHeights.has(segment.id)
      ? measuredHeights.get(segment.id)!
      : segmentHeightEstimate(segment, segmentWidths.get(segment.id)!, coarsePointer.current),
  }))));
  const branchPointCount = $derived(loomBranchJunctions(graph).length);
  const visibleTokenCount = $derived(segments.reduce(
    (total, segment) => total + segment.end - segment.start,
    0,
  ));
  const branchJunctions = $derived(loomBranchJunctions(graph));

  $effect(() => {
    if (!active || loomUiState.view !== "map" || genStatus.active) return;
    void loomTree.rev;
    const currentRows = treeRows;
    untrack(() => {
      for (const row of currentRows) {
        if (row.node.parent_id && row.node.recipe !== null) {
          fetchEdgeLabel(row.node.parent_id, row.node.id);
        }
      }
    });
  });

  function edgeIsActive(edge: LoomLayoutEdge): boolean {
    return Boolean(mapRowById.get(edge.parentId)?.isActivePath && mapRowById.get(edge.childId)?.isActivePath);
  }

  function edgeIsFiltered(edge: LoomLayoutEdge): boolean {
    return mapRowById.get(edge.childId)?.filteredOut ?? false;
  }

  function edgeStyle(edge: LoomLayoutEdge): string {
    const value = loomTree.nodes.get(edge.childId)?.mean_logprob;
    if (typeof value !== "number" || !Number.isFinite(value) || value > 0) return "";
    const intensity = 1 - Math.exp(value);
    return `--edge-width:${(1.25 + 1.75 * intensity).toFixed(2)}px;--edge-opacity:${(0.42 + 0.5 * intensity).toFixed(3)}`;
  }

  function divergenceLabel(node: LoomNodeJSON): string | null {
    if (!node.parent_id || !node.raw_token_ids?.length) return null;
    const siblings = (loomTree.children_of.get(node.parent_id) ?? [])
      .map((id) => loomTree.nodes.get(id))
      .filter((sibling): sibling is LoomNodeJSON =>
        sibling != null && sibling.id !== node.id && Boolean(sibling.raw_token_ids?.length)
      );
    if (siblings.length === 0) return null;

    let deepest = 0;
    for (const sibling of siblings) {
      const other = sibling.raw_token_ids!;
      const limit = Math.min(node.raw_token_ids.length, other.length);
      let index = 0;
      while (index < limit && node.raw_token_ids[index] === other[index]) index += 1;
      if (index < limit || node.raw_token_ids.length !== other.length) {
        deepest = Math.max(deepest, index);
      }
    }
    if (deepest === 0) return "alternate reply";
    const token = node.tokens?.find((row) => row.raw_index === deepest)?.text.trim();
    if (token) {
      const clean = token.replace(/\s+/g, " ").slice(0, 16);
      return `fork · “${clean}”`;
    }
    return `fork · token ${deepest + 1}`;
  }

  // ----------------------------------------- ring decoration --------

  /** Per-node ring fill keyed off the currently-selected highlight
   *  probe. Returns the node's
   *  aggregate reading for ``highlightState.target`` in [-1, 1], or
   *  ``null`` when no probe is selected, the node has no aggregate
   *  readings yet, or the selected probe is missing from this node's
   *  reading map.  ``LoomNode`` renders the ring only when this value
   *  is non-null. */
  function ringFor(node: LoomNodeJSON): number | null {
    const target = highlightState.target;
    if (!target) return null;
    const readings = node.aggregate_readings;
    if (!readings) return null;
    const v = readings[target];
    return typeof v === "number" ? v : null;
  }

  /** Logit-pass: the badge value to render on a node — the node's own
   *  ``mean_logprob``, or null when capture wasn't live (which suppresses
   *  the badge entirely in ``LoomNode``). */
  function weightBadgeFor(node: LoomNodeJSON): number | null {
    const v = node.mean_logprob;
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  }

  /** Steering-delta label for the edge into ``node``. Generated-only:
   *  committed / system nodes don't carry steering. */
  function steerLabelFor(node: LoomNodeJSON): string | null {
    if (!node.parent_id) return null;
    if (node.recipe === null) return null;
    return edgeLabelCache.get(`${node.parent_id}|${node.id}`) ?? null;
  }

  // ----------------------------------------- filter input ------------

  let searchInput: HTMLInputElement | null = $state(null);
  let searchNodeId = $state<string | null>(null);
  let searchSegmentId = $state<string | null>(null);
  let searchRootId = $state(loomTree.root_id);
  const searchResults = $derived(filterState.matchingIds === null ? [] : treeRows.filter(row => !row.filteredOut && (loomUiState.view !== "saved" || row.node.starred)));
  const searchIndex = $derived(searchResults.findIndex(row => row.node.id === searchNodeId));
  const searchResult = $derived(searchResults[Math.max(0, searchIndex)]?.node ?? null);
  const searchExcerpt = $derived(searchResult ? loomSearchExcerpt(searchResult.text ?? "", filterState.mode === "text" ? filterState.expr : "") : null);

  $effect(() => {
    void filterState.expr;
    void filterState.mode;
    searchNodeId = null;
    searchSegmentId = null;
  });
  $effect(() => {
    const rootId = loomTree.root_id;
    void loomTree.rev;
    untrack(() => {
      if (rootId !== searchRootId) {
        searchRootId = rootId;
        clearTreeFilter();
      } else if (filterState.expr.trim()) {
        void applyTreeFilter(filterState.expr);
      }
    });
  });

  function editFilter(value: string): void {
    clearTreeFilter();
    filterState.expr = value;
    if (filterState.mode === "text") void applyTreeFilter(value);
  }

  function changeSearchMode(mode: string): void {
    const expr = filterState.expr;
    clearTreeFilter();
    filterState.mode = mode === "advanced" ? "advanced" : "text";
    filterState.expr = expr;
    if (filterState.mode === "text") void applyTreeFilter(expr);
  }

  async function showSearchMatch(direction = 0): Promise<void> {
    if (!searchResults.length) return;
    const index = searchIndex < 0 ? (direction < 0 ? searchResults.length - 1 : 0)
      : direction === 0 ? searchIndex
      : (searchIndex + direction + searchResults.length) % searchResults.length;
    const node = searchResults[index].node;
    searchNodeId = node.id;
    loomUiState.view = "map";
    await tick();
    const candidates = segments.filter(segment => segment.memberIds.includes(node.id));
    const segment = candidates.find(segment => filterState.mode === "text" && loomTextMatches(
      segment.node.tokens?.slice(segment.start, segment.end).map(token => token.text).join("") || segment.node.text || "", filterState.expr,
    )) ?? candidates.find(segment => segment.id === node.id) ?? candidates[0];
    if (!segment) return;
    searchSegmentId = segment.id;
    if (compactMap) {
      zoomView(1);
      await tick();
    }
    await centerSettled(segment.id);
    const card = viewportEl?.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(segment.id)}"]`);
    const sentence = [...(card?.querySelectorAll<HTMLElement>(".sentence-node") ?? [])].find(element =>
      filterState.mode === "text" && loomTextMatches(element.querySelector(".sentence-tokens")?.textContent ?? "", filterState.expr));
    const placed = graph.nodes.find(node => node.id === segment.id);
    if (sentence && card && placed && viewportEl) {
      const rect = sentence.getBoundingClientRect();
      const cardRect = card.getBoundingClientRect();
      const scale = cardRect.width / placed.width;
      camera = centerLoomRect(viewportSize(), {
        x: placed.x + (rect.left - cardRect.left) / scale,
        y: placed.y + (rect.top - cardRect.top) / scale,
        width: rect.width / scale,
        height: rect.height / scale,
      }, camera.zoom);
    }
  }

  function onFilterKey(ev: KeyboardEvent): void {
    if (ev.isComposing) return;
    if (ev.key === "Enter") {
      ev.preventDefault();
      ev.stopPropagation();
      if (filterState.mode === "advanced" && filterState.matchingIds === null) void applyTreeFilter(filterState.expr);
      else void showSearchMatch(ev.shiftKey ? -1 : 1);
    } else if (ev.key === "Escape") {
      ev.preventDefault();
      ev.stopPropagation();
      clearTreeFilter();
    }
  }

  // ------------------------------------------- focus / keyboard nav --

  let focusedId: string | null = $state(null);
  type LoomView = "weave" | "map" | "path" | "options" | "saved";
  interface PathSnippet {
    key: string;
    node: LoomNodeJSON;
    sentenceIndex: number | null;
    start: number;
    end: number;
    text: string;
  }

  const cursorNode = $derived.by<LoomNodeJSON | null>(() => {
    const id = focusedId ?? loomTree.active_node_id;
    return id ? (loomTree.nodes.get(id) ?? null) : null;
  });
  const activePathRows = $derived.by<Row[]>(() => loomTree.activePath
    .map((id) => rowById.get(id))
    .filter((row): row is Row => row != null));
  const pathSnippets = $derived.by<PathSnippet[]>(() => {
    const snippets: PathSnippet[] = [];
    for (const row of activePathRows) {
      const node = row.node;
      const sentences = node.tokens ? tokenSentences(node.tokens) : [];
      if (sentences.length === 0) {
        snippets.push({
          key: node.id,
          node,
          sentenceIndex: null,
          start: 0,
          end: -1,
          text: node.text,
        });
        continue;
      }
      for (let index = 0; index < sentences.length; index += 1) {
        const sentence = sentences[index];
        snippets.push({
          key: `${node.id}:${sentence.start}:${sentence.end}`,
          node,
          sentenceIndex: index,
          start: sentence.start,
          end: sentence.end,
          text: sentence.text,
        });
      }
    }
    return snippets;
  });
  const optionNodes = $derived.by<LoomNodeJSON[]>(() => {
    const node = cursorNode;
    if (!node) return [];
    const parentId = node.recipe !== null && node.parent_id ? node.parent_id : node.id;
    return _orderedChildren(parentId)
      .map((id) => loomTree.nodes.get(id))
      .filter((child): child is LoomNodeJSON => child != null);
  });
  const savedRows = $derived(treeRows.filter((row) => row.node.starred && !row.filteredOut));
  let viewportEl: HTMLDivElement | null = $state(null);
  let lastViewportSize = $state({ width: 0, height: 0 });
  let camera: LoomCamera = $state({ x: 0, y: 0, zoom: 1 });
  let cameraInitialized = $state(false);
  let cameraFitted = $state(false);
  const compactMap = $derived(coarsePointer.current && (cameraFitted || !cameraInitialized || camera.zoom < 1));
  const visibleMapNodes = $derived(graph.nodes.filter(node => {
    const margin = 120;
    return node.x * camera.zoom + camera.x < lastViewportSize.width + margin
      && (node.x + node.width) * camera.zoom + camera.x > -margin
      && node.y * camera.zoom + camera.y < lastViewportSize.height + margin
      && (node.y + node.height) * camera.zoom + camera.y > -margin;
  }));
  let cameraAnimating = $state(false);
  let cameraAnimationTimer: ReturnType<typeof setTimeout> | null = null;
  let dragging = $state(false);
  let dragPointerId: number | null = null;
  let dragOrigin = { x: 0, y: 0, cameraX: 0, cameraY: 0 };
  const touchPointers = new Map<number, GesturePoint>();
  let pinchGesture: {
    distance: number;
    zoom: number;
    worldX: number;
    worldY: number;
  } | null = null;
  let suppressNodeClickUntil = 0;

  function setCameraMotion(enabled: boolean): void {
    if (cameraAnimationTimer !== null) {
      clearTimeout(cameraAnimationTimer);
      cameraAnimationTimer = null;
    }
    cameraAnimating = enabled;
    if (enabled) {
      cameraAnimationTimer = setTimeout(() => {
        cameraAnimating = false;
        cameraAnimationTimer = null;
      }, 220);
    }
  }

  function viewportSize(): { width: number; height: number } {
    return {
      width: viewportEl?.clientWidth ?? 0,
      height: viewportEl?.clientHeight ?? 0,
    };
  }

  function fitView(animate = true): void {
    if (!viewportEl || graph.nodes.length === 0) return;
    setCameraMotion(animate && cameraInitialized);
    cameraFitted = true;
    camera = fitLoomCamera(viewportSize(), graph);
    cameraInitialized = true;
    lastViewportSize = viewportSize();
  }

  function focusInitialView(): void {
    if (!active || loomUiState.view !== "map" || !viewportEl) return;
    if (coarsePointer.current || viewportSize().width < 768) {
      fitView(false);
      return;
    }
    camera.zoom = 0.9;
    if (loomTree.active_node_id) void centerSettled(loomTree.active_node_id);
    else fitView();
  }

  function zoomView(
    nextZoom: number,
    anchor?: { x: number; y: number },
    animate = true,
  ): void {
    const size = viewportSize();
    cameraFitted = false;
    setCameraMotion(animate);
    camera = zoomLoomAt(camera, nextZoom, anchor ?? {
      x: size.width / 2,
      y: size.height / 2,
    });
    cameraInitialized = true;
  }

  function onViewportWheel(ev: WheelEvent): void {
    ev.preventDefault();
    if (ev.ctrlKey || ev.metaKey) {
      const rect = viewportEl?.getBoundingClientRect();
      if (!rect) return;
      const factor = Math.exp(-ev.deltaY * 0.002);
      zoomView(camera.zoom * factor, {
        x: ev.clientX - rect.left,
        y: ev.clientY - rect.top,
      }, false);
      return;
    }
    setCameraMotion(false);
    cameraFitted = false;
    camera.x -= ev.shiftKey && ev.deltaX === 0 ? ev.deltaY : ev.deltaX;
    camera.y -= ev.shiftKey ? 0 : ev.deltaY;
    cameraInitialized = true;
  }

  function pointInViewport(ev: PointerEvent): GesturePoint | null {
    const rect = viewportEl?.getBoundingClientRect();
    if (!rect) return null;
    return {
      x: ev.clientX - rect.left,
      y: ev.clientY - rect.top,
    };
  }

  function captureViewportPointer(pointerId: number): void {
    if (!viewportEl) return;
    try {
      viewportEl.setPointerCapture(pointerId);
    } catch {
      // The pointer may already have ended between events.
    }
  }

  function releaseViewportPointer(pointerId: number): void {
    if (!viewportEl?.hasPointerCapture(pointerId)) return;
    try {
      viewportEl.releasePointerCapture(pointerId);
    } catch {
      // A cancelled pointer releases capture automatically.
    }
  }

  function touchPair(): [GesturePoint, GesturePoint] | null {
    const points = [...touchPointers.values()];
    return points.length >= 2 ? [points[0], points[1]] : null;
  }

  function beginPinch(): void {
    const pair = touchPair();
    if (!pair) return;
    const metrics = pinchMetrics(pair[0], pair[1]);
    if (metrics.distance <= 0) return;
    setCameraMotion(false);
    dragging = false;
    dragPointerId = null;
    pinchGesture = {
      distance: metrics.distance,
      zoom: camera.zoom,
      worldX: (metrics.center.x - camera.x) / camera.zoom,
      worldY: (metrics.center.y - camera.y) / camera.zoom,
    };
    suppressNodeClickUntil = performance.now() + 350;
    for (const pointerId of touchPointers.keys()) captureViewportPointer(pointerId);
  }

  function updatePinch(): void {
    const pair = touchPair();
    if (!pair || !pinchGesture) return;
    cameraFitted = false;
    const metrics = pinchMetrics(pair[0], pair[1]);
    const zoom = scaleFromPinch(
      pinchGesture.zoom,
      pinchGesture.distance,
      metrics.distance,
      LOOM_MIN_ZOOM,
      LOOM_MAX_ZOOM,
    );
    camera = {
      x: metrics.center.x - pinchGesture.worldX * zoom,
      y: metrics.center.y - pinchGesture.worldY * zoom,
      zoom,
    };
    cameraInitialized = true;
  }

  function onViewportPointerDown(ev: PointerEvent): void {
    if (ev.button !== 0) return;
    const target = ev.target as HTMLElement | null;
    if (ev.pointerType === "touch") {
      const point = pointInViewport(ev);
      if (!point) return;
      touchPointers.set(ev.pointerId, point);
      if (touchPointers.size >= 2) {
        beginPinch();
        ev.preventDefault();
        return;
      }
    }
    if (target?.closest(".tree-node-wrap, .loom-view-controls")) return;
    setCameraMotion(false);
    dragging = true;
    dragPointerId = ev.pointerId;
    dragOrigin = {
      x: ev.clientX,
      y: ev.clientY,
      cameraX: camera.x,
      cameraY: camera.y,
    };
    captureViewportPointer(ev.pointerId);
    ev.preventDefault();
  }

  function onViewportPointerMove(ev: PointerEvent): void {
    if (ev.pointerType === "touch" && touchPointers.has(ev.pointerId)) {
      const point = pointInViewport(ev);
      if (point) touchPointers.set(ev.pointerId, point);
      if (pinchGesture && touchPointers.size >= 2) {
        updatePinch();
        suppressNodeClickUntil = performance.now() + 350;
        ev.preventDefault();
        return;
      }
    }
    if (!dragging || dragPointerId !== ev.pointerId) return;
    cameraFitted = false;
    camera.x = dragOrigin.cameraX + ev.clientX - dragOrigin.x;
    camera.y = dragOrigin.cameraY + ev.clientY - dragOrigin.y;
    cameraInitialized = true;
    ev.preventDefault();
  }

  function endViewportGesture(ev: PointerEvent): void {
    releaseViewportPointer(ev.pointerId);
    if (ev.pointerType === "touch") {
      const wasPinching = pinchGesture !== null;
      touchPointers.delete(ev.pointerId);
      if (wasPinching) {
        if (touchPointers.size >= 2) beginPinch();
        else pinchGesture = null;
      }
    }
    if (dragPointerId === ev.pointerId) {
      dragging = false;
      dragPointerId = null;
    }
  }

  function toggleChildren(nodeId: string): void {
    if (collapsedIds.has(nodeId)) collapsedIds.delete(nodeId);
    else collapsedIds.add(nodeId);
    void tick().then(() => fitView());
  }

  let lastGraphNodeCount = 0;
  $effect(() => {
    const viewport = viewportEl;
    if (!active || loomUiState.view !== "map" || !viewport) return;
    const observer = new ResizeObserver(() => {
      const size = viewportSize();
      if (size.width === 0 || size.height === 0) return;
      if (!cameraInitialized) focusInitialView();
      else if (cameraFitted) fitView(false);
      else if (lastViewportSize.width > 0 && lastViewportSize.height > 0) {
        setCameraMotion(false);
        camera = resizeLoomCamera(camera, lastViewportSize, size);
      }
      lastViewportSize = size;
    });
    observer.observe(viewport);
    return () => observer.disconnect();
  });

  $effect(() => {
    if (!active || loomUiState.view !== "map") return;
    const nodeCount = graph.nodes.length;
    void graph.width;
    void graph.height;
    const previousCount = lastGraphNodeCount;
    lastGraphNodeCount = nodeCount;
    if (nodeCount === 0) {
      cameraInitialized = false;
      return;
    }
    void tick().then(() => {
      if (!cameraInitialized || previousCount === 0) focusInitialView();
      else if (cameraFitted) fitView(false);
      else if (nodeCount !== previousCount) centerCurrent();
    });
  });

  // When the active node changes, focus moves to it (only when there's
  // no manual focus selected yet).
  $effect(() => {
    const activeId = loomTree.active_node_id;
    const activeIsVisible = activeId !== null && treeRowById.has(activeId);
    const focusedIsVisible = focusedId !== null && treeRowById.has(focusedId);
    if (!focusedIsVisible) {
      focusedId = activeIsVisible ? activeId : (treeRows[0]?.node.id ?? null);
    }
  });

  function focusedIndex(): number {
    if (!focusedId) return -1;
    return treeRows.findIndex((r) => r.node.id === focusedId);
  }

  async function focusNode(nodeId: string, scroll = true): Promise<void> {
    focusedId = nodeId;
    if (scroll) centerNode(nodeId);
    await tick();
    const el = viewportEl?.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(nodeId)}"]`);
    if (!el) return;
    el.focus({ preventScroll: true });
  }

  function placedRect(nodeId: string): LoomRect | null {
    const placed = graph.nodes.find((node) => node.id === nodeId);
    return placed ? { x: placed.x, y: placed.y, width: placed.width, height: placed.height } : null;
  }

  function centerOn(rect: LoomRect): LoomCamera {
    cameraFitted = false;
    setCameraMotion(cameraInitialized);
    camera = centerLoomRect(viewportSize(), rect, camera.zoom);
    cameraInitialized = true;
    lastViewportSize = viewportSize();
    return { ...camera };
  }

  function centerNode(nodeId: string): void {
    const rect = placedRect(nodeId);
    if (rect && viewportEl) centerOn(rect);
  }

  const nextFrames = () => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
  /** A newer centring supersedes a settling one. */
  let centering = 0;

  /** Centre on a turn and follow it while the turns brought into view replace their estimated
   *  heights with measured ones (which moves everything below and beside them). */
  async function centerSettled(nodeId: string): Promise<void> {
    if (!viewportEl) return;
    const run = ++centering;
    await settleLoomCenter({
      placement: () => (run === centering && viewportEl ? placedRect(nodeId) : null),
      center: centerOn,
      camera: () => ({ ...camera }),
      remeasured: () => nextFrames().then(() => tick()),
    });
  }

  function centerCurrent(): void {
    if (!loomTree.active_node_id) return;
    if (compactMap) void expandMapNode(loomTree.active_node_id);
    else void centerSettled(loomTree.active_node_id);
  }

  async function expandMapNode(nodeId: string): Promise<void> {
    if (performance.now() < suppressNodeClickUntil) return;
    zoomView(1);
    await tick();
    await centerSettled(nodeId);
    await focusNode(nodeId, false);
  }

  function setLoomView(view: LoomView): void {
    loomUiState.view = view;
    closeMenu(false);
    if (view === "map") {
      void tick().then(() => {
        if (loomTree.active_node_id) void focusNode(loomTree.active_node_id, false);
        focusInitialView();
      });
    }
  }

  function readableNodeText(node: LoomNodeJSON): string {
    return loomNodeSnippet(node);
  }

  function nodeRole(node: LoomNodeJSON): string {
    return node.role_label?.trim() || node.role;
  }

  async function usePath(node: LoomNodeJSON): Promise<void> {
    focusedId = node.id;
    if (loomTree.active_node_id !== node.id) await loomNavigate(node.id);
  }

  function moveVisible(dir: -1 | 1): void {
    const index = focusedIndex();
    const next = treeRows[index + dir];
    if (next) void focusNode(next.node.id);
  }

  function moveParent(): void {
    if (!focusedId) return;
    const parentId = loomTree.nodes.get(focusedId)?.parent_id;
    if (parentId && treeRows.some((row) => row.node.id === parentId)) {
      void focusNode(parentId);
    }
  }

  function moveFirstChild(): void {
    if (!focusedId) return;
    const firstChild = _orderedChildren(focusedId)[0];
    if (firstChild && treeRows.some((row) => row.node.id === firstChild)) {
      void focusNode(firstChild);
    }
  }

  // ----------------------------------------- context menu state --

  interface MenuState {
    open: boolean;
    nodeId: string | null;
  }
  let menu: MenuState = $state({ open: false, nodeId: null });
  let menuEl: HTMLElement | null = $state(null);
  let menuPosition = $state({ x: 12, y: 12 });
  let menuPositioned = $state(false);
  let menuOpener: HTMLElement | null = null;
  let menuTrackingFrame: number | null = null;

  function portal(node: HTMLElement): { destroy: () => void } {
    document.body.appendChild(node);
    return {
      destroy: () => node.remove(),
    };
  }

  function openMenu(ev: MouseEvent, nodeId: string): void {
    ev.preventDefault();
    ev.stopPropagation();
    const opener = (ev.currentTarget as HTMLElement | null) ??
      viewportEl?.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(nodeId)}"]`) ?? null;
    void showMenu(nodeId, opener);
  }

  async function showMenu(
    nodeId: string,
    opener: HTMLElement | null,
  ): Promise<void> {
    menuOpener = opener;
    menuPositioned = false;
    focusedId = nodeId;
    setCameraMotion(false);
    await tick();
    menu = {
      open: true,
      nodeId,
    };
    await tick();
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    positionOpenMenu();
    await tick();
    menuPositioned = true;
    trackOpenMenu();
    await tick();
    menuItems()[0]?.focus();
  }

  function positionOpenMenu(): void {
    if (!menuEl || !menu.nodeId) return;
    // Resolve the visible tree item on every frame. The canvas is transformed
    // for pan/zoom, so a stale trigger or its untransformed wrapper can place
    // the fixed, portalled menu hundreds of pixels away from the card.
    const card = viewportEl?.querySelector<HTMLElement>(
      `[data-node-id="${CSS.escape(menu.nodeId)}"]`,
    ) ?? (menuOpener?.isConnected ? menuOpener : null);
    const anchor = card?.getBoundingClientRect();
    const margin = 12;
    const gap = 12;
    const menuRect = { width: menuEl.offsetWidth, height: menuEl.offsetHeight };
    const anchorRect = anchor ?? {
      left: margin,
      right: margin,
      top: margin,
      bottom: margin,
    };
    let x = anchorRect.right + gap;
    let y = anchorRect.top;
    if (x + menuRect.width > window.innerWidth - margin) {
      x = anchorRect.left - menuRect.width - gap;
    }
    if (x < margin) {
      x = anchorRect.left;
      y = anchorRect.bottom + gap;
    }
    if (y + menuRect.height > window.innerHeight - margin) {
      y = Math.max(margin, window.innerHeight - menuRect.height - margin);
    }
    const nextX = Math.max(margin, Math.min(x, window.innerWidth - menuRect.width - margin));
    const nextY = Math.max(margin, y);
    if (
      Math.abs(menuPosition.x - nextX) > 0.5 ||
      Math.abs(menuPosition.y - nextY) > 0.5
    ) {
      menuPosition = { x: nextX, y: nextY };
    }
  }

  function trackOpenMenu(): void {
    if (menuTrackingFrame !== null) cancelAnimationFrame(menuTrackingFrame);
    const update = () => {
      if (!menu.open) {
        menuTrackingFrame = null;
        return;
      }
      positionOpenMenu();
      menuTrackingFrame = requestAnimationFrame(update);
    };
    menuTrackingFrame = requestAnimationFrame(update);
  }

  function openMenuFromKeyboard(nodeId: string): void {
    const opener = viewportEl?.querySelector<HTMLElement>(
      `[data-node-id="${CSS.escape(nodeId)}"]`,
    ) ?? null;
    void showMenu(nodeId, opener);
  }

  function menuItems(): HTMLButtonElement[] {
    if (!menuEl) return [];
    return Array.from(
      menuEl.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)'),
    );
  }

  function onMenuKey(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      ev.stopPropagation();
      closeMenu();
      return;
    }
    if (ev.key === "Tab") {
      closeMenu(false);
      return;
    }
    const items = menuItems();
    if (items.length === 0) return;
    const index = Math.max(0, items.indexOf(document.activeElement as HTMLButtonElement));
    let next = -1;
    if (ev.key === "ArrowDown") next = (index + 1) % items.length;
    else if (ev.key === "ArrowUp") next = (index - 1 + items.length) % items.length;
    else if (ev.key === "Home") next = 0;
    else if (ev.key === "End") next = items.length - 1;
    if (next >= 0) {
      ev.preventDefault();
      ev.stopPropagation();
      items[next].focus();
    }
  }

  function closeMenu(restoreFocus = true): void {
    const opener = menuOpener;
    const nodeId = menu.nodeId;
    menu = { open: false, nodeId: null };
    menuPositioned = false;
    if (menuTrackingFrame !== null) cancelAnimationFrame(menuTrackingFrame);
    menuTrackingFrame = null;
    menuEl = null;
    if (restoreFocus) {
      void tick().then(() => {
        if (opener?.isConnected) opener.focus({ preventScroll: true });
        else if (nodeId) void focusNode(nodeId, false);
      });
    }
  }

  // ----------------------------------------- dialogs --

  let dialog: LoomDialog | null = $state(null);
  let asideEl: HTMLElement | null = $state(null);
  let createdBranchId = $state<string | null>(null);

  /** After a dialog closes, return focus to its trigger, else to the
   *  focused (or active) turn. */
  function restoreFocusAfterDialog(trigger: HTMLElement | null): void {
    if (trigger?.isConnected) trigger.focus({ preventScroll: true });
    else {
      const fallback = focusedId && treeRowById.has(focusedId)
        ? focusedId
        : (loomTree.active_node_id ?? treeRows[0]?.node.id ?? null);
      if (fallback) void focusNode(fallback, false);
    }
  }

  async function showFirstSearchMatch(): Promise<void> {
    loomUiState.view = "map";
    await tick();
    searchInput?.focus();
    await showSearchMatch();
  }

  /** When App.svelte fires Ctrl+R/E/B/N/D it bumps
   *  ``loomUiState.modalRequest.seq``.  Mirror that into the local
   *  modal state. */
  let _lastSeenSeq = $state(0);
  $effect(() => {
    const req = loomUiState.modalRequest;
    if (req.seq !== _lastSeenSeq && req.kind) {
      _lastSeenSeq = req.seq;
      void dialog?.open(req.kind, req.nodeId, req.text, req.n);
    }
  });

  // --------------------------------------- click + global keys --

  function onNodeClick(node: LoomNodeJSON, ev: MouseEvent): void {
    if (performance.now() < suppressNodeClickUntil) {
      ev.preventDefault();
      ev.stopPropagation();
      return;
    }
    // Ctrl/Cmd-click on a generated node toggles its multi-select
    // membership for the cross-branch diff drawer.  Plain click
    // still navigates.
    if ((ev.ctrlKey || ev.metaKey) && node.recipe !== null) {
      ev.preventDefault();
      ev.stopPropagation();
      toggleNodeSelection(node.id);
      void focusNode(node.id, false);
      return;
    }
    void focusNode(node.id, false);
    void loomNavigate(node.id);
  }

  async function growFromNode(node: LoomNodeJSON): Promise<void> {
    if (node.recipe === null) {
      await loomContinueFromCommitted(node.id, { n: 1 });
      return;
    }
    if (loomTree.active_node_id !== node.id) await loomNavigate(node.id);
    await loomRegenerateActive(1);
  }

  function branchFromNode(node: LoomNodeJSON, trigger: HTMLElement | null): void {
    void dialog?.open("branch", node.id, node.text ?? "", 1, trigger);
  }

  /** Keep the generated prefix through this token and resample everything
   * after it. Passing the boundary's original token id through the existing
   * fork protocol preserves every token up to the branch point exactly. */
  async function branchAfterToken(node: LoomNodeJSON, tokenIndex: number): Promise<void> {
    if (genStatus.active) {
      pushToast("Finish or stop the current reply before starting another branch.");
      return;
    }
    const token = node.tokens?.[tokenIndex];
    if (token?.raw_index == null || token.token_id == null) {
      pushToast("This saved reply does not contain an exact token boundary.", { kind: "error" });
      return;
    }
    try {
      await sendFork(node.id, token.raw_index, token.token_id, true);
    } catch (error) {
      pushToast(userFacingError(
        error,
        "A new continuation could not be started from that word.",
      ), { kind: "error" });
    }
  }

  /** A Loom token is a branch point, not an immediate generation command.
   * Move onto its saved path, then open the shared token workbench at the
   * exact token so replacement, readouts, and continuation remain separate,
   * reversible choices. */
  async function inspectLoomToken(node: LoomNodeJSON, tokenIndex: number): Promise<void> {
    if (!node.tokens?.[tokenIndex]) {
      pushToast("That token is no longer available in this branch.", { kind: "error" });
      return;
    }
    if (loomTree.active_node_id !== node.id) {
      await loomNavigate(node.id);
    }
    await tick();
    const turnIdx = chatLog.turns.findIndex((turn) => turn.nodeId === node.id);
    if (turnIdx < 0) {
      pushToast("That branch could not be opened for token editing.", { kind: "error" });
      return;
    }
    openDrawer("token_drilldown", {
      turnIdx,
      tokenIdx: tokenIndex,
      initialTab: "logits",
    });
  }

  /** True when a keydown originated in a text-entry element (an ``<input>``,
   *  ``<textarea>``, ``<select>``, or a ``contenteditable`` host) — used to
   *  keep the sidebar's bare-key nav shortcuts from stealing keystrokes meant
   *  for a nested field. */
  function _isEditableTarget(target: EventTarget | null): boolean {
    const el = target as HTMLElement | null;
    if (!el) return false;
    const tag = el.tagName;
    return (
      tag === "INPUT" ||
      tag === "TEXTAREA" ||
      tag === "SELECT" ||
      el.isContentEditable
    );
  }

  function onSidebarKey(ev: KeyboardEvent): void {
    if (dialog?.isOpen() || menu.open) return;
    const target = ev.target as HTMLElement | null;
    const treeItem = target?.closest<HTMLElement>('[role="treeitem"]') ?? null;
    if (!treeItem || !viewportEl?.contains(treeItem)) return;
    if (target !== treeItem && target?.closest("button, a, [role='button'], [role='menuitem']")) {
      return;
    }
    const k = ev.key;
    if (k !== "Escape" && _isEditableTarget(ev.target)) return;
    if (k === "ArrowDown" || k === "j") {
      ev.preventDefault();
      moveVisible(+1);
      return;
    }
    if (k === "ArrowUp" || k === "k") {
      ev.preventDefault();
      moveVisible(-1);
      return;
    }
    if (k === "ArrowLeft" || k === "h") {
      ev.preventDefault();
      moveParent();
      return;
    }
    if (k === "ArrowRight" || k === "l") {
      ev.preventDefault();
      moveFirstChild();
      return;
    }
    if (k === "Home" && treeRows.length > 0) {
      ev.preventDefault();
      void focusNode(treeRows[0].node.id);
      return;
    }
    if (k === "End" && treeRows.length > 0) {
      ev.preventDefault();
      void focusNode(treeRows[treeRows.length - 1].node.id);
      return;
    }
    if (k === "ContextMenu" || (k === "F10" && ev.shiftKey)) {
      if (focusedId) {
        ev.preventDefault();
        openMenuFromKeyboard(focusedId);
      }
      return;
    }
    if (k === "Enter" || k === " ") {
      if (focusedId) {
        ev.preventDefault();
        if (compactMap) void expandMapNode(treeItem.dataset.nodeId ?? focusedId);
        else void loomNavigate(focusedId);
      }
      return;
    }
    if (k === "s" && focusedId) {
      ev.preventDefault();
      const node = loomTree.nodes.get(focusedId);
      void loomStar(focusedId, !node?.starred);
      return;
    }
    if (k === "n" && focusedId) {
      ev.preventDefault();
      const node = loomTree.nodes.get(focusedId);
      void dialog?.open("note", focusedId, node?.notes ?? "");
      return;
    }
    if (k === "/") {
      ev.preventDefault();
      void dialog?.open("search", focusedId ?? loomTree.active_node_id);
      return;
    }
    if (k === "Escape") {
      if (loomUiState.filterHelpOpen) {
        ev.preventDefault();
        loomUiState.filterHelpOpen = false;
      }
      return;
    }
  }

  // Close context menu on outside click or Escape (window-level).
  function onWindowClick(ev: MouseEvent): void {
    if (!menu.open) return;
    const t = ev.target as HTMLElement | null;
    if (t && t.closest(".loom-menu")) return;
    closeMenu(false);
  }
  function onWindowKey(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      if (menu.open) { closeMenu(); ev.preventDefault(); return; }
      if (dialog?.isOpen()) { dialog.close(); ev.preventDefault(); return; }
    }
  }

  // ---------------------------------------- menu actions --

  async function menuRegenerate(): Promise<void> {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    if (loomTree.active_node_id !== nid) await loomNavigate(nid);
    await dialog?.open("regenerate", nid, "", 1, trigger);
  }
  async function menuEdit(): Promise<void> {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    await dialog?.open("edit", nid, node?.text ?? "", 1, trigger);
  }
  async function menuBranch(): Promise<void> {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    await dialog?.open("branch", nid, node?.text ?? "", 1, trigger);
  }
  async function menuSwapSeat(): Promise<void> {
    // Seat-swap branch (the cast model): a sibling with the same text
    // and the seat flipped — identical bytes, one seat bit different,
    // the controlled experiment on the seat prior.  No modal: the text
    // is by definition unchanged.
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    const newId = await loomSwapSeat(nid);
    if (newId) await loomNavigate(newId);
  }
  async function menuNavigate(): Promise<void> {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    await loomNavigate(nid);
  }
  async function menuDelete(): Promise<void> {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    await dialog?.open("delete", nid, "", 1, trigger);
  }
  async function menuStar(): Promise<void> {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    await loomStar(nid, !node?.starred);
  }
  async function menuNote(): Promise<void> {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    await dialog?.open("note", nid, node?.notes ?? "", 1, trigger);
  }

  // ---------------------------------- phase 5 context-menu actions --

  function menuPin(): void {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    pinNodeForComparison(nid);
  }

  function menuFanOut(): void {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    // Anchor the fan-out on the committed turn.  If the user clicked a
    // generated node, walk up to its parent.
    let anchorId = nid;
    if (node && node.recipe !== null && node.parent_id) {
      anchorId = node.parent_id;
    }
    void dialog?.open("fanout", anchorId, "0.0, 0.3, 0.6", 1, trigger);
  }

  function menuCompareBranch(): void {
    const nid = menu.nodeId;
    closeMenu(false);
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    // A committed node compares generated children; a generated node
    // compares its generated sibling set.
    let parentId: string | null = null;
    if (node && node.recipe === null && node.role !== "system") parentId = nid;
    else if (node && node.recipe !== null) parentId = node.parent_id;
    if (!parentId) return;
    const generatedChildren = (loomTree.children_of.get(parentId) ?? []).filter(
      (id) => loomTree.nodes.get(id)?.recipe !== null,
    );
    if (generatedChildren.length < 2) return;
    openDrawer("node_compare", {
      node_ids: generatedChildren,
      parent_id: parentId,
    });
  }

  function menuToggleSelection(): void {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    toggleNodeSelection(nid);
  }

  function menuCompareSelected(): void {
    if (menu.open) closeMenu(false);
    if (nodeSelection.ids.length < 2) return;
    openDrawer("node_compare", { node_ids: [...nodeSelection.ids] });
  }

  function menuRegenWithMode(): void {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    void dialog?.open("regen_mode", nid, "", 1, trigger);
  }

  // ---------------------------------------- refresh / error UI --

  function fullRefresh(): void {
    // Re-fetch — useful if the user suspects drift.
    void refreshLoomTree();
  }

  onMount(() => {
    const initialId = focusedId ?? loomTree.active_node_id ?? treeRows[0]?.node.id;
    if (initialId && treeRows.some((row) => row.node.id === initialId)) {
      void focusNode(initialId, false);
    }
    void tick().then(focusInitialView);
    return () => {
      if (cameraAnimationTimer !== null) clearTimeout(cameraAnimationTimer);
      if (menuTrackingFrame !== null) cancelAnimationFrame(menuTrackingFrame);
      touchPointers.clear();
    };
  });

  onInterfaceMount(() => registerInterfaceController("loom", {
    schema: loomViewSchema,
    read: () => ({ values: { zoom: camera.zoom, camera: { x: camera.x, y: camera.y }, collapsed_ids: [...collapsedIds] }, search: { count: searchResults.length, index: searchIndex, node_id: searchNodeId }, visible: active }),
    update: async (change) => {
      if (change.collapsed_ids && (change.collapsed_ids as string[]).some(id => !loomTree.nodes.has(id))) throw new ToolError("NOT_FOUND", "A collapsed node is no longer in this tree.");
      if ((change.zoom !== undefined || change.camera || change.search_match) && (!active || !viewportEl || viewportSize().width === 0)) throw new ToolError("NOT_READY", "Open the visible Loom map before adjusting its camera.");
      if (change.camera && graph.nodes.length === 0) throw new ToolError("NOT_FOUND", "The Loom map has no nodes to frame.");
      if (change.search_match && !searchResults.length) throw new ToolError("NOT_FOUND", "This search has no matching nodes.");
      if (change.collapsed_ids !== undefined) { collapsedIds.clear(); for (const id of change.collapsed_ids as string[]) collapsedIds.add(id); }
      if (change.zoom !== undefined) zoomView(change.zoom as number, undefined, false);
      if (change.camera === "fit") fitView(false);
      if (change.camera === "current") centerCurrent();
      if (change.search_match) { if (change.search_match === "first") searchNodeId = null; await showSearchMatch(change.search_match === "previous" ? -1 : change.search_match === "next" ? 1 : 0); }
    },
  }));
</script>

<svelte:window onclick={onWindowClick} onkeydown={onWindowKey} onresize={positionOpenMenu} />

<aside
  class="loom-sidebar"
  aria-label="Conversation loom panel"
  bind:this={asideEl}
>
  <div class="loom-top">
    <Tabs label="Loom views" scroll value={loomUiState.view} onchange={(view) => setLoomView(view)} options={[
      { value: "weave", label: "Weave" },
      { value: "map", label: "Map" },
      { value: "path", label: "Path", count: String(pathSnippets.length) },
      { value: "options", label: "Options", count: String(optionNodes.length) },
      { value: "saved", label: "Starred", count: String(savedRows.length) },
    ]} />
  </div>

  {#if headersVisible}
  <!-- Tree-scope actions: clear, save and open act on the whole conversation tree, not on the
       active path, so they live with the loom rather than in the chat. -->
  <div class="loom-actions" id="loom-tools-header" inert={!headersVisible} transition:expand>
    <button type="button" class="dw-text" onclick={compareBranch} disabled={comparableNodes.length < 2}
      {...{ "aria-description": "Compare alternate replies to the same message" }}>
      {comparableNodes.length >= 2 ? `Compare ${comparableNodes.length}` : "Compare"}
    </button>
    <button type="button" class="dw-text" onclick={() => openDrawer("save_conversation")}>Save</button>
    <button type="button" class="dw-text" onclick={() => openDrawer("load_conversation")}>Open</button>
    <button type="button" class="dw-text" onclick={clearChat}
      {...{ "aria-description": "Start a new path from the root; keep all existing branches" }}>Start over</button>
    <button type="button" class="dw-text" disabled={!loomTree.active_node_id || loomTree.active_node_id === loomTree.root_id || loomNodeIntersectsGeneration(loomTree.active_node_id)}
      {...{ "aria-description": "Remove the current turn and all branches that follow it" }}
      onclick={() => void dialog?.open("delete", loomTree.active_node_id)}>Cut branch…</button>
    <button type="button" class="dw-text" disabled={genStatus.active || !loomTree.loaded || loomTree.nodes.size <= 1}
      {...{ "aria-description": "Save this conversation and start with an empty loom" }}
      onclick={() => void dialog?.open("clear", loomTree.root_id)}>Clear loom…</button>
    <button type="button" class="dw-text" onclick={fullRefresh} aria-label="Refresh">Refresh</button>
  </div>
  {/if}

  {#if loomUiState.view === "map" || loomUiState.view === "saved"}
    <div class="filter-bar">
      <label class="filter-field">
        <span class="sr-only">{filterState.mode === "text" ? "Search messages" : "Advanced filters"}</span>
        <input
          type="search"
          class="filter-input"
          bind:this={searchInput}
          value={filterState.expr}
          oninput={(event) => editFilter(event.currentTarget.value)}
          onkeydown={onFilterKey}
          placeholder={filterState.mode === "text" ? "Search every branch" : "starred, text:fox"}
          aria-invalid={filterState.error ? true : undefined}
          aria-describedby="loom-search-status"
        />
        {#if filterState.expr}
          <button type="button" class="clear" onclick={() => { clearTreeFilter(); searchInput?.focus(); }}
            aria-label="Clear search"><Icon name="close" /></button>
        {/if}
      </label>
      <Segmented label="Search mode" value={filterState.mode} onchange={changeSearchMode}
        options={[{ value: "text", label: "Text" }, { value: "advanced", label: "Advanced" }]} />
      {#if filterState.mode === "advanced"}
        <button type="button" class="dw-quiet" onclick={() => void applyTreeFilter(filterState.expr)}>Apply</button>
        <!-- The filter grammar, opened in place without taking focus from the field; it mirrors
             tree_filter.py's forms plus the client-side sort directive. -->
        <button type="button" class="dw-text" aria-pressed={loomUiState.filterHelpOpen}
          onclick={() => (loomUiState.filterHelpOpen = !loomUiState.filterHelpOpen)}
          aria-label="How advanced filters work" aria-expanded={loomUiState.filterHelpOpen}>How</button>
      {/if}
    </div>
    <div class="search-feedback">
      <p id="loom-search-status" class="meta" role="status">
        <MorphText text={filterState.loading ? "Searching…" : filterState.error ?? (filterState.matchingIds !== null
          ? searchResults.length === 0 ? "No messages match"
            : `${searchIndex >= 0 ? `${searchIndex + 1} of ` : ""}${searchResults.length} matching ${searchResults.length === 1 ? "message" : "messages"}`
          : "")} />
      </p>
      {#if searchResult && searchExcerpt}
        <div class="search-result">
          <button type="button" class="search-excerpt" aria-label="Show match in map" onclick={() => void showSearchMatch()}>
            <span class="meta">{nodeRole(searchResult)}</span>
            <span dir="auto">{searchExcerpt.before}<mark>{searchExcerpt.match}</mark>{searchExcerpt.after}</span>
          </button>
          <IconButton icon="up" label="Previous match" onclick={() => void showSearchMatch(-1)} />
          <IconButton icon="down" label="Next match" onclick={() => void showSearchMatch(1)} />
        </div>
      {/if}
    </div>
  {/if}

  {#if filterState.mode === "advanced" && loomUiState.filterHelpOpen && (loomUiState.view === "map" || loomUiState.view === "saved")}
    <div class="filter-help" role="region" aria-label="Filter grammar help" transition:expand>
      <p>Commas join filters; a branch must match every one.</p>
      <ul>
        <li><code>text:&lt;words&gt;</code> searches message text</li>
        <li><code>starred</code> keeps saved branches</li>
        <li><code>&lt;measurement&gt; &gt; &lt;number&gt;</code> filters by a reading</li>
        <li><code>agg:</code>, <code>any:</code> or <code>last:</code> chooses where to measure</li>
        <li><code>sort:surprise</code> or <code>sort:confidence</code> orders replies</li>
      </ul>
      <ul class="examples">
        <li><code>agg:angry.calm &gt; 0.4</code></li>
        <li><code>starred, text:fox</code></li>
        <li><code>sort:surprise, agg:honest &lt; 0</code></li>
      </ul>
    </div>
  {/if}

  {#if loomUiState.siblingSort !== "default" || nodeSelection.ids.length > 0 || cursorNode}
    <div class="loom-status meta">
      <!-- Edges are always surprise-weighted (stroke width and opacity thicken toward surprising
           children); a sort directive orders the replies too. -->
      {#if loomUiState.siblingSort !== "default"}<span {...{ "aria-description": "sibling sort" }}>sort:{loomUiState.siblingSort}</span>{/if}
      {#if cursorNode}
        <span class="focused-branch" aria-label="Focused branch summary">
          {#if cursorNode.id === createdBranchId}<span role="status">Branch created</span>{/if}
          <span><MorphText text={cursorNode.role_label || cursorNode.role} numbers={false} /></span>
          <span><MorphText text={`depth ${rowById.get(cursorNode.id)?.depth ?? 0}`} /></span>
          <span><MorphText text={`${cursorNode.tokens?.length ?? 0} tokens`} /></span>
        </span>
      {/if}
      {#if nodeSelection.ids.length > 0}
        <span class="selection-bar">
          <span><MorphText text={`${nodeSelection.ids.length} selected`} /></span>
          <button type="button" class="dw-text" onclick={menuCompareSelected} disabled={nodeSelection.ids.length < 2}>Compare</button>
          <button type="button" class="dw-text" onclick={clearNodeSelection}>Clear</button>
        </span>
      {/if}
    </div>
  {/if}

  <div class="weave-host" hidden={loomUiState.view !== "weave" || !!loomTree.error}>
    {#key loomTree.root_id}
      <LoomWeave
        active={active && loomUiState.view === "weave"}
        oninspect={(node, index) => void inspectLoomToken(node, index)}
        onwrite={branchFromNode}
      />
    {/key}
  </div>

  {#if loomTree.error}
    <div class="empty" role="alert">
      <p>The loom could not load.</p>
      <button type="button" class="dw-quiet" onclick={fullRefresh}>Retry</button>
    </div>
  {:else if loomUiState.view !== "weave"}
  {#if treeRows.length === 0}
    <div class="empty">
      <p>No branches yet</p>
    </div>
  {:else if loomUiState.view === "map"}
    <div class="loom-map-bar">
      <div class="loom-map-summary meta">
        <span><MorphText text={treeRows.length} /> {treeRows.length === 1 ? "turn" : "turns"}</span>
        <span><MorphText text={branchPointCount} /> {branchPointCount === 1 ? "fork" : "forks"}</span>
        {#if visibleTokenCount > 0}
          <span><MorphText text={visibleTokenCount} /> {visibleTokenCount === 1 ? "token" : "tokens"}</span>
        {/if}
        <span class="path-key"><i aria-hidden="true"></i>current path</span>
      </div>
      <div class="loom-view-controls" role="group" aria-label="Loom view controls">
        <button type="button" class="step" onclick={() => zoomView(camera.zoom - 0.12)} data-cursor="zoom-out"
          disabled={camera.zoom <= LOOM_MIN_ZOOM} aria-label="Zoom out">−</button>
        <output class="meta" aria-label="Loom zoom"><RollingNumber value={Math.round(camera.zoom * 100)} />%</output>
        <button type="button" class="step" onclick={() => zoomView(camera.zoom + 0.12)} data-cursor="zoom-in"
          disabled={camera.zoom >= LOOM_MAX_ZOOM} aria-label="Zoom in">+</button>
        <button type="button" class="dw-text" onclick={() => fitView()} aria-label="Fit whole loom">Fit</button>
        <button type="button" class="dw-text" onclick={centerCurrent} aria-label="Center current path">Current</button>
      </div>
    </div>
    <div
      class="tree-scroll loom-viewport"
      class:dragging
      class:camera-animating={cameraAnimating}
      role="tree"
      aria-label="Conversation loom"
      aria-describedby="loom-touch-help"
      aria-multiselectable="true"
      tabindex="-1"
      bind:this={viewportEl}
      onwheel={onViewportWheel}
      onpointerdown={onViewportPointerDown}
      onpointermove={onViewportPointerMove}
      onpointerup={endViewportGesture}
      onpointercancel={endViewportGesture}
      onlostpointercapture={endViewportGesture}
    >
      <p id="loom-touch-help" class="sr-only">Drag to move, pinch to zoom</p>
      {#if active}
      <div
        class="loom-canvas"
        class:camera-animating={cameraAnimating}
        style={`transform:translate(${camera.x}px,${camera.y}px) scale(${camera.zoom})`}
        data-loom-nodes={graph.nodes.length}
        data-loom-edges={graph.edges.length}
        data-loom-zoom={camera.zoom.toFixed(2)}
        data-loom-camera-x={camera.x.toFixed(2)}
        data-loom-camera-y={camera.y.toFixed(2)}
      >
        <svg
          class="loom-edges"
          width={lastViewportSize.width / camera.zoom}
          height={lastViewportSize.height / camera.zoom}
          style={`left:${-camera.x / camera.zoom}px;top:${-camera.y / camera.zoom}px`}
          viewBox={`${-camera.x / camera.zoom} ${-camera.y / camera.zoom} ${lastViewportSize.width / camera.zoom} ${lastViewportSize.height / camera.zoom}`}
          aria-hidden="true"
        >
          {#each graph.edges as edge (`${edge.parentId}|${edge.childId}`)}
            <path
              d={edge.path}
              class="edge-line"
              class:active-edge={edgeIsActive(edge)}
              class:filtered-edge={edgeIsFiltered(edge)}
              style={edgeStyle(edge)}
              data-loom-edge={`${edge.parentId}|${edge.childId}`}
            />
          {/each}
          {#each branchJunctions as junction (junction.id)}
            <circle
              class="junction"
              class:active-junction={mapRowById.get(junction.id)?.isActivePath}
              cx={junction.x}
              cy={junction.y}
              r="3.5"
              data-loom-junction={junction.id}
            />
          {/each}
        </svg>

        {#each visibleMapNodes as placed (placed.id)}
          {@const row = mapRowById.get(placed.id)!}
          {@const segment = segmentById.get(placed.id)!}
          <div
            class="tree-node-wrap"
            role="none"
            class:filtered-out={row.filteredOut}
            class:search-match={filterState.matchingIds !== null && !row.filteredOut}
            class:search-current={searchSegmentId === placed.id}
            class:selected={!segment.shared && selectionSet.has(row.node.id)}
            class:pinned={!segment.shared && pinnedComparison.nodeId === row.node.id}
            class:active-path-node={row.isActivePath}
            class:current-node={!segment.shared && loomTree.active_node_id === row.node.id}
            class:dead-node={row.isDead}
            data-loom-node-id={segment.id}
            data-loom-shared={segment.shared ? segment.memberIds.length : undefined}
            data-loom-depth={placed.depth}
            style={`left:${placed.x}px;top:${placed.y}px;width:${placed.width}px;height:${placed.height}px;--branch-depth:${placed.depth}`}
          >
            {#if compactMap}
              <button
                type="button"
                class="map-summary"
                role="treeitem"
                aria-level={row.depth + 1}
                aria-selected={selectionSet.has(row.node.id)}
                aria-label={`Zoom into ${nodeRole(row.node)}: ${readableNodeText(row.node).slice(0, 100)}`}
                data-node-id={segment.id}
                onfocus={() => { focusedId = row.node.id; }}
                onkeydown={onSidebarKey}
                onclick={() => void expandMapNode(placed.id)}
              >
                <span class="meta">{nodeRole(row.node)}{segment.shared ? ` · ${segment.memberIds.length} paths` : ""}</span>
                <span class="map-summary-text">{readableNodeText(row.node).slice(0, 240)}</span>
                <span class="meta">{segment.end - segment.start} tokens</span>
              </button>
            {:else}
            <LoomNode
              onmeasure={(height) => { if (measuredHeights.get(segment.id) !== height) measuredHeights.set(segment.id, height); }}
              node={row.node}
              displayId={segment.id}
              tokenStart={segment.start}
              tokenEnd={segment.end}
              sharedCount={segment.shared ? segment.memberIds.length : 0}
              sharedUsesActivePath={segment.shared && activePathSet.has(row.node.id)}
              onActivePath={row.isActivePath}
              focused={!segment.shared && focusedId === row.node.id}
              current={!segment.shared && loomTree.active_node_id === row.node.id}
              selected={selectionSet.has(row.node.id)}
              level={row.depth + 1}
              hasChildren={(loomTree.children_of.get(row.node.id)?.length ?? 0) > 0}
              collapsed={collapsedIds.has(row.node.id)}
              dead={row.isDead}
              streaming={!segment.shared && loomTree.pendingNodeId === row.node.id}
              ring={segment.shared ? null : ringFor(row.node)}
              weightBadge={segment.shared ? null : weightBadgeFor(row.node)}
              steerLabel={segment.shared ? null : steerLabelFor(row.node)}
              forkLabel={segment.shared ? null : segment.start > 0 ? "continuation" : divergenceLabel(row.node)}
              onclick={segment.shared ? undefined : (ev) => onNodeClick(row.node, ev)}
              onfocus={segment.shared ? undefined : () => { focusedId = row.node.id; }}
              onkeydown={segment.shared ? undefined : onSidebarKey}
              oncontextmenu={segment.shared ? undefined : (ev) => openMenu(ev, row.node.id)}
              onactions={(ev) => openMenu(ev, row.node.id)}
              ongrow={() => void growFromNode(row.node)}
              onbranch={(ev) => branchFromNode(row.node, ev.currentTarget as HTMLElement)}
              onselecttoken={(tokenIndex) => void inspectLoomToken(row.node, tokenIndex)}
              onbranchsentence={(tokenIndex) => void branchAfterToken(row.node, tokenIndex)}
              sentenceBranchAvailable={!genStatus.active}
              ontogglechildren={() => toggleChildren(row.node.id)}
            />
            {/if}
          </div>
        {/each}
      </div>
      {/if}
    </div>
  {:else if loomUiState.view === "path"}
    <section class="loom-projection" aria-label="Current path">
      <div class="path-stream" role="list" aria-label="Current conversation path">
        {#each pathSnippets as snippet, index (snippet.key)}
          {@const boundary = snippet.end >= 0 ? snippet.node.tokens?.[snippet.end] : null}
          {@const canBranch = snippet.sentenceIndex !== null && !genStatus.active && boundary?.raw_index != null && boundary?.token_id != null}
          <article
            class="path-snippet"
            class:current-snippet={loomTree.active_node_id === snippet.node.id && index === pathSnippets.length - 1}
            class:model={snippet.node.role === "assistant"}
            data-path-snippet={snippet.key}
            role="listitem"
          >
            <span class="dw-pip" class:is-model={snippet.node.role === "assistant"} aria-hidden="true"></span>
            <div class="snippet-body">
              <header class="row-head meta">
                <span>{nodeRole(snippet.node)}{snippet.sentenceIndex !== null ? ` · sentence ${snippet.sentenceIndex + 1}` : ""}</span>
                <button type="button" class="more" aria-label={`Actions for ${nodeRole(snippet.node)}`}
                  onclick={(ev) => openMenu(ev, snippet.node.id)}>•••</button>
              </header>
              {#if snippet.sentenceIndex !== null && snippet.node.tokens}
                <p class="snippet-tokens">{#each tokenWords(snippet.node.tokens, snippet.start, snippet.end) as word (word.start)}{@const lead = leadingSpace(snippet.node.tokens[word.start].text)}{lead}<span class="word">{#each snippet.node.tokens.slice(word.start, word.end + 1) as token, offset (`${token.raw_index ?? word.start + offset}:${token.token_id ?? token.text}`)}{@const tokenIndex = word.start + offset}<button
                      type="button"
                      class="path-token"
                      data-cursor="inspect"
                      data-loom-token-node={snippet.node.id}
                      data-token-index={tokenIndex}
                      {...{ "aria-description": "Open branch-point tools" }}
                      aria-label={`Open token ${tokenIndex + 1}: ${token.text.trim() || "whitespace"}`}
                      aria-haspopup="dialog"
                      onclick={() => void inspectLoomToken(snippet.node, tokenIndex)}
                    >{(offset === 0 ? token.text.slice(lead.length) : token.text) || "∅"}</button>{/each}</span>{/each}</p>
              {:else}
                <button type="button" class="snippet-text" onclick={() => void usePath(snippet.node)}>
                  {readableNodeText(snippet.node)}
                </button>
              {/if}
              <footer class="dw-row">
                <button type="button" class="dw-text" onclick={() => void usePath(snippet.node)}>Use this point</button>
                {#if snippet.sentenceIndex !== null}
                  <button
                    type="button"
                    class="dw-text"
                    disabled={!canBranch}
                    {...{ "aria-description": (canBranch
                      ? "Keep everything through this sentence and generate a new continuation"
                      : "Exact sentence replay is unavailable while the model is busy") }}
                    onclick={() => void branchAfterToken(snippet.node, snippet.end)}
                  >Branch after this</button>
                {/if}
              </footer>
            </div>
          </article>
        {/each}
      </div>
      {#if loomTree.active_node_id}
        <button type="button" class="dw-text see-options" onclick={() => setLoomView("options")}>See alternatives</button>
      {/if}
    </section>
  {:else if loomUiState.view === "options"}
    <section class="loom-projection" aria-label="Next options">
      {#if cursorNode}
        <div class="projection-head">
          <p class="anchor" dir="auto">{readableNodeText(cursorNode)}</p>
          <div class="dw-row">
            <button type="button" class="dw-button" disabled={genStatus.active} onclick={() => void growFromNode(cursorNode)}>
              <MorphText text={genStatus.active ? "Generating…" : "Generate another"} />
            </button>
            {#if cursorNode.parent_id}
              <button type="button" class="dw-quiet" onclick={(ev) => branchFromNode(cursorNode, ev.currentTarget as HTMLElement)}>Write an alternative</button>
            {/if}
          </div>
        </div>
      {:else}
        <p class="meta">Choose a point on the map</p>
      {/if}
      {#if optionNodes.length === 0}
        <div class="empty"><p>No alternatives here yet</p></div>
      {:else}
        <div class="option-grid" role="list" aria-label="Alternative paths">
          {#each optionNodes as node, index (node.id)}
            <article
              class="option-card"
              class:active-option={loomTree.active_node_id === node.id}
              role="listitem"
              data-option-node={node.id}
            >
              <header class="row-head meta">
                <span>{index + 1} · {nodeRole(node)}{loomTree.active_node_id === node.id ? " · current" : ""}{node.starred ? " · starred" : ""}</span>
                <button type="button" class="more" aria-label={`Actions for option ${index + 1}`} onclick={(ev) => openMenu(ev, node.id)}>•••</button>
              </header>
              <button type="button" class="option-preview" dir="auto" onclick={() => void usePath(node)}>
                {readableNodeText(node)}
              </button>
              <footer class="dw-row">
                <button type="button" class="dw-quiet" onclick={() => void usePath(node)}>Use this path</button>
                <button type="button" class="dw-text" onclick={() => void loomStar(node.id, !node.starred)}>
                  <MorphText text={node.starred ? "Unstar" : "Star"} />
                </button>
              </footer>
            </article>
          {/each}
        </div>
      {/if}
    </section>
  {:else}
    <section class="loom-projection" aria-label="Starred points">
      {#if savedRows.length === 0}
        <div class="empty">
          <p><MorphText text={filterState.matchingIds !== null ? "No starred messages match" : "Nothing starred yet"} /></p>
        </div>
      {:else}
        <div class="saved-list" role="list" aria-label="Starred conversation points">
          {#each savedRows as row (row.node.id)}
            <article class="saved-card" role="listitem" data-saved-node={row.node.id}>
              <span class="dw-pip" class:is-model={row.node.role === "assistant"} aria-hidden="true"></span>
              <div class="saved-copy">
                <span class="meta">{nodeRole(row.node)}</span>
                <strong dir="auto">{readableNodeText(row.node)}</strong>
                {#if row.node.notes}<p>{row.node.notes}</p>{/if}
              </div>
              <div class="dw-row saved-actions">
                <button type="button" class="dw-text" onclick={() => void usePath(row.node)}>Use this path</button>
                <button type="button" class="dw-text" onclick={() => void loomStar(row.node.id, false)}>Unstar</button>
                <button type="button" class="more" aria-label={`Actions for ${nodeRole(row.node)}`} onclick={(ev) => openMenu(ev, row.node.id)}>•••</button>
              </div>
            </article>
          {/each}
        </div>
      {/if}
    </section>
  {/if}
  {/if}

</aside>

{#if menu.open && menu.nodeId}
  {@const menuNode = loomTree.nodes.get(menu.nodeId)}
  {@const menuPreview = menuNode ? loomNodeSnippet(menuNode) : "Empty turn"}
  {@const menuRole = menuNode?.role_label?.trim() || menuNode?.role || "turn"}
  {@const cmpParentId =
    menuNode?.recipe === null && menuNode?.role !== "system"
      ? menu.nodeId
      : menuNode?.recipe !== null
        ? (menuNode?.parent_id ?? null)
        : null}
  {@const cmpCount = cmpParentId
    ? (loomTree.children_of.get(cmpParentId) ?? []).filter(
        (id) => loomTree.nodes.get(id)?.recipe !== null,
      ).length
    : 0}
  <div
    class="loom-menu"
    class:positioned={menuPositioned}
    style={`left:${menuPosition.x}px;top:${menuPosition.y}px`}
    role="menu"
    aria-label="Node actions"
    tabindex="-1"
    bind:this={menuEl}
    use:portal
    onkeydown={onMenuKey}
    out:fadeOut
  >
    <div class="loom-menu-context" role="presentation">
      <span class="meta">{menuRole}</span>
      <strong>{menuPreview}</strong>
    </div>
    <div class="loom-menu-section" role="presentation">
      <button type="button" role="menuitem" onclick={menuNavigate}>Use this path</button>
      <button type="button" role="menuitem" onclick={menuRegenerate}>Generate another reply…</button>
      <button type="button" role="menuitem" onclick={menuRegenWithMode}>Generate several replies…</button>
      <button type="button" role="menuitem" onclick={menuEdit}>Edit this turn…</button>
      <button type="button" role="menuitem" onclick={menuBranch}>Start a branch…</button>
      {#if (sessionState.info?.scene_mode ?? false) && (menuNode?.role === "user" || menuNode?.role === "assistant")}
        <button
          type="button"
          role="menuitem"
          {...{ "aria-description": "Swap the speaker and create a branch" }}
          onclick={menuSwapSeat}
        >Swap speaker on a new branch</button>
      {/if}
    </div>
    <div class="loom-menu-section" role="presentation">
      <button type="button" role="menuitem" onclick={menuStar}>
        <MorphText text={menuNode?.starred ? "Unstar this branch" : "Star this branch"} />
      </button>
      <button type="button" role="menuitem" onclick={menuNote}>Add a note…</button>
    </div>
    <div class="loom-menu-section" role="presentation">
      {#if menuNode?.recipe !== null}
        <button type="button" role="menuitem" onclick={menuPin}>
          {pinnedComparison.nodeId === menu.nodeId
            ? "Remove comparison pin"
            : "Pin for comparison"}
        </button>
        <button type="button" role="menuitem" onclick={menuToggleSelection}>
          {selectionSet.has(menu.nodeId)
            ? "Remove from comparison"
            : "Add to comparison"}
        </button>
      {/if}
      <button
        type="button"
        role="menuitem"
        onclick={menuCompareBranch}
        disabled={cmpCount < 2}
        {...{ "aria-description": (cmpCount < 2 ? "Create at least two replies first" : "") }}
      >Compare replies…</button>
      <button type="button" role="menuitem" onclick={menuFanOut}>Create guided variations…</button>
    </div>
    <div class="loom-menu-section" role="presentation">
      <button
        type="button"
        role="menuitem"
        onclick={menuDelete}
        disabled={menu.nodeId !== null && loomNodeIntersectsGeneration(menu.nodeId)}
        {...{ "aria-description": (menu.nodeId !== null && loomNodeIntersectsGeneration(menu.nodeId)
          ? LOOM_DELETE_DURING_GENERATION_MESSAGE
          : "") }}
      >Delete this branch…</button>
    </div>
  </div>
{/if}

<LoomDialog
  bind:this={dialog}
  background={asideEl}
  onsearchmatch={showFirstSearchMatch}
  onbranchcreated={(id) => { createdBranchId = id; }}
  onclosed={restoreFocusAfterDialog}
/>

<style>
  /* Clipped top and bottom only: every row inside handles its own width, and a focus ring on a
   * control flush with either side (the search field) must not lose its edge. */
  .loom-sidebar { display: flex; flex-direction: column; gap: var(--s3); height: 100%; min-width: 0; min-height: 0; overflow-x: visible; overflow-y: clip; color: var(--ink2); font: var(--type-small); }
  .loom-top, .loom-actions, .filter-bar, .search-feedback, .loom-status, .loom-map-bar { flex: none; min-width: 0; }
  .loom-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s4); margin-top: calc(var(--s2) * -1); }
  .loom-actions .dw-text { font: var(--type-meta); }
  .weave-host { display: flex; flex: 1; min-height: 0; min-width: 0; container: loom / inline-size; }
  .weave-host[hidden] { display: none; }

  /* ---------- search */
  .filter-bar { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2); }
  .filter-field { position: relative; display: flex; flex: 1 1 240px; min-width: 0; }
  .filter-input { flex: 1 1 auto; min-width: 0; height: 40px; padding: 0 40px 0 var(--s4); border: 0; border-radius: var(--r-pill); background: var(--wash); color: var(--ink); font: var(--type-small); }
  .filter-input::-webkit-search-cancel-button { -webkit-appearance: none; }
  .clear { position: absolute; top: 4px; right: 4px; display: grid; place-items: center; width: 32px; height: 32px; padding: 0; border: 0; border-radius: var(--r-pill); background: transparent; color: var(--ink3); cursor: pointer; transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease); }
  .clear:hover { background: var(--wash2); color: var(--ink); }
  .search-feedback { display: grid; gap: var(--s2); }
  .search-feedback p:empty { display: none; }
  .search-result { display: flex; align-items: center; gap: var(--s1); }
  .search-excerpt { display: grid; flex: 1; gap: 2px; min-width: 0; padding: var(--s2) var(--s3); border: 0; border-radius: var(--r-box); background: var(--wash); color: var(--ink2); font: var(--type-small); text-align: start; overflow-wrap: anywhere; cursor: pointer; transition: background-color var(--fast) var(--ease); }
  .search-excerpt:hover { background: var(--wash2); }
  .search-excerpt mark { --mark: color-mix(in oklch, var(--tint) 22%, transparent); border-radius: 2px; background: var(--mark); box-shadow: 0 0 0 2px var(--mark); color: var(--ink); }
  .search-excerpt > span:last-child { display: -webkit-box; -webkit-line-clamp: 3; line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
  .filter-help { display: grid; gap: var(--s2); max-height: 224px; overflow: auto; padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); color: var(--ink2); font: var(--type-small); }
  .filter-help ul { display: grid; gap: 2px; }
  .filter-help code { color: var(--ink); font-weight: 500; }
  .filter-help .examples { color: var(--ink3); }
  .loom-status { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s4); }
  .focused-branch, .selection-bar { display: inline-flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s3); }
  .selection-bar { color: var(--ink); }
  .selection-bar .dw-text, .loom-map-bar .dw-text { font: var(--type-meta); }

  /* ---------- the map: a drawing on paper. Turns are fills, edges thin ink lines at low alpha,
   * the current path in the depth accent. */
  .loom-map-bar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s2) var(--s4); }
  .loom-map-summary { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s3); min-width: 0; }
  .path-key { display: inline-flex; align-items: center; gap: var(--s2); }
  .path-key i { display: block; width: 20px; height: 2px; border-radius: 1px; background: var(--tint); }
  .loom-view-controls { display: flex; align-items: center; gap: var(--s1) var(--s3); }
  .loom-view-controls output { min-width: 40px; text-align: center; }
  .step { display: grid; place-items: center; width: 32px; height: 32px; padding: 0; border: 0; border-radius: var(--r-pill); background: var(--wash); color: var(--ink2); font: var(--type-control); cursor: pointer; transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .step:hover:not(:disabled) { background: var(--wash2); color: var(--ink); }
  .step:active:not(:disabled) { scale: 0.97; }
  .step:disabled { opacity: 0.4; cursor: default; }

  .tree-scroll { position: relative; flex: 1 1 auto; min-height: 0; overflow: clip; overscroll-behavior: contain; touch-action: none; cursor: grab; border-radius: var(--r-row); background: var(--paper); }
  .tree-scroll.dragging, .tree-scroll.dragging :global(*) { cursor: grabbing !important; }
  .loom-canvas { position: relative; z-index: 1; width: 100%; height: 100%; transform-origin: 0 0; }
  .loom-canvas.camera-animating { transition: transform var(--glide) var(--ease); }
  .loom-edges { position: absolute; inset: 0; overflow: hidden; pointer-events: none; }
  .loom-edges path { fill: none; stroke-linecap: round; stroke-linejoin: round; vector-effect: non-scaling-stroke; transition: stroke var(--fast) var(--ease), opacity var(--fast) var(--ease); }
  .loom-edges .edge-line { stroke: var(--ink3); stroke-width: var(--edge-width, 1.25px); opacity: calc(var(--edge-opacity, 0.55) * 0.6); }
  .loom-edges .edge-line.active-edge { stroke: var(--tint); stroke-width: calc(var(--edge-width, 1.25px) + 0.75px); opacity: 1; }
  .loom-edges .edge-line.filtered-edge { opacity: 0.12; }
  .loom-edges .junction { fill: var(--ink3); opacity: 0.5; }
  .loom-edges .junction.active-junction { fill: var(--tint); opacity: 1; }
  .map-summary { display: flex; flex-direction: column; justify-content: space-between; gap: var(--s2); width: 100%; height: 100%; padding: var(--s3) var(--s4); overflow: hidden; border: 0; border-radius: var(--r-box); background: var(--wash); color: var(--ink2); font: var(--type-small); text-align: start; cursor: zoom-in; }
  .active-path-node .map-summary { background: color-mix(in oklch, var(--tint) 6%, var(--paper)); color: var(--ink); }
  .current-node .map-summary { background: color-mix(in oklch, var(--tint) 11%, var(--paper)); }
  .map-summary-text { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 3; line-clamp: 3; overflow: hidden; }
  .tree-node-wrap { position: absolute; z-index: calc(10 + var(--branch-depth, 0)); min-width: 0; min-height: 0; transition: left var(--in) var(--ease), top var(--in) var(--ease), opacity var(--fast) var(--ease); }
  .tree-node-wrap.active-path-node { z-index: calc(30 + var(--branch-depth, 0)); }
  .tree-node-wrap.current-node { z-index: 60; }
  .tree-node-wrap.dead-node { z-index: calc(5 + var(--branch-depth, 0)); }
  .tree-node-wrap.filtered-out { opacity: 0.45; }
  .tree-node-wrap.filtered-out:hover { opacity: 1; }
  /* Search hits: wash2 for a match, a 22% tint fill for the one in view. */
  .tree-node-wrap.search-match :global(.node) { background: var(--wash2); }
  .tree-node-wrap.search-current :global(.node) { background: color-mix(in oklch, var(--tint) 22%, var(--paper)); }
  /* Chosen for comparison: an ink ring (interaction is ink); a pin, a softer one. */
  .tree-node-wrap.selected :global(.node) { box-shadow: 0 0 0 2px var(--ink); }
  .tree-node-wrap.pinned :global(.node) { box-shadow: 0 0 0 2px var(--ink3); }

  /* ---------- path, options, starred: rows and washes in the reading column */
  .loom-projection { display: flex; flex: 1 1 auto; flex-direction: column; gap: var(--s5); min-height: 0; margin: 0 -4px; padding: 4px 4px var(--s6); overflow: auto; overscroll-behavior: contain; }
  .row-head { display: flex; align-items: center; justify-content: space-between; gap: var(--s3); min-height: 28px; }
  .more { display: grid; place-items: center; flex: none; width: 28px; height: 28px; padding: 0; border: 0; border-radius: var(--r-pill); background: transparent; color: var(--ink3); font: var(--type-meta); cursor: pointer; transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease); }
  .more:hover { background: var(--wash2); color: var(--ink); }
  .path-stream { display: grid; gap: 2px; }
  .path-snippet { display: grid; grid-template-columns: 12px minmax(0, 1fr); gap: var(--s2); align-items: start; padding: var(--s2) var(--s3); border-radius: var(--r-box); transition: background-color var(--fast) var(--ease); }
  .path-snippet:hover { background: var(--wash); }
  .path-snippet.current-snippet { background: var(--wash2); }
  .path-snippet .dw-pip { margin-top: 11px; }
  .snippet-body { display: grid; gap: var(--s1); min-width: 0; }
  .snippet-tokens, .snippet-text { margin: 0; color: var(--ink); font: var(--type-body); line-height: 28px; overflow-wrap: anywhere; }
  .snippet-text { padding: 0; border: 0; background: transparent; text-align: start; white-space: pre-wrap; cursor: pointer; }
  /* A word holds together (a token is a button, atomic in the line) and a longer one wraps inside;
   * the space before a word is text, which a wrapped line hangs at its end. */
  .snippet-tokens { white-space: pre-wrap; }
  .snippet-tokens .word { display: inline-block; max-width: 100%; }
  .path-token { display: inline; padding: 0; border: 0; border-radius: 4px; background: transparent; color: inherit; font: inherit; white-space: pre-wrap; cursor: pointer; transition: background-color var(--fast) var(--ease), box-shadow var(--fast) var(--ease); }
  .path-token:hover { background: var(--paper); box-shadow: 0 0 0 2px var(--paper); }
  .path-snippet:not(:hover):not(.current-snippet) .path-token:hover { background: var(--wash2); box-shadow: 0 0 0 2px var(--wash2); }
  .path-snippet footer { gap: var(--s4); }
  .path-snippet footer .dw-text { font: var(--type-meta); }
  .see-options { align-self: start; }

  .projection-head { display: grid; gap: var(--s3); }
  .anchor { margin: 0; color: var(--ink2); display: -webkit-box; -webkit-line-clamp: 3; line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; overflow-wrap: anywhere; }
  .option-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 280px), 1fr)); gap: var(--s3); }
  .option-card { display: flex; flex-direction: column; gap: var(--s2); min-width: 0; padding: var(--s3) var(--s4) var(--s4); border-radius: var(--r-row); background: var(--wash); }
  .option-card.active-option { background: var(--wash2); }
  .option-card .dw-quiet { background: var(--wash2); }
  .option-card.active-option .dw-quiet, .option-card .dw-quiet:hover { background: var(--paper); }
  .option-card footer { gap: var(--s4); margin-top: auto; padding-top: var(--s2); }
  .option-preview { display: -webkit-box; flex: 1 1 auto; overflow: hidden; -webkit-box-orient: vertical; -webkit-line-clamp: 7; line-clamp: 7; padding: 0; border: 0; background: transparent; color: var(--ink); font: var(--type-small); line-height: 22px; text-align: start; overflow-wrap: anywhere; cursor: pointer; }
  .saved-list { display: grid; gap: 2px; }
  .saved-card { display: grid; grid-template-columns: 12px minmax(0, 1fr) auto; gap: var(--s2) var(--s3); align-items: start; padding: var(--s2) var(--s3); border-radius: var(--r-box); transition: background-color var(--fast) var(--ease); }
  .saved-card:hover { background: var(--wash); }
  .saved-card .dw-pip { margin-top: 7px; }
  .saved-copy { display: grid; gap: 2px; min-width: 0; }
  .saved-copy strong { overflow: hidden; color: var(--ink); font: var(--type-small); font-weight: 500; text-overflow: ellipsis; white-space: nowrap; }
  .saved-copy p { color: var(--ink3); font: var(--type-meta); }
  .saved-actions { gap: var(--s4); }
  .saved-actions .dw-text { font: var(--type-meta); }

  .empty { display: grid; gap: var(--s3); justify-items: start; color: var(--ink3); font: var(--type-small); }

  /* ---------- the node menu: a wash2 popup, no line and no shadow */
  .loom-menu {
    position: fixed;
    z-index: 100;
    display: grid;
    gap: var(--s2);
    width: min(280px, calc(100vw - 24px));
    max-height: min(680px, calc(100dvh - 24px));
    margin: 0;
    padding: var(--s1);
    overflow-y: auto;
    overscroll-behavior: contain;
    border-radius: var(--r-box);
    background: var(--wash2);
    opacity: 0;
    pointer-events: none;
  }
  .loom-menu:focus { outline: none; }
  .loom-menu.positioned { opacity: 1; pointer-events: auto; }
  .loom-menu-context { display: grid; gap: 2px; min-width: 0; padding: var(--s2) var(--s3) var(--s1); }
  .loom-menu-context strong { overflow: hidden; color: var(--ink); font: var(--type-small); font-weight: 500; text-overflow: ellipsis; white-space: nowrap; }
  .loom-menu-section { display: grid; }
  .loom-menu button { display: flex; align-items: center; min-height: 40px; padding: 0 var(--s3); border: 0; border-radius: var(--r-col); background: transparent; color: var(--ink2); font: var(--type-control); text-align: start; cursor: pointer; transition: color var(--fast) var(--ease), background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .loom-menu button:active:not(:disabled) { scale: 0.97; }
  .loom-menu button:disabled { opacity: 0.4; cursor: default; }
  .loom-menu button:focus-visible { outline-offset: -2px; }
  @media (hover: hover) { .loom-menu button:hover:not(:disabled) { background: var(--paper); color: var(--ink); } }

  @media (prefers-reduced-motion: reduce) {
    .tree-node-wrap, .loom-edges path { transition: none; }
  }
  @media (pointer: coarse) {
    /* Every target 44 px on touch: the text actions grow to it, a token's line is 44 px tall. */
    .loom-menu button { min-height: 44px; }
    .step, .more { width: 44px; height: 44px; }
    .filter-input { height: 44px; }
    .clear { top: 6px; }
    .loom-actions { gap: 0 var(--s4); }
    .dw-text { min-height: 44px; }
    .loom-view-controls .dw-text { min-width: 44px; }
    .snippet-tokens { line-height: 44px; }
  }
  @media (max-width: 720px) {
    /* One row of actions that scrolls sideways, as the view tabs do. */
    .loom-actions { flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; overscroll-behavior-inline: contain; mask-image: linear-gradient(to right, var(--ink) calc(100% - 24px), transparent); padding-right: var(--s5); }
    .loom-actions::-webkit-scrollbar { display: none; }
    .loom-actions .dw-text { flex: none; white-space: nowrap; }
    .loom-map-summary { display: none; }
    .loom-map-bar { justify-content: flex-end; }
    .tree-scroll { min-height: min(96px, 25dvh); }
    .filter-input { font-size: 16px; }
    .saved-card { grid-template-columns: 12px minmax(0, 1fr); }
    .saved-actions { grid-column: 2; }
  }
</style>
