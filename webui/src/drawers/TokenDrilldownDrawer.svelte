<script lang="ts">
  // The selected token, in the Inspector (and the phone's sheet): the token and its probability
  // line, the alternatives the sampler had, then the readings at this token in one order: the lens
  // by depth and layer by layer, the probes, the SAE features. Branching from the token is one
  // quiet action at the end. (The Main and Phone canvases, token selected.)
  //
  // Navigation is a cursor that walks the conversation (drawers/token/cursor.ts): ←/→ step tokens
  // and roll across segment and turn boundaries, ↑/↓ jump turns, Home/End jump to the segment's
  // ends, and a fresh selection snaps the cursor back to it. The selection's params are
  // { turnIdx, tokenIdx, isThinking? } into chatLog.turns.
  //
  // Each reading prefers the token's loom-owned measurements envelope (what was captured) and
  // otherwise replays through the family's token-readout route (keyed and shared, one at a time).
  // Where the runtime has the node replay, the lens side is the token's anatomy instead (one
  // request: by depth, layer by layer, the units that pushed it, the top heads' attention); a token
  // it cannot replay (no lens ready, no replay record) keeps the lens readout's captured view.
  // "Unsteered" replays everything without the recipe's steering; the alternatives stay as
  // sampled, since a replay has no sampler.
  import { IconButton, MorphText, Segmented, expand, tokenParts } from "@stdlogits/ledger";
  import { onDestroy, onMount, tick } from "svelte";
  import { showLoom } from "../lib/stores/workbench.svelte";
  import { pushLensToken } from "../lib/steerActions";
  import { tokenInspectorSchema } from "../lib/interfaceSchemas";
  import { registerInterfaceController } from "../lib/workspaceController";
  import { ToolError } from "../lib/webmcp/types";
  import { hideTokenDetails } from "../lib/stores/drawers.svelte";
  import {
    closeDrawer,
    chatLog,
    loomTree,
    loomUiState,
    samplingState,
    sessionState,
    effectiveRawMode,
    probeRack,
    lensSourceState,
    saeSourceState,
    instrumentFamily,
    saeLoaded,
    genStatus,
    sendFork,
    sendTextFork,
    attachProbe,
    refreshLensSources,
  } from "../lib/stores.svelte";
  import {
    LENS_NAMESPACES,
    lensNamespaceInUse,
    lensNamespaceReady,
    lensPairState,
    type LensNamespace,
  } from "../lib/stores/instruments.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type {
    ChatTurn,
    LensTokenReadoutJSON,
    SaeTokenReadoutJSON,
    ScalarReadingJSON,
    TokenAnatomy as TokenAnatomyRecord,
    TokenScore,
  } from "../lib/types";
  import { replayCapabilities } from "../lib/runtime/contracts";
  import { getRuntimeClient } from "../lib/runtime/registry";
  import TokenAnatomy from "./token/TokenAnatomy.svelte";
  import { UnitLabels } from "./token/labels.svelte";
  import { clampCursor, jumpTurn, segmentTokens, segmentsOf, stepCursor, type SegmentKind, type TokenCursor } from "./token/cursor";
  import { ReplayReadout, type GeometryTokenReadout } from "./token/readout.svelte";
  import { drilldownUi, type DrilldownTab } from "./token/drilldown.svelte";
  import Alternatives from "./token/Alternatives.svelte";
  import LensReading from "./token/LensReading.svelte";
  import ProbeReadings from "./token/ProbeReadings.svelte";
  import FeatureReadings from "./token/FeatureReadings.svelte";
  import { formatP, lastLayer } from "./token/layers";
  import { geometryReplayView, lensReplay, replayAnchor, tokenPins } from "./token/readings";
  import { resolveReadoutTopK } from "../lib/readouts";
  import { geometryTracesKept, rowGeometry, wholeEnvelope } from "../lib/tokenRows";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import { currentCatalogInstrumentAvailability, type CatalogInstrumentAvailability } from "../lib/runtime/instrumentAvailability";

  interface DrawerParams {
    turnIdx: number;
    tokenIdx: number;
    /** The click came from the thinking body, so the anchor segment is ``turn.thinkingTokens``. */
    isThinking?: boolean;
    /** The section to bring into view (the loom's entry points ask for the alternatives). */
    initialTab?: DrilldownTab;
  }

  let { params: inputParams, docked = false, active = true, mobile = false, onclose }: { params?: unknown; docked?: boolean; active?: boolean; mobile?: boolean; onclose?: () => void } = $props();
  let inspectorEl: HTMLElement | null = $state(null);
  const params = $derived(inputParams as DrawerParams | null);

  function closeDetails(): void {
    if (onclose) onclose();
    else if (docked) hideTokenDetails();
    else closeDrawer();
  }

  // ---- the cursor ---------------------------------------------------------

  /** The token the user selected: a fresh selection snaps the cursor back here. */
  const anchor = $derived.by<TokenCursor | null>(() => {
    if (!params) return null;
    return { turnIdx: params.turnIdx, seg: params.isThinking ? "thinking" : "response", tokenIdx: params.tokenIdx };
  });

  let cursor = $state<TokenCursor | null>(null);

  /** "primary" is the turn as generated; "shadow" its A/B twin, when it has one. Per turn: walking
   *  into another turn resets it. */
  type Branch = "primary" | "shadow";
  let branch: Branch = $state<Branch>("primary");

  $effect(() => {
    const a = anchor;
    cursor = a ? { ...a } : null;
    branch = "primary";
  });

  const turnsView = $derived<ChatTurn[]>(
    chatLog.turns.map((t, i) => (i === (cursor?.turnIdx ?? -1) && branch === "shadow" && t.abPair ? t.abPair : t)),
  );
  const segments = $derived(segmentsOf(turnsView));
  /** The cursor clamped onto the current segments (token lists change while streaming). */
  const effCursor = $derived(cursor ? clampCursor(segments, cursor) : null);

  let lastBranchTurn = -1;
  $effect(() => {
    const t = effCursor?.turnIdx ?? -1;
    if (t !== lastBranchTurn) {
      lastBranchTurn = t;
      if (branch !== "primary") branch = "primary";
    }
  });

  const turn = $derived<ChatTurn | null>(
    effCursor != null && effCursor.turnIdx >= 0 && effCursor.turnIdx < chatLog.turns.length ? chatLog.turns[effCursor.turnIdx] : null,
  );
  /** The turn inspected (the twin, when it is chosen). */
  const inspected = $derived<ChatTurn | null>(effCursor ? (turnsView[effCursor.turnIdx] ?? null) : null);
  const tokenList = $derived<TokenScore[]>(effCursor ? segmentTokens(inspected, effCursor.seg) : []);
  const token = $derived<TokenScore | null>(
    effCursor != null && effCursor.tokenIdx >= 0 && effCursor.tokenIdx < tokenList.length ? tokenList[effCursor.tokenIdx] : null,
  );
  const identity = $derived(effCursor ? `${effCursor.turnIdx}:${effCursor.seg}:${effCursor.tokenIdx}` : "none");

  const loomNodeId = $derived.by(() => {
    if (!effCursor) return null;
    if (branch === "shadow") return inspected?.nodeId ?? null;
    if (turn?.nodeId) return turn.nodeId;
    if (effCursor.turnIdx < 0 || loomTree.activePath.length === 0) return null;
    const visible = loomTree.activePath
      .map((id) => loomTree.nodes.get(id))
      .filter(Boolean)
      .filter((n) => !(n!.parent_id === null && n!.role === "system" && !n!.text));
    return visible[effCursor.turnIdx]?.id ?? null;
  });

  /** Where the token replays and under which steering (``replayAnchor``): a generated token at
   *  its raw decode index, a captured prompt token at its place in its node's capture. Only a
   *  generated token branches (``rawIndex``). */
  const replayTarget = $derived(replayAnchor(
    loomNodeId ? loomTree.nodes.get(loomNodeId) : null, token, inspected?.appliedSteering,
  ));
  const replayAt = $derived(replayTarget.index);
  const hasReplayContext = $derived(loomNodeId != null && replayAt != null);
  const canBranch = $derived(branch === "primary" && loomNodeId != null && token?.rawIndex != null && !genStatus.active);

  // ---- branching ----------------------------------------------------------

  let branchOpen = $state(false);
  let branchText = $state("");
  let branching = $state(false);
  let branchError = $state<string | null>(null);
  let branchInput = $state<HTMLInputElement | null>(null);

  $effect(() => {
    void loomNodeId;
    void token?.rawIndex;
    branchError = null;
    branchOpen = false;
    branchText = "";
  });

  async function openBranch(): Promise<void> {
    if (!canBranch || !token) return;
    branchText = token.text;
    branchOpen = true;
    branchError = null;
    await tick();
    branchInput?.focus();
    const first = branchText.search(/\S/);
    const last = branchText.trimEnd().length;
    branchInput?.setSelectionRange(first >= 0 ? first : 0, last > 0 ? last : branchText.length);
  }

  /** After a branch starts, the loom shows it (on the phone the sheet makes way). */
  function toLoom(): void {
    if (!docked || mobile) closeDetails();
    loomUiState.view = "map";
    showLoom();
  }

  async function run(start: () => Promise<void>, fallback: string): Promise<void> {
    branchError = null;
    branching = true;
    try {
      await start();
      toLoom();
    } catch (error) {
      branchError = userFacingError(error, fallback);
    } finally {
      branching = false;
    }
  }

  /** The token kept as it is continues from here with a fresh seed; edited text replaces it. */
  async function startBranch(): Promise<void> {
    if (!canBranch || !loomNodeId || token?.rawIndex == null) return;
    if (branchText.length === 0) {
      branchError = "Write the text to branch with";
      branchInput?.focus();
      return;
    }
    const nodeId = loomNodeId;
    const rawIndex = token.rawIndex;
    if (branchText === token.text && token.tokenId != null) {
      const tokenId = token.tokenId;
      await run(() => sendFork(nodeId, rawIndex, tokenId, true), "Unable to continue this branch from the selected token.");
    } else {
      const text = branchText;
      await run(() => sendTextFork(nodeId, rawIndex, text), "Unable to replace this token and start a new branch.");
    }
  }

  /** An alternative swaps in for the token and the rest resamples as a sibling. */
  async function branchWith(altId: number): Promise<void> {
    if (!canBranch || !loomNodeId || token?.rawIndex == null) return;
    const nodeId = loomNodeId;
    const rawIndex = token.rawIndex;
    await run(() => sendFork(nodeId, rawIndex, altId), "Unable to create a branch from this word choice. Try another alternative.");
  }

  // ---- the generation recipe ---------------------------------------------

  const loomNode = $derived(loomNodeId ? (loomTree.nodes.get(loomNodeId) ?? null) : null);
  const recipeSampling = $derived(loomNode?.recipe?.sampling ?? null);
  /** The steering the token was generated under (a prompt token: the steering its capture read it
   *  under); an empty expression is none. */
  const recipeSteering = $derived(replayTarget.steering);
  function fmtSetting(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return "–";
    return String(Number(value.toFixed(2)));
  }
  /** The sampling this token was drawn with, as the composer's sampling line writes it. */
  const recipeLine = $derived.by<string>(() => {
    const sampling = recipeSampling;
    const recipe = loomNode?.recipe;
    if (!sampling && !recipe) return "";
    const parts = [
      `temperature ${fmtSetting(sampling?.temperature)}`,
      `top-p ${fmtSetting(sampling?.top_p)}`,
      ...(sampling?.top_k ? [`top-k ${sampling.top_k}`] : []),
      `${fmtSetting(sampling?.max_tokens)} tokens`,
    ];
    const seed = recipe?.seed ?? sampling?.seed;
    if (seed != null) parts.push(`seed ${seed}`);
    if (sampling?.presence_penalty) parts.push(`presence ${fmtSetting(sampling.presence_penalty)}`);
    if (sampling?.frequency_penalty) parts.push(`frequency ${fmtSetting(sampling.frequency_penalty)}`);
    if (recipe?.thinking != null) parts.push(recipe.thinking ? "thinking" : "no thinking");
    if ((recipe?.probes.length ?? 0) > 0) parts.push(`${recipe!.probes.length} probes`);
    return parts.join(" · ");
  });

  // ---- navigation ---------------------------------------------------------

  function moveTo(next: TokenCursor | null): void {
    if (next) cursor = next;
  }
  const canStepBack = $derived(effCursor != null && stepCursor(segments, effCursor, -1) !== null);
  const canStepFwd = $derived(effCursor != null && stepCursor(segments, effCursor, 1) !== null);
  function step(delta: 1 | -1): void {
    if (effCursor) moveTo(stepCursor(segments, effCursor, delta));
  }
  function turnHop(delta: 1 | -1): void {
    if (effCursor) moveTo(jumpTurn(segments, effCursor, delta));
  }
  function segEdge(where: "home" | "end"): void {
    if (!effCursor) return;
    cursor = { ...effCursor, tokenIdx: where === "home" ? 0 : Math.max(0, tokenList.length - 1) };
  }
  const atAnchor = $derived(
    effCursor != null && anchor != null && effCursor.turnIdx === anchor.turnIdx && effCursor.seg === anchor.seg && effCursor.tokenIdx === anchor.tokenIdx,
  );
  function resetToAnchor(): void {
    if (anchor) cursor = { ...anchor };
  }
  /** The turn's other segment (thinking or response), when it has one. */
  const otherSeg = $derived.by<SegmentKind | null>(() => {
    if (!effCursor) return null;
    const other: SegmentKind = effCursor.seg === "thinking" ? "response" : "thinking";
    return segments.some((s) => s.turnIdx === effCursor.turnIdx && s.seg === other) ? other : null;
  });
  function toggleSeg(): void {
    if (!effCursor || !otherSeg) return;
    cursor = { turnIdx: effCursor.turnIdx, seg: otherSeg, tokenIdx: 0 };
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.defaultPrevented || !active) return;
    if ((mobile || docked) && !inspectorEl?.contains(ev.target as Node)) return;
    if (ev.key === "Escape") {
      ev.preventDefault();
      if (!docked || mobile) closeDetails();
      return;
    }
    const t = ev.target as HTMLElement | null;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable || t.closest('[role="slider"], [role="listbox"]'))) return;
    switch (ev.key) {
      case "ArrowLeft": step(-1); break;
      case "ArrowRight": step(1); break;
      case "ArrowUp": turnHop(-1); break;
      case "ArrowDown": turnHop(1); break;
      case "Home": segEdge("home"); break;
      case "End": segEdge("end"); break;
      default: return;
    }
    ev.preventDefault();
  }

  // ---- the probability line ----------------------------------------------

  const named = $derived(tokenParts(token?.text ?? ""));
  const chosenP = $derived(token?.logprob != null ? Math.exp(token.logprob) : null);
  /** The chosen token's rank among its captured alternatives. */
  const chosenRank = $derived.by<number | null>(() => {
    const alts = token?.topAlts;
    if (!alts || alts.length === 0 || token?.tokenId == null) return null;
    const i = alts.findIndex((a) => a.id === token.tokenId);
    return i >= 0 ? i + 1 : null;
  });
  /** A few tokens either side of the cursor, the one inspected marked; a click moves there. */
  const context = $derived.by(() => {
    if (!effCursor) return [];
    const at = effCursor.tokenIdx;
    const from = Math.max(0, at - 12);
    return tokenList.slice(from, Math.min(tokenList.length, at + 7)).map((tok, k) => ({ text: tok.text.replace(/\n/g, "⏎"), i: from + k }));
  });
  const hasAbPair = $derived(turn?.abPair != null);

  // ---- the readings (captured, or replayed) -------------------------------

  const jlensFitted = $derived(sessionState.info?.jlens_fitted === true);
  const saeResident = $derived(saeLoaded());
  let saeAvailability = $state<CatalogInstrumentAvailability>("unknown");
  onMount(() => {
    let mounted = true;
    void currentCatalogInstrumentAvailability("sae").then((availability) => {
      if (mounted) saeAvailability = availability;
    });
    return () => {
      mounted = false;
    };
  });
  const lensTokenReplayAvailable = $derived(instrumentFamily("lens")?.capabilities.token_readout === true);
  const saeTokenReplayAvailable = $derived(instrumentFamily("sae")?.capabilities.token_readout === true);
  const geometryTokenReplayAvailable = $derived(instrumentFamily("geometry")?.capabilities.token_readout === true);

  // The lens and SAE read as wide as the logit alternatives (eight when those are off).
  const readoutTopK = $derived(resolveReadoutTopK(samplingState.return_top_k));

  const lensReadout = new ReplayReadout<LensTokenReadoutJSON>();
  const saeReadout = new ReplayReadout<SaeTokenReadoutJSON>();
  const geometryReadout = new ReplayReadout<GeometryTokenReadout>();
  /** The token layer by layer (the node replay): one request in place of the lens readout, where
   *  the runtime has it (drowse.ai does not, and keeps the lens readout). */
  const anatomyAvailable = replayCapabilities(getRuntimeClient().instruments).tokenAnatomy;
  const anatomyReadout = new ReplayReadout<TokenAnatomyRecord>();
  const unitLabels = typeof getRuntimeClient().instruments.unitLabels === "function" ? new UnitLabels() : null;
  onDestroy(() => {
    lensReadout.dispose();
    saeReadout.dispose();
    geometryReadout.dispose();
    anatomyReadout.dispose();
  });
  let lensSteered = $state(true);
  let saeSteered = $state(true);
  let geometrySteered = $state(true);

  /** Steered or unsteered, for all three at once (an agent may still set one family alone). */
  const steeredAll = $derived(lensSteered && saeSteered && geometrySteered);
  /** The lens side replays as the anatomy where the runtime has it, else as the lens readout; it is
   *  ready with an R lens, or (for the anatomy) any active lens. The anatomy shows where a lens is
   *  ready and the token has its replay record; otherwise the lens readout's view stands in (the
   *  capture, or why there is none). The WebMCP inspector gates on the same answer. */
  const lensGate = $derived(lensReplay({
    anatomy: anatomyAvailable,
    tokenReadout: lensTokenReplayAvailable,
    rLens: jlensFitted,
    activeLens: lensSourceState.sources.some((source) => source.active),
    replayContext: hasReplayContext,
  }));
  const lensReplayAvailable = $derived(lensGate.available);
  const anatomyShown = $derived(lensGate.anatomy);
  const anyReplay = $derived(lensReplayAvailable || saeTokenReplayAvailable || geometryTokenReplayAvailable);
  const showSteering = $derived(anyReplay && hasReplayContext && (recipeSteering !== null || !steeredAll));
  function setSteered(next: "steered" | "unsteered"): void {
    const on = next === "steered";
    lensSteered = on || !lensReplayAvailable;
    saeSteered = on || !saeTokenReplayAvailable;
    geometrySteered = on || !geometryTokenReplayAvailable;
  }

  // A reply whose steering a replay cannot reproduce (a thinking-phase window on a thinking model)
  // is read unsteered, the one way it replays: the first refusal switches every family over, and
  // the view says so once instead of each section printing the server's reason.
  let unreplayableNode: string | null = $state(null);
  const steeringUnreplayable = $derived(unreplayableNode !== null && unreplayableNode === loomNodeId);
  $effect(() => {
    const refused = geometryReadout.refused || anatomyReadout.refused || lensReadout.refused || saeReadout.refused;
    if (!refused || !loomNodeId || unreplayableNode === loomNodeId) return;
    unreplayableNode = loomNodeId;
    setSteered("unsteered");
  });

  /** At least one geometry probe on the roster: the geometry replay refuses an empty one. */
  const hasGeometryProbes = $derived(probeRack.active.some((name) => probeRack.entries.get(name)?.info.family === "geometry"));

  /** The token's whole envelope (a live token's, or a row kept whole), or null for a generated
   *  row, which keeps only its scores and geometry shares and coordinates: the rest replays. */
  const whole = $derived(wholeEnvelope(token?.measurements));

  const capturedLensData = $derived.by<LensTokenReadoutJSON | null>(() => {
    const lens = whole?.instruments.lens;
    if (!lens?.readout || !token) return null;
    return {
      node_id: loomNodeId ?? "",
      raw_index: replayAt ?? -1,
      token_id: token.tokenId ?? -1,
      token_text: token.text,
      steering: lens.binding.steering,
      aggregate: lens.readout.aggregate,
      layers: lens.readout.layers ?? [],
    };
  });
  /** The captured lens readout kept its layer rows (a live token, a row kept whole). */
  const capturedLensLayers = $derived(whole?.instruments.lens?.readout?.layers !== undefined);
  const capturedSaeData = $derived.by<SaeTokenReadoutJSON | null>(() => {
    const sae = whole?.instruments.sae;
    if (!sae?.readout || !token) return null;
    return {
      node_id: loomNodeId ?? "",
      raw_index: replayAt ?? -1,
      token_id: token.tokenId ?? -1,
      token_text: token.text,
      steering: sae.binding.steering,
      layer: sae.binding.layer ?? -1,
      features: sae.readout.features,
    };
  });
  const capturedGeometryData = $derived.by<GeometryTokenReadout | null>(() => {
    const geometry = whole?.instruments.geometry;
    if (!geometry || Object.keys(geometry.readings ?? {}).length === 0) return null;
    return { steering: geometry.binding?.steering ?? null, readings: geometry.readings };
  });
  /** The captured geometry readings kept their layers (a live token, a row kept whole). */
  const capturedGeometryLayers = $derived(geometryTracesKept(capturedGeometryData?.readings));

  /** The lens and SAE probes, as the shown reads pinned them (captured with a live token, or
   *  replayed with the read: a generated row keeps only their values). */
  const pinned = $derived<Record<string, ScalarReadingJSON>>(tokenPins({
    anatomy: anatomyShown,
    steered: lensSteered,
    captured: whole?.instruments.lens?.readings,
    anatomyPins: anatomyReadout.pins,
    lensPins: lensReadout.pins,
    saePins: saeReadout.pins,
    saeSteered,
    scores: token?.measurements?.scores ?? null,
  }));

  // One effect per family: adopt the capture while steered, else replay. Declared geometry, SAE,
  // lens: the replay queue runs the newest interactive request first, so the lens reads first.
  $effect(() => {
    if (!active) return;
    void token;
    void genStatus.active;
    void loomNodeId;
    const captured = capturedGeometryData;
    if (!geometryTokenReplayAvailable && !geometrySteered) geometrySteered = true;
    const nodeId = loomNodeId;
    const rawIndex = replayAt;
    const canReplay = geometryTokenReplayAvailable && hasGeometryProbes && !!nodeId && rawIndex != null;
    if ((geometrySteered || !geometryTokenReplayAvailable) && captured && (capturedGeometryLayers || !canReplay)) {
      geometryReadout.adopt(captured, null);
      return;
    }
    geometryReadout.clear();
    if (!canReplay || !nodeId || rawIndex == null) return;
    const steered = geometrySteered;
    // Steered, a saved summary shows while the replay brings its layers, and stays if it fails.
    const base = steered && captured ? { data: captured, source: null } : null;
    // The map is bound to this token's row now: the replay may land after the selection moved.
    const map = geometryReplayView({ steered, captured, row: rowGeometry(token?.measurements) });
    geometryReadout.replay("geometry", nodeId, rawIndex, { steered, raw: effectiveRawMode() }, map, base);
  });

  $effect(() => {
    if (!active) return;
    void token;
    void genStatus.active;
    void loomNodeId;
    const captured = capturedSaeData;
    if (!saeTokenReplayAvailable && !saeSteered) saeSteered = true;
    if ((saeSteered || !saeTokenReplayAvailable) && captured) {
      saeReadout.adopt(captured, whole?.instruments.sae?.binding.source ?? null, whole?.instruments.sae?.readings ?? null);
      return;
    }
    saeReadout.clear();
    if (!saeTokenReplayAvailable) return;
    if (!saeResident) return;
    const nodeId = loomNodeId;
    const rawIndex = replayAt;
    if (!nodeId || rawIndex == null) return;
    const replayTokenId = token?.tokenId ?? -1;
    const replayTokenText = token?.text ?? "";
    const fallbackSource = saeSourceState.sources.find((source) => source.active)?.source ?? instrumentFamily("sae")?.source ?? null;
    saeReadout.replay("sae", nodeId, rawIndex, { topK: readoutTopK, steered: saeSteered, raw: effectiveRawMode() }, (m) => {
      const sae = m.instruments.sae;
      if (!sae?.readout) throw new Error("No SAE reading came back. Check the active feature source.");
      return {
        data: { node_id: nodeId, raw_index: rawIndex, token_id: replayTokenId, token_text: replayTokenText, steering: sae?.binding.steering ?? null, layer: sae?.binding.layer ?? -1, features: sae.readout.features },
        source: sae?.binding.source ?? fallbackSource ?? null,
        pins: sae.readings ?? null,
      };
    });
  });

  $effect(() => {
    if (!active || !anatomyAvailable) return;
    void token;
    // A reply ending changes the conversation the replay reads (and drops a replay that was
    // refused or cut short while it streamed): read the selected token again then.
    void genStatus.active;
    const nodeId = loomNodeId;
    const rawIndex = replayAt;
    if (!anatomyShown || !nodeId || rawIndex == null) { anatomyReadout.clear(); return; }
    anatomyReadout.anatomy(nodeId, rawIndex, { k: Math.min(256, Math.max(1, readoutTopK)), units: 8, patterns: 4, steered: lensSteered, raw: effectiveRawMode() }, (m) => {
      const anatomy = m.anatomy;
      if (!anatomy) throw new Error("No anatomy came back for this token. Select it again to retry.");
      return { data: anatomy, source: anatomy.binding.sources.R ?? null, pins: m.instruments.lens?.readings ?? null };
    });
  });

  // The lens readout: where the anatomy shows, nothing; where it cannot (no lens ready, or no replay
  // record), the capture still draws, and only a runtime without the anatomy replays the readout.
  $effect(() => {
    if (!active) return;
    if (anatomyShown) { lensReadout.clear(); return; }
    void token;
    void genStatus.active;
    void loomNodeId;
    const captured = capturedLensData;
    if (!lensReplayAvailable && !lensSteered) lensSteered = true;
    const nodeId = loomNodeId;
    const rawIndex = replayAt;
    const canReplay = !anatomyAvailable && lensTokenReplayAvailable && jlensFitted && !!nodeId && rawIndex != null;
    if ((lensSteered || !lensReplayAvailable) && captured && (capturedLensLayers || !canReplay)) {
      lensReadout.adopt(captured, whole?.instruments.lens?.binding.sources.R ?? null, whole?.instruments.lens?.readings ?? null);
      return;
    }
    lensReadout.clear();
    if (!canReplay || !nodeId || rawIndex == null) return;
    const replayTokenId = token?.tokenId ?? -1;
    const replayTokenText = token?.text ?? "";
    // The readout is the R family's: its source, never an active S lens.
    const fallbackSource = lensSourceState.sources.find((source) => source.active && source.family !== "S")?.source ?? null;
    // Steered, the saved chips show while the replay brings the layer rows, and stay if it fails.
    const base = lensSteered && captured
      ? { data: captured, source: whole?.instruments.lens?.binding.sources.R ?? null, pins: whole?.instruments.lens?.readings ?? null }
      : null;
    lensReadout.replay("lens", nodeId, rawIndex, { topK: readoutTopK, steered: lensSteered, raw: effectiveRawMode(), layers: "all" }, (m) => {
      const lens = m.instruments.lens;
      if (!lens?.readout) throw new Error("No lens reading came back. Check the active lens source.");
      return {
        data: { node_id: nodeId, raw_index: rawIndex, token_id: replayTokenId, token_text: replayTokenText, steering: lens?.binding.steering ?? null, aggregate: lens.readout.aggregate, layers: lens.readout.layers ?? [] },
        source: lens?.binding.sources.R ?? fallbackSource ?? null,
        pins: lens.readings ?? null,
      };
    }, base);
  });

  /** The model's deepest layer, as the lens readout names it (the SAE layer's place on the ramp). */
  const deepest = $derived(
    anatomyReadout.data ? anatomyReadout.data.layers.length - 1 : lensReadout.data ? lastLayer(lensReadout.data.layers) : null,
  );

  // ---- sections, for the WebMCP inspector's "tab" ------------------------

  const sections: Partial<Record<DrilldownTab, HTMLElement>> = {};
  function sectionRef(node: HTMLElement, tab: DrilldownTab) {
    sections[tab] = node;
    return { destroy: () => { if (sections[tab] === node) delete sections[tab]; } };
  }
  async function bringIntoView(tab: DrilldownTab): Promise<void> {
    drilldownUi.tab = tab;
    await tick();
    sections[tab]?.scrollIntoView({ block: "nearest" });
  }
  let introduced: DrawerParams | null = null;
  $effect(() => {
    if (!active || !params || introduced === params || !token) return;
    introduced = params;
    if (params.initialTab) void bringIntoView(params.initialTab);
  });

  $effect(() => {
    if (!active) return;
    return registerInterfaceController("token_inspector", {
      schema: tokenInspectorSchema,
      read: () => ({ values: { cursor: effCursor, branch, tab: drilldownUi.tab, apply_recipe: { geometry: geometrySteered, lens: lensSteered, sae: saeSteered }, presentation: docked ? "dock" : "drawer" }, node_id: loomNodeId, raw_index: token?.rawIndex ?? null, shadow_available: hasAbPair, replay: { geometry: geometryTokenReplayAvailable, lens: lensReplayAvailable, sae: saeTokenReplayAvailable } }),
      update: async (change) => {
        if (change.move && (change.cursor || change.branch)) throw new ToolError("INVALID_INPUT", "Choose a relative move or an explicit cursor and branch.");
        const target = (change.cursor as TokenCursor | undefined) ?? effCursor;
        const nextBranch = (change.branch as Branch | undefined) ?? (change.cursor && target?.turnIdx !== effCursor?.turnIdx ? "primary" : branch);
        if (!target) throw new ToolError("NOT_FOUND", "Select a token before changing the inspector.");
        const primary = chatLog.turns[target.turnIdx];
        if (nextBranch === "shadow" && !primary?.abPair) throw new ToolError("NOT_FOUND", "This turn has no shadow comparison.");
        const chosen = nextBranch === "shadow" ? primary?.abPair : primary;
        const targetToken = segmentTokens(chosen ?? null, target.seg)[target.tokenIdx];
        if (!targetToken) throw new ToolError("NOT_FOUND", "The requested token is not present on that branch and segment.");
        const recipe = change.apply_recipe as Record<string, boolean> | undefined;
        for (const [family, supported] of [["geometry", geometryTokenReplayAvailable], ["lens", lensReplayAvailable], ["sae", saeTokenReplayAvailable]] as const) {
          if (recipe?.[family] === false && (!supported || targetToken.rawIndex == null)) throw new ToolError("UNAVAILABLE", `Unsteered ${family} inspection requires token replay support and a recorded raw token index.`);
          if (recipe?.[family] === false && ((family === "geometry" && !hasGeometryProbes) || (family === "lens" && !lensGate.ready) || (family === "sae" && !saeResident))) throw new ToolError("UNAVAILABLE", `Prepare the ${family} instrument before requesting an unsteered replay.`);
        }
        let moved: TokenCursor | null = null;
        if (change.move) {
          if (change.move === "next" || change.move === "previous") moved = stepCursor(segments, target, change.move === "next" ? 1 : -1);
          else if (change.move === "next_turn" || change.move === "previous_turn") moved = jumpTurn(segments, target, change.move === "next_turn" ? 1 : -1);
          else if (change.move === "segment_start" || change.move === "segment_end") moved = { ...target, tokenIdx: change.move === "segment_start" ? 0 : tokenList.length - 1 };
          else if (change.move === "anchor") moved = anchor;
          else if (otherSeg) moved = { ...target, seg: otherSeg, tokenIdx: 0 };
          if (!moved) throw new ToolError("NOT_FOUND", "There is no token in that direction.");
        }
        if (change.cursor || moved) { cursor = moved ?? { ...target }; await tick(); }
        if (change.branch !== undefined) branch = nextBranch;
        if (recipe?.geometry !== undefined) geometrySteered = recipe.geometry;
        if (recipe?.lens !== undefined) lensSteered = recipe.lens;
        if (recipe?.sae !== undefined) saeSteered = recipe.sae;
        await tick();
        if (change.tab !== undefined) await bringIntoView(change.tab as DrilldownTab);
        if (change.presentation === "hide") closeDetails();
      },
    });
  });

  // ---- the canvas's token actions: a lens probe on it, a lens term toward it ----------------
  // As said (``jlens/``) or as read (``slens/``), through the pair the Lens panel shares; a model
  // with one lens family offers it alone. Addressed by id (``jlens/#<id>``): the text may not name
  // the token (a leading space, a byte piece), and the id always does.
  const saidReady = $derived(lensNamespaceReady("jlens"));
  const readReady = $derived(lensNamespaceReady("slens"));
  const lensAs = $derived(lensNamespaceInUse());
  const lensPair = $derived(LENS_NAMESPACES.find((row) => row.value === lensAs) ?? LENS_NAMESPACES[0]);
  onMount(() => {
    if (lensSourceState.sources.length === 0) void refreshLensSources();
  });
  let probing = $state(false);
  const tokenProbed = $derived(token?.tokenId != null && probeRack.active.some((name) => {
    const info = probeRack.entries.get(name)?.info;
    return info?.family === "lens" && info.namespace === lensAs && info.token_id === token?.tokenId;
  }));
  async function probeToken(): Promise<void> {
    const id = token?.tokenId;
    if (id == null || probing) return;
    probing = true;
    const selector = `${lensAs}/#${id}`;
    try {
      await attachProbe(selector);
      pushToast(`Probing ${named.empty ? "∅" : named.text.trim() || JSON.stringify(named.text)}`, { kind: "info" });
    } catch (e) {
      pushToast(`Couldn't probe ${selector}: ${userFacingError(e)}`, { kind: "error" });
    } finally {
      probing = false;
    }
  }
  function steerTowardToken(): void {
    const id = token?.tokenId;
    if (id == null) return;
    // The atom names the exact id; its chip reads the token's text and opens in the strip.
    pushLensToken({ id, text: token?.text ?? "" }, lensAs);
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="token" class:mobile bind:this={inspectorEl} data-morph-snapshot={identity} aria-label="Token" role="region" tabindex="-1">
  {#if token && effCursor}
    <section class="dw-sec top" use:sectionRef={"logits"} aria-label="This token">
      <div class="title-row">
        <h2 class="dw-title" data-tip={`id ${token.tokenId ?? "–"}${token.rawIndex != null ? ` · position ${token.rawIndex}` : ""}`}>
          <span class="dw-ws">{named.lead ? "·" : ""}</span><MorphText text={named.empty ? "∅" : named.text} numbers={false} {identity} />
        </h2>
        <span class="steps">
          <IconButton icon="left" label="Previous token" disabled={!canStepBack} onclick={() => step(-1)} />
          <IconButton icon="right" label="Next token" disabled={!canStepFwd} onclick={() => step(1)} />
        </span>
      </div>
      <div class="dw-facts">
        <span>token <b><MorphText text={String(effCursor.tokenIdx + 1)} /></b> of {tokenList.length}</span>
        {#if chosenP !== null}<span data-tip={`sampled p · log p ${token.logprob?.toFixed(3)} · after the sampler's temperature and top-k/top-p, not the model's own probability`}>p <b><MorphText text={formatP(chosenP)} {identity} /></b></span>{/if}
        {#if chosenRank !== null}<span>rank <b>{chosenRank}</b></span>{/if}
        {#if otherSeg}
          <button type="button" class="fact" onclick={toggleSeg} aria-label={`Show the ${otherSeg}`}>{effCursor.seg}</button>
        {:else if effCursor.seg === "thinking"}<span>thinking</span>{/if}
        {#if hasAbPair}<button type="button" class="fact" aria-pressed={branch === "shadow"} data-tip="The same turn, generated without steering" onclick={() => (branch = branch === "shadow" ? "primary" : "shadow")}>A/B twin</button>{/if}
        {#if replayAt == null}<span data-tip="The generation record is missing, so this token cannot be replayed or branched">not replayable</span>{/if}
        {#if !atAnchor}<button type="button" class="fact" onclick={resetToAnchor}>Back</button>{/if}
      </div>
      {#if recipeSteering || recipeLine}
        <p class="recipe meta">{#if recipeSteering}<span class="expr">{recipeSteering}</span>{/if}{#if recipeSteering && recipeLine}<span aria-hidden="true">{" · "}</span>{/if}{recipeLine}</p>
      {/if}
      {#if context.length > 1}
        <!-- Mouse targets only (the arrow keys walk the same tokens), so focus stays on the view. -->
        <p class="ctx" aria-hidden="true">{#each context as item (item.i)}{#if item.i === effCursor.tokenIdx && item.text.startsWith(" ")}{" "}{/if}<button type="button" class="ctok" class:here={item.i === effCursor.tokenIdx} tabindex="-1" onclick={() => { if (effCursor) cursor = { ...effCursor, tokenIdx: item.i }; }}>{item.i === effCursor.tokenIdx && item.text.startsWith(" ") ? item.text.slice(1) : item.text}</button>{/each}</p>
      {/if}
      <Alternatives {token} canBranch={canBranch} busy={branching} onbranch={(id) => void branchWith(id)} />
    </section>

    {#if steeringUnreplayable}
      <p class="meta readings-note">A replay can't reproduce where this reply's steering fired, so these readings are replayed without it.</p>
    {:else if showSteering}
      <div class="dw-row readings-switch">
        <Segmented label="Readings" value={steeredAll ? "steered" : "unsteered"} onchange={setSteered}
          options={[{ value: "steered", label: "Steered", hint: recipeSteering ?? undefined }, { value: "unsteered", label: "Unsteered", hint: "Replayed without the steering" }]} />
      </div>
    {/if}

    <div class="part" use:sectionRef={"lens"}>
      {#if anatomyShown}
        <TokenAnatomy readout={anatomyReadout} tokenText={token.text} labels={unitLabels} />
      {:else}
        <!-- The capture, or why there is none: no lens ready, or no replay record. -->
        <LensReading readout={lensReadout} tokenText={token.text} jlensFitted={lensGate.ready} {hasReplayContext}
          modelId={sessionState.info?.model_id ?? null} replayAvailable={lensReplayAvailable} />
      {/if}
    </div>
    <div class="part" use:sectionRef={"geometry"}>
      <ProbeReadings readout={geometryReadout} {pinned} {hasGeometryProbes} {hasReplayContext}
        replayAvailable={geometryTokenReplayAvailable}
        returnToToken={{ turnIdx: effCursor.turnIdx, tokenIdx: effCursor.tokenIdx, isThinking: effCursor.seg === "thinking" }} />
    </div>
    <div class="part" use:sectionRef={"sae"}>
      <FeatureReadings readout={saeReadout} saeLoaded={saeResident} availability={saeAvailability} {hasReplayContext}
        replayAvailable={saeTokenReplayAvailable} lastLayer={deepest} />
    </div>

    <section class="dw-sec" aria-label="Act on this token">
      {#if branchOpen}
        <form class="branch" transition:expand onsubmit={(event) => { event.preventDefault(); void startBranch(); }}>
          <label class="sr-only" for="token-branch-text">Branch with</label>
          <!-- Escape here closes the field, before the window's Escape would close the token. -->
          <input id="token-branch-text" bind:this={branchInput} bind:value={branchText} autocomplete="off" spellcheck="true"
            onkeydown={(event) => { if (event.key === "Escape") { event.preventDefault(); branchOpen = false; branchError = null; } }} />
          <button type="submit" class="dw-button" disabled={!canBranch || branching || branchText.length === 0}><MorphText text={branching ? "Starting…" : "Branch"} numbers={false} /></button>
          <button type="button" class="dw-text" onclick={() => { branchOpen = false; branchError = null; }}>Cancel</button>
        </form>
      {:else}
        <div class="dw-row">
          {#if (saidReady || readReady) && token.tokenId != null}
            {#if saidReady && readReady}
              <Segmented label="Read this token" value={lensAs} options={[...LENS_NAMESPACES]}
                onchange={(value: LensNamespace) => (lensPairState.namespace = value)} />
            {/if}
            <button type="button" class="dw-quiet" disabled={probing || tokenProbed}
              data-tip={tokenProbed ? "Already a lens probe" : `Read this token at every layer, every token from now on: ${lensPair.hint}`}
              onclick={() => void probeToken()}>Probe <span class="dw-ws">{named.lead ? "·" : ""}</span>{named.empty ? "∅" : named.text}</button>
            <button type="button" class="dw-quiet" data-tip={`Add a lens term that pushes this way: ${lensPair.hint}`}
              onclick={steerTowardToken}>Steer toward it</button>
          {/if}
          <button type="button" class="dw-quiet" disabled={!canBranch || branching}
            data-tip={canBranch ? "Keep the text up to here and write a new ending; the original stays" : genStatus.active ? "Finish or stop the generation first" : "This token has no replay boundary"}
            onclick={() => void openBranch()}>Branch here</button>
        </div>
      {/if}
      {#if branchError}<p class="meta" role="alert">{branchError}</p>{/if}
    </section>
  {:else}
    <p class="meta">{params ? "This token is gone; select another" : "Select a token"}</p>
  {/if}
</div>

<style>
  .token {
    /* What a piece on this surface fills with: wash on paper, wash2 on the phone's wash sheet. */
    --token-fill: var(--wash);
    /* The readings size their columns to the view, not the window: the Inspector is 300 px at a
     * 768 px window and wider on a phone. */
    container: token / inline-size;
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: var(--s7);
    min-width: 0;
    align-content: start;
  }
  .token.mobile { --token-fill: var(--wash2); gap: var(--s6); }
  .token:focus { outline: none; }
  .top { gap: var(--s4); }
  .part { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s7); min-width: 0; }
  .mobile .part { gap: var(--s6); }
  /* The switch heads the readings below it. */
  .readings-switch { margin-bottom: calc(-1 * var(--s4)); }
  .readings-note { margin: 0; }

  .title-row { display: flex; align-items: center; gap: var(--s3); min-width: 0; }
  .title-row .dw-title { flex: 1 1 auto; min-width: 0; overflow: hidden; white-space: pre; text-overflow: ellipsis; cursor: help; }
  .steps { display: flex; flex: none; gap: var(--s1); }
  .fact {
    padding: 0;
    border: 0;
    background: transparent;
    color: var(--ink2);
    font: inherit;
    cursor: pointer;
    transition: color var(--fast) var(--ease);
  }
  .fact:hover, .fact[aria-pressed="true"] { color: var(--ink); }
  .dw-facts [data-tip] { cursor: help; }
  .recipe { overflow-wrap: anywhere; }
  .recipe .expr { color: var(--ink2); }

  /* Around the token: a wash with the token marked in ink, as the page marks it. */
  .ctx {
    margin: 0;
    padding: var(--s2) var(--s3);
    border-radius: var(--r-box);
    background: var(--token-fill);
    color: var(--ink2);
    font: var(--type-small);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .ctok { display: inline; padding: 0; border: 0; border-radius: 4px; background: transparent; color: inherit; font: inherit; white-space: inherit; text-align: inherit; cursor: pointer; transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease); }
  @media (hover: hover) { .ctok:hover { color: var(--ink); } }
  .ctok.here { background: var(--ink); color: var(--paper); box-shadow: 0 0 0 2px var(--ink); }

  .branch { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2); min-width: 0; }
  .branch input { flex: 1 1 12rem; min-width: 0; background: var(--token-fill); color: var(--ink); white-space: pre; }
  /* The field meets the Inspector's scroll edge, so its ring sits inside it. */
  .branch input:focus-visible { outline-offset: -2px; }

  /* On the phone's wash sheet the switch's track steps up to wash2, as the sheet's other pieces do. */
  .token.mobile .readings-switch :global(.seg) { background: var(--token-fill); }

  @media (max-width: 720px) {
    /* Phone sizes (the Phone canvas): every target 44 px. */
    .steps :global(button.icon) { width: 44px; height: 44px; }
    .readings-switch :global(.seg > button) { position: relative; height: 36px; }
    .readings-switch :global(.seg > button::after) { content: ""; position: absolute; inset: -4px 0; }
    .fact { position: relative; }
    .fact::after { content: ""; position: absolute; inset: -14px -8px; }
    /* Too small to tap and too close to stretch: on the phone the context is to read, and ‹ ›
     * (or the page above the sheet) move the token. */
    .ctok { pointer-events: none; }
    .branch input { min-height: 44px; }
    .branch .dw-text { min-height: 44px; }
  }
</style>
