<script lang="ts">
  import { createTokenArrival } from "../lib/tokenArrival";
  const arrivals = createTokenArrival();
  $effect.pre(() => {
    const pending = chatLog.pendingIndex === null ? null : chatLog.turns[chatLog.pendingIndex];
    arrivals.track(pending?.tokens ?? [], genStatus.active, "response");
    arrivals.track(pending?.thinkingTokens ?? [], genStatus.active, "thinking");
  });
  import { tokenInspectorUi } from "../lib/stores/drawers.svelte";
  // Flat completion buffer — the chat surface for base (non-chat)
  // models.  No bubbles, no role labels: the loom active path is joined
  // into a single continuous editable text surface, ``white-space:
  // pre-wrap``.  Per-token tinting is preserved for the assistant span
  // when a highlight probe is selected.
  //
  // "generate" runs with ``raw: true`` from the active leaf — the
  // whole buffer is effectively a prefill.  A mid-buffer edit lands as
  // a ``loomEdit`` (active leaf) or ``loomBranch`` (interior node)
  // first; toggling render mode never mutates the tree, only generation
  // and explicit edits do.

  import { onMount, tick } from "svelte";
  import { registerRawBufferController } from "../lib/workspaceController";
  import StatusFooter from "./StatusFooter.svelte";
  import PendingBubbles from "./PendingBubbles.svelte";
  import {
    chatLog,
    loomTree,
    genStatus,
    sendSubmit,
    sendGenerate,
    pendingActions,
    refreshLoomTree,
    sendStop,
    loomNavigate,
    highlightState,
    openDrawer,
    beginTokenHover,
    endTokenHover,
  } from "../lib/stores.svelte";
  import BaseModelTag from "../lib/ui/BaseModelTag.svelte";
  import { sessionState } from "../lib/stores/session.svelte";
  import { apiTree } from "../lib/runtime/services";
  import { completionAnchor } from "../lib/completionSelection";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import { Segmented } from "@stdlogits/ledger";
  import Select from "../lib/Select.svelte";
  import TokenLogitsPopover from "../lib/ui/TokenLogitsPopover.svelte";
  import { completionTokenViews, projectEditedTokens, type RawTokenView } from "../lib/rawTokenInspection";
  import { tokenRowToScore } from "../lib/stores/loom.svelte";
  import {
    ENTROPY_TARGET,
    PROBABILITY_TARGET,
    SURPRISE_TARGET,
    tokenLead,
  } from "../lib/tokens";
  import {
    highlightScoreFor,
    highlightStyleString,
  } from "../lib/highlight";

  // ---------- buffer text ----------

  /** The conversation flattened to one string — every turn's text
   *  concatenated in order. This is the raw prefix the model extends. */
  const bufferText = $derived(
    chatLog.turns.map((t) => t.text ?? "").join(""),
  );

  // Local editable mirror.  Synced from ``bufferText`` whenever the
  // tree changes and the user isn't mid-edit; user edits write here
  // and send / append land them on the tree.
  let draft = $state("");
  let dirty = $state(false);
  // Set the moment a send/append is fired: the tree hasn't caught
  // up to the locally-edited draft yet, so the buffer→draft sync below
  // is held until ``bufferText`` reaches the draft — without this the
  // draft would briefly snap back to the pre-edit text (a flash of the
  // user's typed tail vanishing) for the server round-trip.
  let committing = $state(false);
  let submissionFinishedAt: number | null = $state(null);
  let textareaRef: HTMLTextAreaElement | null = $state(null);
  let editError = $state<string | null>(null);
  let editFocusOrigin = $state<Element | null>(null);
  let editedTokenViews = $state<RawTokenView[] | null>(null);
  let selection = $state<{ start: number; end: number; text: string; nodeId: string | null } | null>(null);
  let preparingSelection = $state(false);

  // Re-sync the draft from the tree when the buffer changes and the
  // user has no pending edit.  ``dirty`` guards against clobbering an
  // in-progress edit; ``committing`` holds the just-sent draft until
  // the server-side node lands, after which streamed tokens flow in.
  $effect(() => {
    const text = bufferText;
    if (preparingSelection) return;
    if (committing) {
      // Hold until the tree genuinely carries the submitted draft as a
      // prefix — a *content* check, not a length one.  A length compare
      // releases early when a fast-streaming generation fills the
      // buffer to the right size before the authored span has landed
      // (or under a transient wrong-parent active path), snapping the
      // draft onto text that doesn't contain what the user wrote.
      if (!text.startsWith(draft)) {
        if (!genStatus.active && genStatus.finishedAt !== null && genStatus.finishedAt !== submissionFinishedAt) {
          committing = false;
          dirty = true;
        }
        return;
      }
      committing = false;
    }
    if (!dirty) {
      draft = text;
      editedTokenViews = null;
    }
  });

  function onInput(ev: Event): void {
    editError = null;
    const editor = ev.currentTarget as HTMLTextAreaElement;
    const next = editor.value;
    selection = { start: editor.selectionStart, end: editor.selectionEnd, text: next, nodeId: activeLeaf };
    editedTokenViews = next === bufferText ? null : projectEditedTokens(editedTokenViews ?? tokenViews, next);
    draft = next;
    dirty = draft !== bufferText;
    // Any keystroke supersedes a pending submission — resume live sync.
    committing = false;
  }

  function rememberSelection(): void {
    if (!textareaRef || genStatus.active || committing || preparingSelection) return;
    selection = { start: textareaRef.selectionStart, end: textareaRef.selectionEnd, text: draft, nodeId: activeLeaf };
  }

  onMount(() => registerRawBufferController({
    read: () => ({ text: draft, dirty, busy: committing || preparingSelection, selection: selection ? { start: selection.start, end: selection.end } : null }),
    update: async (change) => {
      editError = null;
      if (change.text !== undefined) {
        editedTokenViews = change.text === bufferText ? null : projectEditedTokens(editedTokenViews ?? tokenViews, change.text);
        draft = change.text;
        dirty = draft !== bufferText;
        committing = false;
        mode = "edit";
      }
      if (change.selection !== undefined) selection = { ...change.selection, text: draft, nodeId: activeLeaf };
      await tick();
      if (selection && textareaRef) textareaRef.setSelectionRange(selection.start, selection.end);
    },
    save: async () => {
      await appendEdit();
      if (editError) throw new Error(editError);
    },
    revert: revertEdit,
    prepareSelection: async (recomplete) => {
      const point = await completeFromSelection(recomplete, false);
      if (!point) throw new Error("Select a committed completion point before continuing. Save any draft edit first.");
      return point;
    },
  }));

  async function completeFromSelection(recomplete = false, generate = true): Promise<{ parent_node_id: string } | null> {
    if (!selectionCurrent || !selection || genStatus.active || committing || preparingSelection || dirty || pendingActions.queue.length || !loomTree.root_id) return null;
    const offset = recomplete ? selection.start : selection.end;
    const original = draft;
    const originalNodeId = activeLeaf;
    const originalRootId = loomTree.root_id;
    preparingSelection = true;
    editError = null;
    editFocusOrigin = document.activeElement;
    try {
      const anchor = completionAnchor(chatLog.turns, loomTree.root_id, offset);
      let parentNodeId = anchor.parentNodeId;
      if (anchor.branch) {
        const result = await apiTree.branch(anchor.branch.nodeId, anchor.branch.text, undefined, "user");
        parentNodeId = result.node_id;
      } else {
        await apiTree.navigate(parentNodeId);
      }
      await refreshLoomTree();
      if (loomTree.root_id !== originalRootId || loomTree.active_node_id !== parentNodeId) {
        throw new Error("The selected completion point changed. Select the text again.");
      }
      draft = original.slice(0, offset);
      editedTokenViews = null;
      selection = null;
      scrolledUp = false;
      committing = true;
      submissionFinishedAt = genStatus.finishedAt;
      if (generate) await sendGenerate({ raw: true, parent_node_id: parentNodeId, append_same_role: false, n: 1 });
      else { committing = false; dirty = false; }
      return { parent_node_id: parentNodeId };
    } catch (error) {
      if (loomTree.root_id !== originalRootId) {
        committing = false;
        if (!generate) throw error;
        return null;
      }
      draft = original;
      if (originalNodeId) {
        try {
          await apiTree.navigate(originalNodeId);
          await refreshLoomTree();
        } catch {
          // Keep the original text editable if the connection also prevents returning.
        }
      }
      restoreFailedEdit(error);
      if (!generate) throw error;
      return null;
    } finally {
      preparingSelection = false;
    }
  }

  $effect(() => {
    if (!editFocusOrigin || committing || genStatus.active || preparingSelection) return;
    const origin = editFocusOrigin;
    editFocusOrigin = null;
    void tick().then(() => {
      if (document.activeElement === origin || document.activeElement === document.body) {
        textareaRef?.focus({ preventScroll: true });
      }
    });
  });

  // ---------- send / generate / append ----------
  //
  // Flat mode is non-linear: editing text anywhere in the buffer is the
  // same operation as appending to the end.  Both collapse to a single
  // "divergence" — the first character that differs from the committed
  // tree text — and everything from there becomes one new span.  The
  // node containing the divergence (and its whole subtree) is preserved
  // untouched as the original branch; the edited tail is recorded as a
  // fresh span branched at that point.  Appending is just the special
  // case where the divergence sits at the very end of the buffer.

  /** The active leaf node — what a clean generation extends. */
  const activeLeaf = $derived(
    loomTree.loaded ? (loomTree.active_node_id ?? null) : null,
  );
  const selectionCurrent = $derived(selection !== null && selection.text === draft && selection.nodeId === activeLeaf);

  interface Divergence {
    /** New text — from the divergence offset to the end of the draft. */
    tail: string;
    /** Where the new span hangs: the diverging node's parent for a
     *  mid-buffer edit, the active leaf for a pure append.
     *  ``undefined`` means the buffer is clean — generate from the leaf
     *  with no new span at all. */
    parentNodeId: string | null | undefined;
  }

  /** Diff the draft against the committed buffer and locate the single
   *  span the change collapses to. */
  function resolveDivergence(): Divergence {
    if (!dirty) return { tail: "", parentNodeId: undefined };
    const turns = chatLog.turns;
    let startOffset = 0;
    for (let i = 0; i < turns.length; i++) {
      const turnText = turns[i].text ?? "";
      const slice = draft.slice(startOffset, startOffset + turnText.length);
      if (slice !== turnText) {
        // Divergence inside turns[i].  That node + its subtree stay as
        // the original branch; the tail (this node's start → end of
        // draft) becomes a new span branched as its sibling — i.e. a
        // child of the node's parent.
        const nid = turns[i].nodeId ?? null;
        const parentNodeId = nid
          ? (loomTree.nodes.get(nid)?.parent_id ?? null)
          : (activeLeaf ?? null);
        return { tail: draft.slice(startOffset), parentNodeId };
      }
      startOffset += turnText.length;
    }
    // No node diverged — the draft runs past the joined buffer.  The
    // appended tail hangs under the active leaf as a fresh child.
    return { tail: draft.slice(startOffset), parentNodeId: activeLeaf ?? null };
  }

  async function submitBuffer(): Promise<void> {
    if (genStatus.active || committing || preparingSelection) return;
    mode = "edit";
    scrolledUp = false;
    const d = resolveDivergence();
    // The divergence tail is an authored user span; generation occupies the
    // assistant role. A clean buffer omits the authored half entirely.
    const text = d.tail === "" ? null : d.tail;
    committing = true;
    editError = null;
    submissionFinishedAt = genStatus.finishedAt;
    dirty = false;
    try {
      await sendSubmit(text, text === null ? null : "user", "assistant",
        { raw: true, parent_node_id: d.parentNodeId });
    } catch (error) {
      restoreFailedEdit(error);
    }
  }

  // ---------- append without generating ----------

  /** Land the pending edit on the tree without generating — the same
   *  divergence branch ``submitBuffer`` would take, minus the decode. */
  async function appendEdit(): Promise<void> {
    if (!dirty || genStatus.active || committing || preparingSelection) return;
    editFocusOrigin = document.activeElement;
    const d = resolveDivergence();
    if (d.tail === "") {
      // A boundary deletion selects the shorter path; it must not silently
      // restore the discarded suffix on the next tree update.
      committing = true;
      try {
        await loomNavigate(d.parentNodeId ?? loomTree.root_id!);
        dirty = draft !== bufferText;
      } catch (error) {
        restoreFailedEdit(error);
      } finally {
        committing = false;
      }
      return;
    }
    dirty = false;
    committing = true;
    editError = null;
    submissionFinishedAt = genStatus.finishedAt;
    try {
      await sendSubmit(d.tail, "user", null,
        { raw: true, parent_node_id: d.parentNodeId ?? null });
    } catch (error) {
      restoreFailedEdit(error);
    }
  }

  function restoreFailedEdit(error: unknown): void {
    committing = false;
    dirty = draft !== bufferText;
    editError = userFacingError(error, "The edit could not be saved.");
  }

  function revertEdit(): void {
    draft = bufferText;
    dirty = false;
    committing = false;
    editError = null;
    editedTokenViews = null;
    void tick().then(() => textareaRef?.focus({ preventScroll: true }));
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey) && !ev.isComposing) {
      ev.preventDefault();
      submitBuffer();
      return;
    }
    if (ev.key === "Escape" && genStatus.active) {
      ev.preventDefault();
      sendStop();
    }
  }

  // ---------- token inspection ----------

  const tokenViews = $derived(completionTokenViews(chatLog.turns));
  const currentTokenViews = $derived(editedTokenViews ?? tokenViews);
  const earlierGenerations = $derived(
    [...loomTree.nodes.values()]
      .filter(node => node.recipe && node.tokens?.length && !loomTree.activePath.includes(node.id))
      .sort((a, b) => b.created_at - a.created_at),
  );
  let inspectionSource = $state("current");
  const inspectionSources = $derived([
    { value: "current", label: dirty ? "Current draft" : "Current text" },
    ...(dirty && tokenViews.some(view => view.tok) ? [{ value: "recorded", label: "Original text before edit" }] : []),
    ...earlierGenerations.map((node, index) => ({
      value: node.id,
      label: `Earlier generation ${earlierGenerations.length - index} · ${node.text.slice(0, 48).replace(/\s+/g, " ")}`,
    })),
  ]);
  const inspectedTokenViews = $derived.by(() => {
    if (inspectionSource === "recorded") return tokenViews;
    const node = earlierGenerations.find(node => node.id === inspectionSource);
    if (node) return completionTokenViews([{
      nodeId: node.id, role: node.role, text: node.text, generated: true,
      tokens: node.tokens!.map(tokenRowToScore),
    }]).map(view => ({ ...view, turnIdx: null }));
    return currentTokenViews;
  });
  $effect(() => {
    if (!inspectionSources.some(source => source.value === inspectionSource)) inspectionSource = "current";
  });
  let tokenPopup = $state<{ view: RawTokenView; anchor: HTMLElement } | null>(null);

  // ---------- edit / inspect mode ----------
  //
  // Editing changes the draft projection, never the recorded token rows.
  // Inspection can show either that projection or an original generation.
  type Mode = "edit" | "inspect";
  let mode: Mode = $state<Mode>("edit");

  const showColorMirror = $derived(
    mode === "edit" && !committing && (dirty || (highlightState.target !== null && currentTokenViews.some(view => view.tok))) &&
    currentTokenViews.map((token) => token.text).join("") === (dirty ? draft : bufferText),
  );
  const colorNotice = $derived.by(() => {
    if (!highlightState.target) return null;
    if (committing || !tokenViews.some(view => view.tok)) return null;
    const targets = [highlightState.target, ...(highlightState.compareTwo && highlightState.compareTarget
      ? [highlightState.compareTarget] : [])];
    const scores = tokenViews.flatMap(({ tok }) => targets.flatMap((target) => {
      const score = tok ? highlightScoreFor(tok, target) : undefined;
      return score === undefined ? [] : [score];
    }));
    if (scores.length === 0) return "No readings were recorded for this color.";
    if (scores.some((score) => score !== 0)) return null;
    const reading = targets.length > 1 ? "values are" : highlightState.target === SURPRISE_TARGET ? "token surprisal is"
      : highlightState.target === ENTROPY_TARGET ? "sampler entropy is" : "values are";
    return `Recorded ${reading} zero, so these tokens stay uncolored.`;
  });

  /** Whether any clickable (generated) token exists to inspect. */
  const hasClickableTokens = $derived(
    currentTokenViews.some(view => view.tok !== null),
  );
  const inspectableTokenIndices = $derived(
    inspectedTokenViews.flatMap((token, index) => token.tok === null ? [] : [index]),
  );
  let focusedTokenViewIndex = $state(-1);

  $effect(() => {
    const indices = inspectableTokenIndices;
    if (indices.length === 0) {
      focusedTokenViewIndex = -1;
    } else if (!indices.includes(focusedTokenViewIndex)) {
      focusedTokenViewIndex = indices[0];
    }
  });

  const hasRecordedTokens = $derived(hasClickableTokens || tokenViews.some(view => view.tok) || earlierGenerations.length > 0);
  const canInspect = $derived(hasRecordedTokens && !committing && !genStatus.active && !preparingSelection);
  const modeItems = $derived([
    { value: "edit" as const, label: "Edit text", title: "Edit the completion text" },
    {
      value: "inspect" as const,
      label: "Inspect tokens",
      disabled: mode !== "inspect" && !canInspect,
      title: canInspect || mode === "inspect"
        ? "Choose a token to inspect readings or branch"
        : genStatus.active
          ? "Stop or finish generation before inspecting"
          : committing
          ? "Wait for the edit to finish saving"
          : "Continue text to record tokens",
    },
  ]);

  /** Fall back to edit when nothing remains to inspect (buffer cleared,
   *  conversation reset) so the surface never strands on an empty view. */
  $effect(() => {
    if (mode === "inspect" && !hasRecordedTokens) mode = "edit";
  });

  function changeMode(next: Mode): void {
    if (modeItems.find(item => item.value === next)?.disabled) return;
    tokenPopup = null;
    if (next === "inspect" && !hasClickableTokens) {
      inspectionSource = tokenViews.some(view => view.tok) && dirty ? "recorded" : earlierGenerations[0]?.id ?? "current";
    }
    mode = next;
  }

  function inspectTooltip(v: RawTokenView): string {
    const tok = v.tok;
    if (!tok) return "";
    const parts: string[] = [];
    const sc = highlightScoreFor(tok, highlightState.target);
    if (sc !== undefined && highlightState.target) {
      if (highlightState.target === SURPRISE_TARGET) {
        if (tok.logprob != null) {
          parts.push(`token surprisal ${(-tok.logprob).toFixed(3)} nats`);
        }
      } else if (highlightState.target === PROBABILITY_TARGET && tok.logprob != null) {
        parts.push(`token probability ${(Math.exp(Math.min(0, tok.logprob)) * 100).toFixed(1)}%`);
      } else if (highlightState.target === ENTROPY_TARGET && tok.samplerEntropy != null) {
        parts.push(`sampler entropy ${tok.samplerEntropy.toFixed(3)} nats`);
      } else {
        parts.push(`${highlightState.target} ${sc >= 0 ? "+" : ""}${sc.toFixed(3)}`);
      }
    }
    const n = tok.topAlts?.length ?? 0;
    parts.push(n > 0 ? `Select to see ${n} alternatives` : "Select to see the recorded probability");
    return parts.join(" · ");
  }

  function tokenAccessibleName(v: RawTokenView): string {
    const token = v.text.trim() || "whitespace";
    const alternatives = v.tok?.topAlts?.length ?? 0;
    return alternatives > 0
      ? `Inspect token ${token}, ${alternatives} alternatives`
      : `Inspect token ${token}`;
  }

  function focusInspectableToken(index: number): void {
    focusedTokenViewIndex = index;
    queueMicrotask(() => {
      logRef?.querySelector<HTMLElement>(`[data-token-view-index="${index}"]`)?.focus();
    });
  }

  function onInspectableTokenKeydown(ev: KeyboardEvent, viewIndex: number, token: RawTokenView): void {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      openToken(token, ev.currentTarget as HTMLElement);
      return;
    }
    const current = inspectableTokenIndices.indexOf(viewIndex);
    if (current < 0) return;
    let next = current;
    if (ev.key === "ArrowRight" || ev.key === "ArrowDown") {
      next = (current + 1) % inspectableTokenIndices.length;
    } else if (ev.key === "ArrowLeft" || ev.key === "ArrowUp") {
      next = (current - 1 + inspectableTokenIndices.length) % inspectableTokenIndices.length;
    } else if (ev.key === "Home") {
      next = 0;
    } else if (ev.key === "End") {
      next = inspectableTokenIndices.length - 1;
    } else {
      return;
    }
    ev.preventDefault();
    focusInspectableToken(inspectableTokenIndices[next]);
  }

  function showTokenDetails(v: RawTokenView): void {
    if (v.turnIdx === null) return;
    openDrawer("token_drilldown", {
      turnIdx: v.turnIdx,
      tokenIdx: v.tokenIdx,
      isThinking: v.isThinking,
      initialTab: "logits",
    });
  }

  function openToken(view: RawTokenView, anchor: HTMLElement): void {
    if (!view.tok) return;
    if (tokenInspectorUi.docked && view.turnIdx !== null) {
      tokenPopup = null;
      openDrawer("token_drilldown", { turnIdx: view.turnIdx, tokenIdx: view.tokenIdx, isThinking: view.isThinking });
      return;
    }
    tokenPopup = { view, anchor };
  }

  function hoverToken(v: RawTokenView): void {
    if (!v.tok) return;
    beginTokenHover(v.tok, v.nodeId ?? undefined);
  }

  let logRef: HTMLDivElement | null = $state(null);
  let colorMirrorRef: HTMLDivElement | null = $state(null);
  let scrolledUp = $state(false);
  function syncColorScroll(): void {
    if (!colorMirrorRef || !textareaRef) return;
    colorMirrorRef.scrollTop = textareaRef.scrollTop;
    colorMirrorRef.scrollLeft = textareaRef.scrollLeft;
  }
  function onScroll(ev: Event): void {
    const el = ev.currentTarget as HTMLElement;
    scrolledUp = el.scrollHeight - el.scrollTop - el.clientHeight >= 8;
    if (mode === "edit") syncColorScroll();
  }
  function followLatest(): void {
    scrolledUp = false;
    const el = mode === "edit" ? textareaRef : logRef;
    if (el) el.scrollTop = el.scrollHeight;
    syncColorScroll();
  }
  $effect(() => {
    if (showColorMirror) void tick().then(syncColorScroll);
  });
  $effect(() => {
    void bufferText;
    if (!dirty && !scrolledUp) {
      void tick().then(() => {
        if (!dirty && !scrolledUp) followLatest();
      });
    }
  });

  onMount(() => {
    draft = bufferText;
  });
</script>

<div class="raw-buffer" aria-label="Completion buffer">
  <div class="raw-head">
    <span class="head-label"><label for="completion-buffer">Text completion</label>{#if sessionState.info?.is_base_model}<BaseModelTag />{/if}{#if dirty}<span class="draft-state" role="status">unsaved edit</span>{/if}</span>
    <Segmented label="Buffer mode" value={mode} onchange={changeMode} options={modeItems.map(item => ({ value: item.value, label: item.label, hint: item.title }))} />
  </div>
  <p class="sr-only" id="completion-hint">{mode === "inspect"
    ? "Select a recorded token to see its probabilities. Arrow keys move between tokens."
    : "Continue at the end, cursor, or selection. Originals stay saved in the loom."}</p>
  {#if colorNotice}<p class="color-notice meta" role="status">{colorNotice}</p>{/if}
  {#if mode === "inspect" && inspectionSources.length > 1}
    <div class="inspection-source" data-tip={inspectionSource !== "current" ? "As recorded; your text is unchanged" : undefined}>
      <Select value={inspectionSource} options={inspectionSources} ariaLabel="Text to inspect"
        ariaDescribedby={inspectionSource !== "current" ? "inspection-source-note" : undefined}
        onchange={(value) => { tokenPopup = null; inspectionSource = value; }} />
      <span class="sr-only" id="inspection-source-note">As recorded; your text is unchanged</span>
    </div>
  {/if}

  <div class="surface" class:inspecting={mode === "inspect"} class:generation-active={genStatus.active} bind:this={logRef} onscroll={mode === "inspect" ? onScroll : undefined}>
    {#if mode === "inspect"}
      <div class="inspect" dir="auto" aria-label="Completion tokens">
        {#each inspectedTokenViews as v, i (i)}
          {#if !v.tok}
            <span class="seg plain" class:origin-user={v.source === "user"} class:origin-draft={v.source === "draft"}
              {...{ "aria-description": (v.source === "draft" ? "Unsaved user edit · no recorded probabilities" : v.source === "model" ? "Model text · no recorded probabilities" : "User text · no recorded probabilities") }}>{v.text}</span>
          {:else}
            {@const parts = tokenLead(v.text)}
            {parts.lead}<span
              class="seg tok clickable"
              class:origin-user={v.source === "user"}
              class:origin-model={v.source === "model"}
              class:context-changed={v.contextChanged}
              data-cursor="inspect"
              class:tinted={highlightState.target !== null}
              class:has-alts={(v.tok?.topAlts?.length ?? 0) > 0}
              style={v.tok ? highlightStyleString(v.tok) : ""}
              {...{ "aria-description": (inspectTooltip(v)) }}
              role="button"
              tabindex={i === focusedTokenViewIndex ? 0 : -1}
              data-token-view-index={i}
              aria-label={tokenAccessibleName(v)}
              aria-haspopup="dialog"
              aria-expanded={tokenPopup?.anchor.dataset.tokenViewIndex === String(i)}
              onpointerenter={() => hoverToken(v)}
              onpointerleave={endTokenHover}
              onfocus={() => hoverToken(v)}
              onblur={endTokenHover}
              onclick={(event) => {
                focusedTokenViewIndex = i;
                openToken(v, event.currentTarget);
              }}
              onkeydown={(ev) => onInspectableTokenKeydown(ev, i, v)}
            >{parts.body}</span>
          {/if}
        {/each}
      </div>
    {:else}
      {#if showColorMirror}
        <div class="color-mirror" bind:this={colorMirrorRef} dir="auto" aria-hidden="true">
          {#each currentTokenViews as token, i (i)}{@const parts = tokenLead(token.text)}{parts.lead}<span use:arrivals.reveal={!genStatus.active || dirty || document.activeElement === textareaRef ? null : token.tok} class:origin-draft={token.source === "draft"} style={token.tok ? highlightStyleString(token.tok) : ""}>{parts.body}</span>{/each}<span>{"\u200b"}</span>
        </div>
      {/if}
      <textarea
        id="completion-buffer"
        class="buffer-text"
        bind:this={textareaRef}
        dir="auto"
        value={draft}
        oninput={onInput}
        onkeydown={onKeydown}
        onselect={rememberSelection}
        onkeyup={rememberSelection}
        onpointerup={rememberSelection}
        onscroll={onScroll}
        spellcheck="false"
        placeholder="The door opened onto…"
        aria-label="Editable completion buffer"
        aria-describedby={editError ? "completion-hint completion-error" : "completion-hint"}
        aria-keyshortcuts="Meta+Enter Control+Enter"
        aria-invalid={editError ? true : undefined}
        readonly={genStatus.active || committing || preparingSelection}
      ></textarea>
    {/if}
    {#if scrolledUp && !dirty && mode === "edit"}
      <div class="latest-action"><button type="button" class="dw-quiet" onclick={() => { followLatest(); textareaRef?.focus({ preventScroll: true }); }}>Latest</button></div>
    {/if}
  </div>

  {#if tokenPopup?.view.tok}
    {@const view = tokenPopup.view}
    {#key tokenPopup.anchor}
      <TokenLogitsPopover token={tokenPopup.view.tok} anchor={tokenPopup.anchor}
        isCurrent={anchor => tokenPopup?.anchor === anchor}
        source={inspectionSource !== "current" ? "Original recorded token" : tokenPopup.view.source === "user" ? "User text · recorded" : "Model token"}
        contextChanged={tokenPopup.view.contextChanged}
        onclose={() => tokenPopup = null}
        ondetails={view.turnIdx === null ? undefined : () => showTokenDetails(view)} />
    {/key}
  {/if}

  {#if editError}<p class="completion-error meta" id="completion-error" role="alert">{editError} Your text is still here.</p>{/if}
  <StatusFooter savedEdit={!genStatus.active && genStatus.finishReason === "stop" && genStatus.tokensSoFar === 0 && chatLog.turns.at(-1)?.generated === false} />
  <PendingBubbles />

  <div class="actions" role="group" aria-label="Completion actions">
    <div class="edit-actions" class:shown={dirty}>
      <button type="button" class="dw-text" onclick={() => void appendEdit()}
        disabled={!dirty || genStatus.active || committing || preparingSelection} data-tip="Save this edit without generating">Save edit</button>
      <button type="button" class="dw-text" onclick={revertEdit} disabled={!dirty || genStatus.active || committing || preparingSelection}>Discard</button>
    </div>
    <div class="generation-actions">
      <button type="button" class="dw-text" onclick={() => completeFromSelection()}
        data-tip="Save edits first; the original stays saved"
        disabled={mode !== "edit" || !selectionCurrent || !selection || !draft.length || dirty || genStatus.active || committing || preparingSelection || pendingActions.queue.length > 0}>
        {selection && selection.start !== selection.end ? "Continue after selection" : "Continue from cursor"}
      </button>
      <button type="button" class="dw-text" onclick={() => completeFromSelection(true)}
        data-tip="Replaces the selection; the original stays saved"
        disabled={mode !== "edit" || !selectionCurrent || !selection || selection.start === selection.end || dirty || genStatus.active || committing || preparingSelection || pendingActions.queue.length > 0}>
        Re-complete selection
      </button>
      {#if genStatus.active}
        <button type="button" class="dw-quiet" onclick={sendStop} data-tip="Esc">Stop</button>
      {:else}
        <button type="button" class="dw-button" onclick={submitBuffer} disabled={committing || preparingSelection}
          aria-busy={committing || undefined} data-tip="⌘ or Ctrl + Enter">Continue text</button>
      {/if}
    </div>
  </div>
</div>

<style>
  .head-label { display: inline-flex; align-items: baseline; gap: var(--s3); min-width: 0; color: var(--ink3); font: var(--type-meta); }
  .draft-state { color: var(--ink2); }
  .color-notice { overflow-wrap: anywhere; }
  .completion-error { color: var(--ink2); }
  .inspection-source { display: flex; align-self: flex-start; align-items: center; min-width: 0; max-width: 100%; }
  .inspection-source :global(.sk-select) { min-width: 0; max-width: min(30rem, 100%); }
  .raw-buffer {
    display: flex;
    flex: 1 0 auto;
    flex-direction: column;
    min-height: 0;
    gap: var(--s4);
    color: var(--ink);
    font: var(--type-body);
  }
  .raw-head { display: flex; flex: 0 0 auto; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s2) var(--s4); }
  /* The completion as one reading surface: a wash, no line, the text in the body face at the
   * reading column's line height (room for the tint's spread). */
  .surface {
    flex: 1 0 calc(2lh + var(--s4) * 2);
    position: relative;
    overflow: clip;
    min-height: calc(2lh + var(--s4) * 2);
    border-radius: var(--r-box);
    background: var(--wash);
    line-height: 32px;
  }
  .surface.inspecting { overflow-y: auto; }
  .latest-action { position: absolute; inset-inline-end: var(--s3); bottom: var(--s3); z-index: 2; }
  .latest-action .dw-quiet { background: var(--wash2); }
  .buffer-text,
  .color-mirror,
  .inspect {
    margin: 0;
    padding: var(--s4);
    font: var(--type-body);
    line-height: 32px;
    white-space: pre-wrap;
    word-break: break-word;
    box-sizing: border-box;
  }
  .buffer-text, .color-mirror { scrollbar-gutter: stable; }
  .color-mirror { position: absolute; inset: 0; overflow: hidden; color: transparent; pointer-events: none; }
  .inspect { min-height: 100%; color: var(--ink); }
  /* Prompt and authored text: present, quieter. */
  .inspect .plain, .inspect .tok.origin-user { color: var(--ink2); }
  .inspect .tok {
    display: inline;
    min-width: 0;
    min-height: 0;
    padding: 0;
    border: 0;
    border-radius: 4px;
    background: transparent;
    font: inherit;
    color: inherit;
    white-space: inherit;
    text-align: inherit;
    vertical-align: baseline;
    transition: background-color var(--fast) var(--ease), box-shadow var(--fast) var(--ease), color var(--fast) var(--ease);
  }
  .inspect .context-changed { text-decoration: underline dashed var(--ink3); text-underline-offset: 4px; }
  .inspect .clickable { cursor: pointer; }
  @media (hover: hover) { .inspect .clickable:hover { background: var(--wash2); padding-block: 2px; } }
  .inspect .clickable[aria-expanded="true"] { background: var(--ink); color: var(--paper); padding-block: 2px; }
  .origin-draft { text-decoration: underline dashed var(--ink3); text-underline-offset: 4px; }
  /* Tokens that recorded their alternatives carry a faint underline, so the forkable ones show. */
  .inspect .has-alts { text-decoration: underline dotted var(--ink3); text-underline-offset: 4px; }
  .buffer-text {
    display: block;
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 100%;
    border: 0;
    background: transparent;
    color: var(--ink);
    resize: none;
  }
  .buffer-text:focus { outline: none; }
  .buffer-text:focus-visible { outline: 2px solid var(--ink); outline-offset: -2px; border-radius: var(--r-box); }
  .buffer-text::placeholder { color: var(--ink3); }

  .actions { display: flex; flex: 0 0 auto; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s2) var(--s4); }
  .edit-actions, .generation-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s4); }
  /* Save and Discard take space only while there is an edit, so no empty row holds their place (on
   * a narrow page they wrap onto a line of their own). They fade in when the edit starts. */
  .edit-actions:not(.shown) { display: none; }
  .edit-actions.shown { transition: opacity var(--fast) var(--ease); }
  @starting-style { .edit-actions.shown { opacity: 0; } }
  .generation-actions { margin-left: auto; }
  @media (max-width: 720px) {
    /* The phone's 16 px gutter, as the conversation's log and dock keep it. */
    .raw-buffer { padding: var(--s4) var(--s4) var(--s2); }
    .surface, .buffer-text, .color-mirror, .inspect { font-size: 16px; }
    .actions .dw-text { min-height: 44px; }
    /* Phone sizes (the Phone canvas): the mode switch's options reach 44 px. */
    .raw-head :global(.seg > button) { position: relative; height: 36px; }
    .raw-head :global(.seg > button::after) { content: ""; position: absolute; inset: -4px 0; }
  }
</style>
