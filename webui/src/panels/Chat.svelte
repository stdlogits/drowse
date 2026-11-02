<script lang="ts">
  import { Switch } from "@stdlogits/ledger";
  import { chatViewSchema } from "../lib/interfaceSchemas";
  import { registerInterfaceController } from "../lib/workspaceController";
  import { ToolError } from "../lib/webmcp/types";
  import { onMount as onInterfaceMount } from "svelte";

  import { createTokenArrival } from "../lib/tokenArrival";
  const arrivals = createTokenArrival();
  $effect.pre(() => {
    const pending = chatLog.pendingIndex === null ? null : chatLog.turns[chatLog.pendingIndex];
    arrivals.track(pending?.tokens ?? [], genStatus.active, "response");
    arrivals.track(pending?.thinkingTokens ?? [], genStatus.active, "thinking");
  });
  import { MorphText } from "@stdlogits/ledger";
  import { aggregateReadingText } from "../lib/charts/depth";
  import { createTokenViewCache } from "../lib/runtime/tokenViews";
  import { Icon } from "@stdlogits/ledger";
  import { Expander } from "@stdlogits/ledger";
  import RecipeBar from "./RecipeBar.svelte";
  import { selection, clearSelection } from "../lib/stores/selection.svelte";
  import { workbenchUi } from "../lib/stores/workbench.svelte";
  /** On a phone the workbench lays the log, the inspector sheet and this dock out as one column. */
  let { phone = false }: { phone?: boolean } = $props();
  // Chat panel: collapsible thinking per turn, per-token tinted spans,
  // optional compare-two stripe overlay, click-token drilldown, send /
  // stop, and an A/B split-view container.
  //
  // Single source of truth for state lives in ``lib/stores.svelte`` —
  // this file is presentation + local-only UI bits (textarea state,
  // scroll bookkeeping, per-turn collapse state).  The WS lifecycle and
  // gen-status accounting belongs to the store.
  //
  // Turn surface follows the cast model:
  // every speaker gets one neutral card, with the chat's blob avatar for
  // the model and an accent marker for the user. Role labels remain
  // arbitrary strings. Analysis badges appear only when
  // their backing artifacts exist. System turns render as stage directions (a
  // note about the scene, not a speaker). The ``speaking as`` chips in
  // the composer are the promoted SamplingStrip role boxes — same client
  // state, same wire block.
  //
  // Token rhythm: strip leading whitespace after </think>, with plain-text
  // fall-through when no probe is selected.

  import { onMount, tick, untrack } from "svelte";
  import { registerComposerController } from "../lib/workspaceController";
  import { SvelteMap } from "svelte/reactivity";
  import { expand, fadeOut, reducedMotion, rise, GLIDE } from "@stdlogits/ledger";
  import StatusFooter from "./StatusFooter.svelte";
  import PendingBubbles from "./PendingBubbles.svelte";
  import RawBuffer from "./RawBuffer.svelte";
  import {
    autoRegenState,
    abState,
    chatLog,
    highlightState,
    loomTree,
    pinnedComparison,
    setHighlightTarget,
    setCompareTarget,
    toggleCompareTwo,
    setCompareTwo,
    unpinComparison,
    probeRack,
    sendSubmit,
    sendShadowGenerate,
    currentSteeringExpression,
    sendStop,
    genStatus,
    openDrawer,
    inputHistory,
    inputRestore,
    pushInputHistory,
    navigateInputHistory,
    cancelInputPull,
    consumePulledSlot,
    loomRegenerateNode,
    enqueuePending,
    pendingActions,
    cancelPendingAction,
    isPendingBusy,
    toggleAutoRegen,
    disableAutoRegen,
    setAutoRegenMode,
    setAutoRegenCustom,
    customComparisonProblem,
    effectiveRawMode,
    genUiMode,
    roleDisplayLabel,
    castState,
    samplingState,
    sessionState,
    clearChat,
    beginTokenHover,
    endTokenHover,
  } from "../lib/stores.svelte";
  import type { AutoRegenMode } from "../lib/stores.svelte";
  import type { ChatRole, ChatTurn, TokenScore } from "../lib/types";
  import {
    ENTROPY_TARGET,
    SURPRISE_TARGET,
    tokenLead,
  } from "../lib/tokens";
  import {
    highlightStyleString,
  } from "../lib/highlight";
  import Select from "../lib/Select.svelte";
  import Combobox from "../lib/Combobox.svelte";
  import { runtimeOperationAvailability } from "../lib/runtime/ui-capabilities";

  function startsCompact(): boolean {
    return typeof window !== "undefined" && window.matchMedia(
      "(max-width: 720px), (max-height: 600px), (pointer: coarse) and (max-width: 1120px)",
    ).matches;
  }

  // --------------------------------------------------------------- input --

  let input = $state("");
  let chatRef: HTMLDivElement | null = $state(null);
  let textareaRef: HTMLTextAreaElement | null = $state(null);
  let clearConversationArmed = $state(false);
  let rolePlanOpen = $state(false);
  let rolePlanShell: HTMLDivElement | null = $state(null);
  let rolePlanAnimation: Animation | null = null;
  let composerHeight: number | null = $state(null);
  let composerMaxHeight = $state(420);
  let composerManuallySized = $state(false);
  let composerPointerHeld = $state(false);
  let resizeStart = $state<{
    pointerId: number;
    y: number;
    height: number;
  } | null>(null);
  const COMPOSER_MIN_HEIGHT = 48;
  const COMPOSER_DEFAULT_HEIGHT = 48;
  const COMPACT_COMPOSER_MIN_HEIGHT = 32;
  const COMPACT_COMPOSER_DEFAULT_HEIGHT = 32;
  let composerMinHeight = $state(
    startsCompact() ? COMPACT_COMPOSER_MIN_HEIGHT : COMPOSER_MIN_HEIGHT,
  );
  let composerDefaultHeight = $state(
    startsCompact() ? COMPACT_COMPOSER_DEFAULT_HEIGHT : COMPOSER_DEFAULT_HEIGHT,
  );
  const composerControlHeight = $derived(Math.round(Math.max(
    composerMinHeight,
    Math.min(composerMaxHeight, composerHeight ?? composerDefaultHeight),
  )));
  const canClearConversation = $derived(
    loomTree.root_id !== null && loomTree.active_node_id !== loomTree.root_id,
  );

  function requestClearConversation(): void {
    if (!clearConversationArmed) {
      clearConversationArmed = true;
      return;
    }
    clearConversationArmed = false;
    clearChat();
  }

  /** Auto-grow the textarea from its comfortable default through 6 rows.
   *  With ``box-sizing: border-box`` set in CSS, ``el.scrollHeight``
   *  includes top/bottom padding — which is exactly what we want to write
   *  back into ``style.height``, so a one-line draft sits flush with no
   *  residual scrollbar.  The vertical scrollbar is suppressed unless the
   *  content actually overflows the 6-row cap. */
  function autosize(): void {
    const el = textareaRef;
    if (!el) return;
    if (composerManuallySized && composerHeight !== null) {
      el.style.height = `${composerHeight}px`;
      el.style.overflowY = el.scrollHeight > el.clientHeight ? "auto" : "hidden";
      return;
    }
    el.style.height = "auto";
    const rowHeight = 22; // mono line-height fudge — matches font-size-base
    const maxH = Math.min(rowHeight * 6, composerMaxHeight);
    const next = Math.min(Math.max(el.scrollHeight, composerDefaultHeight), maxH);
    el.style.height = `${next}px`;
    // Only show the scrollbar once we've actually hit the cap.  Without
    // this the browser's "always reserve a scrollbar gutter" heuristic
    // paints a 1-2px up/down nub on single-line input.
    el.style.overflowY = el.scrollHeight > maxH ? "auto" : "hidden";
  }

  function updateComposerBounds(): void {
    if (!chatRef) return;
    chatRef.style.setProperty("--chat-scrollbar-width", `${chatRef.offsetWidth - chatRef.clientWidth}px`);
    const compact = chatRef.clientHeight < 560 || chatRef.clientWidth <= 620 ||
      (window.visualViewport?.height ?? window.innerHeight) < 600;
    composerMinHeight = compact ? COMPACT_COMPOSER_MIN_HEIGHT : COMPOSER_MIN_HEIGHT;
    composerDefaultHeight = compact
      ? COMPACT_COMPOSER_DEFAULT_HEIGHT
      : COMPOSER_DEFAULT_HEIGHT;
    const reservedHeight = compact ? 190 : 300;
    composerMaxHeight = Math.max(
      composerMinHeight,
      Math.min(480, Math.floor(chatRef.clientHeight - reservedHeight)),
    );
    if (composerHeight !== null && composerHeight > composerMaxHeight) {
      composerHeight = composerMaxHeight;
    }
    queueMicrotask(autosize);
  }

  // The reading tools (word colours, automatic comparison) fold under the composer row.
  let readingOpen = $state(false);
  // On a phone the composer's extras (who writes, reasoning, colours, clear) fold behind More.
  let extrasOpen = $state(false);
  // Stop's slot stays a moment after the reply ends (the button hidden and inert): the row's
  // controls shift when it goes, and a quick second click on Stop would land on Clear.
  const STOP_SLOT_HOLD_MS = 1200;
  let stopSlotHeld = $state(false);
  $effect(() => {
    if (genStatus.active) { stopSlotHeld = true; return; }
    const timer = setTimeout(() => { stopSlotHeld = false; }, STOP_SLOT_HOLD_MS);
    return () => clearTimeout(timer);
  });
  const modelShort = $derived(sessionState.info?.model_id.split("/").at(-1) ?? "drowse");
  /** A sampling value as the line shows it: the user's, else the model's default. */
  function shown(value: number | null | undefined): string {
    return value === null || value === undefined || !Number.isFinite(value) ? "default" : String(Number(value.toFixed(2)));
  }
  const samplingSummary = $derived([
    `temperature ${shown(samplingState.temperature ?? sessionState.info?.config.temperature)}`,
    `top-p ${shown(samplingState.top_p ?? sessionState.info?.config.top_p)}`,
    `${samplingState.max_tokens ?? sessionState.info?.config.max_tokens ?? "default"} tokens`,
  ].join(" · "));

  /** The sampling line opens the instruments: their Sampling section, or the phone's sheet. */
  function openSampling(): void {
    clearSelection();
    if (phone) {
      workbenchUi.sheet = true;
      workbenchUi.sheetOpen = true;
    }
    void tick().then(() => document.getElementById("instrument-sampling")?.scrollIntoView({ block: "start", behavior: "smooth" }));
  }

  /** A seat reads You and the model's short name; a role label the user set reads as itself. */
  function seatLabel(turn: ChatTurn): string {
    const label = roleDisplayLabel(turn.role, turn.roleLabel);
    if (turn.role === "user" && label === defaultUserLabel) return "You";
    if (turn.role === "assistant" && label === defaultAssistantLabel) return modelShort;
    return label;
  }

  /** What the words are coloured by, for the key under a reply (the swatch and its name). */
  const tintKey = $derived.by(() => {
    const target = highlightState.target;
    if (!target) return null;
    const label = highlightOptions.find((option) => option.value === target)?.label ?? target;
    const second = highlightState.compareTwo ? highlightState.compareTarget : null;
    if (!second) return label;
    return `${label} and ${compareOptions.find((option) => option.value === second)?.label ?? second}`;
  });

  // A selected token stays in view: on a phone the sheet rises under it and shrinks the log. Out of
  // view, the log stops following the reply's end (which would scroll it away again) and brings
  // the token in.
  $effect(() => {
    if (selection.current.kind !== "token") return;
    const delay = phone && !reducedMotion() ? GLIDE + 40 : 0;
    const timer = setTimeout(() => {
      const token = chatRef?.querySelector<HTMLElement>(".tok.sel");
      if (!token || !logRef) return;
      const view = logRef.getBoundingClientRect();
      const at = token.getBoundingClientRect();
      if (at.top >= view.top && at.bottom <= view.bottom) return;
      pauseAutoScroll();
      token.scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" });
    }, delay);
    return () => clearTimeout(timer);
  });

  function isSelectedToken(turnIdx: number, tokenIdx: number, thinking: boolean): boolean {
    const current = selection.current;
    if (current.kind !== "token") return false;
    const at = current.params as { turnIdx?: number; tokenIdx?: number; isThinking?: boolean } | null;
    return at?.turnIdx === turnIdx && at?.tokenIdx === tokenIdx && Boolean(at?.isThinking) === thinking;
  }

  $effect(() => {
    // Run autosize whenever ``input`` changes — bind:value + an effect is
    // simpler than wiring an oninput handler that has to cooperate with
    // bind:.
    void input;
    autosize();
  });

  // --- Unified submission -----------------------------------------------
  // The composer exposes the native submit contract directly: one role for
  // an authored line, plus an independently selected continuation role (or
  // none). This keeps arbitrary scene sequences visible and makes same-role
  // prefill a single send: author a prefix and continue the same role.
  const rawMode = $derived.by(() => {
    void genUiMode.mode;
    return effectiveRawMode();
  });

  const activeNodeId = $derived(
    loomTree.loaded ? (loomTree.active_node_id ?? null) : null,
  );
  const sceneMode = $derived(sessionState.info?.scene_mode ?? false);
  type ContinuationRole = ChatRole | "none";
  const ROLE_SLUG_RE = /^[a-z0-9._-]+$/;

  let authoredRole = $state<ChatRole>("user");
  let continuationRole = $state<ContinuationRole>("assistant");
  $effect(() => {
    // Legacy renderers only support the ordinary user → assistant path. Keep
    // append-only available, but collapse unsupported role choices when a
    // model does not expose the scene stitcher.
    if (!sceneMode) {
      if (authoredRole !== "user") authoredRole = "user";
      if (continuationRole === "user") continuationRole = "assistant";
    }
  });
  const generatedRole = $derived<ChatRole | null>(
    continuationRole === "none" ? null : continuationRole,
  );
  const defaultUserLabel = $derived(
    sessionState.info?.default_user_role || "user",
  );
  const defaultAssistantLabel = $derived(
    sessionState.info?.default_assistant_role || "assistant",
  );
  const userLabel = $derived(
    samplingState.user_role.trim()
      || defaultUserLabel,
  );
  const assistantLabel = $derived(
    samplingState.assistant_role.trim()
      || defaultAssistantLabel,
  );
  const authoredLabel = $derived(authoredRole === "user" ? userLabel : assistantLabel);
  const authoredRoleValue = $derived(
    authoredRole === "user" ? samplingState.user_role : samplingState.assistant_role,
  );
  const continuationRoleValue = $derived(
    continuationRole === "user" ? samplingState.user_role : samplingState.assistant_role,
  );
  const userRoleSupported = $derived(
    sessionState.info?.is_base_model === false
      && sessionState.info?.user_role_supported === true,
  );
  const assistantRoleSupported = $derived(
    sessionState.info?.is_base_model === false
      && sessionState.info?.role_substitution_supported === true,
  );
  const customRoleLabels = $derived.by(() => {
    const labels = new Set(
      Object.keys(castState.roster).filter((key) => (
        key !== "user"
        && key !== "assistant"
        && key !== defaultUserLabel
        && key !== defaultAssistantLabel
      )),
    );
    if (userLabel !== defaultUserLabel && userLabel !== defaultAssistantLabel) {
      labels.add(userLabel);
    }
    if (
      assistantLabel !== defaultUserLabel
      && assistantLabel !== defaultAssistantLabel
    ) {
      labels.add(assistantLabel);
    }
    return [...labels].sort((a, b) => a.localeCompare(b));
  });

  const roleLabelOptions = $derived(
    [...new Set([defaultUserLabel, defaultAssistantLabel, ...customRoleLabels])]
      .map((value) => ({ value, label: value })),
  );

  function roleLabelValid(value: string): boolean {
    const label = value.trim();
    return label === "" || ROLE_SLUG_RE.test(label);
  }

  const authoredRoleValid = $derived(roleLabelValid(authoredRoleValue));
  const continuationRoleValid = $derived(
    continuationRole === "none" || roleLabelValid(continuationRoleValue),
  );
  const rolesValid = $derived(authoredRoleValid && continuationRoleValid);
  const rolePlanSummary = $derived(
    generatedRole === null
      ? `${authoredLabel} only`
      : `${authoredLabel} → ${generatedRole === "user" ? userLabel : assistantLabel}`,
  );

  $effect(() => {
    if (!rolesValid) rolePlanOpen = true;
  });

  async function setRolePlanOpen(open: boolean): Promise<void> {
    if (rolePlanOpen === open) return;
    const shell = rolePlanShell;
    const startHeight = shell?.getBoundingClientRect().height ?? 0;
    const restoreFocus = Boolean(shell?.contains(document.activeElement));
    rolePlanAnimation?.cancel();
    rolePlanAnimation = null;
    shell?.style.removeProperty("overflow");
    rolePlanOpen = open;
    await tick();
    if (!shell) return;
    if (restoreFocus) {
      shell.querySelector<HTMLElement>(open ? ".plan-minimize" : ".turn-plan-summary")
        ?.focus({ preventScroll: true });
    }
    const duration = reducedMotion() ? 0 : GLIDE;
    if (duration === 0) return;
    const endHeight = shell.getBoundingClientRect().height;
    const easing = getComputedStyle(shell).getPropertyValue("--ease").trim();
    shell.style.overflow = "hidden";
    const animation = shell.animate(
      [{ height: `${startHeight}px` }, { height: `${endHeight}px` }],
      { duration, easing },
    );
    rolePlanAnimation = animation;
    const finish = () => {
      if (rolePlanAnimation !== animation) return;
      rolePlanAnimation = null;
      shell.style.removeProperty("overflow");
    };
    animation.onfinish = finish;
    animation.oncancel = finish;
  }

  function roleSupportsCustomLabel(role: ChatRole): boolean {
    return role === "user" ? userRoleSupported : assistantRoleSupported;
  }
  const continuationRoleEditable = $derived(
    continuationRole !== "none" && roleSupportsCustomLabel(continuationRole),
  );

  function applyRoleChoice(role: ChatRole, label: string): void {
    if (role === "user") samplingState.user_role = label;
    else samplingState.assistant_role = label;
  }

  function selectAuthoredRole(label: string): void {
    applyRoleChoice(authoredRole, label);
  }

  function selectContinuationRole(label: string): void {
    if (continuationRole === "none") return;
    applyRoleChoice(continuationRole, label);
  }

  function setReplyEnabled(enabled: boolean): void {
    if (!enabled) {
      continuationRole = "none";
      return;
    }
    continuationRole = sceneMode && authoredRole === "assistant" ? "user" : "assistant";
  }
  const canSwapPlan = $derived(
    sceneMode && generatedRole !== null && generatedRole !== authoredRole,
  );

  function swapPlanRoles(): void {
    if (!canSwapPlan || generatedRole === null) return;
    const previousAuthored = authoredRole;
    authoredRole = generatedRole;
    continuationRole = previousAuthored;
  }

  // Authored-thinking input: a block the next authored line carries,
  // rendered through the family think delimiters.  Strip families keep
  // it for one turn only — the warning under the box says so before
  // submit (a9 convention 3).
  const thinkingInputSupported = $derived(
    sessionState.info?.thinking_input_supported ?? false,
  );
  const stripsHistoryThinking = $derived(
    sessionState.info?.strips_history_thinking ?? false,
  );
  let thinkingOpen = $state(false);
  let thinkingDraft = $state("");

  onMount(() => registerComposerController({
    read: () => ({ text: input, authoredRole, generatedRole, authoredThinking: thinkingDraft }),
    update: async (change) => {
      if (change.text !== undefined) input = change.text;
      if (change.authoredRole !== undefined) authoredRole = change.authoredRole;
      if (change.generatedRole !== undefined) continuationRole = change.generatedRole ?? "none";
      if (change.authoredThinking !== undefined) {
        thinkingDraft = change.authoredThinking;
        thinkingOpen = thinkingDraft.length > 0;
      }
      await tick();
      autosize();
    },
  }));

  const hasText = $derived(input.trim() !== "");
  const appendSelected = $derived(generatedRole === null);
  const primaryDisabled = $derived(
    !loomTree.loaded || !rolesValid || (!hasText && generatedRole === null),
  );

  const inputPlaceholder = $derived(
    authoredRole === "user" && authoredLabel === defaultUserLabel ? `Message ${modelShort}` : `Write as ${authoredLabel}…`,
  );
  // One short label, so the compose row stays one line; the tip says what an empty send does.
  const sendLabel = $derived(appendSelected ? "Add" : "Send");

  function doSend(): void {
    const text = hasText ? input : "";
    const replaceSlot = consumePulledSlot();
    if (!text) {
      if (replaceSlot !== null) {
        cancelPendingAction(pendingActions.queue[replaceSlot]?.id ?? "");
        return;
      }
      if (generatedRole === null) return;
    }
    const parent = isPendingBusy()
      ? ("active@drain" as const)
      : activeNodeId;
    const thinking = text && thinkingDraft.trim() !== "" ? thinkingDraft : null;
    if (thinking !== null) {
      thinkingDraft = "";
      thinkingOpen = false;
    }
    if (text) {
      pushInputHistory(text);
      input = "";
    }
    void sendSubmit(
      text || null,
      text ? authoredRole : null,
      generatedRole,
      {
        parent_node_id: parent,
        replaceSlot,
        raw: rawMode,
        authored_thinking: thinking,
      },
    );
    scrolledUp = false;
    queueScrollToBottom(true);
    if (window.matchMedia("(pointer: coarse)").matches) {
      const focused = document.activeElement;
      if (focused instanceof HTMLElement && focused.closest("#conversation-composer")) {
        focused.blur();
      }
    }
    queueMicrotask(autosize);
  }

  /** Edge-only multi-line policy: ↑ recalls history when the cursor
   *  sits on the first line of the draft; ↓ goes forward only on the
   *  last line.  In-between lines fall through to the textarea's
   *  native cursor nav so multi-line editing isn't hijacked. */
  function shouldRecallUp(ta: HTMLTextAreaElement): boolean {
    const value = ta.value;
    const cursor = ta.selectionStart ?? 0;
    const firstNL = value.indexOf("\n");
    return firstNL === -1 || cursor <= firstNL;
  }

  function shouldRecallDown(ta: HTMLTextAreaElement): boolean {
    const value = ta.value;
    const cursorEnd = ta.selectionEnd ?? value.length;
    const lastNL = value.lastIndexOf("\n");
    return lastNL === -1 || cursorEnd > lastNL;
  }

  function applyRecalled(text: string): void {
    input = text;
    // Defer cursor placement past the bind:value flush so the textarea
    // reflects the new value before we set the selection.
    queueMicrotask(() => {
      const el = textareaRef;
      if (el) {
        el.setSelectionRange(el.value.length, el.value.length);
        autosize();
      }
    });
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Enter") {
      // Shift-Enter is a newline; Enter executes the visible plan.  Append
      // is selected explicitly by choosing no continuation.
      if (ev.shiftKey) return;
      ev.preventDefault();
      doSend();
      return;
    }
    if (ev.key === "Escape") {
      // Esc is context-sensitive:
      //   1. Gen in flight → stop the gen.  Queue keeps its items.
      //   2. No gen, pull in flight → cancel the pull (restore the
      //      stash, leave the queued slot untouched).
      //   3. Otherwise → fall through to default Escape behavior.
      if (genStatus.active) {
        ev.preventDefault();
        sendStop();
        return;
      }
      if (inputHistory.pulledSlot !== null) {
        ev.preventDefault();
        const stash = cancelInputPull();
        if (stash !== null) {
          input = stash;
          queueMicrotask(autosize);
        }
        return;
      }
    }
    if (ev.key === "ArrowUp" || ev.key === "ArrowDown") {
      const ta = textareaRef;
      if (!ta) return;
      const goingUp = ev.key === "ArrowUp";
      if (goingUp ? !shouldRecallUp(ta) : !shouldRecallDown(ta)) return;
      // ↓ at the live slot (no recall in flight, no pending pull) is
      // a no-op — leave the keystroke for the textarea so it can move
      // within an empty last line or trigger the browser's native
      // end-of-input nudge.  Checking only ``index`` misses the
      // pulled-pending case (``pulledSlot`` is what tracks that),
      // which would otherwise strand the user inside an edited
      // pending item with no ↓ exit.
      if (
        !goingUp
        && inputHistory.index === null
        && inputHistory.pulledSlot === null
      ) return;
      const recalled = navigateInputHistory(goingUp ? -1 : +1, input);
      if (recalled === null) return;
      ev.preventDefault();
      applyRecalled(recalled);
    }
  }

  // ------------------------------------------------------------- highlight --

  /** All probe names available to the highlight dropdowns.  Sourced from
   * the live probe-rack — same source the ProbeRack panel uses. */
  const probeNames = $derived([...probeRack.active]);

  function onHighlightChange(value: string): void {
    setHighlightTarget(value === "" ? null : value);
  }

  function onCompareChange(value: string): void {
    setCompareTarget(value === "" ? null : value);
  }

  function onCompareToggle(): void {
    if (!highlightState.compareTwo) {
      const first = compareOptions[0]?.value ?? null;
      if (first === null) return;
      setCompareTarget(first);
    }
    toggleCompareTwo();
  }

  /** Highlight options for one probe.  A rank-1 flat probe (a 2-node concept
   *  axis) and every curved probe stay a single bare-name option — the bare
   *  channel is the pole coordinate / subspace fraction.  A multi-axis flat
   *  probe (the ``personas`` fan, a flat ``emotions``) fans out into one
   *  option per coordinate so a token can be tinted by each PC; axis 0 keeps
   *  the bare-name value (the channel that survives reload) while axis ``i``
   *  uses the ``name[i]`` form that lines up with the ``@when:name[i]`` gate. */
  function axisOptionsFor(name: string): { value: string; label: string }[] {
    const info = probeRack.entries.get(name)?.info;
    const dim = info?.intrinsic_dim ?? 0;
    const flat = info?.family === "geometry" ? info.is_affine : true;
    // Labels strip the namespace prefix (``default/emotions`` → ``emotions``)
    // to match the probe cards; the option value keeps the full registered
    // name so lookups stay unambiguous.
    const display = name.split("/").pop() ?? name;
    if (flat && dim > 1) {
      return Array.from({ length: dim }, (_, i) => ({
        value: i === 0 ? name : `${name}[${i}]`,
        label: `${display}[${i}]`,
      }));
    }
    return [{ value: name, label: display }];
  }

  /** Highlight-target picker options: built-in generation measurements plus
   *  live probe names, fanned out per coordinate axis for multi-axis probes. */
  const samplerEntropyAvailable = $derived(
    chatLog.turns.some((turn) => (
      [...(turn.thinkingTokens ?? []), ...(turn.tokens ?? [])]
        .some((token) => token.samplerEntropy != null && Number.isFinite(token.samplerEntropy))
    )),
  );
  const hasColorableTokens = $derived(
    chatLog.turns.some((turn) => (
      (turn.thinkingTokens?.length ?? 0) > 0 || (turn.tokens?.length ?? 0) > 0
    )),
  );

  const builtInHighlightOptions = $derived.by<{ value: string; label: string }[]>(() => [
    { value: SURPRISE_TARGET, label: "Token surprisal" },
    ...(samplerEntropyAvailable
      ? [{ value: ENTROPY_TARGET, label: "Sampler entropy" }]
      : []),
  ]);

  const highlightOptions = $derived.by<{ value: string; label: string }[]>(
    () => {
      const opts: { value: string; label: string }[] = [
        { value: "", label: "No color" },
        ...builtInHighlightOptions,
      ];
      for (const name of probeNames) opts.push(...axisOptionsFor(name));
      return opts;
    },
  );

  /** Compare-target picker — same shape but filtered so the A and B targets
   *  don't pick the same axis.  Distinct axes of one probe (PC0 vs PC1) are
   *  allowed — that's a useful two-stripe compare. */
  const compareOptions = $derived.by<{ value: string; label: string }[]>(() => {
    const opts = builtInHighlightOptions.filter(
      (option) => option.value !== highlightState.target,
    );
    for (const name of probeNames) {
      for (const opt of axisOptionsFor(name)) {
        if (opt.value !== highlightState.target) opts.push(opt);
      }
    }
    return opts;
  });

  const compareColorAvailable = $derived(
    hasColorableTokens && highlightState.target !== null && compareOptions.length > 0,
  );

  $effect(() => {
    if (!compareColorAvailable) {
      if (highlightState.compareTwo) setCompareTwo(false);
      if (highlightState.compareTarget !== null) setCompareTarget(null);
      return;
    }
    if (
      highlightState.compareTwo &&
      !compareOptions.some((option) => option.value === highlightState.compareTarget)
    ) {
      setCompareTarget(compareOptions[0]?.value ?? null);
    }
  });

  // -------------------------------------------------- conversation actions --
  //
  // clear / transcript / auto-regen used to live on the Topbar;
  // they act on the conversation, so they belong here.  The mutating
  // ones route through ``enqueuePending`` so clicking them mid-gen
  // queues rather than racing the WS.

  const AUTO_REGEN_MODES: { value: AutoRegenMode; label: string }[] = [
    { value: "unsteered", label: "Original behavior" },
    { value: "inverted", label: "Opposite guidance" },
    { value: "reseed", label: "New random seed" },
    { value: "cool", label: "More focused" },
    { value: "hot", label: "More varied" },
    { value: "custom", label: "Custom recipe…" },
  ];

  const generationAvailability = runtimeOperationAvailability("generation");
  const automaticComparisonAvailable = $derived(
    generationAvailability.available &&
    sessionState.info !== null &&
    sessionState.info.is_base_model !== true &&
    !rawMode,
  );

  $effect(() => {
    if (!automaticComparisonAvailable && autoRegenState.enabled) {
      disableAutoRegen();
    }
  });

  function regenMessage(turn: ChatTurn): void {
    const nodeId = turn.nodeId;
    if (!nodeId) return;
    if (isPendingBusy()) {
      enqueuePending({
        label: "regen",
        text: null,
        apply: () => void loomRegenerateNode(nodeId, 1),
        awaitsGen: true,
        rebuild: null,
      });
    } else {
      void loomRegenerateNode(nodeId, 1);
    }
  }

  // Save / load act on the whole conversation tree; they live here at
  // the chat's edge rather than buried in a rail menu.  Regenerate-N and
  // fan-out used to sit here too — both were redundant (the loom right-
  // click menu carries "regenerate…" and "fan out…", and the experiment
  // lab is one click away in the analysis menu) so they were removed.
  function onAutoRegenModeChange(v: AutoRegenMode): void {
    setAutoRegenMode(v);
  }

  // ------------------------------------------------------------- A/B split --

  /** Do not open an empty comparison pane merely because the setting is on.
   * It appears when a shadow has started or a completed comparison exists. */
  let manualComparisonVisible = $state(false);
  const nextSteering = $derived(currentSteeringExpression());
  const autoRegenActive = $derived(
    (autoRegenState.enabled || manualComparisonVisible) &&
    (abState.processingAb || chatLog.turns.some((turn) => turn.abPair !== undefined)),
  );

  function compareWithoutSteering(turnIdx: number): void {
    if (genStatus.active) return;
    manualComparisonVisible = true;
    unpinComparison();
    void sendShadowGenerate(turnIdx, "unsteered");
  }

  function hideComparison(): void {
    manualComparisonVisible = false;
    disableAutoRegen();
  }

  /** A manually pinned Loom branch takes precedence over the automatic
   * comparison while it remains in the authoritative tree. */
  const pinnedActive = $derived(
    pinnedComparison.nodeId !== null &&
    loomTree.nodes.has(pinnedComparison.nodeId),
  );

  /** Render the conversation up to (and including) the pinned node by
   *  walking parent pointers from the pinned id back to root.  Skips
   *  the synthetic root.  Used by the right column when pinned. */
  const pinnedPath = $derived.by<ChatTurn[]>(() => {
    if (!pinnedActive || !pinnedComparison.nodeId) return [];
    const out: ChatTurn[] = [];
    let cursor: string | null = pinnedComparison.nodeId;
    const seen = new Set<string>();
    while (cursor && !seen.has(cursor)) {
      seen.add(cursor);
      const node = loomTree.nodes.get(cursor);
      if (!node) break;
      // Skip the synthetic root.
      if (!(node.parent_id === null && node.role === "system" && !node.text)) {
        out.push({
          role: node.role,
          text: node.text ?? "",
          roleLabel: node.role_label,
          nodeId: node.id,
          generated: node.recipe !== null,
          appliedSteering: node.applied_steering ?? null,
          aggregateReadings: node.aggregate_readings ?? undefined,
          finishReason: node.finish_reason ?? undefined,
        });
      }
      cursor = node.parent_id;
    }
    return out.reverse();
  });

  /** The right column is visible when EITHER auto-regen is on (which
   *  subsumes the v1.x A/B toggle) or a node is pinned for comparison. */
  const twoColumns = $derived(pinnedActive || autoRegenActive);

  // ----------------------------------------------------------- per-turn UI --

  /** Per-turn thinking-collapsed state.  Keyed by turn index so a re-
   * render of the chat log preserves user-explicit collapse choices.  We
   * default to "collapsed" on creation and auto-expand when the first
   * thinking token lands; on done the turn collapses again unless the
   * user manually expanded.
   *
   * SvelteMap (not plain Map) — Svelte 5's $state doesn't track plain
   * Map mutations, so a bare Map.set wouldn't re-render the toggle UI. */
  const collapsedThinking: SvelteMap<number, boolean> = $state(new SvelteMap());

  function turnCollapsed(turnIdx: number, turn: ChatTurn): boolean {
    const explicit = collapsedThinking.get(turnIdx);
    if (explicit !== undefined) return explicit;
    // Default: expanded while in-flight (so the user can watch it
    // generate), collapsed once the gen lands.
    const inFlight =
      chatLog.pendingIndex === turnIdx && (turn.thinkingTokens?.length ?? 0) > 0;
    return !inFlight;
  }

  function toggleThinking(turnIdx: number): void {
    const cur = collapsedThinking.get(turnIdx) ?? true;
    collapsedThinking.set(turnIdx, !cur);
  }

  // Keep the streamed transcript outside a live region: announcing every
  // token makes generation unusable with a screen reader. This stable,
  // atomic status reports only lifecycle transitions and completed
  // conversation updates.
  let streamAnnouncement = $state("");
  let announcementInitialized = false;
  let previousGenerationActive = false;
  let previousTurnCount = 0;
  $effect(() => {
    const active = genStatus.active;
    const finishReason = genStatus.finishReason;
    const pendingIndex = chatLog.pendingIndex;
    const turnCount = chatLog.turns.length;

    untrack(() => {
      if (!announcementInitialized) {
        announcementInitialized = true;
      } else if (active && !previousGenerationActive) {
        streamAnnouncement = rawMode ? "Updating completion." : "Generating response.";
      } else if (!active && previousGenerationActive) {
        streamAnnouncement = finishReason === "cancelled"
          ? "Generation stopped."
          : finishReason === null
            ? "Generation ended."
            : rawMode
              ? genStatus.tokensSoFar === 0 && chatLog.turns.at(-1)?.generated === false
                ? "Edit saved."
                : finishReason === "length" ? "Token limit reached." : "Completion finished."
              : "Response complete.";
      } else if (
        !active
        && pendingIndex === null
        && turnCount !== previousTurnCount
      ) {
        streamAnnouncement = rawMode
          ? "Completion updated."
          : `Conversation updated. ${turnCount} ${turnCount === 1 ? "message" : "messages"}.`;
      }

      previousGenerationActive = active;
      previousTurnCount = turnCount;
    });
  });

  // Cross-component input restore: when a queue drain pops the slot
  // the user was currently editing, ``drainNextPendingAction`` parks
  // the stash on ``inputRestore`` and bumps ``rev``.  This $effect
  // copies it back into the textarea on the next tick.
  let _restoreRev = $state(0);
  $effect(() => {
    if (inputRestore.rev !== _restoreRev) {
      _restoreRev = inputRestore.rev;
      input = inputRestore.text;
      queueMicrotask(autosize);
    }
  });

  // ------------------------------------------------------ scroll bookkeeping --

  let logRef: HTMLDivElement | null = $state(null);
  let logContentRef: HTMLDivElement | null = $state(null);
  /** True iff the user has manually scrolled up — freezes auto-scroll
   * until they hit the bottom again. */
  let scrolledUp = $state(false);
  let previousScrollTop = 0;
  let scrollingPointer = false;
  let scrollTouchY = 0;
  let forceScroll = false;

  function pauseAutoScroll(): void {
    scrolledUp = true;
    forceScroll = false;
  }

  function onScroll(ev: Event): void {
    const el = ev.currentTarget as HTMLElement;
    // 8px slop so a scrollbar that doesn't quite hit the floor still
    // counts as "at bottom".
    const atBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < 8;
    if (atBottom && el.scrollTop > previousScrollTop) scrolledUp = false;
    else if (scrollingPointer && el.scrollTop < previousScrollTop - 1) {
      pauseAutoScroll();
    }
    previousScrollTop = el.scrollTop;
    if (!scrolledUp) queueScrollToBottom();
  }

  function scrollToBottom(): void {
    const el = logRef;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    previousScrollTop = el.scrollTop;
  }

  let _scrollFrame = 0;
  function queueScrollToBottom(force = false): void {
    forceScroll ||= force;
    if (_scrollFrame) return;
    _scrollFrame = requestAnimationFrame(() => {
      _scrollFrame = 0;
      if (forceScroll || !scrolledUp) scrollToBottom();
      forceScroll = false;
    });
  }

  // Auto-scroll on new turns or token deltas.  Reads (not writes) the
  // length-aggregates that drive the chat — Svelte 5 tracks these via
  // the runes graph.
  $effect(() => {
    // Touch the things we care about so the effect re-runs on changes.
    void chatLog.turns.length;
    const lastTurn = chatLog.turns[chatLog.turns.length - 1];
    void lastTurn?.tokens?.length;
    void lastTurn?.thinkingTokens?.length;
    void lastTurn?.text;
    untrack(() => queueScrollToBottom());
  });

  $effect(() => {
    if (!logRef || !logContentRef) return;
    const observer = new ResizeObserver(() => queueScrollToBottom());
    observer.observe(logRef);
    observer.observe(logContentRef);
    const releasePointer = () => { scrollingPointer = false; };
    window.addEventListener("pointerup", releasePointer);
    window.addEventListener("pointercancel", releasePointer);
    queueScrollToBottom(true);
    return () => {
      observer.disconnect();
      window.removeEventListener("pointerup", releasePointer);
      window.removeEventListener("pointercancel", releasePointer);
    };
  });

  onMount(() => {
    autosize();
    scrollToBottom();
    if (!window.matchMedia("(pointer: coarse)").matches) textareaRef?.focus();
    updateComposerBounds();
    let resizeFrame = 0;
    let releaseComposerFrame = 0;
    const holdComposer = (event: PointerEvent) => {
      if (event.button !== 0 || !textareaRef?.closest("form")?.contains(document.activeElement)) return;
      cancelAnimationFrame(releaseComposerFrame);
      composerPointerHeld = true;
    };
    const releaseComposer = () => {
      if (!composerPointerHeld) return;
      cancelAnimationFrame(releaseComposerFrame);
      releaseComposerFrame = requestAnimationFrame(() => { composerPointerHeld = false; });
    };
    document.addEventListener("pointerdown", holdComposer, true);
    window.addEventListener("pointerup", releaseComposer);
    window.addEventListener("pointercancel", releaseComposer);
    window.addEventListener("blur", releaseComposer);
    const scheduleComposerBounds = () => {
      if (resizeFrame) return;
      resizeFrame = requestAnimationFrame(() => {
        resizeFrame = 0;
        updateComposerBounds();
      });
    };
    const observer = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(scheduleComposerBounds);
    if (chatRef) observer?.observe(chatRef);
    window.visualViewport?.addEventListener("resize", scheduleComposerBounds);
    return () => {
      observer?.disconnect();
      cancelAnimationFrame(resizeFrame);
      cancelAnimationFrame(releaseComposerFrame);
      document.removeEventListener("pointerdown", holdComposer, true);
      window.removeEventListener("pointerup", releaseComposer);
      window.removeEventListener("pointercancel", releaseComposer);
      window.removeEventListener("blur", releaseComposer);
      cancelAnimationFrame(_scrollFrame);
      _scrollFrame = 0;
      window.visualViewport?.removeEventListener("resize", scheduleComposerBounds);
      rolePlanAnimation?.cancel();
    };
  });

  // ----------------------------------------------------------- token render --

  /** Drop whitespace-only tokens from the head of the response so the gap below ``</think>``
   * goes away in plain-text mode too.  Returns the surviving slice
   * starting at the first non-whitespace token. */
  const visibleResponseTokens = createTokenViewCache<TokenScore>();

  const tokenFocusIndices = new SvelteMap<string, number>();

  function tokenClicked(
    turnIdx: number,
    tokenIdx: number,
    ev: MouseEvent | KeyboardEvent,
    isThinking: boolean = false,
  ): void {
    ev.stopPropagation();
    const turn = chatLog.turns[turnIdx];
    const token = (isThinking ? turn?.thinkingTokens : turn?.tokens)?.[tokenIdx];
    if (!token) return;
    openDrawer("token_drilldown", { turnIdx, tokenIdx, isThinking });
  }

  function tokenKeydown(ev: KeyboardEvent, turnIdx: number, tokenIdx: number, isThinking = false): void {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      tokenClicked(turnIdx, tokenIdx, ev, isThinking);
      return;
    }
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(ev.key)) return;
    const target = ev.currentTarget as HTMLElement;
    const tokens = [...target.parentElement!.querySelectorAll<HTMLElement>(".tok")];
    const current = tokens.indexOf(target);
    const next = ev.key === "Home" ? 0 : ev.key === "End" ? tokens.length - 1
      : (current + (ev.key === "ArrowRight" ? 1 : -1) + tokens.length) % tokens.length;
    ev.preventDefault();
    tokens[next].focus();
  }

  /** One keyboard stop per message opens the drawer at a real token; the
   * drawer's arrow-key controls then traverse every token and segment. */
  function inspectTurnTokens(turn: ChatTurn, turnIdx: number): void {
    const firstResponse = visibleResponseTokens(turn.tokens ?? [])[0];
    if (firstResponse) {
      openDrawer("token_drilldown", {
        turnIdx,
        tokenIdx: firstResponse.originalIdx,
      });
      return;
    }
    if ((turn.thinkingTokens?.length ?? 0) > 0) {
      openDrawer("token_drilldown", {
        turnIdx,
        tokenIdx: 0,
        isThinking: true,
      });
      return;
    }
    if ((turn.tokens?.length ?? 0) > 0) {
      openDrawer("token_drilldown", { turnIdx, tokenIdx: 0 });
    }
  }

  // The bare-text form for plain (no-highlight) rendering still strips
  // leading whitespace when no probe is selected.
  function plainResponseText(turn: ChatTurn): string {
    if (!turn.tokens || turn.tokens.length === 0) {
      // Fall back to the accumulated text if the per-token list is
      // unset (extremely early in a stream, or for non-streamed loads).
      return (turn.text ?? "").replace(/^\s+/, "");
    }
    return visibleResponseTokens(turn.tokens)
      .map(({ tok }) => tok.text)
      .join("");
  }

  onInterfaceMount(() => registerInterfaceController("chat", {
    schema: chatViewSchema,
    read: () => ({ values: { comparison_visible: pinnedActive || autoRegenActive }, comparison_available: chatLog.turns.some(turn => turn.abPair !== undefined) }),
    update: (change) => {
      if (change.comparison_visible === true && !pinnedActive && !chatLog.turns.some(turn => turn.abPair !== undefined)) throw new ToolError("NOT_FOUND", "Generate or pin a comparison before showing it.");
      if (change.comparison_visible === false) { hideComparison(); unpinComparison(); }
      if (change.comparison_visible === true) manualComparisonVisible = true;
    },
  }));
</script>

<!-- What the words are coloured by: under the composer in a chat, under the completion's head in
     raw completion (RawBuffer's Colors). -->
{#snippet readingTools()}
  <div class="reading" id="chat-reading-tools" transition:expand>
    <label class="ctl">
      <span class="ctl-label" data-tip="Color words by token surprisal, sampler entropy, or an attached reading. Select a word for its exact value.">Color words by</span>
      <Select value={highlightState.target ?? ""} options={highlightOptions} onchange={onHighlightChange}
        ariaLabel={rawMode ? "Color completion tokens by" : "Color generated words by"} />
    </label>
    {#if compareColorAvailable}
      <button type="button" class="dw-text" aria-pressed={highlightState.compareTwo} onclick={onCompareToggle}>
        {highlightState.compareTwo ? "One color" : "Compare two colors"}
      </button>
    {/if}
    {#if highlightState.compareTwo}
      <label class="ctl">
        <span class="ctl-label">With</span>
        <Select value={highlightState.compareTarget ?? ""} options={compareOptions} onchange={onCompareChange}
          disabled={!highlightState.compareTwo} ariaLabel="Second word color" />
      </label>
    {/if}
    {#if automaticComparisonAvailable}
      <Switch checked={autoRegenState.enabled} onchange={toggleAutoRegen} label="Compare replies"
        hint="Create a second version after every reply using the comparison settings." />
      {#if autoRegenState.enabled}
        <Select value={autoRegenState.mode} options={AUTO_REGEN_MODES} onchange={onAutoRegenModeChange}
          disabled={abState.processingAb} ariaLabel="Automatic comparison style" />
        {#if autoRegenState.mode === "custom"}
          {@const customProblem = customComparisonProblem(autoRegenState.custom)}
          <input type="text" class="ctl-input" value={autoRegenState.custom} disabled={abState.processingAb}
            oninput={(ev) => setAutoRegenCustom((ev.currentTarget as HTMLInputElement).value)}
            placeholder="seed=42, temperature=1.5" aria-label="Custom automatic comparison sampling"
            aria-invalid={customProblem !== null} aria-describedby={customProblem ? "custom-comparison-problem" : undefined} />
          {#if customProblem}<span id="custom-comparison-problem" class="ctl-label" role="alert">{customProblem}</span>{/if}
        {/if}
      {/if}
    {/if}
  </div>
{/snippet}

<div class="chat-panel">
<div class="chat" aria-label={rawMode ? "Text completion" : "Chat"} bind:this={chatRef}>
  <div class="sr-only" role="status" aria-live="polite" aria-atomic="true">{streamAnnouncement}</div>

  {#if rawMode}
    <div class="raw-host"><RawBuffer colors={readingTools} bind:colorsOpen={readingOpen} /></div>
  {:else}
  <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions (The scrollable conversation supports keyboard scrolling.) -->
  <div
    class="log"
    class:ab={twoColumns}
    bind:this={logRef}
    onscroll={onScroll}
    onwheel={(event) => { if (event.deltaY < 0) pauseAutoScroll(); }}
    onpointerdown={(event) => { scrollingPointer = event.pointerType === "mouse"; }}
    ontouchstart={(event) => { scrollTouchY = event.touches[0].clientY; }}
    ontouchmove={(event) => {
      const y = event.touches[0].clientY;
      if (y > scrollTouchY) pauseAutoScroll();
      scrollTouchY = y;
    }}
    onkeydown={(event) => {
      if (["ArrowUp", "PageUp", "Home"].includes(event.key)) pauseAutoScroll();
      if (event.target === logRef && event.key === "Home") {
        event.preventDefault();
        event.currentTarget.scrollTop = 0;
      } else if (event.target === logRef && event.key === "End") {
        event.preventDefault();
        scrolledUp = false;
        queueScrollToBottom(true);
      }
    }}
    tabindex="0"
    role="region"
    aria-label="Conversation"
  >
    <div class="log-content" bind:this={logContentRef}>
    {#if chatLog.turns.length === 0}
      <div class="empty" in:rise out:fadeOut>
        <span class="dw-mark">{modelShort}</span>
        <p class="meta">{sessionState.info?.is_base_model ? "Write the opening text below" : "Write the first message below"}</p>
      </div>
    {:else if twoColumns && !pinnedActive}
      <!-- An automatic comparison, one row a turn so each comparison sits beside the reply it
           compares: a prompt spans both columns, a reply pairs with its comparison. Comparisons
           are stateless and never alter the loom. -->
      <div class="ab-grid ab-rows">
        <span aria-hidden="true"></span>
        <div class="pin-head">
          <span class="head">Comparison</span>
          <button type="button" class="dw-text" onclick={hideComparison}>Hide</button>
        </div>
        {#each chatLog.turns as turn, turnIdx (turnIdx)}
          {#if !turn.generated || turn.role === "system"}
            <div class="ab-span">{@render bubble(turn, turnIdx, false)}</div>
          {:else}
            <div class="ab-cell">{@render bubble(turn, turnIdx, false)}</div>
            <div class="ab-cell">
              {#if turn.abPair}
                {@render bubble(turn.abPair, turnIdx, true)}
              {:else}
                <div class="dw-turn" aria-hidden="true">
                  <span class="dw-seat">{seatLabel(turn)}</span>
                  <span class="meta">{abState.pendingTurnIdx === turnIdx ? "Writing the comparison…" : "Not compared"}</span>
                </div>
              {/if}
            </div>
          {/if}
        {/each}
      </div>
    {:else if twoColumns}
      <!-- Two columns: the right is a pinned branch path, which runs its own length. -->
      <div class="ab-grid">
        <div class="ab-col">
          {#each chatLog.turns as turn, turnIdx (turnIdx)}
            {@render bubble(turn, turnIdx, false)}
          {/each}
        </div>
        <div class="ab-col">
          <div class="pin-head">
            <span class="head">Pinned branch</span>
            <button type="button" class="dw-text" onclick={unpinComparison}>Unpin</button>
          </div>
          {#each pinnedPath as turn, idx (idx)}
            {@render bubble(turn, idx, true)}
          {/each}
        </div>
      </div>
    {:else}
      {#each chatLog.turns as turn, turnIdx (turnIdx)}
        {@render bubble(turn, turnIdx, false)}
      {/each}
    {/if}
    </div>
  </div>

  <div class="dock">
    <PendingBubbles />
    {#if readingOpen}{@render readingTools()}{/if}

    <StatusFooter />
    <RecipeBar />

    <div class="turn-plan-shell" bind:this={rolePlanShell}>
      {#if rolePlanOpen}
        <div id="turn-role-controls" class="turn-plan" aria-label="Next turn" in:rise>
          <label class="plan-card">
            <span class="head">You write as</span>
            <Combobox value={authoredRoleValue} options={roleLabelOptions} onchange={selectAuthoredRole}
              disabled={!roleSupportsCustomLabel(authoredRole)} invalid={!authoredRoleValid}
              placeholder={authoredRole === "user" ? defaultUserLabel : defaultAssistantLabel}
              ariaLabel="You write as" ariaDescribedby={!authoredRoleValid ? "role-label-error" : undefined} />
          </label>
          <label class="plan-card">
            <span class="head">The model writes as</span>
            <Combobox value={continuationRole === "none" ? "" : continuationRoleValue} options={roleLabelOptions}
              onchange={selectContinuationRole} disabled={!continuationRoleEditable} invalid={!continuationRoleValid}
              placeholder={continuationRole === "user" ? defaultUserLabel : defaultAssistantLabel}
              ariaLabel="Model writes as" ariaDescribedby={!continuationRoleValid ? "role-label-error" : undefined} />
          </label>
          <div class="dw-row plan-actions">
            <button type="button" class="dw-text" disabled={!canSwapPlan} onclick={swapPlanRoles}>Swap</button>
            <Switch checked={generatedRole !== null} onchange={setReplyEnabled} label="Reply" hint="Generate a model reply" />
            <button type="button" class="dw-text" disabled={!rolesValid} onclick={() => openDrawer("cast")}>Role settings</button>
            <button type="button" class="dw-text plan-minimize" aria-expanded="true" aria-controls="turn-role-controls"
              onclick={() => setRolePlanOpen(false)}>Done</button>
          </div>
        </div>
      {/if}
    </div>

    {#if !rolesValid}
      <p id="role-label-error" class="meta role-error" role="alert">
        Use lowercase letters, numbers, periods, underscores, or hyphens for role names.
      </p>
    {/if}

    {#if thinkingInputSupported && thinkingOpen}
      <div class="thinking-box" transition:expand>
        <textarea class="thinking-input" bind:value={thinkingDraft} dir="auto" rows="2" spellcheck="false"
          placeholder="Reasoning for this line" aria-label="Reasoning for the next authored line"></textarea>
        {#if stripsHistoryThinking}<p class="meta">This model uses it for one turn only.</p>{/if}
      </div>
    {/if}

    <form
      id="conversation-composer"
      class="dw-compose"
      class:pointer-held={composerPointerHeld}
      onsubmit={(ev) => { ev.preventDefault(); doSend(); }}
    >
      <textarea
        class="input"
        bind:this={textareaRef}
        bind:value={input}
        dir="auto"
        onkeydown={onKeydown}
        placeholder={inputPlaceholder}
        rows={phone ? 1 : 2}
        aria-label={`Compose as ${authoredLabel}`}
      ></textarea>
      <div class="dw-compose-row">
        <button type="button" class="dw-sampling" onclick={openSampling} data-tip="Sampling">{samplingSummary}</button>
        {#if phone}
          <button type="button" class="dw-text small more" aria-expanded={extrasOpen} aria-controls="composer-extras"
            onclick={() => (extrasOpen = !extrasOpen)}>More</button>
        {/if}
        <span class="extras" id="composer-extras" class:folded={phone && !extrasOpen}>
        <button type="button" class="dw-text small turn-plan-summary" aria-expanded={rolePlanOpen} aria-controls="turn-role-controls"
          onclick={() => setRolePlanOpen(!rolePlanOpen)} data-tip="Who writes the next turn">{rolePlanSummary}</button>
        {#if thinkingInputSupported}
          <button type="button" class="dw-text small" aria-pressed={thinkingOpen || thinkingDraft.trim() !== ""}
            onclick={() => (thinkingOpen = !thinkingOpen)}>Reasoning</button>
        {/if}
        <button type="button" class="dw-text small" aria-expanded={readingOpen} aria-controls="chat-reading-tools"
          onclick={() => (readingOpen = !readingOpen)}>Colors</button>
        {#if canClearConversation}
          <button type="button" class="dw-text small clear-conversation" class:armed={clearConversationArmed}
            onmousedown={(event) => {
              if (event.button !== 0) return;
              event.preventDefault();
              event.currentTarget.focus({ preventScroll: true });
            }}
            onclick={requestClearConversation}
            onblur={() => (clearConversationArmed = false)}
            onkeydown={(event) => { if (event.key === "Escape") clearConversationArmed = false; }}
            data-tip={clearConversationArmed ? "Start a new path; the branches stay in the loom" : "Start a blank conversation; the branches stay in the loom"}
            aria-label={clearConversationArmed ? "Confirm clear conversation" : "Clear conversation"}
          >{clearConversationArmed ? "Confirm clear" : "Clear"}</button>
        {/if}
        </span>
        <span class="send">
          {#if genStatus.active || stopSlotHeld}
            <button type="button" class="dw-quiet stop" class:held={!genStatus.active} inert={!genStatus.active}
              aria-hidden={!genStatus.active} onclick={sendStop} data-tip="Escape stops the reply">Stop</button>
          {/if}
          <button type="submit" class="dw-button" disabled={primaryDisabled}
            data-tip={appendSelected ? "Enter adds your message only" : hasText ? "Enter sends and generates a reply" : "Generates the next reply"}>
            <MorphText text={sendLabel} numbers={false} />
          </button>
        </span>
      </div>
    </form>
  </div>
  {/if}
</div>

</div>

{#snippet bubble(turn: ChatTurn, turnIdx: number, isShadow: boolean)}
  {#if turn.role === "system"}
    <!-- A stage direction: a note about the scene, not a speaker. -->
    <div class="dw-turn stage" class:shadow={isShadow} in:rise>
      <span class="dw-seat">System</span>
      <p class="stage-text" dir="auto">{turn.text}</p>
    </div>
  {:else}
  {@const generating = genStatus.active && (isShadow ? abState.pendingTurnIdx === turnIdx : !abState.processingAb && chatLog.pendingIndex === turnIdx)}
  {@const user = turn.role === "user"}
  <div class="dw-turn msg" class:user class:shadow={isShadow} class:historical={turnIdx < chatLog.turns.length - 8} in:rise>
    <span class="dw-seat" class:end={user}>{seatLabel(turn)}</span>

    {#if (turn.thinkingTokens?.length ?? 0) > 0 || turn.thinking}
      <div class="thinking-block">
        <button type="button" class="dw-text small thinking-toggle" onclick={() => toggleThinking(turnIdx)}
          aria-expanded={!turnCollapsed(turnIdx, turn)}>
          <span class="caret" class:open={!turnCollapsed(turnIdx, turn)}><Icon name="down" /></span>Thinking
        </button>
        {#if !turnCollapsed(turnIdx, turn)}
          <div class="thinking-body" dir="auto" transition:expand>
            {#each turn.thinkingTokens ?? [] as tok, tokenIdx (tokenIdx)}
              {@const tint = highlightStyleString(tok)}
              {@const chosen = isSelectedToken(turnIdx, tokenIdx, true)}
              {@const parts = tokenLead(tok.text)}
              {parts.lead}<span
                class="tok"
                class:sel={chosen}
                use:arrivals.reveal={genStatus.active ? tok : null}
                data-cursor="inspect"
                class:tinted={tint !== ""}
                style={chosen ? "" : tint}
                onpointerenter={() => beginTokenHover(tok, turn.nodeId)}
                onpointerleave={endTokenHover}
                onfocus={() => { tokenFocusIndices.set(`${turn.nodeId ?? turnIdx}:thinking`, tokenIdx); beginTokenHover(tok, turn.nodeId); }}
                onblur={endTokenHover}
                onclick={(ev) => tokenClicked(turnIdx, tokenIdx, ev, true)}
                onkeydown={(ev) => tokenKeydown(ev, turnIdx, tokenIdx, true)}
                role="button"
                tabindex={tokenIdx === (tokenFocusIndices.get(`${turn.nodeId ?? turnIdx}:thinking`) ?? 0) ? 0 : -1}
                aria-pressed={chosen}
                aria-label={`Inspect token ${tok.text.trim() || "whitespace"}`}
              >{parts.body}</span>
            {/each}
          </div>
        {/if}
      </div>
    {/if}

    <div class={user ? "dw-user" : "dw-said"} dir="auto">
      {#if (turn.tokens?.length ?? 0) > 0}
        {#each visibleResponseTokens(turn.tokens ?? []) as { tok, originalIdx }, visibleIdx (originalIdx)}
          {@const tint = highlightStyleString(tok)}
          {@const chosen = isSelectedToken(turnIdx, originalIdx, false)}
          {@const parts = tokenLead(tok.text)}
          {parts.lead}<span
            class="tok"
            class:sel={chosen}
            use:arrivals.reveal={genStatus.active ? tok : null}
            data-cursor="inspect"
            class:tinted={tint !== ""}
            style={chosen ? "" : tint}
            onpointerenter={() => beginTokenHover(tok, turn.nodeId)}
            onpointerleave={endTokenHover}
            onfocus={() => { tokenFocusIndices.set(`${turn.nodeId ?? turnIdx}:response`, originalIdx); beginTokenHover(tok, turn.nodeId); }}
            onblur={endTokenHover}
            onclick={(ev) => tokenClicked(turnIdx, originalIdx, ev, false)}
            onkeydown={(ev) => tokenKeydown(ev, turnIdx, originalIdx)}
            role="button"
            tabindex={(tokenFocusIndices.has(`${turn.nodeId ?? turnIdx}:response`) ? tokenFocusIndices.get(`${turn.nodeId ?? turnIdx}:response`) === originalIdx : visibleIdx === 0) ? 0 : -1}
            aria-pressed={chosen}
            aria-label={`Inspect token ${tok.text.trim() || "whitespace"}`}
          >{parts.body}</span>
        {/each}
      {:else}
        <span class="plain">{plainResponseText(turn)}</span>
      {/if}
    </div>

    <div class="dw-under" class:end={user}>
      {#if generating}
        <span role="status"><MorphText text={(turn.tokens?.length ?? 0) > 0 ? "Writing" : (turn.thinkingTokens?.length ?? 0) > 0 ? "Thinking" : "Preparing the reply"} numbers={false} /></span>
      {:else if turn.generated && (turn.tokens?.length ?? 0) > 0}
        <span>{turn.tokens?.length} tokens</span>
      {/if}
      {#if !generating && turn.finishReason === "cancelled"}
        <span class="dw-cut" data-tip="A Stop or a closed page ended this reply before the model finished">stopped early</span>
      {/if}
      {#if turn.meanLogprob != null && Number.isFinite(turn.meanLogprob)}
        <span data-tip="Sequence perplexity">perplexity <MorphText text={Math.exp(-turn.meanLogprob).toFixed(1)} /></span>
      {/if}
      {#if turn.generated && turn.finishReason != null && turn.appliedSteering !== undefined}
        <span data-tip={turn.appliedSteering ?? "No steering"}>{turn.appliedSteering ? "steered" : "unsteered"}</span>
      {/if}
      {#if tintKey && turn.generated && (turn.tokens?.length ?? 0) > 0}
        <button type="button" class="dw-tintkey" aria-expanded={readingOpen} aria-controls="chat-reading-tools"
          onclick={() => (readingOpen = !readingOpen)} data-tip="What the words are coloured by"><span class="dw-swatch" aria-hidden="true"></span>{tintKey}</button>
      {/if}
      {#if turn.nodeId}
        <button type="button" class="dw-text small" onclick={() => regenMessage(turn)}
          aria-label={`Reroll ${roleDisplayLabel(turn.role, turn.roleLabel)} message`}>Reroll</button>
      {/if}
      {#if (turn.tokens?.length ?? 0) > 0 || (turn.thinkingTokens?.length ?? 0) > 0}
        <button type="button" class="dw-text small" onclick={() => inspectTurnTokens(turn, turnIdx)}
          aria-label={`Inspect tokens in ${roleDisplayLabel(turn.role, turn.roleLabel)} message`}>Inspect</button>
      {/if}
      {#if turn.generated && !isShadow && turn.appliedSteering && automaticComparisonAvailable}
        <button type="button" class="dw-text small" disabled={genStatus.active} onclick={() => compareWithoutSteering(turnIdx)}
          data-tip="Replay the preceding messages with the saved sampling settings and no steering">Compare unsteered</button>
      {/if}
    </div>
    {#if turn.generated && turn.finishReason != null && Object.keys(turn.aggregateReadings ?? {}).length > 0}
      <div class="readings">
        <Expander label="Readings">
          <dl>
            {#each Object.entries(turn.aggregateReadings ?? {}) as [name, value] (name)}
              {#if Number.isFinite(value)}
                <div><dt>{name}</dt><dd>{aggregateReadingText(name, value)}</dd></div>
              {/if}
            {/each}
          </dl>
        </Expander>
      </div>
    {/if}
  </div>
  {/if}
{/snippet}

<style>
  .chat-panel, .chat { display: flex; flex-direction: column; flex: 1 1 0; min-height: 0; min-width: 0; }
  .log { flex: 1 1 0; min-height: 0; overflow-y: auto; overscroll-behavior: contain; scrollbar-width: thin; scrollbar-gutter: stable; }
  .log:focus-visible { outline-offset: -2px; border-radius: var(--r-box); }
  .log-content { display: grid; align-content: start; gap: var(--s6); min-width: 0; padding: var(--s1) var(--s1) var(--s5); }
  .raw-host { flex: 1 1 0; min-height: 0; display: flex; flex-direction: column; }
  .empty { display: grid; gap: var(--s2); justify-items: start; padding-top: var(--s7); }

  .dw-seat.end { justify-self: end; }
  .dw-under.end { justify-content: flex-end; }
  .msg.historical { content-visibility: auto; contain-intrinsic-size: auto 120px; }
  .msg.user .dw-user { display: block; }
  .dw-said, .dw-user { min-width: 0; }
  .stage-text { color: var(--ink2); font: var(--type-small); font-style: italic; white-space: pre-wrap; overflow-wrap: anywhere; }
  .shadow .dw-said { color: var(--ink2); }
  .plain { white-space: pre-wrap; }

  /* One token: joined to its word, a wash on hover, ink when it is the one selected. The
   * highlight's own fill rides its inline style. The space before a word sits outside the box
   * (lib/tokens.ts::tokenLead), so a token that wraps starts its box on its own line. The wash and
   * the ink grow only up and down (inline padding never moves the line), so a chip never paints
   * over the glyph beside it. */
  .tok { border-radius: 4px; cursor: pointer; transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease); }
  .tok:not(.tinted):hover { background: var(--wash2); padding-block: 2px; }
  .tok.sel, .tok.sel:hover { background: var(--ink); color: var(--paper); padding-block: 2px; }
  .tok:focus-visible { outline-offset: 1px; }

  .thinking-block { display: grid; gap: var(--s2); min-width: 0; }
  .thinking-toggle { display: inline-flex; align-items: center; gap: var(--s1); justify-self: start; }
  .caret { display: grid; rotate: -90deg; transition: rotate var(--glide) var(--ease); }
  .caret.open { rotate: 0deg; }
  .thinking-body { color: var(--ink2); font: var(--type-small); font-style: italic; line-height: 24px; white-space: pre-wrap; overflow-wrap: anywhere; }
  .readings { min-width: 0; }
  .readings dl { display: grid; gap: var(--s1); margin: 0; }
  .readings dl > div { display: flex; justify-content: space-between; gap: var(--s4); color: var(--ink2); font: var(--type-small); }
  .readings dt { overflow-wrap: anywhere; }
  .readings dd { margin: 0; color: var(--ink); font-variant-numeric: tabular-nums; }

  .ab-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s5); min-width: 0; }
  .ab-col { display: grid; align-content: start; gap: var(--s6); min-width: 0; }
  .ab-rows { row-gap: var(--s6); align-items: start; }
  .ab-span { grid-column: 1 / -1; min-width: 0; }
  .ab-cell { min-width: 0; }
  .pin-head { display: flex; align-items: center; justify-content: space-between; gap: var(--s2); }

  .dock { display: grid; gap: var(--s2); flex: none; min-width: 0; padding-top: var(--s2); }
  .reading { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s4); padding: var(--s2) 0; }
  .ctl { display: inline-flex; align-items: center; gap: var(--s2); color: var(--ink2); font: var(--type-control); }
  .ctl-input { height: 36px; padding: 0 var(--s3); border: 0; border-radius: var(--r-pill); background: var(--wash); color: var(--ink); font: var(--type-small); }
  .turn-plan { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s3) var(--s4); padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .plan-card { display: grid; gap: var(--s1); min-width: 0; }
  .plan-actions { grid-column: 1 / -1; gap: var(--s4); }
  .role-error { color: var(--ink); }
  .thinking-box { display: grid; gap: var(--s1); padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); }
  .thinking-input { width: 100%; min-height: 40px; padding: 0; border: 0; outline: 0; resize: vertical; background: transparent; color: var(--ink); font: var(--type-small); font-style: italic; }

  .dw-compose .input { width: 100%; }
  .dw-compose-row { gap: var(--s2) var(--s4); }
  .extras { display: contents; }
  .small { font: var(--type-meta); }
  /* The sampling summary takes the row's free space, so Send keeps the right edge on one line and,
   * when the row wraps, its own auto margin keeps it there instead of flush left. */
  .dw-sampling { min-width: 0; flex: 1 1 auto; }
  .send { display: inline-flex; align-items: center; gap: var(--s2); margin-left: auto; }
  .stop.held { visibility: hidden; }
  .turn-plan-summary { max-width: 24ch; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .clear-conversation.armed { color: var(--ink); }
  .dw-compose .dw-button, .dw-compose .dw-quiet { flex: none; }

  @media (max-width: 720px) {
    .log { padding: var(--s4) var(--s4) 0; }
    .ab-grid { grid-template-columns: minmax(0, 1fr); }
    .dock { padding: var(--s2) var(--s4) var(--s4); }
    .turn-plan { grid-template-columns: minmax(0, 1fr); }
    .dw-compose { gap: var(--s1); padding: var(--s2) var(--s2) var(--s2) var(--s4); }
    .dw-compose .input { min-height: 32px; }
    .dw-compose-row { min-height: 44px; flex-wrap: wrap; gap: 0 var(--s3); }
    .dw-compose-row .dw-sampling { flex: 1 1 0; min-width: 0; min-height: 44px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; margin-right: 0; }
    .more { flex: none; min-height: 44px; }
    .extras { order: 3; display: flex; flex: 1 0 100%; gap: var(--s4); overflow-x: auto; scrollbar-width: none; }
    .extras.folded { display: none; }
    .extras > * { flex: none; min-height: 44px; }
    .extras .turn-plan-summary { max-width: none; }
    .dw-compose .dw-button, .dw-compose .dw-quiet { height: 44px; }
  }
  @media (max-width: 720px) and (hover: none) {
    .tok:not(.tinted):hover { background: transparent; box-shadow: none; }
  }
</style>
