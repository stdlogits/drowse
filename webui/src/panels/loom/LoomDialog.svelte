<script lang="ts">
  // The loom's dialogs: regenerate, edit, branch, delete, clear, note,
  // go-to-turn, search, guided fan-out, and regenerate-with-variation.  The
  // sidebar opens one through ``open`` (bound instance) and gets focus back
  // through ``onclosed``; while a dialog is open the rest of the workspace
  // is inert.
  import { tick } from "svelte";
  import { IconButton, MorphText } from "@stdlogits/ledger";
  import NumberInput from "../../lib/NumberInput.svelte";
  import Select from "../../lib/Select.svelte";
  import { containTab } from "../../lib/focusTrap";
  import { fadeIn, fadeOut, rise } from "@stdlogits/ledger";
  import { apiTree } from "../../lib/runtime/services";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { flushConversationAutosave, savedConversationState } from "../../lib/stores/savedConversations.svelte";
  import {
    applyTreeFilter,
    clearNodeSelection,
    clearTreeFilter,
    filterState,
    genStatus,
    LOOM_DELETE_DURING_GENERATION_MESSAGE,
    loomBranch,
    loomContinueFromCommitted,
    loomDelete,
    loomEdit,
    loomNavigate,
    loomNodeIntersectsGeneration,
    loomNote,
    loomRegenerateActive,
    loomTree,
    pushToast,
    refreshLoomTree,
  } from "../../lib/stores.svelte";
  import { fanoutConcepts as flatConcepts, fanoutRecipe, parseStrengthList, resolveNodeIdPrefix } from "./loomCommands";
  import { steerRack } from "../../lib/stores/steering.svelte";
  import { goto } from "../../lib/stores/rooms.svelte";

  let {
    background = null,
    onsearchmatch,
    onbranchcreated,
    onclosed,
  }: {
    /** The loom panel; it and its workspace siblings go inert while open. */
    background?: HTMLElement | null;
    /** A search matched: reveal the first match. */
    onsearchmatch?: () => Promise<void>;
    onbranchcreated?: (nodeId: string) => void;
    /** The dialog closed; ``trigger`` is the control that opened it. */
    onclosed?: (trigger: HTMLElement | null) => void;
  } = $props();

  interface ModalState {
    kind:
      | null
      | "regenerate"
      | "edit"
      | "branch"
      | "delete"
      | "clear"
      | "note"
      | "navpicker"
      | "search"
      | "fanout"
      | "regen_mode";
    nodeId: string | null;
    text: string;
    n: number;
    /** The fan-out's concept (a fitted flat fit, ``ns/name``) and the node it steers toward: picked,
     *  never typed (people write recipes with the chips, not as text). */
    concept?: string;
    toward?: string;
    /** Mode for the regen-with-modifier modal. */
    mode?: string;
    /** Inline validation error.  Non-empty means commit failed; the
     *  modal stays open with this message rendered below the input. */
    error?: string;
  }
  let modal: ModalState = $state({
    kind: null,
    nodeId: null,
    text: "",
    n: 1,
    concept: "",
    toward: "",
    mode: "unsteered",
    error: "",
  });

  /** The concepts a fan-out sweeps (loomCommands.ts::fanoutConcepts). */
  const fanoutConcepts = $derived(flatConcepts(steerRack.catalog));

  function fanoutNodes(concept: string): string[] {
    return fanoutConcepts.find((row) => `${row.namespace}/${row.name}` === concept)?.node_labels ?? [];
  }

  /** The fan-out's opening pick: the rack's first flat concept chip (toward its node), else the
   *  first fitted concept toward its first node. */
  function fanoutDefault(): { concept: string; toward: string } {
    const names = new Set(fanoutConcepts.map((row) => `${row.namespace}/${row.name}`));
    for (const [name, entry] of steerRack.entries) {
      if (entry.mode === "subspace" && !entry.ablate && names.has(name)) {
        const nodes = fanoutNodes(name);
        return { concept: name, toward: entry.label !== null && nodes.includes(entry.label) ? entry.label : nodes[0] ?? "" };
      }
    }
    const first = fanoutConcepts[0];
    return first ? { concept: `${first.namespace}/${first.name}`, toward: first.node_labels[0] ?? "" } : { concept: "", toward: "" };
  }
  /** Focusable ref shared by every modal input variant — the regenerate
   *  / regen_mode number-of-siblings field swaps in a themed
   *  ``NumberInput`` which exposes a structurally-matching
   *  ``focus()`` / ``select()`` pair, so the modal-open auto-focus
   *  works regardless of which branch is rendered. */
  type FocusableRef = {
    focus: () => void;
    select?: () => void;
  };
  let modalInput: FocusableRef | null = $state(null);
  let modalEl: HTMLElement | null = $state(null);
  let modalCancelButton: HTMLButtonElement | null = $state(null);
  let modalTrigger: HTMLElement | null = null;
  let backgroundStates: Array<{
    element: HTMLElement;
    inert: boolean;
    ariaHidden: string | null;
  }> = [];

  function hideModalBackground(): void {
    if (backgroundStates.length > 0) return;
    const zone = background?.closest<HTMLElement>(".loom-zone");
    const layout = zone?.parentElement;
    if (!zone || !layout) return;
    backgroundStates = Array.from(layout.children)
      .filter((element): element is HTMLElement => element instanceof HTMLElement && element !== zone)
      .map((element) => ({
        element,
        inert: element.hasAttribute("inert"),
        ariaHidden: element.getAttribute("aria-hidden"),
      }));
    for (const state of backgroundStates) {
      state.element.setAttribute("inert", "");
      state.element.setAttribute("aria-hidden", "true");
    }
  }

  function restoreModalBackground(): void {
    for (const state of backgroundStates) {
      if (!state.element.isConnected) continue;
      if (!state.inert) state.element.removeAttribute("inert");
      if (state.ariaHidden === null) state.element.removeAttribute("aria-hidden");
      else state.element.setAttribute("aria-hidden", state.ariaHidden);
    }
    backgroundStates = [];
  }

  function focusInitialModalControl(): void {
    if (modalInput) {
      modalInput.focus();
      modalInput.select?.();
    } else {
      modalCancelButton?.focus();
    }
  }

  async function openModal(
    kind: ModalState["kind"],
    nodeId: string | null,
    initialText: string = "",
    initialN: number = 1,
    trigger?: HTMLElement | null,
  ): Promise<void> {
    if (kind === "delete" && nodeId && loomNodeIntersectsGeneration(nodeId)) {
      pushToast(LOOM_DELETE_DURING_GENERATION_MESSAGE, { kind: "warning" });
      return;
    }
    modalTrigger = trigger ??
      (document.activeElement instanceof HTMLElement ? document.activeElement : null);
    modalInput = null;
    modal = {
      kind,
      nodeId,
      text: initialText,
      n: initialN,
      ...(kind === "fanout" ? fanoutDefault() : { concept: "", toward: "" }),
      mode: "unsteered",
      error: "",
    };
    await tick();
    focusInitialModalControl();
    background?.setAttribute("inert", "");
    background?.setAttribute("aria-hidden", "true");
    hideModalBackground();
  }

  function closeModal(): void {
    const trigger = modalTrigger;
    background?.removeAttribute("inert");
    background?.removeAttribute("aria-hidden");
    restoreModalBackground();
    modal = {
      kind: null,
      nodeId: null,
      text: "",
      n: 1,
      concept: "",
      toward: "",
      mode: "unsteered",
      error: "",
    };
    modalInput = null;
    modalEl = null;
    modalCancelButton = null;
    modalTrigger = null;
    void tick().then(() => onclosed?.(trigger));
  }

  function onModalKey(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      ev.stopPropagation();
      closeModal();
      return;
    }
    if (modalEl) containTab(ev, modalEl);
  }

  function setModalError(message: string): void {
    // Clear input focus so the error reads; modal stays open.
    modal = { ...modal, error: message };
  }

  let modalBusy = $state(false);
  async function commitModal(): Promise<void> {
    if (modalBusy) return;
    modalBusy = true;
    try { await applyModal(); }
    finally { modalBusy = false; }
  }

  async function applyModal(): Promise<void> {
    const m = modal;
    if (!m.kind || !m.nodeId) return closeModal();
    switch (m.kind) {
      case "clear":
        if (genStatus.active) { setModalError("Stop the current reply before clearing the loom."); return; }
        try {
          await flushConversationAutosave();
          await apiTree.reset();
          savedConversationState.activeId = null;
          savedConversationState.avatarSeed = null;
          savedConversationState.accent = "purple";
          await refreshLoomTree();
          clearNodeSelection();
          clearTreeFilter();
          pushToast("Loom cleared. The previous conversation is still in Saved chats.", { kind: "info" });
        } catch (error) {
          setModalError(userFacingError(error, "The loom could not be cleared. Your saved chats were not deleted."));
          return;
        }
        break;
      case "regenerate":
        await loomRegenerateActive(Math.max(1, Math.floor(m.n)));
        break;
      case "edit":
        await loomEdit(m.nodeId, m.text);
        break;
      case "branch": {
        const newId = await loomBranch(m.nodeId, m.text);
        if (newId) { await loomNavigate(newId); onbranchcreated?.(newId); }
        break;
      }
      case "delete":
        if (loomNodeIntersectsGeneration(m.nodeId)) {
          setModalError(LOOM_DELETE_DURING_GENERATION_MESSAGE);
          return;
        }
        if (!await loomDelete(m.nodeId)) {
          setModalError(loomTree.error ?? "The branch could not be removed. Try again.");
          return;
        }
        break;
      case "note":
        await loomNote(m.nodeId, m.text);
        break;
      case "navpicker": {
        // Resolve a node by id-prefix or full id.  Ambiguity keeps the
        // modal open with a list of matches so the user can disambiguate
        // rather than navigating to a random first match (Map insertion
        // order leaks the bug otherwise).
        const r = resolveNodeIdPrefix(m.text, loomTree.nodes.keys(), loomTree.root_id);
        if (r.id) {
          await loomNavigate(r.id);
          break;
        }
        if (r.matches.length === 0) {
          setModalError(`no node matches '${m.text}'`);
          return;
        }
        const preview = r.matches.slice(0, 6).map((s) => s.slice(0, 8)).join(", ");
        setModalError(
          `ambiguous: ${r.matches.length} matches (${preview}` +
            (r.matches.length > 6 ? ", …" : "") + ")",
        );
        return;
      }
      case "search": {
        await applyTreeFilter(m.text, "text");
        if (!filterState.matchingIds?.size) {
          setModalError(`No text match for '${m.text}'.`);
          return;
        }
        closeModal();
        await onsearchmatch?.();
        return;
      }
      case "fanout": {
        // Fan-out: anchor on the user node, send one regen per strength as a sibling, each steered
        // by the picked concept alone at that strength (a partial recipe's ``steering`` replaces
        // the steering), written as the chips write it.
        const info = fanoutConcepts.find((row) => `${row.namespace}/${row.name}` === m.concept);
        if (!info || fanoutRecipe(info, m.toward ?? "", 0) === null) {
          setModalError("Pick a concept and the node to steer toward.");
          return;
        }
        const alphas = parseStrengthList(m.text);
        if (alphas.length === 0) {
          setModalError(
            "couldn't parse alphas, try a comma list (0.0, 0.3, 0.7) " +
            "or linspace(-1, 1, 5)",
          );
          return;
        }
        const userId = m.nodeId;
        for (const alpha of alphas) {
          await loomContinueFromCommitted(userId, {
            n: 1,
            recipe_override: fanoutRecipe(info, m.toward ?? "", alpha),
          });
        }
        break;
      }
      case "regen_mode": {
        // Manual regen-with-modifier: regenerate a generated node, or
        // author fresh continuations under a committed node, and dispatch N
        // siblings under the
        // chosen mode.  Modes are resolved engine-side.
        const node = loomTree.nodes.get(m.nodeId);
        const mode = (m.mode ?? "unsteered").trim();
        const N = Math.max(1, Math.floor(m.n));
        if (node?.recipe === null) {
          await loomContinueFromCommitted(m.nodeId, {
            n: N,
            recipe_override: mode,
          });
        } else {
          // Generated node — set it active then regenerate in its own seat.
          if (loomTree.active_node_id !== m.nodeId) {
            await loomNavigate(m.nodeId);
          }
          await loomRegenerateActive(N, { recipe_override: mode });
        }
        break;
      }
    }
    closeModal();
  }

  // Unmounting with a dialog open must not leave the workspace inert.
  $effect(() => () => restoreModalBackground());

  export function open(
    kind: Exclude<ModalState["kind"], null>,
    nodeId: string | null,
    initialText = "",
    initialN = 1,
    trigger?: HTMLElement | null,
  ): Promise<void> {
    return openModal(kind, nodeId, initialText, initialN, trigger);
  }

  export function close(): void {
    closeModal();
  }

  export function isOpen(): boolean {
    return modal.kind !== null;
  }
</script>

{#if modal.kind}
  <div
    class="loom-modal-backdrop"
    aria-hidden="true"
    onclick={closeModal}
    in:fadeIn
    out:fadeOut
  ></div>
  <div
    class="loom-modal"
    role="dialog"
    aria-modal="true"
    aria-labelledby="loom-modal-title"
    tabindex="-1"
    bind:this={modalEl}
    onkeydown={onModalKey}
    in:rise
    out:fadeOut
  >
    <header class="modal-header">
      <h2 id="loom-modal-title">
        {#if modal.kind === "regenerate"}Generate alternatives
        {:else if modal.kind === "edit"}Edit turn
        {:else if modal.kind === "branch"}Start a branch
        {:else if modal.kind === "delete"}Delete branch
        {:else if modal.kind === "clear"}Clear loom
        {:else if modal.kind === "note"}Add note
        {:else if modal.kind === "navpicker"}Go to turn
        {:else if modal.kind === "search"}Search conversation
        {:else if modal.kind === "fanout"}Create guided variations
        {:else if modal.kind === "regen_mode"}Generate several replies
        {/if}
      </h2>
      <IconButton icon="close" label="Close" onclick={closeModal} />
    </header>
    <div class="modal-body">
      {#if modal.kind === "regenerate"}
        <label>
          <span class="meta">Replies</span>
          <NumberInput
            bind:this={modalInput}
            value={modal.n}
            min={1}
            max={16}
            step={1}
            oninput={(v) => { if (v !== null) modal.n = v; }}
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
      {:else if modal.kind === "delete"}
        <blockquote>{loomTree.nodes.get(modal.nodeId ?? "")?.text.slice(0, 240) || "Empty turn"}</blockquote>
        <p>This turn and everything after it go, here and in the saved chat. Earlier turns and sibling branches stay. It cannot be undone.</p>
      {:else if modal.kind === "clear"}
        <p>The whole conversation, every branch with it, stays in Saved chats; the loom starts empty.</p>
      {:else if modal.kind === "navpicker" || modal.kind === "search"}
        <input
          bind:this={modalInput as HTMLInputElement}
          bind:value={modal.text}
          type="text"
          aria-label={modal.kind === "navpicker" ? "Node ID prefix" : "Search node text"}
          placeholder={modal.kind === "navpicker" ? "node id prefix (or 'root')" : "search node text"}
          onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
        />
      {:else if modal.kind === "fanout"}
        {#if fanoutConcepts.length === 0}
          <p class="meta">No concept for this model</p>
          <div class="dw-row"><button type="button" class="dw-quiet" onclick={() => { closeModal(); goto("artifacts", "build"); }}>Build a probe</button></div>
        {:else}
          <label>
            <span class="meta">Concept</span>
            <Select
              value={modal.concept ?? ""}
              options={fanoutConcepts.map((row) => ({ value: `${row.namespace}/${row.name}`, label: `${row.namespace}/${row.name}` }))}
              ariaLabel="The concept each variation steers with"
              onchange={(value) => { modal.concept = value; modal.toward = fanoutNodes(value)[0] ?? ""; }}
            />
          </label>
          <label>
            <span class="meta">Toward</span>
            <Select
              value={modal.toward ?? ""}
              options={fanoutNodes(modal.concept ?? "").map((node) => ({ value: node, label: node }))}
              ariaLabel="The node each variation steers toward"
              onchange={(value) => { modal.toward = value; }}
            />
          </label>
        {/if}
        <label>
          <span class="meta">Strengths</span>
          <input
            bind:this={modalInput as HTMLInputElement}
            bind:value={modal.text}
            type="text"
            placeholder="0.0, 0.3, 0.6 · linspace(-1, 1, 5) · 0:1:0.25"
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
      {:else if modal.kind === "regen_mode"}
        <label>
          <span class="meta">Variation</span>
          <Select
            value={modal.mode ?? "unsteered"}
            options={[
              { value: "unsteered", label: "unsteered" },
              { value: "inverted", label: "inverted" },
              { value: "reseed", label: "reseed" },
              { value: "cool", label: "cool" },
              { value: "hot", label: "hot" },
            ]}
            onchange={(v) => { modal.mode = v; }}
            ariaLabel="regen mode"
          />
        </label>
        <label>
          <span class="meta">Replies</span>
          <NumberInput
            bind:this={modalInput}
            value={modal.n}
            min={1}
            max={16}
            step={1}
            oninput={(v) => { if (v !== null) modal.n = v; }}
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
      {:else}
        <textarea
          bind:this={modalInput as HTMLTextAreaElement}
          bind:value={modal.text}
          rows="6"
          aria-label={modal.kind === "edit" ? "Node text" : modal.kind === "branch" ? "Branch text" : "Node note"}
          placeholder={modal.kind === "branch" ? "(empty = branch from blank)" : ""}
          onkeydown={(ev) => {
            if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
              ev.preventDefault();
              void commitModal();
            }
            // Prevent browser bold-formatting on Ctrl+B inside the
            // textarea for branch buffers.
            if (ev.key === "b" && (ev.metaKey || ev.ctrlKey)) {
              ev.preventDefault();
            }
          }}
        ></textarea>
        <p class="meta">⌘⏎</p>
      {/if}
      {#if modal.error}
        <p class="modal-error" role="alert">{modal.error}</p>
      {/if}
    </div>
    <footer class="modal-footer">
      <button
        type="button"
        class="dw-text"
        bind:this={modalCancelButton}
        onclick={closeModal}
      >Cancel</button>
      <button
        type="button"
        class="dw-button"
        onclick={() => void commitModal()}
        disabled={modalBusy || (modal.kind === "clear" && genStatus.active) || (modal.kind === "delete" && modal.nodeId !== null &&
          loomNodeIntersectsGeneration(modal.nodeId))}
        {...{ "aria-description": (modal.kind === "delete" && modal.nodeId !== null &&
          loomNodeIntersectsGeneration(modal.nodeId)
          ? LOOM_DELETE_DURING_GENERATION_MESSAGE
          : "") }}
      >
        <MorphText text={modalBusy ? (modal.kind === "branch" ? "Creating branch…" : "Applying…")
          : modal.kind === "delete" ? "Delete branch"
          : modal.kind === "clear" ? "Save and clear loom"
          : modal.kind === "edit" || modal.kind === "note" ? "Save"
          : modal.kind === "branch" ? "Create branch"
          : modal.kind === "navpicker" ? "Go"
          : modal.kind === "search" ? "Search" : "Generate"} numbers={false} />
      </button>
    </footer>
  </div>
{/if}

<style>
  /* A small dialog: paper over a wash2 scrim, as the shell's dialogs are. */
  .loom-modal-backdrop { position: fixed; inset: 0; z-index: 100; background: color-mix(in oklab, var(--wash2) 88%, transparent); -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px); cursor: pointer; }
  .loom-modal {
    position: fixed;
    top: 50%;
    left: 50%;
    z-index: 101;
    display: flex;
    flex-direction: column;
    gap: var(--s4);
    width: min(560px, calc(100vw - var(--s4) * 2));
    max-height: calc(100dvh - var(--s6));
    min-width: 0;
    padding: var(--s5) var(--s6) var(--s6);
    border-radius: var(--r-row);
    background: var(--paper);
    color: var(--ink2);
    font: var(--type-small);
    transform: translate(-50%, -50%);
  }
  .loom-modal:focus { outline: none; }
  .modal-header { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); }
  .modal-header h2 { margin: 0; color: var(--ink); font: var(--type-lead); }
  /* Inset by the focus ring's reach, so a focused field's ring is not clipped by the scroll box. */
  .modal-body { display: flex; flex-direction: column; gap: var(--s3); min-width: 0; margin: -4px; padding: 4px; overflow-y: auto; }
  .modal-body label { display: grid; gap: var(--s1); }
  .modal-body input, .modal-body textarea { width: 100%; min-width: 0; padding: var(--s2) var(--s3); border: 0; border-radius: var(--r-col); background: var(--wash); color: var(--ink); font: var(--type-body); resize: vertical; }
  .modal-body input { height: 40px; }
  .modal-body blockquote { margin: 0; padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); color: var(--ink); overflow-wrap: anywhere; }
  .modal-footer { display: flex; align-items: center; justify-content: flex-end; gap: var(--s4); }
  .modal-error { padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); color: var(--ink); overflow-wrap: anywhere; }
  @media (max-width: 720px) {
    .loom-modal { padding: var(--s4) var(--s4) var(--s5); }
    .modal-body input, .modal-body textarea { font-size: 16px; }
  }
  @media (forced-colors: active) {
    .modal-error { background: Canvas; color: CanvasText; }
  }
  @media (pointer: coarse) { .dw-text, .modal-body input { min-height: 44px; } }
</style>
