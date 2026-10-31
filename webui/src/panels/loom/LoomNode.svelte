<script lang="ts">
  import { MorphText } from "@stdlogits/ledger";
  // A completion node in the loom canvas. Generated completions expose their
  // actual model-token rows directly; authored text remains one editable
  // snippet because it has no authoritative tokenizer record.

  import type { LoomNodeJSON } from "../../lib/types";
  import { tokenRowToScore } from "../../lib/stores.svelte";
  import { highlightStyleString } from "../../lib/highlight";
  import { splitTokenSentences } from "./loomSentences";

  interface Props {
    node: LoomNodeJSON;
    displayId?: string;
    onmeasure?: (height: number) => void;
    tokenStart?: number;
    tokenEnd?: number;
    sharedCount?: number;
    sharedUsesActivePath?: boolean;
    /** Active path membership — bold + ring + accent. */
    onActivePath: boolean;
    /** Current roving-tab-stop node for sidebar keyboard navigation. */
    focused: boolean;
    /** Exact active node, distinct from active-path ancestry. */
    current: boolean;
    /** Multi-selection state used by branch comparison. */
    selected: boolean;
    /** One-based visual tree depth. */
    level: number;
    /** Whether this item owns visible child items. */
    hasChildren: boolean;
    /** Dead branch (not on active path) — retain readable text, quiet the chrome. */
    dead: boolean;
    /** In-flight target — pulse the node so the user sees streaming. */
    streaming: boolean;
    /** Click handler — navigate to this node. */
    onclick?: (ev: MouseEvent) => void;
    /** Focus handler keeps the sidebar's roving tab stop synchronized. */
    onfocus?: (ev: FocusEvent) => void;
    /** Tree keyboard handler supplied by the owning tree. */
    onkeydown?: (ev: KeyboardEvent) => void;
    /** Right-click handler — open the context menu. */
    oncontextmenu?: (ev: MouseEvent) => void;
    /** Visible action-menu trigger for pointer and touch users. */
    onactions?: (ev: MouseEvent) => void;
    /** Generate another continuation at this point. */
    ongrow?: (ev: MouseEvent) => void;
    /** Create an editable sibling path. */
    onbranch?: (ev: MouseEvent) => void;
    /** Select this exact model token and open its branch-point tools. */
    onselecttoken?: (tokenIndex: number, ev: MouseEvent) => void;
    /** Continue from the exact token at the end of a sentence as a sibling. */
    onbranchsentence?: (tokenIndex: number, ev: MouseEvent) => void;
    /** Runtime and node both support exact forced-token replay. */
    sentenceBranchAvailable?: boolean;
    /** Hide or reveal descendants in the canvas. */
    ontogglechildren?: (ev: MouseEvent) => void;
    /** Whether descendants are currently hidden in this view. */
    collapsed?: boolean;
    /** Optional probe ring fill in [-1, 1] (null when no probe aggregate). */
    ring?: number | null;
    /** Logit-pass: per-turn ``mean_logprob`` to render as a numeric
     *  badge.  Null (capture wasn't live) suppresses the badge. */
    weightBadge?: number | null;
    /** Steering-delta label for the edge into this node (e.g.
     *  ``0.45 angry.calm``).  Rendered as a trailing chip — it used to
   *  appears in the card so it stays readable at every zoom. */
    steerLabel?: string | null;
    /** How this sibling diverged from another reply. */
    forkLabel?: string | null;
  }

  let {
    node,
    displayId,
    onmeasure,
    tokenStart = 0,
    tokenEnd,
    sharedCount = 0,
    sharedUsesActivePath = false,
    onActivePath,
    focused,
    current,
    selected,
    level,
    hasChildren,
    dead,
    streaming,
    onclick,
    onfocus,
    onkeydown,
    oncontextmenu,
    onactions,
    ongrow,
    onbranch,
    onselecttoken,
    onbranchsentence,
    sentenceBranchAvailable = false,
    ontogglechildren,
    collapsed = false,
    ring = null,
    weightBadge = null,
    steerLabel = null,
    forkLabel = null,
  }: Props = $props();

  function measureCard(element: HTMLElement) {
    let frame = 0;
    let lastHeight = 0;
    const measure = () => {
      frame = 0;
      if (!element.clientWidth) return;
      const style = getComputedStyle(element);
      const children = [...element.children].filter(child => getComputedStyle(child).display !== "none") as HTMLElement[];
      let height = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom)
        + Math.max(0, children.length - 1) * parseFloat(style.rowGap);
      for (const child of children) {
        if (child.classList.contains("token-field")) {
          const field = getComputedStyle(child);
          const sentences = [...child.children] as HTMLElement[];
          height += parseFloat(field.paddingTop) + parseFloat(field.paddingBottom)
            + Math.max(0, sentences.length - 1) * parseFloat(field.rowGap)
            + sentences.reduce((sum, sentence) => sum + sentence.offsetHeight, 0);
        } else height += child.offsetHeight;
      }
      height = Math.ceil(height);
      if (height !== lastHeight) {
        lastHeight = height;
        onmeasure?.(height);
      }
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(measure); };
    const resize = new ResizeObserver(schedule);
    const observe = () => {
      resize.disconnect();
      resize.observe(element);
      element.querySelectorAll<HTMLElement>(":scope > *, .sentence-node").forEach(child => resize.observe(child));
      schedule();
    };
    const mutations = new MutationObserver(observe);
    mutations.observe(element, { childList: true, subtree: true, characterData: true });
    observe();
    return { destroy() { resize.disconnect(); mutations.disconnect(); cancelAnimationFrame(frame); } };
  }

  const PREVIEW_CHARS = 600;

  const preview = $derived.by(() => {
    const t = (node.text ?? "").replace(/\s+/g, " ").trim();
    if (!t) return node.role === "system" && !node.parent_id ? "root" : "(empty)";
    return t.length > PREVIEW_CHARS ? t.slice(0, PREVIEW_CHARS) + "…" : t;
  });

  const roleName = $derived(node.role_label?.trim() || node.role);
  const visibleTokens = $derived(node.tokens?.slice(tokenStart, tokenEnd) ?? []);
  const sentences = $derived(splitTokenSentences(visibleTokens).map(sentence => ({
    ...sentence, start: sentence.start + tokenStart, end: sentence.end + tokenStart,
  })));

  function tokenName(text: string): string {
    const readable = text.replace(/\n/g, "↵").replace(/\t/g, "⇥").trim();
    return readable || "whitespace";
  }

  function tokenTitle(index: number, text: string, logprob: number | null): string {
    if (typeof logprob !== "number" || !Number.isFinite(logprob)) {
      return `Token ${index + 1}: ${tokenName(text)}`;
    }
    const probability = Math.min(1, Math.max(0, Math.exp(logprob)));
    return `Token ${index + 1}: ${tokenName(text)} · ${(probability * 100).toFixed(1)}%`;
  }

  // The highlight probe's aggregate for this turn, as a dot: the depth accent for a positive
  // reading, faint ink for a negative one, its strength as the dot's opacity.
  const ringStyle = $derived(ring === null ? null
    : `--ring:${ring >= 0 ? "var(--tint)" : "var(--ink3)"};opacity:${(0.35 + 0.65 * Math.min(1, Math.abs(ring))).toFixed(2)}`);
</script>

<div
  use:measureCard
  class="node"
  class:shared={sharedCount > 0}
  class:active={onActivePath}
  class:focused
  class:dead
  class:streaming
  class:generation-active={streaming}
  class:starred={!sharedCount && node.starred}
  class:user={node.role === "user"}
  class:assistant={node.role === "assistant"}
  class:system={node.role === "system"}
  role="treeitem"
  aria-label={sharedCount ? `Shared prefix across ${sharedCount} replies` : undefined}
  aria-level={level}
  aria-expanded={!sharedCount && hasChildren ? !collapsed : undefined}
  aria-current={current ? "true" : undefined}
  aria-selected={sharedCount ? undefined : selected}
  tabindex={focused ? 0 : -1}
  data-node-id={displayId ?? node.id}
  data-cursor={oncontextmenu ? "context-menu" : undefined}
  {onclick}
  {onfocus}
  {onkeydown}
  {oncontextmenu}
>
  <span class="node-head">
    <span class="identity">
      <span class="pip" aria-hidden="true"></span>
      <span class="role-name"><MorphText text={sharedCount ? `shared by ${sharedCount}` : roleName} numbers={false} /></span>
      {#if ringStyle}<span class="ring" style={ringStyle} aria-hidden="true"></span>{/if}
      {#if streaming}<span class="generation-label" role="status">Writing…</span>{/if}
    </span>
    {#if !sharedCount}
      <span class="head-meta">
        {#if node.starred}<span class="star" {...{ "aria-description": "Saved branch" }} aria-hidden="true">★</span>{/if}
        <button
          type="button"
          class="node-actions"
          aria-label={`Actions for ${roleName}: ${preview}`}
          {...{ "aria-description": "Message actions" }}
          onclick={(ev) => {
            ev.stopPropagation();
            onactions?.(ev);
          }}
        >•••</button>
      </span>
    {/if}
  </span>
  {#if node.tokens && visibleTokens.length > 0}
    <div class="token-field" aria-label={`${sentences.length} sentence branch points`}>
      {#each sentences as sentence, sentenceIndex (`${sentence.start}:${sentence.end}`)}
        {@const boundary = node.tokens[sentence.end]}
        {@const canBranch = sentenceBranchAvailable &&
          boundary?.raw_index != null && boundary?.token_id != null}
        <div
          class="sentence-node"
          data-loom-sentence={sentenceIndex}
          data-sentence-start={sentence.start}
          data-sentence-end={sentence.end}
        >
          <span class="sentence-tokens">{#each node.tokens.slice(sentence.start, sentence.end + 1) as token, offset (`${token.raw_index ?? sentence.start + offset}:${token.token_id ?? token.text}`)}{@const tokenIndex = sentence.start + offset}<button
                type="button"
                class="token-node"
                data-cursor="inspect"
                style={sharedCount ? "" : highlightStyleString(tokenRowToScore(token))}
                data-loom-token-node={node.id}
                data-token-index={tokenIndex}
                data-raw-index={token.raw_index ?? undefined}
                {...{ "aria-description": (`${tokenTitle(tokenIndex, token.text, sharedCount ? null : token.logprob)} · Open branch-point tools`) }}
                aria-label={`Open token ${tokenIndex + 1}: ${tokenName(token.text)}`}
                aria-haspopup="dialog"
                onclick={(ev) => {
                  ev.stopPropagation();
                  onselecttoken?.(tokenIndex, ev);
                }}
              >{token.text || "∅"}</button>{/each}</span><button
            type="button"
            class="sentence-branch"
            disabled={!canBranch}
            {...{ "aria-description": (canBranch
              ? "Keep this sentence and generate a different continuation"
              : "This saved reply does not have an exact replay boundary") }}
            aria-label={`Branch after sentence ${sentenceIndex + 1}: ${sentence.text.trim()}`}
            onclick={(ev) => {
              ev.stopPropagation();
              if (canBranch) onbranchsentence?.(sentence.end, ev);
            }}
          >Branch here</button>
        </div>
      {/each}
    </div>
  {:else}
    <span class="preview"><MorphText text={streaming ? "Preparing continuation…" : tokenStart > 0 ? "End of reply" : preview} /></span>
  {/if}
  {#if sharedCount}
    <span class="shared-hint">Readings from {sharedUsesActivePath ? "the current" : "the first"} path</span>
  {/if}
  {#if !sharedCount && (forkLabel || steerLabel || weightBadge != null || node.notes)}
    <span class="node-meta">
      {#if forkLabel}<span class="fork" {...{ "aria-description": "Where this reply diverged" }}>{forkLabel}</span>{/if}
      {#if steerLabel}<span class="steer" {...{ "aria-description": "Guidance changed on this branch" }}>{steerLabel}</span>{/if}
      {#if weightBadge != null}
        <span class="weight" {...{ "aria-description": "Average token log probability" }}>
          <span aria-hidden="true">log p</span> <MorphText text={weightBadge.toFixed(2)} />
        </span>
      {/if}
      {#if node.notes}<span class="note-mark" data-tip={node.notes} {...{ "aria-description": (node.notes) }} aria-label="Has a note">note</span>{/if}
    </span>
  {/if}
  {#if !sharedCount}
  <span class="node-tools" aria-label="Branch actions">
    <button
      type="button"
      {...{ "aria-description": "Generate another path from here" }}
      aria-label={`Generate another path from ${roleName}`}
      onclick={(ev) => { ev.stopPropagation(); ongrow?.(ev); }}
    >Generate</button>
    <button
      type="button"
      {...{ "aria-description": "Write an alternative at this point" }}
      aria-label={`Write an alternative to ${roleName}`}
      onclick={(ev) => { ev.stopPropagation(); onbranch?.(ev); }}
    >Write</button>
    {#if hasChildren}
      <button
        type="button"
        {...{ "aria-description": (collapsed ? "Show paths after this point" : "Hide paths after this point") }}
        aria-label={collapsed ? "Show child paths" : "Hide child paths"}
        aria-pressed={collapsed}
        onclick={(ev) => { ev.stopPropagation(); ontogglechildren?.(ev); }}
      >{collapsed ? "Show" : "Hide"}</button>
    {/if}
  </span>
  {/if}
</div>

<style>
  /* A turn on the map: a fill, no line. Off the path a wash; on the current path the depth accent
   * at 6%, the current turn at 11%. Its text runs as text, each token a tinted span. */
  .node {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: var(--s2);
    width: 100%;
    height: 100%;
    min-width: 0;
    min-height: 36px;
    padding: var(--s3) var(--s4);
    overflow: hidden;
    border-radius: var(--r-box);
    background: var(--wash);
    color: var(--ink2);
    font: var(--type-small);
    cursor: pointer;
    user-select: none;
    transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease), box-shadow var(--fast) var(--ease);
  }
  .node > * { flex-shrink: 0; }
  .node:hover { background: var(--wash2); }
  .node.active { background: color-mix(in oklch, var(--tint) 6%, var(--paper)); color: var(--ink); }
  .node.active:hover { background: color-mix(in oklch, var(--tint) 9%, var(--paper)); }
  .node[aria-current="true"], .node.streaming { background: color-mix(in oklch, var(--tint) 11%, var(--paper)); color: var(--ink); }
  .node.dead { color: var(--ink3); }
  .node.dead:hover { color: var(--ink2); }
  .node:focus-visible { outline-offset: 2px; }
  .node.shared { height: auto; max-height: 100%; cursor: default; }
  .node.shared:hover { background: var(--wash); }
  .shared .token-field { flex-grow: 0; }
  .shared-hint { color: var(--ink3); font: var(--type-meta); }

  .node-head, .identity, .head-meta, .node-meta { display: flex; align-items: center; min-width: 0; }
  .node-head { flex: none; justify-content: space-between; gap: var(--s3); height: 20px; }
  .identity { gap: var(--s2); }
  .head-meta { gap: var(--s1); }
  .pip { flex: none; width: 6px; height: 6px; border-radius: 50%; background: var(--ink3); }
  .assistant .pip { background: var(--ink); }
  .role-name { overflow: hidden; color: var(--ink3); font: var(--type-meta); white-space: nowrap; text-overflow: ellipsis; }
  .ring { flex: none; width: 8px; height: 8px; border-radius: 2px; background: var(--ring); }
  .star { color: var(--ink2); font: var(--type-meta); }
  .node-actions {
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    margin-right: calc(var(--s2) * -1);
    padding: 0;
    border: 0;
    border-radius: var(--r-pill);
    background: transparent;
    color: var(--ink3);
    font: var(--type-meta);
    cursor: pointer;
    transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease), scale var(--fast) var(--ease);
  }
  .node-actions:hover { background: var(--paper); color: var(--ink); }
  .node-actions:active { scale: 0.97; }

  .preview {
    display: -webkit-box;
    overflow: hidden;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 6;
    line-clamp: 6;
    min-width: 0;
    overflow-wrap: anywhere;
  }

  /* Sentences as rows: a sentence lifts to a soft wash on hover and offers its branch. */
  .token-field { display: flex; flex-direction: column; gap: 2px; min-width: 0; margin: 0 calc(var(--s2) * -1); }
  .sentence-node { padding: 2px var(--s2); border-radius: var(--r-col); transition: background-color var(--fast) var(--ease); }
  .node:not(.shared) .sentence-node:hover, .node:not(.shared) .sentence-node:focus-within { background: color-mix(in oklab, var(--paper) 55%, transparent); }
  .sentence-tokens { line-height: 22px; overflow-wrap: anywhere; }
  .token-node {
    display: inline;
    padding: 0;
    border: 0;
    border-radius: 4px;
    background: transparent;
    color: inherit;
    font: inherit;
    text-align: start;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    cursor: pointer;
    transition: background-color var(--fast) var(--ease), box-shadow var(--fast) var(--ease), color var(--fast) var(--ease);
  }
  .token-node:not([style*="background"]):hover { background: var(--wash2); box-shadow: 0 0 0 2px var(--wash2); }
  .token-node:hover { color: var(--ink); }
  .token-node:focus-visible { outline-offset: 1px; }
  .sentence-branch {
    margin-left: var(--s2);
    padding: 0;
    border: 0;
    background: transparent;
    color: var(--ink3);
    font: var(--type-meta);
    white-space: nowrap;
    cursor: pointer;
    opacity: 0;
    transition: opacity var(--fast) var(--ease), color var(--fast) var(--ease);
  }
  .sentence-node:hover .sentence-branch, .sentence-branch:focus-visible { opacity: 1; }
  .sentence-branch:hover { color: var(--ink); }
  .sentence-branch:disabled { display: none; }

  .node-meta { flex: none; flex-wrap: wrap; gap: 2px var(--s3); overflow: hidden; color: var(--ink3); font: var(--type-meta); }
  .fork, .steer { max-width: 22ch; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .steer { color: var(--ink2); }
  .weight { font-variant-numeric: tabular-nums; }

  /* The tools wait out of sight until the turn is hovered, focused or current. */
  .node-tools {
    display: flex;
    flex: none;
    align-items: center;
    gap: var(--s4);
    margin-top: auto;
    opacity: 0;
    transition: opacity var(--fast) var(--ease);
  }
  .node:hover .node-tools, .node:focus-within .node-tools, .node.focused .node-tools, .node[aria-current="true"] .node-tools { opacity: 1; }
  .node-tools button {
    min-height: 28px;
    padding: 0;
    border: 0;
    background: transparent;
    color: var(--ink3);
    font: var(--type-meta);
    cursor: pointer;
    transition: color var(--fast) var(--ease);
  }
  .node-tools button:hover, .node-tools button[aria-pressed="true"] { color: var(--ink); }

  @media (pointer: coarse) {
    .node-actions { width: 44px; height: 44px; }
    .node-tools { opacity: 1; }
    .node-tools button { min-height: 44px; }
    .sentence-branch { opacity: 1; }
  }
</style>
