<script lang="ts">
  import { Expander } from "@stdlogits/ledger";
  import Select from "../../lib/Select.svelte";
  import { tick, untrack } from "svelte";
  import { SvelteSet } from "svelte/reactivity";
  import {
    effectiveRawMode, genStatus, loomNavigate, loomStar, loomTree,
    sendFork, sendGenerate, sendStop, sendSubmit,
  } from "../../lib/stores.svelte";
  import type { LoomNodeJSON } from "../../lib/types";
  import { userFacingError } from "../../lib/runtime/userFacingError";

  let { active, oninspect, onwrite }: {
    active: boolean;
    oninspect: (node: LoomNodeJSON, index: number) => void;
    onwrite: (node: LoomNodeJSON, trigger: HTMLElement) => void;
  } = $props();

  let anchorId = $state<string | null>(null);
  let count = $state(3);
  let prompt = $state("");
  let startingPrompt = $state<string | null>(null);
  const expandedTokens = new SvelteSet<string>();
  let dispatching = $state(false);
  let error = $state("");
  let choicesHeading: HTMLHeadingElement;
  const busy = $derived(dispatching || genStatus.active);
  const anchor = $derived(anchorId ? loomTree.nodes.get(anchorId) : undefined);
  const choices = $derived((anchorId ? loomTree.children_of.get(anchorId) ?? [] : [])
    .map(id => loomTree.nodes.get(id)).filter((node): node is LoomNodeJSON => !!node));
  const path = $derived(loomTree.activePath.map(id => loomTree.nodes.get(id))
    .filter((node): node is LoomNodeJSON => !!node && !(node.parent_id === null && !node.text)));
  const empty = $derived(path.length === 0);
  const chosen = $derived(choices.find(node => loomTree.activePath.includes(node.id)));
  const parent = $derived(anchor?.parent_id ? loomTree.nodes.get(anchor.parent_id) : undefined);

  $effect(() => {
    if (!active) return;
    if (anchorId && loomTree.nodes.has(anchorId)) return;
    const current = untrack(() => loomTree.nodes.get(loomTree.active_node_id ?? ""));
    anchorId = current?.recipe && current.parent_id ? current.parent_id
      : current?.id ?? loomTree.root_id;
  });

  function role(node: LoomNodeJSON): string {
    return node.role_label || node.role;
  }

  async function explore(node: LoomNodeJSON): Promise<void> {
    anchorId = node.id;
    error = "";
    await tick();
    choicesHeading?.focus({ preventScroll: true });
    choicesHeading?.scrollIntoView({ block: "nearest" });
  }

  async function choose(node: LoomNodeJSON): Promise<void> {
    if (busy) return;
    dispatching = true;
    try { await loomNavigate(node.id); }
    finally { dispatching = false; }
  }

  async function generate(event?: SubmitEvent): Promise<void> {
    event?.preventDefault();
    if (busy || !anchorId) return;
    dispatching = true;
    error = "";
    try {
      if (empty) {
        if (!prompt.trim()) {
          error = "Enter some starting text before generating alternatives.";
          return;
        }
        startingPrompt = prompt;
        await sendSubmit(prompt, "user", "assistant", {
          parent_node_id: anchorId, n: count, raw: effectiveRawMode(),
        });
      } else {
        const generatedRole = choices.find(node => node.recipe !== null)?.role;
        await sendGenerate({
          parent_node_id: anchorId, n: count, raw: effectiveRawMode(),
          append_same_role: false,
          generate_seat: generatedRole === "user" ? "user" : "assistant",
        });
      }
    } catch (cause) {
      startingPrompt = null;
      error = userFacingError(cause, "Could not generate alternatives. Please try again.");
    } finally { dispatching = false; }
  }

  function lastBoundary(node: LoomNodeJSON): number {
    const index = (node.tokens?.length ?? 0) - 1;
    const token = node.tokens?.[index];
    return node.recipe && token?.raw_index != null && token.token_id != null ? index : -1;
  }

  async function extend(node: LoomNodeJSON): Promise<void> {
    const token = node.tokens?.[lastBoundary(node)];
    if (busy || token?.raw_index == null || token.token_id == null) return;
    dispatching = true;
    error = "";
    try {
      await sendFork(node.id, token.raw_index, token.token_id, true);
    } catch (cause) {
      error = userFacingError(cause, "Could not continue this text. Please try again.");
    } finally { dispatching = false; }
  }

  // A first prompt creates its own branch point; subsequent generations keep
  // that point fixed while each new sibling arrives.
  $effect(() => {
    if (startingPrompt === null) return;
    const first = choices.find(node => node.role === "user" && node.text === startingPrompt);
    if (first) { anchorId = first.id; prompt = ""; startingPrompt = null; }
  });
</script>

<div class="weave" class:empty>
  <section class="reading" aria-labelledby="weave-reading-title">
    <header class="section-heading">
      <h2 id="weave-reading-title" class="head">Path</h2>
      <span class="meta">{path.length} {path.length === 1 ? "turn" : "turns"}</span>
    </header>
    {#if empty}
      <p class="meta">Nothing yet</p>
    {:else}
      <div class="reading-path" role="list" aria-label="Weave current path">
        {#each path as node, index (node.id)}
          <article role="listitem" class="passage" class:branch-point={anchorId === node.id} data-weave-passage={node.id}>
            <header class="row-head">
              <span class="meta">{role(node)}{anchorId === node.id ? " · branch point" : ""}</span>
              <button type="button" class="dw-text" onclick={() => void explore(node)}
                aria-label={`Explore here after turn ${index + 1}`} aria-pressed={anchorId === node.id}>Explore here</button>
            </header>
            <div class="passage-text" dir="auto">{node.text}</div>
            {#if node.tokens?.length}
              <Expander label="Tokens" bind:open={() => expandedTokens.has(node.id), (open) => { if (open) expandedTokens.add(node.id); else expandedTokens.delete(node.id); }}>
                <div class="toks tokens">
                  {#each node.tokens as token, tokenIndex}
                    <button type="button" data-cursor="inspect" disabled={busy} {...{ "aria-description": "Inspect or branch from this token" }}
                      aria-label={`Inspect token ${tokenIndex + 1}: ${token.text.trim() || "whitespace"}`}
                      onclick={() => oninspect(node, tokenIndex)}>{token.text || "∅"}</button>
                  {/each}
                </div>
              </Expander>
            {/if}
          </article>
        {/each}
      </div>
    {/if}
  </section>

  <section class="possibilities" aria-labelledby="weave-choices-title">
    <header class="section-heading">
      <h2 id="weave-choices-title" class="head" tabindex="-1" bind:this={choicesHeading}>Continuations</h2>
      {#if parent}<button type="button" class="dw-text" onclick={() => void explore(parent)}>Back one point</button>{/if}
    </header>
    {#if !empty && anchor}
      <div class="context">
        <Expander label={`After ${role(anchor)} · ${anchor.text.slice(0, 70) || "Start of conversation"}${anchor.text.length > 70 ? "…" : ""}`}>
          <div class="passage-text">{anchor.text}</div>
        </Expander>
      </div>
    {/if}
    <form onsubmit={generate}>
      {#if empty}
        <label class="sr-only" for="weave-prompt">Starting text</label>
        <textarea id="weave-prompt" name="prompt" dir="auto" bind:value={prompt} required rows="5"
          placeholder="Starting text" disabled={busy}
          onkeydown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}></textarea>
      {/if}
      <div class="generation-controls">
        <span class="count"><Select bind:value={count} disabled={busy} ariaLabel="Alternatives" options={[1, 2, 3, 4, 6, 8].map(value => ({ value, label: String(value) }))} /></span>
        <button type="submit" class="dw-button" disabled={busy || !loomTree.loaded}
          {...{ "aria-description": "Create sibling turns using the current model and response controls" }}>Generate {count}</button>
        {#if genStatus.active}<button type="button" class="dw-text" onclick={sendStop}>Stop</button>{/if}
      </div>
    </form>
    <p class="meta" role="status">{genStatus.active ? "Generating…" : `${choices.length} available${chosen ? " · one chosen" : ""}`}</p>
    {#if error}<p class="error" role="alert">{error}</p>{/if}
    <div class="choice-list" role="list" aria-label="Weave alternatives">
      {#each choices as node, index (node.id)}
        <article class="choice" class:chosen={chosen?.id === node.id} class:generation-active={genStatus.active && loomTree.pendingNodeId === node.id} role="listitem" data-weave-choice={node.id}>
          <header class="row-head">
            <span class="meta">{index + 1} · {role(node)}{chosen?.id === node.id ? " · chosen" : ""}</span>
            {#if genStatus.active && loomTree.pendingNodeId === node.id}<span class="generation-label" role="status">Writing…</span>{/if}
            <button type="button" class="dw-text star" aria-label={`${node.starred ? "Starred" : "Star"} option ${index + 1}`} aria-pressed={node.starred}
              onclick={() => void loomStar(node.id, !node.starred)}>{node.starred ? "Starred" : "Star"}</button>
          </header>
          <div class="passage-text" dir="auto">{node.text || (genStatus.active ? "Generating…" : "Empty turn")}</div>
          <footer>
            <button type="button" class="dw-quiet" disabled={busy || chosen?.id === node.id}
              onclick={() => void choose(node)}>{chosen?.id === node.id ? "Chosen" : "Choose"}</button>
            <button type="button" class="dw-text" disabled={busy} onclick={async () => { await choose(node); if (loomTree.active_node_id === node.id) await explore(node); }}
              {...{ "aria-description": "Choose this path and explore its next turns" }}>Explore next</button>
            <button type="button" class="dw-text" disabled={busy || lastBoundary(node) < 0}
              {...{ "aria-description": (lastBoundary(node) < 0
                ? "This turn has no saved token boundary to continue from"
                : "Keep every existing token and sample more text with this turn’s saved settings; the original stays intact") }}
              onclick={() => void extend(node)}>Continue</button>
            <button type="button" class="dw-text" disabled={busy || !node.parent_id} onclick={(event) => onwrite(node, event.currentTarget)}>Write another</button>
          </footer>
        </article>
      {/each}
    </div>
  </section>
</div>

<style>
  /* The weave: the path on the left as rows, the branch point's continuations on the right as
   * washes. */
  .weave { display: grid; grid-template-columns: minmax(0, 1fr) minmax(280px, 0.9fr); gap: var(--s6); flex: 1; min-width: 0; min-height: 0; overflow: hidden; }
  section { display: flex; flex-direction: column; gap: var(--s3); min-width: 0; overflow-y: auto; overscroll-behavior: contain; padding-bottom: var(--s6); }
  .section-heading, .row-head, footer, .generation-controls { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s4); }
  .section-heading, .row-head { justify-content: space-between; }
  .section-heading { min-height: 36px; }
  h2 { margin: 0; }
  h2:focus { outline: none; }
  .reading-path { display: grid; gap: 2px; }
  .passage { display: grid; gap: var(--s1); min-width: 0; padding: var(--s2) var(--s3) var(--s3); border-radius: var(--r-box); transition: background-color var(--fast) var(--ease); }
  .passage:hover { background: var(--wash); }
  .passage.branch-point { background: var(--wash2); }
  .row-head { min-height: 28px; }
  .row-head .dw-text { font: var(--type-meta); }
  .passage-text { color: var(--ink); font: var(--type-small); line-height: 22px; white-space: pre-wrap; overflow-wrap: anywhere; }
  .passage .passage-text { font: var(--type-body); line-height: 28px; }
  .tokens { padding: var(--s1) 0; }
  .tokens button { max-width: 100%; padding: 2px 6px; border: 0; border-radius: var(--r-chip); background: var(--paper); color: var(--ink2); font: var(--type-small); white-space: pre-wrap; overflow-wrap: anywhere; cursor: pointer; transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease); }
  .passage:not(.branch-point) .tokens button { background: var(--wash); }
  .tokens button:hover:not(:disabled) { color: var(--ink); background: var(--wash2); }
  .tokens button:disabled { opacity: 0.4; cursor: default; }
  .context { color: var(--ink2); }
  form { display: grid; gap: var(--s3); }
  textarea { width: 100%; min-height: 120px; padding: var(--s3) var(--s4); border: 0; border-radius: var(--r-box); background: var(--wash); color: var(--ink); font: var(--type-body); resize: vertical; }
  .generation-controls { gap: var(--s2) var(--s3); }
  .count { width: 88px; }
  .choice-list { display: grid; gap: var(--s3); }
  .choice { display: grid; gap: var(--s2); min-width: 0; padding: var(--s3) var(--s4) var(--s4); border-radius: var(--r-row); background: var(--wash); transition: background-color var(--fast) var(--ease); }
  .choice.chosen, .choice.generation-active { background: var(--wash2); }
  .choice .dw-quiet { background: var(--wash2); }
  .choice.chosen .dw-quiet, .choice .dw-quiet:hover:not(:disabled) { background: var(--paper); }
  .star { margin-left: auto; }
  footer { gap: var(--s2) var(--s4); padding-top: var(--s1); }
  footer .dw-text { font: var(--type-meta); }
  .error { color: var(--ink); }
  @media (max-width: 960px), (max-height: 700px) {
    .weave { display: block; overflow-y: auto; }
    section { overflow: visible; }
    section + section { margin-top: var(--s5); }
    textarea { font-size: 16px; }
  }
  @media (pointer: coarse) {
    .row-head .dw-text, footer .dw-text { min-height: 44px; }
  }
</style>
