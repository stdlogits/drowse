<script lang="ts" module>
  /** How many tokens either side of the cursor the line draws (the track reaches the rest). */
  export const STRIP_REACH = 160;
</script>

<script lang="ts">
  // The token cursor at the top of the Units, Lens and Probes rooms (lib/stores/tokenCursor): the
  // conversation's tokens on one line, prompt and reply alike, each turn opened by its seat, the
  // cursor's token marked; under it a track of the whole conversation, a turn a stretch, where a
  // click lands on the token at that place. ←/→ step (rolling across turns), Home/End go to the
  // ends of the cursor's segment, and Whole reply reads the reply instead of one token.
  import { tick } from "svelte";
  import { IconButton, MorphText } from "@stdlogits/ledger";
  import { chatLog, roleDisplayLabel } from "../../lib/stores/chat.svelte";
  import { segmentTokens } from "../../drawers/token/cursor";
  import {
    canStep,
    moveCursor,
    segmentEdge,
    showWholeReply,
    stepToken,
    tokenCursor,
    type TokenCursor,
  } from "../../lib/stores/tokenCursor.svelte";

  const at = $derived(tokenCursor.at);
  const whole = $derived(at === null);
  const reply = $derived(tokenCursor.reply);
  const total = $derived(tokenCursor.total);

  /** One drawn token: where the cursor lands on it, its text, whether it opens its turn. */
  interface Item { key: string; at: TokenCursor; text: string; opens: string | null; ordinal: number }

  /** The walk's ordinal where each segment starts. */
  const starts = $derived.by(() => {
    let n = 0;
    return tokenCursor.segments.map((s) => {
      const start = n;
      n += s.length;
      return start;
    });
  });
  const replyRange = $derived.by((): [number, number] | null => {
    if (!reply) return null;
    let first = -1;
    let last = -1;
    tokenCursor.segments.forEach((s, i) => {
      if (s.turnIdx !== reply.turnIdx) return;
      if (first < 0) first = starts[i];
      last = starts[i] + s.length - 1;
    });
    return first < 0 ? null : [first, last];
  });
  /** The ordinal the line centres on: the cursor's token, or the whole reply's start. */
  const centre = $derived(tokenCursor.token?.ordinal ?? replyRange?.[0] ?? 0);

  const items = $derived.by((): Item[] => {
    const from = Math.max(0, centre - STRIP_REACH);
    const to = Math.min(total - 1, centre + STRIP_REACH);
    const out: Item[] = [];
    let lastTurn = -1;
    tokenCursor.segments.forEach((s, i) => {
      const start = starts[i];
      if (start + s.length <= from || start > to) return;
      const turn = chatLog.turns[s.turnIdx];
      const tokens = segmentTokens(turn, s.seg);
      for (let k = Math.max(0, from - start); k < s.length && start + k <= to; k += 1) {
        const opens = s.turnIdx !== lastTurn ? roleDisplayLabel(turn?.role ?? "", turn?.roleLabel) : null;
        lastTurn = s.turnIdx;
        out.push({
          key: `${s.turnIdx}:${s.seg}:${k}`,
          at: { turnIdx: s.turnIdx, seg: s.seg, tokenIdx: k },
          text: (tokens[k]?.text ?? "").replace(/\n/g, "⏎"),
          opens,
          ordinal: start + k,
        });
      }
    });
    return out;
  });

  const here = (item: Item) => at !== null && item.at.turnIdx === at.turnIdx && item.at.seg === at.seg && item.at.tokenIdx === at.tokenIdx;
  const inReply = (item: Item) => whole && replyRange !== null && item.ordinal >= replyRange[0] && item.ordinal <= replyRange[1];

  /** The track: each turn a stretch of the conversation, in proportion to its tokens. */
  const stretches = $derived.by(() => {
    const out: { turnIdx: number; from: number; size: number; prompt: boolean }[] = [];
    tokenCursor.segments.forEach((s, i) => {
      const prev = out.at(-1);
      if (prev && prev.turnIdx === s.turnIdx) prev.size += s.length;
      else out.push({ turnIdx: s.turnIdx, from: starts[i], size: s.length, prompt: chatLog.turns[s.turnIdx]?.generated !== true });
    });
    return out;
  });
  const pct = (n: number) => `${total > 0 ? (n / total) * 100 : 0}%`;

  const status = $derived.by(() => {
    if (whole) {
      if (!reply || !replyRange) return "";
      const n = replyRange[1] - replyRange[0] + 1;
      return `${roleDisplayLabel(reply.turn.role, reply.turn.roleLabel)} · ${n} ${n === 1 ? "token" : "tokens"}`;
    }
    const token = tokenCursor.token;
    return token ? `${roleDisplayLabel(token.turn.role, token.turn.roleLabel)} · token ${token.ordinal + 1} of ${total}` : "";
  });

  function jumpTo(ordinal: number): void {
    const target = Math.max(0, Math.min(total - 1, ordinal));
    let i = 0;
    while (i + 1 < starts.length && starts[i + 1] <= target) i += 1;
    const s = tokenCursor.segments[i];
    if (s) moveCursor({ turnIdx: s.turnIdx, seg: s.seg, tokenIdx: target - starts[i] });
  }
  function onTrack(event: MouseEvent): void {
    const box = (event.currentTarget as HTMLElement).getBoundingClientRect();
    if (box.width <= 0) return;
    jumpTo(Math.floor(((event.clientX - box.left) / box.width) * total));
  }

  // Keep the cursor's token in the line's view without scrolling the page.
  let line: HTMLElement | null = $state(null);
  $effect(() => {
    void centre;
    void items;
    void tick().then(() => {
      const el = line?.querySelector<HTMLElement>(".t.on") ?? line?.querySelector<HTMLElement>(".t.lit");
      if (!line || !el) return;
      const target = el.offsetLeft - (line.clientWidth - el.offsetWidth) / 2;
      line.scrollTo({ left: Math.max(0, target), behavior: "auto" });
    });
  });

  function onKey(event: KeyboardEvent): void {
    if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
    const target = event.target as HTMLElement | null;
    if (target && (target.closest("input, textarea, select, [contenteditable='true']")
      || target.closest('[role="slider"], [role="radiogroup"], [role="tablist"], [role="listbox"], [role="menu"], [role="dialog"]'))) return;
    switch (event.key) {
      case "ArrowLeft": stepToken(-1); break;
      case "ArrowRight": stepToken(1); break;
      case "Home": segmentEdge("first"); break;
      case "End": segmentEdge("last"); break;
      default: return;
    }
    event.preventDefault();
  }
</script>

<svelte:window onkeydown={onKey} />

{#if total > 0}
  <section class="strip" aria-label="Tokens">
    <div class="bar">
      <button type="button" class="whole" aria-pressed={whole} disabled={!reply} onclick={showWholeReply}>Whole reply</button>
      <span class="steps">
        <IconButton icon="left" label="Previous token" disabled={!canStep(-1)} onclick={() => stepToken(-1)} />
        <IconButton icon="right" label="Next token" disabled={!canStep(1)} onclick={() => stepToken(1)} />
      </span>
      <span class="meta status"><MorphText text={status} /></span>
    </div>
    <div class="line" bind:this={line}>
      {#if (items[0]?.ordinal ?? 0) > 0}<span class="more" aria-hidden="true">…</span>{/if}
      {#each items as item (item.key)}
        {#if item.opens !== null}<span class="seat" aria-hidden="true">{item.opens}</span>{/if}
        <button type="button" class="t" class:on={here(item)} class:lit={inReply(item)} class:prompt={chatLog.turns[item.at.turnIdx]?.generated !== true}
          tabindex="-1" aria-label={`${item.text} (token ${item.ordinal + 1})`} onclick={() => moveCursor(item.at)}>{item.text}</button>
      {/each}
      {#if (items.at(-1)?.ordinal ?? total - 1) < total - 1}<span class="more" aria-hidden="true">…</span>{/if}
    </div>
    <!-- The whole conversation, a turn a stretch: a click lands on the token at that place. -->
    <button type="button" class="track" tabindex="-1" aria-label="The conversation's tokens; choose a place to jump there" onclick={onTrack}>
      {#each stretches as stretch (stretch.turnIdx)}
        <i class:prompt={stretch.prompt} class:lit={whole && reply?.turnIdx === stretch.turnIdx} style:left={pct(stretch.from)} style:width={pct(stretch.size)}></i>
      {/each}
      {#if tokenCursor.token}<b style:left={pct(tokenCursor.token.ordinal + 0.5)}></b>{/if}
    </button>
  </section>
{/if}

<style>
  .strip { display: grid; gap: var(--s2); min-width: 0; }
  .bar { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s3); min-width: 0; }
  .whole { height: 36px; padding: 0 var(--s4); border: 0; border-radius: var(--r-pill); background: transparent; color: var(--ink2); font: var(--type-control); white-space: nowrap; cursor: pointer; transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease); }
  .whole:hover:not(:disabled) { background: var(--wash); color: var(--ink); }
  .whole[aria-pressed="true"] { background: var(--wash2); color: var(--ink); }
  .whole:disabled { cursor: default; opacity: 0.4; }
  .steps { display: inline-flex; gap: var(--s1); }
  .status { min-width: 0; color: var(--ink3); font: var(--type-meta); font-variant-numeric: tabular-nums; }

  .line { display: flex; align-items: center; min-width: 0; height: 40px; overflow-x: auto; overflow-y: hidden; scrollbar-width: none; white-space: pre; }
  .line::-webkit-scrollbar { display: none; }
  .seat { flex: none; margin: 0 var(--s2) 0 var(--s4); color: var(--ink3); font: var(--type-meta); }
  .seat:first-of-type { margin-left: 0; }
  .more { flex: none; padding: 0 var(--s2); color: var(--ink3); font: var(--type-meta); }
  .t { flex: none; height: 32px; padding: 0 1px; border: 0; border-radius: 4px; background: transparent; color: var(--ink2); font: var(--type-body); white-space: pre; cursor: pointer; transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease); }
  .t.prompt { color: var(--ink3); }
  .t:hover { background: var(--wash); color: var(--ink); }
  .t.lit { background: var(--wash); color: var(--ink); }
  .t.on { background: color-mix(in oklch, var(--deep) 22%, transparent); color: var(--ink); }

  .track { position: relative; display: block; width: 100%; height: 16px; padding: 0; border: 0; background: transparent; cursor: pointer; }
  .track > i { position: absolute; top: 6px; height: 4px; border-radius: 2px; background: var(--wash2); transition: background-color var(--fast) var(--ease); }
  .track > i:not(.prompt) { background: color-mix(in oklch, var(--deep) 18%, var(--wash2)); }
  .track > i.lit { background: color-mix(in oklch, var(--deep) 55%, transparent); }
  .track > b { position: absolute; top: 2px; width: 2px; height: 12px; margin-left: -1px; border-radius: 1px; background: var(--ink); transition: left var(--glide) var(--ease); }

  @media (max-width: 720px), (pointer: coarse) {
    .whole { height: 44px; }
    .line { height: 48px; }
    .t { height: 44px; padding: 0 3px; }
    .track { height: 28px; }
    .track > i { top: 12px; }
    .track > b { top: 8px; }
  }
</style>
