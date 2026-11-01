<script lang="ts">
  // A value that can change on every decoded token: the status line's counts and the rack's
  // readings. While a reply streams it changes faster than a letter morph plays, and each morph
  // (Torph's per-letter animations, then `getAnimations` over the page's animations) costs the frame
  // that the next token needed: with the rack and the status line morphing, the page read about 15
  // token frames a second, fell behind a 50 tok/s decode, and the server closed its stream (1013)
  // and cut the reply. So the value is plain text while a reply streams and morphs once it settles.
  import { MorphText } from "@stdlogits/ledger";
  import { genStatus } from "../stores/chat.svelte";

  let { text, numbers = true }: { text: string | number; numbers?: boolean } = $props();
</script>

{#if genStatus.active}<span class="live">{text}</span>{:else}<MorphText {text} {numbers} />{/if}
