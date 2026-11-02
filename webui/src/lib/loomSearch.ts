export function searchableLoomText(text: string): string {
  return text.replace(/\s+/g, " ").trim().toLowerCase();
}

export function loomTextMatches(text: string, query: string): boolean {
  const needle = searchableLoomText(query);
  return needle.length > 0 && searchableLoomText(text).includes(needle);
}

export function loomSearchExcerpt(text: string, query: string): { before: string; match: string; after: string } {
  const readable = text.replace(/\s+/g, " ").trim();
  const needle = searchableLoomText(query);
  const index = needle ? readable.toLowerCase().indexOf(needle) : -1;
  if (index < 0) return { before: readable.slice(0, 150), match: "", after: readable.length > 150 ? "…" : "" };
  const start = Math.max(0, index - 45);
  const end = Math.min(readable.length, index + needle.length + 90);
  return {
    before: (start > 0 ? "…" : "") + readable.slice(start, index),
    match: readable.slice(index, index + needle.length),
    after: readable.slice(index + needle.length, end) + (end < readable.length ? "…" : ""),
  };
}

/** A node's text on one line for a list row, or what stands in for it: a reply that is all
 *  thinking (a thinking model that hit its token budget inside the block) says so, with its
 *  thinking token count when the rows are loaded, rather than reading as an empty turn. */
export function loomNodeSnippet(
  node: { text: string | null; thinking_text: string | null; thinking_tokens?: readonly unknown[] | null },
  empty = "Empty turn",
): string {
  const text = (node.text ?? "").replace(/\s+/g, " ").trim();
  if (text) return text;
  if ((node.thinking_text ?? "").trim()) {
    const n = node.thinking_tokens?.length ?? 0;
    return n > 0 ? `Thinking only · ${n} token${n === 1 ? "" : "s"}` : "Thinking only";
  }
  return empty;
}
