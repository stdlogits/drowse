// The thread column's way down (store-free, so the node suites load it directly). The column lists
// the active path, then the turns below the active node the way you last came down: choosing an
// earlier turn keeps the reply under it and everything after, dimmed, until you choose one again.

/** Remember, for every node on a path, the child it went on to. */
export function rememberWay(way: Map<string, string>, path: readonly string[]): void {
  for (let i = 0; i + 1 < path.length; i++) way.set(path[i], path[i + 1]);
}

/** The turns after ``from``, down to a leaf: at each node the child you last went on to while it
 *  is still one of its children, else its newest. */
export function wayDown(
  from: string,
  childrenOf: ReadonlyMap<string, readonly string[]>,
  way: ReadonlyMap<string, string>,
): string[] {
  const out: string[] = [];
  const seen = new Set([from]);
  for (let at = from; ;) {
    const kids = childrenOf.get(at) ?? [];
    if (kids.length === 0) return out;
    const kept = way.get(at);
    const next = kept !== undefined && kids.includes(kept) ? kept : kids[kids.length - 1];
    if (seen.has(next)) return out;
    seen.add(next);
    out.push(next);
    at = next;
  }
}
