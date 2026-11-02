// Server-rendered markup as the tests read it. Each removal repeats until nothing changes, since
// one pass can join the halves of a split match into a new one (CodeQL's incomplete
// multi-character sanitization); the tests only read the text, they never serve it.

/** ``html`` without Svelte's hydration comments (``<!--[-->`` and the like). */
export function withoutComments(html) {
  let text = html;
  let previous;
  do {
    previous = text;
    text = text.replace(/<!--[^]*?-->/g, "");
  } while (text !== previous);
  return text;
}

/** ``html``'s text: no comments and no tags. */
export function textOf(html) {
  let text = withoutComments(html);
  let previous;
  do {
    previous = text;
    text = text.replace(/<[^>]*>/g, "");
  } while (text !== previous);
  return text;
}

// Styles as the layout tests read them: no cascade, just each rule with its media.
/** A component's ``<style>`` as rules (see :func:`cssRules`). */
export const styleRules = (source) => cssRules(source.slice(source.indexOf("<style>") + 7, source.indexOf("</style>")));
/** A stylesheet as rules ``{ media, selectors, decls }``, ``media`` the enclosing at-rule's head
 *  (or ``null``); comments dropped, declarations by property. */
export function cssRules(text) {
  const css = text.replace(/\/\*[^]*?\*\//g, "");
  const rules = [];
  (function walk(text, media) {
    let i = 0;
    while (i < text.length) {
      const open = text.indexOf("{", i);
      if (open < 0) break;
      let depth = 1;
      let close = open + 1;
      for (; depth > 0; close += 1) depth += text[close] === "{" ? 1 : text[close] === "}" ? -1 : 0;
      const head = text.slice(i, open).trim();
      const body = text.slice(open + 1, close - 1);
      if (head.startsWith("@")) walk(body, head);
      else {
        const decls = {};
        for (const part of body.split(";")) {
          const colon = part.indexOf(":");
          if (colon > 0) decls[part.slice(0, colon).trim()] = part.slice(colon + 1).trim();
        }
        rules.push({ media, selectors: head.split(",").map(s => s.trim()), decls });
      }
      i = close;
    }
  })(css, null);
  return rules;
}
