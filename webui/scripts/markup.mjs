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
