export interface SentenceToken {
  text: string;
}

export interface TokenSentence {
  start: number;
  end: number;
  text: string;
}

const SENTENCE_END = /[.!?…](?:["'”’)\]}]+)?$/u;

/** Group model tokens into human-readable sentence branch points without
 * changing their indexes or text. The end index is always an exact token
 * boundary that can be replayed by the runtime. */
export function splitTokenSentences(
  tokens: readonly SentenceToken[],
): TokenSentence[] {
  const sentences: TokenSentence[] = [];
  let start = 0;

  for (let index = 0; index < tokens.length; index += 1) {
    const text = tokens[index]?.text ?? "";
    const visible = text.trimEnd();
    const endsSentence = SENTENCE_END.test(visible);
    if (!endsSentence) continue;

    sentences.push({
      start,
      end: index,
      text: tokens.slice(start, index + 1).map((token) => token.text).join(""),
    });
    start = index + 1;
  }

  if (start < tokens.length) {
    sentences.push({
      start,
      end: tokens.length - 1,
      text: tokens.slice(start).map((token) => token.text).join(""),
    });
  }

  return sentences;
}

export interface TokenWord {
  /** First and last token of the word, inclusive, in the caller's indexes. */
  start: number;
  end: number;
}

const LEADING_SPACE = /^\s/u;
const TRAILING_SPACE = /\s$/u;

/** The tokens of ``tokens[start..end]`` (inclusive) grouped into the words they spell, so text
 *  drawn one box per token breaks lines only where it has a space. A word starts at ``start``, at
 *  a token that begins with whitespace and after one that ends with it; punctuation stays with
 *  the word before it, and text with no spaces is one word. */
export function tokenWords(
  tokens: readonly SentenceToken[],
  start = 0,
  end = tokens.length - 1,
): TokenWord[] {
  const words: TokenWord[] = [];
  if (start > end) return words;
  let wordStart = start;
  for (let index = start + 1; index <= end; index += 1) {
    if (LEADING_SPACE.test(tokens[index]?.text ?? "") || TRAILING_SPACE.test(tokens[index - 1]?.text ?? "")) {
      words.push({ start: wordStart, end: index - 1 });
      wordStart = index;
    }
  }
  words.push({ start: wordStart, end });
  return words;
}

/** The spaces a word's first token opens with, drawn as text before the word's box so a wrapped
 *  line hangs them at its end instead of starting indented. Only plain spaces ahead of something
 *  visible: a line break stays in its token, and a token of nothing but spaces keeps them. */
export function leadingSpace(text: string): string {
  return /^ +(?=\S)/u.exec(text)?.[0] ?? "";
}

const sentenceCache = new WeakMap<readonly SentenceToken[], TokenSentence[]>();

/** ``splitTokenSentences`` memoized on the token array's identity: saved
 *  replies never change their arrays in place, and the live overlay hands
 *  out a fresh array each frame. */
export function tokenSentences(tokens: readonly SentenceToken[]): TokenSentence[] {
  let sentences = sentenceCache.get(tokens);
  if (!sentences) sentenceCache.set(tokens, sentences = splitTokenSentences(tokens));
  return sentences;
}
