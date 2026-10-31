<script lang="ts">
  // Help: the one page where sentences belong. Where things are (the rooms), the words the
  // dashboard uses, then the keys and the steering grammar folded away. Rows on paper, no cards.
  import { Expander, IconButton } from "@stdlogits/ledger";
  import { closeDrawer } from "../lib/stores.svelte";
  import { getRuntimeManifoldFitMaxIntrinsicDim } from "../lib/runtime/registry";
  import { CONTACT_ADDRESS } from "../lib/contact";

  // The platform's modifier, so the key hints are right on Linux and Windows too. ``navigator``
  // may be undefined in non-browser tests: fall back to the Mac's.
  const mod =
    typeof navigator === "undefined" || /Mac|iPhone|iPad|iPod/.test(navigator.platform) ? "⌘" : "Ctrl";
  let shortcutsOpen = $state(false);
  let syntaxOpen = $state(false);
  const manifoldFitMaxDim = getRuntimeManifoldFitMaxIntrinsicDim();

  const places = [
    { name: "Workbench", what: "The thread you are on, the conversation, and whatever you select: a token layer by layer, a probe, a steering term. With nothing selected, the instruments." },
    { name: "Manifolds", what: "Build, install, combine and test the concepts you steer with and read by, and the roles that speak." },
    { name: "Units", what: "The model's neurons and heads, by layer." },
    { name: "Lens", what: "Which lens and which SAE read each layer, and where they come from: fetched, fitted or trained." },
    { name: "Model", what: "Health, the settings reset, and API access." },
    { name: "Loom", what: "Every branch of the conversation. Weave shows your text beside its alternatives; Map shows where paths divide (drag to move, pinch to zoom)." },
  ];

  const terms = [
    { term: "Token", def: "One unit of text for the model: a word, part of one, or punctuation." },
    { term: "Token surprisal", def: "How unlikely the chosen token was after the sampling settings. It measures a word choice, not whether the reply is true." },
    { term: "Sampler entropy", def: "How spread out the next-token probabilities were after the sampling settings. Restrictive settings can make it low even when the reply is wrong." },
    { term: "Direction", def: "An internal pattern used to steer a reply toward or away from a trait." },
    { term: "Reading", def: "A measurement of an internal pattern. It observes the reply without changing it." },
    { term: "J-lens", def: "A fitted projection of a layer's activations into token probabilities. Aggregate strength is the mean across fitted layers, not the sampler's probability or a record of the model's thoughts." },
    { term: "SAE feature", def: "A learned activation pattern. Its label is an interpretation, not proof of a concept or cause. Normalized strength compares activation with a reference maximum and can exceed 1." },
    { term: "Captured or replayed", def: "Captured readings were recorded during the original run. Replayed ones are computed later, with their source and steering beside them; an unsteered replay is a separate measurement." },
  ];

  const keys = [
    { group: "Anywhere", rows: [
      { keys: [mod, "K"], what: "Searches everything" },
      { keys: ["Esc"], what: "Closes a dialog, stops a generation, or clears the selection" },
    ] },
    { group: "Composer", rows: [
      { keys: ["Enter"], what: "Sends the turn" },
      { keys: ["Shift", "Enter"], what: "Starts a new line" },
    ] },
    { group: "Workbench", rows: [
      { keys: [mod, "R"], what: "Generates another answer" },
      { keys: [mod, "E"], what: "Edits the current turn" },
      { keys: [mod, "B"], what: "Branches from the current turn" },
      { keys: [mod, "N"], what: "Opens the map navigator" },
      { keys: [mod, "D"], what: "Deletes the branch and its continuations" },
    ] },
    { group: "Loom, while it has focus", rows: [
      { keys: ["j", "k"], or: true, what: "Moves between alternatives" },
      { keys: ["h", "l"], or: true, what: "Moves to an earlier turn or its first continuation" },
      { keys: ["Enter"], what: "Opens the branch" },
      { keys: ["s"], what: "Stars the turn" },
      { keys: ["n"], what: "Adds a note" },
      { keys: ["/"], what: "Searches the conversation" },
    ] },
  ];

  const examples = [
    { expr: "0.3 honest", what: "A light push toward honesty" },
    { expr: "0.4 warm@response", what: "Warmth only during the reply" },
    { expr: "0.5 personas%pirate", what: "The pirate point on the persona manifold" },
    { expr: "!sycophantic", what: "The sycophantic direction removed" },
    { expr: "0.3 a + 0.5 b", what: "Two directions together" },
  ];

  // The grammar as rows (name, then its forms), so it lines up without a monospace face.
  const grammar = [
    { name: "expr", forms: ["term ((\"+\" | \"-\") term)*"] },
    { name: "term", forms: ["[coeff \"*\"?] [\"!\"] selector [\"@\" trigger]"] },
    { name: "selector", forms: ["atom ((\"~\" | \"|\") atom | \"%\" position)?"] },
    { name: "position", forms: ["signed_num (\",\" signed_num)* | label"] },
    { name: "atom", forms: ["[ns \"/\"] NAME [\".\" NAME] [\":\" variant] | \"sae/\" INT"] },
    { name: "trigger", forms: ["before | after | both | thinking | response", "prompt | generated | when:<probe><op><num>"] },
    { name: "variant", forms: ["raw | sae | sae-<release>", "role-<name> | from-<source>"] },
  ];
</script>

<section class="help" aria-label="Help drawer">
  <header class="dlg-head">
    <h2>Help</h2>
    <IconButton icon="close" label="Close" onclick={closeDrawer} />
  </header>

  <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard access for overflow content) -->
  <div class="body" role="region" aria-label="Help reference" tabindex="0">
    <section class="dw-sec" aria-labelledby="help-places">
      <span class="head" id="help-places">Where things are</span>
      <dl class="rows">
        {#each places as place (place.name)}
          <div class="row"><dt>{place.name}</dt><dd>{place.what}</dd></div>
        {/each}
        {#if manifoldFitMaxDim !== null}
          <div class="row"><dt>Browser fitting</dt><dd>Up to {manifoldFitMaxDim} dimensions; larger fits need the Python server.</dd></div>
        {/if}
      </dl>
    </section>

    <section class="dw-sec" aria-labelledby="help-terms">
      <span class="head" id="help-terms">Words</span>
      <dl class="rows">
        {#each terms as row (row.term)}
          <div class="row"><dt>{row.term}</dt><dd>{row.def}</dd></div>
        {/each}
      </dl>
    </section>

    <div class="folds">
      <Expander bind:open={shortcutsOpen} label="Keyboard shortcuts">
        <div class="keys">
          {#each keys as group (group.group)}
            <section class="dw-sec">
              <span class="head">{group.group}</span>
              <div class="rows">
                {#each group.rows as row (row.what)}
                  <div class="row">
                    <span class="combo">{#each row.keys as key, i (i)}{#if i > 0}<span class="join">{row.or ? "/" : "+"}</span>{/if}<kbd>{key}</kbd>{/each}</span>
                    <span class="what">{row.what}</span>
                  </div>
                {/each}
              </div>
            </section>
          {/each}
        </div>
      </Expander>

      <Expander bind:open={syntaxOpen} label="Steering expressions">
        <div class="syntax">
          <section class="dw-sec">
            <span class="head">Examples</span>
            <div class="rows">
              {#each examples as row (row.expr)}
                <div class="row"><code class="expr">{row.expr}</code><span class="what">{row.what}</span></div>
              {/each}
            </div>
          </section>
          <section class="dw-sec">
            <span class="head">Grammar</span>
            <dl class="rows grammar" aria-label="Steering expression grammar">
              {#each grammar as rule (rule.name)}
                <div class="row">
                  <dt><code class="expr">{rule.name}</code></dt>
                  <dd class="forms">{#each rule.forms as form (form)}<code>{form}</code>{/each}</dd>
                </div>
              {/each}
            </dl>
          </section>
        </div>
      </Expander>
    </div>

    <p class="contact">Questions or feedback: <a class="prose-link" href={`mailto:${CONTACT_ADDRESS}`}>{CONTACT_ADDRESS}</a></p>
  </div>
</section>

<style>
  .help { display: flex; flex-direction: column; height: 100%; min-height: 0; color: var(--ink2); }
  .dlg-head { display: flex; flex: none; align-items: center; justify-content: space-between; gap: var(--s4); padding: var(--s5) var(--s5) 0; }
  .dlg-head h2 { margin: 0; color: var(--ink); font: var(--type-mark); }
  .body { display: grid; flex: 1 1 auto; grid-template-columns: minmax(0, 1fr); gap: var(--s6); align-content: start; min-height: 0; padding: var(--s5); overflow-y: auto; overscroll-behavior: contain; }
  .body:focus-visible { outline-offset: -2px; }

  /* A name and what it is, on one line where there is room. */
  .rows { display: grid; gap: var(--s2); margin: 0; }
  .row { display: grid; grid-template-columns: 160px minmax(0, 1fr); gap: var(--s1) var(--s4); align-items: baseline; }
  .row dt { color: var(--ink); font: var(--type-control); }
  .row dd { margin: 0; color: var(--ink2); font: var(--type-small); text-wrap: pretty; }
  .what { color: var(--ink2); font: var(--type-small); }

  .folds { display: grid; gap: var(--s2); }
  .keys, .syntax { display: grid; gap: var(--s5); padding-bottom: var(--s2); }
  .combo { display: inline-flex; flex-wrap: wrap; align-items: baseline; gap: var(--s1); }
  .join { color: var(--ink3); font: var(--type-meta); }
  kbd { display: inline-block; min-width: 24px; padding: 0 var(--s2); border-radius: var(--r-chip); background: var(--wash); color: var(--ink); font: var(--type-meta); text-align: center; }
  code { font: inherit; }
  .expr { color: var(--ink); font: var(--type-control); overflow-wrap: anywhere; }
  .forms { display: grid; gap: 2px; min-width: 0; color: var(--ink2); font: var(--type-small); overflow-wrap: anywhere; }
  .grammar .row { grid-template-columns: 96px minmax(0, 1fr); }

  .contact { color: var(--ink3); font: var(--type-meta); }

  @media (max-width: 720px) {
    .dlg-head { padding: var(--s4) var(--s4) 0; }
    .body { padding: var(--s4); }
    .row { grid-template-columns: minmax(0, 1fr); }
    .grammar .row { grid-template-columns: 80px minmax(0, 1fr); }
  }
</style>
