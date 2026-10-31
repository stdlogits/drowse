# web/ — the dashboard

`drowse/web/` is the Python half of the dashboard. `routes.py` mounts the committed bundle
in `dist/`, and `__init__.py` re-exports `register_web_routes` and `dist_path`. The Svelte 5
+ Vite source lives in `webui/` at the repo root, which has no AGENTS.md of its own, so this
file covers it too. For the route catalogue, WS frame fields and auth policy, see
`drowse/server/AGENTS.md`.

## Mount (`routes.py`)

`create_app(..., web=True)` registers the mount after every API route, so its catch-all
can never shadow `/v1/*`, `/api/*` or `/drowse/v1/*`. `drowse serve` mounts by default and
`--no-web` opts out; `create_app` defaults to `web=False`. Mount once per app: a second
call adds shadowed duplicates without raising.

`/assets` and `/icons` are `StaticFiles` mounts, and `GET /` serves `index.html`. The
catch-all `GET /{full_path:path}` returns 404 for `api`, `v1`, `drowse` and anything under
them. Otherwise it serves top-level dist files from an allowlist built at mount time, and
falls back to `index.html`. Because `full_path` is only ever used as a dict key, `..`
traversal is impossible.

Only split JS chunks are content-hashed. `assets/drowse.js` and all CSS, font and JSON
assets keep fixed names, so StaticFiles' ETag/Last-Modified revalidation is load-bearing.
Never add an immutable cache header. `dist_path()` resolves through `importlib.resources`.
If `dist/index.html` is missing, mounting raises `WebUINotBuilt` (500).

The shell is public and holds no data. `server/security.py` authenticates only the API
prefixes, sets CSP and anti-framing headers on every response, and marks API responses
`no-store`. The CSP's `script-src 'self'` bans inline scripts, which is why the theme
bootstrap is the separate file `/theme-preflight.js`: Ledger's pre-paint script, which
`vite.config.ts` serves in dev and writes into the build straight from `@stdlogits/ledger`
(index.html loads it with `data-ledger-theme-key="drowse.theme"`). Its `connect-src` allows only self,
`https://www.neuronpedia.org` (SAE descriptions) and `https://drowse.ai/api/contact`, so
any new third-party fetch needs a CSP entry.

## Build and the committed bundle

```bash
cd webui && npm ci   # Node 24.x
npm run dev          # :5173, proxies /drowse (incl. WS), /v1, /api to :8000
npm run check        # all webui gates
npm run build        # vite build into ../drowse/web/dist + check-build-isolation
```

- `drowse/web/dist/` is committed package data and is exactly what wheel users run.
  Rebuild and commit it in the same change as anything the bundle is built from:
  `webui/src`, `public`, `index.html`, `vite.config.ts`, `scripts/dashboard-assets.mjs`,
  or the npm deps. CI's `webui` job runs `npm ci`, `check` and `build`, then fails on any
  `git status` entry under `dist/`, untracked files included. `emptyOutDir` wipes `dist/`
  on every build, so never put files there by hand.
- `webui/public/` also holds drowse.ai's brand set (`social/`, `icons/`, the favicons and
  the Fluent licence). `scripts/dashboard-assets.mjs::shipsInDashboard` keeps all of it out
  of the dashboard, whose tab icon is the static drowse mark (`drowse-mark-{light,dark}.svg`:
  a tile on the depth ramp, a paper-coloured closed eye).
  `scripts/check-build-isolation.mjs` fails the build on hosted leaks (`sw.js`,
  `manifest.webmanifest`, `_headers`, wasm, MLC/WebLLM files, non-allowlisted public
  files, a PWA bootstrap), a missing allowlisted icon or SAE description index, or an
  `index.html` reference the build omits.
- The SAE description indexes in `lib/data/sae-descriptions/` ship on purpose (as
  `assets/gemma-3-{270m,1b,4b}-it.json`): they label local SAELens gemma-scope-2 releases
  offline. Regenerate them with `node scripts/build-sae-descriptions.mjs`.
- CI's `build` job seeds a stale `build/lib` cache, then runs
  `scripts/check_package_isolation.py check`. The wheel's dashboard files must equal the
  sdist's and the source tree's.
- `npm run check` runs `svelte-check`, `check-theme.mjs` (every referenced CSS custom
  property must be declared, here or in `@stdlogits/ledger`), `check-ledger.mjs` (the Ledger
  ratchet, below), `check-runtime-boundary.mjs`, the `test:*` node suites and
  `check-test-coverage.mjs`. The last requires every `scripts/*.test.mjs` to be reachable
  from `check`, so a new test file needs a line in `package.json`.
- `check-ledger.mjs` counts, per file, the uses of the bridge's old token names
  (`lib/style/bridge.css`) and the departures from Ledger: borders; outlines outside a
  `:focus`/`:focus-visible` rule; shadows other than a marked token's 2px same-colour spread
  (the tip's `--float` is Ledger's own, so any use here counts); monospace; capitals;
  letter-spaced text; easing off the one curve; animations that loop; colours outside the
  tokens; colour mixes outside oklch (the accent's) and oklab (a grey's: Chrome renders a
  near-grey mixed in oklch with a red cast); and text coloured by the depth accent (`--tint`,
  `--deep`, `--late` or a bridge name for them). It reads `style:prop={…}` directives as
  `prop: …` declarations. It fails if any count differs from `webui/ledger-ratchet.json`. A
  count that rises fails; one that falls asks for `node scripts/check-ledger.mjs --write`,
  which lowers the ratchet and never raises it. A category the ratchet does not record yet
  fails the check until `--write` takes today's count as its first ceiling. `--cut` (the 6.0
  cut) requires every count at zero and the bridge deleted.

## Generated REST types (`lib/types.gen.ts`)

Never edit it by hand. `scripts/generate_webui_types.py` builds `create_app(MagicMock())`,
with no model and no network, and renders TS for the **response** schemas of every
`/drowse/v1/*` route plus everything those schemas reference.

To change a response shape, edit its TypedDict (`server/response_models.py`, or
`readout/measurements.py` for the envelope), then run
`python scripts/generate_webui_types.py`. `--check` diffs without writing.
`SCHEMA_RENAMES` maps engine names to dashboard names (`Measurements` →
`MeasurementsEnvelopeJSON`, …), and every other schema keeps its Python name.
`PROPERTY_OVERRIDES` types the deliberately opaque payloads: loom token rows, fit
diagnostics, manifold domains.

`lib/types.ts` is hand-written: request bodies, WS and SSE frames, and client-local types.
Hand-writing requests is safe because `extra="forbid"` makes request drift a loud 400.
Response drift would fail silently, which is why responses are generated. `types.ts`
re-exports `types.gen.ts` and is the only module that imports it.

CI's `typecheck` job runs `--check`. `tests/invariants/test_webui_types_parity.py` also
requires a declared response schema on every non-streaming native route, and kebab-case
native paths. One gotcha: the script imports `drowse` from the environment, not from its
own checkout, so in a worktree run it from the repo root with `PYTHONPATH=.`.

## Talking to the server

- The client always uses the literal session id `default`. `lib/api.ts` has methods only
  for routes the dashboard calls, so a missing method does not mean the server lacks the
  route.
- The key lives only in memory (`lib/runtime/http-auth.ts`). It is read once from
  `<meta name="api-key">` if an embedding page supplies one; drowse never injects that tag.
  Otherwise the API-access drawer sets it via `setApiKey`. HTTP sends
  `Authorization: Bearer`. The WebSocket offers the subprotocols `drowse.v1` and
  `drowse.auth.<base64url-utf8-key>`. Never put a credential in a URL.
- The WS stream is `/drowse/v1/sessions/default/stream`. `submit` is the composer's one
  door: authored and generated roles are independent, `generated_role: null` appends
  without generating, and `text: null` continues from the selected leaf. `generate`
  covers continuation, explicit message replay (the A/B shadow), the token fork and
  `generate_seat`. The third client frame is `stop`. Server frames are `started`,
  `token`, `done`, `error`, `request_complete` and `tree_mutated`. The shared types also
  carry browser-runtime extensions: the `generation_progress` frame, which Python never
  emits, and `position_spans`/`position_indices`, which Python's `extra="forbid"` rejects.
- An error frame's `code` (SSE, polled or WS) is the exception class name (`RequestError`,
  `PoisednessError`, …). Render failures with `describeError(e)` from
  `lib/runtime/errors.ts`.

**Measurements.** The read side has one record, the versioned `measurements` envelope from
`drowse/readout/measurements.py`. Its scope is `"token"` on each measured `token` frame,
where it is the same object the loom token row stores. It is `"aggregate"` on
`done.result.measurements`, and `"replay"` (with `provenance: "replayed"`) from
`.../instruments/{family}/token-readout`.

`mergedReadings` folds the three families' readings into one name-keyed map. Branch on a
reading's shape, never on probe-info flags: `isScalarReading` (the reading has a `value`)
means a lens or SAE reading, and geometry readings are `ProbeReadingJSON`.
`SessionInfo.instruments` is the only representation of instrument state.

Hover and the drilldown read the envelopes the loom owns, so they survive refresh and
save/load. A channel that was off during the generation is replayed through
`runtime/tokenReadoutCache.ts` (keyed, deduplicated). Replayed values are never written
back and never relabelled as captured.

## The drowse.ai seam

drowse.ai ([stdlogits/drowse.ai](https://github.com/stdlogits/drowse.ai)) pins this repo as
a submodule and builds `webui/src` with a WebGPU runtime in place of the HTTP client. Its
local Vite config (used by the ad-hoc servers in its test scripts) imports this repo's
`webui/vite.config.ts`, whose default export must stay a function of the command.

Anything drowse.ai imports or reads is its interface. That covers shared modules it
imports by path (the stores barrel, `lib/types.ts`,
`lib/runtime/{contracts,registry,services}.ts`, `lib/theme.ts`, `lib/brandMigration.ts`,
`lib/ui/*`, …) and the component source text and CSS class names its tests read. Before
moving or renaming any of these, grep drowse.ai's `webui/{src,hosted,scripts,e2e}`,
`browser-runtime/` and `tests/`, and land its side with the submodule bump.

**RuntimeClient.** The contract lives in `lib/runtime/contracts.ts`. It has a `mode`
(`http` | `browser` | `fake`); seven services (`sessions`, `profiles`, `probes`,
`manifolds`, `templates`, `tree`, `instruments`); an `events` channel; and `dispose()`.
`RUNTIME_SERVICE_METHODS` is compile-checked against the interfaces. A member a runtime may
lack is optional (`?`) with an entry there and a check the UI tests before showing it:
`instruments.tokenAnatomy?` and `instruments.unitActivity?` (the node replay), reported by
`replayCapabilities(instruments)`; the unit reads (`unitRead?` and its siblings, by
`unitReadsAvailable`); and `unitLabels?`, the catalog's labels for the token anatomy's rows.
The token view shows the anatomy (`drawers/token/TokenAnatomy.svelte`, queued with the
token readouts in `runtime/tokenReadoutCache.ts`) where `tokenAnatomy` exists, and the lens
readout (`LensReading.svelte`) where it does not.

`main.ts` installs `HttpRuntimeClient` into `runtime/registry.ts` before it imports `App`.
The registry refuses a second runtime. drowse.ai also installs `RuntimeCapabilities` and a
`HostedController` there.

Shared code reaches the runtime only through `runtimeClient` or `runtime/services.ts`.
`check-runtime-boundary.mjs` is AST-based, so it also covers `.svelte` scripts and dynamic
imports; it fails if any file other than `runtime/http-client.ts` imports `lib/api.ts`.

`DRAWER_REQUIREMENTS` (`runtime/ui-capabilities.ts`) gates drawers on capability
operations. With no capabilities installed, as in the HTTP build, everything is available.

**Aliases.** Declare each alias in both `vite.config.ts` and `tsconfig.json`. Add one only
when the two builds need different components at one import site.

| alias | here | drowse.ai |
|---|---|---|
| `@runtime-jlens-source` | `panels/JLensSourceSection.svelte` | `src/hosted/ui/JLensSourceSection.svelte` |
| `@runtime-sae-source` | `panels/SaeSourceSection.svelte` | `src/hosted/ui/SaeSourceSection.svelte` |
| `@runtime-jlens-missing` | `drawers/token/JLensMissingState.svelte` | `src/hosted/ui/JLensMissingState.svelte` |
| `@runtime-session-drawer` | `drawers/SessionAdminDrawer.svelte` | this repo's `UnavailableRuntimeDrawer.svelte` |
| `@runtime-local-drawer` | `drawers/UnavailableRuntimeDrawer.svelte` | `src/drawers/LocalRuntimeDrawer.svelte` |

**Hosted-only code** lives in drowse.ai's `webui/src/hosted/`: the WebGPU runtime and
workers, the catalog and downloads, the PWA, the hosted shell and onboarding, and its own
steering parser and archive mirror. This tree carries only the contracts, so never import
WebGPU, MLC or worker code here. The hosted-only pieces still in this tree must not grow:
`lib/shell.ts`, `lib/homeController.ts`, `lib/webmcp/{hostedTools,homeTools}.ts`,
`runtime/gpuRecovery.ts` and the WebGPU/catalog rows of `runtime/userFacingError.ts`, the
catalog/download/worker half of `runtime/contracts.ts`, and the hosted session store in
`lib/sessionPersistence.ts`.

**`lib/brandMigration.ts` stays.** drowse.ai users still hold browser data under the
earlier names `polythetic` and `saklas`: localStorage keys, IndexedDB databases, the OPFS
root, and record fields such as `saklas_version`. The module migrates that data
opportunistically and never blocks the current store. `lib/theme.ts` (the stored theme),
`lib/chatBackup.ts` and `runtime/localData.ts` recognize the old names for the same reason.

## Source layout (`webui/src/`)

```
main.ts / App.svelte  runtime install + mount / the shell: header, rooms, dialog host
panels/workbench/     Workbench (Thread | Page | Inspector, the phone's sheet), Thread
panels/inspector/     Inspector (by selection), Instruments, TermView, UnitView (+ UnitTrace)
panels/rooms/         RoomSections (Manifolds, Model), LensRoom, UnitsRoom (+ UnitAtlas)
panels/               Chat (the Page), RawBuffer, RecipeBar (the recipe chips), SamplingStrip,
                      ProbeRack, JLensPanel, SaePanel, CommandPalette; loom/, rack/
drawers/              one component per DrawerName; index.ts registry; manifold/, token/
lib/api.ts            HTTP + SSE + WS client (imported only by runtime/http-client.ts)
lib/runtime/          RuntimeClient contract, registry, HTTP impl, errors, capability gating
lib/types.ts          the import surface (re-exports types.gen.ts)
lib/stores.svelte.ts  barrel over lib/stores/ slices
lib/expression.ts     rack → steering expression (serialize only)
lib/highlight.ts      the one per-token highlight; tokens.ts is its store-free ramp math
lib/templates.ts      client mirror of the template validator
lib/manifolds/        fitKind.ts (the one flat/curved classifier), diagnostics
lib/units.ts          unit IDs, reads, activity and atlas arithmetic (store-free); stores/units
lib/unitTerms.ts      a unit's recipe terms and their one validity rule (import-free)
lib/webmcp/           WebMCP tools, loaded only when document.modelContext exists
lib/style/            drowse.css (the shell's Ledger classes), global.css, bridge.css (temporary)
lib/theme.ts          createTheme({ key: "drowse.theme" }); lib/ui/ holds shared primitives
```

The design is Ledger, from the package `@stdlogits/ledger` (pinned as a git dependency at a
tag; its `DESIGN.md` is the design, and it supplies the tokens, the Wix Madefor faces, motion,
the tip, the theme and the pieces: Segmented, Tabs, Switch, IconButton, Icon, Expander, Meter,
Token, Context, Swap, MorphText, ThemeButton). Use a piece rather than drawing a look-alike;
the dev server leaves the package out of dependency optimization (`optimizeDeps.exclude`)
because it ships TypeScript sources.

## State

`lib/stores/` holds one `$state` slice per concern. `lib/stores.svelte.ts` re-exports the
slices that stream, loom, chat and instruments share. Narrower slices (`selection`,
`rooms`, `workbench`, `manifoldJobs`, `preparations`, …) are imported directly.

The shell's own state is three small slices. `rooms.svelte.ts` is the route (`#/lens`,
`#/manifolds/packs`): the room, its section and in-memory params for the tool it hosts.
`selection.svelte.ts` is what the Inspector shows (none, token, probe, term, unit).
`workbench.svelte.ts` is what the Page region shows (the conversation or the loom) and the
phone's sheet.

The workbench has three layouts. From 1240 px the three regions sit side by side and each
scrolls on its own. From the phone up to 1240 px it is Page | Inspector, with the thread
folded into a row over the page, so a selection always shows beside the conversation. On a
phone (720 px and under) it is one column: the Inspector is a sheet between the log and the
composer, and the loom page takes the column (no sheet while it shows). The SAE source picker
lives in the Lens room beside the lens sources; the Inspector's Features shows a "Choose an
SAE" button that leads there until an SAE is loaded.

Slices import each other directly, never through the barrel. The import graph has cycles
(loom ↔ ws, chat ↔ pending). These are safe only because every cross-slice reference sits
inside a function body, so no module-level initializer may read an imported binding.

`$state` does not track `Map.set`, `Set.add`, or writes inside stored objects. Shared
collections are therefore `SvelteMap`/`SvelteSet`, and mutators replace the value
(`entries.set(name, { ...e, coords })`). An in-place `entry.current = v` on the per-token
probe path would freeze every sparkline for the whole generation.

`createPreparationSlice` builds the four background jobs (lens `fetch`/`fit`, SAE
`fetch`/`train`), which share one poll loop. A family's two operations share one server
resource, so `check()` resumes only its own operation's job.

## Drawers

The registry is `drawers/index.ts::DRAWERS: Record<DrawerName, DrawerEntry>`. Each row
has a lazy `load()` (rendered by `DrawerOutlet` and cached after the first open), optional
fixed `params` (how one `RackDrawer` serves both `subspace` and `manifolds`), `narrow`,
and either a `launcher` or `launcher: null` with a `via` string naming the surface that
opens it. `RAIL_CATEGORIES` (the command palette's tools) is derived from the launchers.

`openDrawer(name, params)` (`lib/stores/drawers.svelte.ts`) is the one entry point, and where
a tool opens depends on what it is. `token_drilldown` and `probe_inspector` are selections,
shown in the workbench's Inspector. The tools a room hosts (`ROOM_SECTIONS` in
`lib/stores/rooms.svelte.ts`: the manifold builder, packs, merge, template lab and roles in
Manifolds; health, the local runtime and API access in Model) go to their room and section,
where their close returns to the workbench. Everything else opens as a dialog over the page.

To add a drawer, add its name to `DrawerName` in `lib/types.ts` and add three rows: one in
`DRAWERS`, one in `DRAWER_REQUIREMENTS`, and its WebMCP coverage entry in
`lib/webmcp/catalogue.ts::INTERFACE_CATALOGUE`. All three are `Record<DrawerName, …>`, so
the compiler enforces them; nothing else is needed. A new runtime service method likewise
needs a `RUNTIME_SERVICE_CATALOGUE` entry in the same file. drowse.ai's tests regex-parse the
`DRAWERS` rows (`  name: {`) and pin their count, so adding or removing a drawer touches
drowse.ai too.

## Surface invariants

- **Expression.** `lib/expression.ts` only serializes; the server's grammar is the only
  parser. Terms come out as subspace terms (at the shared `subspaceAlong`), `jlens/`
  atoms, `sae/` atoms, then manifold terms (at their own `along[,onto]`). Keep `Variant`
  in step with the grammar (`raw`, `sae`, `sae-*`, `role`, `role-*`, `from`, `from-*`).
  `applyCustomSteeringExpression` clears the visual rack, because the rack cannot show
  every projection or gate form and must not misstate what generation uses.
- **Rack families.** `fitKind.ts::fitGeometry` (`resolved_fit_mode ?? fit_mode`) maps
  `pca`/`baked` to subspace, `spectral`/`authored` to manifold, and an unfitted `auto` to
  unresolved. Subspace terms share one `subspaceAlong` because the composer merges them
  into one affine subspace that slides once. J-lens and SAE atoms carry their own α
  (`ATOM_DEFAULT_ALPHA = 0.3`): a single token or feature direction runs hotter than a
  concept vector, and α ≥ 0.5 over-steers into repetition. A unit term (`mode: "unit"`,
  keyed by the full `neuron/…` or `head/…` reference) is one per unit: a neuron pushed
  (signed α, 0.2 by default; below zero its − pole) or a neuron or head ablated (`!…`,
  written bare at α 1, the unit held at its neutral mean). A head has no push. Switching a
  term between push and ablation takes that operation's own α (`setUnitOperation`), since a
  push's α kept as an ablation's would blend away from the mean and a negative one amplify;
  `unitTerms.ts::unitTermProblem` refuses a negative ablation α and a head push in the
  snapshot validator and in WebMCP's rack edit alike.
- **Units.** The unit view (selection `{ kind: "unit", id }`) and the Units room read
  through `lib/stores/units.svelte.ts`: plain getters inside `$derived`, the matching
  `request*` from an `$effect`, nothing asked while a generation runs. Both hide where
  `unitReadsAvailable(runtimeClient)` is false (`shownRooms()` drops the room). A neuron
  reads one side at a time, and both views read the side that fired here (`peakPole`, the
  sign of its trace's peak): the view's writes, label and link, the room's summary. Heads
  rank by their own write at the atlas's token (`rankHeads` over the population), never
  against the neurons' peaks, which on SmolLM2 hold no head. Unit probes are the `units`
  family: they list in the probe rack (`UnitProbeCard`, a row like a geometry probe's, detached
  from the unit view it selects) and their readings merge with the other families'
  (`mergedReadings`).
- **Templates.** `validateTemplateDraft` mirrors `drowse/store/templates.py`
  (`_validate_body`, `_validate_context`, `_slug_value`, `_LABEL_REGEX`) rule for rule.
  Change both together.
- **Pending queue.** A submit during a generation, or behind queued items, is deferred,
  never raced. Rack mutations go through `enqueueOrApply` and coalesce under `coalesceKey`
  `"rack"`. The queue drains FIFO, one item per WS `done`/`error`. In `RawBuffer`, an edit
  never overwrites text: it forks a branch at the first changed character
  (`resolveDivergence`).
- **Persistence.** The server loom tree is authoritative and is never first-painted from a
  browser cache, because node ids go stale when the server restarts.
  `ConversationSnapshotV7` is validated exactly and backs both the IndexedDB library
  (written by `ConversationAutosave` while no generation runs) and the backup files
  (`lib/chatBackup.ts`). localStorage holds presentation preferences only, via
  `safeLocalStorage*`.
- **Visual.** Ledger: flat and borderless (fills only: paper, wash, wash2), the one shadow
  under the tip, Wix Madefor only with tabular numbers, no capitals, one easing curve.
  Colour is depth: the one accent is `--tint`, a layer's place on the `--deep` → `--late`
  ramp, and it only fills; signed negatives are ink at low alpha, interaction is ink. The
  components not yet redrawn read old token names through `lib/style/bridge.css`, which the
  ratchet counts down. Token tints are the depth accent (or ink, for a negative) mixed with
  transparent in oklch up to 22%, with a 2px spread in the same mix. Popups, menus and toasts
  are wash2 fills with no shadow; a dialog and the command palette are paper over a wash2 scrim.
- **A departure from the canvas, for a9 to ratify.** A selected steering term opens in the
  Inspector (`panels/inspector/TermView.svelte`, the rack's controls) rather than in place
  under the chips as the Steering artboard draws it (a wash surface with Strength, When, Where
  and the live expression). The in-place term editor is the term-editor lane's.

## Testing

The node suites load modules through Vite SSR (`ssrLoadModule`) with a stub
`installRuntimeClient`. `saeDescriptions.ts` and `tokens.ts` are also imported by plain
Node (here or in drowse.ai), so they must not use extensionless value imports.
`tests/web/test_web.py` covers the mount: files served as files (the marks, the theme
preflight), the 404 for a missing asset, the SPA fallback, traversal, a public shell under an
API key, `web=False`, and an empty dist. `tests/invariants/test_webui_types_parity.py`
gates the generated types, and `test_layering` places `web` in the top layer.

## Deliberately absent

The dashboard has no client-side expression parser; drowse.ai's parser is parity-tested
against `tests/fixtures/browser_steering_expression.json`. WS client frames have no
commit or prefill fields, and nothing flat sits beside `measurements`. The rack has no
`~`/`|` authoring. There is no session switching and no persistent credential storage.
