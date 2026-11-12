# Drowse web UI

Svelte 5 + Vite source for the dashboard served by `drowse serve`. Production
builds go directly to `../drowse/web/dist/`; that committed directory is package
data and is the UI users receive from the wheel.

## Development

```bash
cd webui
npm ci

# In another terminal:
drowse serve <model>

npm run dev
```

Vite runs on `http://localhost:5173` and proxies `/drowse`, `/v1`, and `/api`
(including the native WebSocket) to `http://localhost:8000`.

Before committing a UI change:

```bash
npm run check
npm run build
git diff --exit-code ../drowse/web/dist
```

`npm run check` runs Svelte/TypeScript checks, the theme-token validator and the Ledger ratchet.
`npm run build` wipes and regenerates the committed production bundle. CI repeats
both commands and fails if the rebuilt bundle differs.

### The hosted build

[drowse.ai](https://drowse.ai) builds this same source with a WebGPU runtime in
place of the HTTP client. That shell, its runtime, and its tests live in
[stdlogits/drowse.ai](https://github.com/stdlogits/drowse.ai), which pins this
repository as a submodule. The `@runtime-*` aliases in `vite.config.ts` are the
seams it swaps; keep shared code behind `lib/runtime/client.ts` (enforced by
`scripts/check-runtime-boundary.mjs`) rather than importing the HTTP client.

## Application shape

The header holds the rooms as routes: Workbench, Units, Lens, Probes, Artifacts
and Model. The workbench is three regions: the thread (the loom as the path you are
on, and earlier chats), the page (the conversation or the loom's map) and the
Inspector, which shows what is selected (a token's drilldown and anatomy, a
probe, a unit) and, with nothing selected, the instruments; a steering term
opens under the recipe's chips and leaves the Inspector as it was.
From 1240 px the regions sit side by side; narrower it is Page | Inspector with
the thread folded over the page; on a phone it is one column with the
Inspector as a sheet. Units, Lens and Probes read the conversation at one
token cursor (`lib/stores/tokenCursor.svelte.ts`, drawn by
`panels/rooms/TokenStrip.svelte`), which is the workbench's token selection;
Artifacts is where what they read with is fitted, fetched, built and
installed, and every setup link leads to its section. A tool a room hosts
opens in that room; any other opens as a dialog over the page. The command
palette is the header's search (⌘K, or **Menu → Search everything**).

The steering recipe is chips over the composer, with the term editor in place
under them, and it is the one place steering lives: "+ Steer" picks every kind
and the instruments' Push and Silence add a chip through `lib/steerActions.ts`.
The dashboard serializes chips into an expression
(`lib/expression.ts`) and never parses one: a recipe from elsewhere comes back
from the server's validate route as structured `parts`, and one the chips
cannot draw stays a single read-only chip.

## Source map

```text
src/
  main.ts, App.svelte   runtime install and mount; the shell (header, rooms, dialog host)
  panels/workbench/     the workbench and its thread
  panels/inspector/     the Inspector, the instruments, the unit view
  panels/rooms/         the Units, Lens, Probes, Artifacts and Model rooms; the token strip
  panels/               the page (Chat, RawBuffer), the recipe chips, the probe rack, loom/, rack/
  lib/steerActions.ts   the one way into the recipe
  drawers/              one component per tool; token/ is the token drilldown
  lib/runtime/          the RuntimeClient contract and the HTTP runtime
  lib/stores/           one $state slice per concern
  lib/steering/         the term editor's logic
  lib/style/            the shell's Ledger classes
```

The full map, with each module's ownership, is in `../drowse/web/AGENTS.md`.

The server-owned wire contract is documented beside the implementation in
`../drowse/server/AGENTS.md`; dashboard-specific ownership and interaction
contracts live in `../drowse/web/AGENTS.md`. Prefer those sources over copying a
route inventory into this file.

## State and component rules

- Put state in the smallest matching file under `lib/stores/`. Use
  `stores.svelte.ts` only for state shared across the WebSocket, loom, chat, and
  instruments.
- Svelte collection state uses `SvelteMap`/`SvelteSet`. Replace stored objects
  when mutating them so subscribers observe the change.
- Add shared primitives under `lib/ui/`; do not duplicate button, tab, card, or
  drawer chrome inside a feature.
- Keep `lib/types.ts` aligned with the Pydantic/native WebSocket schemas. New UI
  code reads the canonical `measurements` envelope on each token; the pre-5.x
  top-level `captured` / per-token readout aliases are gone.
- Long-running generate/fit/train work uses one updateable toast or the existing
  progress surface, not a stream of transient notifications.

## Adding a drawer

1. Add `src/drawers/FooDrawer.svelte` and accept `params: unknown` through
   `$props()`.
2. Add its name to `DrawerName` in `lib/types.ts` and its rows in `drawers/index.ts`
   (`narrow` for a form or picker; a `launcher` puts it in the command palette),
   `runtime/ui-capabilities.ts` and `lib/webmcp/catalogue.ts`.
3. If a room hosts it, add it to `ROOM_SECTIONS` in `lib/stores/rooms.svelte.ts`;
   otherwise it opens as a dialog.
4. Run the check/build sequence above and commit the regenerated bundle.

## Visual system

The dashboard is drawn in Ledger, the Standard Logits design language, from the
`@stdlogits/ledger` package, pinned at the v0.1.4 tag's GitHub tarball (its
`DESIGN.md` is the design): white paper, two
washes, three inks, one accent, no lines. The theme is auto (the system's look),
light or dark, behind one theme button. The accent is depth, the ramp from
`--deep` to `--late`, and it only fills. Wix Madefor is the only face, with
tabular numbers. `src/lib/style/drowse.css` holds the shell's classes.
