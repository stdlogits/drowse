<script lang="ts">
  // The Model room's API access section (a dialog anywhere else): the bearer key this tab sends,
  // and the server's sessions as rows, the current one on wash2. drowse.ai swaps this file for the
  // unavailable one through the @runtime-session-drawer alias.
  import { IconButton, MorphText } from "@stdlogits/ledger";
  import { onMount } from "svelte";
  import Button from "../lib/ui/Button.svelte";
  import { getApiKey, setApiKey } from "../lib/runtime/http-auth";
  import { apiSessions } from "../lib/runtime/services";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import type { SessionInfo } from "../lib/types";
  import { closeDrawer, refreshSession, sessionState } from "../lib/stores.svelte";

  let { params = null }: { params?: unknown } = $props();
  const embedded = $derived((params as { embedded?: boolean } | null)?.embedded === true);

  let key = $state(getApiKey() ?? "");
  let sessions: SessionInfo[] = $state([]);
  let loaded = $state(false);
  let busy = $state(false);
  let errorMsg: string | null = $state(null);
  let saved = $state(false);
  const uid = $props.id();

  async function loadSessions(): Promise<void> {
    busy = true;
    errorMsg = null;
    try {
      const r = await apiSessions.list();
      sessions = r.sessions;
      loaded = true;
    } catch (e) {
      errorMsg = userFacingError(e, "Unable to load server sessions. Check the server connection and try again.");
    } finally {
      busy = false;
    }
  }

  async function saveKey(): Promise<void> {
    setApiKey(key);
    saved = true;
    await refreshSession();
    await loadSessions();
  }

  onMount(() => { void loadSessions(); });
</script>

<section class="access" class:embedded aria-label={embedded ? "API access" : "Session and auth drawer"}>
  {#if !embedded}
    <header class="dlg-head">
      <h2>API access</h2>
      <IconButton icon="close" label="Close" onclick={closeDrawer} />
    </header>
  {/if}

  <div class="body">
    <section class="dw-sec" aria-labelledby={`${uid}-key-head`}>
      <div class="dw-sec-head"><span class="head" id={`${uid}-key-head`}>Key</span><span class="meta">This tab, until you reload</span></div>
      <form class="key-row" onsubmit={(event) => { event.preventDefault(); void saveKey(); }}>
        <label class="sr-only" for={`${uid}-key`}>Drowse API key</label>
        <input
          id={`${uid}-key`}
          type="password"
          bind:value={key}
          placeholder="DROWSE_API_KEY"
          autocomplete="off"
          oninput={() => (saved = false)}
        />
        <Button type="submit" variant="solid">Apply</Button>
        <Button variant="flat" onclick={() => { key = ""; void saveKey(); }}>Clear</Button>
      </form>
      <p class="meta" role="status"><MorphText text={saved ? (key ? "Applied" : "Cleared") : ""} numbers={false} /></p>
    </section>

    <section class="dw-sec" aria-labelledby={`${uid}-sessions-head`}>
      <div class="dw-sec-head">
        <span class="head" id={`${uid}-sessions-head`}>Sessions</span>
        <button type="button" class="dw-text" aria-busy={busy} disabled={busy} onclick={loadSessions}>
          <MorphText text={busy ? "Loading…" : "Refresh"} numbers={false} />
        </button>
      </div>
      {#if errorMsg}
        <p class="error" role="alert">{errorMsg}</p>
      {/if}
      {#if !loaded && busy}
        <div class="dw-blocks" aria-hidden="true"><i></i></div>
      {:else if loaded && sessions.length === 0}
        <p class="meta">None on this server</p>
      {:else}
        <ul class="sessions">
          {#each sessions as s (s.id)}
            <li class="session" aria-current={sessionState.info?.id === s.id ? "true" : undefined}>
              <span class="id">{s.id}</span>
              <span class="model">{s.model_id}</span>
              <span class="meta">{s.device}/{s.dtype} · {s.profiles.length} profiles · {s.probes.length} probes{sessionState.info?.id === s.id ? " · this tab" : ""}</span>
            </li>
          {/each}
        </ul>
      {/if}
    </section>
  </div>
</section>

<style>
  .access { display: flex; flex-direction: column; min-height: 0; color: var(--ink2); }
  .dlg-head { display: flex; flex: none; align-items: center; justify-content: space-between; gap: var(--s4); padding: var(--s5) var(--s5) 0; }
  .dlg-head h2 { margin: 0; color: var(--ink); font: var(--type-mark); }
  .body { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s6); align-content: start; max-width: 760px; padding: var(--s5); overflow: auto; }
  .embedded .body { padding: 0; overflow: visible; }

  .key-row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s3); }
  .key-row input { flex: 1 1 280px; min-width: 0; }
  .meta:empty { display: none; }
  .error { color: var(--ink); font: var(--type-small); overflow-wrap: anywhere; }

  /* The sessions as rows (the Lens room's source rows): an ID, the model, its facts. */
  .sessions { display: grid; gap: 2px; }
  .session { display: grid; grid-template-columns: 104px minmax(0, 1fr); gap: 2px var(--s3); align-items: baseline; min-height: 40px; padding: var(--s2) var(--s3); border-radius: var(--r-box); }
  .session[aria-current="true"] { background: var(--wash2); }
  .session .id { color: var(--ink); font: 600 14px/20px var(--display); letter-spacing: -0.01em; overflow-wrap: anywhere; }
  .session .model { color: var(--ink); font: var(--type-control); overflow-wrap: anywhere; }
  .session .meta { grid-column: 2; }
  .dw-blocks > i { height: 40px; }

  @media (max-width: 720px) {
    .dlg-head { padding: var(--s4) var(--s4) 0; }
    .body { padding: var(--s4); }
    .key-row input { flex-basis: 100%; min-height: 44px; }
    .session { grid-template-columns: minmax(0, 1fr); }
    .session .meta { grid-column: 1; }
  }
</style>
