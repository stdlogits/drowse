<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { IconButton } from "@stdlogits/ledger";
  // Cast manager (phase 3 of the cast model) — the tree's roster of
  // named labels, each with a standing steering recipe.  A member's
  // recipe is the *weakest* tier at generation: it fills only fields
  // the send left unset, so the rack and per-send controls always win.
  // Identity is auto-derived from labels observed anywhere in the tree;
  // configuration adds only a standing recipe/notes layer.  Mutations and
  // new observed labels reconcile through the inlined effective roster.
  //
  // This drawer also owns the two active structural-role labels.  The
  // composer chooses between those labels without separately exposing their
  // canonical user/assistant roles; a turn labeled with a member's slug
  // generates under its recipe.

  import { apiTree } from "../lib/runtime/services";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import {
    castState,
    pushToast,
    samplingState,
    sessionState,
  } from "../lib/stores.svelte";
  import { closeDrawer } from "../lib/stores/drawers.svelte";
  import { IDENTITY_HELP, SAMPLING_HELP } from "../lib/parameterHelp";
  import { registerCastController } from "../lib/workspaceController";
  import { SAMPLING_SEED_MAX, samplingSeedMinimum } from "../lib/runtime/samplingCapabilities";
  import { getRuntimeClient } from "../lib/runtime/registry";
  import NumberInput from "../lib/NumberInput.svelte";
  import Select from "../lib/Select.svelte";
  import type { CastMemberJSON } from "../lib/types";

  let steering = $state("");
  let notes = $state("");
  let thinking = $state("inherit");
  let seed = $state<number | null>(null);
  let editingLabel = $state<string | null>(null);
  let editingKey = $state<string | null>(null);
  let busy = $state(false);
  let err = $state<string | null>(null);
  let lastSavedKey = $state<string | null>(null);
  let saveTimer: ReturnType<typeof setTimeout> | null = null;
  let saveRequest: Promise<boolean> | null = null;

  const defaultUserRole = $derived(
    sessionState.info?.default_user_role ?? "user",
  );
  const defaultAssistantRole = $derived(
    sessionState.info?.default_assistant_role ?? "assistant",
  );
  const activeRoles = $derived.by(() => [...new Set([
    defaultUserRole,
    defaultAssistantRole,
    samplingState.user_role.trim(),
    samplingState.assistant_role.trim(),
  ].filter(Boolean))]);
  // The server roster is keyed by Drowse's structural roles. Present those
  // two entries through the model's actual chat-template vocabulary, while
  // retaining the hidden key so structural cast recipes still round-trip.
  const roster = $derived.by(() => {
    const byLabel = new Map<
      string,
      { key: string; label: string; member: CastMemberJSON | null }
    >();
    for (const [key, member] of Object.entries(castState.roster)) {
      const displayLabel =
        key === "user"
          ? defaultUserRole
          : key === "assistant"
            ? defaultAssistantRole
            : key;
      const existing = byLabel.get(displayLabel);
      const structural = key === "user" || key === "assistant";
      if (!existing || structural) {
        byLabel.set(displayLabel, { key, label: displayLabel, member });
      }
    }
    for (const role of activeRoles) {
      if (byLabel.has(role)) continue;
      const key = role === defaultUserRole
        ? "user"
        : role === defaultAssistantRole
          ? "assistant"
          : role;
      byLabel.set(role, {
        key,
        label: role,
        member: castState.roster[key] ?? null,
      });
    }
    const standard = activeRoles
      .map((role) => byLabel.get(role))
      .filter((row) => row !== undefined);
    const custom = [...byLabel.values()]
      .filter((row) => !activeRoles.includes(row.label))
      .sort((a, b) => a.label.localeCompare(b.label));
    return [...standard, ...custom];
  });

  const selectedMember = $derived(
    editingKey ? (castState.roster[editingKey] ?? null) : null,
  );
  const dirty = $derived(
    editingKey !== null && (
      steering.trim() !== (selectedMember?.recipe?.steering ?? "")
      || notes.trim() !== (selectedMember?.notes ?? "")
      || thinking !== (typeof selectedMember?.recipe?.thinking === "boolean" ? String(selectedMember.recipe.thinking) : "inherit")
      || seed !== (selectedMember?.recipe?.seed ?? null)
    ),
  );
  const saveStatus = $derived(
    busy
      ? "Saving…"
      : dirty
        ? "Changes save automatically"
        : lastSavedKey === editingKey
          ? "Saved"
          : "Changes save automatically",
  );

  function loadMember(row: (typeof roster)[number]): void {
    editingLabel = row.label;
    editingKey = row.key;
    steering = row.member?.recipe?.steering ?? "";
    notes = row.member?.notes ?? "";
    thinking = typeof row.member?.recipe?.thinking === "boolean" ? String(row.member.recipe.thinking) : "inherit";
    seed = row.member?.recipe?.seed ?? null;
    err = null;
  }

  onMount(() => registerCastController({
    read: () => ({ busy: busy || saveTimer !== null, dirty, values: { label: editingKey, steering, thinking, seed, notes } }),
    sync: () => {
      const row = roster.find(row => row.key === editingKey);
      if (row) loadMember(row);
      else { steering = ""; notes = ""; thinking = "inherit"; seed = null; }
    },
  }));

  function cancelScheduledSave(): void {
    if (saveTimer === null) return;
    clearTimeout(saveTimer);
    saveTimer = null;
  }

  function scheduleSave(): void {
    err = null;
    lastSavedKey = null;
    cancelScheduledSave();
    saveTimer = setTimeout(() => {
      saveTimer = null;
      void saveCurrent();
    }, 450);
  }

  $effect(() => {
    const rows = roster;
    if (editingKey !== null && rows.some((row) => row.key === editingKey)) return;
    if (rows[0]) loadMember(rows[0]);
  });

  async function saveCurrent(): Promise<boolean> {
    cancelScheduledSave();
    if (saveRequest) {
      const saved = await saveRequest;
      if (!saved) return false;
    }
    const target = editingKey;
    if (!target || !dirty) return true;
    const targetLabel = editingLabel;
    const nextSteering = steering.trim();
    const nextNotes = notes.trim();
    const nextThinking = thinking === "inherit" ? null : thinking === "true";
    const nextSeed = seed;
    busy = true;
    err = null;
    const request = (async (): Promise<boolean> => {
      try {
        const r = await apiTree.castPut(target, {
          steering: nextSteering === "" ? null : nextSteering,
          notes: nextNotes,
          thinking: nextThinking,
          seed: nextSeed,
        });
        // Optimistic merge — the ``op="cast"`` frame confirms shortly.
        castState.roster = { ...castState.roster, [target]: r.member };
        if (editingKey === target) lastSavedKey = target;
        return true;
      } catch (e) {
        err = userFacingError(e, `Unable to save ${targetLabel ?? "this role"}. Check the fields and try again.`);
        return false;
      } finally {
        busy = false;
      }
    })();
    saveRequest = request;
    const saved = await request;
    if (saveRequest === request) saveRequest = null;
    if (saved && editingKey === target && dirty) {
      return saveCurrent();
    }
    return saved;
  }

  async function selectMember(row: (typeof roster)[number]): Promise<void> {
    if (row.key === editingKey) return;
    if (!(await saveCurrent())) return;
    loadMember(row);
  }

  async function closeWithSave(): Promise<void> {
    if (await saveCurrent()) closeDrawer();
  }

  async function remove(slug: string): Promise<void> {
    try {
      await apiTree.castDelete(slug);
      const nextRoster = { ...castState.roster };
      delete nextRoster[slug];
      castState.roster = nextRoster;
      if (editingKey === slug) {
        steering = "";
        notes = "";
        thinking = "inherit";
        seed = null;
        lastSavedKey = null;
      }
    } catch (e) {
      pushToast(userFacingError(e, "Unable to remove this speaker. Try again."), {
        kind: "error",
      });
    }
  }

  async function clearSelectedSettings(): Promise<void> {
    cancelScheduledSave();
    if (saveRequest && !(await saveRequest)) return;
    if (editingKey) await remove(editingKey);
  }

  onDestroy(() => {
    cancelScheduledSave();
    if (dirty) void saveCurrent();
  });
</script>

<section class="drawer-shell" aria-label="Role settings drawer">
  <div class="body">
    <section class="dw-sec role-picker" aria-labelledby="conversation-roles-title">
      <div class="dw-sec-head">
        <h3 id="conversation-roles-title" class="head">Conversation roles</h3>
        <IconButton icon="close" label="Close" onclick={() => void closeWithSave()} />
      </div>
      <ul class="roster" aria-label="Conversation roles">
        {#each roster as row (row.label)}
          <li class="member" class:editing={editingKey === row.key}>
            <button
              type="button"
              class="member-main"
              aria-pressed={editingKey === row.key}
              onclick={() => void selectMember(row)}
            >
              <span class="dw-pip" class:is-model={row.label === "assistant"} aria-hidden="true"></span>
              <span class="member-text">
                <span class="member-label">{row.label}</span>
                {#if row.member?.recipe && Object.keys(row.member.recipe).length > 0}
                  <span class="meta">standing recipe</span>
                {:else}
                  <span class="meta">reply settings</span>
                {/if}
                {#if row.member?.notes}
                  <span class="meta member-notes">{row.member.notes}</span>
                {/if}
              </span>
            </button>
          </li>
        {/each}
      </ul>
    </section>

    {#if editingKey && editingLabel}
      <section class="dw-sec form" aria-labelledby="selected-role-title">
        <div class="dw-sec-head">
          <h3 id="selected-role-title" class="role-title">{editingLabel}</h3>
          <p class="meta" role="status" aria-live="polite">{saveStatus}</p>
        </div>
        <label class="field">
          <span class="meta" data-tip={IDENTITY_HELP.cast}>Steering</span>
          <input
            bind:value={steering}
            placeholder="None"
            spellcheck="false"
            autocomplete="off"
            aria-label={`Steering expression for ${editingLabel}`}
            oninput={scheduleSave}
          />
        </label>
        <div class="pair">
          <div class="field">
            <span class="meta" data-tip={SAMPLING_HELP.thinking}>Thinking</span>
            <Select
              value={thinking}
              options={[{ value: "inherit", label: "Reply setting" }, { value: "true", label: "On" }, { value: "false", label: "Off" }]}
              disabled={!sessionState.info?.supports_thinking || !sessionState.info?.thinking_is_optional}
              ariaLabel={`Thinking default for ${editingLabel}`}
              onchange={(value) => { thinking = value; scheduleSave(); }}
            />
          </div>
          <div class="field">
            <span class="meta" data-tip={SAMPLING_HELP.seed}>Seed</span>
            <NumberInput value={seed} min={samplingSeedMinimum(getRuntimeClient().mode)} max={SAMPLING_SEED_MAX} step={1} allowEmpty placeholder="Reply setting" ariaLabel={`Seed default for ${editingLabel}`} onchange={(value) => { seed = value === null ? null : Math.floor(value); scheduleSave(); }} />
          </div>
        </div>
        <label class="field">
          <span class="meta">Private note</span>
          <input
            bind:value={notes}
            placeholder="A note for yourself"
            autocomplete="off"
            oninput={scheduleSave}
          />
        </label>
        {#if err}
          <p class="error" role="alert">{err}</p>
        {/if}
        {#if selectedMember?.origin === "configured"}
          <div class="dw-row">
            <button type="button" class="dw-text" disabled={busy} onclick={() => void clearSelectedSettings()}>Clear settings</button>
          </div>
        {/if}
      </section>
    {/if}
  </div>
</section>

<style>
  /* Roles: the roster as rows (the one being edited in wash2), its settings beside it. */
  .drawer-shell { min-width: 0; color: var(--ink2); }
  .body { display: grid; grid-template-columns: minmax(220px, 300px) minmax(0, 1fr); gap: var(--s7); align-items: start; }
  .roster { display: grid; gap: 2px; }
  .member-main { display: grid; grid-template-columns: 12px minmax(0, 1fr); gap: var(--s2); align-items: start; width: 100%; padding: var(--s2) var(--s3); border: 0; border-radius: var(--r-box); background: transparent; color: var(--ink2); font: var(--type-small); text-align: left; cursor: pointer; transition: background-color var(--fast) var(--ease); }
  .member-main:hover { background: var(--wash); }
  .member.editing .member-main { background: var(--wash2); }
  .member-text { display: grid; gap: 2px; min-width: 0; }
  .member-label { color: var(--ink); font: var(--type-control); }
  .member-notes { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .role-title { margin: 0; color: var(--ink); font: var(--type-lead); }
  .form { gap: var(--s4); }
  .field { display: grid; gap: var(--s1); min-width: 0; }
  .field input { width: 100%; height: 40px; padding: 0 var(--s3); border: 0; border-radius: var(--r-col); background: var(--wash); color: var(--ink); font: var(--type-small); }
  .pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s4); }
  .error { color: var(--ink); font: var(--type-small); }
  @media (max-width: 860px) {
    .body { grid-template-columns: minmax(0, 1fr); gap: var(--s6); }
  }
  @media (max-width: 720px) {
    .pair { grid-template-columns: minmax(0, 1fr); }
    .field input { font-size: 16px; }
  }
</style>
