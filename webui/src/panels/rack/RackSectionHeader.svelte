<script lang="ts" generics="T extends string">
  import { MorphText, Switch } from "@stdlogits/ledger";
  // The header of an instrument rack (probes, lens and SAE steering and readouts). This
  // owns the title/live/count/sort rhythm so future tabs cannot drift by a
  // pixel or silently omit one of the shared controls.

  import Select from "../../lib/Select.svelte";

  interface SortOption<U> {
    value: U;
    label: string;
    disabled?: boolean;
  }

  let {
    title,
    help,
    count,
    live = null,
    liveBusy = false,
    liveTitle = "",
    liveLabel,
    liveHelp,
    onLiveToggle,
    sortValue,
    sortOptions = [],
    sortAriaLabel = "Sort cards by",
    onSortChange,
  }: {
    title: string;
    help?: string;
    count?: string;
    live?: boolean | null;
    liveBusy?: boolean;
    liveTitle?: string;
    liveLabel?: string;
    liveHelp?: string;
    onLiveToggle?: () => void;
    sortValue?: T;
    sortOptions?: SortOption<T>[];
    sortAriaLabel?: string;
    onSortChange?: (value: T) => void;
  } = $props();

  const hasSort = $derived(sortValue !== undefined && sortOptions.length > 0);
  const contextualLiveLabel = $derived(liveLabel ?? `${title} live readings`);
</script>

<header class="header">
  <div class="header-text">
    <span class="title" data-tip={help}>{title}</span>
    {#if count}<span class="meta count" aria-live="polite"><MorphText text={count} /></span>{/if}
  </div>
  <div class="header-end">
    {#if live !== null}
      <span class="live-control" inert={liveBusy} {...{ "aria-description": liveTitle }}>
        <Switch checked={live} onchange={() => onLiveToggle?.()} label="Live" hint={liveHelp ?? `Turn ${live ? "off" : "on"} ${contextualLiveLabel}`} />
      </span>
    {/if}
    {#if hasSort}
      <span class="sort">
        <Select value={sortValue as T} options={sortOptions} onchange={onSortChange} ariaLabel={sortAriaLabel} />
      </span>
    {/if}
  </div>
</header>

<style>
  .header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s1) var(--s4); min-height: 36px; }
  .header-text { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s2); min-width: 0; }
  .title { color: var(--ink); font: var(--type-control); white-space: nowrap; }
  .header-end { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s3); min-width: 0; }
  .live-control[inert] { opacity: 0.4; }
  .sort { display: inline-flex; min-width: 8em; }
</style>
