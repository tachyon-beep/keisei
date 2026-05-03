<script>
  import { leagueEntries } from '../stores/league.js'
  import { showcaseGame, showcaseQueue, queueDepth, sidecarAlive, showcaseSpeed } from '../stores/showcase.js'
  import { sendShowcaseCommand } from './ws.js'

  let selectedEntry1 = ''
  let selectedEntry2 = ''
  /**
   * Whether to render the full setup form. When a match is in progress we
   * collapse to a single "+ New match" button and the spectator UI gets the
   * screen real estate. The user can expand explicitly.
   */
  export let collapsed = false
  let expanded = false

  // Approximate ms-per-move so users understand what slow/normal/fast mean.
  // Keep in sync with backend SHOWCASE_SPEEDS in keisei/showcase/runner.py if
  // the table moves; this is purely informational.
  const SPEED_HINTS = {
    slow: '~2 s / move — easy to follow',
    normal: '~700 ms / move — balanced',
    fast: '~150 ms / move — quick games',
  }

  $: activeEntries = ($leagueEntries || []).filter(e => e.status === 'active')

  // canStart depends only on local form state + sidecar/queue. Each negative
  // condition produces a distinct hint so the user knows *why* the button is
  // disabled.
  $: disabledReason = (() => {
    if (!$sidecarAlive) return 'Showcase engine is offline — start the sidecar to enable matches.'
    if ($queueDepth >= 5) return `Queue is full (${$queueDepth} pending). Wait for one to start before adding more.`
    if (!selectedEntry1 || !selectedEntry2) return 'Pick a player for both Black and White.'
    if (selectedEntry1 === selectedEntry2) return 'Black and White must be different players.'
    return null
  })()
  $: canStart = disabledReason === null

  function requestMatch() {
    if (!canStart) return
    sendShowcaseCommand({
      type: 'request_showcase_match',
      entry_id_1: selectedEntry1,
      entry_id_2: selectedEntry2,
      speed: $showcaseSpeed,
    })
    if (collapsed) expanded = false
  }

  function changeSpeed(newSpeed) {
    showcaseSpeed.set(newSpeed)
    sendShowcaseCommand({ type: 'change_showcase_speed', speed: newSpeed })
  }

  function toggleExpanded() {
    expanded = !expanded
  }
</script>

{#if collapsed && !expanded}
  <div class="collapsed-row">
    <button
      class="new-match-btn"
      on:click={toggleExpanded}
      aria-label="Open match setup form"
    >
      + New match
    </button>
    <div class="speed-controls compact" role="group" aria-label="Playback speed">
      {#each ['slow', 'normal', 'fast'] as s}
        <button
          class:active={$showcaseSpeed === s}
          aria-pressed={$showcaseSpeed === s}
          on:click={() => changeSpeed(s)}
          title={SPEED_HINTS[s]}
        >{s}</button>
      {/each}
    </div>
  </div>
{:else}
  <div class="match-controls" role="group" aria-label="Match setup">
    {#if collapsed}
      <button
        class="collapse-btn"
        on:click={toggleExpanded}
        aria-label="Hide match setup form"
        title="Hide setup"
      >×</button>
    {/if}
    <div class="entry-selectors">
      <select bind:value={selectedEntry1} aria-label="Black player">
        <option value="">Select black…</option>
        {#each activeEntries as entry}
          <option value={String(entry.id)}>
            {entry.display_name} ({entry.elo_rating?.toFixed(0) ?? '?'})
          </option>
        {/each}
      </select>
      <span class="vs">vs</span>
      <select bind:value={selectedEntry2} aria-label="White player">
        <option value="">Select white…</option>
        {#each activeEntries as entry}
          <option value={String(entry.id)}>
            {entry.display_name} ({entry.elo_rating?.toFixed(0) ?? '?'})
          </option>
        {/each}
      </select>
    </div>
    <div class="speed-controls" role="group" aria-label="Playback speed">
      <span class="label">Speed:</span>
      {#each ['slow', 'normal', 'fast'] as s}
        <button
          class:active={$showcaseSpeed === s}
          aria-pressed={$showcaseSpeed === s}
          on:click={() => changeSpeed(s)}
          title={SPEED_HINTS[s]}
        >{s}</button>
      {/each}
    </div>
    <button
      class="start-btn"
      on:click={requestMatch}
      disabled={!canStart}
      title={disabledReason || 'Start a new showcase match'}
      aria-describedby={disabledReason ? 'start-disabled-reason' : undefined}
    >Start Match</button>
    {#if disabledReason}
      <div id="start-disabled-reason" class="disabled-reason" role="status">{disabledReason}</div>
    {/if}
  </div>
{/if}

<style>
  .collapsed-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
  }

  .new-match-btn {
    padding: 6px 14px;
    min-height: 36px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid var(--accent-teal);
    border-radius: 4px;
    background: transparent;
    color: var(--accent-teal);
    cursor: pointer;
  }
  .new-match-btn:hover { background: var(--badge-bg-teal); }
  .new-match-btn:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }

  .match-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border-bottom: 1px solid var(--border);
  }

  .collapse-btn {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 16px;
    line-height: 1;
    min-width: 28px;
    min-height: 28px;
    padding: 0;
  }
  .collapse-btn:hover { color: var(--text-primary); border-color: var(--text-secondary); }
  .collapse-btn:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }

  .entry-selectors { display: flex; align-items: center; gap: 8px; }
  .vs { font-weight: 600; color: var(--text-muted); font-size: 13px; }

  select {
    padding: 6px 8px;
    min-height: 36px;
    font-size: 13px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
  }
  select:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }

  .speed-controls { display: flex; align-items: center; gap: 4px; }
  .speed-controls .label { font-size: 12px; color: var(--text-secondary); }

  .speed-controls button {
    padding: 4px 10px;
    min-height: 32px;
    font-size: 12px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    text-transform: capitalize;
  }
  .speed-controls.compact button { min-height: 28px; padding: 2px 8px; font-size: 11px; }

  .speed-controls button[aria-pressed='true'] {
    border-color: var(--tab-active-border);
    color: var(--tab-active-border);
    background: var(--tab-active-bg);
  }
  .speed-controls button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }

  .start-btn {
    padding: 6px 16px;
    min-height: 36px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid var(--accent-teal);
    border-radius: 4px;
    background: var(--accent-teal);
    color: #fff;
    cursor: pointer;
  }
  .start-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .start-btn:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }

  .disabled-reason {
    font-size: 12px;
    color: var(--accent-gold);
    flex-basis: 100%;
  }
</style>
