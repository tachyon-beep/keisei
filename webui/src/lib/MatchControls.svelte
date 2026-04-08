<script>
  import { leagueEntries } from '../stores/league.js'
  import { showcaseQueue, queueDepth, sidecarAlive, showcaseSpeed } from '../stores/showcase.js'
  import { sendShowcaseCommand } from './ws.js'

  let selectedEntry1 = ''
  let selectedEntry2 = ''
  let speed = 'normal'

  $: activeEntries = ($leagueEntries || []).filter(e => e.status === 'active')
  $: canStart = selectedEntry1 && selectedEntry2 && selectedEntry1 !== selectedEntry2
    && $sidecarAlive && $queueDepth < 5

  function requestMatch() {
    if (!canStart) return
    sendShowcaseCommand({
      type: 'request_showcase_match',
      entry_id_1: selectedEntry1,
      entry_id_2: selectedEntry2,
      speed,
    })
  }

  function changeSpeed(newSpeed) {
    speed = newSpeed
    showcaseSpeed.set(newSpeed)
    sendShowcaseCommand({ type: 'change_showcase_speed', speed: newSpeed })
  }
</script>

<div class="match-controls">
  <div class="entry-selectors">
    <select bind:value={selectedEntry1} aria-label="Black player">
      <option value="">Select black...</option>
      {#each activeEntries as entry}
        <option value={String(entry.id)}>
          {entry.display_name} ({entry.elo_rating?.toFixed(0) ?? '?'})
        </option>
      {/each}
    </select>
    <span class="vs">vs</span>
    <select bind:value={selectedEntry2} aria-label="White player">
      <option value="">Select white...</option>
      {#each activeEntries as entry}
        <option value={String(entry.id)}>
          {entry.display_name} ({entry.elo_rating?.toFixed(0) ?? '?'})
        </option>
      {/each}
    </select>
  </div>
  <div class="speed-controls">
    <span class="label">Speed:</span>
    {#each ['slow', 'normal', 'fast'] as s}
      <button class:active={speed === s} on:click={() => changeSpeed(s)}>{s}</button>
    {/each}
  </div>
  <button class="start-btn" on:click={requestMatch} disabled={!canStart}>Start Match</button>
  {#if !$sidecarAlive}
    <div class="warning">Showcase engine is offline</div>
  {:else if $queueDepth >= 5}
    <div class="warning">Queue full ({$queueDepth} pending)</div>
  {/if}
</div>

<style>
  .match-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; padding: 12px; border-bottom: 1px solid var(--border); }
  .entry-selectors { display: flex; align-items: center; gap: 8px; }
  .vs { font-weight: 600; color: var(--text-muted); font-size: 13px; }
  select { padding: 6px 8px; min-height: 36px; font-size: 13px; border: 1px solid var(--border); border-radius: 4px; background: var(--bg-primary); color: var(--text-primary); }
  .speed-controls { display: flex; align-items: center; gap: 4px; }
  .speed-controls .label { font-size: 12px; color: var(--text-secondary); }
  .speed-controls button { padding: 4px 10px; min-height: 32px; font-size: 12px; border: 1px solid var(--border); border-radius: 4px; background: transparent; color: var(--text-secondary); cursor: pointer; text-transform: capitalize; }
  .speed-controls button.active { border-color: var(--tab-active-border); color: var(--tab-active-border); background: var(--tab-active-bg); }
  .start-btn { padding: 6px 16px; min-height: 36px; font-size: 13px; font-weight: 600; border: 1px solid var(--accent-teal); border-radius: 4px; background: var(--accent-teal); color: #fff; cursor: pointer; }
  .start-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .warning { font-size: 12px; color: var(--accent-gold); }
</style>
