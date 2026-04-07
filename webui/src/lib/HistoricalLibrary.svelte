<script>
  import { historicalLibrary, gauntletResults } from '../stores/league.js'
  import { trainingState } from '../stores/training.js'

  $: currentEpoch = $trainingState?.current_epoch || 0
  $: maxGauntletEpoch = $gauntletResults.length > 0
    ? Math.max(...$gauntletResults.map(g => g.epoch))
    : null
  $: staleness = maxGauntletEpoch != null && currentEpoch > 0
    ? currentEpoch - maxGauntletEpoch
    : null

  // Group gauntlet results by epoch (most recent first)
  $: gauntletByEpoch = (() => {
    const map = new Map()
    for (const g of $gauntletResults) {
      if (!map.has(g.epoch)) map.set(g.epoch, [])
      map.get(g.epoch).push(g)
    }
    return [...map.entries()].sort((a, b) => b[0] - a[0])
  })()
</script>

<div class="historical-library">
  <div class="slots-section">
    <h4 class="section-label">
      Library Slots
      {#if staleness != null}
        <span class="staleness">Last gauntlet: {staleness} epoch{staleness !== 1 ? 's' : ''} ago</span>
      {/if}
    </h4>
    {#if $historicalLibrary.length === 0}
      <p class="empty">No historical slots configured</p>
    {:else}
      <table>
        <caption class="sr-only">Historical library slot assignments</caption>
        <thead>
          <tr>
            <th scope="col" class="num">#</th>
            <th scope="col">Entry</th>
            <th scope="col" class="num">Target</th>
            <th scope="col" class="num">Actual</th>
            <th scope="col">Mode</th>
          </tr>
        </thead>
        <tbody>
          {#each $historicalLibrary as slot}
            <tr>
              <td class="num">{slot.slot_index}</td>
              <td>{slot.entry_name || '—'}</td>
              <td class="num">{slot.target_epoch}</td>
              <td class="num">{slot.actual_epoch ?? '—'}</td>
              <td class="mode">{slot.selection_mode}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>

  {#if gauntletByEpoch.length > 0}
    <div class="gauntlet-section">
      <h4 class="section-label">Gauntlet Results</h4>
      {#each gauntletByEpoch.slice(0, 5) as [epoch, results]}
        <div class="gauntlet-epoch">
          <span class="epoch-header">Epoch {epoch}</span>
          {#each results as g}
            <div class="gauntlet-row">
              <span class="slot-tag">Slot {g.historical_slot}</span>
              <span class="wld">{g.wins}W {g.losses}L {g.draws}D</span>
              {#if g.elo_before != null && g.elo_after != null}
                {@const delta = Math.round(g.elo_after - g.elo_before)}
                <span class="elo-delta" class:positive={delta > 0} class:negative={delta < 0}>
                  {delta > 0 ? '+' : ''}{delta}
                </span>
              {/if}
            </div>
          {/each}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .historical-library { padding: 10px 14px; display: flex; flex-direction: column; gap: 12px; }
  .section-label {
    font-size: 12px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--text-muted); margin: 0 0 6px;
    display: flex; align-items: center; gap: 8px;
  }
  .staleness { font-weight: 400; font-size: 12px; color: var(--accent-gold); text-transform: none; letter-spacing: 0; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  thead { color: var(--text-muted); font-size: 12px; }
  th, td { text-align: left; padding: 3px 8px; }
  th.num, td.num { text-align: right; }
  .mode { font-size: 12px; color: var(--text-muted); }
  .gauntlet-epoch { margin-bottom: 6px; }
  .epoch-header {
    font-size: 12px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--text-muted); display: block; margin-bottom: 2px;
  }
  .gauntlet-row {
    display: flex; align-items: center; gap: 8px; font-size: 12px;
    padding: 2px 4px; border-radius: 3px;
  }
  .gauntlet-row:hover { background: var(--bg-card); }
  .slot-tag { font-size: 12px; color: var(--text-muted); min-width: 48px; }
  .wld { font-family: monospace; font-size: 12px; color: var(--text-secondary); }
  .elo-delta { font-family: monospace; font-size: 12px; font-weight: 600; }
  .elo-delta.positive { color: var(--accent-teal); }
  .elo-delta.negative { color: var(--danger); }
  .empty { color: var(--text-muted); font-size: 12px; text-align: center; padding: 12px; }
</style>
