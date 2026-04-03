<script>
  import { leagueResults, leagueEntries } from '../stores/league.js'

  const MAX_ITEMS = 30

  $: entryMap = new Map($leagueEntries.map(e => [e.id, e]))

  // Count total clashes between each pair (both directions)
  $: clashCounts = (() => {
    const map = new Map()
    for (const r of $leagueResults) {
      const key = [r.learner_id, r.opponent_id].sort().join('-')
      map.set(key, (map.get(key) || 0) + 1)
    }
    return map
  })()

  function pairClashes(a, b) {
    const key = [a, b].sort().join('-')
    return clashCounts.get(key) || 0
  }

  // Build recent items with epoch separators
  $: items = (() => {
    const results = $leagueResults.slice(0, MAX_ITEMS)
    const out = []
    let lastEpoch = null

    for (const r of results) {
      if (r.epoch !== lastEpoch) {
        out.push({ type: 'separator', epoch: r.epoch })
        lastEpoch = r.epoch
      }

      const learner = entryMap.get(r.learner_id)
      const opponent = entryMap.get(r.opponent_id)
      const total = (r.wins || 0) + (r.losses || 0) + (r.draws || 0)
      const learnerWon = (r.wins || 0) > (r.losses || 0)
      const draw = (r.wins || 0) === (r.losses || 0)
      const clashes = pairClashes(r.learner_id, r.opponent_id)
      const winPct = total > 0 ? Math.round(((r.wins || 0) / total) * 100) : 0

      out.push({
        type: 'match',
        ...r,
        learnerName: learner?.display_name || learner?.architecture || `#${r.learner_id}`,
        opponentName: opponent?.display_name || opponent?.architecture || `#${r.opponent_id}`,
        total,
        learnerWon,
        draw,
        clashes,
        winPct,
      })
    }
    return out
  })()
</script>

<div class="recent-card">
  <h2 class="section-header">Recent Matches</h2>
  {#if items.length === 0}
    <p class="empty">No matches yet.</p>
  {:else}
    <div class="feed">
      {#each items as item}
        {#if item.type === 'separator'}
          <div class="epoch-separator">
            {#if item.epoch === -1}
              Tournament
            {:else}
              Epoch {item.epoch}
            {/if}
          </div>
        {:else}
          <div class="match-item">
            <div class="match-top">
              <span class="name" class:winner={item.learnerWon}>{item.learnerName}</span>
              <span class="vs">vs</span>
              <span class="name" class:winner={!item.learnerWon && !item.draw}>{item.opponentName}</span>
              <span class="match-score" class:win={item.learnerWon} class:loss={!item.learnerWon && !item.draw} class:tied={item.draw}>
                {item.wins}W {item.losses}L {item.draws}D
              </span>
            </div>
            <div class="match-detail">
              <span>{item.total} games</span>
              <span class="sep">·</span>
              <span>{item.winPct}% win</span>
              <span class="sep">·</span>
              <span>clash #{item.clashes}</span>
            </div>
          </div>
        {/if}
      {/each}
    </div>
  {/if}
</div>

<style>
  .recent-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    min-height: 100px;
    flex: 1;
  }

  .section-header {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    flex-shrink: 0;
  }

  .feed {
    overflow-y: scroll;
    min-height: 0;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .epoch-separator {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    padding: 6px 8px 3px;
    border-bottom: 1px solid var(--border-subtle);
    margin-top: 4px;
  }

  .epoch-separator:first-child {
    margin-top: 0;
  }

  .match-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 5px 8px;
    border-radius: 4px;
    font-size: 12px;
  }

  .match-item:hover {
    background: var(--bg-card);
  }

  .match-top {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .name {
    color: var(--text-secondary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .name.winner {
    color: var(--text-primary);
    font-weight: 600;
  }

  .vs {
    color: var(--text-muted);
    font-size: 10px;
    flex-shrink: 0;
  }

  .match-score {
    font-family: monospace;
    font-size: 11px;
    font-weight: 600;
    flex-shrink: 0;
    padding: 2px 6px;
    border-radius: 3px;
    margin-left: auto;
  }

  .match-score.win { color: var(--accent-teal); background: rgba(77, 184, 168, 0.1); }
  .match-score.loss { color: var(--danger); background: rgba(224, 80, 80, 0.1); }
  .match-score.tied { color: var(--accent-gold); background: rgba(200, 150, 46, 0.1); }

  .match-detail {
    font-size: 10px;
    color: var(--text-muted);
    display: flex;
    gap: 4px;
    padding-left: 2px;
  }

  .sep {
    color: var(--border);
  }

  .empty {
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    padding: 24px;
  }
</style>
