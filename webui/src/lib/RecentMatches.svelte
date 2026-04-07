<script>
  import { leagueResults, leagueEntries } from '../stores/league.js'
  import { getRoleIcon } from './roleIcons.js'

  const MAX_ITEMS = 30

  $: entryMap = new Map($leagueEntries.map(e => [e.id, e]))

  // Count total clashes between each pair (both directions)
  $: clashCounts = (() => {
    const map = new Map()
    for (const r of $leagueResults) {
      const key = [r.entry_a_id, r.entry_b_id].sort().join('-')
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

      const entryA = entryMap.get(r.entry_a_id)
      const entryB = entryMap.get(r.entry_b_id)
      const nameA = entryA?.display_name || entryA?.architecture || `#${r.entry_a_id}`
      const nameB = entryB?.display_name || entryB?.architecture || `#${r.entry_b_id}`
      const roleA = entryA?.role
      const roleB = entryB?.role
      const statusA = entryA?.status
      const statusB = entryB?.status
      const winsA = r.wins_a || 0
      const winsB = r.wins_b || 0
      const draws = r.draws || 0
      const total = winsA + winsB + draws
      const aWon = winsA > winsB
      const draw = winsA === winsB
      const clashes = pairClashes(r.entry_a_id, r.entry_b_id)

      // Always show from winner's perspective (or A's if draw)
      const winnerName = aWon || draw ? nameA : nameB
      const loserName = aWon || draw ? nameB : nameA
      const winnerRole = aWon || draw ? roleA : roleB
      const loserRole = aWon || draw ? roleB : roleA
      const winnerStatus = aWon || draw ? statusA : statusB
      const loserStatus = aWon || draw ? statusB : statusA
      const w = aWon || draw ? winsA : winsB
      const l = aWon || draw ? winsB : winsA
      const winPct = total > 0 ? Math.round((w / total) * 100) : 0
      const eloDeltaA = r.elo_after_a != null && r.elo_before_a != null ? Math.round(r.elo_after_a - r.elo_before_a) : 0
      const eloDeltaB = r.elo_after_b != null && r.elo_before_b != null ? Math.round(r.elo_after_b - r.elo_before_b) : 0
      const eloWinner = aWon || draw ? eloDeltaA : eloDeltaB
      const eloLoser = aWon || draw ? eloDeltaB : eloDeltaA

      out.push({
        type: 'match',
        ...r,
        winnerName,
        loserName,
        winnerRole,
        loserRole,
        winnerStatus,
        loserStatus,
        w, l, draws,
        total,
        aWon,
        draw,
        clashes,
        winPct,
        eloWinner,
        eloLoser,
      })
    }
    return out
  })()
</script>

<div class="recent-card">
  <h2 class="section-header">Recent Matches</h2>
  {#if items.length === 0}
    <p class="empty">No matches played yet.</p>
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
              <span class="name winner"><span class="role-icon" aria-hidden="true">{getRoleIcon(item.winnerRole, item.winnerStatus)}</span>{item.winnerName}</span>
              <span class="vs">vs</span>
              <span class="name"><span class="role-icon" aria-hidden="true">{getRoleIcon(item.loserRole, item.loserStatus)}</span>{item.loserName}</span>
              <span class="match-score" class:win={item.aWon} class:loss={!item.aWon && !item.draw} class:tied={item.draw}>
                {item.w}W {item.l}L {item.draws}D
              </span>
            </div>
            <div class="match-detail">
              <span>{item.total} games</span>
              <span class="sep">·</span>
              <span>{item.winPct}% win</span>
              <span class="sep">·</span>
              <span>clash #{item.clashes}</span>
              {#if item.eloWinner !== 0 || item.eloLoser !== 0}
                <span class="sep">·</span>
                <span class="elo-delta" class:positive={item.eloWinner > 0} class:negative={item.eloWinner < 0}>{item.eloWinner > 0 ? '+' : ''}{item.eloWinner}</span>
                <span class="elo-slash">/</span>
                <span class="elo-delta" class:positive={item.eloLoser > 0} class:negative={item.eloLoser < 0}>{item.eloLoser > 0 ? '+' : ''}{item.eloLoser}</span>
              {/if}
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

  .feed {
    overflow-y: auto;
    min-height: 0;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .epoch-separator {
    font-size: 12px;
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

  .role-icon {
    font-size: 12px;
    margin-right: 3px;
  }

  .vs {
    color: var(--text-muted);
    font-size: 12px;
    flex-shrink: 0;
  }

  .match-score {
    font-family: monospace;
    font-size: 12px;
    font-weight: 600;
    flex-shrink: 0;
    padding: 2px 6px;
    border-radius: 3px;
    margin-left: auto;
  }

  .match-score.win { color: var(--accent-teal); background: var(--badge-bg-teal); }
  .match-score.loss { color: var(--danger); background: var(--badge-bg-danger); }
  .match-score.tied { color: var(--accent-gold); background: var(--badge-bg-gold); }

  .match-detail {
    font-size: 12px;
    color: var(--text-muted);
    display: flex;
    gap: 4px;
    padding-left: 2px;
  }

  .sep {
    color: var(--border);
  }

  .elo-delta {
    font-family: monospace;
    font-weight: 600;
  }
  .elo-delta.positive { color: var(--accent-teal); }
  .elo-delta.negative { color: var(--danger); }
  .elo-slash { color: var(--border); }

  .empty {
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    padding: 24px;
  }
</style>
