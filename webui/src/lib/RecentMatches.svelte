<script>
  import { leagueResults, leagueEntries } from '../stores/league.js'

  const MAX_ITEMS = 20

  $: entryMap = new Map($leagueEntries.map(e => [e.id, e]))

  $: recent = $leagueResults
    .slice(0, MAX_ITEMS)
    .map(r => {
      const learner = entryMap.get(r.learner_id)
      const opponent = entryMap.get(r.opponent_id)
      const total = (r.wins || 0) + (r.losses || 0) + (r.draws || 0)
      const learnerWon = (r.wins || 0) > (r.losses || 0)
      const draw = (r.wins || 0) === (r.losses || 0)
      return {
        ...r,
        learnerName: learner?.display_name || learner?.architecture || `#${r.learner_id}`,
        opponentName: opponent?.display_name || opponent?.architecture || `#${r.opponent_id}`,
        total,
        learnerWon,
        draw,
        summary: `${r.wins}W ${r.losses}L ${r.draws}D`,
      }
    })
</script>

<div class="recent-card">
  <h2 class="section-header">Recent Matches</h2>
  {#if recent.length === 0}
    <p class="empty">No matches yet.</p>
  {:else}
    <div class="feed">
      {#each recent as m}
        <div class="match-item">
          <span class="epoch-badge">E{m.epoch}</span>
          <span class="match-names">
            <span class="name" class:winner={m.learnerWon}>{m.learnerName}</span>
            <span class="vs">vs</span>
            <span class="name" class:winner={!m.learnerWon && !m.draw}>{m.opponentName}</span>
          </span>
          <span class="match-score" class:win={m.learnerWon} class:loss={!m.learnerWon && !m.draw} class:tied={m.draw}>
            {m.summary}
          </span>
        </div>
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

  .match-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 8px;
    border-radius: 4px;
    font-size: 12px;
  }

  .match-item:hover {
    background: var(--bg-card);
  }

  .epoch-badge {
    font-size: 10px;
    font-weight: 700;
    font-family: monospace;
    color: var(--text-muted);
    background: var(--bg-primary);
    padding: 2px 5px;
    border-radius: 3px;
    flex-shrink: 0;
    min-width: 32px;
    text-align: center;
  }

  .match-names {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    overflow: hidden;
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
  }

  .match-score.win { color: var(--accent-teal); background: rgba(77, 184, 168, 0.1); }
  .match-score.loss { color: var(--danger); background: rgba(224, 80, 80, 0.1); }
  .match-score.tied { color: var(--accent-gold); background: rgba(200, 150, 46, 0.1); }

  .empty {
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    padding: 24px;
  }
</style>
