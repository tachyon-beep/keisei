<script>
  import { leagueResults, leagueEntries, headToHead, styleProfiles } from '../stores/league.js'
  import { getRoleInfo } from './roleIcons.js'

  export let entryId
  export let headingEl = undefined

  $: entryMap = new Map($leagueEntries.map(e => [e.id, e]))
  $: entry = entryMap.get(entryId)
  $: profile = $styleProfiles.get(entryId)
  $: hasProfile = profile && profile.profile_status !== 'insufficient'

  // Last Round: matches from the entry's most recent epoch
  $: entryMatches = $leagueResults.filter(
    r => r.entry_a_id === entryId || r.entry_b_id === entryId
  )
  $: maxEpoch = entryMatches.length > 0
    ? Math.max(...entryMatches.map(r => r.epoch))
    : null
  $: lastRound = maxEpoch != null
    ? entryMatches.filter(r => r.epoch === maxEpoch)
    : []

  // Overall Record: aggregate by opponent from headToHead store
  $: overallOpponents = (() => {
    const opponents = []
    for (const [key, rec] of $headToHead) {
      const [aId, bId] = key.split('-').map(Number)
      if (aId === entryId) {
        const opp = entryMap.get(bId)
        if (opp) {
          opponents.push({ ...rec, opponent: opp })
        }
      }
    }
    return opponents.sort((a, b) => b.total - a.total)
  })()

  function matchPerspective(m) {
    const isA = m.entry_a_id === entryId
    const oppId = isA ? m.entry_b_id : m.entry_a_id
    const opp = entryMap.get(oppId)
    return {
      opponent: opp,
      w: isA ? (m.wins_a || 0) : (m.wins_b || 0),
      l: isA ? (m.wins_b || 0) : (m.wins_a || 0),
      d: m.draws || 0,
      eloDelta: isA
        ? Math.round((m.elo_after_a || 0) - (m.elo_before_a || 0))
        : Math.round((m.elo_after_b || 0) - (m.elo_before_b || 0)),
    }
  }
</script>

<div class="entry-detail">
  <p class="sr-only" aria-live="polite">{entry ? `Viewing ${entry.display_name || entry.architecture}` : ''}</p>
  {#if !entry}
    <p class="empty">Select an entry to view details</p>
  {:else}
    <h3 class="sr-only" tabindex="-1" bind:this={headingEl}>{entry.display_name || entry.architecture} — Details</h3>

    <div class="detail-sections">
      <div class="detail-section">
        <h4 class="section-label">Last Round {#if maxEpoch != null}<span class="epoch-tag">Epoch {maxEpoch}</span>{/if}</h4>
        {#if lastRound.length === 0}
          <p class="empty-small">No matches in the current round</p>
        {:else}
          <div class="match-list">
            {#each lastRound.map(matchPerspective) as m}
              {#if m.opponent}
                <div class="match-row">
                  <span class="opp-name">
                    <span class="role-icon" aria-hidden="true">{getRoleInfo(m.opponent.role, m.opponent.status).icon}</span>
                    {m.opponent.display_name || m.opponent.architecture}
                  </span>
                  <span class="wld">{m.w}W {m.l}L {m.d}D</span>
                  <span class="elo-delta" class:positive={m.eloDelta > 0} class:negative={m.eloDelta < 0}>
                    {m.eloDelta > 0 ? '+' : ''}{m.eloDelta}
                  </span>
                </div>
              {/if}
            {/each}
          </div>
        {/if}
      </div>

      <div class="detail-section">
        <h4 class="section-label">Overall Record</h4>
        {#if overallOpponents.length === 0}
          <p class="empty-small">No match history</p>
        {:else}
          <div class="match-list">
            {#each overallOpponents as rec}
              <div class="match-row">
                <span class="opp-name">
                  <span class="role-icon" aria-hidden="true">{getRoleInfo(rec.opponent.role, rec.opponent.status).icon}</span>
                  {rec.opponent.display_name || rec.opponent.architecture}
                </span>
                <span class="wld">{rec.w}W {rec.l}L {rec.d}D</span>
                <span class="win-pct">{rec.total > 0 ? Math.round(rec.winRate * 100) : 0}%</span>
                <span class="game-count">{rec.total}g</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      {#if hasProfile}
        <div class="detail-section style-section">
          <h4 class="section-label">Play Style {#if profile.profile_status === 'provisional'}<span class="epoch-tag">(provisional)</span>{/if}</h4>
          {#if profile.primary_style}
            <div class="style-primary">{profile.primary_style}</div>
          {/if}
          {#if profile.secondary_traits?.length}
            <div class="style-traits">
              {#each profile.secondary_traits as trait}
                <span class="style-trait">{trait}</span>
              {/each}
            </div>
          {/if}
          {#if profile.commentary?.length}
            <div class="commentary-list">
              {#each profile.commentary as fact}
                <div class="commentary-item" class:high-conf={fact.confidence === 'high'}>{fact.text}</div>
              {/each}
            </div>
          {/if}
        </div>
      {/if}

      {#if entry}
        <div class="detail-section role-stats">
          <h4 class="section-label">Role-Specific</h4>
          <div class="stat-row">
            <span class="mini-stat"><span class="mini-label">Frontier</span> {entry.elo_frontier != null ? Math.round(entry.elo_frontier) : '—'}</span>
            <span class="mini-stat"><span class="mini-label">Dynamic</span> {entry.elo_dynamic != null ? Math.round(entry.elo_dynamic) : '—'}</span>
            <span class="mini-stat"><span class="mini-label">Recent</span> {entry.elo_recent != null ? Math.round(entry.elo_recent) : '—'}</span>
            <span class="mini-stat"><span class="mini-label">Historical</span> {entry.elo_historical != null ? Math.round(entry.elo_historical) : '—'}</span>
          </div>
          {#if entry.games_vs_frontier != null}
            <div class="stat-row games">
              <span class="mini-stat"><span class="mini-label">vs Frontier</span> {entry.games_vs_frontier}</span>
              <span class="mini-stat"><span class="mini-label">vs Dynamic</span> {entry.games_vs_dynamic}</span>
              <span class="mini-stat"><span class="mini-label">vs Recent</span> {entry.games_vs_recent}</span>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .entry-detail { padding: 10px 14px; }
  .detail-sections { display: flex; gap: 16px; flex-wrap: wrap; }
  .detail-section { flex: 1; min-width: 200px; }
  .section-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--text-muted); margin: 0 0 6px;
  }
  .epoch-tag {
    font-weight: 400; color: var(--text-muted); font-size: 10px;
    margin-left: 6px; text-transform: none; letter-spacing: 0;
  }
  .match-list { display: flex; flex-direction: column; gap: 2px; }
  .match-row {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; padding: 2px 4px; border-radius: 3px;
  }
  .match-row:hover { background: var(--bg-card); }
  .opp-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-primary); }
  .role-icon { font-size: 10px; margin-right: 3px; }
  .wld { font-family: monospace; font-size: 11px; color: var(--text-secondary); flex-shrink: 0; }
  .elo-delta { font-family: monospace; font-size: 11px; font-weight: 600; flex-shrink: 0; min-width: 36px; text-align: right; }
  .elo-delta.positive { color: var(--accent-teal); }
  .elo-delta.negative { color: var(--danger); }
  .win-pct { font-family: monospace; font-size: 11px; color: var(--text-muted); flex-shrink: 0; }
  .game-count { font-size: 10px; color: var(--text-muted); flex-shrink: 0; }
  .role-stats { min-width: 100%; }
  .stat-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .stat-row.games { margin-top: 4px; }
  .mini-stat { font-family: monospace; font-size: 12px; color: var(--text-primary); }
  .mini-label { font-size: 10px; color: var(--text-muted); margin-right: 4px; font-family: inherit; }
  .style-section { min-width: 200px; }
  .style-primary { font-size: 13px; font-weight: 600; color: var(--accent-teal); margin-bottom: 4px; }
  .style-traits { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }
  .style-trait {
    font-size: 11px; padding: 1px 6px; border-radius: 3px;
    color: var(--text-secondary); background: rgba(128, 128, 128, 0.12);
  }
  .commentary-list { display: flex; flex-direction: column; gap: 2px; }
  .commentary-item {
    font-size: 11px; font-style: italic; color: var(--text-muted); padding: 1px 0;
  }
  .commentary-item.high-conf { color: var(--text-secondary); }
  .empty { color: var(--text-muted); font-size: 13px; text-align: center; padding: 24px; }
  .empty-small { color: var(--text-muted); font-size: 12px; padding: 8px 0; }
</style>
