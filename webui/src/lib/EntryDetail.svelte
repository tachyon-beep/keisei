<script>
  import { leagueResults, leagueEntries, headToHead, styleProfiles, eloHistory } from '../stores/league.js'
  import { getRoleInfo } from './roleIcons.js'
  import MetricsChart from './MetricsChart.svelte'

  export let entryId
  export let headingEl = undefined

  const PRIMARY_ELO = {
    frontier_static: 'elo_frontier',
    dynamic: 'elo_dynamic',
    recent_fixed: 'elo_recent',
  }
  const ELO_LABELS = {
    elo_frontier: 'Frontier',
    elo_dynamic: 'Dynamic',
    elo_recent: 'Recent',
    elo_historical: 'Historical',
  }
  const ALL_ELO_COLS = ['elo_frontier', 'elo_dynamic', 'elo_recent', 'elo_historical']

  $: entryMap = new Map($leagueEntries.map(e => [e.id, e]))
  $: entry = entryMap.get(entryId)
  $: profile = $styleProfiles.get(entryId)

  /** Secondary Elos: non-primary columns that have moved from 1000 */
  $: secondaryElos = entry ? ALL_ELO_COLS
    .filter(col => col !== PRIMARY_ELO[entry.role] && entry[col] !== 1000)
    .map(col => ({ label: ELO_LABELS[col], value: Math.round(entry[col]) }))
    : []
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

  // Elo sparkline: history for this entry + top 2 most-played opponents
  const SPARK_COLORS = ['#4ade80', '#60a5fa', '#f59e0b']
  $: sparkData = (() => {
    if (!entry) return { xData: [], series: [] }
    const myHistory = $eloHistory.filter(h => h.entry_id === entryId && h.epoch >= 0)
    if (myHistory.length < 2) return { xData: [], series: [] }

    // Build epoch set from this entry's history
    const epochSet = new Set(myHistory.map(h => h.epoch))
    // Add top 2 opponents' history (most games played against)
    const topOpps = overallOpponents.slice(0, 2).map(r => r.opponent.id)
    const oppHistories = topOpps.map(id =>
      $eloHistory.filter(h => h.entry_id === id && h.epoch >= 0)
    )
    for (const oh of oppHistories) {
      for (const h of oh) epochSet.add(h.epoch)
    }
    const xData = [...epochSet].sort((a, b) => a - b)
    const epochIndex = new Map(xData.map((e, i) => [e, i]))

    const buildSeries = (history, label, color) => {
      const data = new Array(xData.length).fill(null)
      for (const h of history) {
        const idx = epochIndex.get(h.epoch)
        if (idx != null) data[idx] = h.elo_rating
      }
      return { label, data, color }
    }

    const series = [
      buildSeries(myHistory, entry.display_name || entry.architecture, SPARK_COLORS[0]),
    ]
    topOpps.forEach((id, i) => {
      const opp = entryMap.get(id)
      if (opp) {
        series.push(buildSeries(
          oppHistories[i],
          opp.display_name || opp.architecture,
          SPARK_COLORS[i + 1],
        ))
      }
    })
    return { xData, series }
  })()

  function matchPerspective(m) {
    const isA = m.entry_a_id === entryId
    const oppId = isA ? m.entry_b_id : m.entry_a_id
    const opp = entryMap.get(oppId)
    const w = isA ? (m.wins_a || 0) : (m.wins_b || 0)
    const l = isA ? (m.wins_b || 0) : (m.wins_a || 0)
    const d = m.draws || 0
    const myElo = isA ? m.elo_before_a : m.elo_before_b
    const oppElo = isA ? m.elo_before_b : m.elo_before_a
    const won = w > l
    const draw = w === l
    // Upset: this entry won despite being 100+ Elo below opponent
    const upset = won && myElo != null && oppElo != null && oppElo - myElo >= 100
    // Opponent upset: opponent won despite being 100+ Elo below this entry
    const oppUpset = !won && !draw && myElo != null && oppElo != null && myElo - oppElo >= 100
    return {
      opponent: opp,
      w, l, d,
      oppElo: oppElo != null ? Math.round(oppElo) : null,
      upset: upset || oppUpset,
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
    <h3 class="detail-heading" tabindex="-1" bind:this={headingEl}>
      <span class="role-icon" aria-hidden="true">{getRoleInfo(entry.role, entry.status).icon}</span>
      {entry.display_name || entry.architecture}
      <span class="heading-elo">{Math.round(entry.elo_rating)}</span>
    </h3>

    <div class="detail-sections">
      {#if sparkData.xData.length > 0}
        <div class="detail-section spark-section">
          <h4 class="section-label">Elo Trend</h4>
          <div class="spark-chart">
            <MetricsChart
              title=""
              xData={sparkData.xData}
              series={sparkData.series}
              height={160}
              xLabel="Epoch"
            />
          </div>
        </div>
      {/if}

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
                    <span class="role-icon" title={getRoleInfo(m.opponent.role, m.opponent.status).tooltip} aria-hidden="true">{getRoleInfo(m.opponent.role, m.opponent.status).icon}</span>
                    {m.opponent.display_name || m.opponent.architecture}{#if m.oppElo != null}<span class="name-elo">({m.oppElo})</span>{/if}
                  </span>
                  {#if m.upset}<span class="upset-badge" title="Upset: lower-rated player won">!</span>{/if}
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
                  <span class="role-icon" title={getRoleInfo(rec.opponent.role, rec.opponent.status).tooltip} aria-hidden="true">{getRoleInfo(rec.opponent.role, rec.opponent.status).icon}</span>
                  {rec.opponent.display_name || rec.opponent.architecture}<span class="name-elo">({Math.round(rec.opponent.elo_rating)})</span>
                </span>
                <span class="wld">{rec.w}W {rec.l}L {rec.d}D</span>
                <span class="win-pct">{rec.total > 0 ? Math.round(rec.winRate * 100) : 0}%</span>
                <span class="game-count">{rec.total}g</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      {#if secondaryElos.length > 0}
        <div class="detail-section role-stats">
          <h4 class="section-label">Other Ratings</h4>
          <div class="stat-row">
            {#each secondaryElos as elo}
              <span class="mini-stat"><span class="mini-label">{elo.label}</span> {elo.value}</span>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .entry-detail { padding: 14px 18px; }
  .detail-heading {
    font-size: 16px; font-weight: 700; color: var(--text-primary);
    margin: 0 0 12px; display: flex; align-items: center; gap: 6px;
    padding-right: 28px; /* space for close button */
  }
  .detail-heading:focus { outline: none; }
  .detail-heading:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .heading-elo { font-family: monospace; font-weight: 400; color: var(--text-muted); font-size: 14px; }
  .detail-sections { display: flex; flex-direction: column; gap: 16px; }
  .detail-section { min-width: 0; padding-bottom: 14px; border-bottom: 1px solid var(--border); }
  .detail-section:last-child { border-bottom: none; padding-bottom: 0; }
  .section-label {
    font-size: 13px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--text-muted); margin: 0 0 8px;
  }
  .epoch-tag {
    font-weight: 400; color: var(--text-muted); font-size: 13px;
    margin-left: 6px; text-transform: none; letter-spacing: 0;
  }
  .match-list { display: flex; flex-direction: column; gap: 4px; }
  .match-row {
    display: flex; align-items: center; gap: 10px;
    font-size: 14px; padding: 4px 6px; border-radius: 4px;
  }
  .match-row:hover { background: var(--bg-card); }
  .opp-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-primary); }
  .name-elo { font-family: monospace; font-size: 12px; color: var(--text-muted); font-weight: 400; margin-left: 3px; }
  .upset-badge { font-size: 12px; font-weight: 700; color: var(--accent-gold); opacity: 0.7; flex-shrink: 0; }
  .role-icon { font-size: 14px; margin-right: 4px; }
  .wld { font-family: monospace; font-size: 14px; color: var(--text-secondary); flex-shrink: 0; }
  .elo-delta { font-family: monospace; font-size: 14px; font-weight: 600; flex-shrink: 0; min-width: 40px; text-align: right; }
  .elo-delta.positive { color: var(--accent-teal); }
  .elo-delta.negative { color: var(--danger); }
  .win-pct { font-family: monospace; font-size: 14px; color: var(--text-muted); flex-shrink: 0; }
  .game-count { font-size: 14px; color: var(--text-muted); flex-shrink: 0; }
  .role-stats { }
  .stat-row { display: flex; gap: 14px; flex-wrap: wrap; }
  .mini-stat { font-family: monospace; font-size: 14px; color: var(--text-primary); }
  .mini-label { font-size: 13px; color: var(--text-muted); margin-right: 4px; font-family: inherit; }
  .style-section { }
  .style-primary { font-size: 15px; font-weight: 600; color: var(--accent-teal); margin-bottom: 6px; }
  .style-traits { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
  .style-trait {
    font-size: 13px; padding: 2px 8px; border-radius: 3px;
    color: var(--text-secondary); background: rgba(128, 128, 128, 0.12);
  }
  .commentary-list { display: flex; flex-direction: column; gap: 4px; }
  .commentary-item {
    font-size: 13px; font-style: italic; color: var(--text-muted); padding: 2px 0;
  }
  .commentary-item.high-conf { color: var(--text-secondary); }
  .spark-section { }
  .spark-chart { height: 160px; }
  .spark-chart :global(.chart-wrapper) {
    border: none;
    padding: 0;
    background: transparent;
  }
  .empty { color: var(--text-muted); font-size: 14px; text-align: center; padding: 24px; }
  .empty-small { color: var(--text-muted); font-size: 13px; padding: 10px 0; }
</style>
