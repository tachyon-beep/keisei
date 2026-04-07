<script>
  import { tick } from 'svelte'
  import { leagueRanked, entryWLD, eloDelta, focusedEntryId, leagueByRole, styleProfiles, displayElo } from '../stores/league.js'
  import { getRoleInfo } from './roleIcons.js'

  /** Current learner's display_name (used to highlight their row) */
  export let learnerName = null
  /** Total slots shown in leaderboard (empty ones are placeholders) */
  export let totalSlots = 20

  const ROLE_CAPACITY = { frontier_static: 5, recent_fixed: 5, dynamic: 10, historical: 5 }
  const ROLE_ORDER = ['frontier_static', 'recent_fixed', 'dynamic', 'historical', 'other']
  const ROLE_LABELS = {
    frontier_static: '🛡 Frontier',
    recent_fixed: '✦ Recent',
    dynamic: '⚔ Dynamic',
    historical: '📜 Historical',
    other: '? Other',
  }

  let viewMode = 'flat' // 'flat' | 'grouped'

  function handleViewToggleKeydown(e) {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault()
      viewMode = viewMode === 'flat' ? 'grouped' : 'flat'
      tick().then(() => {
        document.querySelector('.view-toggle [role="radio"][tabindex="0"]')?.focus()
      })
    }
  }

  let sortColumn = 'elo_rating'
  let sortAsc = false

  $: wld = $entryWLD
  $: deltas = $eloDelta

  $: sorted = (() => {
    const entries = [...$leagueRanked]
    entries.sort((a, b) => {
      let av, bv
      if (sortColumn === 'wins') {
        av = (wld.get(a.id)?.w || 0); bv = (wld.get(b.id)?.w || 0)
      } else if (sortColumn === 'losses') {
        av = (wld.get(a.id)?.l || 0); bv = (wld.get(b.id)?.l || 0)
      } else if (sortColumn === 'draws') {
        av = (wld.get(a.id)?.d || 0); bv = (wld.get(b.id)?.d || 0)
      } else if (sortColumn === 'elo_delta') {
        av = (deltas.get(a.id) || 0); bv = (deltas.get(b.id) || 0)
      } else if (sortColumn === 'elo_rating') {
        av = displayElo(a).value; bv = displayElo(b).value
      } else {
        av = a[sortColumn]; bv = b[sortColumn]
      }
      return sortAsc ? (av > bv ? 1 : -1) : (bv > av ? 1 : -1)
    })
    return entries.map((e, i) => ({ ...e, rank: i + 1 }))
  })()

  function toggleSort(col) {
    if (sortColumn === col) {
      sortAsc = !sortAsc
    } else {
      sortColumn = col
      sortAsc = false
    }
  }

  function toggleExpand(id) {
    focusedEntryId.update(current => current === id ? null : id)
  }

  function sortIndicator(col) {
    if (sortColumn !== col) return ''
    return sortAsc ? ' ▲' : ' ▼'
  }

  function ariaSortValue(col) {
    if (sortColumn !== col) return 'none'
    return sortAsc ? 'ascending' : 'descending'
  }

  function isLearner(entry) {
    return learnerName && entry.display_name === learnerName
  }

  $: profiles = $styleProfiles

  function getStyle(entryId) {
    const p = profiles.get(entryId)
    if (!p || p.profile_status === 'insufficient') return null
    return p.primary_style || null
  }

  function getStyleTooltip(entryId) {
    const p = profiles.get(entryId)
    if (!p || p.profile_status === 'insufficient') return ''
    const parts = []
    if (p.primary_style) parts.push(p.primary_style)
    const traits = p.secondary_traits || []
    if (traits.length) parts.push(traits.join(', '))
    const m = p.raw_metrics || {}
    if (m.avg_game_length != null) parts.push(`Avg length: ${Math.round(m.avg_game_length)}`)
    if (m.drops_per_game != null) parts.push(`Drops/game: ${m.drops_per_game.toFixed(1)}`)
    return parts.join(' | ')
  }

  $: placeholderCount = Math.max(0, totalSlots - sorted.length)
  $: placeholders = Array.from({ length: placeholderCount }, (_, i) => sorted.length + i + 1)
</script>

<div class="league-table-card">
  <h2 class="section-header">Elo Leaderboard {#if placeholderCount > 0}<span class="slot-count">{sorted.length} / {totalSlots}</span>{/if}</h2>
  <div class="view-toggle" role="radiogroup" aria-label="Leaderboard view">
    <button role="radio" aria-checked={viewMode === 'flat'} tabindex={viewMode === 'flat' ? 0 : -1} on:click={() => viewMode = 'flat'} on:keydown={handleViewToggleKeydown} class:active={viewMode === 'flat'}>Flat</button>
    <button role="radio" aria-checked={viewMode === 'grouped'} tabindex={viewMode === 'grouped' ? 0 : -1} on:click={() => viewMode = 'grouped'} on:keydown={handleViewToggleKeydown} class:active={viewMode === 'grouped'}>Grouped</button>
  </div>
    <div class="table-scroll">
      <table>
        <caption class="sr-only">Elo leaderboard, sorted by {sortColumn} {sortAsc ? 'ascending' : 'descending'}</caption>
        <thead>
          <tr>
            <th class="num" aria-sort="none">#</th>
            <th aria-sort={ariaSortValue('display_name')}>
              <button class="sort-btn" title="Sort by name" on:click={() => toggleSort('display_name')}>Name{sortIndicator('display_name')}</button>
            </th>
            <th class="num" aria-sort={ariaSortValue('elo_rating')}>
              <button class="sort-btn" title="Sort by Elo rating" on:click={() => toggleSort('elo_rating')}>Elo{sortIndicator('elo_rating')}</button>
            </th>
            <th class="num" aria-sort={ariaSortValue('elo_delta')}>
              <button class="sort-btn" title="Sort by Elo change" on:click={() => toggleSort('elo_delta')}>±{sortIndicator('elo_delta')}</button>
            </th>
            <th class="num" aria-sort={ariaSortValue('games_played')}>
              <button class="sort-btn" title="Sort by games played" on:click={() => toggleSort('games_played')}>GP{sortIndicator('games_played')}</button>
            </th>
            <th class="num wld-col" aria-sort={ariaSortValue('wins')}>
              <button class="sort-btn" title="Sort by wins" on:click={() => toggleSort('wins')}>W{sortIndicator('wins')}</button>
            </th>
            <th class="num wld-col" aria-sort={ariaSortValue('losses')}>
              <button class="sort-btn" title="Sort by losses" on:click={() => toggleSort('losses')}>L{sortIndicator('losses')}</button>
            </th>
            <th class="num wld-col" aria-sort={ariaSortValue('draws')}>
              <button class="sort-btn" title="Sort by draws" on:click={() => toggleSort('draws')}>D{sortIndicator('draws')}</button>
            </th>
            <th class="style-col" aria-sort="none">Style</th>
            <th class="num" aria-sort={ariaSortValue('created_epoch')}>
              <button class="sort-btn" title="Sort by creation epoch" on:click={() => toggleSort('created_epoch')}>Epoch{sortIndicator('created_epoch')}</button>
            </th>
            <th class="expand-col" aria-sort="none"></th>
          </tr>
        </thead>
        <tbody>
          {#if viewMode === 'grouped'}
            {#each ROLE_ORDER as role}
              {#if $leagueByRole.has(role)}
                <tr class="group-header">
                  <td colspan="11" class="group-heading" role="separator">
                    <span aria-hidden="true">{ROLE_LABELS[role]?.split(' ')[0]}</span>
                    {ROLE_LABELS[role]?.split(' ').slice(1).join(' ') || role} · {$leagueByRole.get(role).length}/{ROLE_CAPACITY[role] || '?'}
                  </td>
                </tr>
                {#each $leagueByRole.get(role) as entry}
                  <tr
                    class:top={entry.rank === 1}
                    class:learner={isLearner(entry)}
                    class:focused={$focusedEntryId === entry.id}
                    aria-label="Rank {entry.rank}: {entry.display_name || entry.architecture}, {displayElo(entry).tag || 'Elo'} {Math.round(displayElo(entry).value)}, {wld.get(entry.id)?.w || 0}W {wld.get(entry.id)?.l || 0}L {wld.get(entry.id)?.d || 0}D"
                    on:click={() => toggleExpand(entry.id)}
                    tabindex="0"
                    on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleExpand(entry.id) }}}
                  >
                    <td class="num rank">{entry.rank}</td>
                    <td class="name-cell">
                      {entry.display_name || entry.architecture}
                      {#if isLearner(entry)}<span class="learner-badge" title="This policy is currently being trained against the other entries in the league">TRAINEE</span>{/if}
                      {#if entry.protection_remaining > 0}
                        <span class="protection-badge" title="Protected from retirement for {entry.protection_remaining} more epochs" aria-label="Protected for {entry.protection_remaining} epochs"><span aria-hidden="true">🛡</span> {entry.protection_remaining}</span>
                      {/if}
                    </td>
                    <td class="num elo">{Math.round(displayElo(entry).value)}{#if displayElo(entry).tag} <span class="elo-tag">{displayElo(entry).tag}</span>{/if}</td>
                    <td class="num delta" class:delta-pos={(deltas.get(entry.id) || 0) > 0} class:delta-neg={(deltas.get(entry.id) || 0) < 0}>
                      {(deltas.get(entry.id) || 0) > 0 ? '+' : ''}{deltas.get(entry.id) || 0}
                    </td>
                    <td class="num">{entry.games_played}</td>
                    <td class="num win">{wld.get(entry.id)?.w || 0}</td>
                    <td class="num loss">{wld.get(entry.id)?.l || 0}</td>
                    <td class="num draw">{wld.get(entry.id)?.d || 0}</td>
                    <td class="style-cell" title={getStyleTooltip(entry.id)}>{getStyle(entry.id) || ''}</td>
                    <td class="num">{entry.created_epoch}</td>
                    <td class="expand-chevron" aria-hidden="true">{$focusedEntryId === entry.id ? '▾' : '▸'}</td>
                  </tr>
                {/each}
                {#each Array(Math.max(0, (ROLE_CAPACITY[role] || 0) - ($leagueByRole.get(role)?.length || 0))) as _, i}
                  <tr class="placeholder-row" aria-hidden="true">
                    <td class="num rank placeholder-text">{($leagueByRole.get(role)?.length || 0) + i + 1}</td>
                    <td class="placeholder-text">—</td>
                    <td class="num placeholder-text">—</td>
                    <td class="num placeholder-text"></td>
                    <td class="num placeholder-text"></td>
                    <td class="num placeholder-text"></td>
                    <td class="num placeholder-text"></td>
                    <td class="num placeholder-text"></td>
                    <td class="num placeholder-text"></td>
                    <td class="num placeholder-text"></td>
                    <td class="placeholder-text"></td>
                  </tr>
                {/each}
              {/if}
            {/each}
          {:else}
            {#each sorted as entry}
              <tr
                class:top={entry.rank === 1}
                class:learner={isLearner(entry)}
                class:focused={$focusedEntryId === entry.id}
                aria-expanded={$focusedEntryId === entry.id}
                aria-label="Rank {entry.rank}: {entry.display_name || entry.architecture}, {displayElo(entry).tag || 'Elo'} {Math.round(displayElo(entry).value)}, {wld.get(entry.id)?.w || 0}W {wld.get(entry.id)?.l || 0}L {wld.get(entry.id)?.d || 0}D"
                on:click={() => toggleExpand(entry.id)}
                tabindex="0"
                on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleExpand(entry.id) }}}
              >
                <td class="num rank">
                  {#if entry.rank === 1}
                    <span class="crown" aria-hidden="true">♛</span>
                    <span class="sr-only">1 (champion)</span>
                  {:else}
                    {entry.rank}
                  {/if}
                </td>
                <td class="name-cell">
                  <span class="role-badge {getRoleInfo(entry.role).cssClass}" title={getRoleInfo(entry.role).tooltip} aria-label="{getRoleInfo(entry.role).label} tier">{getRoleInfo(entry.role).icon}</span>
                  {entry.display_name || entry.architecture}
                  {#if isLearner(entry)}
                    <span class="learner-badge" title="This policy is currently being trained against the other entries in the league">TRAINEE</span>
                  {/if}
                  {#if entry.protection_remaining > 0}
                    <span class="protection-badge" title="Protected from retirement for {entry.protection_remaining} more epochs" aria-label="Protected for {entry.protection_remaining} epochs"><span aria-hidden="true">🛡</span> {entry.protection_remaining}</span>
                  {/if}
                </td>
                <td class="num elo">{Math.round(displayElo(entry).value)}{#if displayElo(entry).tag} <span class="elo-tag">{displayElo(entry).tag}</span>{/if}</td>
                <td class="num delta" class:delta-pos={(deltas.get(entry.id) || 0) > 0} class:delta-neg={(deltas.get(entry.id) || 0) < 0}>
                  {(deltas.get(entry.id) || 0) > 0 ? '+' : ''}{deltas.get(entry.id) || 0}
                </td>
                <td class="num">{entry.games_played}</td>
                <td class="num win">{wld.get(entry.id)?.w || 0}</td>
                <td class="num loss">{wld.get(entry.id)?.l || 0}</td>
                <td class="num draw">{wld.get(entry.id)?.d || 0}</td>
                <td class="style-cell" title={getStyleTooltip(entry.id)}>{getStyle(entry.id) || ''}</td>
                <td class="num">{entry.created_epoch}</td>
                <td class="expand-chevron" aria-hidden="true">{$focusedEntryId === entry.id ? '▾' : '▸'}</td>
              </tr>
            {/each}
            {#each placeholders as slot}
              <tr class="placeholder-row" aria-hidden="true">
                <td class="num rank placeholder-text">{slot}</td>
                <td class="placeholder-text">—</td>
                <td class="num placeholder-text">—</td>
                <td class="num placeholder-text"></td>
                <td class="num placeholder-text"></td>
                <td class="num placeholder-text"></td>
                <td class="num placeholder-text"></td>
                <td class="num placeholder-text"></td>
                <td class="num placeholder-text"></td>
                <td class="num placeholder-text"></td>
                <td class="placeholder-text"></td>
              </tr>
            {/each}
          {/if}
        </tbody>
      </table>
    </div>
</div>

<style>
  .league-table-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .table-scroll {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
  }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead { color: var(--text-muted); font-size: 13px; position: sticky; top: 0; background: var(--bg-secondary); z-index: 1; }
  th, td { text-align: left; padding: 6px 10px; }
  th.num, td.num { text-align: right; }
  th.wld-col { min-width: 36px; }

  .sort-btn {
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
  }
  .sort-btn:hover {
    color: var(--text-primary);
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .sort-btn:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }

  tbody tr {
    border-bottom: 1px solid var(--border-subtle);
    cursor: pointer;
    color: var(--text-primary);
    transition: background 0.1s;
  }
  tbody tr:hover { background: var(--bg-card); }
  tbody tr:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: -2px; }

  /* Rank #1 row accent */
  tr.top {
    border-left: 3px solid var(--accent-gold);
  }
  tr.top .rank { color: var(--accent-gold); font-weight: 700; }
  tr.top .elo { color: var(--accent-teal); font-weight: 700; }

  .crown {
    color: var(--accent-gold);
    font-size: 15px;
  }

  /* Current learner highlight */
  tr.learner {
    background: rgba(77, 184, 168, 0.08);
    border-left: 3px solid var(--accent-teal);
  }
  tr.learner:hover {
    background: rgba(77, 184, 168, 0.14);
  }

  .learner-badge {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: var(--accent-teal);
    background: var(--badge-bg-teal);
    padding: 1px 5px;
    border-radius: 3px;
    margin-left: 6px;
    vertical-align: middle;
  }

  .name-cell {
    white-space: nowrap;
    width: 28%;
  }

  .role-badge {
    display: inline-block;
    font-size: 12px;
    width: 18px;
    text-align: center;
    margin-right: 4px;
    vertical-align: middle;
    border-radius: 3px;
    padding: 1px 0;
  }
  .role-badge.role-frontier { color: #7b8fa8; }
  .role-badge.role-recent { color: #c8962e; }
  .role-badge.role-dynamic { color: var(--accent-teal); }
  .role-badge.role-historical { color: #9b7ec8; }
  .role-badge.role-unknown { color: var(--text-muted); }
  .role-badge.role-retired { color: #888; opacity: 0.6; }

  tr.focused { background: var(--bg-card); }

  .elo { font-family: monospace; white-space: nowrap; }
  .elo-tag { font-size: 10px; font-weight: 600; color: var(--text-muted); margin-left: 2px; }
  .delta { font-family: monospace; font-size: 12px; color: var(--text-muted); }
  .delta-pos { color: var(--accent-teal); }
  .delta-neg { color: var(--danger); }
  .win { color: var(--accent-teal); font-family: monospace; }
  .loss { color: var(--danger); font-family: monospace; }
  .draw { color: var(--accent-gold); font-family: monospace; }

  .slot-count {
    font-weight: 400;
    color: var(--text-muted);
    font-size: 12px;
    margin-left: 6px;
  }

  .placeholder-row {
    cursor: default;
  }
  .placeholder-row:hover { background: transparent; }
  .placeholder-text {
    color: var(--text-muted);
    opacity: 0.5;
    font-size: 12px;
    user-select: none;
  }

  .style-col { font-size: 12px; min-width: 80px; }
  .style-cell {
    font-size: 12px;
    color: var(--accent-teal);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 140px;
    cursor: help;
  }

  .empty { color: var(--text-muted); font-size: 13px; padding: 24px; text-align: center; }

  .view-toggle {
    display: flex;
    gap: 2px;
    margin-bottom: 8px;
  }
  .view-toggle button {
    padding: 6px 14px;
    min-height: 36px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: none;
    color: var(--text-muted);
    cursor: pointer;
  }
  .view-toggle button.active {
    background: var(--bg-card);
    color: var(--text-primary);
    border-color: var(--accent-teal);
  }
  .view-toggle button:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }

  .group-heading {
    padding: 10px 10px 6px;
    font-size: 12px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-primary);
  }

  .protection-badge {
    font-size: 12px;
    color: var(--text-muted);
    margin-left: 4px;
    vertical-align: middle;
  }

  .expand-col { width: 24px; }
  .expand-chevron {
    text-align: center;
    color: var(--text-muted);
    font-size: 12px;
    width: 24px;
    padding: 6px 4px;
  }

  @media (prefers-reduced-motion: reduce) {
    tbody tr { transition: none; }
  }
</style>
