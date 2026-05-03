<script>
  import { leagueEntries, headToHead, displayElo } from '../stores/league.js'
  import { getRoleInfo } from './roleIcons.js'

  /** Showcase game payload from backend (showcase_games row + queue context). */
  export let game
  /** The move currently being displayed (live tail OR scrubbed-to ply). */
  export let displayedMove = null
  /** True when user is scrubbing through history rather than watching live. */
  export let scrubbing = false
  /** Estimated typical game length in plies, used for the progress bar. */
  export let estimatedTotalPly = 140

  // Lookup the league entries by entry_id so we can decorate each player with
  // tier badge and architecture. entry_id_black/white are TEXT in the DB so
  // we string-coerce both sides of the comparison.
  $: entryById = (() => {
    const map = new Map()
    for (const e of $leagueEntries) map.set(String(e.id), e)
    return map
  })()
  $: blackEntry = game ? entryById.get(String(game.entry_id_black)) : null
  $: whiteEntry = game ? entryById.get(String(game.entry_id_white)) : null

  $: blackRole = blackEntry ? getRoleInfo(blackEntry.role, blackEntry.status) : null
  $: whiteRole = whiteEntry ? getRoleInfo(whiteEntry.role, whiteEntry.status) : null

  $: blackArchSummary = archSummary(blackEntry)
  $: whiteArchSummary = archSummary(whiteEntry)

  function archSummary(entry) {
    if (!entry) return null
    const mp = entry.model_params || {}
    const parts = []
    if (mp.num_blocks && mp.channels) parts.push(`b${mp.num_blocks}c${mp.channels}`)
    if (mp.se_reduction) parts.push(`SE-${mp.se_reduction}`)
    return { arch: entry.architecture || '', topology: parts.join(' · ') }
  }

  // Use display ELO when available (handles tier-specific ratings); fall back
  // to the snapshot ELO captured on the showcase game row.
  $: blackElo = blackEntry ? Math.round(displayElo(blackEntry).value) : (game?.elo_black != null ? Math.round(game.elo_black) : null)
  $: whiteElo = whiteEntry ? Math.round(displayElo(whiteEntry).value) : (game?.elo_white != null ? Math.round(game.elo_white) : null)

  // Head-to-head for these two players (canonical key built by store).
  $: h2h = (() => {
    if (!blackEntry || !whiteEntry) return null
    const key = `${blackEntry.id}-${whiteEntry.id}`
    return $headToHead.get(key) || null
  })()

  // Whose turn it is, derived from displayed move (the side-to-move *after*
  // that move was played).
  $: turn = displayedMove?.current_player || 'black'
  $: liveTotalPly = game?.total_ply ?? 0
  $: viewedPly = displayedMove?.ply ?? liveTotalPly
  $: progressPct = Math.min(100, (liveTotalPly / Math.max(estimatedTotalPly, 1)) * 100)

  $: isFinished = game?.status && game.status !== 'in_progress'
  $: resultLabel = isFinished ? game.status.replaceAll('_', ' ') : null
</script>

<section class="scorecard" aria-label="Match scorecard">
  <div class="player player-black" class:active-turn={!isFinished && turn === 'black'}>
    <span class="role-icon" aria-hidden="true">☗</span>
    <span class="role-label">Black</span>
    <span class="player-name">{game?.name_black ?? '—'}</span>
    {#if blackEntry?.role}
      <span class="tier-badge {blackRole?.cssClass}" title={blackRole?.tooltip}>
        {blackRole?.icon} {blackRole?.label}
      </span>
    {/if}
    {#if blackArchSummary?.arch || blackArchSummary?.topology}
      <span class="arch-info">{blackArchSummary.arch}{#if blackArchSummary.topology} · {blackArchSummary.topology}{/if}</span>
    {/if}
    <span class="spacer"></span>
    {#if !isFinished && turn === 'black'}
      <span class="turn-indicator" aria-label="Black to move">
        <span class="turn-dot" aria-hidden="true"></span> to move
      </span>
    {/if}
    {#if blackElo != null}
      <span class="elo-pill">{blackElo}</span>
    {/if}
  </div>

  <div class="player player-white" class:active-turn={!isFinished && turn === 'white'}>
    <span class="role-icon" aria-hidden="true">☖</span>
    <span class="role-label">White</span>
    <span class="player-name">{game?.name_white ?? '—'}</span>
    {#if whiteEntry?.role}
      <span class="tier-badge {whiteRole?.cssClass}" title={whiteRole?.tooltip}>
        {whiteRole?.icon} {whiteRole?.label}
      </span>
    {/if}
    {#if whiteArchSummary?.arch || whiteArchSummary?.topology}
      <span class="arch-info">{whiteArchSummary.arch}{#if whiteArchSummary.topology} · {whiteArchSummary.topology}{/if}</span>
    {/if}
    <span class="spacer"></span>
    {#if !isFinished && turn === 'white'}
      <span class="turn-indicator" aria-label="White to move">
        <span class="turn-dot" aria-hidden="true"></span> to move
      </span>
    {/if}
    {#if whiteElo != null}
      <span class="elo-pill">{whiteElo}</span>
    {/if}
  </div>

  <div class="footer-strip">
    {#if isFinished}
      <span class="result-badge" role="status">{resultLabel}</span>
    {:else if scrubbing}
      <span class="ply-label scrub-label" role="status">
        Ply {viewedPly} <span class="of">/ live ply {liveTotalPly}</span>
      </span>
    {:else}
      <span class="ply-label" role="status">
        Ply {viewedPly} <span class="of">of ~{estimatedTotalPly}</span>
      </span>
    {/if}
    <div
      class="ply-progress"
      role="progressbar"
      aria-valuenow={liveTotalPly}
      aria-valuemin={0}
      aria-valuemax={estimatedTotalPly}
      aria-label="Game progress: ply {liveTotalPly} of about {estimatedTotalPly}"
    >
      <div class="ply-progress-fill" style="width: {progressPct}%"></div>
    </div>
    {#if h2h && h2h.total > 0}
      <span class="h2h" title="Head-to-head: Black has {h2h.w} wins, {h2h.l} losses, {h2h.d} draws">
        H2H {h2h.w}–{h2h.l}{#if h2h.d > 0}–{h2h.d}{/if}
      </span>
    {/if}
  </div>
</section>

<style>
  .scorecard {
    display: flex;
    flex-direction: column;
    border-bottom: 1px solid var(--border);
  }

  .player {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    transition: background-color 0.2s, box-shadow 0.2s;
  }

  /* Alternate shading: black on the lighter shade, white on the darker — keeps
     the pair visually distinct without a centerline divider, and mirrors the
     piece colours (black piece on light board square). */
  .player-black { background: var(--bg-secondary); }
  .player-white { background: var(--bg-card); }

  .player.active-turn {
    box-shadow: inset 4px 0 0 var(--accent-teal);
  }

  .role-icon { font-size: 18px; color: var(--text-primary); flex-shrink: 0; }

  .role-label {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    width: 48px;
    flex-shrink: 0;
  }

  .player-name {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.15;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .tier-badge {
    font-size: 11px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 3px;
    flex-shrink: 0;
  }
  .tier-badge.role-frontier { color: #7b8fa8; background: rgba(123, 143, 168, 0.12); }
  .tier-badge.role-recent { color: var(--accent-gold); background: var(--badge-bg-gold); }
  .tier-badge.role-dynamic { color: var(--accent-teal); background: var(--badge-bg-teal); }
  .tier-badge.role-historical { color: #9b7ec8; background: rgba(155, 126, 200, 0.12); }
  .tier-badge.role-unknown { color: var(--text-muted); background: rgba(128, 128, 128, 0.12); }

  .arch-info {
    font-size: 12px;
    color: var(--text-muted);
    font-family: monospace;
    flex-shrink: 0;
  }

  .spacer { flex: 1; }

  .turn-indicator {
    font-size: 12px;
    color: var(--accent-teal);
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }

  .turn-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-teal);
    box-shadow: 0 0 0 2px rgba(77, 184, 168, 0.25);
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.55; transform: scale(0.85); }
  }
  @media (prefers-reduced-motion: reduce) {
    .turn-dot { animation: none; }
  }

  .elo-pill {
    font-family: monospace;
    font-size: 14px;
    font-weight: 700;
    color: var(--text-primary);
    background: var(--bg-primary);
    padding: 3px 12px;
    border-radius: 12px;
    border: 1px solid var(--border);
    flex-shrink: 0;
  }

  .footer-strip {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 18px;
    background: var(--bg-primary);
    border-top: 1px solid var(--border);
    font-size: 12px;
    color: var(--text-secondary);
  }

  .ply-label {
    font-family: monospace;
    font-weight: 700;
    color: var(--text-primary);
    flex-shrink: 0;
  }
  .ply-label .of {
    color: var(--text-muted);
    font-weight: 400;
  }
  .scrub-label .of { color: var(--accent-gold); font-weight: 600; }

  .ply-progress {
    flex: 1;
    height: 4px;
    background: var(--bg-card);
    border-radius: 2px;
    overflow: hidden;
    min-width: 80px;
  }
  .ply-progress-fill {
    height: 100%;
    background: var(--accent-teal);
    transition: width 0.3s ease;
  }

  .h2h {
    font-family: monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    flex-shrink: 0;
  }

  .result-badge {
    background: var(--badge-bg-teal);
    color: var(--accent-teal);
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 4px;
    text-transform: capitalize;
    flex-shrink: 0;
  }

  @media (max-width: 768px) {
    .player {
      padding: 8px 12px;
      gap: 8px;
      flex-wrap: wrap;
    }
    .player-name { font-size: 16px; }
    .role-label { width: auto; }
    .arch-info { display: none; }
    .footer-strip {
      padding: 6px 12px;
      flex-wrap: wrap;
    }
    .ply-progress { flex-basis: 100%; }
  }
</style>
