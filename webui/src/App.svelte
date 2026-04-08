<script>
  import { onMount } from 'svelte'
  import { connect, disconnect } from './lib/ws.js'
  import { games, selectedGame, selectedOpponent } from './stores/games.js'
  import { activeTab } from './stores/navigation.js'
  import { trainingState } from './stores/training.js'
  import { leagueResults, learnerEntry } from './stores/league.js'
  import { latestMetrics } from './stores/metrics.js'
  import StatusIndicator from './lib/StatusIndicator.svelte'
  import GameThumbnail from './lib/GameThumbnail.svelte'
  import Board from './lib/Board.svelte'
  import PieceTray from './lib/PieceTray.svelte'
  import MoveLog from './lib/MoveLog.svelte'
  import EvalBar from './lib/EvalBar.svelte'
  import MetricsGrid from './lib/MetricsGrid.svelte'
  import PlayerCard from './lib/PlayerCard.svelte'
  import LeagueView from './lib/LeagueView.svelte'
  import ShowcaseView from './lib/ShowcaseView.svelte'
  import ShogiLegend from './lib/ShogiLegend.svelte'
  import { safeParse } from './lib/safeParse.js'

  onMount(() => {
    connect()
    return disconnect
  })

  $: game = $selectedGame
  $: board = game ? safeParse(game.board_json, game.board || []) : []
  $: hands = game ? safeParse(game.hands_json, game.hands || {}) : {}
  $: moveHistory = game?.move_history_json || '[]'

  let thumbPanelHeight = 0

  $: lastMoveIdx = (() => {
    try {
      const history = safeParse(moveHistory, [])
      if (history.length === 0) return -1
      return -1
    } catch { return -1 }
  })()

  // Learner info from training state
  $: learnerName = $trainingState?.display_name || $trainingState?.model_arch || 'Learner'
  $: learnerElo = $learnerEntry?.elo_rating ?? null
  $: learnerDetail = $trainingState
    ? `${$trainingState.model_arch || ''} · Epoch ${$trainingState.current_epoch || 0} · ${($trainingState.current_step || 0).toLocaleString()} steps`
    : ''

  // Seeded RNG for stable learner fun facts (keyed on display name)
  function seededPick(seed, pool) {
    let h = 0
    for (let i = 0; i < seed.length; i++) h = ((h << 5) - h + seed.charCodeAt(i)) | 0
    return pool[Math.abs(h) % pool.length]
  }

  const flavourPools = {
    'Favourite piece': ['Gold General', 'Silver General', 'Knight', 'Lance', 'Bishop', 'Rook', 'Promoted Pawn', 'Dragon Horse', 'Dragon King', 'King'],
    'Favourite philosopher': ['Musashi', 'Sun Tzu', 'Confucius', 'Turing', 'Shannon', 'Von Neumann', 'Bellman', 'Lao Tzu', 'Leibniz', 'Miyamoto'],
    'Favourite snack': ['Onigiri', 'Mochi', 'Dango', 'Taiyaki', 'Gradient soup', 'Loss crumble', 'Entropy tea', 'Batch noodles', 'Tensor rolls', 'Senbei'],
    'Training motto': ['"Loss goes down"', '"Explore everything"', '"Patience is policy"', '"Trust the gradient"', '"Variance is the enemy"', '"Clip wisely"', '"Entropy is freedom"', '"Value the position"', '"Every ply counts"', '"Promote early"'],
    'Lucky number': ['0.0001', '0.99', '42', '3.14', '2048', '0.95', '1e-8', '256', '0.2', '7.5M'],
  }

  $: learnerFlavour = (() => {
    const name = learnerName || 'Learner'
    const cats = Object.keys(flavourPools)
    const facts = []
    for (let i = 0; i < 3 && i < cats.length; i++) {
      const cat = cats[(Math.abs(name.length * 31 + i * 7) | 0) % cats.length]
      // avoid picking the same category twice
      if (!facts.find(f => f[0] === cat)) {
        facts.push([cat, seededPick(name + cat, flavourPools[cat])])
      }
    }
    return facts
  })()

  // Learner stats for PlayerCard
  $: learnerStats = (() => {
    const s = []
    if ($trainingState?.model_arch) s.push(['Architecture', $trainingState.model_arch])
    // Compact architecture summary from config
    try {
      const cfg = typeof $trainingState?.config_json === 'string'
        ? JSON.parse($trainingState.config_json) : $trainingState?.config_json
      if (cfg) {
        const mp = cfg.model?.params || cfg.model_params || {}
        const parts = []
        if (mp.num_blocks && mp.channels) parts.push(`b${mp.num_blocks}c${mp.channels}`)
        if (mp.se_reduction) parts.push(`SE-${mp.se_reduction}`)
        if (mp.global_pool_channels) parts.push(`gp${mp.global_pool_channels}`)
        if (parts.length) s.push(['Topology', parts.join(' · ')])
      }
    } catch { /* config not available yet */ }
    const m = $latestMetrics
    if (m) {
      if (m.policy_loss != null) s.push(['Policy loss', m.policy_loss.toFixed(4)])
      if (m.value_loss != null) s.push(['Value loss', m.value_loss.toFixed(4)])
      if (m.entropy != null) s.push(['Entropy', m.entropy.toFixed(4)])
      if (m.value_accuracy != null) s.push(['Value accuracy', (m.value_accuracy * 100).toFixed(1) + '%'])
    }
    // W/L/D from recent league results (last 10 epochs)
    if ($leagueResults.length > 0) {
      const recent = $leagueResults.slice(0, 10)
      const totals = recent.reduce((a, r) => ({
        w: a.w + (r.wins || 0), l: a.l + (r.losses || 0), d: a.d + (r.draws || 0)
      }), { w: 0, l: 0, d: 0 })
      s.push(['Recent W/L/D', `${totals.w} / ${totals.l} / ${totals.d}`])
    }
    // Fun facts
    for (const [label, value] of learnerFlavour) {
      s.push([label, value])
    }
    return s
  })()

  // Opponent info from selected game
  $: opp = $selectedOpponent
  $: opponentName = opp ? opp.display_name : 'Self-play'
  $: opponentElo = opp?.elo_rating ?? null
  $: opponentDetail = opp
    ? `${opp.architecture} · Epoch ${opp.created_epoch}`
    : ''

  // Opponent stats for PlayerCard
  $: opponentStats = (() => {
    if (!opp) return []
    const s = []
    s.push(['Architecture', opp.architecture])
    const mp = opp.model_params || {}
    const parts = []
    if (mp.num_blocks && mp.channels) parts.push(`b${mp.num_blocks}c${mp.channels}`)
    if (mp.se_reduction) parts.push(`SE-${mp.se_reduction}`)
    if (mp.global_pool_channels) parts.push(`gp${mp.global_pool_channels}`)
    if (parts.length) s.push(['Topology', parts.join(' · ')])
    s.push(['Snapshot epoch', String(opp.created_epoch)])
    s.push(['Games played', String(opp.games_played)])
    if (opp.created_at) {
      const d = new Date(opp.created_at)
      s.push(['Born', d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })])
    }
    // Append flavour facts from the league entry
    if (opp.flavour_facts) {
      for (const [label, value] of opp.flavour_facts) {
        s.push([label, value])
      }
    }
    return s
  })()
</script>

<div class="app">
  <a href={$activeTab === 'training' ? '#game-panel' : '#league-main'} class="skip-nav">Skip to content</a>
  <StatusIndicator />

  {#if $activeTab === 'training'}
    <div class="main-content">
      <aside class="thumbnail-panel" aria-label="Game list" bind:clientHeight={thumbPanelHeight} style="width: {thumbPanelHeight - 94}px">
        <h2 class="section-label">Games ({Math.min($games.length, 16)}{#if $games.length > 16} / {$games.length}{/if})</h2>
        <div class="thumb-grid">
          {#each $games.slice(0, 16) as g (g.game_id)}
            <GameThumbnail game={g} />
          {/each}
        </div>
      </aside>

      <div class="player-panel">
        <PlayerCard role="learner" name={learnerName} elo={learnerElo} detail={learnerDetail} stats={learnerStats} />
        <div class="vs-separator">VS</div>
        <PlayerCard role="opponent" name={opponentName} elo={opponentElo} detail={opponentDetail} stats={opponentStats} tierRole={opp?.role} />
      </div>

      <main id="game-panel" class="game-panel" aria-label="Game viewer">
        {#if game}
          <div class="game-view">
            <div class="board-area">
              <PieceTray color="white" hand={hands.white || {}} />
              <Board
                board={board}
                inCheck={!!game.in_check}
                currentPlayer={game.current_player || 'black'}
                lastMoveIdx={lastMoveIdx}
              />
              <PieceTray color="black" hand={hands.black || {}} />
            </div>

            <div class="eval-area">
              <EvalBar
                value={game.value_estimate || 0}
                currentPlayer={game.current_player || 'black'}
              />
            </div>

            <div class="info-area">
              <div class="game-info">
                <div class="info-row">
                  <span class="label">Game {(game.game_id || 0) + 1}</span>
                  <span class="value">{game.current_player || 'black'} to move</span>
                </div>
                <div class="info-row">
                  <span class="label">Ply</span>
                  <span class="value">{game.ply || 0}</span>
                </div>
                <div class="info-row">
                  <span class="label">Result</span>
                  <span class="value result"
                    class:in-progress={game.result === 'in_progress'}
                    class:terminal={game.result !== 'in_progress'}
                  >
                    {#if game.result === 'in_progress'}In progress{:else}&#10003; {(game.result || '').replaceAll('_', ' ')}{/if}
                  </span>
                </div>
              </div>

              <MoveLog
                moveHistoryJson={moveHistory}
                currentPlayer={game.current_player || 'black'}
              />
            </div>

            <div class="legend-area">
              <ShogiLegend />
            </div>
          </div>
        {:else}
          <div class="no-game">
            <p>Waiting for game data&hellip;</p>
            <p class="no-game-hint">Connect a training session to see live games.</p>
          </div>
        {/if}
      </main>
    </div>

    <section class="metrics-panel" aria-label="Training metrics">
      <MetricsGrid />
    </section>
  {:else if $activeTab === 'league'}
    <LeagueView />
  {:else if $activeTab === 'showcase'}
    <ShowcaseView />
  {/if}
</div>

<style>
  .skip-nav {
    position: absolute;
    left: -9999px;
    top: 0;
    z-index: 100;
    padding: 8px 16px;
    background: var(--accent-ink);
    color: #fff;
    font-size: 14px;
    font-weight: 600;
    text-decoration: none;
    border-radius: 0 0 4px 0;
  }

  .skip-nav:focus { left: 0; }

  .app {
    display: grid;
    grid-template-rows: auto 1fr auto;
    height: 100dvh;
    overflow: hidden;
    background: var(--bg-primary);
  }

  .main-content {
    display: flex;
    gap: 0;
    align-items: stretch;
    overflow: hidden;
    min-height: 0;
    border-bottom: 1px solid var(--border);
  }

  .thumbnail-panel {
    flex: 0 0 auto;
    min-width: 200px;
    border-right: 1px solid var(--border);
    padding: 8px;
    overflow: hidden;
  }

  .section-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }

  .thumb-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
  }

  .player-panel {
    flex: 0 0 auto;
    width: 315px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    justify-content: stretch;
    gap: 4px;
    border-right: 1px solid var(--border);
  }

  .vs-separator {
    text-align: center;
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
  }

  .game-panel {
    flex: 1 1 auto;
    padding: 8px;
    overflow: hidden;
    min-height: 0;
  }

  .game-view {
    display: flex;
    align-items: stretch;
    gap: 16px;
    height: 100%;
  }

  .board-area {
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    justify-content: center;
  }

  .eval-area {
    display: flex;
    flex-shrink: 0;
  }

  .info-area {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 40ch;
    min-height: 0;
    overflow: hidden;
  }

  .game-info {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
  }

  .info-row {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    font-size: 13px;
  }

  .info-row .label { color: var(--text-secondary); }
  .info-row .value { color: var(--text-primary); }

  .result.in-progress { color: var(--accent-gold); }
  .result.terminal { color: var(--accent-teal); }

  .no-game {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    color: var(--text-muted);
    gap: 8px;
  }

  .no-game-hint { font-size: 13px; color: var(--text-muted); }

  .legend-area {
    flex: 0 1 auto;
    min-width: 0;
    min-height: 0;
    overflow: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }

  .metrics-panel { padding: 12px 16px; }

  @media (max-width: 768px) {
    .main-content { flex-direction: column; }

    .thumbnail-panel {
      width: 100%;
      border-right: none;
      border-bottom: 1px solid var(--border);
      max-height: 160px;
    }

    .thumb-grid {
      grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    }

    .player-panel {
      width: 100%;
      flex-direction: row;
      border-right: none;
      border-bottom: 1px solid var(--border);
      justify-content: center;
    }

    .vs-separator { writing-mode: horizontal-tb; }

    .game-view { flex-direction: column; }
    .board-area { align-self: center; }
    .info-area { min-width: unset; }
  }

  @media (max-width: 480px) {
    .game-panel { padding: 8px; }
    .metrics-panel { padding: 8px; }
  }
</style>
