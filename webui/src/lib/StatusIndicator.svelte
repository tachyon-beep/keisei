<script>
  import { onDestroy } from 'svelte'
  import { trainingState, trainingAlive } from '../stores/training.js'
  import { connectionState } from './ws.js'
  import { getIndicator } from './indicator.js'
  import { buildConfigTooltip } from './configTooltip.js'
  import { parseUTC, formatElapsed } from './timeFormat.js'
  import TabBar from './TabBar.svelte'

  $: status = $trainingState?.status || 'unknown'
  $: epoch = $trainingState?.current_epoch || 0
  $: step = $trainingState?.current_step || 0
  $: alive = $trainingAlive
  $: displayName = $trainingState?.display_name || 'Player'
  $: modelArch = $trainingState?.model_arch || ''
  $: stats = $trainingState?.system_stats || {}
  $: gpus = stats.gpus || []

  $: totalEpochs = $trainingState?.total_epochs || null
  $: phase = $trainingState?.phase || ''
  $: indicator = getIndicator(alive, status)

  $: configTooltip = buildConfigTooltip($trainingState?.config_json, modelArch)

  // Wall clock: real time since training started (ticks every second)
  // Train clock: time the trainer has been active (heartbeat_at - started_at, freezes when stopped)

  $: startedAt = parseUTC($trainingState?.started_at)
  $: heartbeatAt = parseUTC($trainingState?.heartbeat_at)
  let wallTime = ''
  let trainTime = ''
  let wallTimer = null

  function tick() {
    if (startedAt) {
      wallTime = formatElapsed(Date.now() - startedAt.getTime())
    }
    if (startedAt && heartbeatAt) {
      trainTime = formatElapsed(heartbeatAt.getTime() - startedAt.getTime())
    }
  }

  $: if (startedAt && !wallTimer) {
    tick()
    wallTimer = setInterval(tick, 1000)
  }
  // Update train clock when heartbeat changes
  $: if (heartbeatAt) tick()

  onDestroy(() => {
    if (wallTimer) clearInterval(wallTimer)
  })
</script>

<header class="status-bar" role="banner">
  <div class="left">
    <img src="/favicon.svg" alt="" class="app-icon" aria-hidden="true" />
    <h1>
      {#if displayName && displayName !== 'Player'}
        {displayName}
      {:else}
        Keisei
      {/if}
    </h1>
    {#if alive || epoch > 0}
      <div class="stats">
        {#if !alive && indicator.dot === 'yellow'}
          <span class="phase-badge stale" aria-live="polite">&#9679; STALE</span>
        {:else if !alive}
          <span class="phase-badge stopped" aria-live="polite">&#9632; {indicator.text.replace('Training ', '').toUpperCase()}</span>
        {:else if phase === 'update'}
          <span class="phase-badge update" aria-live="polite">&#9650; PPO UPDATE</span>
        {:else if phase === 'rollout'}
          <span class="phase-badge rollout" aria-live="polite">&#9654; ROLLOUT</span>
        {:else}
          <span class="phase-badge alive" aria-live="polite">&#9679; ALIVE</span>
        {/if}
        <span class="sep">|</span>
        <span class="stat">Epoch {epoch.toLocaleString()}{#if totalEpochs} / {totalEpochs.toLocaleString()}{/if}</span>
        <span class="sep">|</span>
        <span class="stat">Step {step.toLocaleString()}</span>
        <span class="sep">|</span>
        <span class="stat">Games {($trainingState?.episodes || 0).toLocaleString()}</span>
        {#if wallTime}
          <span class="sep hide-mobile">|</span>
          <span class="stat hide-mobile" title="Wall clock: real time since training started">Wall {wallTime}</span>
        {/if}
        {#if trainTime}
          <span class="sep hide-mobile">|</span>
          <span class="stat hide-mobile" title="Train clock: active training time (freezes when stopped)">Train {trainTime}</span>
        {/if}
        {#if stats.cpu_percent != null}
          <span class="sep hide-narrow">|</span>
          <span class="stat hide-narrow">CPU {stats.cpu_percent}%</span>
        {/if}
        {#each gpus as gpu, i}
          <span class="sep hide-narrow">|</span>
          <span class="stat hide-narrow">GPU{i} {gpu.util_percent}% ({gpu.mem_used_mb}MB)</span>
        {/each}
      </div>
    {/if}
  </div>
  <div class="right">
    <TabBar />
  </div>
</header>
{#if $connectionState === 'connecting'}
  <div class="connecting-banner" role="status">
    Connecting to training server&hellip;
  </div>
{:else if $connectionState === 'reconnecting'}
  <div class="reconnect-banner" role="alert">
    Disconnected from server — reconnecting&hellip;
  </div>
{/if}

<style>
  header.status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
  }

  .left {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .app-icon {
    height: 24px;
    width: auto;
    flex-shrink: 0;
  }

  h1 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    font-family: Georgia, 'Times New Roman', serif;
  }

  .stats {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--text-muted);
    font-family: monospace;
  }

  .stat {
    color: var(--text-secondary);
  }

  .sep {
    color: var(--border);
  }

  .phase-badge {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 1px 6px;
    border-radius: 3px;
  }

  .phase-badge.update {
    color: var(--danger);
    background: var(--badge-bg-danger);
  }

  .phase-badge.rollout {
    color: var(--accent-teal);
    background: var(--badge-bg-teal);
  }

  .phase-badge.stale {
    color: var(--warning);
    background: var(--badge-bg-gold);
  }

  .phase-badge.alive {
    color: var(--accent-teal);
    background: var(--badge-bg-teal);
  }

  .phase-badge.stopped {
    color: var(--danger);
    background: var(--badge-bg-danger);
  }

  .right {
    font-size: 14px;
  }

  .player-name {
    color: var(--accent-teal);
    font-weight: 600;
    cursor: help;
  }

  .connecting-banner {
    background: rgba(77, 184, 168, 0.1);
    border-bottom: 1px solid var(--accent-teal);
    color: var(--accent-teal);
    font-size: 13px;
    font-weight: 600;
    text-align: center;
    padding: 6px 16px;
  }

  .reconnect-banner {
    background: rgba(224, 80, 80, 0.15);
    border-bottom: 1px solid var(--danger);
    color: var(--danger);
    font-size: 13px;
    font-weight: 600;
    text-align: center;
    padding: 6px 16px;
  }

  .stats {
    flex-wrap: wrap;
  }

  @media (max-width: 768px) {
    .hide-mobile { display: none; }
  }

  @media (max-width: 480px) {
    .hide-narrow { display: none; }
    header.status-bar {
      flex-direction: column;
      gap: 8px;
      align-items: flex-start;
    }
  }
</style>
