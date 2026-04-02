<script>
  import { trainingState, trainingAlive } from '../stores/training.js'
  import { getIndicator } from './indicator.js'
  import { buildConfigTooltip } from './configTooltip.js'
  import TabBar from './TabBar.svelte'

  $: status = $trainingState?.status || 'unknown'
  $: epoch = $trainingState?.current_epoch || 0
  $: step = $trainingState?.current_step || 0
  $: alive = $trainingAlive
  $: displayName = $trainingState?.display_name || 'Player'
  $: modelArch = $trainingState?.model_arch || ''
  $: stats = $trainingState?.system_stats || {}
  $: gpus = stats.gpus || []

  $: indicator = getIndicator(alive, status)

  $: configTooltip = buildConfigTooltip($trainingState?.config_json, modelArch)
</script>

<header class="status-bar" role="banner">
  <div class="left">
    <h1>Keisei Training Dashboard</h1>
    <div class="indicator">
      <span class="dot" aria-hidden="true" style="background: {indicator.dot === 'green' ? 'var(--accent-teal)' : indicator.dot === 'yellow' ? 'var(--warning)' : 'var(--danger)'}"></span>
      <span class="text">{indicator.text}</span>
    </div>
    {#if alive}
      <div class="stats" aria-live="polite">
        <span class="stat">Epoch {epoch}</span>
        <span class="sep">|</span>
        <span class="stat">Step {step.toLocaleString()}</span>
        <span class="sep">|</span>
        <span class="stat">Games {($trainingState?.episodes || 0).toLocaleString()}</span>
        {#if stats.cpu_percent != null}
          <span class="sep">|</span>
          <span class="stat">CPU {stats.cpu_percent}%</span>
        {/if}
        {#each gpus as gpu, i}
          <span class="sep">|</span>
          <span class="stat">GPU{i} {gpu.util_percent}% ({gpu.mem_used_mb}MB)</span>
        {/each}
      </div>
    {/if}
  </div>
  <div class="right">
    <TabBar />
  </div>
</header>

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

  h1 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    font-family: 'Noto Serif', Georgia, serif;
  }

  .indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
  }

  .stats {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-muted);
    font-family: monospace;
  }

  .stat {
    color: var(--text-secondary);
  }

  .sep {
    color: var(--border);
  }

  .right {
    font-size: 14px;
  }

  .player-name {
    color: var(--accent-teal);
    font-weight: 600;
    cursor: help;
  }
</style>
