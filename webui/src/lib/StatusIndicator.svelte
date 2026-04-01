<script>
  import { trainingState, trainingAlive } from '../stores/training.js'

  $: status = $trainingState?.status || 'unknown'
  $: epoch = $trainingState?.current_epoch || 0
  $: step = $trainingState?.current_step || 0
  $: alive = $trainingAlive
  $: displayName = $trainingState?.display_name || 'Player'
  $: modelArch = $trainingState?.model_arch || ''
  $: stats = $trainingState?.system_stats || {}
  $: gpus = stats.gpus || []

  $: indicator = alive
    ? { dot: 'green', text: `Training alive` }
    : status === 'completed'
      ? { dot: 'red', text: 'Training completed' }
      : status === 'paused'
        ? { dot: 'red', text: 'Training paused' }
        : { dot: 'yellow', text: 'Training stale' }

  $: configTooltip = (() => {
    try {
      const cfg = typeof $trainingState?.config_json === 'string'
        ? JSON.parse($trainingState.config_json)
        : $trainingState?.config_json
      if (!cfg) return ''
      const lines = []
      lines.push(`Architecture: ${modelArch}`)
      if (cfg.training) {
        lines.push(`Algorithm: ${cfg.training.algorithm || '?'}`)
        lines.push(`Games: ${cfg.training.num_games || '?'}`)
      }
      if (cfg.model) {
        lines.push(`Architecture: ${cfg.model.architecture || '?'}`)
      }
      return lines.join('\n')
    } catch { return modelArch }
  })()
</script>

<header class="status-bar" role="banner">
  <div class="left">
    <h1>Keisei Training Dashboard</h1>
    <div class="indicator">
      <span class="dot" aria-hidden="true" style="background: {indicator.dot === 'green' ? 'var(--accent-green)' : indicator.dot === 'yellow' ? 'var(--warning)' : 'var(--danger)'}"></span>
      <span class="text">{indicator.text}</span>
    </div>
    {#if alive}
      <div class="stats">
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
    <span class="player-name" title={configTooltip}>☗ {displayName}</span>
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
    color: var(--accent-green);
    font-weight: 600;
    cursor: help;
  }
</style>
