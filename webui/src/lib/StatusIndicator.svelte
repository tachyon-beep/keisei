<script>
  import { trainingState, trainingAlive } from '../stores/training.js'

  $: status = $trainingState?.status || 'unknown'
  $: epoch = $trainingState?.current_epoch || 0
  $: alive = $trainingAlive
  $: displayName = $trainingState?.display_name || 'Player'

  $: indicator = alive
    ? { dot: 'green', text: `Training alive (epoch ${epoch})` }
    : status === 'completed'
      ? { dot: 'red', text: 'Training completed' }
      : status === 'paused'
        ? { dot: 'red', text: 'Training paused' }
        : { dot: 'yellow', text: 'Training stale' }
</script>

<header class="status-bar" role="banner">
  <div class="left">
    <h1>Keisei Training Dashboard</h1>
    <div class="indicator">
      <span class="dot" aria-hidden="true" style="background: {indicator.dot === 'green' ? 'var(--accent-green)' : indicator.dot === 'yellow' ? 'var(--warning)' : 'var(--danger)'}"></span>
      <span class="text">{indicator.text}</span>
    </div>
  </div>
  <div class="right">
    <span class="player-name">☗ {displayName}</span>
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

  .right {
    font-size: 14px;
  }

  .player-name {
    color: var(--accent-green);
    font-weight: 600;
  }
</style>
