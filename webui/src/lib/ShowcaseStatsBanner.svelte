<script>
  import { showcaseGame, showcaseQueue, queueDepth, sidecarAlive } from '../stores/showcase.js'

  // The banner is intentionally minimal — three glanceable cards:
  //  1. Engine status (live alive dot)
  //  2. Active match ply count
  //  3. Queue depth (pending matches)
  //
  // We deliberately do NOT compute "matches today" or "avg game length" here:
  // those need backend aggregates that don't exist yet. When they do, add a
  // fourth card. Surfacing zero/unknown values would just be noise.
  $: livePly = $showcaseGame?.total_ply ?? 0
  $: matchActive = $showcaseGame != null && $showcaseGame.status === 'in_progress'
  $: pending = $queueDepth
</script>

<section class="stats-banner" aria-label="Showcase summary">
  <div class="stat-card" class:alive={$sidecarAlive} class:dead={!$sidecarAlive}>
    <span class="stat-value">
      <span class="dot" aria-hidden="true"></span>
      {$sidecarAlive ? 'Online' : 'Offline'}
    </span>
    <span class="stat-label">Showcase Engine</span>
  </div>
  <div class="stat-card" class:highlight={matchActive}>
    <span class="stat-value">
      {#if matchActive}
        Ply {livePly}
      {:else}
        —
      {/if}
    </span>
    <span class="stat-label">
      {#if matchActive}
        Live Match
      {:else}
        No Active Match
      {/if}
    </span>
  </div>
  <div class="stat-card" class:warn={pending >= 5}>
    <span class="stat-value">{pending}</span>
    <span class="stat-label">
      {#if pending >= 5}
        Queue Full
      {:else}
        Pending in Queue
      {/if}
    </span>
  </div>
</section>

<style>
  .stats-banner {
    display: flex;
    gap: 12px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .stat-card {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 12px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
  }

  .stat-card.highlight {
    border-color: var(--accent-teal);
    background: var(--badge-bg-teal);
  }

  .stat-card.warn {
    border-color: var(--accent-gold);
    background: var(--badge-bg-gold);
  }

  .stat-card.dead {
    border-color: var(--danger);
    background: var(--badge-bg-danger);
  }

  .stat-value {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    font-family: monospace;
  }

  .stat-card.highlight .stat-value { color: var(--accent-teal); }
  .stat-card.warn .stat-value { color: var(--accent-gold); }
  .stat-card.dead .stat-value { color: var(--danger); }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
  }

  .stat-card.alive .dot {
    background: var(--accent-teal);
    box-shadow: 0 0 0 2px rgba(77, 184, 168, 0.25);
  }

  .stat-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  @media (max-width: 768px) {
    .stats-banner { padding: 8px 10px; gap: 6px; flex-wrap: wrap; }
    .stat-card { padding: 8px 10px; min-width: 110px; }
    .stat-value { font-size: 14px; }
  }
</style>
