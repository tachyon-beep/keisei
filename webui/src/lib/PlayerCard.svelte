<script>
  /** @type {'learner' | 'opponent'} */
  export let role = 'learner'
  /** @type {string} */
  export let name = ''
  /** @type {number | null} */
  export let elo = null
  /** @type {string} */
  export let detail = ''

  $: icon = role === 'learner' ? '☗' : '☖'
  $: roleLabel = role === 'learner' ? 'Learner' : 'Opponent'
  $: colorClass = role === 'learner' ? 'learner' : 'opponent'
</script>

<div
  class="player-card {colorClass}"
  aria-label="{roleLabel}: {name}{elo != null ? ', Elo ' + Math.round(elo) : ''}"
>
  <div class="header">
    <span class="role">{icon} {roleLabel}</span>
    {#if elo != null}
      <span class="elo-badge">{Math.round(elo)}</span>
    {/if}
  </div>
  <div class="name">{name || '—'}</div>
  {#if detail}
    <div class="detail">{detail}</div>
  {/if}
</div>

<style>
  .player-card {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    background: var(--bg-secondary);
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .role {
    font-weight: 700;
    font-size: 12px;
  }

  .player-card.learner .role { color: var(--player-learner); }
  .player-card.opponent .role { color: var(--player-opponent); }

  .elo-badge {
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    font-family: monospace;
  }

  .player-card.learner .elo-badge {
    background: #1a3a2a;
    color: var(--player-learner);
  }

  .player-card.opponent .elo-badge {
    background: #1a1a2e;
    color: var(--player-opponent);
  }

  .name {
    font-size: 12px;
    color: var(--text-primary);
    margin-top: 6px;
  }

  .detail {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 2px;
  }
</style>
