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

  const learnerFacts = [
    ['Favourite opening', 'Static Rook'],
    ['Spirit animal', 'Capybara'],
    ['Favourite food', 'Gradient soup'],
    ['Motto', '"Loss goes down"'],
    ['Lucky number', '0.0001'],
    ['Hobby', 'Backpropagation'],
  ]

  const opponentFacts = [
    ['Favourite opening', 'Ranging Rook'],
    ['Spirit animal', 'Tanuki'],
    ['Favourite food', 'Random rollouts'],
    ['Motto', '"Explore everything"'],
    ['Lucky number', '42'],
    ['Hobby', 'Self-play'],
  ]

  $: facts = role === 'learner' ? learnerFacts : opponentFacts
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
  <div class="facts">
    {#each facts as [label, value]}
      <div class="fact-row">
        <span class="fact-label">{label}</span>
        <span class="fact-value">{value}</span>
      </div>
    {/each}
  </div>
</div>

<style>
  .player-card {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    background: var(--bg-secondary);
    flex: 1;
    display: flex;
    flex-direction: column;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .role {
    font-weight: 700;
    font-size: 16px;
  }

  .player-card.learner .role { color: var(--player-learner); }
  .player-card.opponent .role { color: var(--player-opponent); }

  .elo-badge {
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
    font-family: monospace;
  }

  .player-card.learner .elo-badge {
    background: var(--elo-bg-learner);
    color: var(--player-learner);
  }

  .player-card.opponent .elo-badge {
    background: var(--elo-bg-opponent);
    color: var(--player-opponent);
  }

  .name {
    font-size: 16px;
    color: var(--text-primary);
    margin-top: 8px;
  }

  .detail {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 4px;
  }

  .facts {
    margin-top: 10px;
    border-top: 1px solid var(--border);
    padding-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex: 1;
    justify-content: space-evenly;
  }

  .fact-row {
    display: flex;
    justify-content: space-between;
    font-size: 14px;
  }

  .fact-label {
    color: var(--text-muted);
  }

  .fact-value {
    color: var(--text-secondary);
    text-align: right;
  }
</style>
