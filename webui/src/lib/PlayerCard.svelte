<script>
  import { getRoleInfo } from './roleIcons.js'

  /** @type {'learner' | 'opponent'} */
  export let role = 'learner'
  /** @type {string | null} Tier role: frontier_static, recent_fixed, dynamic, historical */
  export let tierRole = null
  /** @type {string} */
  export let name = ''
  /** @type {number | null} */
  export let elo = null
  /** @type {string} */
  export let detail = ''
  /** @type {Array<[string, string]>} */
  export let stats = []

  $: icon = role === 'learner' ? '☗' : '☖'
  $: roleLabel = role === 'learner' ? 'Learner' : 'Opponent'
  $: colorClass = role === 'learner' ? 'learner' : 'opponent'
  $: tierInfo = tierRole ? getRoleInfo(tierRole) : null
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
  {#if tierInfo}
    <span class="tier-badge {tierInfo.cssClass}">{tierInfo.icon} {tierInfo.label}</span>
  {/if}
  <div class="name">{name || '—'}</div>
  {#if detail}
    <div class="detail">{detail}</div>
  {/if}
  {#if stats.length > 0}
    <div class="facts">
      {#each stats as [label, value]}
        <div class="fact-row">
          <span class="fact-label">{label}</span>
          <span class="fact-value">{value}</span>
        </div>
      {/each}
    </div>
  {/if}
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

  .tier-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
    padding: 2px 8px;
    border-radius: 3px;
    margin-top: 6px;
  }
  .tier-badge.role-frontier { color: #7b8fa8; background: rgba(123, 143, 168, 0.12); }
  .tier-badge.role-recent { color: #c8962e; background: rgba(200, 150, 46, 0.12); }
  .tier-badge.role-dynamic { color: var(--accent-teal); background: rgba(77, 184, 168, 0.12); }
  .tier-badge.role-historical { color: #9b7ec8; background: rgba(155, 126, 200, 0.12); }
  .tier-badge.role-unknown { color: var(--text-muted); background: rgba(128, 128, 128, 0.12); }

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
