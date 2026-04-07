<script>
  import { onDestroy } from 'svelte'
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
  /** @type {object | null} Style profile from StyleProfiler */
  export let styleProfile = null

  $: icon = role === 'learner' ? '☗' : '☖'
  $: roleLabel = role === 'learner' ? 'Learner' : 'Opponent'
  $: colorClass = role === 'learner' ? 'learner' : 'opponent'
  $: tierInfo = tierRole ? getRoleInfo(tierRole) : null

  // Style profile data
  $: hasStyle = styleProfile && styleProfile.profile_status !== 'insufficient'
  $: primaryStyle = hasStyle ? styleProfile.primary_style : null
  $: secondaryTraits = hasStyle ? (styleProfile.secondary_traits || []) : []
  $: commentary = hasStyle ? (styleProfile.commentary || []) : []
  $: profileStatus = styleProfile ? styleProfile.profile_status : null

  // Rotating commentary
  let commentaryIdx = 0
  $: currentCommentary = commentary.length > 0 ? commentary[commentaryIdx % commentary.length] : null

  let rotationInterval = null
  function updateRotationInterval() {
    if (rotationInterval) clearInterval(rotationInterval)
    rotationInterval = null
    if (commentary.length > 1) {
      rotationInterval = setInterval(() => {
        commentaryIdx = (commentaryIdx + 1) % commentary.length
      }, 20000)
    }
  }
  $: {
    commentary.length
    updateRotationInterval()
  }
  onDestroy(() => { if (rotationInterval) clearInterval(rotationInterval) })
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
  {#if primaryStyle}
    <div class="style-label">{primaryStyle}</div>
  {/if}
  {#if secondaryTraits.length > 0}
    <div class="traits">
      {#each secondaryTraits as trait}
        <span class="trait-badge">{trait}</span>
      {/each}
    </div>
  {/if}
  {#if detail}
    <div class="detail">{detail}</div>
  {/if}
  {#if currentCommentary}
    <div class="commentary" title="{currentCommentary.confidence} confidence">
      {currentCommentary.text}
    </div>
  {/if}
  {#if profileStatus === 'provisional'}
    <div class="profile-status">provisional profile</div>
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

  .style-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--accent-teal);
    margin-top: 4px;
  }

  .traits {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 4px;
  }

  .trait-badge {
    font-size: 11px;
    padding: 1px 6px;
    border-radius: 3px;
    color: var(--text-secondary);
    background: rgba(128, 128, 128, 0.12);
    white-space: nowrap;
  }

  .commentary {
    font-size: 12px;
    font-style: italic;
    color: var(--text-muted);
    margin-top: 6px;
    min-height: 1.4em;
    transition: opacity 0.3s ease;
  }

  .profile-status {
    font-size: 10px;
    color: var(--text-muted);
    opacity: 0.6;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
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
