<script>
  import { getHandPieces } from './handPieces.js'

  export let color = 'black'
  export let hand = {}

  const label = color === 'black' ? '☗ Black' : '☖ White'

  $: pieces = getHandPieces(hand)
</script>

<div class="tray" class:black={color === 'black'} class:white={color === 'white'}>
  <span class="label">{label}</span>
  <div class="pieces">
    {#each pieces as p}
      <div class="hand-piece" title="{p.type} ×{p.count}">
        <span class="kanji" lang="ja">{p.kanji}</span>
        {#if p.count > 1}
          <span class="count">{p.count}</span>
        {/if}
      </div>
    {/each}
    {#if pieces.length === 0}
      <span class="empty">—</span>
    {/if}
  </div>
</div>

<style>
  .tray {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border: 1px solid var(--border);
    border-radius: 6px;
    height: 46px;
  }

  .tray.black {
    background: var(--bg-selected-black);
    border-radius: 0 0 6px 6px;
    border-top: none;
  }

  .tray.white {
    background: var(--bg-selected-white);
    border-radius: 6px 6px 0 0;
    border-bottom: none;
  }

  .label {
    font-size: 12px;
    color: var(--text-secondary);
    min-width: 55px;
  }

  .pieces {
    display: flex;
    align-items: center;
    gap: 4px;
    min-height: 32px;
  }

  .hand-piece {
    width: 28px;
    height: 32px;
    border: 1px solid var(--border);
    border-radius: 3px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    position: relative;
    background: var(--bg-card);
  }

  .count {
    position: absolute;
    top: -5px;
    right: -5px;
    background: var(--accent-teal);
    color: #000;
    font-size: 9px;
    border-radius: 50%;
    width: 14px;
    height: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
  }

  .tray.white .hand-piece {
    border-color: var(--border-white-piece);
  }

  .tray.black .hand-piece {
    border-color: var(--accent-teal);
  }

  .empty {
    color: var(--text-muted);
    font-size: 12px;
  }
</style>
