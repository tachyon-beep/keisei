<script>
  import MoveDots from './MoveDots.svelte'
  import { PATTERNS } from './movePatterns.js'

  const pieces = [
    { kanji: '王', name: 'King', promoted: null },
    { kanji: '飛', name: 'Rook', promoted: '龍', promName: 'Dragon' },
    { kanji: '角', name: 'Bishop', promoted: '馬', promName: 'Horse' },
    { kanji: '金', name: 'Gold', promoted: null },
    { kanji: '銀', name: 'Silver', promoted: '全', promName: 'Gold' },
    { kanji: '桂', name: 'Knight', promoted: '圭', promName: 'Gold' },
    { kanji: '香', name: 'Lance', promoted: '杏', promName: 'Gold' },
    { kanji: '歩', name: 'Pawn', promoted: 'と', promName: 'Tokin' },
  ]
</script>

<div class="legend">
  <h3 class="legend-title">Shogi Piece Guide</h3>

  <div class="piece-grid">
    {#each pieces as p}
      {@const pat = PATTERNS[p.name]}
      <div class="base-col">
        <span class="kanji">{p.kanji}</span>
        <span class="piece-name">{p.name}</span>
        <MoveDots pattern={pat.base} extra={pat.extra || null} />
      </div>
      <div class="arrow-col">
        {#if p.promoted}<span class="arrow">→</span>{/if}
      </div>
      <div class="prom-col">
        {#if p.promoted && pat.promoted}
          <span class="kanji promoted">{p.promoted}</span>
          <span class="piece-name prom">{p.promName}</span>
          <MoveDots pattern={pat.promoted} promoted={true} />
        {/if}
      </div>
    {/each}
  </div>

  <div class="legend-footer">
    <div class="legend-key">
      <span class="key-item"><span class="key-square">■</span> = one square</span>
      <span class="key-item"><span class="key-arrow">→</span> = slides</span>
    </div>
    <p>☗ Black (Sente) moves first — pieces point ↑</p>
    <p>☖ White (Gote) moves second — pieces point ↓</p>
  </div>

  <div class="chess-diff">
    <h4 class="diff-title">How Shogi differs from Chess</h4>
    <div class="diff-list">
      <div class="diff-item">
        <span class="diff-icon">♻</span>
        <div><strong>Drops</strong> — captured pieces join your army and can be placed back on any empty square on your turn</div>
      </div>
      <div class="diff-item">
        <span class="diff-icon">⬆</span>
        <div><strong>Promotion</strong> — most pieces promote when reaching the last 3 ranks (opponent's territory), gaining new moves</div>
      </div>
      <div class="diff-item">
        <span class="diff-icon">♟</span>
        <div><strong>Weaker pieces</strong> — no Queen; the Rook and Bishop are the strongest. Pawns only move and capture forward</div>
      </div>
      <div class="diff-item">
        <span class="diff-icon">🏰</span>
        <div><strong>No castling</strong> — Kings are defended by building "castles" from Gold and Silver generals over multiple moves</div>
      </div>
      <div class="diff-item">
        <span class="diff-icon">♾</span>
        <div><strong>No draws (almost)</strong> — drops mean material never leaves the game, so stalemates are extremely rare</div>
      </div>
    </div>
  </div>
</div>

<style>
  .legend {
    height: 100%;
    display: flex;
    flex-direction: column;
    padding: 12px 16px;
  }

  .legend-title {
    font-size: 14px;
    font-weight: 700;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
  }

  .piece-grid {
    flex: 1;
    display: grid;
    grid-template-columns: auto auto auto;
    justify-content: space-between;
    align-content: space-evenly;
  }

  .arrow-col {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .base-col, .prom-col {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .kanji {
    font-size: 24px;
    width: 32px;
    text-align: center;
    color: var(--text-piece);
    background: var(--bg-board);
    border: 1px solid var(--border-board);
    border-radius: 2px;
    line-height: 1.3;
    flex-shrink: 0;
  }

  .kanji.promoted {
    color: var(--promoted);
  }

  .piece-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    min-width: 42px;
  }

  .piece-name.prom {
    color: var(--promoted);
  }

  .arrow {
    color: var(--text-muted);
    font-size: 28px;
    flex-shrink: 0;
  }

  .legend-footer {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .legend-key {
    display: flex;
    gap: 12px;
    margin-bottom: 4px;
  }

  .key-item {
    font-size: 13px;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 3px;
  }

  .key-square {
    color: var(--accent-teal);
    font-size: 12px;
  }

  .key-arrow {
    color: var(--accent-gold);
    font-size: 13px;
  }

  .legend-footer p {
    font-size: 13px;
    color: var(--text-muted);
  }

  .chess-diff {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
  }

  .diff-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }

  .diff-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .diff-item {
    display: flex;
    gap: 8px;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.4;
  }

  .diff-icon {
    flex-shrink: 0;
    font-size: 14px;
    width: 20px;
    text-align: center;
  }

  .diff-item strong {
    color: var(--text-primary);
  }
</style>
