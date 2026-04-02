<script>
  const pieces = [
    { kanji: '王', name: 'King', move: '1 step any direction', promoted: null },
    { kanji: '飛', name: 'Rook', move: 'Any number straight', promoted: '龍', promName: 'Dragon', promMove: '+ 1 step diagonal' },
    { kanji: '角', name: 'Bishop', move: 'Any number diagonal', promoted: '馬', promName: 'Horse', promMove: '+ 1 step straight' },
    { kanji: '金', name: 'Gold', move: '1 step (not back-diag)', promoted: null },
    { kanji: '銀', name: 'Silver', move: '1 step (not sides/back)', promoted: '全', promName: '→ Gold', promMove: 'Moves like Gold' },
    { kanji: '桂', name: 'Knight', move: '2 fwd + 1 side (jump)', promoted: '圭', promName: '→ Gold', promMove: 'Moves like Gold' },
    { kanji: '香', name: 'Lance', move: 'Any number forward', promoted: '杏', promName: '→ Gold', promMove: 'Moves like Gold' },
    { kanji: '歩', name: 'Pawn', move: '1 step forward', promoted: 'と', promName: 'Tokin', promMove: 'Moves like Gold' },
  ]
</script>

<div class="legend">
  <h3 class="legend-title">Shogi Piece Guide</h3>
  <p class="legend-intro">Japanese chess — captured pieces switch sides and can be dropped back onto the board.</p>

  <div class="piece-grid">
    {#each pieces as p}
      <div class="base-col">
        <span class="kanji">{p.kanji}</span>
        <div class="piece-info">
          <span class="piece-name">{p.name}</span>
          <span class="piece-move">{p.move}</span>
        </div>
      </div>
      <div class="prom-col">
        {#if p.promoted}
          <span class="arrow">→</span>
          <span class="kanji promoted">{p.promoted}</span>
          <div class="piece-info">
            <span class="piece-name prom">{p.promName}</span>
            <span class="piece-move">{p.promMove}</span>
          </div>
        {/if}
      </div>
    {/each}
  </div>

  <div class="legend-footer">
    <p>☗ Black (Sente) moves first — pieces point ↑</p>
    <p>☖ White (Gote) moves second — pieces point ↓</p>
    <p><span class="promoted-color">Red text</span> = promoted piece</p>
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

  .legend-intro {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 12px;
    line-height: 1.4;
  }

  .piece-grid {
    flex: 1;
    display: grid;
    grid-template-columns: auto 1fr;
    align-content: space-evenly;
    gap: 0 16px;
  }

  .base-col, .prom-col {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .kanji {
    font-size: 24px;
    width: 32px;
    text-align: center;
    color: #1a1a1a;
    background: var(--bg-board);
    border: 1px solid var(--border-board);
    border-radius: 2px;
    line-height: 1.3;
    flex-shrink: 0;
  }

  .kanji.promoted {
    color: var(--promoted);
  }

  .piece-info {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }

  .piece-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .piece-name.prom {
    color: var(--promoted);
  }

  .piece-move {
    font-size: 11px;
    color: var(--text-muted);
  }

  .arrow {
    color: var(--text-muted);
    font-size: 14px;
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

  .legend-footer p {
    font-size: 11px;
    color: var(--text-muted);
  }

  .promoted-color {
    color: var(--promoted);
    font-weight: 600;
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
    font-size: 12px;
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
