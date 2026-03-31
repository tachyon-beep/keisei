# Plan 4: Board + Piece Trays

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the Shogi board component (9x9 grid with kanji pieces) and piece tray components (captured pieces with count badges).

**Architecture:** CSS Grid for the board, kanji characters for pieces. White's pieces rotated 180deg per shogi convention. Piece trays above (White) and below (Black) the board.

**Tech Stack:** Svelte, CSS Grid

---

### Task 1: Piece Data Mapping

**Files:**
- Create: `webui/src/lib/pieces.js`

- [ ] **Step 1: Create piece type to kanji mapping**

`webui/src/lib/pieces.js`:
```js
/**
 * Shogi piece kanji mappings and rendering helpers.
 */

export const PIECE_KANJI = {
  pawn:   { base: '歩', promoted: 'と' },
  lance:  { base: '香', promoted: '杏' },
  knight: { base: '桂', promoted: '圭' },
  silver: { base: '銀', promoted: '全' },
  gold:   { base: '金', promoted: null },
  bishop: { base: '角', promoted: '馬' },
  rook:   { base: '飛', promoted: '龍' },
  king:   { base: '玉', promoted: null },
}

// King uses different kanji per player (convention)
export const KING_KANJI = {
  black: '王',
  white: '玉',
}

/** Order for piece trays: highest value first. */
export const HAND_PIECE_ORDER = ['rook', 'bishop', 'gold', 'silver', 'knight', 'lance', 'pawn']

/**
 * Get the display kanji for a piece.
 * @param {string} type - piece type name
 * @param {boolean} promoted - whether promoted
 * @param {string} color - "black" or "white"
 * @returns {string} kanji character
 */
export function pieceKanji(type, promoted, color) {
  if (type === 'king') return KING_KANJI[color] || '玉'
  const entry = PIECE_KANJI[type]
  if (!entry) return '?'
  if (promoted && entry.promoted) return entry.promoted
  return entry.base
}
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/pieces.js
git commit -m "feat: shogi piece kanji mappings and hand piece ordering"
```

---

### Task 2: Board Component

**Files:**
- Create: `webui/src/lib/Board.svelte`

- [ ] **Step 1: Implement Board.svelte**

`webui/src/lib/Board.svelte`:
```svelte
<script>
  import { pieceKanji } from './pieces.js'

  /** @type {Array<null|{type: string, color: string, promoted: boolean, row: number, col: number}>} */
  export let board = []

  /** @type {boolean} */
  export let inCheck = false

  /** @type {string} */
  export let currentPlayer = 'black'

  // Column labels: 9 down to 1 (shogi convention)
  const colLabels = [9, 8, 7, 6, 5, 4, 3, 2, 1]
  // Row labels: a through i
  const rowLabels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
</script>

<div class="board-container">
  <!-- Column labels -->
  <div class="col-labels">
    {#each colLabels as label}
      <span>{label}</span>
    {/each}
  </div>

  <div class="board-with-rows">
    <!-- 9x9 grid -->
    <div class="board">
      {#each Array(81) as _, idx}
        {@const piece = board[idx]}
        {@const row = Math.floor(idx / 9)}
        {@const col = idx % 9}
        <div
          class="square"
          class:has-piece={piece != null}
        >
          {#if piece}
            <span
              class="piece"
              class:white={piece.color === 'white'}
              class:promoted={piece.promoted}
            >
              {pieceKanji(piece.type, piece.promoted, piece.color)}
            </span>
          {/if}
        </div>
      {/each}
    </div>

    <!-- Row labels -->
    <div class="row-labels">
      {#each rowLabels as label}
        <span>{label}</span>
      {/each}
    </div>
  </div>
</div>

<style>
  .board-container {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .col-labels {
    display: grid;
    grid-template-columns: repeat(9, 1fr);
    padding-left: 0;
    padding-right: 20px;
    font-size: 11px;
    color: var(--text-muted);
    text-align: center;
  }

  .board-with-rows {
    display: flex;
    gap: 4px;
  }

  .board {
    display: grid;
    grid-template-columns: repeat(9, 36px);
    grid-template-rows: repeat(9, 36px);
    border: 2px solid #8B7355;
    background: var(--bg-board);
  }

  .square {
    border: 1px solid #b8956a;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    position: relative;
  }

  .piece {
    cursor: default;
    user-select: none;
    line-height: 1;
  }

  .piece.white {
    transform: rotate(180deg);
  }

  .piece.promoted {
    color: #c00;
  }

  .row-labels {
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    font-size: 11px;
    color: var(--text-muted);
    padding: 0 2px;
  }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/Board.svelte
git commit -m "feat: 9x9 shogi board with kanji pieces and coordinate labels"
```

---

### Task 3: PieceTray Component

**Files:**
- Create: `webui/src/lib/PieceTray.svelte`

- [ ] **Step 1: Implement PieceTray.svelte**

`webui/src/lib/PieceTray.svelte`:
```svelte
<script>
  import { PIECE_KANJI, HAND_PIECE_ORDER } from './pieces.js'

  /** @type {"black" | "white"} */
  export let color = 'black'

  /** @type {Object<string, number>} e.g. {"pawn": 2, "lance": 0, ...} */
  export let hand = {}

  const label = color === 'black' ? '☗ Black' : '☖ White'

  $: pieces = HAND_PIECE_ORDER
    .filter(type => (hand[type] || 0) > 0)
    .map(type => ({
      type,
      kanji: PIECE_KANJI[type].base,
      count: hand[type],
    }))
</script>

<div class="tray" class:black={color === 'black'} class:white={color === 'white'}>
  <span class="label">{label}</span>
  <div class="pieces">
    {#each pieces as p}
      <div class="hand-piece">
        <span class="kanji">{p.kanji}</span>
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
  }

  .tray.black {
    background: #1a2a1a;
    border-radius: 0 0 6px 6px;
    border-top: none;
  }

  .tray.white {
    background: #1a1a2e;
    border-radius: 6px 6px 0 0;
    border-bottom: none;
  }

  .label {
    font-size: 11px;
    color: var(--text-secondary);
    min-width: 55px;
  }

  .pieces {
    display: flex;
    gap: 4px;
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
    background: var(--accent-green);
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
    border-color: #555;
  }

  .tray.black .hand-piece {
    border-color: var(--accent-green);
  }

  .empty {
    color: var(--text-muted);
    font-size: 12px;
  }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/PieceTray.svelte
git commit -m "feat: piece tray component with kanji tiles and count badges"
```
