<script>
  import { selectedGameId } from '../stores/games.js'
  import { pieceKanji } from './pieces.js'
  import { parseBoard, getStatusText, getAdvantage } from './gameThumbnail.js'

  export let game

  $: selected = $selectedGameId === game.game_id
  $: board = parseBoard(game)
  $: statusText = getStatusText(game)
  $: ({ confident, favours } = getAdvantage(game))

  function handleClick() {
    selectedGameId.set(game.game_id)
  }
</script>

<button
  class="thumbnail"
  class:selected
  on:click={handleClick}
  aria-label="Game {game.game_id + 1}, {statusText}"
  aria-pressed={selected}
>
  <div class="mini-board" aria-hidden="true">
    {#each Array(81) as _, idx}
      {@const piece = board[idx]}
      <div class="cell">
        {#if piece}
          <span class:white={piece.color === 'white'} class:promoted={piece.promoted} lang="ja">
            {pieceKanji(piece.type, piece.promoted, piece.color)}
          </span>
        {/if}
      </div>
    {/each}
  </div>
  <div class="label">
    {#if confident}<span class="confidence-dot" class:black-dot={favours === 'black'} class:white-dot={favours === 'white'}></span>{/if}
    G{game.game_id + 1} — {statusText}
  </div>
</button>

<style>
  .thumbnail {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px;
    cursor: pointer;
    text-align: center;
    transition: border-color 0.15s;
    width: 100%;
  }

  .thumbnail:hover {
    border-color: var(--text-secondary);
  }

  .thumbnail:focus-visible {
    outline: 2px solid var(--accent-blue);
    outline-offset: 2px;
  }

  .thumbnail.selected {
    border: 2px solid var(--accent-green);
    background: var(--bg-selected);
  }

  .mini-board {
    display: grid;
    grid-template-columns: repeat(9, 1fr);
    grid-template-rows: repeat(9, 1fr);
    width: 100%;
    aspect-ratio: 1;
    margin: 0 auto;
    background: var(--bg-board);
    border: 1px solid var(--border-board);
    border-radius: 2px;
  }

  .cell {
    display: flex;
    align-items: center;
    justify-content: center;
    border: 0.5px solid var(--border-board-inner);
    font-size: 6px;
    line-height: 1;
  }

  .cell span.white {
    transform: rotate(180deg);
  }

  .cell span.promoted {
    color: var(--promoted);
  }

  .label {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
  }

  .confidence-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .confidence-dot.black-dot {
    background: #1a1a1a;
    border: 1px solid var(--text-muted);
  }

  .confidence-dot.white-dot {
    background: #e0e0e0;
    border: 1px solid var(--text-muted);
  }

  .thumbnail.selected .label {
    color: var(--accent-green);
  }

  @media (prefers-reduced-motion: reduce) {
    .thumbnail {
      transition: none;
    }
  }
</style>
