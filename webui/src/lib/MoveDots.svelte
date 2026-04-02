<script>
  /**
   * Tiny directional grid showing how a piece moves.
   * ■ = one square, arrow = slides (unlimited range), □ = can't move.
   *
   * Grid is always 3x3. Centre cell is the piece's position (shown as ·).
   * Optional `extra` row sits above the grid for knight-style jumps.
   */

  /** @type {Array<Array<'step' | 'slide' | null>>} 3x3 grid */
  export let pattern = []

  /** @type {Array<'step' | 'slide' | null>} optional extra row above grid (3 cells) */
  export let extra = null

  /** Whether this is a promoted piece (tints the markers) */
  export let promoted = false

  function arrowFor(r, c) {
    const dr = r - 1
    const dc = c - 1
    if (dr < 0 && dc === 0) return '↑'
    if (dr > 0 && dc === 0) return '↓'
    if (dr === 0 && dc < 0) return '←'
    if (dr === 0 && dc > 0) return '→'
    if (dr < 0 && dc < 0) return '↖'
    if (dr < 0 && dc > 0) return '↗'
    if (dr > 0 && dc < 0) return '↙'
    if (dr > 0 && dc > 0) return '↘'
    return ''
  }

  function glyph(cell) {
    if (cell === 'step') return '■'
    if (cell === 'slide') return '→'
    return '□'
  }
</script>

<div class="move-dots" class:promoted class:has-extra={extra != null} role="img" aria-hidden="true">
  {#if extra}
    <div class="extra-row">
      {#each extra as cell}
        <span class="cell" class:step={cell === 'step'} class:slide={cell === 'slide'} class:empty={cell === null}>
          {#if cell === 'step'}■{:else if cell === 'slide'}↑{:else}□{/if}
        </span>
      {/each}
    </div>
  {/if}
  <div class="grid">
    {#each pattern as row, r}
      {#each row as cell, c}
        {@const isCentre = r === 1 && c === 1}
        <span class="cell" class:centre={isCentre} class:step={!isCentre && cell === 'step'} class:slide={!isCentre && cell === 'slide'} class:empty={!isCentre && cell === null}>
          {#if isCentre}
            ·
          {:else if cell === 'step'}
            ■
          {:else if cell === 'slide'}
            {arrowFor(r, c)}
          {:else}
            □
          {/if}
        </span>
      {/each}
    {/each}
  </div>
</div>

<style>
  .move-dots {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex-shrink: 0;
    width: 30px;
  }

  .extra-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    width: 100%;
    height: 10px;
    margin-bottom: 1px;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(3, 1fr);
    width: 100%;
    height: 30px;
  }

  .cell {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    line-height: 1;
    color: transparent;
  }

  .cell.centre {
    color: var(--text-muted);
    font-size: 10px;
  }

  .cell.empty {
    color: var(--border);
  }

  .cell.step {
    color: var(--accent-teal);
  }

  .cell.slide {
    color: var(--accent-gold);
  }

  .move-dots.promoted .cell.step {
    color: var(--promoted-bright);
  }

  .move-dots.promoted .cell.slide {
    color: var(--promoted-bright);
  }
</style>
