<script>
  /**
   * Tiny directional grid showing how a piece moves.
   * ■ = one square, single arrow = slides, double arrow = knight leap, □ = can't move.
   *
   * Grid is always 3x3. Centre cell is the piece's position (shown as ·).
   */

  /** @type {Array<Array<'step' | 'slide' | 'jump' | null>>} 3x3 grid */
  export let pattern = []

  /** Whether this is a promoted piece (tints the markers) */
  export let promoted = false

  function slideArrow(r, c) {
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

  // Knight only leaps forward; column tells us which diagonal.
  function jumpArrow(_r, c) {
    return c < 1 ? '⇖' : '⇗'
  }
</script>

<div class="move-dots" class:promoted role="img" aria-hidden="true">
  <div class="grid">
    {#each pattern as row, r}
      {#each row as cell, c}
        {@const isCentre = r === 1 && c === 1}
        <span
          class="cell"
          class:centre={isCentre}
          class:step={!isCentre && cell === 'step'}
          class:slide={!isCentre && cell === 'slide'}
          class:jump={!isCentre && cell === 'jump'}
          class:empty={!isCentre && cell === null}
        >
          {#if isCentre}
            ·
          {:else if cell === 'step'}
            ■
          {:else if cell === 'slide'}
            {slideArrow(r, c)}
          {:else if cell === 'jump'}
            {jumpArrow(r, c)}
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

  .cell.jump {
    color: var(--accent-gold);
    font-size: 11px;
  }

  .move-dots.promoted .cell.step {
    color: var(--promoted-bright);
  }

  .move-dots.promoted .cell.slide {
    color: var(--promoted-bright);
  }

  .move-dots.promoted .cell.jump {
    color: var(--promoted-bright);
  }
</style>
