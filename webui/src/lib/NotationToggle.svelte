<script>
  import { notationStyle, NOTATION_STYLES, NOTATION_LABELS, NOTATION_NAMES, cycleNotation } from '../stores/notation.js'

  $: nextStyle = NOTATION_STYLES[(NOTATION_STYLES.indexOf($notationStyle) + 1) % NOTATION_STYLES.length]
</script>

<span class="notation-toggle-wrap">
  <span class="notation-label" aria-hidden="true">Notation:</span>
  <button
    class="notation-toggle"
    on:click={cycleNotation}
    title="Switch to {NOTATION_NAMES[nextStyle]} notation"
    aria-label="Notation: {NOTATION_NAMES[$notationStyle]}. Click to switch to {NOTATION_NAMES[nextStyle]}."
  >
    {NOTATION_LABELS[$notationStyle]}
  </button>
</span>

<style>
  .notation-toggle-wrap {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .notation-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .notation-toggle {
    background: none;
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 4px 10px;
    min-width: 44px;
    min-height: 44px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .notation-toggle:hover {
    color: var(--text-primary);
    border-color: var(--text-secondary);
  }
  .notation-toggle:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }
</style>
