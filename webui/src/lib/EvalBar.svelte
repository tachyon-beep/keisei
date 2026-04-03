<script>
  import { computeEval } from './evalCalc.js'

  /** Value estimate from the model's value head. Range roughly -1 to +1. */
  export let value = 0
  /** Who is currently to move. */
  export let currentPlayer = 'black'

  $: ({ blackPct, displayValue } = computeEval(value, currentPlayer))
</script>

<div class="eval-bar" title="Value estimate: {displayValue}">
  <div class="label top">☖</div>
  <div class="bar">
    <div class="white-fill" style="flex: {100 - blackPct}"></div>
    <div class="midline"></div>
    <div class="black-fill" style="flex: {blackPct}"></div>
  </div>
  <div class="value-label">{displayValue}</div>
  <div class="label bottom">☗</div>
</div>

<style>
  .eval-bar {
    display: flex;
    flex-direction: column;
    align-items: center;
    align-self: stretch;
    gap: 4px;
    width: 40px;
    flex-shrink: 0;
  }

  .label {
    font-size: 14px;
    color: var(--text-muted);
    line-height: 1;
  }

  .bar {
    flex: 1;
    width: 24px;
    border-radius: 4px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    position: relative;
    border: 1px solid var(--border);
    min-height: 200px;
  }

  .white-fill {
    background: var(--eval-white);
    min-height: 0;
    transition: flex 0.3s ease;
  }

  .black-fill {
    background: var(--eval-black);
    min-height: 0;
    transition: flex 0.3s ease;
  }

  .midline {
    position: absolute;
    top: 50%;
    left: 0;
    right: 0;
    height: 1px;
    background: var(--accent-gold);
    opacity: 0.6;
  }

  .value-label {
    font-size: 11px;
    font-family: monospace;
    color: var(--text-secondary);
    white-space: nowrap;
  }
</style>
