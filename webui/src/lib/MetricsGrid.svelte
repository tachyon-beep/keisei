<script>
  import { metrics } from '../stores/metrics.js'
  import MetricsChart from './MetricsChart.svelte'
  import { extractColumns } from './metricsColumns.js'

  $: columns = extractColumns($metrics)

  let expandedIndex = 0

  const charts = [
    { title: 'Policy & Value Loss', xKey: 'steps', series: (c) => [
      { label: 'Policy', data: c.policyLoss, color: '#f59e0b' },
      { label: 'Value', data: c.valueLoss, color: '#60a5fa' },
    ], annotation: 'Both should fall together — divergence may indicate overfitting' },
    { title: 'Win Rate', xKey: 'epochs', series: (c) => [
      { label: '☗ Black', data: c.blackWinRate, color: '#e0e0e0' },
      { label: '☖ White', data: c.whiteWinRate, color: '#60a5fa' },
      { label: 'Draw', data: c.drawRate, color: '#f59e0b' },
    ], annotation: 'Black has first-move advantage — expect ~55/45 split at convergence' },
    { title: 'Avg Episode Length', xKey: 'epochs', series: (c) => [
      { label: 'Episode Length', data: c.avgEpLen, color: '#a78bfa' },
    ], annotation: 'Longer games = more strategic play' },
    { title: 'Policy Entropy', xKey: 'steps', series: (c) => [
      { label: 'Entropy', data: c.entropy, color: '#f472b6' },
    ], annotation: 'Falling entropy = agent becoming more decisive' },
  ]

  function handleClick(index) {
    expandedIndex = expandedIndex === index ? null : index
  }
</script>

<div class="metrics-grid">
  <h2 class="grid-header">
    Training Metrics {#if $metrics.length > 0}— Epoch {$metrics[$metrics.length - 1]?.epoch ?? '?'}{/if}
  </h2>

  <div class="layout">
    <div class="mini-column">
      {#each charts as chart, i}
        <button
          class="mini-chart-btn"
          class:active={expandedIndex === i}
          on:click={() => handleClick(i)}
          aria-expanded={expandedIndex === i}
          aria-label="{chart.title} — click to {expandedIndex === i ? 'collapse' : 'expand'}"
        >
          <MetricsChart
            title={chart.title}
            xData={columns[chart.xKey]}
            series={chart.series(columns)}
            height={140}
            compact={true}
          />
        </button>
      {/each}
    </div>

    <div class="expanded-area">
      {#key expandedIndex}
        {#if expandedIndex != null}
          {@const chart = charts[expandedIndex]}
          <div class="expanded-chart">
            <button class="collapse-btn" on:click={() => expandedIndex = null} aria-label="Collapse chart">✕</button>
            <MetricsChart
              title={chart.title}
              xData={columns[chart.xKey]}
              series={chart.series(columns)}
              height={280}
              annotation={chart.annotation || null}
              compact={false}
            />
          </div>
        {:else}
          <p class="expand-hint">Click a chart to expand</p>
        {/if}
      {/key}
    </div>
  </div>
</div>

<style>
  .metrics-grid {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 12px;
  }

  h2.grid-header {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }

  .layout {
    display: flex;
    gap: 10px;
  }

  .mini-column {
    flex: 0 0 880px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    align-content: start;
  }

  .expanded-area {
    flex: 1 1 auto;
    min-width: 0;
  }

  .expand-hint {
    color: var(--text-muted);
    font-size: 12px;
    text-align: center;
    padding-top: 40px;
  }

  .mini-chart-btn {
    all: unset;
    cursor: pointer;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    transition: border-color 0.15s;
  }

  .mini-chart-btn:hover {
    border-color: var(--text-secondary);
  }

  .mini-chart-btn:focus-visible {
    outline: 2px solid var(--accent-blue);
    outline-offset: 2px;
  }

  .mini-chart-btn.active {
    border-color: var(--accent-green);
  }

  .expanded-chart {
    position: relative;
  }

  .collapse-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    z-index: 1;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 2px 6px;
    font-size: 12px;
  }

  .collapse-btn:hover {
    color: var(--text-primary);
    border-color: var(--text-secondary);
  }

  @media (max-width: 768px) {
    .layout {
      flex-direction: column;
    }

    .mini-column {
      flex: 0 0 auto;
      grid-template-columns: 1fr 1fr;
    }
  }

  @media (max-width: 480px) {
    .mini-column {
      grid-template-columns: 1fr;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .mini-chart-btn { transition: none; }
  }
</style>
