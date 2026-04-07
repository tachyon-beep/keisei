<script>
  import { metrics } from '../stores/metrics.js'
  import { theme } from '../stores/theme.js'
  import MetricsChart from './MetricsChart.svelte'
  import { extractColumns } from './metricsColumns.js'

  $: columns = extractColumns($metrics)

  let expandedIndex = 0

  // Resolve chart colors from CSS variables so they adapt to the active theme
  function chartColor(varName) {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim()
  }

  // Re-read colors whenever the theme store changes
  let chartColors = {}
  $: if ($theme || $metrics) {
    chartColors = {
      gold: chartColor('--chart-gold'),
      ink: chartColor('--chart-ink'),
      cream: chartColor('--chart-cream'),
      teal: chartColor('--chart-teal'),
      moss: chartColor('--chart-moss'),
    }
  }

  $: charts = [
    { title: 'Policy & Value Loss', xKey: 'steps', xLabel: 'Step', series: (c) => [
      { label: 'Policy', data: c.policyLoss, color: chartColors.gold || '#c8962e' },
      { label: 'Value', data: c.valueLoss, color: chartColors.ink || '#7eb8d4' },
    ], annotation: 'Both should fall together — divergence may indicate overfitting' },
    { title: 'Win Rate', xKey: 'epochs', xLabel: 'Epoch', series: (c) => [
      { label: '☗ Black', data: c.blackWinRate, color: chartColors.cream || '#e8e0d4' },
      { label: '☖ White', data: c.whiteWinRate, color: chartColors.ink || '#7eb8d4' },
      { label: 'Draw', data: c.drawRate, color: chartColors.gold || '#c8962e' },
    ], annotation: 'Black has first-move advantage in shogi' },
    { title: 'Avg Episode Length', xKey: 'epochs', xLabel: 'Epoch', series: (c) => [
      { label: 'Episode Length', data: c.avgEpLen, color: chartColors.teal || '#4db8a8' },
    ], annotation: 'Longer games = more strategic play' },
    { title: 'Policy Entropy', xKey: 'steps', xLabel: 'Step', series: (c) => [
      { label: 'Entropy', data: c.entropy, color: chartColors.moss || '#6b9e6b' },
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

  {#if $metrics.length === 0}
    <p class="empty-state">Metrics will appear once training begins.</p>
  {:else}
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
            xLabel={chart.xLabel || null}
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
              xLabel={chart.xLabel || null}
            />
          </div>
        {:else}
          <p class="expand-hint">Click a chart to expand</p>
        {/if}
      {/key}
    </div>
  </div>
  {/if}
</div>

<style>
  .metrics-grid {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 12px;
  }

  h2.grid-header {
    font-size: 13px;
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
    flex: 0 0 clamp(400px, 60%, 880px);
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    align-content: start;
  }

  .expanded-area {
    flex: 1 1 auto;
    min-width: 0;
  }

  .empty-state {
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    padding: 32px 16px;
  }

  .expand-hint {
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    padding-top: 40px;
  }

  .mini-chart-btn {
    background: none;
    color: inherit;
    font: inherit;
    padding: 0;
    cursor: pointer;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    transition: border-color 0.15s;
    text-align: left;
  }

  .mini-chart-btn:hover {
    border-color: var(--text-secondary);
  }

  .mini-chart-btn:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }

  .mini-chart-btn.active {
    border-color: var(--accent-teal);
  }

  .expanded-chart {
    position: relative;
  }

  .collapse-btn {
    position: absolute;
    top: 4px;
    right: 4px;
    z-index: 1;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    min-width: 44px;
    min-height: 44px;
    padding: 4px 8px;
    font-size: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .collapse-btn:hover {
    color: var(--text-primary);
    border-color: var(--text-secondary);
    background: var(--bg-card);
  }

  .collapse-btn:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
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
