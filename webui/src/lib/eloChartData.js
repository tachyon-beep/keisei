const COLORS = ['#4ade80', '#60a5fa', '#f59e0b', '#a78bfa', '#f472b6']

/**
 * Transform flat elo_history array into grouped chart data for MetricsChart.
 * @param {Array<{entry_id: number, epoch: number, elo_rating: number}>} history
 * @param {Array<{id: number, architecture: string, elo_rating: number}>} entries
 * @returns {{ xData: number[], series: Array<{label: string, data: number[], color: string}> }}
 */
export function buildEloChartData(history, entries) {
  if (history.length === 0) return { xData: [], series: [] }

  // Filter out any legacy epoch=-1 sentinel values
  const filtered = history.filter(h => h.epoch >= 0)
  if (filtered.length === 0) return { xData: [], series: [] }

  const epochSet = new Set(filtered.map(h => h.epoch))
  const xData = [...epochSet].sort((a, b) => a - b)
  const epochIndex = new Map(xData.map((e, i) => [e, i]))

  const entryIds = entries.map(e => e.id)
  const entryMap = new Map(entries.map(e => [e.id, e]))

  const series = entryIds
    .filter(id => filtered.some(h => h.entry_id === id))
    .map((id, i) => {
      const entry = entryMap.get(id)
      const data = new Array(xData.length).fill(null)
      for (const h of filtered) {
        if (h.entry_id === id) {
          data[epochIndex.get(h.epoch)] = h.elo_rating
        }
      }
      const name = entry.display_name || entry.architecture
      return {
        label: `${name} (${Math.round(entry.elo_rating)})`,
        data,
        color: COLORS[i % COLORS.length],
      }
    })

  return { xData, series }
}
