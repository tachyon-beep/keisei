/**
 * Extract columnar arrays from metrics rows for charting.
 * @param {Array<Object>} rows - Metrics row objects from the store.
 * @returns {Object} Columnar arrays keyed by chart-friendly names.
 */
export function extractColumns(rows) {
  const steps = [],
    policyLoss = [],
    valueLoss = [],
    winRate = [],
    lossRate = [],
    blackWinRate = [],
    whiteWinRate = [],
    drawRate = [],
    avgEpLen = [],
    entropy = [],
    epochs = [],
    pvRatio = []
  for (const r of rows) {
    steps.push(r.step || 0)
    policyLoss.push(r.policy_loss ?? null)
    valueLoss.push(r.value_loss ?? null)
    const pl = r.policy_loss ?? null
    const vl = r.value_loss ?? null
    pvRatio.push(pl != null && vl != null && vl !== 0 ? Math.abs(pl / vl) : null)
    winRate.push(r.win_rate ?? null)
    lossRate.push(r.loss_rate ?? null)
    blackWinRate.push(r.black_win_rate ?? null)
    whiteWinRate.push(r.white_win_rate ?? null)
    drawRate.push(r.draw_rate ?? null)
    avgEpLen.push(r.avg_episode_length ?? null)
    entropy.push(r.entropy ?? null)
    epochs.push(r.epoch || 0)
  }
  return {
    steps,
    policyLoss,
    valueLoss,
    winRate,
    lossRate,
    blackWinRate,
    whiteWinRate,
    drawRate,
    avgEpLen,
    entropy,
    epochs,
    pvRatio,
  }
}
