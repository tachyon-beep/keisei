/**
 * Compute eval-bar display values from model value estimate.
 *
 * @param {number} value - Raw value estimate, roughly -1 to +1.
 * @param {string} currentPlayer - 'black' or 'white'.
 * @returns {{ blackPct: number, displayValue: string }}
 */
export function computeEval(value, currentPlayer) {
  const clamped = Math.max(-1, Math.min(1, value))
  const blackAdvantage = currentPlayer === 'black' ? clamped : -clamped
  const blackPct = 50 + blackAdvantage * 50
  const displayValue = Math.abs(blackAdvantage) < 0.005
    ? '0.00'
    : (blackAdvantage > 0 ? '+' : '') + blackAdvantage.toFixed(2)
  return { blackPct, displayValue }
}
