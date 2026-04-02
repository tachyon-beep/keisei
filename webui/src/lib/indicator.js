/**
 * Determine the status indicator dot color and text
 * based on training liveness and status string.
 *
 * @param {boolean} alive - Whether the training heartbeat is fresh.
 * @param {string} status - Training status ('running', 'completed', 'paused', etc.)
 * @returns {{ dot: string, text: string }}
 */
export function getIndicator(alive, status) {
  if (alive) return { dot: 'green', text: 'Training alive' }
  if (status === 'completed') return { dot: 'red', text: 'Training completed' }
  if (status === 'paused') return { dot: 'red', text: 'Training paused' }
  return { dot: 'yellow', text: 'Training stale' }
}
