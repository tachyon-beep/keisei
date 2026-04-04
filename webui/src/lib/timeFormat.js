/**
 * Time formatting helpers extracted from StatusIndicator.svelte
 * for testability.
 */

/**
 * Parse a timestamp string as UTC. Appends 'Z' if missing to ensure
 * the Date constructor treats it as UTC rather than local time.
 */
export function parseUTC(s) {
  if (!s) return null
  return new Date(s + (s.endsWith('Z') ? '' : 'Z'))
}

/**
 * Format a duration in milliseconds as a human-readable elapsed string.
 * Negative values are clamped to 0.
 *
 * Examples:
 *   formatElapsed(90000)    → "01m 30s"
 *   formatElapsed(3661000)  → "01h 01m 01s"
 *   formatElapsed(90061000) → "1d 01h 01m 01s"
 */
export function formatElapsed(ms) {
  if (ms < 0) ms = 0
  const s = Math.floor(ms / 1000)
  const days = Math.floor(s / 86400)
  const hrs = Math.floor((s % 86400) / 3600)
  const mins = Math.floor((s % 3600) / 60)
  const secs = s % 60
  const pad = (n) => String(n).padStart(2, '0')
  if (days > 0) return `${days}d ${pad(hrs)}h ${pad(mins)}m ${pad(secs)}s`
  if (hrs > 0) return `${pad(hrs)}h ${pad(mins)}m ${pad(secs)}s`
  return `${pad(mins)}m ${pad(secs)}s`
}
