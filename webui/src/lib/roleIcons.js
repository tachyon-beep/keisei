/**
 * Role icon/badge definitions for league entry tiers.
 *
 * Design spec: docs/concepts/tiered-opponent-pool.md §15.3
 *
 *   Frontier Static  — shield
 *   Recent Fixed     — spark
 *   Dynamic          — crossed swords
 *   Historical       — scroll
 */

const ROLES = {
  frontier_static: { icon: '🛡', label: 'Frontier', tooltip: 'Frontier — strongest fixed checkpoints that set the performance ceiling', cssClass: 'role-frontier' },
  recent_fixed:    { icon: '✦',  label: 'Recent',   tooltip: 'Recent — recently saved snapshots of the training policy', cssClass: 'role-recent' },
  dynamic:         { icon: '⚔',  label: 'Dynamic',  tooltip: 'Dynamic — evolving opponents that adapt during training', cssClass: 'role-dynamic' },
  historical:      { icon: '📜', label: 'Historical', tooltip: 'Historical — archived policies from earlier training runs', cssClass: 'role-historical' },
}

const RETIRED = { icon: '⏸', label: 'Retired', tooltip: 'Retired — removed from the active opponent pool', cssClass: 'role-retired' }
const UNKNOWN = { icon: '?', label: 'Unknown', tooltip: 'Unknown role', cssClass: 'role-unknown' }

/** Get the full role descriptor { icon, label, tooltip, cssClass } for a role string. */
export function getRoleInfo(role, status) {
  if (status === 'retired') return RETIRED
  return ROLES[role] || UNKNOWN
}

/** Get just the icon character for a role. */
export function getRoleIcon(role, status) {
  if (status === 'retired') return RETIRED.icon
  return (ROLES[role] || UNKNOWN).icon
}

/** Get the short label for a role. */
export function getRoleLabel(role) {
  return (ROLES[role] || UNKNOWN).label
}

/** Get the CSS class for a role badge. */
export function getRoleCssClass(role) {
  return (ROLES[role] || UNKNOWN).cssClass
}
