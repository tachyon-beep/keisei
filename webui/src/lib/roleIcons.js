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
  frontier_static: { icon: '🛡', label: 'Frontier', cssClass: 'role-frontier' },
  recent_fixed:    { icon: '✦',  label: 'Recent',   cssClass: 'role-recent' },
  dynamic:         { icon: '⚔',  label: 'Dynamic',  cssClass: 'role-dynamic' },
  historical:      { icon: '📜', label: 'Historical', cssClass: 'role-historical' },
}

const UNKNOWN = { icon: '?', label: 'Unknown', cssClass: 'role-unknown' }

/** Get the full role descriptor { icon, label, cssClass } for a role string. */
export function getRoleInfo(role) {
  return ROLES[role] || UNKNOWN
}

/** Get just the icon character for a role. */
export function getRoleIcon(role) {
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
