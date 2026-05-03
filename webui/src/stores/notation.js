import { writable } from 'svelte/store'

/**
 * Global notation-style preference shared by all panels that render move
 * notation (MoveLog, CommentaryPanel). Hoisted to a store so toggling the
 * style anywhere updates every panel in sync, and persisted to localStorage
 * so the user's preference survives reloads.
 */
export const NOTATION_STYLES = ['western', 'japanese', 'usi']
export const NOTATION_LABELS = { western: 'W', japanese: '漢', usi: 'USI' }
export const NOTATION_NAMES = { western: 'Western', japanese: 'Japanese', usi: 'USI' }

const STORAGE_KEY = 'notationStyle'

function loadInitial() {
  if (typeof localStorage === 'undefined') return 'western'
  const v = localStorage.getItem(STORAGE_KEY)
  return NOTATION_STYLES.includes(v) ? v : 'western'
}

export const notationStyle = writable(loadInitial())

notationStyle.subscribe((val) => {
  if (typeof localStorage !== 'undefined' && NOTATION_STYLES.includes(val)) {
    localStorage.setItem(STORAGE_KEY, val)
  }
})

/** Cycle to the next notation style. */
export function cycleNotation() {
  notationStyle.update(current => {
    const idx = NOTATION_STYLES.indexOf(current)
    return NOTATION_STYLES[(idx + 1) % NOTATION_STYLES.length]
  })
}
