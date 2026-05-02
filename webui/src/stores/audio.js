import { writable } from 'svelte/store'

// Background-music toggle for the dashboard. The actual <audio> element
// lives in App.svelte and reacts to this store.
//
// Persistence: the user's last on/off choice is saved to localStorage so
// the button reflects the prior state on reload. Audio does NOT autoplay
// on load — browser policy requires a user gesture, and an unprompted
// reload-into-music UX is jarring. App.svelte resets the store to false
// if play() rejects, so a stale "true" from localStorage gracefully
// resolves to "off until clicked".

export const AUDIO_VOLUME = 0.4

const KEY = 'audioEnabled'

function loadInitial() {
  if (typeof localStorage === 'undefined') return false
  return localStorage.getItem(KEY) === 'true'
}

export const audioEnabled = writable(loadInitial())

audioEnabled.subscribe((val) => {
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(KEY, val ? 'true' : 'false')
  }
})
