import { writable } from 'svelte/store'

const stored = typeof localStorage !== 'undefined' ? localStorage.getItem('keisei-theme') : null
export const theme = writable(stored || 'dark')

theme.subscribe((value) => {
  if (typeof document !== 'undefined') {
    document.documentElement.setAttribute('data-theme', value)
  }
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem('keisei-theme', value)
  }
})

export function toggleTheme() {
  theme.update((t) => (t === 'dark' ? 'light' : 'dark'))
}
