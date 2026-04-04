import { writable } from 'svelte/store'

const stored = typeof localStorage !== 'undefined' && localStorage.getItem('activeTab')
export const activeTab = writable(stored || 'training')

activeTab.subscribe(val => {
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem('activeTab', val)
  }
})
