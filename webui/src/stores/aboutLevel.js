import { writable } from 'svelte/store'

// Level scale for "About" tab progressive disclosure.
//   1 = The Big Idea     (plain English, no jargon)
//   2 = Learning Loop    (concepts, self-play loop)
//   3 = Inside the Demo  (observation tensor, architecture, shapes)
//   4 = Algorithmic      (PPO loss math, hyperparameters)
//   5 = Research View    (limitations, assumptions, open questions)
export const ABOUT_LEVELS = [
  { id: 1, label: 'The Big Idea', blurb: 'Plain English, big picture' },
  { id: 2, label: 'Learning Loop', blurb: '+ How self-play works' },
  { id: 3, label: 'Inside the Demo', blurb: '+ Architecture & shapes' },
  { id: 4, label: 'Algorithmic', blurb: '+ PPO loss & hyperparameters' },
  { id: 5, label: 'Research View', blurb: '+ Limitations & open questions' },
]

const KEY = 'aboutLevel'

function loadInitial() {
  if (typeof localStorage === 'undefined') return 2
  const raw = localStorage.getItem(KEY)
  const n = Number.parseInt(raw, 10)
  return Number.isInteger(n) && n >= 1 && n <= 5 ? n : 2
}

export const aboutLevel = writable(loadInitial())

aboutLevel.subscribe((val) => {
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(KEY, String(val))
  }
})
