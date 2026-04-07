// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'

function collapseEvents(events) {
  const out = []
  for (const event of events) {
    const prev = out[out.length - 1]
    if (prev && !prev.collapsed && prev.type === event.type && prev.time === event.time) {
      out[out.length - 1] = {
        collapsed: true, type: event.type, icon: event.icon,
        time: event.time, count: 2, names: [prev.name, event.name],
      }
    } else if (prev?.collapsed && prev.type === event.type && prev.time === event.time) {
      prev.count++
      prev.names.push(event.name)
    } else {
      out.push(event)
    }
  }
  return out
}

describe('collapseEvents', () => {
  it('returns empty array for empty input', () => {
    expect(collapseEvents([])).toEqual([])
  })

  it('passes through a single event unchanged', () => {
    const events = [{ type: 'arrival', time: '12:00:00', icon: '→', name: 'Bot-A', detail: 'joined' }]
    const result = collapseEvents(events)
    expect(result).toHaveLength(1)
    expect(result[0].collapsed).toBeUndefined()
    expect(result[0].name).toBe('Bot-A')
  })

  it('collapses consecutive same-type same-time events', () => {
    const events = [
      { type: 'arrival', time: '12:00:00', icon: '→', name: 'Bot-A', detail: 'joined' },
      { type: 'arrival', time: '12:00:00', icon: '→', name: 'Bot-B', detail: 'joined' },
      { type: 'arrival', time: '12:00:00', icon: '→', name: 'Bot-C', detail: 'joined' },
    ]
    const result = collapseEvents(events)
    expect(result).toHaveLength(1)
    expect(result[0].collapsed).toBe(true)
    expect(result[0].count).toBe(3)
    expect(result[0].names).toEqual(['Bot-A', 'Bot-B', 'Bot-C'])
  })

  it('does not collapse events of different types', () => {
    const events = [
      { type: 'arrival', time: '12:00:00', icon: '→', name: 'Bot-A' },
      { type: 'departure', time: '12:00:00', icon: '←', name: 'Bot-B' },
    ]
    const result = collapseEvents(events)
    expect(result).toHaveLength(2)
  })

  it('does not collapse events at different times', () => {
    const events = [
      { type: 'arrival', time: '12:00:00', icon: '→', name: 'Bot-A' },
      { type: 'arrival', time: '12:00:05', icon: '→', name: 'Bot-B' },
    ]
    const result = collapseEvents(events)
    expect(result).toHaveLength(2)
  })
})
