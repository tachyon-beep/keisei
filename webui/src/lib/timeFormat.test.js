import { describe, it, expect } from 'vitest'
import { parseUTC, formatElapsed } from './timeFormat.js'

describe('parseUTC', () => {
  it('returns null for falsy input', () => {
    expect(parseUTC(null)).toBeNull()
    expect(parseUTC(undefined)).toBeNull()
    expect(parseUTC('')).toBeNull()
  })

  it('parses a Z-terminated ISO string as UTC', () => {
    const d = parseUTC('2026-04-01T12:00:00Z')
    expect(d).toBeInstanceOf(Date)
    expect(d.getUTCHours()).toBe(12)
    expect(d.getUTCMinutes()).toBe(0)
  })

  it('appends Z when missing to force UTC interpretation', () => {
    const d = parseUTC('2026-04-01T12:00:00')
    expect(d.getUTCHours()).toBe(12)
  })

  it('does not double-append Z if already present', () => {
    const a = parseUTC('2026-04-01T00:00:00Z')
    const b = parseUTC('2026-04-01T00:00:00')
    expect(a.getTime()).toBe(b.getTime())
  })
})

describe('formatElapsed', () => {
  it('formats zero milliseconds', () => {
    expect(formatElapsed(0)).toBe('00m 00s')
  })

  it('clamps negative values to zero', () => {
    expect(formatElapsed(-5000)).toBe('00m 00s')
  })

  it('formats seconds only', () => {
    expect(formatElapsed(45000)).toBe('00m 45s')
  })

  it('formats minutes and seconds', () => {
    expect(formatElapsed(90000)).toBe('01m 30s')
  })

  it('formats hours, minutes, and seconds', () => {
    expect(formatElapsed(3661000)).toBe('01h 01m 01s')
  })

  it('formats days, hours, minutes, and seconds', () => {
    expect(formatElapsed(90061000)).toBe('1d 01h 01m 01s')
  })

  it('formats exactly one day', () => {
    expect(formatElapsed(86400000)).toBe('1d 00h 00m 00s')
  })

  it('formats exactly one hour (no days prefix)', () => {
    expect(formatElapsed(3600000)).toBe('01h 00m 00s')
  })

  it('handles multiple days', () => {
    // 3 days, 2 hours, 5 minutes, 30 seconds
    const ms = (3 * 86400 + 2 * 3600 + 5 * 60 + 30) * 1000
    expect(formatElapsed(ms)).toBe('3d 02h 05m 30s')
  })

  it('handles sub-second values (rounds down)', () => {
    expect(formatElapsed(999)).toBe('00m 00s')
    expect(formatElapsed(1500)).toBe('00m 01s')
  })
})
