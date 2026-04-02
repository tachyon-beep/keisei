import { describe, it, expect } from 'vitest'
import { safeParse } from './safeParse.js'

describe('safeParse', () => {
  it('parses valid JSON string', () => {
    expect(safeParse('{"a":1}', {})).toEqual({ a: 1 })
  })

  it('parses valid JSON array string', () => {
    expect(safeParse('[1,2,3]', [])).toEqual([1, 2, 3])
  })

  it('returns fallback for invalid JSON', () => {
    expect(safeParse('{broken', 'default')).toBe('default')
  })

  it('returns fallback for empty string', () => {
    expect(safeParse('', [])).toEqual([])
  })

  it('returns non-string input as-is (object)', () => {
    const obj = { already: 'parsed' }
    expect(safeParse(obj, {})).toBe(obj)
  })

  it('returns non-string input as-is (array)', () => {
    const arr = [1, 2]
    expect(safeParse(arr, [])).toBe(arr)
  })

  it('returns non-string input as-is (null)', () => {
    expect(safeParse(null, 'fallback')).toBeNull()
  })

  it('returns non-string input as-is (undefined)', () => {
    expect(safeParse(undefined, 'fallback')).toBeUndefined()
  })

  it('returns non-string input as-is (number)', () => {
    expect(safeParse(42, 0)).toBe(42)
  })
})
