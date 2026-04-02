import { describe, it, expect } from 'vitest'
import { getIndicator } from './indicator.js'

describe('getIndicator', () => {
  it('returns green when alive regardless of status', () => {
    expect(getIndicator(true, 'running')).toEqual({ dot: 'green', text: 'Training alive' })
    expect(getIndicator(true, 'completed')).toEqual({ dot: 'green', text: 'Training alive' })
    expect(getIndicator(true, 'paused')).toEqual({ dot: 'green', text: 'Training alive' })
  })

  it('returns red for completed when not alive', () => {
    expect(getIndicator(false, 'completed')).toEqual({ dot: 'red', text: 'Training completed' })
  })

  it('returns red for paused when not alive', () => {
    expect(getIndicator(false, 'paused')).toEqual({ dot: 'red', text: 'Training paused' })
  })

  it('returns yellow (stale) for running when not alive', () => {
    expect(getIndicator(false, 'running')).toEqual({ dot: 'yellow', text: 'Training stale' })
  })

  it('returns yellow for unknown status when not alive', () => {
    expect(getIndicator(false, 'unknown')).toEqual({ dot: 'yellow', text: 'Training stale' })
  })

  it('returns yellow for empty status when not alive', () => {
    expect(getIndicator(false, '')).toEqual({ dot: 'yellow', text: 'Training stale' })
  })
})
