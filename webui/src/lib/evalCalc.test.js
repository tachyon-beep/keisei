import { describe, it, expect } from 'vitest'
import { computeEval } from './evalCalc.js'

describe('computeEval', () => {
  it('even position (0) gives 50% and "0.00"', () => {
    const result = computeEval(0, 'black')
    expect(result.blackPct).toBe(50)
    expect(result.displayValue).toBe('0.00')
  })

  it('black winning (+1) as black gives 100%', () => {
    const result = computeEval(1, 'black')
    expect(result.blackPct).toBe(100)
    expect(result.displayValue).toBe('+1.00')
  })

  it('white winning (-1) as black gives 0%', () => {
    const result = computeEval(-1, 'black')
    expect(result.blackPct).toBe(0)
    expect(result.displayValue).toBe('-1.00')
  })

  it('flips sign when currentPlayer is white', () => {
    const result = computeEval(0.5, 'white')
    expect(result.blackPct).toBe(25)
    expect(result.displayValue).toBe('-0.50')
  })

  it('clamps values above 1', () => {
    const result = computeEval(2.5, 'black')
    expect(result.blackPct).toBe(100)
    expect(result.displayValue).toBe('+1.00')
  })

  it('clamps values below -1', () => {
    const result = computeEval(-3, 'black')
    expect(result.blackPct).toBe(0)
    expect(result.displayValue).toBe('-1.00')
  })

  it('near-zero values display as 0.00', () => {
    const result = computeEval(0.004, 'black')
    expect(result.displayValue).toBe('0.00')
  })

  it('just above threshold shows signed value', () => {
    const result = computeEval(0.006, 'black')
    expect(result.displayValue).toBe('+0.01')
  })
})
