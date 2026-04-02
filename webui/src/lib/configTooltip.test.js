import { describe, it, expect } from 'vitest'
import { buildConfigTooltip } from './configTooltip.js'

describe('buildConfigTooltip', () => {
  it('returns empty string for null configJson', () => {
    expect(buildConfigTooltip(null, 'ResNet')).toBe('')
  })

  it('returns empty string for undefined configJson', () => {
    expect(buildConfigTooltip(undefined, 'ResNet')).toBe('')
  })

  it('parses valid JSON string with training and model sections', () => {
    const cfg = JSON.stringify({
      training: { algorithm: 'PPO', num_games: 1000 },
      model: { architecture: 'TransformerV2' }
    })
    const result = buildConfigTooltip(cfg, 'ResNet')
    expect(result).toBe(
      'Architecture: ResNet\nAlgorithm: PPO\nGames: 1000\nArchitecture: TransformerV2'
    )
  })

  it('parses valid JSON string with only training section', () => {
    const cfg = JSON.stringify({
      training: { algorithm: 'A2C', num_games: 500 }
    })
    const result = buildConfigTooltip(cfg, 'ResNet')
    expect(result).toBe('Architecture: ResNet\nAlgorithm: A2C\nGames: 500')
  })

  it('parses valid JSON string with only model section', () => {
    const cfg = JSON.stringify({
      model: { architecture: 'CNN' }
    })
    const result = buildConfigTooltip(cfg, 'ResNet')
    expect(result).toBe('Architecture: ResNet\nArchitecture: CNN')
  })

  it('handles already-parsed object (not a string)', () => {
    const cfg = {
      training: { algorithm: 'DQN', num_games: 200 },
      model: { architecture: 'MLP' }
    }
    const result = buildConfigTooltip(cfg, 'ResNet')
    expect(result).toBe(
      'Architecture: ResNet\nAlgorithm: DQN\nGames: 200\nArchitecture: MLP'
    )
  })

  it('falls back to modelArch for invalid JSON string', () => {
    expect(buildConfigTooltip('not valid json', 'ResNet')).toBe('ResNet')
  })

  it('returns just Architecture line for empty object', () => {
    expect(buildConfigTooltip({}, 'ResNet')).toBe('Architecture: ResNet')
  })

  it("defaults algorithm and num_games to '?' when missing", () => {
    const cfg = { training: {} }
    const result = buildConfigTooltip(cfg, 'ResNet')
    expect(result).toBe('Architecture: ResNet\nAlgorithm: ?\nGames: ?')
  })
})
