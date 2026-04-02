/**
 * Build a tooltip string from training state config.
 * @param {object|string|null} configJson - The config_json field (string or parsed object)
 * @param {string} modelArch - The model architecture string (used as fallback and first line)
 * @returns {string}
 */
export function buildConfigTooltip(configJson, modelArch) {
  try {
    const cfg = typeof configJson === 'string'
      ? JSON.parse(configJson)
      : configJson
    if (!cfg) return ''
    const lines = []
    lines.push(`Architecture: ${modelArch}`)
    if (cfg.training) {
      lines.push(`Algorithm: ${cfg.training.algorithm || '?'}`)
      lines.push(`Games: ${cfg.training.num_games || '?'}`)
    }
    if (cfg.model) {
      lines.push(`Architecture: ${cfg.model.architecture || '?'}`)
    }
    return lines.join('\n')
  } catch { return modelArch }
}
