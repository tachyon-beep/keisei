/**
 * Parse a JSON string, returning a fallback on failure.
 * If the input is not a string, returns it as-is.
 */
export function safeParse(json, fallback) {
  try {
    return typeof json === 'string' ? JSON.parse(json) : json
  } catch {
    return fallback
  }
}
