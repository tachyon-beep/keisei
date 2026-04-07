/**
 * Batch-collapse consecutive events of the same type and time into
 * a single collapsed line. e.g., 3 arrivals at the same timestamp become
 * one entry with count=3 and a names array.
 */
export function collapseEvents(events) {
  const out = []
  for (const event of events) {
    const prev = out[out.length - 1]
    if (prev && !prev.collapsed && prev.type === event.type && prev.time === event.time) {
      out[out.length - 1] = {
        collapsed: true,
        type: event.type,
        icon: event.icon,
        time: event.time,
        count: 2,
        names: [prev.name, event.name],
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
