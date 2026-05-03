// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// Mock WebSocket before importing ws.js
let mockInstances = []

class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3

  constructor(url) {
    this.url = url
    this.readyState = MockWebSocket.CONNECTING
    this.onopen = null
    this.onclose = null
    this.onerror = null
    this.onmessage = null
    this.closeCalled = false
    mockInstances.push(this)
  }

  close() {
    this.closeCalled = true
    this.readyState = MockWebSocket.CLOSED
  }

  // Simulate the server accepting the connection
  _simulateOpen() {
    this.readyState = MockWebSocket.OPEN
    if (this.onopen) this.onopen(new Event('open'))
  }

  _simulateClose() {
    this.readyState = MockWebSocket.CLOSED
    if (this.onclose) this.onclose(new Event('close'))
  }

  _simulateError() {
    if (this.onerror) this.onerror(new Event('error'))
  }
}

// Install mock on globalThis before module import
vi.stubGlobal('WebSocket', MockWebSocket)

// Also need location for WS_URL construction
vi.stubGlobal('location', { protocol: 'http:', host: 'localhost:8001' })

// Use dynamic import so the module picks up mocked globals
let connect, disconnect, connectionState

beforeEach(async () => {
  vi.useFakeTimers()
  mockInstances = []

  // Reset the module to clear singleton state (ws, reconnectAttempt, reconnectTimer)
  vi.resetModules()
  const mod = await import('./ws.js')
  connect = mod.connect
  disconnect = mod.disconnect
  connectionState = mod.connectionState
})

afterEach(() => {
  // Ensure cleanup
  disconnect()
  vi.useRealTimers()
})

describe('connect', () => {
  it('creates a WebSocket with correct URL (ws: for http:)', () => {
    connect()
    expect(mockInstances).toHaveLength(1)
    expect(mockInstances[0].url).toBe('ws://localhost:8001/ws')
  })

  it('does not create a second WebSocket if already connecting', () => {
    connect()
    expect(mockInstances).toHaveLength(1)
    // readyState is CONNECTING (0), which is <= OPEN (1)
    connect()
    expect(mockInstances).toHaveLength(1)
  })

  it('does not create a second WebSocket if already open', () => {
    connect()
    mockInstances[0]._simulateOpen()
    connect()
    expect(mockInstances).toHaveLength(1)
  })

  it('creates a new WebSocket after previous was closed', () => {
    connect()
    const first = mockInstances[0]
    first.readyState = MockWebSocket.CLOSED
    // Now ws.readyState > OPEN, so guard allows new connection
    connect()
    expect(mockInstances).toHaveLength(2)
  })

  it('resets reconnectAttempt on successful open', () => {
    connect()
    mockInstances[0]._simulateOpen()
    // Trigger a close to start reconnecting
    mockInstances[0]._simulateClose()
    // First reconnect fires
    vi.advanceTimersByTime(2000)
    expect(mockInstances).toHaveLength(2)
    // Open the new connection — reconnectAttempt resets
    mockInstances[1]._simulateOpen()
    // Close again — next delay should be base delay (attempt 0), not escalated
    mockInstances[1]._simulateClose()
    // Base delay is 1000ms * (0.5 to 1.5), so within 1500ms it should reconnect
    vi.advanceTimersByTime(1500)
    expect(mockInstances).toHaveLength(3)
  })
})

describe('disconnect', () => {
  it('closes the WebSocket and nulls it', () => {
    connect()
    const ws = mockInstances[0]
    disconnect()
    expect(ws.closeCalled).toBe(true)
  })

  it('is safe to call when not connected', () => {
    expect(() => disconnect()).not.toThrow()
  })

  it('clears a pending reconnect timer', () => {
    connect()
    mockInstances[0]._simulateClose() // schedules reconnect
    disconnect()
    // Advance past any possible reconnect delay
    vi.advanceTimersByTime(60000)
    // No new WebSocket should have been created (only the original one)
    expect(mockInstances).toHaveLength(1)
  })
})

describe('scheduleReconnect (via onclose)', () => {
  it('reconnects after disconnect with base delay', () => {
    connect()
    mockInstances[0]._simulateOpen()
    mockInstances[0]._simulateClose()

    // Should not reconnect immediately
    expect(mockInstances).toHaveLength(1)

    // Base delay = 1000ms * 2^0 * jitter(0.5-1.5) = 500-1500ms
    vi.advanceTimersByTime(1500)
    expect(mockInstances).toHaveLength(2)
  })

  it('uses exponential backoff on repeated disconnects', () => {
    // Seed Math.random to get predictable jitter (factor = 0.5 + 0.5 = 1.0)
    vi.spyOn(Math, 'random').mockReturnValue(0.5)

    connect()
    mockInstances[0]._simulateOpen()
    mockInstances[0]._simulateClose()

    // Attempt 0: base = min(1000 * 2^0, 30000) = 1000, delay = 1000 * 1.0 = 1000ms
    vi.advanceTimersByTime(999)
    expect(mockInstances).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(mockInstances).toHaveLength(2)

    // Simulate immediate close for next attempt
    mockInstances[1]._simulateClose()

    // Attempt 1: base = min(1000 * 2^1, 30000) = 2000, delay = 2000 * 1.0 = 2000ms
    vi.advanceTimersByTime(1999)
    expect(mockInstances).toHaveLength(2)
    vi.advanceTimersByTime(1)
    expect(mockInstances).toHaveLength(3)

    // Simulate immediate close for next attempt
    mockInstances[2]._simulateClose()

    // Attempt 2: base = min(1000 * 2^2, 30000) = 4000, delay = 4000 * 1.0 = 4000ms
    vi.advanceTimersByTime(3999)
    expect(mockInstances).toHaveLength(3)
    vi.advanceTimersByTime(1)
    expect(mockInstances).toHaveLength(4)

    Math.random.mockRestore()
  })

  it('caps backoff at 30 seconds', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.5) // jitter factor = 1.0

    connect()
    mockInstances[0]._simulateOpen()
    mockInstances[0]._simulateClose()

    // Fast-forward through many reconnect attempts to hit the cap
    for (let i = 0; i < 10; i++) {
      vi.advanceTimersByTime(30000)
      const latest = mockInstances[mockInstances.length - 1]
      latest._simulateClose()
    }

    // After many attempts, base = min(1000 * 2^10, 30000) = 30000
    // delay = 30000 * 1.0 = 30000ms — should not exceed this
    const countBefore = mockInstances.length
    vi.advanceTimersByTime(29999)
    expect(mockInstances).toHaveLength(countBefore)
    vi.advanceTimersByTime(1)
    expect(mockInstances).toHaveLength(countBefore + 1)

    Math.random.mockRestore()
  })

  it('applies jitter between 50% and 150% of base delay', () => {
    // With random() = 0, jitter = 0.5, delay = base * 0.5
    vi.spyOn(Math, 'random').mockReturnValue(0)
    connect()
    mockInstances[0]._simulateOpen()
    mockInstances[0]._simulateClose()

    // Attempt 0: base = 1000, delay = 1000 * 0.5 = 500ms
    vi.advanceTimersByTime(499)
    expect(mockInstances).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(mockInstances).toHaveLength(2)

    Math.random.mockRestore()
  })
})

describe('connectionState debounce on close', () => {
  function readState() {
    let value
    const unsubscribe = connectionState.subscribe(v => { value = v })
    unsubscribe()
    return value
  }

  it('does not flip to reconnecting immediately on close', () => {
    connect()
    mockInstances[0]._simulateOpen()
    expect(readState()).toBe('connected')

    mockInstances[0]._simulateClose()
    // Visible state stays 'connected' inside the grace window.
    expect(readState()).toBe('connected')
  })

  it('flips to reconnecting once grace window elapses without recovery', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.5) // jitter = 1.0 → 1000ms

    connect()
    mockInstances[0]._simulateOpen()
    mockInstances[0]._simulateClose()

    // Reconnect attempt fires at ~1000ms, but never opens — grace fires at 3000ms.
    vi.advanceTimersByTime(2999)
    expect(readState()).toBe('connected')
    vi.advanceTimersByTime(1)
    expect(readState()).toBe('reconnecting')

    Math.random.mockRestore()
  })

  it('hides the flicker when reconnect succeeds within grace window', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.5) // jitter = 1.0 → 1000ms

    connect()
    mockInstances[0]._simulateOpen()
    mockInstances[0]._simulateClose()

    // Advance to the first reconnect attempt and let it open.
    vi.advanceTimersByTime(1000)
    expect(mockInstances).toHaveLength(2)
    mockInstances[1]._simulateOpen()

    // Cross the original grace deadline — state should have stayed 'connected'.
    vi.advanceTimersByTime(5000)
    expect(readState()).toBe('connected')

    Math.random.mockRestore()
  })

  it('does not leave a stale grace timer when a second close arrives', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.5)

    connect()
    mockInstances[0]._simulateOpen()
    mockInstances[0]._simulateClose()
    // Second close before the first grace fires (e.g. retry that failed instantly).
    vi.advanceTimersByTime(500)
    mockInstances[0]._simulateClose()

    // Grace was armed at the first close; flips to reconnecting at 3000ms total,
    // not 3000ms after the second close.
    vi.advanceTimersByTime(2499)
    expect(readState()).toBe('connected')
    vi.advanceTimersByTime(1)
    expect(readState()).toBe('reconnecting')

    Math.random.mockRestore()
  })

  it('disconnect() cancels the pending grace timer', () => {
    connect()
    mockInstances[0]._simulateOpen()
    mockInstances[0]._simulateClose()
    expect(readState()).toBe('connected')

    disconnect()
    vi.advanceTimersByTime(10000)
    // Grace timer was cleared, so state never flipped to reconnecting.
    expect(readState()).toBe('connected')
  })
})

describe('onerror', () => {
  it('closes the WebSocket on error', () => {
    connect()
    const ws = mockInstances[0]
    ws._simulateError()
    expect(ws.closeCalled).toBe(true)
  })
})

describe('WS_URL construction', () => {
  it('uses wss: for https: protocol', async () => {
    vi.stubGlobal('location', { protocol: 'https:', host: 'example.com' })
    vi.resetModules()
    mockInstances = []
    const mod = await import('./ws.js')
    mod.connect()
    expect(mockInstances[0].url).toBe('wss://example.com/ws')
    mod.disconnect()
    // Restore for other tests
    vi.stubGlobal('location', { protocol: 'http:', host: 'localhost:8001' })
  })
})
