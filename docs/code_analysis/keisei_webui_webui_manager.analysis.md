# Code Analysis: `keisei/webui/webui_manager.py`

## 1. Purpose & Role

This module implements the `WebUIManager`, the primary WebSocket streaming server for real-time training visualization. It runs parallel to the console-based `DisplayManager`, extracting board state, training metrics, and advanced visualization data from the `Trainer` object and broadcasting them to connected WebSocket clients. It also contains a `WebSocketConnectionManager` that handles client registration, deregistration, and message broadcasting via an async queue.

## 2. Interface Contracts

### `WebSocketConnectionManager` (lines 24-75)
- **`register(websocket)`**: Adds a WebSocket connection to the tracked set.
- **`unregister(websocket)`**: Removes a WebSocket connection from the tracked set.
- **`broadcast(message)`**: Serializes a dict to JSON and sends to all connected clients. Dead connections are cleaned up.
- **`queue_message(message)`**: Puts a message into an `asyncio.Queue` for async broadcast.
- **`message_broadcaster()`**: Long-running coroutine that consumes from the queue and broadcasts.

### `WebUIManager` (lines 78-705)
- **Constructor**: `__init__(config: WebUIConfig)` -- accepts a Pydantic config object.
- **`start() -> bool`**: Starts the WebSocket server in a background thread with its own event loop. Returns `False` if websockets is not installed.
- **`stop()`**: Cancels async tasks via `call_soon_threadsafe`.
- **`update_progress(trainer, speed, pending_updates)`**: Called from the training loop to broadcast progress data. Rate-limited to 2 Hz.
- **`refresh_dashboard_panels(trainer)`**: Called from the training loop to broadcast board state and metrics. Rate-limited by `board_update_rate_hz` and `metrics_update_rate_hz`.
- **Multiple private extraction methods**: `_extract_board_state`, `_extract_metrics_data`, `_get_latest_value_estimate`, `_extract_policy_confidence`, `_extract_skill_metrics`, `_extract_gradient_norms`, `_get_buffer_quality_distribution`.

### Dependencies
- `websockets` (optional, soft dependency with graceful fallback on lines 13-19)
- `keisei.config_schema.WebUIConfig`
- Accesses `trainer.game`, `trainer.metrics_manager`, `trainer.step_manager`, `trainer.experience_buffer`, `trainer.model` at runtime.

## 3. Correctness Analysis

### Attribute `history.win_rates` Does Not Exist (line 592)
In `_calculate_tactical_strength` (line 592), the code accesses `history.win_rates` -- but the `MetricsHistory` class defines `win_rates_history`, not `win_rates`. This will always fall through to the `except Exception` handler on line 596, silently returning the fallback value of 0.5. The tactical strength metric never reflects real data.

### `recent_episodes` Attribute Does Not Exist (lines 517-522, 534-537)
Both `_get_latest_value_estimate` (line 517) and `_extract_policy_confidence` (line 534) attempt to access `trainer.step_manager.recent_episodes`. The `StepManager` class (based on grep results) does not have a `recent_episodes` attribute. These methods will always fall through to their fallback branches: `_get_latest_value_estimate` returns a random float (line 526), and `_extract_policy_confidence` returns random confidence data (line 551). These are not bugs per se (the fallbacks are intentional), but the primary data paths are dead code.

### Bare `except:` Clauses (lines 374, 376)
In the move history extraction block (lines 367-377), two bare `except:` clauses catch all exceptions silently. This suppresses any debugging information and violates Python best practices. If `format_move_with_description` or the string conversion fails, the error is invisible.

### `asyncio.Queue` Created Outside Event Loop (line 29)
The `WebSocketConnectionManager.__init__` creates an `asyncio.Queue()` at line 29 during `WebUIManager.__init__` on the main thread, before any event loop exists. In Python 3.10+, `asyncio.Queue` no longer requires an event loop at construction, so this is safe on Python 3.10+. Given the project uses Python 3.13 (per MEMORY.md), this is not a bug.

### Fallback Data Generation With `random` (lines 525-526, 550-551, 668, 691)
Multiple fallback paths generate synthetic random data using `import random` inside method bodies. When real data is unavailable (which is the common case for `_get_latest_value_estimate` and `_extract_policy_confidence` as noted above), the WebUI displays random noise. This is misleading to users who may interpret it as real training metrics.

### `max_connections` Config Ignored (line 672-673 of config_schema.py vs WebUIManager)
The `WebUIConfig` defines `max_connections: int = 10`, but `WebUIManager` never references this value. The WebSocket server accepts unlimited concurrent connections. This is a configuration contract violation.

### `update_rate_hz` Config Ignored
The `WebUIConfig` defines `update_rate_hz: float = 2.0`, but the manager uses hardcoded rate-limiting values (e.g., 0.5-second interval on line 206) and the separate `board_update_rate_hz` / `metrics_update_rate_hz` config values. The `update_rate_hz` field is dead configuration.

## 4. Robustness & Error Handling

### Graceful Degradation When websockets Not Installed (lines 13-19, 100-101, 105-107)
The module handles the missing `websockets` dependency cleanly: a warning is logged in the constructor, and `start()` returns `False`. The type alias `WebSocketServerProtocol = Any` on line 18 prevents import-time errors.

### Broad Exception Handlers Throughout
Nearly every public and private method wraps its body in a `try/except Exception` block (e.g., lines 409-411, 509-511, 527-528, 552-553, 568, 609, 636, 650, 669-670, 693). This prevents any single extraction failure from crashing the training loop, which is a sound defensive strategy for a non-critical streaming component. However, it also masks bugs (as demonstrated by the `history.win_rates` issue).

### Missing Event Loop Guard in `_queue_message_with_cleanup` (line 703)
The method calls `asyncio.run_coroutine_threadsafe(... , self.event_loop)` at line 703-705. While `update_progress` and `refresh_dashboard_panels` check `if not self._running or not self.event_loop` before calling this method, `_queue_message_with_cleanup` itself does not verify that `self.event_loop` is not closed. If `stop()` is called concurrently and the event loop closes between the check and the `run_coroutine_threadsafe` call, a `RuntimeError` would be raised. However, since `stop()` only cancels tasks and does not close the event loop, this race is unlikely in practice.

### Thread Safety of Rate Limiting Timestamps (lines 92-94, 183-197, 206-208)
The `last_board_update`, `last_metrics_update`, and `last_progress_update` timestamps are read and written from the main training thread without synchronization. Since these methods are called from a single training thread (not from multiple threads), this is not a concurrency issue in practice.

### `message_count` Not Thread-Safe (line 697)
The `message_count` integer is incremented on the main thread but could theoretically be read from the async event loop thread. In CPython this is safe due to the GIL.

## 5. Performance & Scalability

### Data Extraction Overhead
Every call to `refresh_dashboard_panels` invokes `_extract_metrics_data`, which in turn calls six additional extraction methods (`_get_latest_value_estimate`, `_extract_policy_confidence`, `_extract_skill_metrics`, `_extract_gradient_norms`, `_get_buffer_quality_distribution`). The `_extract_gradient_norms` method (lines 652-670) iterates over **all model parameters** calling `param.grad.norm()`, which is an O(N) operation over potentially millions of parameters. At the configured `metrics_update_rate_hz` of 1 Hz, this runs once per second, adding measurable overhead to the training loop.

### List Slicing and Copying
Multiple extraction methods create list copies/slices of history data: e.g., `list(history.policy_losses)[-50:]` (line 470-477) creates a full list copy then slices. For deques of size 1000, this allocates and discards memory on every call.

### Message Queue Unbounded
The `asyncio.Queue()` at line 29 has no `maxsize` argument, so the queue can grow without bound. If WebSocket broadcast is slow (e.g., slow client) but message production continues at training speed, the queue will grow indefinitely, consuming memory.

### No Connection Limit Enforcement
As noted, `max_connections` from config is never enforced. A malicious or misconfigured client could open many connections, each of which receives all broadcast messages, amplifying memory and bandwidth consumption.

## 6. Security & Safety

### Binding to All Interfaces (from config default `0.0.0.0`)
The WebSocket server binds to `0.0.0.0` by default, exposing it on all network interfaces. This is by design for the streaming use case but increases attack surface.

### No Authentication or Authorization
Any client with network access can connect to the WebSocket and receive full training telemetry, including model gradient norms, board state, and all training metrics. There is no token, key, or credential check.

### No Input Validation on WebSocket Messages
The `handle_client` coroutine (lines 153-168) does not process any incoming messages from clients -- it only sends data and waits for the connection to close. This is safe from an injection perspective since no client input is parsed.

### No Rate Limiting on Client Connections
Without `max_connections` enforcement, an attacker could exhaust server resources by opening many WebSocket connections.

### Sensitive Data Exposure
The metrics stream includes detailed model internals: gradient norms per layer (line 657-664), policy confidence distributions, and buffer reward distributions. While not secret in most contexts, this could reveal model architecture details.

### `random` Module Used for Fallback Data
The `random` module (not `secrets`) is used for generating fallback data. Since this data is cosmetic (not security-relevant), this is not a security issue.

## 7. Maintainability

### Excessive Defensive Attribute Probing (lines 278-358)
The `_extract_board_state` method contains extensive `hasattr`/`getattr` chains to probe for piece attributes (`piece_type`, `type`, `name` on lines 280-281; `color`, `player`, `owner` on lines 293-294; `promoted`, `is_promoted` on lines 306-308) and hand attributes (`sente_hand`, `black_hand`, `hands` on lines 327-348; `gote_hand`, `white_hand` on lines 350-358). This duck-typing approach suggests the code was written to be compatible with multiple possible game engine APIs without knowing which one is actually used. The Shogi game engine in this same codebase has well-defined attributes, making most of these fallback paths dead code.

### Dead Fallback Paths
As documented in the correctness section, multiple methods have primary data paths that reference non-existent attributes (`recent_episodes`, `history.win_rates`), making their primary logic dead code while the fallback paths (random data generation) are what always executes.

### Module Size and Complexity
At 705 lines, this is the largest file in the webui package and contains significant extraction logic that is tightly coupled to the `Trainer` object's internal structure. The extraction methods access deep internal state: `trainer.metrics_manager.history.win_rates_history`, `trainer.step_manager.move_log`, `trainer.experience_buffer`, `trainer.model.named_parameters()`. This creates a fragile coupling that will break if any of these internal structures change.

### Unused Import
The `signal` import from `os` path is not present in this file, but `Path` (line 10) is imported but never used in the module.

### Inconsistent Rate Limiting Approach
Board updates use `_should_update_board()` (configurable via `board_update_rate_hz`), metrics use `_should_update_metrics()` (configurable via `metrics_update_rate_hz`), and progress uses a hardcoded 0.5-second interval (line 206). The `update_rate_hz` config field goes unused entirely. This is confusing.

## 8. Verdict

**NEEDS_ATTENTION**

The module functions adequately as a non-critical streaming component, with sound graceful degradation when dependencies are missing and defensive exception handling that prevents crashes in the training loop. However, there are several real correctness issues: `history.win_rates` (line 592) references a non-existent attribute, `recent_episodes` (lines 517, 534) references a non-existent attribute on StepManager (making two data extraction paths always return random data), and `max_connections` configuration is silently ignored. The extensive duck-typing in `_extract_board_state` creates unnecessary complexity since the game engine's API is well-defined within this same codebase. The `_extract_gradient_norms` method iterating over all model parameters at 1 Hz adds measurable overhead. None of these issues crash the system (due to pervasive exception handling), but they mean the WebUI displays fabricated data in several visualization panels without any indication to the user.
