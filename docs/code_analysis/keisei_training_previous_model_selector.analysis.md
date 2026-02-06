# Code Analysis: `keisei/training/previous_model_selector.py`

## 1. Purpose & Role

This module provides a bounded pool of previously saved model checkpoint paths, intended for use in evaluation against historical versions of the agent. It maintains a FIFO queue with a configurable maximum size, supporting random selection of a past checkpoint for evaluation matches. Currently, it is referenced only by tests (`tests/evaluation/test_previous_model_selector.py`) and does not appear to be integrated into the active training pipeline.

## 2. Interface Contracts

### Imports
- `random` -- for random checkpoint selection.
- `collections.deque` -- bounded FIFO container.
- `pathlib.Path` -- filesystem path abstraction.

### `PreviousModelSelector` (lines 9-28)
- **Constructor (line 12)**: `pool_size` (int, default 5). Creates a `deque` with `maxlen=pool_size`.
- **`add_checkpoint` (line 16)**: Takes `path: Path | str`, converts to `Path`, appends to deque. When the deque is at capacity, the oldest entry is automatically evicted.
- **`get_random_checkpoint` (line 20)**: Returns a random `Path` from the pool, or `None` if the pool is empty.
- **`get_all` (line 26)**: Returns a list copy of all stored checkpoints in insertion order.

## 3. Correctness Analysis

- **Deque with maxlen (line 14)**: `deque(maxlen=pool_size)` correctly enforces the pool size limit. Appending beyond capacity evicts the leftmost (oldest) element. This is correct FIFO behavior.
- **Path coercion (line 18)**: `Path(path)` handles both `str` and `Path` inputs. If `path` is already a `Path`, `Path(Path(...))` returns an equivalent `Path`. Correct.
- **Random selection (line 24)**: `random.choice(list(self.checkpoints))` converts the deque to a list for random selection. This is necessary because `random.choice` requires a sequence (deque supports it, but converting to list is explicit and safe). The `if not self.checkpoints` guard on line 22 prevents `IndexError` from `random.choice` on an empty sequence.
- **`get_all` return type (line 27)**: Returns `list(self.checkpoints)`, which is a new list (not a reference to the internal deque). The type annotation says `Iterable[Path]` (line 26), which is correct but less specific than the actual `List[Path]` return type.
- **No path validation**: `add_checkpoint` does not verify that the path exists on disk or points to a valid checkpoint file. This means stale paths (deleted files) can accumulate in the pool.

## 4. Robustness & Error Handling

- **Empty pool handling**: `get_random_checkpoint` returns `None` for an empty pool (line 23). Callers must handle the `None` case.
- **No validation of `pool_size`**: A `pool_size` of 0 would create a `deque(maxlen=0)` that silently discards all appended items, making the selector permanently empty. A negative `pool_size` would raise a `ValueError` from `deque`. Neither case is guarded.
- **No filesystem checks**: Paths are stored as-is without checking existence. A checkpoint file could be deleted between being added and being selected. This is a design choice -- the selector is a path store, not a filesystem validator.
- **Thread safety**: No locking. Concurrent `add_checkpoint` and `get_random_checkpoint` calls could lead to races. Not a concern in the current single-threaded usage.

## 5. Performance & Scalability

- **O(1) add**: Deque append is O(1).
- **O(n) random selection**: `list(self.checkpoints)` creates a copy of up to `pool_size` elements, then `random.choice` is O(1) on the list. With typical `pool_size=5`, this is negligible.
- **O(n) get_all**: Creates a list copy of up to `pool_size` elements. Negligible overhead.
- **Memory**: Stores up to `pool_size` `Path` objects. Minimal memory footprint.

## 6. Security & Safety

- No file I/O (only stores paths, does not read/write files).
- No network access or dynamic code execution.
- Path objects are constructed from caller-provided strings but are not used to access the filesystem within this module.

## 7. Maintainability

- **28 lines, 1 class**: Extremely concise and single-purpose.
- **Docstrings**: Class-level docstring (line 10) and per-method docstrings (lines 17, 21, 27).
- **Type annotations**: Full annotations on all methods, including `Path | str` union type on `add_checkpoint` (line 16, uses Python 3.10+ union syntax).
- **Python version compatibility**: The `Path | str` union syntax on line 16 requires Python 3.10+. The `from __future__ import annotations` on line 1 enables this syntax for annotations in Python 3.7+, so it is compatible with the project's Python 3.13 requirement.
- **Not integrated**: As of the current codebase, this class is not used outside of its dedicated test file. It appears to be infrastructure prepared for a planned evaluation-against-historical-models feature.

## 8. Verdict

**SOUND**

The class is a correct, minimal implementation of a bounded checkpoint pool with random selection. There are no bugs. The lack of integration into the training pipeline means it is currently dead code in production, but it is well-tested via its dedicated test suite. The only concern is the missing validation on `pool_size=0` (silent no-op behavior), which is a minor edge case.
