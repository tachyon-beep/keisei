# SLDataset mmap Cache Hardening

**Date**: 2026-04-03
**Issue**: keisei-ed9dbc77d7
**Status**: Design approved
**Reviewers**: Architecture critic, systems thinker, Python engineer,
quality engineer, PyTorch engineer

## Problem

`SLDataset._mmap_cache` has two defects:

1. **Fork safety**: When `DataLoader` uses `num_workers > 0`, it forks worker
   processes. Any `np.memmap` objects opened before the fork would be inherited.
   For read-only mmaps (`mode="r"`) this is technically safe on Linux (the OS
   page cache is shared), but a future change to writable mmaps or pre-loading
   in `__init__` would silently introduce cross-process corruption. There is
   currently no guard.

2. **Unbounded growth**: Every shard accessed is cached forever. With typical
   usage (10–100 shards) this is harmless, but with large shard counts the
   dict grows without limit.

## Design

### LRU-bounded mmap cache

Replace `dict` with `OrderedDict` to get O(1) LRU eviction.

**`SLDataset.__init__`** gains a `max_cache_size` parameter (default 16):

```python
from collections import OrderedDict

def __init__(self, data_dir: Path, max_cache_size: int = 16) -> None:
    if max_cache_size < 1:
        raise ValueError(f"max_cache_size must be >= 1, got {max_cache_size}")
    ...
    self._mmap_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
    self._max_cache_size = max_cache_size
```

Emit a warning when the shard count exceeds the cache size, so operators
know to increase it for large datasets:

```python
if len(self.shards) > max_cache_size:
    logger.warning(
        "SLDataset has %d shards but max_cache_size=%d; "
        "consider increasing max_cache_size to reduce mmap re-opens",
        len(self.shards), max_cache_size,
    )
```

**`_get_mmap`** becomes LRU-aware:

```python
def _get_mmap(self, path: Path) -> np.ndarray:
    if path in self._mmap_cache:
        self._mmap_cache.move_to_end(path)
        return self._mmap_cache[path]
    mmap = np.memmap(path, dtype=np.uint8, mode="r")
    self._mmap_cache[path] = mmap
    if len(self._mmap_cache) > self._max_cache_size:
        self._mmap_cache.popitem(last=False)  # evict LRU
    return mmap
```

Evicted `np.memmap` objects are dereferenced; CPython's reference counting
reclaims the OS mapping immediately. This relies on `__getitem__` calling
`.copy()` on all `np.frombuffer` views before returning — no caller retains
a live view into the mmap after `__getitem__` completes. If `.copy()` is
ever removed, eviction could invalidate in-flight buffers.

**`clear_cache`** method for explicit cleanup:

```python
def clear_cache(self) -> None:
    """Drop all cached mmap objects.

    Safe to call from worker_init_fn. Not thread-safe — intended for
    single-threaded access per process (the DataLoader worker model).
    """
    self._mmap_cache.clear()
```

### Fork-safe worker init

**`trainer.py`** passes a `worker_init_fn` to `DataLoader` that clears
inherited mmap state. The function walks the `.dataset` chain to handle
wrapper datasets (e.g., `Subset`):

```python
from torch.utils.data import get_worker_info

def _sl_worker_init(worker_id: int) -> None:
    """Clear inherited mmap cache after fork so each worker opens its own."""
    info = get_worker_info()
    if info is None:
        return  # called from main process (e.g., in tests)
    dataset = info.dataset
    # Walk wrapper chain (e.g., Subset wraps .dataset)
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    if hasattr(dataset, "clear_cache"):
        dataset.clear_cache()

self.dataloader = DataLoader(
    self.dataset,
    ...
    worker_init_fn=_sl_worker_init if config.num_workers > 0 and has_data else None,
)
```

Note: `worker_init_fn` is only wired when `has_data` is true, since
`num_workers` is already forced to 0 for empty datasets. `ConcatDataset`
is not handled (it uses `.datasets` plural); if needed in the future,
extend the traversal.

This is defensive — the cache is currently empty at fork time. But it
guarantees correctness even if future code adds shard pre-loading or
wraps `SLDataset` in `Subset` for train/val splits.

### `persistent_workers` interaction

With `persistent_workers=True` (already set in `trainer.py`),
`worker_init_fn` fires exactly once per worker at spawn time. Each
worker's LRU cache then accumulates across epochs — this is correct
and intentional (warm cache). There is no need to clear at epoch
boundaries; the LRU bound governs memory.

### Default cache size rationale

16 shards covers typical training runs (5–50 shards) without eviction
churn. With shuffled DataLoader access, LRU has no temporal locality
advantage — when `num_shards >> max_cache_size`, the effective miss rate
approaches `1 - max_cache_size/num_shards`. In practice, the OS page
cache absorbs most of the re-open cost (pages stay resident after
`munmap`), so this is a syscall overhead concern, not an I/O concern.
For large shard counts, callers should set `max_cache_size >= num_shards`.

## Files changed

| File | Change |
|---|---|
| `keisei/sl/dataset.py` | `OrderedDict` LRU cache, `max_cache_size` param, validation, warning, `clear_cache()` |
| `keisei/sl/trainer.py` | `worker_init_fn=_sl_worker_init` with wrapper-dataset traversal |
| `tests/test_sl_pipeline.py` | New tests (see below) |

## Tests

1. **LRU eviction**: Create dataset with `max_cache_size=2`, access 3 different
   shards via `_get_mmap`. Assert cache size is 2 AND the first shard's path
   is no longer in `_mmap_cache` (verifies eviction target, not just size).
   Also test re-access promotion: access shards A, B, re-access A, then
   access C — assert B was evicted (not A, since A was promoted).

2. **`clear_cache`**: Call `_get_mmap` to populate cache, call `clear_cache()`,
   assert cache is empty. Then call `__getitem__` and verify it succeeds
   (re-opens mmaps cleanly) and returns correct tensor values.

3. **`max_cache_size` validation**: Assert `SLDataset(path, max_cache_size=0)`
   raises `ValueError`.

4. **`worker_init_fn` wired**: Construct `SLTrainer` with `num_workers=2` and
   a data directory containing at least one shard (required — `has_data=False`
   suppresses `worker_init_fn`). Assert `dataloader.worker_init_fn is not None`.
   Also construct with `num_workers=0` and assert `worker_init_fn is None`.

## Out of scope

- Shard pre-loading / prefetching — not needed at current scale.
- Shared-memory cache across workers — adds complexity for minimal gain with
  read-only mmaps (the OS page cache already handles this).
- Structured dtype for `_get_mmap` (use `_SHARD_DTYPE` instead of `uint8`) —
  pre-existing design, worth a separate cleanup.
- `spawn` multiprocessing context — Linux-only project, `fork` is default.
