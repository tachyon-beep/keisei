# SLDataset mmap Cache Hardening

**Date**: 2026-04-03
**Issue**: keisei-ed9dbc77d7
**Status**: Design approved

## Problem

`SLDataset._mmap_cache` has two defects:

1. **Fork safety**: When `DataLoader` uses `num_workers > 0`, it forks worker
   processes. Any `np.memmap` objects opened before the fork would share file
   descriptors across processes. Currently the cache is empty at fork time
   (mmaps are lazy), but there is no guard — a future change that pre-loads
   shards in `__init__` would silently introduce cross-process fd sharing.

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
    ...
    self._mmap_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
    self._max_cache_size = max_cache_size
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

Evicted `np.memmap` objects are dereferenced; the OS reclaims the mapping when
the refcount drops to zero.

**`clear_cache`** method for explicit cleanup:

```python
def clear_cache(self) -> None:
    """Drop all cached mmap objects. Safe to call from worker_init_fn."""
    self._mmap_cache.clear()
```

### Fork-safe worker init

**`trainer.py`** passes a `worker_init_fn` to `DataLoader` that clears
inherited mmap state:

```python
def _sl_worker_init(worker_id: int) -> None:
    """Clear inherited mmap cache after fork so each worker opens its own."""
    import torch.utils.data as data_utils
    dataset = data_utils.get_worker_info().dataset
    dataset.clear_cache()

self.dataloader = DataLoader(
    self.dataset,
    ...
    worker_init_fn=_sl_worker_init if config.num_workers > 0 else None,
)
```

This is defensive — the cache is currently empty at fork time. But it
guarantees correctness even if future code adds shard pre-loading.

### Default cache size rationale

16 shards covers typical training runs (5–50 shards) without eviction churn.
If a dataset has 100+ shards, the LRU naturally keeps the working set warm
while bounding memory. The caller can override via `max_cache_size`.

## Files changed

| File | Change |
|---|---|
| `keisei/sl/dataset.py` | `OrderedDict` LRU cache, `max_cache_size` param, `clear_cache()` |
| `keisei/sl/trainer.py` | `worker_init_fn=_sl_worker_init` in DataLoader |
| `tests/test_sl_pipeline.py` | New tests (see below) |

## Tests

1. **LRU eviction**: Create dataset with `max_cache_size=2`, access 3 different
   shards, verify cache size stays at 2 and the first shard was evicted.

2. **`clear_cache`**: Open some mmaps, call `clear_cache()`, verify cache is
   empty and subsequent access still works (re-opens cleanly).

3. **`worker_init_fn` wired**: Construct `SLTrainer` with `num_workers=2`,
   verify `dataloader.worker_init_fn` is not None. (Functional fork test is
   skipped in CI — too heavy for unit tests.)

## Out of scope

- Shard pre-loading / prefetching — not needed at current scale.
- Shared-memory cache across workers — adds complexity for minimal gain with
  read-only mmaps (the OS page cache already handles this).
