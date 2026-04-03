# SLDataset mmap Cache Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `SLDataset`'s mmap cache fork-safe and bounded so it is production-ready before first real training run.

**Architecture:** Replace the unbounded `dict` cache with an LRU-bounded `OrderedDict`, add `clear_cache()` method, wire a `worker_init_fn` into the DataLoader that clears inherited mmap state after fork. Add input validation and an operability warning.

**Tech Stack:** Python 3.13, numpy (`np.memmap`), PyTorch (`DataLoader`, `get_worker_info`), pytest

**Spec:** `docs/superpowers/specs/2026-04-03-sl-dataset-mmap-hardening-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `keisei/sl/dataset.py` | Modify | `OrderedDict` LRU cache, `max_cache_size` param + validation, warning, `clear_cache()` |
| `keisei/sl/trainer.py` | Modify | `_sl_worker_init` function, wire into `DataLoader` |
| `tests/test_sl_pipeline.py` | Modify | 4 new test methods in a new `TestSLDatasetMmapCache` class |

---

### Task 1: LRU cache and `max_cache_size` validation

**Files:**
- Modify: `keisei/sl/dataset.py:1-10` (imports)
- Modify: `keisei/sl/dataset.py:66-93` (`SLDataset.__init__`, `_get_mmap`)
- Test: `tests/test_sl_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add a new test class at the end of `tests/test_sl_pipeline.py`:

```python
class TestSLDatasetMmapCache:
    """Tests for LRU-bounded mmap cache (keisei-ed9dbc77d7)."""

    def _write_shards(self, tmp_path, n_shards=3, n_positions=5):
        """Helper: write n_shards shard files with n_positions each."""
        rng = np.random.default_rng(42)
        for i in range(n_shards):
            write_shard(
                tmp_path / f"shard_{i:03d}.bin",
                rng.standard_normal((n_positions, 50 * 81)).astype(np.float32),
                rng.integers(0, 11259, size=n_positions).astype(np.int64),
                rng.integers(0, 3, size=n_positions).astype(np.int64),
                rng.standard_normal(n_positions).astype(np.float32),
            )

    def test_max_cache_size_zero_raises(self, tmp_path):
        """max_cache_size < 1 must raise ValueError."""
        self._write_shards(tmp_path, n_shards=1)
        with pytest.raises(ValueError, match="max_cache_size must be >= 1"):
            SLDataset(tmp_path, max_cache_size=0)

    def test_lru_eviction_bounds_cache_size(self, tmp_path):
        """With max_cache_size=2 and 3 shards, cache never exceeds 2 entries."""
        self._write_shards(tmp_path, n_shards=3)
        dataset = SLDataset(tmp_path, max_cache_size=2)

        # Access items from shard 0, 1, 2 (5 positions each)
        _ = dataset[0]   # shard 0
        _ = dataset[5]   # shard 1
        assert len(dataset._mmap_cache) == 2

        _ = dataset[10]  # shard 2 — should evict shard 0
        assert len(dataset._mmap_cache) == 2

        shard_0_path = dataset.shards[0][0]
        assert shard_0_path not in dataset._mmap_cache, "LRU should have evicted shard 0"

    def test_lru_promotion_on_reaccess(self, tmp_path):
        """Re-accessing a shard promotes it; the non-promoted shard is evicted."""
        self._write_shards(tmp_path, n_shards=3)
        dataset = SLDataset(tmp_path, max_cache_size=2)

        _ = dataset[0]   # shard 0 (LRU)
        _ = dataset[5]   # shard 1 (MRU)
        _ = dataset[0]   # re-access shard 0 — promotes to MRU

        _ = dataset[10]  # shard 2 — should evict shard 1 (now LRU), not shard 0
        shard_0_path = dataset.shards[0][0]
        shard_1_path = dataset.shards[1][0]
        assert shard_0_path in dataset._mmap_cache, "Shard 0 was promoted, should survive"
        assert shard_1_path not in dataset._mmap_cache, "Shard 1 was LRU, should be evicted"

    def test_lru_single_slot_boundary(self, tmp_path):
        """max_cache_size=1: every new shard evicts the previous one."""
        self._write_shards(tmp_path, n_shards=2)
        dataset = SLDataset(tmp_path, max_cache_size=1)

        _ = dataset[0]  # shard 0
        assert len(dataset._mmap_cache) == 1

        _ = dataset[5]  # shard 1 — should evict shard 0
        assert len(dataset._mmap_cache) == 1
        shard_0_path = dataset.shards[0][0]
        assert shard_0_path not in dataset._mmap_cache

    def test_warning_when_shards_exceed_cache_size(self, tmp_path, caplog):
        """A warning should be logged when num_shards > max_cache_size."""
        self._write_shards(tmp_path, n_shards=5)
        with caplog.at_level(logging.WARNING, logger="keisei.sl.dataset"):
            SLDataset(tmp_path, max_cache_size=2)
        assert any("5 shards but max_cache_size=2" in msg for msg in caplog.messages)
```

Note: the test file needs `import logging` at the top.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSLDatasetMmapCache -v`
Expected: FAIL — `SLDataset()` does not accept `max_cache_size` parameter

- [ ] **Step 3: Implement LRU cache in dataset.py**

In `keisei/sl/dataset.py`, add `OrderedDict` import and `logging`:

```python
import logging
from collections import OrderedDict
```

Update `__init__` signature and body:

```python
class SLDataset(Dataset):
    """Memory-mapped dataset reading from binary shard files."""

    def __init__(self, data_dir: Path, max_cache_size: int = 16) -> None:
        if max_cache_size < 1:
            raise ValueError(f"max_cache_size must be >= 1, got {max_cache_size}")

        self.data_dir = data_dir
        self.shards: list[tuple[Path, int]] = []  # (path, num_positions)
        self._cumulative: list[int] = []

        shard_files = sorted(data_dir.glob("shard_*.bin"))
        total = 0
        for shard_path in shard_files:
            file_size = shard_path.stat().st_size
            n_positions = file_size // RECORD_SIZE
            if n_positions > 0:
                self.shards.append((shard_path, n_positions))
                total += n_positions
                self._cumulative.append(total)

        self._total = total
        self._mmap_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._max_cache_size = max_cache_size

        if len(self.shards) > max_cache_size:
            logger.warning(
                "SLDataset has %d shards but max_cache_size=%d; "
                "consider increasing max_cache_size to reduce mmap re-opens",
                len(self.shards), max_cache_size,
            )
```

Add `logger` after the imports (near line 10):

```python
logger = logging.getLogger(__name__)
```

Update `_get_mmap`:

```python
def _get_mmap(self, path: Path) -> np.ndarray:
    if path in self._mmap_cache:
        self._mmap_cache.move_to_end(path)
        return self._mmap_cache[path]
    mmap = np.memmap(path, dtype=np.uint8, mode="r")
    self._mmap_cache[path] = mmap
    if len(self._mmap_cache) > self._max_cache_size:
        # Evict LRU entry. Safe because __getitem__ calls .copy() on all
        # np.frombuffer views — no caller retains a live view into the mmap.
        self._mmap_cache.popitem(last=False)
    return mmap
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSLDatasetMmapCache -v`
Expected: 3 PASSED

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `uv run pytest tests/test_sl_pipeline.py -v`
Expected: All existing tests still pass (the `max_cache_size` parameter has a default of 16, so no callers break)

- [ ] **Step 6: Commit**

```bash
git add keisei/sl/dataset.py tests/test_sl_pipeline.py
git commit -m "feat(sl): LRU-bounded mmap cache for SLDataset

Replace unbounded dict with OrderedDict LRU cache. Add max_cache_size
parameter (default 16) with validation. Emit warning when shard count
exceeds cache size. Five new tests cover eviction, promotion, boundary,
warning, and validation."
```

---

### Task 2: `clear_cache()` method

**Files:**
- Modify: `keisei/sl/dataset.py` (add method after `_get_mmap`)
- Test: `tests/test_sl_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add to `TestSLDatasetMmapCache` in `tests/test_sl_pipeline.py`:

```python
    def test_clear_cache_empties_and_reaccess_works(self, tmp_path):
        """clear_cache() empties the cache; subsequent access re-opens cleanly."""
        self._write_shards(tmp_path, n_shards=2)
        dataset = SLDataset(tmp_path, max_cache_size=16)

        # Populate cache
        item_before = dataset[0]
        assert len(dataset._mmap_cache) == 1

        # Clear
        dataset.clear_cache()
        assert len(dataset._mmap_cache) == 0

        # Re-access same item — should work and return identical values
        item_after = dataset[0]
        assert len(dataset._mmap_cache) == 1
        assert torch.equal(item_before["observation"], item_after["observation"])
        assert item_before["policy_target"] == item_after["policy_target"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSLDatasetMmapCache::test_clear_cache_empties_and_reaccess_works -v`
Expected: FAIL — `SLDataset` has no `clear_cache` method

- [ ] **Step 3: Implement clear_cache**

Add to `SLDataset` in `keisei/sl/dataset.py`, after `_get_mmap`:

```python
def clear_cache(self) -> None:
    """Drop all cached mmap objects.

    Safe to call from worker_init_fn after fork. Not thread-safe —
    intended for single-threaded access per process (DataLoader worker model).
    """
    self._mmap_cache.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSLDatasetMmapCache -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add keisei/sl/dataset.py tests/test_sl_pipeline.py
git commit -m "feat(sl): add SLDataset.clear_cache() for fork safety"
```

---

### Task 3: Fork-safe `worker_init_fn` in trainer

**Files:**
- Modify: `keisei/sl/trainer.py:1-16` (imports), `keisei/sl/trainer.py:46-65` (`__init__`)
- Test: `tests/test_sl_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add to `TestSLDatasetMmapCache` in `tests/test_sl_pipeline.py`:

```python
    def test_worker_init_fn_wired_when_workers_nonzero(self, tmp_path):
        """SLTrainer with num_workers > 0 must set worker_init_fn."""
        from keisei.sl.trainer import SLConfig, SLTrainer
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        # Must have data — has_data=False suppresses worker_init_fn
        self._write_shards(tmp_path, n_shards=1, n_positions=10)

        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        model = SEResNetModel(params)
        config = SLConfig(data_dir=str(tmp_path), num_workers=2)
        trainer = SLTrainer(model, config)
        assert trainer.dataloader.worker_init_fn is not None

    def test_worker_init_fn_none_when_workers_zero(self, tmp_path):
        """SLTrainer with num_workers=0 must NOT set worker_init_fn."""
        from keisei.sl.trainer import SLConfig, SLTrainer
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        self._write_shards(tmp_path, n_shards=1, n_positions=10)

        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        model = SEResNetModel(params)
        config = SLConfig(data_dir=str(tmp_path), num_workers=0)
        trainer = SLTrainer(model, config)
        assert trainer.dataloader.worker_init_fn is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSLDatasetMmapCache::test_worker_init_fn_wired_when_workers_nonzero tests/test_sl_pipeline.py::TestSLDatasetMmapCache::test_worker_init_fn_none_when_workers_zero -v`
Expected: FAIL — `worker_init_fn_wired` fails because trainer currently passes no `worker_init_fn`

- [ ] **Step 3: Implement worker_init_fn in trainer.py**

Add import near the top of `keisei/sl/trainer.py` (after existing imports):

```python
from torch.utils.data import DataLoader, get_worker_info
```

(This replaces the existing `from torch.utils.data import DataLoader`.)

Add the `_sl_worker_init` function before the `SLTrainer` class:

```python
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
```

Update the `DataLoader` construction in `SLTrainer.__init__` to add `worker_init_fn`:

```python
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=has_data,  # RandomSampler rejects empty datasets
            num_workers=config.num_workers if has_data else 0,
            pin_memory=is_cuda and config.num_workers > 0 and has_data,
            persistent_workers=config.num_workers > 0 and has_data,
            worker_init_fn=_sl_worker_init if config.num_workers > 0 and has_data else None,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSLDatasetMmapCache -v`
Expected: 6 PASSED

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/test_sl_pipeline.py -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add keisei/sl/trainer.py tests/test_sl_pipeline.py
git commit -m "feat(sl): fork-safe worker_init_fn clears mmap cache after fork

Adds _sl_worker_init that walks wrapper dataset chain (Subset,
ConcatDataset) to find SLDataset and call clear_cache(). Wired into
DataLoader when num_workers > 0 and dataset has data."
```

---

### Task 4: Final integration verification

**Files:**
- None created/modified — verification only

- [ ] **Step 1: Run full project test suite**

Run: `uv run pytest`
Expected: All tests pass, no regressions

- [ ] **Step 2: Verify the warning fires**

Run a quick Python check:

```bash
uv run python -c "
import logging, tempfile, numpy as np
from pathlib import Path
from keisei.sl.dataset import SLDataset, write_shard

logging.basicConfig(level=logging.WARNING)
with tempfile.TemporaryDirectory() as d:
    p = Path(d)
    rng = np.random.default_rng(0)
    for i in range(5):
        write_shard(p / f'shard_{i:03d}.bin',
            rng.standard_normal((2, 50*81)).astype(np.float32),
            rng.integers(0, 11259, size=2).astype(np.int64),
            rng.integers(0, 3, size=2).astype(np.int64),
            rng.standard_normal(2).astype(np.float32))
    ds = SLDataset(p, max_cache_size=2)
"
```

Expected: Warning message `SLDataset has 5 shards but max_cache_size=2...`

- [ ] **Step 3: Close the issue**

```bash
filigree close keisei-ed9dbc77d7 --reason="LRU-bounded OrderedDict cache (max_cache_size=16 default), clear_cache() method, fork-safe worker_init_fn with wrapper-dataset traversal. 6 new tests. Reviewed by 5 specialists."
```
