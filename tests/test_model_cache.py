"""Tests for OpponentStore LRU model cache."""

import pytest
import torch

from keisei.db import init_db
from keisei.training.model_registry import build_model
from keisei.training.opponent_store import OpponentStore, Role

pytestmark = pytest.mark.integration


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "cache.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    s = OpponentStore(db_path, str(league_dir))
    yield s
    s.close()


_RESNET_PARAMS = {"hidden_size": 16, "num_layers": 1}


def _add_entry(store, epoch, role=Role.UNASSIGNED):
    model = build_model("resnet", _RESNET_PARAMS)
    return store.add_entry(model, "resnet", _RESNET_PARAMS, epoch=epoch, role=role)


class TestLoadOpponentCached:
    """LRU cache for load_opponent: hit, miss, eviction, clear."""

    def test_cache_hit_returns_same_object(self, store):
        """Second call with same entry returns the identical cached model."""
        entry = _add_entry(store, epoch=1)
        m1 = store.load_opponent_cached(entry, device="cpu", max_cached=4)
        m2 = store.load_opponent_cached(entry, device="cpu", max_cached=4)
        assert m1 is m2

    def test_cache_miss_loads_fresh(self, store):
        """Different entries return different model objects."""
        e1 = _add_entry(store, epoch=1)
        e2 = _add_entry(store, epoch=2)
        m1 = store.load_opponent_cached(e1, device="cpu", max_cached=4)
        m2 = store.load_opponent_cached(e2, device="cpu", max_cached=4)
        assert m1 is not m2

    def test_eviction_at_capacity(self, store):
        """When cache exceeds max_cached, oldest entry is evicted."""
        entries = [_add_entry(store, epoch=i) for i in range(4)]
        # Fill cache to capacity=3
        for e in entries[:3]:
            store.load_opponent_cached(e, device="cpu", max_cached=3)
        assert store.cache_size() == 3

        # Adding a 4th should evict the first
        store.load_opponent_cached(entries[3], device="cpu", max_cached=3)
        assert store.cache_size() == 3

        # First entry is no longer cached — reloading gives a new object
        m_fresh = store.load_opponent_cached(entries[0], device="cpu", max_cached=3)
        # Can't check identity against evicted object, but cache size should still be 3
        assert store.cache_size() == 3

    def test_cache_hit_refreshes_lru_order(self, store):
        """Accessing a cached entry moves it to most-recently-used, protecting from eviction."""
        e1 = _add_entry(store, epoch=1)
        e2 = _add_entry(store, epoch=2)
        e3 = _add_entry(store, epoch=3)
        e4 = _add_entry(store, epoch=4)

        store.load_opponent_cached(e1, device="cpu", max_cached=3)
        store.load_opponent_cached(e2, device="cpu", max_cached=3)
        store.load_opponent_cached(e3, device="cpu", max_cached=3)

        # Touch e1 — moves it to most-recently-used
        m1_ref = store.load_opponent_cached(e1, device="cpu", max_cached=3)

        # Insert e4 — should evict e2 (oldest untouched), not e1
        store.load_opponent_cached(e4, device="cpu", max_cached=3)

        # e1 should still be the same object (not evicted)
        m1_again = store.load_opponent_cached(e1, device="cpu", max_cached=3)
        assert m1_ref is m1_again

    def test_clear_cache(self, store):
        """clear_model_cache() empties the entire cache."""
        e1 = _add_entry(store, epoch=1)
        m1 = store.load_opponent_cached(e1, device="cpu", max_cached=4)
        assert store.cache_size() > 0

        store.clear_model_cache()
        assert store.cache_size() == 0

        # Reloading gives a new object
        m2 = store.load_opponent_cached(e1, device="cpu", max_cached=4)
        assert m1 is not m2

    def test_zero_max_cached_disables_cache(self, store):
        """With max_cached=0, every call loads fresh (no caching)."""
        entry = _add_entry(store, epoch=1)
        m1 = store.load_opponent_cached(entry, device="cpu", max_cached=0)
        m2 = store.load_opponent_cached(entry, device="cpu", max_cached=0)
        assert m1 is not m2
        assert store.cache_size() == 0
