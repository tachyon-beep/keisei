"""Unit tests for keisei.training.parallel.model_sync.ModelSynchronizer."""

import time

import numpy as np
import pytest
import torch
import torch.nn as nn

from keisei.training.parallel.model_sync import ModelSynchronizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    """A tiny model used across sync tests."""
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))


@pytest.fixture
def syncer():
    """Default ModelSynchronizer with short interval."""
    return ModelSynchronizer(sync_interval=100, compression_enabled=True)


@pytest.fixture
def syncer_uncompressed():
    """ModelSynchronizer with compression disabled."""
    return ModelSynchronizer(sync_interval=100, compression_enabled=False)


# ---------------------------------------------------------------------------
# TestModelSynchronizerInit
# ---------------------------------------------------------------------------


class TestModelSynchronizerInit:
    def test_default_values(self):
        s = ModelSynchronizer()
        assert s.sync_interval == 100
        assert s.compression_enabled is True
        assert s.last_sync_step == 0
        assert s.sync_count == 0

    def test_custom_values(self):
        s = ModelSynchronizer(sync_interval=500, compression_enabled=False)
        assert s.sync_interval == 500
        assert s.compression_enabled is False


# ---------------------------------------------------------------------------
# TestShouldSync
# ---------------------------------------------------------------------------


class TestShouldSync:
    def test_returns_false_below_interval(self, syncer):
        assert syncer.should_sync(50) is False

    def test_returns_true_at_interval(self, syncer):
        assert syncer.should_sync(100) is True

    def test_returns_true_above_interval(self, syncer):
        assert syncer.should_sync(150) is True

    def test_respects_last_sync_step(self, syncer):
        syncer.mark_sync_completed(100)
        assert syncer.should_sync(150) is False
        assert syncer.should_sync(200) is True


# ---------------------------------------------------------------------------
# TestPrepareModelForSync
# ---------------------------------------------------------------------------


class TestPrepareModelForSync:
    def test_compressed_output_structure(self, syncer, small_model):
        result = syncer.prepare_model_for_sync(small_model)
        assert "model_data" in result
        assert "metadata" in result

    def test_metadata_fields(self, syncer, small_model):
        result = syncer.prepare_model_for_sync(small_model)
        meta = result["metadata"]
        assert "sync_count" in meta
        assert "timestamp" in meta
        assert "total_parameters" in meta
        assert "compressed" in meta
        assert "model_keys" in meta
        assert meta["compressed"] is True

    def test_all_weights_converted(self, syncer, small_model):
        result = syncer.prepare_model_for_sync(small_model)
        state_keys = set(small_model.state_dict().keys())
        data_keys = set(result["model_data"].keys())
        assert state_keys == data_keys

    def test_uncompressed_format(self, syncer_uncompressed, small_model):
        result = syncer_uncompressed.prepare_model_for_sync(small_model)
        meta = result["metadata"]
        assert meta["compressed"] is False
        # Each value should have raw numpy arrays with shape/dtype
        for key, val in result["model_data"].items():
            assert "data" in val
            assert "shape" in val
            assert "dtype" in val
            assert isinstance(val["data"], np.ndarray)


# ---------------------------------------------------------------------------
# TestRestoreModelFromSync
# ---------------------------------------------------------------------------


class TestRestoreModelFromSync:
    def test_roundtrip_compressed(self, syncer, small_model):
        original_sd = {k: v.clone() for k, v in small_model.state_dict().items()}
        sync_data = syncer.prepare_model_for_sync(small_model)

        # Create a fresh model and restore into it
        target = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        result = syncer.restore_model_from_sync(sync_data, target)

        assert result is True
        for key in original_sd:
            assert torch.allclose(original_sd[key], target.state_dict()[key])

    def test_roundtrip_uncompressed(self, syncer_uncompressed, small_model):
        original_sd = {k: v.clone() for k, v in small_model.state_dict().items()}
        sync_data = syncer_uncompressed.prepare_model_for_sync(small_model)

        target = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        result = syncer_uncompressed.restore_model_from_sync(sync_data, target)

        assert result is True
        for key in original_sd:
            assert torch.allclose(original_sd[key], target.state_dict()[key])

    def test_corrupted_data_returns_false(self, syncer, small_model):
        import gzip

        # Valid gzip data but wrong size for float32/shape — triggers
        # ValueError in np.frombuffer, which is caught by restore_model_from_sync.
        bad_bytes = gzip.compress(b"hello")
        sync_data = {
            "model_data": {
                "bad_key": {
                    "data": bad_bytes,
                    "shape": (100,),
                    "dtype": "float32",
                    "compressed": True,
                }
            },
            "metadata": {"compressed": True, "sync_count": 0, "total_parameters": 0},
        }
        result = syncer.restore_model_from_sync(sync_data, small_model)
        assert result is False

    def test_mismatched_keys_returns_false(self, syncer, small_model):
        # Prepare sync data from a different-shaped model
        other_model = nn.Linear(3, 7)
        sync_data = syncer.prepare_model_for_sync(other_model)
        result = syncer.restore_model_from_sync(sync_data, small_model)
        assert result is False


# ---------------------------------------------------------------------------
# TestMarkSyncCompleted
# ---------------------------------------------------------------------------


class TestMarkSyncCompleted:
    def test_updates_step_and_count(self, syncer):
        syncer.mark_sync_completed(100)
        assert syncer.last_sync_step == 100
        assert syncer.sync_count == 1

        syncer.mark_sync_completed(200)
        assert syncer.last_sync_step == 200
        assert syncer.sync_count == 2


# ---------------------------------------------------------------------------
# TestGetSyncStats
# ---------------------------------------------------------------------------


class TestGetSyncStats:
    def test_initial_stats(self, syncer):
        stats = syncer.get_sync_stats()
        assert stats["sync_count"] == 0
        assert stats["last_sync_step"] == 0
        assert stats["sync_interval"] == 100
        assert stats["compression_enabled"] is True
        assert stats["average_sync_rate"] == 0

    def test_stats_after_syncs(self, syncer):
        syncer.mark_sync_completed(100)
        syncer.mark_sync_completed(200)
        stats = syncer.get_sync_stats()
        assert stats["sync_count"] == 2
        assert stats["last_sync_step"] == 200
        # average_sync_rate = 2 / max(1, 200) = 0.01
        assert stats["average_sync_rate"] == pytest.approx(0.01)

    def test_zero_step_average(self, syncer):
        # Before any sync, last_sync_step == 0, so rate should be 0
        stats = syncer.get_sync_stats()
        assert stats["average_sync_rate"] == 0
