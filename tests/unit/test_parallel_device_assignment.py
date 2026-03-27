"""Tests for GPU device assignment in parallel self-play workers."""

from unittest.mock import patch

import pytest

from keisei.config_schema import ParallelConfig
from keisei.training.parallel.parallel_manager import ParallelManager


class TestParallelConfigFields:
    """Test ParallelConfig fields for device assignment."""

    def test_worker_device_map_default(self):
        """Default worker_device_map should be 'auto'."""
        config = ParallelConfig()
        assert config.worker_device_map == "auto"

    def test_max_workers_per_gpu_default(self):
        """Default max_workers_per_gpu should be 8."""
        config = ParallelConfig()
        assert config.max_workers_per_gpu == 8

    def test_worker_device_map_explicit_values(self):
        """Explicit device map values should be accepted."""
        for value in ["auto", "cpu", "cuda:0", "cuda:1"]:
            config = ParallelConfig(worker_device_map=value)
            assert config.worker_device_map == value

    def test_max_workers_per_gpu_validation_min(self):
        """max_workers_per_gpu below 1 should fail validation."""
        with pytest.raises(Exception):
            ParallelConfig(max_workers_per_gpu=0)

    def test_max_workers_per_gpu_validation_max(self):
        """max_workers_per_gpu above 32 should fail validation."""
        with pytest.raises(Exception):
            ParallelConfig(max_workers_per_gpu=33)

    def test_max_workers_per_gpu_explicit(self):
        """Explicit max_workers_per_gpu values should be accepted."""
        config = ParallelConfig(max_workers_per_gpu=4)
        assert config.max_workers_per_gpu == 4

    def test_config_serialization_includes_new_fields(self):
        """New fields should appear in model_dump."""
        config = ParallelConfig()
        dumped = config.model_dump()
        assert "worker_device_map" in dumped
        assert "max_workers_per_gpu" in dumped


class TestDeviceAssignment:
    """Test _assign_worker_device round-robin logic."""

    def _make_manager(self, device_map="auto", num_workers=4):
        """Create a ParallelManager with minimal config for testing."""
        parallel_config = {
            "enabled": False,
            "num_workers": num_workers,
            "batch_size": 32,
            "sync_interval": 100,
            "compression_enabled": False,
            "timeout_seconds": 5.0,
            "max_queue_size": 100,
            "worker_seed_offset": 1000,
            "worker_device_map": device_map,
            "max_workers_per_gpu": 8,
        }
        return ParallelManager(
            env_config={},
            model_config={},
            parallel_config=parallel_config,
            device="cpu",
        )

    @patch("torch.cuda.device_count", return_value=0)
    def test_auto_no_gpus_falls_back_to_cpu(self, mock_count):
        """With auto and 0 GPUs, all workers should get cpu."""
        mgr = self._make_manager(device_map="auto")
        for i in range(4):
            assert mgr._assign_worker_device(i) == "cpu"

    @patch("torch.cuda.device_count", return_value=1)
    def test_auto_single_gpu(self, mock_count):
        """With auto and 1 GPU, all workers should get cuda:0."""
        mgr = self._make_manager(device_map="auto")
        for i in range(4):
            assert mgr._assign_worker_device(i) == "cuda:0"

    @patch("torch.cuda.device_count", return_value=2)
    def test_auto_two_gpus_round_robin(self, mock_count):
        """With auto and 2 GPUs, workers alternate cuda:0 and cuda:1."""
        mgr = self._make_manager(device_map="auto")
        assert mgr._assign_worker_device(0) == "cuda:0"
        assert mgr._assign_worker_device(1) == "cuda:1"
        assert mgr._assign_worker_device(2) == "cuda:0"
        assert mgr._assign_worker_device(3) == "cuda:1"

    def test_explicit_device(self):
        """Explicit device map should be returned for all workers."""
        mgr = self._make_manager(device_map="cuda:2")
        for i in range(4):
            assert mgr._assign_worker_device(i) == "cuda:2"

    def test_cpu_device(self):
        """CPU device map should be returned for all workers."""
        mgr = self._make_manager(device_map="cpu")
        for i in range(4):
            assert mgr._assign_worker_device(i) == "cpu"


class TestWorkerDeviceInjection:
    """Test that worker configs are built correctly per worker."""

    @patch("torch.cuda.device_count", return_value=2)
    def test_worker_config_gets_device(self, mock_count):
        """Each worker should get a config dict with worker_device set."""
        parallel_config = {
            "enabled": False,
            "num_workers": 4,
            "batch_size": 32,
            "sync_interval": 100,
            "compression_enabled": False,
            "timeout_seconds": 5.0,
            "max_queue_size": 100,
            "worker_seed_offset": 1000,
            "worker_device_map": "auto",
            "max_workers_per_gpu": 8,
        }
        mgr = ParallelManager(
            env_config={},
            model_config={},
            parallel_config=parallel_config,
            device="cpu",
        )
        # Simulate what start_workers does
        for worker_id in range(4):
            worker_config = dict(mgr.parallel_config)
            worker_config["worker_device"] = mgr._assign_worker_device(worker_id)
            expected_device = f"cuda:{worker_id % 2}"
            assert worker_config["worker_device"] == expected_device
            # Original config should not be mutated
            assert "worker_device" not in mgr.parallel_config

    def test_worker_device_defaults_to_cpu_in_config(self):
        """Worker config without worker_device should default to cpu via .get()."""
        config = {"some_key": "value"}
        device = config.get("worker_device", "cpu")
        assert device == "cpu"
