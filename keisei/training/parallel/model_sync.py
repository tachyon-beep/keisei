"""
Model synchronization utilities for parallel training.

Handles efficient synchronization of model weights between the main training
process and worker processes, including compression and versioning.
"""

import gzip
import logging
import pickle
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from .utils import compress_array, decompress_array

logger = logging.getLogger(__name__)


class ModelSynchronizer:
    """
    Manages synchronization of model weights between main process and workers.

    Provides efficient model weight distribution with optional compression
    and tracks synchronization intervals.
    """

    def __init__(self, sync_interval: int = 100, compression_enabled: bool = True):
        """
        Initialize model synchronizer.

        Args:
            sync_interval: Number of steps between synchronizations
            compression_enabled: Whether to compress model weights for transmission
        """
        self.sync_interval = sync_interval
        self.compression_enabled = compression_enabled
        self.last_sync_step = 0
        self.sync_count = 0

        logger.info(
            "Model synchronizer initialized (interval=%d, compression=%s)",
            sync_interval,
            compression_enabled,
        )

    def should_sync(self, current_step: int) -> bool:
        """
        Check if model should be synchronized at current step.

        Args:
            current_step: Current training step

        Returns:
            True if synchronization should occur
        """
        return current_step - self.last_sync_step >= self.sync_interval

    def prepare_model_for_sync(self, model: nn.Module) -> Dict[str, Any]:
        """
        Prepare model state dict for efficient transmission.

        Args:
            model: PyTorch model to synchronize

        Returns:
            Prepared model data for transmission
        """
        state_dict = model.state_dict()

        # Convert to CPU and numpy for efficient transmission
        prepared_data = {}
        total_params = 0

        for key, tensor in state_dict.items():
            cpu_tensor = tensor.cpu()
            np_array = cpu_tensor.numpy()
            total_params += np_array.size

            if self.compression_enabled:
                prepared_data[key] = self._compress_array(np_array)
            else:
                prepared_data[key] = {
                    "data": np_array,
                    "shape": np_array.shape,
                    "dtype": str(np_array.dtype),
                }

        sync_metadata = {
            "sync_count": self.sync_count,
            "timestamp": time.time(),
            "total_parameters": total_params,
            "compressed": self.compression_enabled,
            "model_keys": list(state_dict.keys()),
        }

        return {"model_data": prepared_data, "metadata": sync_metadata}

    def restore_model_from_sync(
        self, sync_data: Dict[str, Any], model: nn.Module
    ) -> bool:
        """
        Restore model from synchronized data.

        Args:
            sync_data: Data received from synchronization
            model: Model to update

        Returns:
            True if restoration was successful
        """
        try:
            model_data = sync_data["model_data"]
            metadata = sync_data["metadata"]

            # Reconstruct state dict
            state_dict = {}
            for key, data in model_data.items():
                if metadata["compressed"]:
                    np_array = self._decompress_array(data)
                else:
                    np_array = data["data"]
                state_dict[key] = torch.from_numpy(
                    self._ensure_writable_array(np_array)
                )

            # Load into model
            model.load_state_dict(state_dict)

            logger.debug(
                "Model synchronized (sync_count=%d, " "params=%d)",
                metadata["sync_count"],
                metadata["total_parameters"],
            )

            return True

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error("Failed to restore model from sync data: %s", str(e))
            return False

    @staticmethod
    def _ensure_writable_array(array: np.ndarray) -> np.ndarray:
        """Return an array backed by writable storage for safe tensor conversion."""
        if array.flags.writeable:
            return array
        return np.array(array, copy=True)

    def mark_sync_completed(self, current_step: int) -> None:
        """
        Mark synchronization as completed for the current step.

        Args:
            current_step: Current training step
        """
        self.last_sync_step = current_step
        self.sync_count += 1
        logger.debug(
            "Model sync completed at step %d (count=%d)", current_step, self.sync_count
        )

    def _compress_array(self, array: np.ndarray) -> Dict[str, Any]:
        """Wrapper around :func:`compress_array` for backward compatibility."""
        return compress_array(array)

    def _decompress_array(self, compressed_data: Dict[str, Any]) -> np.ndarray:
        """Wrapper around :func:`decompress_array` for backward compatibility."""
        return decompress_array(compressed_data)

    def get_sync_stats(self) -> Dict[str, Any]:
        """
        Get synchronization statistics.

        Returns:
            Dictionary with sync statistics
        """
        return {
            "sync_count": self.sync_count,
            "last_sync_step": self.last_sync_step,
            "sync_interval": self.sync_interval,
            "compression_enabled": self.compression_enabled,
            "average_sync_rate": (
                self.sync_count / max(1, self.last_sync_step)
                if self.last_sync_step > 0
                else 0
            ),
        }
