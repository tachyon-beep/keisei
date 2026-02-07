"""
Communication utilities for parallel training workers.

Provides queue-based communication between main process and worker processes,
including experience collection and model synchronization.
"""

import logging
import multiprocessing as mp
import queue
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils import compress_array

logger = logging.getLogger(__name__)


class WorkerCommunicator:
    """
    Manages communication between main training process and worker processes.

    Uses multiprocessing queues for efficient data exchange:
    - Experience queue: Workers -> Main (collected experiences)
    - Model queue: Main -> Workers (updated model weights)
    - Control queue: Main -> Workers (control commands)
    """

    def __init__(
        self, num_workers: int, max_queue_size: int = 1000, timeout: float = 10.0
    ):
        """
        Initialize communication channels.

        Args:
            num_workers: Number of worker processes
            max_queue_size: Maximum size of communication queues
            timeout: Timeout for queue operations in seconds
        """
        self.num_workers = num_workers
        self.timeout = timeout

        # Create queues for each worker
        self.experience_queues: List[mp.Queue] = [
            mp.Queue(maxsize=max_queue_size) for _ in range(num_workers)
        ]
        self.model_queues: List[mp.Queue] = [
            mp.Queue(maxsize=10)
            for _ in range(num_workers)  # Smaller for model weights
        ]
        self.control_queues: List[mp.Queue] = [
            mp.Queue(maxsize=10) for _ in range(num_workers)  # Control commands
        ]

        logger.info("Initialized worker communication for %d workers", num_workers)

    def send_model_weights(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        compression_enabled: bool = True,
    ) -> None:
        """
        Send updated model weights to all workers.

        Args:
            model_state_dict: PyTorch model state dictionary
            compression_enabled: Whether to compress weights for transmission
        """
        try:
            # Prepare model data for transmission
            model_data = self._prepare_model_data(model_state_dict, compression_enabled)

            # Send to all workers
            for i, model_queue in enumerate(self.model_queues):
                try:
                    model_queue.put(model_data, timeout=self.timeout)
                except queue.Full:
                    logger.warning(
                        "Model queue for worker %d is full, skipping update", i
                    )

        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Failed to send model weights: %s", str(e))

    def collect_experiences(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Collect experiences from all workers.

        Returns:
            List of (worker_id, experiences) tuples
        """
        collected_experiences = []

        for worker_id, exp_queue in enumerate(self.experience_queues):
            try:
                # Non-blocking collection
                while True:
                    try:
                        batch_data = exp_queue.get_nowait()
                        collected_experiences.append((worker_id, batch_data))
                    except queue.Empty:
                        break

            except (RuntimeError, OSError, queue.Empty) as e:
                logger.error(
                    "Failed to collect experiences from worker %d: %s",
                    worker_id,
                    str(e),
                )

        return collected_experiences

    def send_control_command(
        self,
        command: str,
        data: Optional[Dict] = None,
        worker_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Send control command to workers.

        Args:
            command: Command string ('stop', 'reset', 'pause', etc.)
            data: Optional command data
            worker_ids: Specific workers to send to (None = all workers)
        """
        command_msg = {"command": command, "data": data, "timestamp": time.time()}

        target_workers = (
            worker_ids if worker_ids is not None else range(self.num_workers)
        )

        for worker_id in target_workers:
            try:
                self.control_queues[worker_id].put(command_msg, timeout=self.timeout)
            except queue.Full:
                logger.warning("Control queue for worker %d is full", worker_id)
            except (RuntimeError, OSError) as e:
                logger.error(
                    "Failed to send control command to worker %d: %s", worker_id, str(e)
                )

    def _prepare_model_data(
        self, state_dict: Dict[str, torch.Tensor], compress: bool
    ) -> Dict[str, Any]:
        """
        Prepare model state dict for transmission with actual compression.

        Args:
            state_dict: PyTorch model state dictionary
            compress: Whether to compress the data

        Returns:
            Prepared model data for queue transmission
        """
        # Convert tensors to numpy arrays for transmission
        model_data = {}
        total_original_size = 0
        total_compressed_size = 0

        for key, tensor in state_dict.items():
            np_array = tensor.cpu().numpy()

            if compress:
                compressed = compress_array(np_array)
            else:
                compressed = {
                    "data": np_array,
                    "shape": np_array.shape,
                    "dtype": str(np_array.dtype),
                    "compressed": False,
                }

            model_data[key] = compressed
            original_size = (
                compressed.get("original_size", np_array.nbytes)
                if compress
                else np_array.nbytes
            )
            size = compressed.get("compressed_size", np_array.nbytes)

            total_original_size += original_size
            total_compressed_size += size

        compression_ratio = (
            total_original_size / total_compressed_size
            if total_compressed_size > 0
            else 1.0
        )

        return {
            "model_data": model_data,
            "timestamp": time.time(),
            "compressed": compress,
            "compression_ratio": compression_ratio,
            "total_original_size": total_original_size,
            "total_compressed_size": total_compressed_size,
        }

    def get_queue_info(self) -> Dict[str, List[int]]:
        """
        Get current queue sizes for monitoring.

        Returns:
            Dictionary with queue sizes for each worker
        """
        return {
            "experience_queue_sizes": [q.qsize() for q in self.experience_queues],
            "model_queue_sizes": [q.qsize() for q in self.model_queues],
            "control_queue_sizes": [q.qsize() for q in self.control_queues],
        }

    def cleanup(self) -> None:
        """Clean up communication resources."""
        logger.info("Cleaning up worker communication resources")

        # Send stop command to all workers
        self.send_control_command("stop")

        # Clear all queues
        for queues in [self.experience_queues, self.model_queues, self.control_queues]:
            for q in queues:
                try:
                    while not q.empty():
                        q.get_nowait()
                except queue.Empty:
                    pass
                q.close()

        logger.info("Worker communication cleanup complete")
