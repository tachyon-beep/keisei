"""
Parallel training manager for coordinating multiple self-play workers.

This module manages the parallel experience collection system, including
worker processes, communication, and model synchronization.
"""

import logging
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn

from keisei.core.experience_buffer import ExperienceBuffer
from keisei.training.parallel.communication import WorkerCommunicator
from keisei.training.parallel.model_sync import ModelSynchronizer
from keisei.training.parallel.self_play_worker import SelfPlayWorker

logger = logging.getLogger(__name__)


class ParallelManager:
    """
    Manages parallel experience collection using multiple worker processes.

    Coordinates worker processes, handles communication, and synchronizes
    model weights between the main training process and workers.
    """

    def __init__(
        self,
        env_config: Dict,
        model_config: Dict,
        parallel_config: Dict,
        device: str = "cuda",
    ):
        """
        Initialize parallel training manager.

        Args:
            env_config: Environment configuration
            model_config: Model configuration
            parallel_config: Parallel training configuration
            device: Device for main process (typically GPU)
        """
        self.env_config = env_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device = torch.device(device)

        # Parallel configuration
        self.num_workers = parallel_config["num_workers"]
        self.batch_size = parallel_config["batch_size"]
        self.enabled = parallel_config["enabled"]

        # Initialize components
        self.communicator = WorkerCommunicator(
            num_workers=self.num_workers,
            max_queue_size=parallel_config["max_queue_size"],
            timeout=parallel_config["timeout_seconds"],
        )

        self.model_sync = ModelSynchronizer(
            sync_interval=parallel_config["sync_interval"],
            compression_enabled=parallel_config["compression_enabled"],
        )

        # Worker processes
        self.workers: List[SelfPlayWorker] = []
        self.worker_stats: Dict[int, Dict] = {}

        # State tracking
        self._last_synced_state_dict = None
        self._worker_restart_counts: Dict[int, int] = {}
        self._max_restarts_per_worker = 5
        self.total_steps_collected = 0
        self.total_batches_received = 0
        self.last_sync_time = time.time()
        self.is_running = False

        logger.info("ParallelManager initialized with %d workers", self.num_workers)

    def _assign_worker_device(self, worker_id: int) -> str:
        """Assign a CUDA device to a worker based on the device map config.

        When device_map is 'auto', distributes workers round-robin across GPUs
        and logs a warning if the max_workers_per_gpu cap would be exceeded.
        """
        device_map = self.parallel_config.get("worker_device_map", "auto")
        if device_map == "auto":
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                return "cpu"
            gpu_id = worker_id % gpu_count
            max_per_gpu = self.parallel_config.get("max_workers_per_gpu", 8)
            workers_on_gpu = sum(
                1 for wid in range(self.num_workers)
                if wid != worker_id and wid % gpu_count == gpu_id
            )
            if workers_on_gpu >= max_per_gpu:
                logger.warning(
                    "GPU %d already has %d workers (max_workers_per_gpu=%d), "
                    "assigning worker %d to CPU instead",
                    gpu_id, workers_on_gpu, max_per_gpu, worker_id,
                )
                return "cpu"
            return f"cuda:{gpu_id}"
        return device_map

    def start_workers(self, initial_model: nn.Module) -> bool:
        """
        Start all worker processes.

        Args:
            initial_model: Initial model to distribute to workers

        Returns:
            True if all workers started successfully
        """
        if not self.enabled:
            logger.info("Parallel collection disabled, skipping worker startup")
            return True

        # Validate num_workers against GPU capacity
        device_map = self.parallel_config.get("worker_device_map", "auto")
        if device_map == "auto":
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                max_per_gpu = self.parallel_config.get("max_workers_per_gpu", 8)
                max_gpu_workers = gpu_count * max_per_gpu
                if self.num_workers > max_gpu_workers:
                    logger.warning(
                        "num_workers (%d) exceeds GPU capacity (%d GPUs × %d max_workers_per_gpu = %d). "
                        "Excess workers will be assigned to CPU.",
                        self.num_workers, gpu_count, max_per_gpu, max_gpu_workers,
                    )

        try:
            # Create and start worker processes
            for worker_id in range(self.num_workers):
                worker_config = dict(self.parallel_config)
                worker_config["worker_device"] = self._assign_worker_device(worker_id)

                worker = SelfPlayWorker(
                    worker_id=worker_id,
                    env_config=self.env_config,
                    model_config=self.model_config,
                    parallel_config=worker_config,
                    experience_queue=self.communicator.experience_queues[worker_id],
                    model_queue=self.communicator.model_queues[worker_id],
                    control_queue=self.communicator.control_queues[worker_id],
                    seed_offset=self.parallel_config["worker_seed_offset"],
                )

                worker.start()
                self.workers.append(worker)

                logger.info("Started worker %d (PID: %d)", worker_id, worker.pid)

            # Verify workers are alive after spawn (catch immediate crashes)
            time.sleep(1)
            dead = [w for w in self.workers if not w.is_alive()]
            if dead:
                for w in dead:
                    logger.error(
                        "Worker %d died immediately (exit code %s)",
                        w.worker_id,
                        w.exitcode,
                    )
                raise RuntimeError(
                    f"{len(dead)}/{len(self.workers)} workers died on startup"
                )

            # Send initial model to all workers
            self._sync_model_to_workers(initial_model)

            self.is_running = True
            logger.info("All %d workers started successfully", self.num_workers)
            return True

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Failed to start workers: %s", str(e))
            self.stop_workers()
            return False

    def _restart_dead_workers(self) -> int:
        """
        Detect and restart any dead worker processes.

        Iterates over self.workers, replacing any that have died with fresh
        SelfPlayWorker instances using the same worker_id, queues, and a
        new per-worker config (including device assignment).

        Returns:
            Number of workers that were restarted.
        """
        restarted = 0
        for idx, worker in enumerate(self.workers):
            if worker.is_alive():
                continue

            worker_id = worker.worker_id
            restart_count = self._worker_restart_counts.get(worker_id, 0)
            if restart_count >= self._max_restarts_per_worker:
                logger.error(
                    "Worker %d exceeded max restarts (%d), not restarting",
                    worker_id,
                    self._max_restarts_per_worker,
                )
                continue

            logger.warning(
                "Worker %d is dead (exit code %s), restarting (%d/%d)",
                worker_id,
                worker.exitcode,
                restart_count + 1,
                self._max_restarts_per_worker,
            )

            # Reap the dead process to avoid zombies on POSIX
            worker.join(timeout=1.0)

            worker_config = dict(self.parallel_config)
            worker_config["worker_device"] = self._assign_worker_device(worker_id)

            new_worker = SelfPlayWorker(
                worker_id=worker_id,
                env_config=self.env_config,
                model_config=self.model_config,
                parallel_config=worker_config,
                experience_queue=self.communicator.experience_queues[worker_id],
                model_queue=self.communicator.model_queues[worker_id],
                control_queue=self.communicator.control_queues[worker_id],
                seed_offset=self.parallel_config["worker_seed_offset"],
            )

            try:
                new_worker.start()
            except (OSError, RuntimeError) as e:
                logger.error("Failed to restart worker %d: %s", worker_id, e)
                continue
            self.workers[idx] = new_worker
            self._worker_restart_counts[worker_id] = restart_count + 1
            restarted += 1

            logger.info(
                "Restarted worker %d (new PID: %d)", worker_id, new_worker.pid
            )

            # Sync current model weights so the restarted worker doesn't
            # run with uninitialized weights until the next periodic sync.
            if self._last_synced_state_dict is not None:
                try:
                    model_data = self.communicator._prepare_model_data(
                        self._last_synced_state_dict,
                        compress=self.parallel_config["compression_enabled"],
                    )
                    self.communicator.model_queues[worker_id].put(
                        model_data, timeout=self.communicator.timeout
                    )
                except (RuntimeError, OSError) as e:
                    logger.warning(
                        "Could not sync model to restarted worker %d: %s",
                        worker_id, e,
                    )

        return restarted

    def collect_experiences(self, experience_buffer: ExperienceBuffer) -> int:
        """
        Collect experiences from all workers and add to main buffer.

        Args:
            experience_buffer: Main experience buffer to fill

        Returns:
            Number of experiences collected
        """
        if not self.enabled or not self.is_running:
            return 0

        self._restart_dead_workers()

        experiences_collected = 0

        try:
            # Collect from all workers
            worker_batches = self.communicator.collect_experiences()

            for worker_id, batch_data in worker_batches:
                # batch_data is a dictionary with experiences and metadata
                worker_experiences = batch_data["experiences"]
                experience_buffer.add_from_worker_batch(worker_experiences)

                batch_size = batch_data.get("batch_size", 0)
                experiences_collected += batch_size
                self.total_batches_received += 1

                # Update worker stats
                self.worker_stats[worker_id] = {
                    "steps_collected": batch_data.get("steps_collected", 0),
                    "games_played": batch_data.get("games_played", 0),
                    "last_batch_time": batch_data.get("timestamp", time.time()),
                    "last_batch_size": batch_size,
                }

            self.total_steps_collected += experiences_collected

            if experiences_collected > 0:
                logger.debug(
                    "Collected %d experiences from workers", experiences_collected
                )

            return experiences_collected

        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Failed to collect experiences: %s", str(e))
            return 0

    def sync_model_if_needed(self, model: nn.Module, current_step: int) -> bool:
        """
        Synchronize model with workers if needed.

        Args:
            model: Current model to synchronize
            current_step: Current training step

        Returns:
            True if synchronization occurred
        """
        if not self.enabled or not self.is_running:
            return False

        if self.model_sync.should_sync(current_step):
            return self._sync_model_to_workers(model, current_step)

        return False

    def _sync_model_to_workers(self, model: nn.Module, current_step: int = 0) -> bool:
        """
        Synchronize model weights to all workers.

        Args:
            model: Model to synchronize
            current_step: Current training step

        Returns:
            True if synchronization succeeded
        """
        try:
            # Send model weights to workers
            state_dict = model.state_dict()
            self._last_synced_state_dict = state_dict
            self.communicator.send_model_weights(
                state_dict,
                compression_enabled=self.parallel_config["compression_enabled"],
            )

            # Mark sync completed
            self.model_sync.mark_sync_completed(current_step)
            self.last_sync_time = time.time()

            logger.debug("Model synchronized to workers at step %d", current_step)
            return True

        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Model synchronization failed: %s", str(e))
            return False

    def stop_workers(self) -> None:
        """Stop all worker processes gracefully."""
        if not self.workers:
            return

        logger.info("Stopping parallel workers...")

        try:
            # Send stop command to all workers
            self.communicator.send_control_command("stop")

            # Wait for workers to finish
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=5.0)  # 5 second timeout

                    if worker.is_alive():
                        logger.warning("Force terminating worker %d", worker.pid)
                        worker.terminate()
                        worker.join(timeout=2.0)

            # Clean up communication
            self.communicator.cleanup()

            self.workers.clear()
            self.is_running = False

            logger.info("All workers stopped")

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Error stopping workers: %s", str(e))

    def get_parallel_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive parallel training statistics.

        Returns:
            Dictionary with parallel training stats
        """
        queue_info = self.communicator.get_queue_info() if self.is_running else {}
        sync_stats = self.model_sync.get_sync_stats()

        return {
            "enabled": self.enabled,
            "running": self.is_running,
            "num_workers": self.num_workers,
            "total_steps_collected": self.total_steps_collected,
            "total_batches_received": self.total_batches_received,
            "last_sync_time": self.last_sync_time,
            "worker_stats": self.worker_stats.copy(),
            "queue_info": queue_info,
            "sync_stats": sync_stats,
            "collection_rate": self._calculate_collection_rate(),
        }

    def _calculate_collection_rate(self) -> float:
        """Calculate current experience collection rate."""
        if not self.worker_stats:
            return 0.0

        # Simple rate calculation based on recent activity
        current_time = time.time()
        recent_steps = 0

        for stats in self.worker_stats.values():
            last_batch_time = stats.get("last_batch_time", 0)
            if current_time - last_batch_time < 60:  # Last minute
                recent_steps += stats.get("last_batch_size", 0)

        return recent_steps / 60.0  # Steps per second

    def reset_workers(self) -> None:
        """Reset all worker environments."""
        if self.is_running:
            self.communicator.send_control_command("reset")
            logger.info("Reset command sent to all workers")

    def pause_workers(self, duration: float = 1.0) -> None:
        """
        Pause all workers for specified duration.

        Args:
            duration: Pause duration in seconds
        """
        if self.is_running:
            self.communicator.send_control_command("pause", data={"duration": duration})
            logger.info("Pause command sent to all workers (duration=%fs)", duration)

    def is_healthy(self) -> bool:
        """
        Check if parallel system is healthy.

        Returns:
            True if system appears to be functioning normally
        """
        if not self.enabled:
            return True  # Disabled system is considered healthy

        if not self.is_running:
            return False

        # Check if workers are alive
        alive_workers = sum(1 for worker in self.workers if worker.is_alive())
        if alive_workers < self.num_workers:
            logger.warning("Only %d/%d workers alive", alive_workers, self.num_workers)
            return False

        # Check recent activity
        current_time = time.time()
        recent_activity = any(
            current_time - stats.get("last_batch_time", 0) < 30
            for stats in self.worker_stats.values()
        )

        return recent_activity

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.stop_workers()
