"""
Self-play worker process for parallel experience collection.

This module implements worker processes that run self-play games independently
and collect experiences to send back to the main training process.
"""

import logging
import multiprocessing as mp
import queue
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from keisei.core.actor_critic_protocol import ActorCriticProtocol
from keisei.core.experience_buffer import Experience
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils.utils import PolicyOutputMapper

from .utils import decompress_array

logger = logging.getLogger(__name__)


class SelfPlayWorker(mp.get_context("spawn").Process):
    """
    Worker process for parallel self-play experience collection.

    Each worker runs an independent environment and model to collect experiences,
    which are then sent back to the main training process via queues.
    """

    def __init__(
        self,
        worker_id: int,
        env_config: Dict[str, Any],
        model_config: Dict[str, Any],
        parallel_config: Dict[str, Any],
        experience_queue: mp.Queue,
        model_queue: mp.Queue,
        control_queue: mp.Queue,
        seed_offset: int = 1000,
    ):
        """
        Initialize self-play worker.

        Args:
            worker_id: Unique identifier for this worker
            env_config: Environment configuration
            model_config: Model configuration
            parallel_config: Parallel training configuration
            experience_queue: Queue to send experiences to main process
            model_queue: Queue to receive model updates from main process
            control_queue: Queue to receive control commands
            seed_offset: Offset for random seed to ensure diversity
        """
        super().__init__()
        self.worker_id = worker_id
        self.env_config = env_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        # Communication queues
        self.experience_queue = experience_queue
        self.model_queue = model_queue
        self.control_queue = control_queue

        # Worker state
        self.seed = env_config["seed"] + seed_offset + worker_id
        self.running = True
        self.steps_collected = 0
        self.games_played = 0

        # Will be initialized in run()
        self.game: Optional[ShogiGame] = None
        self.model: Optional[ActorCriticProtocol] = None
        self.policy_mapper: Optional[PolicyOutputMapper] = None
        self.device = torch.device(
            self.parallel_config.get("worker_device", "cpu")
        )

        # Internal state tracking
        self._current_obs: Optional[np.ndarray] = None

        logger.info("Worker %d initialized with seed %d", worker_id, self.seed)

    def run(self) -> None:
        """Main worker loop - runs in the worker process."""
        try:
            self._setup_worker()
            self._worker_loop()
        except (ValueError, RuntimeError, ImportError, OSError) as e:
            logger.error("Worker %d failed: %s", self.worker_id, str(e))
        finally:
            self._cleanup_worker()

    def _setup_worker(self) -> None:
        """Initialize worker environment and model."""
        try:
            # Set random seeds for reproducibility and diversity
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Initialize environment
            self.game = ShogiGame()

            # Initialize PolicyOutputMapper for proper action space mapping
            self.policy_mapper = PolicyOutputMapper()

            # Initialize a model architecture for the worker using model_factory
            # This will be replaced when the first model update is received
            from keisei.training.models import model_factory

            input_channels = self.model_config.get("input_channels", 46)
            num_actions = self.model_config.get(
                "num_actions", 13527
            )  # Use correct action space size

            self.model = model_factory(
                model_type=self.model_config.get("model_type", "resnet"),
                obs_shape=(input_channels, 9, 9),
                num_actions=num_actions,
                tower_depth=self.model_config.get("tower_depth", 9),
                tower_width=self.model_config.get("tower_width", 256),
                se_ratio=self.model_config.get("se_ratio", 0.25),
            )
            if self.model is None:
                raise RuntimeError("Failed to create model using model_factory")
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode for inference

            logger.info("Worker %d setup complete", self.worker_id)
        except (ValueError, RuntimeError, ImportError) as e:
            logger.error("Worker %d setup failed: %s", self.worker_id, str(e))
            raise

    def _worker_loop(self) -> None:
        """Main worker experience collection loop."""
        batch_experiences = []
        batch_size = self.parallel_config["batch_size"]

        while self.running:
            # Check for control commands
            self._check_control_commands()

            # Check for model updates
            self._check_model_updates()

            if not self.running:
                break

            # Skip experience collection if model not ready
            if self.model is None:
                time.sleep(0.1)
                continue

            # Collect one experience
            experience = self._collect_single_experience()
            if experience is not None:
                batch_experiences.append(experience)
                self.steps_collected += 1

                # Send batch when full
                if len(batch_experiences) >= batch_size:
                    self._send_experience_batch(batch_experiences)
                    batch_experiences = []

        # Send any remaining experiences
        if batch_experiences:
            self._send_experience_batch(batch_experiences)

    def _collect_single_experience(self) -> Optional[Experience]:
        """
        Collect a single experience from the environment.

        Returns:
            Experience object or None if collection failed
        """
        try:
            if self.game is None or self.model is None:
                return None

            # Reset game if needed
            if self._current_obs is None:
                self._current_obs = self.game.reset()

            # Get current observation
            obs_tensor = torch.from_numpy(self._current_obs).float().to(self.device)

            # Get model predictions
            with torch.no_grad():
                action_logits, value = self.model.forward(obs_tensor.unsqueeze(0))
                action_probs = torch.softmax(action_logits, dim=-1)

                # Get legal moves and create action mask using PolicyOutputMapper
                legal_moves = self.game.get_legal_moves()
                if self.policy_mapper is not None:
                    legal_mask = self.policy_mapper.get_legal_mask(
                        legal_moves, self.device
                    )
                else:
                    logger.error(
                        "Worker %d: PolicyOutputMapper not initialized", self.worker_id
                    )
                    return None

                # Mask illegal actions
                masked_probs = action_probs.squeeze(0) * legal_mask.float()
                masked_probs = masked_probs / (masked_probs.sum() + 1e-8)

                # Sample action
                action_dist = torch.distributions.Categorical(masked_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

            # Convert action index to move using PolicyOutputMapper
            if self.policy_mapper is not None:
                try:
                    selected_move = self.policy_mapper.policy_index_to_shogi_move(
                        int(action.item())
                    )
                except (IndexError, ValueError) as e:
                    logger.error(
                        "Worker %d invalid action %d: %s",
                        self.worker_id,
                        int(action.item()),
                        str(e),
                    )
                    return None
            else:
                logger.error(
                    "Worker %d: PolicyOutputMapper not initialized", self.worker_id
                )
                return None

            # Take action in environment
            next_obs, reward, done, _ = self.game.make_move(selected_move)

            # Create experience
            experience = Experience(
                obs=obs_tensor.cpu(),  # Move back to CPU for transmission
                action=int(action.item()),
                reward=float(reward),
                log_prob=log_prob.item(),
                value=value.squeeze().item(),
                done=bool(done),
                legal_mask=legal_mask.cpu(),
            )

            # Update current observation
            if done:
                self._current_obs = None  # Force environment reset
                self.games_played += 1
            else:
                # next_obs should be numpy array from make_move
                if isinstance(next_obs, np.ndarray):
                    self._current_obs = next_obs
                else:
                    # Fallback: get fresh observation
                    self._current_obs = self.game.get_observation()

            return experience

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(
                "Worker %d experience collection failed: %s", self.worker_id, str(e)
            )
            return None

    def _send_experience_batch(self, experiences: List[Experience]) -> None:
        """
        Send batch of experiences to main process.

        Args:
            experiences: List of Experience objects to send
        """
        try:
            # Convert experiences to batched tensor format for efficient IPC
            batched_tensors = self._experiences_to_batch(experiences)

            # Add worker metadata
            batch_message = {
                "worker_id": self.worker_id,
                "experiences": batched_tensors,  # Now sending batched tensors instead of individual objects
                "batch_size": len(experiences),
                "timestamp": time.time(),
                "steps_collected": self.steps_collected,
                "games_played": self.games_played,
            }

            # Send to main process
            timeout = self.parallel_config.get("timeout_seconds", 10.0)
            self.experience_queue.put(batch_message, timeout=timeout)

            logger.debug(
                "Worker %d sent batch of %d experiences as tensors",
                self.worker_id,
                len(experiences),
            )

        except queue.Full:
            logger.warning(
                "Worker %d experience queue full, dropping batch", self.worker_id
            )
        except (ValueError, RuntimeError, OSError) as e:
            logger.error(
                "Worker %d failed to send experiences: %s", self.worker_id, str(e)
            )

    def _experiences_to_batch(
        self, experiences: List[Experience]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert list of experiences to batched tensor format.

        Args:
            experiences: List of Experience objects

        Returns:
            Dictionary with batched tensors
        """
        obs_list = [exp.obs for exp in experiences]

        return {
            "obs": torch.stack(obs_list, dim=0),
            "actions": torch.tensor(
                [exp.action for exp in experiences], dtype=torch.int64
            ),
            "rewards": torch.tensor(
                [exp.reward for exp in experiences], dtype=torch.float32
            ),
            "log_probs": torch.tensor(
                [exp.log_prob for exp in experiences], dtype=torch.float32
            ),
            "values": torch.tensor(
                [exp.value for exp in experiences], dtype=torch.float32
            ),
            "dones": torch.tensor([exp.done for exp in experiences], dtype=torch.bool),
            "legal_masks": torch.stack([exp.legal_mask for exp in experiences], dim=0),
        }

    def _check_control_commands(self) -> None:
        """Check for control commands from main process."""
        try:
            while True:
                try:
                    command_msg = self.control_queue.get_nowait()
                    self._handle_control_command(command_msg)
                except queue.Empty:
                    break
        except (ValueError, RuntimeError, OSError) as e:
            logger.error(
                "Worker %d control command check failed: %s", self.worker_id, str(e)
            )

    def _check_model_updates(self) -> None:
        """Check for model updates from main process."""
        try:
            while True:
                try:
                    model_data = self.model_queue.get_nowait()
                    self._update_model(model_data)
                except queue.Empty:
                    break
        except (ValueError, RuntimeError, OSError) as e:
            logger.error(
                "Worker %d model update check failed: %s", self.worker_id, str(e)
            )

    def _handle_control_command(self, command_msg: Dict) -> None:
        """
        Handle control command from main process.

        Args:
            command_msg: Command message dictionary
        """
        command = command_msg.get("command")

        if command == "stop":
            logger.info("Worker %d received stop command", self.worker_id)
            self.running = False
        elif command == "reset":
            logger.info("Worker %d received reset command", self.worker_id)
            if self.game:
                self._current_obs = None  # Force environment reset
        elif command == "pause":
            # Simple pause implementation - could be more sophisticated
            pause_duration = command_msg.get("data", {}).get("duration", 1.0)
            time.sleep(pause_duration)
        else:
            logger.warning(
                "Worker %d received unknown command: %s", self.worker_id, command
            )

    def _update_model(self, model_data: Dict) -> None:
        """
        Update worker model with new weights from main process.

        Args:
            model_data: Model weight data from main process
        """
        try:
            if not self.model:
                logger.warning(
                    "Worker %d received model update but model not initialized",
                    self.worker_id,
                )
                return

            # Extract model weights
            weights = model_data["model_data"]

            # Reconstruct state dict
            state_dict = {}
            for key, data in weights.items():
                if isinstance(data, dict) and "data" in data:
                    if data.get("compressed", False):
                        array_np = decompress_array(data)
                        if not isinstance(array_np, np.ndarray):
                            raise ValueError(
                                f"Decompressed data for key '{key}' is not a valid numpy array: {type(array_np)}"
                            )
                    else:
                        array_np = data["data"]
                    if not array_np.flags.writeable:
                        array_np = array_np.copy()
                    state_dict[key] = torch.from_numpy(array_np).to(self.device)
                else:
                    if isinstance(data, np.ndarray) and not data.flags.writeable:
                        data = data.copy()
                    state_dict[key] = torch.from_numpy(data).to(self.device)

            # Load weights
            self.model.load_state_dict(state_dict)

            logger.debug("Worker %d model updated", self.worker_id)

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error("Worker %d model update failed: %s", self.worker_id, str(e))

    def _cleanup_worker(self) -> None:
        """Clean up worker resources."""
        if self.game:
            # Close game resources if needed
            pass

        logger.info(
            "Worker %d cleanup complete (steps=%d, games=%d)",
            self.worker_id,
            self.steps_collected,
            self.games_played,
        )

    def get_worker_stats(self) -> Dict[str, Any]:
        """
        Get worker statistics.

        Returns:
            Dictionary with worker statistics
        """
        return {
            "worker_id": self.worker_id,
            "steps_collected": self.steps_collected,
            "games_played": self.games_played,
            "running": self.running,
            "seed": self.seed,
        }
