"""Training loop orchestrator."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from keisei.config import AppConfig

# ---------------------------------------------------------------------------
# Move notation helpers
# ---------------------------------------------------------------------------

_DROP_PIECE_CHARS = ["P", "L", "N", "S", "G", "B", "R"]


def _sq_to_notation(sq: int) -> str:
    """Convert square index (0-80) to shogi notation like '5g'."""
    row = sq // 9
    col = sq % 9
    return f"{9 - col}{chr(ord('a') + row)}"


def _move_notation(decoded: dict[str, Any]) -> str:
    """Convert a decoded action dict to human-readable notation."""
    if decoded["type"] == "board":
        from_str = _sq_to_notation(decoded["from_sq"])
        to_str = _sq_to_notation(decoded["to_sq"])
        promo = "+" if decoded.get("promote") else ""
        return f"{from_str}\u2192{to_str}{promo}"
    else:  # drop
        idx = decoded["piece_type_idx"]
        piece = _DROP_PIECE_CHARS[idx] if idx < len(_DROP_PIECE_CHARS) else "?"
        to_str = _sq_to_notation(decoded["to_sq"])
        return f"{piece}*{to_str}"
from keisei.db import (
    init_db,
    read_training_state,
    update_heartbeat,
    update_training_progress,
    write_game_snapshots,
    write_metrics,
    write_training_state,
)
from keisei.training.algorithm_registry import PPOParams, validate_algorithm_params
from keisei.training.checkpoint import load_checkpoint, save_checkpoint
from keisei.training.model_registry import build_model
from keisei.training.ppo import PPOAlgorithm, RolloutBuffer

logger = logging.getLogger(__name__)


class TrainingLoop:
    def __init__(self, config: AppConfig, vecenv: Any = None) -> None:
        self.config = config
        self.db_path = config.display.db_path

        init_db(self.db_path)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = build_model(config.model.architecture, config.model.params)
        self.model = self.model.to(self.device)

        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            logger.info("Using %d GPUs via DataParallel", gpu_count)
            self.model = torch.nn.DataParallel(self.model)

        logger.info(
            "Model: %s (%s), params: %d, device: %s, gpus: %d",
            config.model.display_name,
            config.model.architecture,
            sum(p.numel() for p in self.model.parameters()),
            self.device,
            gpu_count,
        )

        ppo_params = validate_algorithm_params(
            config.training.algorithm, config.training.algorithm_params
        )
        assert isinstance(ppo_params, PPOParams)
        # PPO needs the underlying model for optimizer param groups,
        # but uses the DataParallel wrapper for forward passes
        base_model = self.model.module if hasattr(self.model, 'module') else self.model
        self.ppo = PPOAlgorithm(ppo_params, base_model, forward_model=self.model)

        if vecenv is not None:
            self.vecenv = vecenv
        else:
            from shogi_gym import VecEnv

            self.vecenv = VecEnv(
                num_envs=config.training.num_games,
                max_ply=config.training.max_ply,
            )

        from shogi_gym import DefaultActionMapper

        self.action_mapper = DefaultActionMapper()

        self.num_envs = config.training.num_games
        self.buffer = RolloutBuffer(
            num_envs=self.num_envs, obs_shape=(46, 9, 9), action_space=13527
        )
        self.move_histories: list[list[dict[str, Any]]] = [
            [] for _ in range(self.num_envs)
        ]
        self.latest_values: list[float] = [0.0] * self.num_envs
        self.moves_per_minute = config.display.moves_per_minute
        self._last_snapshot_time = 0.0
        self.epoch = 0
        self.global_step = 0
        self._last_heartbeat = time.monotonic()

        self._check_resume()

    def _check_resume(self) -> None:
        state = read_training_state(self.db_path)
        if state is not None and state.get("checkpoint_path"):
            checkpoint_path = Path(state["checkpoint_path"])
            if checkpoint_path.exists():
                logger.warning(
                    "Resuming from checkpoint: %s (epoch %d)",
                    checkpoint_path,
                    state["current_epoch"],
                )
                base_model = self.model.module if hasattr(self.model, 'module') else self.model
                meta = load_checkpoint(
                    checkpoint_path, base_model, self.ppo.optimizer
                )
                self.epoch = meta["epoch"]
                self.global_step = meta["step"]
                return

        write_training_state(
            self.db_path,
            {
                "config_json": json.dumps(
                    {
                        "training": {
                            "num_games": self.config.training.num_games,
                            "algorithm": self.config.training.algorithm,
                        },
                        "model": {"architecture": self.config.model.architecture},
                    }
                ),
                "display_name": self.config.model.display_name,
                "model_arch": self.config.model.architecture,
                "algorithm_name": self.config.training.algorithm,
                "started_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            },
        )

    def run(self, num_epochs: int, steps_per_epoch: int) -> None:
        reset_result = self.vecenv.reset()
        obs = torch.from_numpy(np.array(reset_result.observations)).to(self.device)
        legal_masks = torch.from_numpy(np.array(reset_result.legal_masks)).to(self.device)
        # All games start with Black (0) to move
        current_players = np.zeros(self.num_envs, dtype=np.uint8)

        start_epoch = self.epoch
        for epoch_i in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch_i
            win_count = 0
            black_wins = 0
            white_wins = 0
            draw_count = 0

            for step_i in range(steps_per_epoch):
                self.global_step += 1

                actions, log_probs, values = self.ppo.select_actions(
                    obs, legal_masks
                )
                action_list = actions.tolist()
                # Snapshot who is moving *before* the step
                pre_step_players = current_players.copy()
                step_result = self.vecenv.step(action_list)

                rewards = torch.from_numpy(np.array(step_result.rewards)).to(self.device)
                terminated = torch.from_numpy(np.array(step_result.terminated)).to(self.device)
                truncated = torch.from_numpy(np.array(step_result.truncated)).to(self.device)
                dones = terminated | truncated
                current_players = np.array(step_result.current_players)

                win_count += int((rewards > 0).sum().item())
                self.latest_values = values.tolist()

                for env_i in range(self.num_envs):
                    is_white = bool(pre_step_players[env_i])
                    try:
                        decoded = self.action_mapper.decode(
                            action_list[env_i], is_white
                        )
                        notation = _move_notation(decoded)
                    except Exception:
                        notation = f"a{action_list[env_i]}"
                    self.move_histories[env_i].append(
                        {
                            "action": action_list[env_i],
                            "notation": notation,
                        }
                    )
                    if dones[env_i]:
                        r = rewards[env_i].item()
                        if r > 0:
                            # Last mover won
                            if is_white:
                                white_wins += 1
                            else:
                                black_wins += 1
                        elif r < 0:
                            # Last mover lost — opponent won
                            if is_white:
                                black_wins += 1
                            else:
                                white_wins += 1
                        else:
                            draw_count += 1
                        self.move_histories[env_i] = []
                        current_players[env_i] = 0  # reset to Black

                self.buffer.add(
                    obs, actions, log_probs, values, rewards, dones, legal_masks
                )

                obs = torch.from_numpy(np.array(step_result.observations)).to(self.device)
                legal_masks = torch.from_numpy(np.array(step_result.legal_masks)).to(self.device)

                self._maybe_write_snapshots()
                self._maybe_update_heartbeat()

            self.model.eval()
            with torch.no_grad():
                _, next_values = self.model(obs)
                next_values = next_values.squeeze(-1)

            losses = self.ppo.update(self.buffer, next_values)

            ep_completed = getattr(self.vecenv, "episodes_completed", 0)
            metrics = {
                "epoch": epoch_i,
                "step": self.global_step,
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
                "gradient_norm": losses["gradient_norm"],
                "win_rate": win_count / max(ep_completed, 1)
                if ep_completed
                else None,
                "black_win_rate": black_wins / max(ep_completed, 1)
                if ep_completed
                else None,
                "white_win_rate": white_wins / max(ep_completed, 1)
                if ep_completed
                else None,
                "draw_rate": draw_count / max(ep_completed, 1)
                if ep_completed
                else None,
                "truncation_rate": getattr(
                    self.vecenv, "truncation_rate", None
                ),
                "avg_episode_length": getattr(
                    self.vecenv, "mean_episode_length", None
                ),
                "episodes_completed": ep_completed,
            }
            write_metrics(self.db_path, metrics)

            if hasattr(self.vecenv, "reset_stats"):
                self.vecenv.reset_stats()

            update_training_progress(
                self.db_path, epoch_i, self.global_step
            )

            logger.info(
                "Epoch %d | step %d | policy_loss=%.4f value_loss=%.4f entropy=%.4f",
                epoch_i,
                self.global_step,
                losses["policy_loss"],
                losses["value_loss"],
                losses["entropy"],
            )

            if (epoch_i + 1) % self.config.training.checkpoint_interval == 0:
                ckpt_path = (
                    Path(self.config.training.checkpoint_dir)
                    / f"epoch_{epoch_i:05d}.pt"
                )
                base_model = self.model.module if hasattr(self.model, 'module') else self.model
                save_checkpoint(
                    ckpt_path,
                    base_model,
                    self.ppo.optimizer,
                    epoch_i + 1,
                    self.global_step,
                )
                update_training_progress(
                    self.db_path, epoch_i + 1, self.global_step, str(ckpt_path)
                )
                logger.info("Checkpoint saved: %s", ckpt_path)

    def _maybe_write_snapshots(self) -> None:
        if self.moves_per_minute <= 0:
            return
        now = time.monotonic()
        interval = 60.0 / self.moves_per_minute
        if now - self._last_snapshot_time < interval:
            return
        self._last_snapshot_time = now

        if hasattr(self.vecenv, "get_spectator_data"):
            spectator_data = self.vecenv.get_spectator_data()
            snapshots = []
            for i, game_data in enumerate(spectator_data):
                snapshots.append(
                    {
                        "game_id": i,
                        "board_json": json.dumps(game_data.get("board", [])),
                        "hands_json": json.dumps(game_data.get("hands", {})),
                        "current_player": game_data.get(
                            "current_player", "black"
                        ),
                        "ply": game_data.get("ply", 0),
                        "is_over": int(game_data.get("is_over", False)),
                        "result": game_data.get("result", "in_progress"),
                        "sfen": game_data.get("sfen", ""),
                        "in_check": int(game_data.get("in_check", False)),
                        "move_history_json": json.dumps(
                            self.move_histories[i]
                        ),
                        "value_estimate": self.latest_values[i]
                        if i < len(self.latest_values)
                        else 0.0,
                    }
                )
            write_game_snapshots(self.db_path, snapshots)

    def _maybe_update_heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat >= 10.0:
            self._last_heartbeat = now
            update_training_progress(self.db_path, self.epoch, self.global_step)


def main() -> None:
    import argparse

    from keisei.config import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Keisei training loop")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to TOML config"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=256,
        help="Steps per epoch",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Dispatch to the correct training loop based on algorithm
    if config.training.algorithm == "katago_ppo":
        from keisei.training.katago_loop import KataGoTrainingLoop

        loop = KataGoTrainingLoop(config)
        loop.run(num_epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)
    else:
        loop = TrainingLoop(config)
        loop.run(num_epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)


if __name__ == "__main__":
    main()
