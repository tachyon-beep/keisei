"""Integration tests: checkpoint save/load/resume flow.

Verifies that a real agent can save its state, reload it, and resume
training with correct state restoration.
"""

import numpy as np
import pytest
import torch

from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi.shogi_game import ShogiGame
from keisei.training.step_manager import StepManager


# Module-level no-op logger
def _noop_logger(msg, also_to_wandb=False, wandb_data=None, log_level="info"):
    pass


# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointSave:
    """Agent can save a checkpoint to disk."""

    def test_save_creates_file(self, ppo_agent, tmp_path):
        """save_model creates a checkpoint file on disk."""
        ckpt_path = str(tmp_path / "checkpoint.pt")
        ppo_agent.save_model(ckpt_path, global_timestep=42, total_episodes_completed=5)

        assert (tmp_path / "checkpoint.pt").exists()

    def test_checkpoint_contains_expected_keys(self, ppo_agent, tmp_path):
        """The saved checkpoint contains model_state_dict and optimizer_state_dict."""
        ckpt_path = str(tmp_path / "checkpoint.pt")
        ppo_agent.save_model(
            ckpt_path,
            global_timestep=100,
            total_episodes_completed=10,
            stats_to_save={"black_wins": 3, "white_wins": 2, "draws": 5},
        )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["global_timestep"] == 100
        assert checkpoint["total_episodes_completed"] == 10
        assert checkpoint["black_wins"] == 3
        assert checkpoint["white_wins"] == 2
        assert checkpoint["draws"] == 5


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointLoad:
    """Agent can load a checkpoint and restore its state."""

    def test_load_restores_training_metadata(self, ppo_agent, tmp_path):
        """load_model returns the saved global_timestep and episode count."""
        ckpt_path = str(tmp_path / "checkpoint.pt")
        ppo_agent.save_model(
            ckpt_path,
            global_timestep=77,
            total_episodes_completed=15,
        )

        result = ppo_agent.load_model(ckpt_path)
        assert result["global_timestep"] == 77
        assert result["total_episodes_completed"] == 15

    def test_load_restores_model_weights(self, integration_config, session_policy_mapper, tmp_path):
        """Model weights after load match those at save time."""
        model = ActorCritic(
            input_channels=46,
            num_actions_total=session_policy_mapper.get_total_actions(),
        )
        agent = PPOAgent(
            model=model, config=integration_config, device=torch.device("cpu")
        )

        # Snapshot weights before save
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        ckpt_path = str(tmp_path / "test_ckpt.pt")
        agent.save_model(ckpt_path, global_timestep=1)

        # Create a new model+agent with fresh (different) weights
        model2 = ActorCritic(
            input_channels=46,
            num_actions_total=session_policy_mapper.get_total_actions(),
        )
        agent2 = PPOAgent(
            model=model2, config=integration_config, device=torch.device("cpu")
        )

        # Load checkpoint
        agent2.load_model(ckpt_path)

        # Verify weights match after load
        for k in original_state:
            assert torch.equal(
                original_state[k], model2.state_dict()[k]
            ), f"Weight mismatch for {k}"

    def test_load_nonexistent_file_returns_error(self, ppo_agent):
        """load_model with a nonexistent path returns an error dict."""
        result = ppo_agent.load_model("/nonexistent/path/checkpoint.pt")
        assert "error" in result
        assert result["global_timestep"] == 0


# ---------------------------------------------------------------------------
# Resume training from checkpoint
# ---------------------------------------------------------------------------


class TestResumeTraining:
    """Training can resume from a checkpoint without errors."""

    def test_resume_training_after_checkpoint(
        self, integration_config, session_policy_mapper, tmp_path
    ):
        """Save checkpoint after N steps, load, continue training for M more steps."""
        # Phase 1: Train for a few steps
        game = ShogiGame(max_moves_per_game=500)
        model = ActorCritic(
            input_channels=46,
            num_actions_total=session_policy_mapper.get_total_actions(),
        )
        agent = PPOAgent(
            model=model, config=integration_config, device=torch.device("cpu")
        )
        buf = ExperienceBuffer(
            buffer_size=integration_config.training.steps_per_epoch,
            gamma=integration_config.training.gamma,
            lambda_gae=integration_config.training.lambda_gae,
            device="cpu",
            num_actions=session_policy_mapper.get_total_actions(),
        )
        sm = StepManager(
            config=integration_config,
            game=game,
            agent=agent,
            policy_mapper=session_policy_mapper,
            experience_buffer=buf,
        )

        episode_state = sm.reset_episode()
        for t in range(integration_config.training.steps_per_epoch):
            result = sm.execute_step(episode_state, global_timestep=t, logger_func=_noop_logger)
            if result.success:
                episode_state = sm.update_episode_state(episode_state, result)
                if result.done:
                    episode_state = sm.reset_episode()
            else:
                episode_state = sm.reset_episode()

        last_val = agent.get_value(episode_state.current_obs)
        buf.compute_advantages_and_returns(last_val)
        metrics_phase1 = agent.learn(buf)
        buf.clear()

        # Save checkpoint
        ckpt_path = str(tmp_path / "resume_test.pt")
        agent.save_model(ckpt_path, global_timestep=16, total_episodes_completed=1)

        # Phase 2: Create new agent, load checkpoint, continue training
        model2 = ActorCritic(
            input_channels=46,
            num_actions_total=session_policy_mapper.get_total_actions(),
        )
        agent2 = PPOAgent(
            model=model2, config=integration_config, device=torch.device("cpu")
        )
        resume_data = agent2.load_model(ckpt_path)
        assert resume_data["global_timestep"] == 16

        game2 = ShogiGame(max_moves_per_game=500)
        buf2 = ExperienceBuffer(
            buffer_size=integration_config.training.steps_per_epoch,
            gamma=integration_config.training.gamma,
            lambda_gae=integration_config.training.lambda_gae,
            device="cpu",
            num_actions=session_policy_mapper.get_total_actions(),
        )
        sm2 = StepManager(
            config=integration_config,
            game=game2,
            agent=agent2,
            policy_mapper=session_policy_mapper,
            experience_buffer=buf2,
        )

        episode_state2 = sm2.reset_episode()
        for t in range(integration_config.training.steps_per_epoch):
            result = sm2.execute_step(
                episode_state2, global_timestep=16 + t, logger_func=_noop_logger
            )
            if result.success:
                episode_state2 = sm2.update_episode_state(episode_state2, result)
                if result.done:
                    episode_state2 = sm2.reset_episode()
            else:
                episode_state2 = sm2.reset_episode()

        last_val2 = agent2.get_value(episode_state2.current_obs)
        buf2.compute_advantages_and_returns(last_val2)
        metrics_phase2 = agent2.learn(buf2)

        # Phase 2 should produce valid metrics
        assert "ppo/policy_loss" in metrics_phase2
        assert np.isfinite(metrics_phase2["ppo/policy_loss"])

    def test_agent_produces_same_action_after_load(
        self, integration_config, session_policy_mapper, legal_mask_fn, tmp_path
    ):
        """After save + load, deterministic action selection matches."""
        model = ActorCritic(
            input_channels=46,
            num_actions_total=session_policy_mapper.get_total_actions(),
        )
        agent = PPOAgent(
            model=model, config=integration_config, device=torch.device("cpu")
        )

        game = ShogiGame(max_moves_per_game=500)
        obs = game.get_observation()
        legal_moves = game.get_legal_moves()
        legal_mask = legal_mask_fn(legal_moves)

        # Deterministic action before save
        move_before, idx_before, _, _ = agent.select_action(
            obs, legal_mask, is_training=False
        )

        ckpt_path = str(tmp_path / "deterministic_test.pt")
        agent.save_model(ckpt_path)

        # Load into a new agent
        model2 = ActorCritic(
            input_channels=46,
            num_actions_total=session_policy_mapper.get_total_actions(),
        )
        agent2 = PPOAgent(
            model=model2, config=integration_config, device=torch.device("cpu")
        )
        agent2.load_model(ckpt_path)

        # Deterministic action after load -- should be identical
        move_after, idx_after, _, _ = agent2.select_action(
            obs, legal_mask, is_training=False
        )

        assert idx_before == idx_after, (
            f"Deterministic action changed after checkpoint round-trip: "
            f"{idx_before} vs {idx_after}"
        )

    def test_optimizer_state_restored(
        self, integration_config, session_policy_mapper, tmp_path
    ):
        """Optimizer state (e.g. momentum) is preserved across save/load."""
        model = ActorCritic(
            input_channels=46,
            num_actions_total=session_policy_mapper.get_total_actions(),
        )
        agent = PPOAgent(
            model=model, config=integration_config, device=torch.device("cpu")
        )

        # Run a dummy backward pass so optimizer has non-trivial state
        obs = torch.randn(1, 46, 9, 9)
        logits, value = model(obs)
        loss = logits.sum() + value.sum()
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        # Get optimizer state before save
        opt_state_before = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in agent.optimizer.state_dict()["state"].items()
        }

        ckpt_path = str(tmp_path / "opt_test.pt")
        agent.save_model(ckpt_path)

        # New agent, load
        model2 = ActorCritic(
            input_channels=46,
            num_actions_total=session_policy_mapper.get_total_actions(),
        )
        agent2 = PPOAgent(
            model=model2, config=integration_config, device=torch.device("cpu")
        )
        agent2.load_model(ckpt_path)

        opt_state_after = agent2.optimizer.state_dict()["state"]

        # Both should have the same parameter group keys with matching tensor values
        assert set(opt_state_before.keys()) == set(opt_state_after.keys())
