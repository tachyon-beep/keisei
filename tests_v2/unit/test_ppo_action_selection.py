"""Tests for PPOAgent.select_action with real model and real game."""

import numpy as np
import pytest
import torch

from keisei.config_schema import (
    AppConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    ParallelConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils.utils import PolicyOutputMapper


def make_config(tmp_path):
    """Build a minimal AppConfig pointing at tmp_path for logs/models."""
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            input_channels=46,
            num_actions_total=13527,
            seed=42,
        ),
        training=TrainingConfig(
            total_timesteps=200,
            steps_per_epoch=16,
            ppo_epochs=2,
            minibatch_size=8,
            learning_rate=3e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=100,
            evaluation_interval_timesteps=100,
            enable_torch_compile=False,
        ),
        evaluation=EvaluationConfig(num_games=2),
        logging=LoggingConfig(
            log_file=str(tmp_path / "train.log"),
            model_dir=str(tmp_path / "models"),
        ),
        wandb=WandBConfig(enabled=False),
        parallel=ParallelConfig(enabled=False),
    )


@pytest.fixture
def mapper():
    """Shared PolicyOutputMapper instance."""
    return PolicyOutputMapper()


@pytest.fixture
def agent(tmp_path, mapper):
    """PPOAgent with a real ActorCritic model on CPU."""
    config = make_config(tmp_path)
    model = ActorCritic(
        input_channels=46,
        num_actions_total=mapper.get_total_actions(),
    )
    return PPOAgent(
        model=model, config=config, device=torch.device("cpu")
    )


@pytest.fixture
def game():
    """A fresh ShogiGame at the initial position."""
    return ShogiGame()


def _build_legal_mask(game, mapper):
    """Build a boolean legal-action mask from the current game state."""
    legal_moves = game.get_legal_moves()
    mask = torch.zeros(mapper.get_total_actions(), dtype=torch.bool)
    for move in legal_moves:
        try:
            idx = mapper.shogi_move_to_policy_index(move)
            mask[idx] = True
        except (KeyError, ValueError):
            pass
    return mask


# ---------------------------------------------------------------------------
# 1. Action selection basics
# ---------------------------------------------------------------------------


class TestActionSelectionBasics:
    """Basic sanity checks for select_action return values."""

    def test_returns_four_element_tuple(self, agent, game, mapper):
        """select_action returns a tuple of exactly 4 elements."""
        obs = game.get_observation()
        mask = _build_legal_mask(game, mapper)
        result = agent.select_action(obs, mask)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_policy_index_is_valid_int(self, agent, game, mapper):
        """Returned policy_index is an int in [0, num_actions_total)."""
        obs = game.get_observation()
        mask = _build_legal_mask(game, mapper)
        _, policy_index, _, _ = agent.select_action(obs, mask)
        assert isinstance(policy_index, int)
        assert 0 <= policy_index < mapper.get_total_actions()

    def test_log_prob_is_finite(self, agent, game, mapper):
        """Returned log_prob is a finite float (not nan or inf)."""
        obs = game.get_observation()
        mask = _build_legal_mask(game, mapper)
        _, _, log_prob, _ = agent.select_action(obs, mask)
        assert isinstance(log_prob, float)
        assert np.isfinite(log_prob)

    def test_value_is_finite(self, agent, game, mapper):
        """Returned value estimate is a finite float."""
        obs = game.get_observation()
        mask = _build_legal_mask(game, mapper)
        _, _, _, value = agent.select_action(obs, mask)
        assert isinstance(value, float)
        assert np.isfinite(value)

    def test_shogi_move_is_not_none(self, agent, game, mapper):
        """When legal_mask has True entries, the returned move is not None."""
        obs = game.get_observation()
        mask = _build_legal_mask(game, mapper)
        assert mask.any(), "Initial position should have legal moves"
        shogi_move, _, _, _ = agent.select_action(obs, mask)
        assert shogi_move is not None


# ---------------------------------------------------------------------------
# 2. Legal mask enforcement
# ---------------------------------------------------------------------------


class TestLegalMaskEnforcement:
    """Ensure the agent only picks actions marked as legal."""

    def test_selected_index_is_legal(self, agent, game, mapper):
        """The selected policy_index always has legal_mask[index] == True."""
        obs = game.get_observation()
        mask = _build_legal_mask(game, mapper)
        for _ in range(10):
            _, policy_index, _, _ = agent.select_action(obs, mask)
            assert mask[policy_index], (
                f"Agent selected illegal action index {policy_index}"
            )

    def test_single_legal_action_deterministic(self, agent, game, mapper):
        """With only one legal action, that action is always selected."""
        obs = game.get_observation()
        # Build a mask with exactly one legal action
        legal_moves = game.get_legal_moves()
        first_move = legal_moves[0]
        idx = mapper.shogi_move_to_policy_index(first_move)
        mask = torch.zeros(mapper.get_total_actions(), dtype=torch.bool)
        mask[idx] = True

        selected_indices = set()
        for _ in range(5):
            _, policy_index, _, _ = agent.select_action(obs, mask)
            selected_indices.add(policy_index)
        assert selected_indices == {idx}

    def test_all_actions_legal_still_valid(self, agent, game, mapper):
        """With all actions marked legal, the selected action is still valid."""
        obs = game.get_observation()
        mask = torch.ones(mapper.get_total_actions(), dtype=torch.bool)
        _, policy_index, _, _ = agent.select_action(obs, mask)
        assert isinstance(policy_index, int)
        assert 0 <= policy_index < mapper.get_total_actions()

    def test_selected_index_is_legal_after_moves(self, agent, game, mapper):
        """After several game moves, selected action is still legal."""
        # Play a few random moves to change the game state
        import random

        random.seed(42)
        for _ in range(4):
            legal_moves = game.get_legal_moves()
            if not legal_moves or game.game_over:
                break
            move = random.choice(legal_moves)
            game.make_move(move)

        if not game.game_over:
            obs = game.get_observation()
            mask = _build_legal_mask(game, mapper)
            if mask.any():
                _, policy_index, _, _ = agent.select_action(obs, mask)
                assert mask[policy_index]

    def test_get_value_returns_finite(self, agent, game):
        """get_value returns a finite float for any observation."""
        obs = game.get_observation()
        value = agent.get_value(obs)
        assert isinstance(value, float)
        assert np.isfinite(value)


# ---------------------------------------------------------------------------
# 3. Deterministic vs stochastic
# ---------------------------------------------------------------------------


class TestDeterministicVsStochastic:
    """Verify behaviour differences between training and evaluation mode."""

    def test_deterministic_mode_consistent(self, agent, game, mapper):
        """is_training=False produces the same action on repeated calls."""
        obs = game.get_observation()
        mask = _build_legal_mask(game, mapper)
        results = [
            agent.select_action(obs, mask, is_training=False)[1]
            for _ in range(10)
        ]
        assert len(set(results)) == 1, (
            "Deterministic mode should always select the same action"
        )

    def test_stochastic_mode_varies(self, agent, game, mapper):
        """is_training=True may produce different actions across many calls."""
        obs = game.get_observation()
        mask = _build_legal_mask(game, mapper)
        # Only meaningful when there is more than one legal action
        if mask.sum().item() <= 1:
            pytest.skip("Only one legal action available")
        results = [
            agent.select_action(obs, mask, is_training=True)[1]
            for _ in range(50)
        ]
        assert len(set(results)) > 1, (
            "Stochastic mode should occasionally pick different actions "
            f"(got only index {results[0]} in 50 tries)"
        )

    def test_deterministic_selects_highest_probability(
        self, agent, game, mapper
    ):
        """Deterministic mode selects the highest-probability legal action."""
        obs = game.get_observation()
        mask = _build_legal_mask(game, mapper)

        # Get the action deterministic mode picks
        _, det_action, _, _ = agent.select_action(
            obs, mask, is_training=False
        )

        # Manually compute the highest-probability legal action
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            policy_logits, _ = agent.model.forward(obs_tensor)
        masked_logits = torch.where(
            mask.unsqueeze(0),
            policy_logits,
            torch.tensor(float("-inf")),
        )
        expected_action = int(torch.argmax(masked_logits, dim=-1).item())
        assert det_action == expected_action


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case scenarios for action selection."""

    def test_very_sparse_mask(self, agent, game, mapper):
        """A mask with only one True entry works correctly."""
        obs = game.get_observation()
        legal_moves = game.get_legal_moves()
        # Pick the last legal move to ensure it's not always the first
        last_move = legal_moves[-1]
        idx = mapper.shogi_move_to_policy_index(last_move)
        mask = torch.zeros(mapper.get_total_actions(), dtype=torch.bool)
        mask[idx] = True

        shogi_move, policy_index, log_prob, value = agent.select_action(
            obs, mask
        )
        assert policy_index == idx
        assert shogi_move is not None
        assert np.isfinite(log_prob)
        assert np.isfinite(value)

    def test_log_probability_is_negative(self, agent, game, mapper):
        """Log probability should be negative (log of a probability < 1)."""
        obs = game.get_observation()
        mask = _build_legal_mask(game, mapper)
        # With multiple legal actions, the probability of the chosen one < 1
        if mask.sum().item() <= 1:
            pytest.skip("Need more than one legal action")
        _, _, log_prob, _ = agent.select_action(obs, mask)
        assert log_prob < 0.0, (
            f"Log probability should be negative, got {log_prob}"
        )
