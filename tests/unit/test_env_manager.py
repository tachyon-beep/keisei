"""Unit tests for keisei.training.env_manager: EnvManager."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from keisei.training.env_manager import EnvManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    input_channels=46,
    num_actions_total=13527,
    seed=42,
    max_moves_per_game=500,
):
    return SimpleNamespace(
        env=SimpleNamespace(
            device="cpu",
            input_channels=input_channels,
            num_actions_total=num_actions_total,
            seed=seed,
            max_moves_per_game=max_moves_per_game,
        ),
    )


# ---------------------------------------------------------------------------
# Setup tests
# ---------------------------------------------------------------------------


class TestSetupEnvironment:
    """Tests for setup_environment."""

    def test_returns_game_and_mapper_tuple(self):
        config = _make_config()
        em = EnvManager(config)
        game, mapper = em.setup_environment()
        assert game is not None
        assert mapper is not None

    def test_game_type_is_shogi_game(self):
        config = _make_config()
        em = EnvManager(config)
        game, _ = em.setup_environment()
        from keisei.shogi.shogi_game import ShogiGame

        assert isinstance(game, ShogiGame)

    def test_action_space_matches_config(self):
        config = _make_config()
        em = EnvManager(config)
        em.setup_environment()
        assert em.action_space_size == 13527

    def test_obs_space_shape_set(self):
        config = _make_config()
        em = EnvManager(config)
        em.setup_environment()
        assert em.obs_space_shape == (46, 9, 9)

    def test_action_space_mismatch_raises(self):
        config = _make_config(num_actions_total=9999)
        em = EnvManager(config)
        with pytest.raises(RuntimeError, match="Failed to initialize PolicyOutputMapper"):
            em.setup_environment()


# ---------------------------------------------------------------------------
# Reset and state tests
# ---------------------------------------------------------------------------


class TestResetGame:
    """Tests for reset_game."""

    def test_reset_returns_true(self):
        config = _make_config()
        em = EnvManager(config)
        em.setup_environment()
        assert em.reset_game() is True

    def test_reset_without_setup_returns_false(self):
        config = _make_config()
        em = EnvManager(config)
        assert em.reset_game() is False


class TestInitializeGameState:
    """Tests for initialize_game_state."""

    def test_returns_numpy_array(self):
        config = _make_config()
        em = EnvManager(config)
        em.setup_environment()
        obs = em.initialize_game_state()
        assert isinstance(obs, np.ndarray)

    def test_returns_correct_shape(self):
        config = _make_config()
        em = EnvManager(config)
        em.setup_environment()
        obs = em.initialize_game_state()
        assert obs.shape == (46, 9, 9)

    def test_returns_none_without_setup(self):
        config = _make_config()
        em = EnvManager(config)
        assert em.initialize_game_state() is None


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateEnvironment:
    """Tests for validate_environment."""

    def test_returns_true_for_valid_setup(self):
        config = _make_config()
        em = EnvManager(config)
        em.setup_environment()
        assert em.validate_environment() is True

    def test_returns_false_without_setup(self):
        config = _make_config()
        em = EnvManager(config)
        assert em.validate_environment() is False


# ---------------------------------------------------------------------------
# Info and legal moves tests
# ---------------------------------------------------------------------------


class TestGetEnvironmentInfo:
    """Tests for get_environment_info."""

    def test_returns_dict_with_expected_keys(self):
        config = _make_config()
        em = EnvManager(config)
        em.setup_environment()
        info = em.get_environment_info()
        assert "game" in info
        assert "policy_mapper" in info
        assert "action_space_size" in info
        assert "obs_space_shape" in info
        assert "input_channels" in info
        assert "num_actions_total" in info
        assert "game_type" in info


class TestGetLegalMovesCount:
    """Tests for get_legal_moves_count."""

    def test_returns_positive_for_initial_position(self):
        config = _make_config()
        em = EnvManager(config)
        em.setup_environment()
        count = em.get_legal_moves_count()
        assert count > 0

    def test_returns_zero_without_setup(self):
        config = _make_config()
        em = EnvManager(config)
        assert em.get_legal_moves_count() == 0


# ---------------------------------------------------------------------------
# Seeding tests
# ---------------------------------------------------------------------------


class TestSetupSeeding:
    """Tests for setup_seeding."""

    def test_with_valid_seed_does_not_crash(self):
        config = _make_config()
        em = EnvManager(config)
        em.setup_environment()
        # ShogiGame may or may not have a seed method
        result = em.setup_seeding(seed=123)
        # The result depends on whether the game object supports seeding
        assert isinstance(result, bool)

    def test_with_none_seed_returns_false(self):
        config = _make_config(seed=None)
        em = EnvManager(config)
        em.setup_environment()
        result = em.setup_seeding(seed=None)
        assert result is False

    def test_without_game_returns_false(self):
        config = _make_config()
        em = EnvManager(config)
        result = em.setup_seeding(seed=42)
        assert result is False
