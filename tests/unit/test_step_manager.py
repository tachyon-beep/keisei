"""Unit tests for keisei.training.step_manager: StepManager, EpisodeState, StepResult."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from keisei.training.step_manager import EpisodeState, StepManager, StepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config():
    """Build a minimal config for StepManager."""
    return SimpleNamespace(
        env=SimpleNamespace(device="cpu"),
        display=SimpleNamespace(display_moves=False, turn_tick=0.0),
    )


def _make_obs():
    """Create a dummy observation array (46 channels, 9x9 board)."""
    return np.zeros((46, 9, 9), dtype=np.float32)


def _make_obs_tensor():
    """Create a dummy observation tensor with batch dim."""
    return torch.zeros(1, 46, 9, 9)


def _make_episode_state():
    """Create a fresh EpisodeState."""
    obs = _make_obs()
    obs_tensor = _make_obs_tensor()
    return EpisodeState(
        current_obs=obs,
        current_obs_tensor=obs_tensor,
        episode_reward=0.0,
        episode_length=0,
    )


def _make_step_manager():
    """Create a StepManager with mocked dependencies."""
    config = _make_config()
    game = MagicMock()
    game.reset.return_value = _make_obs()
    game.current_player = MagicMock()
    game.current_player.name = "BLACK"

    agent = MagicMock()
    policy_mapper = MagicMock()
    experience_buffer = MagicMock()

    return StepManager(config, game, agent, policy_mapper, experience_buffer)


def _noop_logger(*args, **kwargs):
    pass


# ---------------------------------------------------------------------------
# EpisodeState tests
# ---------------------------------------------------------------------------


class TestEpisodeState:
    """Tests for EpisodeState dataclass."""

    def test_construction_defaults(self):
        state = _make_episode_state()
        assert state.episode_reward == 0.0
        assert state.episode_length == 0
        assert state.current_obs.shape == (46, 9, 9)
        assert state.current_obs_tensor.shape == (1, 46, 9, 9)


# ---------------------------------------------------------------------------
# StepResult tests
# ---------------------------------------------------------------------------


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_default_success_is_true(self):
        result = StepResult(
            next_obs=_make_obs(),
            next_obs_tensor=_make_obs_tensor(),
            reward=1.0,
            done=False,
            info={},
            selected_move=(0, 0, 1, 0, False),
            policy_index=42,
            log_prob=-0.5,
            value_pred=0.8,
        )
        assert result.success is True
        assert result.error_message is None


# ---------------------------------------------------------------------------
# StepManager.reset_episode tests
# ---------------------------------------------------------------------------


class TestResetEpisode:
    """Tests for StepManager.reset_episode."""

    def test_reset_returns_fresh_episode_state(self):
        sm = _make_step_manager()
        state = sm.reset_episode()
        assert isinstance(state, EpisodeState)
        assert state.episode_reward == 0.0
        assert state.episode_length == 0

    def test_reset_clears_move_history(self):
        sm = _make_step_manager()
        sm.move_history.append((0, 0, 1, 0, False))
        sm.move_log.append("test move")
        sm.reset_episode()
        assert sm.move_history == []
        assert sm.move_log == []

    def test_reset_clears_capture_counters(self):
        sm = _make_step_manager()
        sm.sente_capture_count = 5
        sm.gote_capture_count = 3
        sm.reset_episode()
        assert sm.sente_capture_count == 0
        assert sm.gote_capture_count == 0


# ---------------------------------------------------------------------------
# StepManager.update_episode_state tests
# ---------------------------------------------------------------------------


class TestUpdateEpisodeState:
    """Tests for StepManager.update_episode_state."""

    def test_accumulates_reward_and_length(self):
        sm = _make_step_manager()
        state = _make_episode_state()
        step_result = StepResult(
            next_obs=_make_obs(),
            next_obs_tensor=_make_obs_tensor(),
            reward=1.5,
            done=False,
            info={},
            selected_move=(0, 0, 1, 0, False),
            policy_index=0,
            log_prob=0.0,
            value_pred=0.0,
        )
        updated = sm.update_episode_state(state, step_result)
        assert updated.episode_reward == 1.5
        assert updated.episode_length == 1

    def test_multiple_updates_accumulate(self):
        sm = _make_step_manager()
        state = _make_episode_state()
        for i in range(3):
            step_result = StepResult(
                next_obs=_make_obs(),
                next_obs_tensor=_make_obs_tensor(),
                reward=1.0,
                done=False,
                info={},
                selected_move=(0, 0, 1, 0, False),
                policy_index=0,
                log_prob=0.0,
                value_pred=0.0,
            )
            state = sm.update_episode_state(state, step_result)
        assert state.episode_reward == pytest.approx(3.0)
        assert state.episode_length == 3


# ---------------------------------------------------------------------------
# StepManager.execute_step tests
# ---------------------------------------------------------------------------


class TestExecuteStep:
    """Tests for StepManager.execute_step."""

    def test_calls_agent_select_action(self):
        sm = _make_step_manager()
        sm.game.get_legal_moves.return_value = [(0, 0, 1, 0, False)]
        sm.policy_mapper.get_legal_mask.return_value = torch.ones(13527)
        sm.agent.select_action.return_value = (
            (0, 0, 1, 0, False),  # move
            42,  # policy_index
            -0.5,  # log_prob
            0.8,  # value_pred
        )
        sm.game.make_move.return_value = (_make_obs(), 1.0, False, {})

        state = _make_episode_state()
        result = sm.execute_step(state, global_timestep=0, logger_func=_noop_logger)

        sm.agent.select_action.assert_called_once()
        assert result.success is True

    def test_stores_experience_in_buffer(self):
        sm = _make_step_manager()
        sm.game.get_legal_moves.return_value = [(0, 0, 1, 0, False)]
        sm.policy_mapper.get_legal_mask.return_value = torch.ones(13527)
        sm.agent.select_action.return_value = (
            (0, 0, 1, 0, False),
            42,
            -0.5,
            0.8,
        )
        sm.game.make_move.return_value = (_make_obs(), 1.0, False, {})

        state = _make_episode_state()
        sm.execute_step(state, global_timestep=0, logger_func=_noop_logger)

        sm.experience_buffer.add.assert_called_once()

    def test_handles_no_legal_moves(self):
        sm = _make_step_manager()
        sm.game.get_legal_moves.return_value = []

        state = _make_episode_state()
        result = sm.execute_step(state, global_timestep=0, logger_func=_noop_logger)

        assert result.success is False
        assert result.done is True
        assert "no_legal_moves" in result.info.get("terminal_reason", "")

    def test_handles_agent_returning_none_move(self):
        sm = _make_step_manager()
        sm.game.get_legal_moves.return_value = [(0, 0, 1, 0, False)]
        sm.policy_mapper.get_legal_mask.return_value = torch.ones(13527)
        sm.agent.select_action.return_value = (None, 0, 0.0, 0.0)

        state = _make_episode_state()
        result = sm.execute_step(state, global_timestep=0, logger_func=_noop_logger)

        assert result.success is False


# ---------------------------------------------------------------------------
# StepManager.handle_episode_end tests
# ---------------------------------------------------------------------------


class TestHandleEpisodeEnd:
    """Tests for StepManager.handle_episode_end."""

    def _make_step_result(self, winner=None, reason="Unknown"):
        return StepResult(
            next_obs=_make_obs(),
            next_obs_tensor=_make_obs_tensor(),
            reward=1.0,
            done=True,
            info={"winner": winner, "reason": reason},
            selected_move=(0, 0, 1, 0, False),
            policy_index=0,
            log_prob=0.0,
            value_pred=0.0,
        )

    def test_black_win(self):
        sm = _make_step_manager()
        state = _make_episode_state()
        step_result = self._make_step_result(winner="black", reason="Tsumi")
        game_stats = {"black_wins": 0, "white_wins": 0, "draws": 0}

        new_state, winner = sm.handle_episode_end(
            state, step_result, game_stats, 0, _noop_logger
        )
        assert winner == "black"

    def test_white_win(self):
        sm = _make_step_manager()
        state = _make_episode_state()
        step_result = self._make_step_result(winner="white", reason="Tsumi")
        game_stats = {"black_wins": 0, "white_wins": 0, "draws": 0}

        new_state, winner = sm.handle_episode_end(
            state, step_result, game_stats, 0, _noop_logger
        )
        assert winner == "white"

    def test_draw(self):
        sm = _make_step_manager()
        state = _make_episode_state()
        step_result = self._make_step_result(winner=None, reason="MaxMoves")
        game_stats = {"black_wins": 0, "white_wins": 0, "draws": 0}

        new_state, winner = sm.handle_episode_end(
            state, step_result, game_stats, 0, _noop_logger
        )
        assert winner is None

    def test_does_not_modify_original_game_stats(self):
        sm = _make_step_manager()
        state = _make_episode_state()
        step_result = self._make_step_result(winner="black", reason="Tsumi")
        game_stats = {"black_wins": 5, "white_wins": 3, "draws": 2}
        original_stats = game_stats.copy()

        sm.handle_episode_end(state, step_result, game_stats, 10, _noop_logger)
        assert game_stats == original_stats

    def test_resets_counters_after_episode_end(self):
        sm = _make_step_manager()
        sm.sente_capture_count = 3
        sm.gote_capture_count = 2
        state = _make_episode_state()
        step_result = self._make_step_result(winner="black", reason="Tsumi")
        game_stats = {"black_wins": 0, "white_wins": 0, "draws": 0}

        sm.handle_episode_end(state, step_result, game_stats, 0, _noop_logger)
        assert sm.sente_capture_count == 0
        assert sm.gote_capture_count == 0


# ---------------------------------------------------------------------------
# StepManager capture/promotion tracking
# ---------------------------------------------------------------------------


class TestCapturePromotionTracking:
    """Tests for capture and promotion counter tracking during execute_step."""

    def test_capture_tracking(self):
        sm = _make_step_manager()
        sm.game.get_legal_moves.return_value = [(0, 0, 1, 0, False)]
        sm.policy_mapper.get_legal_mask.return_value = torch.ones(13527)
        sm.agent.select_action.return_value = (
            (0, 0, 1, 0, False),
            42,
            -0.5,
            0.8,
        )
        # Simulate a capture by returning captured_piece_type in info
        from keisei.shogi.shogi_core_definitions import Color

        sm.game.current_player = Color.BLACK
        sm.game.make_move.return_value = (
            _make_obs(),
            1.0,
            False,
            {"captured_piece_type": "PAWN"},
        )

        state = _make_episode_state()
        sm.execute_step(state, global_timestep=0, logger_func=_noop_logger)

        assert sm.sente_capture_count == 1
