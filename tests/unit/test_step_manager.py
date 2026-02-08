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


# ---------------------------------------------------------------------------
# _clear_episode_counters tests
# ---------------------------------------------------------------------------


class TestClearEpisodeCounters:
    """Tests for StepManager._clear_episode_counters."""

    def test_resets_all_counters_to_zero(self):
        sm = _make_step_manager()
        sm.sente_capture_count = 5
        sm.gote_capture_count = 3
        sm.sente_drop_count = 2
        sm.gote_drop_count = 1
        sm.sente_promo_count = 4
        sm.gote_promo_count = 2
        sm.sente_best_capture = "Rook"
        sm.sente_best_capture_value = 10
        sm.gote_best_capture = "Bishop"
        sm.gote_best_capture_value = 8

        sm._clear_episode_counters()

        assert sm.sente_capture_count == 0
        assert sm.gote_capture_count == 0
        assert sm.sente_drop_count == 0
        assert sm.gote_drop_count == 0
        assert sm.sente_promo_count == 0
        assert sm.gote_promo_count == 0
        assert sm.sente_best_capture is None
        assert sm.sente_best_capture_value == 0
        assert sm.gote_best_capture is None
        assert sm.gote_best_capture_value == 0

    def test_clears_move_history_and_log(self):
        sm = _make_step_manager()
        sm.move_history.extend([(0, 0, 1, 0, False), (1, 1, 2, 2, True)])
        sm.move_log.extend(["Move 1", "Move 2"])

        sm._clear_episode_counters()

        assert sm.move_history == []
        assert sm.move_log == []


# ---------------------------------------------------------------------------
# _obs_to_tensor tests
# ---------------------------------------------------------------------------


class TestObsToTensor:
    """Tests for StepManager._obs_to_tensor."""

    def test_correct_shape_and_batch_dim(self):
        sm = _make_step_manager()
        obs = _make_obs()
        tensor = sm._obs_to_tensor(obs)
        assert tensor.shape == (1, 46, 9, 9)

    def test_correct_dtype_and_device(self):
        sm = _make_step_manager()
        obs = _make_obs()
        tensor = sm._obs_to_tensor(obs)
        assert tensor.dtype == torch.float32
        assert tensor.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# _reset_and_fail tests
# ---------------------------------------------------------------------------


class TestResetAndFail:
    """Tests for StepManager._reset_and_fail."""

    def test_normal_path_returns_failure_step_result(self):
        sm = _make_step_manager()
        result = sm._reset_and_fail(
            "test error", done=True, info={"terminal_reason": "test"}
        )
        assert result.success is False
        assert result.done is True
        assert result.error_message == "test error"
        assert result.info == {"terminal_reason": "test"}
        assert result.next_obs.shape == (46, 9, 9)

    def test_done_false_preserved(self):
        sm = _make_step_manager()
        result = sm._reset_and_fail("test error", done=False, info={})
        assert result.done is False

    def test_double_failure_with_fallback(self):
        sm = _make_step_manager()
        sm.game.reset.side_effect = RuntimeError("reset crashed")
        fallback_obs = _make_obs()
        fallback_tensor = _make_obs_tensor()

        result = sm._reset_and_fail(
            "original error",
            done=False,
            info={},
            fallback_obs=fallback_obs,
            fallback_obs_tensor=fallback_tensor,
        )
        assert result.success is False
        assert result.done is True  # Forced to True on double failure
        assert "Reset also failed" in result.error_message
        assert "original error" in result.error_message

    def test_double_failure_without_fallback_raises(self):
        sm = _make_step_manager()
        sm.game.reset.side_effect = RuntimeError("reset crashed")

        with pytest.raises(RuntimeError, match="reset crashed"):
            sm._reset_and_fail("original error", done=False, info={})

    def test_double_failure_logs_when_logger_provided(self):
        sm = _make_step_manager()
        sm.game.reset.side_effect = RuntimeError("reset crashed")
        logger = MagicMock()

        sm._reset_and_fail(
            "original error",
            done=False,
            info={},
            fallback_obs=_make_obs(),
            fallback_obs_tensor=_make_obs_tensor(),
            logger_func=logger,
        )
        logger.assert_called_once()
        logged_msg = logger.call_args[0][0]
        assert "Reset failed during error recovery" in logged_msg


# ---------------------------------------------------------------------------
# _track_move_stats tests
# ---------------------------------------------------------------------------


class TestTrackMoveStats:
    """Tests for StepManager._track_move_stats."""

    def test_sente_capture_increments_count(self):
        from keisei.shogi.shogi_core_definitions import Color

        sm = _make_step_manager()
        move = (0, 0, 1, 0, False)  # Board move, no promotion
        info = {"captured_piece_type": "PAWN"}
        sm._track_move_stats(Color.BLACK, move, info)
        assert sm.sente_capture_count == 1
        assert sm.gote_capture_count == 0

    def test_gote_capture_increments_count(self):
        from keisei.shogi.shogi_core_definitions import Color

        sm = _make_step_manager()
        move = (0, 0, 1, 0, False)
        info = {"captured_piece_type": "ROOK"}
        sm._track_move_stats(Color.WHITE, move, info)
        assert sm.gote_capture_count == 1
        assert sm.sente_capture_count == 0

    def test_best_capture_tracking(self):
        from keisei.shogi.shogi_core_definitions import Color

        sm = _make_step_manager()
        # First capture: PAWN (value 1)
        sm._track_move_stats(
            Color.BLACK, (0, 0, 1, 0, False), {"captured_piece_type": "PAWN"}
        )
        assert sm.sente_best_capture == "Pawn"
        assert sm.sente_best_capture_value == 1
        # Second capture: ROOK (value 10) should upgrade
        sm._track_move_stats(
            Color.BLACK, (0, 0, 1, 0, False), {"captured_piece_type": "ROOK"}
        )
        assert sm.sente_best_capture == "Rook"
        assert sm.sente_best_capture_value == 10
        # Third capture: LANCE (value 3) should NOT downgrade
        sm._track_move_stats(
            Color.BLACK, (0, 0, 1, 0, False), {"captured_piece_type": "LANCE"}
        )
        assert sm.sente_best_capture == "Rook"
        assert sm.sente_best_capture_value == 10

    def test_promoted_piece_capture_uses_base_name(self):
        from keisei.shogi.shogi_core_definitions import Color

        sm = _make_step_manager()
        sm._track_move_stats(
            Color.BLACK,
            (0, 0, 1, 0, False),
            {"captured_piece_type": "PROMOTED_SILVER"},
        )
        assert sm.sente_best_capture == "Silver"
        assert sm.sente_best_capture_value == 5

    def test_drop_tracking(self):
        from keisei.shogi.shogi_core_definitions import Color

        sm = _make_step_manager()
        drop_move = (None, None, 3, 4, False)  # Drop move
        sm._track_move_stats(Color.BLACK, drop_move, {})
        assert sm.sente_drop_count == 1
        sm._track_move_stats(Color.WHITE, drop_move, {})
        assert sm.gote_drop_count == 1

    def test_promotion_tracking(self):
        from keisei.shogi.shogi_core_definitions import Color

        sm = _make_step_manager()
        promo_move = (0, 0, 1, 0, True)  # Promotion move
        sm._track_move_stats(Color.BLACK, promo_move, {})
        assert sm.sente_promo_count == 1
        sm._track_move_stats(Color.WHITE, promo_move, {})
        assert sm.gote_promo_count == 1

    def test_no_stats_on_normal_move(self):
        from keisei.shogi.shogi_core_definitions import Color

        sm = _make_step_manager()
        normal_move = (0, 0, 1, 0, False)  # No capture, no drop, no promo
        sm._track_move_stats(Color.BLACK, normal_move, {})
        assert sm.sente_capture_count == 0
        assert sm.sente_drop_count == 0
        assert sm.sente_promo_count == 0


# ---------------------------------------------------------------------------
# execute_step broadened exception handling tests
# ---------------------------------------------------------------------------


class TestExecuteStepExceptionHandling:
    """Tests that execute_step catches RuntimeError, TypeError, and IndexError."""

    def _setup_step_manager_with_move_error(self, error):
        """Set up a StepManager where make_move raises the given error."""
        sm = _make_step_manager()
        sm.game.get_legal_moves.return_value = [(0, 0, 1, 0, False)]
        sm.policy_mapper.get_legal_mask.return_value = torch.ones(13527)
        sm.agent.select_action.return_value = (
            (0, 0, 1, 0, False),
            42,
            -0.5,
            0.8,
        )
        sm.game.make_move.side_effect = error
        return sm

    def test_catches_runtime_error(self):
        sm = self._setup_step_manager_with_move_error(
            RuntimeError("tensor shape mismatch")
        )
        result = sm.execute_step(
            _make_episode_state(), global_timestep=0, logger_func=_noop_logger
        )
        assert result.success is False
        assert "tensor shape mismatch" in result.error_message

    def test_catches_type_error(self):
        sm = self._setup_step_manager_with_move_error(
            TypeError("unsupported operand type")
        )
        result = sm.execute_step(
            _make_episode_state(), global_timestep=0, logger_func=_noop_logger
        )
        assert result.success is False
        assert "unsupported operand type" in result.error_message

    def test_catches_index_error(self):
        sm = self._setup_step_manager_with_move_error(
            IndexError("move tuple index out of range")
        )
        result = sm.execute_step(
            _make_episode_state(), global_timestep=0, logger_func=_noop_logger
        )
        assert result.success is False
        assert "move tuple index out of range" in result.error_message


# ---------------------------------------------------------------------------
# handle_episode_end reset delegation tests
# ---------------------------------------------------------------------------


class TestHandleEpisodeEndResetDelegation:
    """Tests that handle_episode_end delegates to reset_episode."""

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

    def test_delegates_to_reset_episode(self):
        sm = _make_step_manager()
        state = _make_episode_state()
        step_result = self._make_step_result(winner="black", reason="Tsumi")
        game_stats = {"black_wins": 0, "white_wins": 0, "draws": 0}

        with patch.object(sm, "reset_episode", wraps=sm.reset_episode) as mock_reset:
            sm.handle_episode_end(state, step_result, game_stats, 0, _noop_logger)
            mock_reset.assert_called_once()

    def test_returns_old_state_on_reset_failure(self):
        sm = _make_step_manager()
        sm.game.reset.side_effect = RuntimeError("game engine crash")
        state = _make_episode_state()
        step_result = self._make_step_result(winner="white", reason="Tsumi")
        game_stats = {"black_wins": 0, "white_wins": 0, "draws": 0}

        new_state, winner = sm.handle_episode_end(
            state, step_result, game_stats, 0, _noop_logger
        )
        # Should return original state on failure
        assert new_state is state
        assert winner == "white"
