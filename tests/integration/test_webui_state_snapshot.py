"""Integration tests: WebUI state snapshot building.

Verifies that the state_snapshot module correctly extracts data from
real training components into a JSON-serializable snapshot that conforms
to the v1 BroadcastStateEnvelope contract.
"""

import time
from types import SimpleNamespace

import pytest
import torch

from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi.shogi_game import ShogiGame
from keisei.training.metrics_manager import MetricsManager
from keisei.training.step_manager import StepManager
from keisei.webui.state_snapshot import (
    build_snapshot,
    extract_board_state,
    extract_buffer_info,
    extract_metrics,
    extract_step_info,
)
from keisei.webui.view_contracts import SCHEMA_VERSION, validate_envelope


# Module-level no-op logger
def _noop_logger(msg, also_to_wandb=False, wandb_data=None, log_level="info"):
    pass


# ---------------------------------------------------------------------------
# Helper: minimal trainer-like object for build_snapshot
# ---------------------------------------------------------------------------


def _make_trainer_stub(game, metrics_manager, step_manager, experience_buffer):
    """Build a lightweight object that satisfies build_snapshot's protocol."""
    return SimpleNamespace(
        game=game,
        metrics_manager=metrics_manager,
        step_manager=step_manager,
        experience_buffer=experience_buffer,
        last_gradient_norm=0.0,
    )


# ---------------------------------------------------------------------------
# Board state extraction
# ---------------------------------------------------------------------------


class TestBoardStateExtraction:
    """extract_board_state reads real game data correctly."""

    def test_board_state_has_expected_keys(self, shogi_game):
        """Snapshot board state contains board, current_player, etc."""
        state = extract_board_state(shogi_game)

        assert "board" in state
        assert "current_player" in state
        assert "move_count" in state
        assert "game_over" in state
        assert "winner" in state
        assert "black_hand" in state
        assert "white_hand" in state

    def test_board_is_9x9_grid(self, shogi_game):
        """Board is a 9x9 grid of piece dicts or None."""
        state = extract_board_state(shogi_game)
        board = state["board"]

        assert len(board) == 9
        for row in board:
            assert len(row) == 9

    def test_initial_position_has_pieces(self, shogi_game):
        """Initial board has pieces in expected starting positions."""
        state = extract_board_state(shogi_game)
        board = state["board"]

        # Row 0 should have white's back rank pieces
        assert board[0][4] is not None
        assert board[0][4]["type"] == "king"
        assert board[0][4]["color"] == "white"

        # Row 8 should have black's back rank pieces
        assert board[8][4] is not None
        assert board[8][4]["type"] == "king"
        assert board[8][4]["color"] == "black"

        assert state["current_player"] == "black"
        assert state["game_over"] is False
        assert state["move_count"] == 0


# ---------------------------------------------------------------------------
# Full snapshot
# ---------------------------------------------------------------------------


class TestBuildSnapshot:
    """build_snapshot assembles a v1 BroadcastStateEnvelope."""

    def test_snapshot_contains_all_sections(
        self,
        integration_config,
        shogi_game,
        ppo_agent,
        session_policy_mapper,
        experience_buffer,
    ):
        """Envelope has required top-level keys and training sub-keys."""
        mm = MetricsManager()
        sm = StepManager(
            config=integration_config,
            game=shogi_game,
            agent=ppo_agent,
            policy_mapper=session_policy_mapper,
            experience_buffer=experience_buffer,
        )
        trainer = _make_trainer_stub(shogi_game, mm, sm, experience_buffer)

        snapshot = build_snapshot(trainer, speed=10.5, pending_updates={"epoch": 1})

        # Envelope-level keys
        assert snapshot["schema_version"] == SCHEMA_VERSION
        assert "timestamp" in snapshot
        assert snapshot["speed"] == 10.5
        assert "mode" in snapshot
        assert snapshot["active_views"] == ["training"]
        assert "health" in snapshot
        assert snapshot["pending_updates"] == {"epoch": 1}

        # Training view sub-keys
        training = snapshot["training"]
        assert "board_state" in training
        assert "metrics" in training
        assert "step_info" in training
        assert "buffer_info" in training
        assert "model_info" in training

    def test_snapshot_validates_against_contract(
        self,
        integration_config,
        shogi_game,
        ppo_agent,
        session_policy_mapper,
        experience_buffer,
    ):
        """Snapshot passes validate_envelope() with zero errors."""
        mm = MetricsManager()
        sm = StepManager(
            config=integration_config,
            game=shogi_game,
            agent=ppo_agent,
            policy_mapper=session_policy_mapper,
            experience_buffer=experience_buffer,
        )
        trainer = _make_trainer_stub(shogi_game, mm, sm, experience_buffer)

        snapshot = build_snapshot(trainer, speed=10.5, pending_updates={"epoch": 1})
        errors = validate_envelope(snapshot)
        assert errors == [], f"Envelope validation errors: {errors}"

    def test_snapshot_updates_after_gameplay(
        self,
        integration_config,
        ppo_agent,
        session_policy_mapper,
    ):
        """Snapshot reflects game state changes after moves are played."""
        game = ShogiGame(max_moves_per_game=500)
        buf = ExperienceBuffer(
            buffer_size=16, gamma=0.99, lambda_gae=0.95, device="cpu",
            num_actions=session_policy_mapper.get_total_actions(),
        )
        mm = MetricsManager()
        sm = StepManager(
            config=integration_config,
            game=game,
            agent=ppo_agent,
            policy_mapper=session_policy_mapper,
            experience_buffer=buf,
        )
        trainer = _make_trainer_stub(game, mm, sm, buf)

        # Snapshot before any moves
        snap_before = build_snapshot(trainer)
        assert snap_before["training"]["board_state"]["move_count"] == 0

        # Play a few steps
        episode_state = sm.reset_episode()
        for t in range(3):
            result = sm.execute_step(episode_state, global_timestep=t, logger_func=_noop_logger)
            if result.success:
                episode_state = sm.update_episode_state(episode_state, result)
                if result.done:
                    episode_state = sm.reset_episode()

        # Snapshot after moves
        snap_after = build_snapshot(trainer)
        assert snap_after["training"]["board_state"]["move_count"] > 0
        assert snap_after["training"]["buffer_info"]["size"] > 0

    def test_step_info_tracks_move_log(
        self,
        integration_config,
        ppo_agent,
        session_policy_mapper,
    ):
        """Step info section contains move log entries after gameplay."""
        game = ShogiGame(max_moves_per_game=500)
        config_with_display = integration_config.model_copy(
            update={"display": integration_config.display.model_copy(
                update={"display_moves": True, "turn_tick": 0.0}
            )}
        )
        buf = ExperienceBuffer(
            buffer_size=16, gamma=0.99, lambda_gae=0.95, device="cpu",
            num_actions=session_policy_mapper.get_total_actions(),
        )
        sm = StepManager(
            config=config_with_display,
            game=game,
            agent=ppo_agent,
            policy_mapper=session_policy_mapper,
            experience_buffer=buf,
        )

        episode_state = sm.reset_episode()
        for t in range(3):
            result = sm.execute_step(episode_state, global_timestep=t, logger_func=_noop_logger)
            if result.success:
                episode_state = sm.update_episode_state(episode_state, result)
                if result.done:
                    episode_state = sm.reset_episode()

        step_info = extract_step_info(sm)
        assert "move_log" in step_info
        assert len(step_info["move_log"]) > 0

    def test_metrics_extraction(self):
        """extract_metrics returns expected keys from MetricsManager."""
        mm = MetricsManager()
        metrics = extract_metrics(mm)

        assert "global_timestep" in metrics
        assert "total_episodes" in metrics
        assert "black_wins" in metrics
        assert "white_wins" in metrics
        assert "draws" in metrics
        assert "learning_curves" in metrics
        assert "hot_squares" in metrics
        assert isinstance(metrics["learning_curves"], dict)
