"""Tests for keisei.webui.state_snapshot — clean state extraction."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from keisei.shogi.shogi_core_definitions import Color, Piece, PieceType
from keisei.shogi.shogi_game import ShogiGame
from keisei.webui.state_snapshot import (
    build_snapshot,
    extract_board_state,
    extract_buffer_info,
    extract_metrics,
    extract_step_info,
    write_snapshot_atomic,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def real_game():
    """A real ShogiGame in its initial state."""
    return ShogiGame()


@pytest.fixture
def metrics_manager():
    """A mock MetricsManager with populated history."""
    mm = MagicMock()
    mm.global_timestep = 5000
    mm.total_episodes_completed = 42
    mm.black_wins = 20
    mm.white_wins = 15
    mm.draws = 7
    mm.processing = False
    mm.get_hot_squares.return_value = ["5e", "4d", "6f"]

    history = MagicMock()
    history.policy_losses = [0.5, 0.4, 0.35]
    history.value_losses = [1.0, 0.9, 0.85]
    history.entropies = [2.5, 2.4, 2.3]
    history.kl_divergences = [0.01, 0.015, 0.012]
    history.clip_fractions = [0.1, 0.12, 0.09]
    history.learning_rates = [3e-4, 3e-4, 2.9e-4]
    history.episode_lengths = [100, 120, 90]
    history.episode_rewards = [0.5, 0.8, -0.3]
    history.win_rates_history = [
        {"win_rate_black": 50.0, "win_rate_white": 30.0, "win_rate_draw": 20.0}
    ]
    mm.history = history
    return mm


@pytest.fixture
def step_manager():
    """A mock StepManager."""
    sm = MagicMock()
    sm.move_log = ["P7g-7f", "P3c-3d", "P2g-2f"]
    sm.sente_capture_count = 3
    sm.gote_capture_count = 2
    sm.sente_drop_count = 1
    sm.gote_drop_count = 0
    sm.sente_promo_count = 2
    sm.gote_promo_count = 1
    return sm


@pytest.fixture
def experience_buffer():
    """A mock ExperienceBuffer."""
    buf = MagicMock()
    buf.size.return_value = 512
    buf.capacity.return_value = 2048
    return buf


@pytest.fixture
def trainer(real_game, metrics_manager, step_manager, experience_buffer):
    """A mock Trainer with all components wired."""
    t = MagicMock()
    t.game = real_game
    t.metrics_manager = metrics_manager
    t.step_manager = step_manager
    t.experience_buffer = experience_buffer
    t.last_gradient_norm = 0.42
    return t


# ---------------------------------------------------------------------------
# extract_board_state
# ---------------------------------------------------------------------------


class TestExtractBoardState:
    def test_initial_board_has_correct_dimensions(self, real_game):
        state = extract_board_state(real_game)
        assert len(state["board"]) == 9
        assert all(len(row) == 9 for row in state["board"])

    def test_initial_board_black_moves_first(self, real_game):
        state = extract_board_state(real_game)
        assert state["current_player"] == "black"

    def test_initial_board_not_game_over(self, real_game):
        state = extract_board_state(real_game)
        assert state["game_over"] is False
        assert state["winner"] is None
        assert state["move_count"] == 0

    def test_initial_board_pieces_correct(self, real_game):
        """Verify a few known initial positions."""
        state = extract_board_state(real_game)
        board = state["board"]

        # Row 0, col 0 should be White Lance (top-left corner)
        piece = board[0][0]
        assert piece is not None
        assert piece["type"] == "lance"
        assert piece["color"] == "white"
        assert piece["promoted"] is False

        # Row 8, col 4 should be Black King (bottom center)
        king = board[8][4]
        assert king is not None
        assert king["type"] == "king"
        assert king["color"] == "black"

    def test_empty_hands_at_start(self, real_game):
        state = extract_board_state(real_game)
        assert state["black_hand"] == {}
        assert state["white_hand"] == {}

    def test_hands_after_capture(self, real_game):
        """Manually place a piece to capture and verify hands."""
        # Put a white pawn where black can capture it
        real_game.board[6][0] = Piece(PieceType.PAWN, Color.WHITE)
        # Remove the existing black pawn at 6,0 to make room
        # (initial position has black pawns at row 6)
        # Actually row 6 already has black pawns, so let's put white on row 5
        real_game.board[5][0] = Piece(PieceType.PAWN, Color.WHITE)
        # Directly add to hand to test extraction
        real_game.hands[0][PieceType.PAWN] = 2
        real_game.hands[1][PieceType.ROOK] = 1

        state = extract_board_state(real_game)
        assert state["black_hand"]["pawn"] == 2
        assert state["white_hand"]["rook"] == 1

    def test_board_state_is_json_serializable(self, real_game):
        state = extract_board_state(real_game)
        result = json.dumps(state)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# extract_metrics
# ---------------------------------------------------------------------------


class TestExtractMetrics:
    def test_basic_stats(self, metrics_manager):
        m = extract_metrics(metrics_manager)
        assert m["global_timestep"] == 5000
        assert m["total_episodes"] == 42
        assert m["black_wins"] == 20
        assert m["white_wins"] == 15
        assert m["draws"] == 7

    def test_learning_curves_present(self, metrics_manager):
        m = extract_metrics(metrics_manager)
        curves = m["learning_curves"]
        assert "policy_losses" in curves
        assert "value_losses" in curves
        assert "entropies" in curves
        assert len(curves["policy_losses"]) == 3

    def test_hot_squares(self, metrics_manager):
        m = extract_metrics(metrics_manager)
        assert m["hot_squares"] == ["5e", "4d", "6f"]

    def test_metrics_json_serializable(self, metrics_manager):
        m = extract_metrics(metrics_manager)
        result = json.dumps(m)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# extract_step_info
# ---------------------------------------------------------------------------


class TestExtractStepInfo:
    def test_move_log(self, step_manager):
        info = extract_step_info(step_manager)
        assert info["move_log"] == ["P7g-7f", "P3c-3d", "P2g-2f"]

    def test_capture_counts(self, step_manager):
        info = extract_step_info(step_manager)
        assert info["sente_capture_count"] == 3
        assert info["gote_capture_count"] == 2

    def test_promo_counts(self, step_manager):
        info = extract_step_info(step_manager)
        assert info["sente_promo_count"] == 2
        assert info["gote_promo_count"] == 1


# ---------------------------------------------------------------------------
# extract_buffer_info
# ---------------------------------------------------------------------------


class TestExtractBufferInfo:
    def test_size_and_capacity(self, experience_buffer):
        info = extract_buffer_info(experience_buffer)
        assert info["size"] == 512
        assert info["capacity"] == 2048


# ---------------------------------------------------------------------------
# build_snapshot
# ---------------------------------------------------------------------------


class TestBuildSnapshot:
    def test_full_snapshot_is_json_serializable(self, trainer):
        snapshot = build_snapshot(trainer, speed=123.4, pending_updates={"ep_metrics": "L:10 R:0.5"})
        result = json.dumps(snapshot)
        assert isinstance(result, str)

    def test_snapshot_contains_all_sections(self, trainer):
        snapshot = build_snapshot(trainer)
        assert "timestamp" in snapshot
        assert "speed" in snapshot
        assert "board_state" in snapshot
        assert "metrics" in snapshot
        assert "step_info" in snapshot
        assert "buffer_info" in snapshot
        assert "model_info" in snapshot

    def test_snapshot_with_none_game(self, trainer):
        trainer.game = None
        snapshot = build_snapshot(trainer)
        assert snapshot["board_state"] is None

    def test_pending_updates_filter_non_scalars(self, trainer):
        snapshot = build_snapshot(
            trainer,
            pending_updates={"speed": 10.0, "complex": {"nested": True}},
        )
        assert "speed" in snapshot["pending_updates"]
        assert "complex" not in snapshot["pending_updates"]


# ---------------------------------------------------------------------------
# write_snapshot_atomic
# ---------------------------------------------------------------------------


class TestWriteSnapshotAtomic:
    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "state.json"
        snapshot = {"timestamp": 1.0, "data": "test"}
        write_snapshot_atomic(snapshot, path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded == snapshot

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "state.json"
        write_snapshot_atomic({"ok": True}, path)
        assert path.exists()

    def test_atomic_no_partial_writes(self, tmp_path):
        """Verify the file either has valid content or doesn't exist."""
        path = tmp_path / "state.json"
        # Write initial content
        write_snapshot_atomic({"version": 1}, path)
        # Overwrite
        write_snapshot_atomic({"version": 2}, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["version"] == 2

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "state.json"
        write_snapshot_atomic({"a": 1}, path)
        write_snapshot_atomic({"b": 2}, path)
        with open(path) as f:
            loaded = json.load(f)
        assert "b" in loaded
        assert "a" not in loaded
