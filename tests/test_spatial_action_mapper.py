"""Integration tests for the SpatialActionMapper via VecEnv and direct Python bindings."""

import numpy as np
import pytest

from shogi_gym import SpatialActionMapper, VecEnv


@pytest.fixture
def spatial_env():
    """Single-env VecEnv with spatial action mode."""
    return VecEnv(num_envs=1, max_ply=100, observation_mode="default", action_mode="spatial")


@pytest.fixture
def mapper():
    return SpatialActionMapper()


class TestSpatialActionMapperBasic:
    def test_action_space_size(self, mapper):
        assert mapper.action_space_size == 11259

    def test_action_space_on_env(self, spatial_env):
        assert spatial_env.action_space_size == 11259

    def test_legal_mask_shape(self, spatial_env):
        result = spatial_env.reset()
        masks = np.array(result.legal_masks)
        assert masks.shape == (1, 11259)

    def test_legal_mask_has_legal_moves(self, spatial_env):
        result = spatial_env.reset()
        masks = np.array(result.legal_masks)
        assert masks[0].sum() > 0, "Startpos should have legal moves"


class TestSpatialFlatIndexContract:
    def test_flat_index_is_square_times_139_plus_slot(self, mapper):
        """Verify: flat_index = square * 139 + move_type"""
        # Encode a drop (P*e5, sq 40, piece 0=Pawn)
        idx = mapper.encode_drop_move(40, 0, False)
        expected = 40 * 139 + 132  # slot 132 = first drop piece type
        assert idx == expected, f"Expected {expected}, got {idx}"


class TestSpatialRoundTrip:
    def test_board_move_roundtrip(self, mapper):
        """A simple board move: from sq 40 to sq 31 (one step N from e5 to e4)."""
        idx = mapper.encode_board_move(40, 31, False, False)
        decoded = mapper.decode(idx, False)
        assert decoded["type"] == "board"
        assert decoded["from_sq"] == 40
        assert decoded["to_sq"] == 31
        assert decoded["promote"] is False

    def test_drop_move_roundtrip(self, mapper):
        """Drop a pawn at e5."""
        idx = mapper.encode_drop_move(40, 0, False)
        decoded = mapper.decode(idx, False)
        assert decoded["type"] == "drop"
        assert decoded["to_sq"] == 40
        assert decoded["piece_type_idx"] == 0

    def test_white_perspective_roundtrip(self, mapper):
        """Verify white perspective flipping works for board moves."""
        idx = mapper.encode_board_move(40, 31, False, True)
        decoded = mapper.decode(idx, True)
        assert decoded["from_sq"] == 40
        assert decoded["to_sq"] == 31


class TestSpatialKnightMoves:
    """Knight moves use a separate encoding path (slots 128-131) — verify via PyO3."""

    def test_black_knight_roundtrip(self, mapper):
        # Black knight at sq 58 (e7, row 6 col 4) jumps to sq 49 (d6, row 5 col 3)
        # This is a left-forward knight jump: dr=-2, dc=-1
        idx = mapper.encode_board_move(58, 49, False, False)
        decoded = mapper.decode(idx, False)
        assert decoded["type"] == "board"
        assert decoded["from_sq"] == 58
        assert decoded["to_sq"] == 49
        slot = idx % 139
        assert 128 <= slot <= 131, f"Knight slot {slot} out of expected range 128-131"

    def test_white_knight_roundtrip(self, mapper):
        # White knight: perspective-flipped. Encode from White's view.
        idx = mapper.encode_board_move(22, 31, False, True)
        decoded = mapper.decode(idx, True)
        assert decoded["type"] == "board"
        assert decoded["from_sq"] == 22
        assert decoded["to_sq"] == 31

    def test_black_white_perspective_produces_different_indices(self, mapper):
        """Same physical squares must produce different flat indices for Black vs White."""
        idx_black = mapper.encode_board_move(40, 31, False, False)
        idx_white = mapper.encode_board_move(40, 31, False, True)
        assert idx_black != idx_white, \
            "Black and White perspectives must produce different flat indices"

    def test_knight_promote_roundtrip(self, mapper):
        """Knight move with promotion flag."""
        idx = mapper.encode_board_move(58, 49, True, False)  # promote=True
        decoded = mapper.decode(idx, False)
        assert decoded["from_sq"] == 58
        assert decoded["to_sq"] == 49
        assert decoded["promote"] is True
        slot = idx % 139
        assert 128 <= slot <= 131


class TestSpatialBoardMoveIndexContract:
    """Verify the flat_index = square * 139 + slot contract for board moves (not just drops)."""

    def test_board_move_flat_index(self, mapper):
        # One step North from sq 40 (e5) to sq 31 (e4): dir=N, dist=1
        # N direction = index 0 in the direction table, slot = 0*8 + (1-1) = 0
        idx = mapper.encode_board_move(40, 31, False, False)
        expected_sq = 40  # Black perspective: sq 40 is already correct
        slot = idx - expected_sq * 139
        assert 0 <= slot < 139, f"Slot {slot} out of range for sq {expected_sq}"


class TestSpatialStepExecution:
    def test_step_with_spatial_action(self, spatial_env):
        """Execute a full reset-step cycle with spatial action encoding."""
        result = spatial_env.reset()
        masks = np.array(result.legal_masks)
        action = int(np.argmax(masks[0]))  # pick first legal action
        step_result = spatial_env.step([action])

        obs = np.array(step_result.observations)
        assert obs.shape[0] == 1  # one env
        assert len(step_result.rewards) == 1

    def test_multi_step_stability(self, spatial_env):
        """Run 20 steps without crashing."""
        result = spatial_env.reset()
        for _ in range(20):
            masks = np.array(result.legal_masks)
            if masks[0].sum() == 0:
                # Game ended, explicitly reset
                result = spatial_env.reset()
                continue
            step_result = spatial_env.step([int(np.argmax(masks[0]))])
            result = step_result


class TestSpatialInvalidMode:
    def test_invalid_action_mode(self):
        with pytest.raises(ValueError, match="Unknown action_mode"):
            VecEnv(num_envs=1, max_ply=100, action_mode="invalid")


class TestKataGoFullConfig:
    """Test VecEnv with both KataGo observation and spatial action modes."""

    def test_full_katago_config(self):
        env = VecEnv(
            num_envs=2,
            max_ply=50,
            observation_mode="katago",
            action_mode="spatial",
        )
        assert env.observation_channels == 50
        assert env.action_space_size == 11259

        result = env.reset()
        obs = np.array(result.observations)
        masks = np.array(result.legal_masks)
        assert obs.shape == (2, 50, 9, 9)
        assert masks.shape == (2, 11259)

        # Step
        actions = [int(np.argmax(masks[i])) for i in range(2)]
        step_result = env.step(actions)
        assert len(step_result.rewards) == 2
