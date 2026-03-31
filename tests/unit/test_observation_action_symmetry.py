"""
Regression test for observation-action perspective alignment.

Verifies that the spatial structure of the observation matches the spatial
structure of the action space for both Black and White. This test would have
caught the original bug where the observation was flipped for White but the
action space was not.
"""

import numpy as np
import pytest
import torch

from keisei.shogi_python_reference import Color, ShogiGame
from keisei.utils.utils import PolicyOutputMapper

pytestmark = pytest.mark.unit

# Channel constants (from OBS_UNPROMOTED_ORDER: PAWN=0, LANCE=1, KNIGHT=2, SILVER=3,
#                   GOLD=4, BISHOP=5, ROOK=6, KING=7)
OBS_CHANNEL_PAWN = 0
OBS_CHANNEL_KING = 7
OBS_OPP_UNPROMOTED_START = 14  # OBS_OPP_PLAYER_UNPROMOTED_START
OBS_CHANNEL_OPP_KING = OBS_OPP_UNPROMOTED_START + 7  # channel 21


@pytest.fixture(scope="module")
def mapper():
    """Module-scoped PolicyOutputMapper (expensive to create)."""
    return PolicyOutputMapper()


class TestObservationActionSymmetry:
    """Regression tests for observation-action perspective alignment."""

    def test_black_pawn_push_spatial_alignment(self, mapper):
        """At start position (Black's turn), get the observation, find own pawns
        (channel 0, should be at row 6). For each pawn at (r,c), the push action
        (r,c)->(r-1,c,False) should be legal in the perspective mask (is_white=False).
        """
        game = ShogiGame()
        assert game.current_player == Color.BLACK

        obs = game.get_observation()
        legal_moves = game.get_legal_moves()
        mask = mapper.get_legal_mask_perspective(
            legal_moves, device=torch.device("cpu"), is_white=False
        )

        # Find Black's own pawns in obs channel 0
        pawn_positions = list(zip(*np.where(obs[OBS_CHANNEL_PAWN] == 1.0)))
        assert len(pawn_positions) == 9, (
            f"Expected 9 pawns in channel 0 at start, got {len(pawn_positions)}"
        )

        # All Black pawns should be at row 6 in the observation
        for r, c in pawn_positions:
            assert r == 6, f"Black pawn at obs row {r}, expected row 6"

            # The push action (r,c)->(r-1,c,False) should be legal in perspective mask
            push_move = (r, c, r - 1, c, False)
            idx = mapper.shogi_move_to_policy_index(push_move)
            assert mask[idx] == 1.0, (
                f"Perspective mask for Black should be 1 at index {idx} "
                f"for pawn push {push_move}"
            )

    def test_white_pawn_push_spatial_alignment(self, mapper):
        """After Black moves (6,6)->(5,6,False), it's White's turn. Get observation,
        find White's own pawns (channel 0, should be at row 6 in flipped obs — these
        are absolute row 2 pawns flipped). For each pawn at (r,c) in obs, the push
        action (r,c)->(r-1,c,False) should be legal in the perspective mask
        (is_white=True). Also verify the absolute-space index for the same pawn push
        is NOT set (would have caught the original bug).
        """
        game = ShogiGame()
        game.make_move((6, 6, 5, 6, False))
        assert game.current_player == Color.WHITE

        obs = game.get_observation()
        legal_moves = game.get_legal_moves()
        mask = mapper.get_legal_mask_perspective(
            legal_moves, device=torch.device("cpu"), is_white=True
        )

        # Find White's own pawns in obs channel 0 (White sees itself as current player)
        pawn_positions = list(zip(*np.where(obs[OBS_CHANNEL_PAWN] == 1.0)))
        assert len(pawn_positions) == 9, (
            f"Expected 9 pawns in channel 0 for White's obs, got {len(pawn_positions)}"
        )

        for r, c in pawn_positions:
            assert r == 6, (
                f"White pawn appears at obs row {r}, expected row 6 in flipped obs"
            )

            # The perspective push action (r,c)->(r-1,c,False) should be legal
            perspective_push = (r, c, r - 1, c, False)
            persp_idx = mapper.shogi_move_to_policy_index(perspective_push)
            assert mask[persp_idx] == 1.0, (
                f"Perspective mask for White should be 1 at index {persp_idx} "
                f"for perspective push {perspective_push}"
            )

            # The absolute-space index for the same coordinates should NOT be set
            # (r,c) in perspective corresponds to absolute (8-r, 8-c) for White.
            # The absolute push would be (8-r, 8-c) -> (8-(r-1), 8-c) = (9-r, 8-c)
            # but we use (r,c) directly as absolute — this is the bug vector.
            # The absolute move for the same perspective coords used as absolute
            # is the non-flipped version which should NOT be set in the mask.
            abs_wrong_idx = mapper.shogi_move_to_policy_index(perspective_push)
            # The perspective_push IS the perspective move; what should NOT be set
            # is the absolute move at the same raw indices WITHOUT flipping.
            # Specifically, the absolute move corresponding to this perspective position:
            abs_r, abs_c = 8 - r, 8 - c
            abs_push = (abs_r, abs_c, abs_r + 1, abs_c, False)
            abs_idx = mapper.shogi_move_to_policy_index(abs_push)
            # The absolute move index should NOT appear in the perspective mask
            # (unless it coincidentally maps to the same index, which it won't for row 6)
            if abs_idx != persp_idx:
                assert mask[abs_idx] == 0.0, (
                    f"Perspective mask for White should NOT be 1 at absolute index "
                    f"{abs_idx} for absolute move {abs_push}"
                )

    def test_mirror_symmetry_at_start(self, mapper):
        """Compare Black's obs at move 1 with White's obs at move 2 (after a neutral
        pawn push). Both should have:
          - Own king at obs (8,4) (channel 7)
          - Own pawns at row 6 (channel 0) — Black has 9, White has 9
          - Opponent king at obs (0,4) (channel 21)
        """
        # Black's observation at start
        game_black = ShogiGame()
        obs_black = game_black.get_observation()

        # After a neutral pawn push by Black, White's perspective
        game_white = ShogiGame()
        game_white.make_move((6, 4, 5, 4, False))  # Black pushes center pawn
        assert game_white.current_player == Color.WHITE
        obs_white = game_white.get_observation()

        # Both players should see their own king at (8,4)
        assert obs_black[OBS_CHANNEL_KING, 8, 4] == 1.0, (
            "Black should see own king at obs (8,4) in channel 7"
        )
        assert obs_white[OBS_CHANNEL_KING, 8, 4] == 1.0, (
            "White should see own king at obs (8,4) in channel 7"
        )

        # Both players should see their own pawns at row 6
        black_pawn_rows = set(r for r, c in zip(*np.where(obs_black[OBS_CHANNEL_PAWN] == 1.0)))
        white_pawn_rows = set(r for r, c in zip(*np.where(obs_white[OBS_CHANNEL_PAWN] == 1.0)))
        assert black_pawn_rows == {6}, (
            f"Black's pawns should all be at row 6, got rows {black_pawn_rows}"
        )
        assert white_pawn_rows == {6}, (
            f"White's pawns should all be at row 6 in perspective, got rows {white_pawn_rows}"
        )

        # Both should have 9 own pawns (after one pawn push, still 9 pawns each)
        black_pawn_count = int(np.sum(obs_black[OBS_CHANNEL_PAWN]))
        white_pawn_count = int(np.sum(obs_white[OBS_CHANNEL_PAWN]))
        assert black_pawn_count == 9, (
            f"Black should have 9 pawns in obs, got {black_pawn_count}"
        )
        assert white_pawn_count == 9, (
            f"White should have 9 pawns in obs, got {white_pawn_count}"
        )

        # Both players should see opponent king at obs (0,4)
        assert obs_black[OBS_CHANNEL_OPP_KING, 0, 4] == 1.0, (
            "Black should see opponent king at obs (0,4) in channel 21"
        )
        assert obs_white[OBS_CHANNEL_OPP_KING, 0, 4] == 1.0, (
            "White should see opponent king at obs (0,4) in channel 21"
        )

    def test_all_white_legal_moves_roundtrip(self, mapper):
        """After Black moves, get all White legal moves. Create perspective mask.
        Count of legal indices should equal count of legal moves. Every index should
        roundtrip via perspective_index_to_absolute_move(idx, is_white=True) to a
        move in the original legal_moves list.
        """
        game = ShogiGame()
        game.make_move((6, 0, 5, 0, False))  # Black moves
        assert game.current_player == Color.WHITE

        legal_moves = game.get_legal_moves()
        mask = mapper.get_legal_mask_perspective(
            legal_moves, device=torch.device("cpu"), is_white=True
        )

        legal_indices = torch.where(mask)[0]

        # Count of legal indices should equal count of legal moves
        assert len(legal_indices) == len(legal_moves), (
            f"Expected {len(legal_moves)} legal indices in perspective mask, "
            f"got {len(legal_indices)}"
        )

        # Every index should roundtrip to a move in the original legal_moves list
        legal_moves_set = set(legal_moves)
        for idx_tensor in legal_indices:
            idx = idx_tensor.item()
            absolute_move = mapper.perspective_index_to_absolute_move(
                idx, is_white=True
            )
            assert absolute_move in legal_moves_set, (
                f"Perspective index {idx} roundtripped to {absolute_move}, "
                f"which is not in legal_moves"
            )
