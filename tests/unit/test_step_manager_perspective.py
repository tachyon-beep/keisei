"""Tests for perspective-aware action mapping in StepManager.

These tests verify that PolicyOutputMapper's perspective methods produce correct
legal masks and roundtrip conversions when used with real Shogi game states.
"""

import numpy as np
import pytest
import torch

from keisei.shogi_python_reference import Color, ShogiGame
from keisei.utils.utils import PolicyOutputMapper

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def mapper():
    """Module-scoped PolicyOutputMapper (expensive to create)."""
    return PolicyOutputMapper()


class TestStepManagerPerspective:
    """Tests for perspective-aware action mapping in the training step loop."""

    def test_observation_action_alignment_initial_position(self, mapper):
        """At start position (Black's turn), Black's pawn push (6,0)->(5,0)
        should be legal in the perspective mask (is_white=False)."""
        game = ShogiGame()
        assert game.current_player == Color.BLACK

        legal_moves = game.get_legal_moves()
        # Black pawn push: from (6,0) to (5,0), no promotion
        pawn_push = (6, 0, 5, 0, False)
        assert pawn_push in legal_moves, "Black pawn push (6,0)->(5,0) should be legal"

        mask = mapper.get_legal_mask_perspective(
            legal_moves, device=torch.device("cpu"), is_white=False
        )
        idx = mapper.shogi_move_to_policy_index(pawn_push)
        assert mask[idx] == 1.0, (
            f"Perspective mask for Black should have a 1 at index {idx} "
            f"for pawn push (6,0)->(5,0)"
        )

    def test_observation_action_alignment_white_turn(self, mapper):
        """After Black moves (6,6)->(5,6), it's White's turn. White's pawn at
        absolute (2,0) appears at obs position (6,8). The perspective mask
        (is_white=True) should have a 1 at the index for perspective move
        (6,8)->(5,8), and should NOT have a 1 at the absolute index (2,0)->(3,0)."""
        game = ShogiGame()
        # Black moves pawn at (6,6) to (5,6)
        black_move = (6, 6, 5, 6, False)
        game.make_move(black_move)
        assert game.current_player == Color.WHITE

        legal_moves = game.get_legal_moves()
        # White pawn push: absolute (2,0)->(3,0), no promotion
        abs_pawn_push = (2, 0, 3, 0, False)
        assert abs_pawn_push in legal_moves, (
            "White pawn push (2,0)->(3,0) should be legal"
        )

        mask = mapper.get_legal_mask_perspective(
            legal_moves, device=torch.device("cpu"), is_white=True
        )

        # In perspective space, (2,0)->(3,0) becomes (6,8)->(5,8)
        perspective_move = (6, 8, 5, 8, False)
        perspective_idx = mapper.shogi_move_to_policy_index(perspective_move)
        assert mask[perspective_idx] == 1.0, (
            f"Perspective mask for White should have a 1 at index {perspective_idx} "
            f"for perspective move (6,8)->(5,8)"
        )

        # The absolute-space index should NOT be set in the perspective mask
        abs_idx = mapper.shogi_move_to_policy_index(abs_pawn_push)
        assert mask[abs_idx] == 0.0, (
            f"Perspective mask for White should NOT have a 1 at absolute index "
            f"{abs_idx} for move (2,0)->(3,0)"
        )

    def test_perspective_move_roundtrip_produces_valid_game_move(self, mapper):
        """After Black moves, for every legal White perspective index,
        perspective_index_to_absolute_move(idx, is_white=True) should produce
        a move that's in the original legal_moves list."""
        game = ShogiGame()
        # Black moves pawn at (6,6) to (5,6)
        black_move = (6, 6, 5, 6, False)
        game.make_move(black_move)
        assert game.current_player == Color.WHITE

        legal_moves = game.get_legal_moves()
        mask = mapper.get_legal_mask_perspective(
            legal_moves, device=torch.device("cpu"), is_white=True
        )

        # Get all indices where the perspective mask is 1
        legal_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
        assert len(legal_indices) == len(legal_moves), (
            f"Number of legal perspective indices ({len(legal_indices)}) should match "
            f"number of legal moves ({len(legal_moves)})"
        )

        for idx in legal_indices:
            abs_move = mapper.perspective_index_to_absolute_move(idx, is_white=True)
            assert abs_move in legal_moves, (
                f"Perspective index {idx} -> absolute move {abs_move} "
                f"should be in legal_moves but is not"
            )
