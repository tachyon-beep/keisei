"""Tests for PolicyOutputMapper roundtrip mapping between Shogi moves and
policy network output indices."""

import pytest

from keisei.utils.utils import PolicyOutputMapper
from keisei.shogi.shogi_core_definitions import PieceType, get_unpromoted_types

EXPECTED_TOTAL_ACTIONS = 13527


@pytest.fixture(scope="module")
def mapper():
    """Module-scoped PolicyOutputMapper (expensive to create)."""
    return PolicyOutputMapper()


class TestTotalActions:
    """Tests for the total action count."""

    def test_total_actions_is_13527(self, mapper):
        """The total number of mapped actions must be exactly 13527."""
        assert mapper.get_total_actions() == EXPECTED_TOTAL_ACTIONS, (
            f"Expected {EXPECTED_TOTAL_ACTIONS} total actions, "
            f"got {mapper.get_total_actions()}"
        )


class TestRoundtrip:
    """Tests for move <-> index roundtrip mapping."""

    def test_roundtrip_board_move_no_promotion(self, mapper):
        """A board move without promotion should survive a roundtrip mapping."""
        move = (0, 0, 1, 1, False)  # from (0,0) to (1,1), no promotion
        idx = mapper.shogi_move_to_policy_index(move)
        recovered = mapper.policy_index_to_shogi_move(idx)
        assert recovered == move, (
            f"Roundtrip failed: {move} -> idx {idx} -> {recovered}"
        )

    def test_roundtrip_board_move_with_promotion(self, mapper):
        """A board move with promotion should survive a roundtrip mapping."""
        move = (2, 3, 0, 3, True)  # from (2,3) to (0,3), with promotion
        idx = mapper.shogi_move_to_policy_index(move)
        recovered = mapper.policy_index_to_shogi_move(idx)
        assert recovered == move, (
            f"Roundtrip failed: {move} -> idx {idx} -> {recovered}"
        )

    def test_roundtrip_drop_move(self, mapper):
        """A drop move should survive a roundtrip mapping."""
        move = (None, None, 4, 4, PieceType.PAWN)  # Drop pawn at (4,4)
        idx = mapper.shogi_move_to_policy_index(move)
        recovered = mapper.policy_index_to_shogi_move(idx)
        assert recovered == move, (
            f"Roundtrip failed: {move} -> idx {idx} -> {recovered}"
        )

    def test_roundtrip_all_drop_piece_types(self, mapper):
        """Roundtrip should work for every droppable piece type."""
        for piece_type in get_unpromoted_types():
            move = (None, None, 0, 0, piece_type)
            idx = mapper.shogi_move_to_policy_index(move)
            recovered = mapper.policy_index_to_shogi_move(idx)
            assert recovered == move, (
                f"Roundtrip failed for {piece_type.name}: "
                f"{move} -> idx {idx} -> {recovered}"
            )


class TestAllIndicesValid:
    """Tests for complete coverage and bijectivity of the mapping."""

    def test_all_indices_map_to_valid_moves(self, mapper):
        """Every index from 0 to total-1 should map to a valid move."""
        total = mapper.get_total_actions()
        for idx in range(total):
            move = mapper.policy_index_to_shogi_move(idx)
            assert move is not None, f"Index {idx} mapped to None"
            assert len(move) == 5, (
                f"Index {idx} mapped to move with {len(move)} elements, expected 5"
            )

    def test_no_duplicate_indices(self, mapper):
        """The mapping must be bijective: no two different moves share an index."""
        total = mapper.get_total_actions()
        seen_moves = set()
        for idx in range(total):
            move = mapper.policy_index_to_shogi_move(idx)
            assert move not in seen_moves, (
                f"Duplicate move found at index {idx}: {move}"
            )
            seen_moves.add(move)
        assert len(seen_moves) == total


class TestMoveStructure:
    """Tests for the structure of board moves and drop moves."""

    def test_board_moves_have_correct_structure(self, mapper):
        """Board moves should be (from_r, from_c, to_r, to_c, bool)."""
        # Pick a known board move (first entry is a board move by construction)
        move = mapper.idx_to_move[0]
        from_r, from_c, to_r, to_c, promote = move
        assert isinstance(from_r, int), f"from_r type is {type(from_r)}, expected int"
        assert isinstance(from_c, int), f"from_c type is {type(from_c)}, expected int"
        assert isinstance(to_r, int), f"to_r type is {type(to_r)}, expected int"
        assert isinstance(to_c, int), f"to_c type is {type(to_c)}, expected int"
        assert isinstance(promote, bool), (
            f"promote type is {type(promote)}, expected bool"
        )
        # Coordinates should be on the 9x9 board
        for coord in [from_r, from_c, to_r, to_c]:
            assert 0 <= coord <= 8, f"Coordinate {coord} out of 0..8 range"

    def test_drop_moves_have_correct_structure(self, mapper):
        """Drop moves should be (None, None, to_r, to_c, PieceType)."""
        # Find the first drop move in the index
        drop_move = None
        for move in mapper.idx_to_move:
            if move[0] is None:
                drop_move = move
                break
        assert drop_move is not None, "No drop moves found in mapping"

        none1, none2, to_r, to_c, piece_type = drop_move
        assert none1 is None, f"from_r should be None for drop, got {none1}"
        assert none2 is None, f"from_c should be None for drop, got {none2}"
        assert isinstance(to_r, int), f"to_r type is {type(to_r)}, expected int"
        assert isinstance(to_c, int), f"to_c type is {type(to_c)}, expected int"
        assert isinstance(piece_type, PieceType), (
            f"piece_type is {type(piece_type)}, expected PieceType"
        )
        assert 0 <= to_r <= 8
        assert 0 <= to_c <= 8


class TestBoundaryAndConsistency:
    """Tests for boundary conditions and internal consistency."""

    def test_invalid_index_raises_index_error(self, mapper):
        """Accessing an out-of-bounds index should raise IndexError."""
        with pytest.raises(IndexError):
            mapper.policy_index_to_shogi_move(EXPECTED_TOTAL_ACTIONS)
        with pytest.raises(IndexError):
            mapper.policy_index_to_shogi_move(-1)

    def test_idx_to_move_length_matches_total(self, mapper):
        """idx_to_move list length must equal get_total_actions()."""
        assert len(mapper.idx_to_move) == mapper.get_total_actions(), (
            f"idx_to_move length {len(mapper.idx_to_move)} != "
            f"get_total_actions() {mapper.get_total_actions()}"
        )

    def test_all_moves_are_valid_5_tuples(self, mapper):
        """Every move in idx_to_move must be a tuple of length 5."""
        for idx, move in enumerate(mapper.idx_to_move):
            assert isinstance(move, tuple), (
                f"Move at index {idx} is {type(move).__name__}, expected tuple"
            )
            assert len(move) == 5, (
                f"Move at index {idx} has length {len(move)}, expected 5"
            )
