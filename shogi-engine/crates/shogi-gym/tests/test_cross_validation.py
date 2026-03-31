"""Cross-validate Rust DefaultActionMapper against Python PolicyOutputMapper.

This test ensures bit-for-bit encoding parity between the Rust and Python
action mappers. A divergence here would produce silent training bugs.
"""
import sys

sys.path.insert(0, "/home/john/keisei")

import pytest
from shogi_gym import DefaultActionMapper
from keisei.utils.utils import PolicyOutputMapper
from keisei.shogi_python_reference.shogi_core_definitions import (
    PieceType,
    get_unpromoted_types,
)


class TestCrossValidation:
    def setup_method(self):
        self.rust_mapper = DefaultActionMapper()
        self.python_mapper = PolicyOutputMapper()

    def test_action_space_sizes_match(self):
        assert self.rust_mapper.action_space_size == self.python_mapper.get_total_actions()

    def test_all_board_moves_match(self):
        """Compare all 12,960 board move indices between Rust and Python."""
        mismatches = []
        for from_row in range(9):
            for from_col in range(9):
                from_sq = from_row * 9 + from_col
                for to_row in range(9):
                    for to_col in range(9):
                        if from_row == to_row and from_col == to_col:
                            continue
                        to_sq = to_row * 9 + to_col
                        for promote in [False, True]:
                            rust_idx = self.rust_mapper.encode_board_move(
                                from_sq, to_sq, promote, is_white=False
                            )
                            py_move = (from_row, from_col, to_row, to_col, promote)
                            py_idx = self.python_mapper.shogi_move_to_policy_index(
                                py_move
                            )
                            if rust_idx != py_idx:
                                mismatches.append(
                                    f"board ({from_sq}->{to_sq}, promo={promote}): "
                                    f"rust={rust_idx}, python={py_idx}"
                                )
        assert not mismatches, (
            f"{len(mismatches)} mismatches found. First 10:\n"
            + "\n".join(mismatches[:10])
        )

    def test_all_drop_moves_match(self):
        """Compare all 567 drop move indices between Rust and Python."""
        hand_piece_types = get_unpromoted_types()

        mismatches = []
        for to_row in range(9):
            for to_col in range(9):
                to_sq = to_row * 9 + to_col
                for pt_idx, pt_enum in enumerate(hand_piece_types):
                    rust_idx = self.rust_mapper.encode_drop_move(
                        to_sq, pt_idx, is_white=False
                    )
                    py_move = (None, None, to_row, to_col, pt_enum)
                    py_idx = self.python_mapper.shogi_move_to_policy_index(py_move)
                    if rust_idx != py_idx:
                        mismatches.append(
                            f"drop (sq={to_sq}, type={pt_idx}/{pt_enum.name}): "
                            f"rust={rust_idx}, python={py_idx}"
                        )
        assert not mismatches, (
            f"{len(mismatches)} mismatches found. First 10:\n"
            + "\n".join(mismatches[:10])
        )

    def test_full_index_coverage(self):
        """Verify that both mappers cover the full 13,527 action space."""
        rust_indices = set()
        python_indices = set()

        # Board moves
        for from_row in range(9):
            for from_col in range(9):
                from_sq = from_row * 9 + from_col
                for to_row in range(9):
                    for to_col in range(9):
                        if from_row == to_row and from_col == to_col:
                            continue
                        to_sq = to_row * 9 + to_col
                        for promote in [False, True]:
                            rust_indices.add(
                                self.rust_mapper.encode_board_move(
                                    from_sq, to_sq, promote, is_white=False
                                )
                            )
                            py_move = (from_row, from_col, to_row, to_col, promote)
                            python_indices.add(
                                self.python_mapper.shogi_move_to_policy_index(py_move)
                            )

        # Drop moves
        hand_piece_types = get_unpromoted_types()
        for to_row in range(9):
            for to_col in range(9):
                to_sq = to_row * 9 + to_col
                for pt_idx, pt_enum in enumerate(hand_piece_types):
                    rust_indices.add(
                        self.rust_mapper.encode_drop_move(
                            to_sq, pt_idx, is_white=False
                        )
                    )
                    py_move = (None, None, to_row, to_col, pt_enum)
                    python_indices.add(
                        self.python_mapper.shogi_move_to_policy_index(py_move)
                    )

        expected = set(range(13_527))
        assert rust_indices == expected, (
            f"Rust mapper missing indices: {expected - rust_indices}, "
            f"extra indices: {rust_indices - expected}"
        )
        assert python_indices == expected, (
            f"Python mapper missing indices: {expected - python_indices}, "
            f"extra indices: {python_indices - expected}"
        )

    def test_perspective_flip_board_moves(self):
        """Verify that the Rust perspective flip matches Python's flip_move."""
        # Test a representative sample of board moves
        test_cases = [
            (0, 80, False),  # corner to corner
            (40, 41, True),  # center moves
            (76, 4, False),  # near-king squares
            (0, 1, False),   # adjacent squares
            (8, 72, True),   # diagonal corners with promotion
        ]
        for from_sq, to_sq, promote in test_cases:
            # Rust: white perspective flips internally
            rust_white = self.rust_mapper.encode_board_move(
                from_sq, to_sq, promote, is_white=True
            )
            # Python: manually flip, then encode
            flipped_from = 80 - from_sq
            flipped_to = 80 - to_sq
            fr, fc = flipped_from // 9, flipped_from % 9
            tr, tc = flipped_to // 9, flipped_to % 9
            py_flipped = (fr, fc, tr, tc, promote)
            py_white = self.python_mapper.shogi_move_to_policy_index(py_flipped)
            assert rust_white == py_white, (
                f"perspective flip mismatch for ({from_sq}->{to_sq}, promo={promote}): "
                f"rust_white={rust_white}, python_flipped={py_white}"
            )

    def test_perspective_flip_drop_moves(self):
        """Verify that the Rust perspective flip for drops matches Python's flip_move."""
        hand_piece_types = get_unpromoted_types()
        test_cases = [
            (0, 0),   # top-left corner, pawn
            (40, 3),  # center, silver
            (80, 6),  # bottom-right corner, rook
        ]
        for to_sq, pt_idx in test_cases:
            pt_enum = hand_piece_types[pt_idx]
            rust_white = self.rust_mapper.encode_drop_move(
                to_sq, pt_idx, is_white=True
            )
            flipped_to = 80 - to_sq
            fr, fc = flipped_to // 9, flipped_to % 9
            py_flipped = (None, None, fr, fc, pt_enum)
            py_white = self.python_mapper.shogi_move_to_policy_index(py_flipped)
            assert rust_white == py_white, (
                f"drop perspective flip mismatch for (sq={to_sq}, type={pt_idx}): "
                f"rust_white={rust_white}, python_flipped={py_white}"
            )
