"""Tests for DefaultActionMapper Python bindings."""
import pytest
from shogi_gym import DefaultActionMapper


class TestDefaultActionMapper:
    def setup_method(self):
        self.mapper = DefaultActionMapper()

    def test_action_space_size(self):
        assert self.mapper.action_space_size == 13_527

    def test_board_move_roundtrip(self):
        idx = self.mapper.encode_board_move(from_sq=0, to_sq=1, promote=False, is_white=False)
        decoded = self.mapper.decode(idx, is_white=False)
        assert decoded["type"] == "board"
        assert decoded["from_sq"] == 0
        assert decoded["to_sq"] == 1
        assert decoded["promote"] is False

    def test_board_move_with_promotion(self):
        idx = self.mapper.encode_board_move(from_sq=10, to_sq=1, promote=True, is_white=False)
        decoded = self.mapper.decode(idx, is_white=False)
        assert decoded["type"] == "board"
        assert decoded["promote"] is True

    def test_drop_move_roundtrip(self):
        idx = self.mapper.encode_drop_move(to_sq=40, piece_type_idx=0, is_white=False)
        decoded = self.mapper.decode(idx, is_white=False)
        assert decoded["type"] == "drop"
        assert decoded["to_sq"] == 40
        assert decoded["piece_type_idx"] == 0

    def test_all_drop_indices_in_range(self):
        for sq in range(81):
            for pt in range(7):
                idx = self.mapper.encode_drop_move(to_sq=sq, piece_type_idx=pt, is_white=False)
                assert 12_960 <= idx < 13_527

    def test_perspective_flip_board_move(self):
        idx_black = self.mapper.encode_board_move(0, 1, False, is_white=False)
        idx_white = self.mapper.encode_board_move(80, 79, False, is_white=True)
        assert idx_black == idx_white

    def test_decode_out_of_range(self):
        with pytest.raises(ValueError):
            self.mapper.decode(13_527, is_white=False)

    def test_invalid_square(self):
        with pytest.raises(ValueError):
            self.mapper.encode_board_move(81, 0, False, is_white=False)

    def test_invalid_piece_type(self):
        with pytest.raises(ValueError):
            self.mapper.encode_drop_move(0, 7, is_white=False)
