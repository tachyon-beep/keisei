"""Tests for the SL data preparation CLI."""

import numpy as np
import pytest

from keisei.sl.dataset import SCORE_NORMALIZATION, SLDataset
from keisei.sl.parsers import GameOutcome
from keisei.sl.prepare import prepare_sl_data


@pytest.fixture
def sample_sfen_dir(tmp_path):
    """Create a directory with a small SFEN game file."""
    sfen_content = (
        "result:win_black\n"
        "startpos\n"
        "7g7f\n"
        "3c3d\n"
        "2g2f\n"
        "8c8d\n"
    )
    sfen_file = tmp_path / "games" / "test.sfen"
    sfen_file.parent.mkdir()
    sfen_file.write_text(sfen_content)
    return tmp_path / "games"


def test_prepare_creates_shards(sample_sfen_dir, tmp_path):
    output_dir = tmp_path / "processed"
    prepare_sl_data(
        game_sources=[str(sample_sfen_dir)],
        output_dir=str(output_dir),
        min_ply=2,
    )
    dataset = SLDataset(output_dir)
    # 4 moves in the game -> 4 positions encoded
    assert len(dataset) == 4


def test_prepare_filters_short_games(tmp_path):
    # Game with only 1 move — should be filtered by min_ply=2
    sfen_content = "result:win_black\nstartpos\n7g7f\n"
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    (games_dir / "short.sfen").write_text(sfen_content)

    output_dir = tmp_path / "processed"
    prepare_sl_data(
        game_sources=[str(games_dir)],
        output_dir=str(output_dir),
        min_ply=2,
    )
    dataset = SLDataset(output_dir)
    assert len(dataset) == 0


def test_prepare_filters_by_min_rating(tmp_path):
    """Games with low rating metadata should be filtered out."""
    sfen_content = (
        "result:win_black\nrating:1200\n"
        "startpos\n7g7f\n3c3d\n"
    )
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    (games_dir / "low_rated.sfen").write_text(sfen_content)

    output_dir = tmp_path / "processed"
    prepare_sl_data(
        game_sources=[str(games_dir)],
        output_dir=str(output_dir),
        min_ply=1,
        min_rating=1500,
    )
    dataset = SLDataset(output_dir)
    assert len(dataset) == 0


def test_prepare_skips_corrupted_files(tmp_path):
    """Corrupted files should be skipped, other files still processed."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()

    # Good game file
    good_content = "result:win_black\nstartpos\n7g7f\n3c3d\n"
    (games_dir / "good.sfen").write_text(good_content)

    # Corrupted file (binary garbage with .sfen extension)
    (games_dir / "bad.sfen").write_bytes(b"\x00\xff\xfe" * 100)

    output_dir = tmp_path / "processed"
    prepare_sl_data(
        game_sources=[str(games_dir)],
        output_dir=str(output_dir),
        min_ply=1,
    )
    dataset = SLDataset(output_dir)
    # Good file has 2 moves -> 2 positions; bad file skipped
    assert len(dataset) == 2


def test_prepare_handles_mixed_extensions(tmp_path):
    """Both .sfen and .csa files in the same directory should be processed."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()

    sfen_content = "result:win_black\nstartpos\n7g7f\n3c3d\n"
    (games_dir / "game1.sfen").write_text(sfen_content)

    csa_content = (
        "V2.2\nN+A\nN-B\n"
        "P1-KY-KE-GI-KI-OU-KI-GI-KE-KY\n"
        "P2 * -HI *  *  *  *  * -KA * \n"
        "P3-FU-FU-FU-FU-FU-FU-FU-FU-FU\n"
        "P4 *  *  *  *  *  *  *  *  * \n"
        "P5 *  *  *  *  *  *  *  *  * \n"
        "P6 *  *  *  *  *  *  *  *  * \n"
        "P7+FU+FU+FU+FU+FU+FU+FU+FU+FU\n"
        "P8 * +KA *  *  *  *  * +HI * \n"
        "P9+KY+KE+GI+KI+OU+KI+GI+KE+KY\n"
        "+\n+7776FU\n-3334FU\n%TORYO\n"
    )
    (games_dir / "game2.csa").write_text(csa_content)

    output_dir = tmp_path / "processed"
    prepare_sl_data(
        game_sources=[str(games_dir)],
        output_dir=str(output_dir),
        min_ply=1,
    )
    dataset = SLDataset(output_dir)
    # SFEN: 2 moves -> 2 positions; CSA: 2 moves -> 2 positions
    assert len(dataset) == 4


class TestParserRegistryDuplicateExtension:
    """M1: _build_parser_registry raises ValueError on duplicate extensions."""

    def test_duplicate_extension_raises_value_error(self):
        """Patching parser list to include a duplicate extension triggers ValueError."""
        from unittest.mock import patch

        from keisei.sl.parsers import SFENParser
        from keisei.sl.prepare import _build_parser_registry

        # Patch the parser class list so both entries are SFENParser,
        # which means ".sfen" will be registered twice.
        with patch(
            "keisei.sl.prepare.SFENParser", SFENParser
        ), patch(
            "keisei.sl.prepare.CSAParser", SFENParser
        ):
            with pytest.raises(ValueError, match="Duplicate parser for extension"):
                _build_parser_registry()


class TestValueEncodingCorrectness:
    """H5: Value encoding from side-to-move perspective (W/D/L)."""

    def _make_game_file(self, tmp_path, outcome_str, num_moves=4):
        """Create an SFEN game file with a given outcome and number of moves."""
        moves = ["7g7f", "3c3d", "2g2f", "8c8d", "6g6f", "4c4d"][:num_moves]
        content = f"result:{outcome_str}\nstartpos\n" + "\n".join(moves) + "\n"
        games_dir = tmp_path / "games"
        games_dir.mkdir(exist_ok=True)
        (games_dir / "test.sfen").write_text(content)
        return games_dir

    def test_win_black_value_encoding(self, tmp_path):
        """WIN_BLACK: black-to-move positions get value_cat=0 (W), white gets 2 (L)."""
        games_dir = self._make_game_file(tmp_path, "win_black", num_moves=4)
        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(games_dir)],
            output_dir=str(output_dir),
            min_ply=1,
        )
        dataset = SLDataset(output_dir)
        assert len(dataset) == 4

        for i in range(4):
            item = dataset[i]
            value_cat = item["value_target"].item()
            if i % 2 == 0:  # black to move
                assert value_cat == 0, f"Move {i} (black-to-move): expected W(0), got {value_cat}"
            else:  # white to move
                assert value_cat == 2, f"Move {i} (white-to-move): expected L(2), got {value_cat}"

    def test_win_white_reverses_assignment(self, tmp_path):
        """WIN_WHITE: black-to-move positions get value_cat=2 (L), white gets 0 (W)."""
        games_dir = self._make_game_file(tmp_path, "win_white", num_moves=4)
        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(games_dir)],
            output_dir=str(output_dir),
            min_ply=1,
        )
        dataset = SLDataset(output_dir)
        assert len(dataset) == 4

        for i in range(4):
            item = dataset[i]
            value_cat = item["value_target"].item()
            if i % 2 == 0:  # black to move
                assert value_cat == 2, f"Move {i} (black-to-move): expected L(2), got {value_cat}"
            else:  # white to move
                assert value_cat == 0, f"Move {i} (white-to-move): expected W(0), got {value_cat}"

    def test_draw_produces_value_cat_1_for_all(self, tmp_path):
        """DRAW: all positions get value_cat=1 regardless of side to move."""
        games_dir = self._make_game_file(tmp_path, "draw", num_moves=4)
        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(games_dir)],
            output_dir=str(output_dir),
            min_ply=1,
        )
        dataset = SLDataset(output_dir)
        assert len(dataset) == 4

        for i in range(4):
            item = dataset[i]
            value_cat = item["value_target"].item()
            assert value_cat == 1, f"Move {i}: expected Draw(1), got {value_cat}"

    def test_score_target_normalization(self, tmp_path):
        """Score targets should be normalized by SCORE_NORMALIZATION."""
        games_dir = self._make_game_file(tmp_path, "win_black", num_moves=4)
        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(games_dir)],
            output_dir=str(output_dir),
            min_ply=1,
        )
        dataset = SLDataset(output_dir)
        assert len(dataset) == 4

        for i in range(4):
            item = dataset[i]
            score = item["score_target"].item()
            is_black_to_move = i % 2 == 0
            # raw_score = 1.0 for black-to-move, -1.0 for white-to-move
            expected_raw = 1.0 if is_black_to_move else -1.0
            expected_normalized = expected_raw / SCORE_NORMALIZATION
            np.testing.assert_almost_equal(
                score, expected_normalized, decimal=5,
                err_msg=f"Move {i}: score {score} != expected {expected_normalized}"
            )

    def test_draw_score_target_is_zero(self, tmp_path):
        """DRAW positions should have score_target = 0.0."""
        games_dir = self._make_game_file(tmp_path, "draw", num_moves=4)
        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(games_dir)],
            output_dir=str(output_dir),
            min_ply=1,
        )
        dataset = SLDataset(output_dir)
        for i in range(4):
            score = dataset[i]["score_target"].item()
            assert score == 0.0, f"Move {i}: draw score should be 0.0, got {score}"
