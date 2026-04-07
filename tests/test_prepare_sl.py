"""Tests for the SL data preparation CLI."""

import json

import numpy as np
import pytest

from keisei.sl.dataset import SCORE_NORMALIZATION, SLDataset
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
    dataset = SLDataset(output_dir, allow_placeholder=True)
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
    dataset = SLDataset(output_dir, allow_placeholder=True)
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
    dataset = SLDataset(output_dir, allow_placeholder=True)
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
    dataset = SLDataset(output_dir, allow_placeholder=True)
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
    dataset = SLDataset(output_dir, allow_placeholder=True)
    # SFEN: 2 moves -> 2 positions; CSA: 2 moves -> 2 positions
    assert len(dataset) == 4


class TestCSASourceIntegration:
    """HIGH-1: CSA file → shard end-to-end through prepare_sl_data()."""

    def _make_csa_game(self, games_dir):
        """Create a minimal CSA game file with 2 moves."""
        csa_content = (
            "V2.2\nN+PlayerA\nN-PlayerB\n"
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
        (games_dir / "game.csa").write_text(csa_content)

    def test_csa_only_source_creates_shards(self, tmp_path):
        """CSA-only source directory should produce shards via prepare_sl_data()."""
        games_dir = tmp_path / "games"
        games_dir.mkdir()
        self._make_csa_game(games_dir)

        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(games_dir)],
            output_dir=str(output_dir),
            min_ply=1,
        )
        dataset = SLDataset(output_dir, allow_placeholder=True)
        # CSA game has 2 moves → 2 positions
        assert len(dataset) == 2

    def test_csa_value_encoding(self, tmp_path):
        """CSA game ending in %TORYO: side-to-move resigns, last_mover wins.

        In the test game: +7776FU, -3334FU, %TORYO
        After -3334FU, last_mover="-" (gote). %TORYO means side-to-move
        (sente, "+") resigns → last_mover (gote, "-") wins → WIN_WHITE.
        """
        games_dir = tmp_path / "games"
        games_dir.mkdir()
        self._make_csa_game(games_dir)

        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(games_dir)],
            output_dir=str(output_dir),
            min_ply=1,
        )
        dataset = SLDataset(output_dir, allow_placeholder=True)
        # WIN_WHITE: black-to-move positions get value_cat=2 (L)
        assert dataset[0]["value_target"].item() == 2
        # WIN_WHITE: white-to-move positions get value_cat=0 (W)
        assert dataset[1]["value_target"].item() == 0


class TestShardSizeBoundary:
    """HIGH-1: shard_size flush boundary correctness.

    Note: the flush check fires after each *game*, not after each *position*.
    So to trigger a mid-processing flush, we need multiple games where the
    cumulative positions cross the shard_size threshold between games.
    """

    def test_multiple_games_trigger_shard_split(self, tmp_path):
        """Two 2-move games with shard_size=2: first game fills buffer → flush → second game → flush."""
        games_dir = tmp_path / "games"
        games_dir.mkdir()
        # Two separate game files, each with 2 moves
        (games_dir / "game1.sfen").write_text("result:win_black\nstartpos\n7g7f\n3c3d\n")
        (games_dir / "game2.sfen").write_text("result:win_white\nstartpos\n2g2f\n8c8d\n")

        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(games_dir)],
            output_dir=str(output_dir),
            min_ply=1,
            shard_size=2,
        )
        shard_files = sorted(output_dir.glob("shard_*.bin"))
        assert len(shard_files) == 2, f"Expected 2 shards, got {len(shard_files)}"

        dataset = SLDataset(output_dir, allow_placeholder=True)
        assert len(dataset) == 4

    def test_single_game_exceeding_shard_size(self, tmp_path):
        """A single game with 4 moves and shard_size=2 produces 2 shards (flush per position)."""
        games_dir = tmp_path / "games"
        games_dir.mkdir()
        sfen_content = "result:win_black\nstartpos\n7g7f\n3c3d\n2g2f\n8c8d\n"
        (games_dir / "game.sfen").write_text(sfen_content)

        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(games_dir)],
            output_dir=str(output_dir),
            min_ply=1,
            shard_size=2,
        )
        # Flush fires inside the per-move loop, so shard_size is a true cap.
        # 4 positions with shard_size=2 → 2 shards of 2 positions each.
        shard_files = sorted(output_dir.glob("shard_*.bin"))
        assert len(shard_files) == 2
        dataset = SLDataset(output_dir, allow_placeholder=True)
        assert len(dataset) == 4

    def test_remainder_flushed(self, tmp_path):
        """Positions that don't fill a shard should still be flushed at the end."""
        games_dir = tmp_path / "games"
        games_dir.mkdir()
        sfen_content = "result:win_black\nstartpos\n7g7f\n"
        (games_dir / "game.sfen").write_text(sfen_content)

        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(games_dir)],
            output_dir=str(output_dir),
            min_ply=1,
            shard_size=1000,  # much larger than our 1 position
        )
        shard_files = sorted(output_dir.glob("shard_*.bin"))
        assert len(shard_files) == 1, "Remainder should be flushed even if < shard_size"
        dataset = SLDataset(output_dir, allow_placeholder=True)
        assert len(dataset) == 1


class TestDeterministicFileOrdering:
    """Regression: Path.glob() order is filesystem-dependent; prepare_sl_data must sort."""

    def test_files_processed_in_sorted_order(self, tmp_path):
        """Two game files with distinct outcomes: shard positions must reflect sorted filename order."""
        games_dir = tmp_path / "games"
        games_dir.mkdir()

        # "aaa.sfen" → win_black, "zzz.sfen" → win_white
        # If sorted, aaa comes first → positions 0,1 are win_black (value 0,2)
        # then zzz → positions 2,3 are win_white (value 2,0)
        (games_dir / "zzz.sfen").write_text("result:win_white\nstartpos\n7g7f\n3c3d\n")
        (games_dir / "aaa.sfen").write_text("result:win_black\nstartpos\n2g2f\n8c8d\n")

        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(games_dir)],
            output_dir=str(output_dir),
            min_ply=1,
        )
        dataset = SLDataset(output_dir, allow_placeholder=True)
        assert len(dataset) == 4

        # Sorted order: aaa.sfen (win_black) first, then zzz.sfen (win_white)
        # aaa.sfen positions: move 0 (black-to-move, W=0), move 1 (white-to-move, L=2)
        assert dataset[0]["value_target"].item() == 0, "aaa.sfen pos 0: black wins, black-to-move → W(0)"
        assert dataset[1]["value_target"].item() == 2, "aaa.sfen pos 1: black wins, white-to-move → L(2)"
        # zzz.sfen positions: move 0 (black-to-move, L=2), move 1 (white-to-move, W=0)
        assert dataset[2]["value_target"].item() == 2, "zzz.sfen pos 0: white wins, black-to-move → L(2)"
        assert dataset[3]["value_target"].item() == 0, "zzz.sfen pos 1: white wins, white-to-move → W(0)"


class TestStaleShardsRemoved:
    """Regression: re-running prepare_sl_data must not leave stale shards from prior runs."""

    def test_fewer_shards_second_run_removes_stale(self, tmp_path):
        """If run 1 produces 2 shards and run 2 produces 1, stale shard_001 must be gone."""
        games_dir = tmp_path / "games"
        games_dir.mkdir()
        output_dir = tmp_path / "processed"

        # Run 1: 4 positions with shard_size=2 → 2 shards
        (games_dir / "game.sfen").write_text(
            "result:win_black\nstartpos\n7g7f\n3c3d\n2g2f\n8c8d\n"
        )
        prepare_sl_data(
            game_sources=[str(games_dir)],
            output_dir=str(output_dir),
            min_ply=1,
            shard_size=2,
        )
        assert len(list(output_dir.glob("shard_*.bin"))) == 2

        # Run 2: 1 position → 1 shard
        (games_dir / "game.sfen").write_text(
            "result:win_black\nstartpos\n7g7f\n"
        )
        prepare_sl_data(
            game_sources=[str(games_dir)],
            output_dir=str(output_dir),
            min_ply=1,
            shard_size=1000,
        )
        shard_files = list(output_dir.glob("shard_*.bin"))
        assert len(shard_files) == 1, (
            f"Stale shards from prior run remain: {[f.name for f in shard_files]}"
        )

        dataset = SLDataset(output_dir, allow_placeholder=True)
        assert len(dataset) == 1


class TestNonexistentSourcePath:
    """HIGH-1: prepare_sl_data with a non-existent source path."""

    def test_nonexistent_source_produces_zero_shards(self, tmp_path):
        """A source path that doesn't exist should produce zero shards, not crash."""
        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(tmp_path / "no_such_dir")],
            output_dir=str(output_dir),
            min_ply=1,
        )
        shard_files = list(output_dir.glob("shard_*.bin"))
        assert len(shard_files) == 0


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
        dataset = SLDataset(output_dir, allow_placeholder=True)
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
        dataset = SLDataset(output_dir, allow_placeholder=True)
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
        dataset = SLDataset(output_dir, allow_placeholder=True)
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
        dataset = SLDataset(output_dir, allow_placeholder=True)
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
        dataset = SLDataset(output_dir, allow_placeholder=True)
        for i in range(4):
            score = dataset[i]["score_target"].item()
            assert score == 0.0, f"Move {i}: draw score should be 0.0, got {score}"


class TestShardMetadataGuard:
    """Tests for shard_meta.json placeholder guard."""

    def test_prepare_writes_shard_meta_json(self, sample_sfen_dir, tmp_path):
        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(sample_sfen_dir)],
            output_dir=str(output_dir),
            min_ply=2,
        )
        meta_path = output_dir / "shard_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["placeholder"] is True
        assert meta["num_shards"] >= 1
        assert meta["num_games"] >= 1

    def test_placeholder_guard_rejects_without_flag(self, sample_sfen_dir, tmp_path):
        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(sample_sfen_dir)],
            output_dir=str(output_dir),
            min_ply=2,
        )
        with pytest.raises(ValueError, match="placeholder"):
            SLDataset(output_dir)

    def test_placeholder_guard_allows_with_flag(self, sample_sfen_dir, tmp_path):
        output_dir = tmp_path / "processed"
        prepare_sl_data(
            game_sources=[str(sample_sfen_dir)],
            output_dir=str(output_dir),
            min_ply=2,
        )
        dataset = SLDataset(output_dir, allow_placeholder=True)
        assert len(dataset) > 0

    def test_no_metadata_file_allows_loading(self, tmp_path):
        """SLDataset loads normally when shard_meta.json is absent."""
        # Empty dir with no shards and no meta — should not raise
        dataset = SLDataset(tmp_path)
        assert len(dataset) == 0
