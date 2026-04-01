"""Tests for the SL data preparation CLI."""

import numpy as np
import pytest

from keisei.sl.dataset import SLDataset
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
