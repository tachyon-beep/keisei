# tests/test_sl_pipeline.py
"""Tests for the supervised learning pipeline."""

import numpy as np
import pytest

from keisei.sl.dataset import SLDataset, write_shard
from keisei.sl.parsers import (
    CSAParser,
    GameOutcome,
    GameRecord,
    ParsedMove,
    SFENParser,
)


class TestSFENParser:
    def test_supported_extensions(self):
        parser = SFENParser()
        assert ".sfen" in parser.supported_extensions()

    def test_parse_single_game(self, tmp_path):
        # SFEN format: one game per block, first line = metadata, rest = moves
        sfen_content = (
            "result:win_black\n"
            "startpos\n"
            "7g7f\n"
            "3c3d\n"
        )
        sfen_file = tmp_path / "test.sfen"
        sfen_file.write_text(sfen_content)

        parser = SFENParser()
        games = list(parser.parse(sfen_file))
        assert len(games) == 1
        assert games[0].outcome == GameOutcome.WIN_BLACK
        assert len(games[0].moves) == 2
        assert games[0].moves[0].move_usi == "7g7f"


CSA_GAME = """V2.2
N+Player1
N-Player2
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA *
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  *
P5 *  *  *  *  *  *  *  *  *
P6 *  *  *  *  *  *  *  *  *
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI *
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
+
+7776FU
-3334FU
%TORYO
"""


class TestCSAParser:
    def test_supported_extensions(self):
        parser = CSAParser()
        assert ".csa" in parser.supported_extensions()

    def test_parse_single_game(self, tmp_path):
        csa_file = tmp_path / "test.csa"
        csa_file.write_text(CSA_GAME)

        parser = CSAParser()
        games = list(parser.parse(csa_file))
        assert len(games) == 1
        # After -3334FU (White moves), it's Black's turn. %TORYO = side-to-move resigns.
        # Black resigns -> White wins. last_mover="-" -> WIN_WHITE.
        assert games[0].outcome == GameOutcome.WIN_WHITE
        assert len(games[0].moves) == 2
        assert games[0].metadata.get("player_black") == "Player1"
        assert games[0].metadata.get("player_white") == "Player2"

    def test_parse_csa_move_format(self, tmp_path):
        csa_file = tmp_path / "test.csa"
        csa_file.write_text(CSA_GAME)

        parser = CSAParser()
        games = list(parser.parse(csa_file))
        # CSA move "+7776FU" should be converted to USI "7g7f"
        assert games[0].moves[0].move_usi == "7g7f"
        assert games[0].moves[1].move_usi == "3c3d"


class TestSLDataset:
    def test_write_and_read_shard(self, tmp_path):
        """Write a shard with 10 positions and read them back."""
        observations = np.random.randn(10, 50 * 81).astype(np.float32)
        policy_targets = np.random.randint(0, 11259, size=10).astype(np.int64)
        value_targets = np.random.randint(0, 3, size=10).astype(np.int64)
        score_targets = np.random.randn(10).astype(np.float32)

        shard_path = tmp_path / "shard_000.bin"
        write_shard(shard_path, observations, policy_targets, value_targets, score_targets)

        dataset = SLDataset(tmp_path)
        assert len(dataset) == 10

        item = dataset[0]
        assert item["observation"].shape == (50, 9, 9)
        assert item["policy_target"].shape == ()  # scalar
        assert item["value_target"].shape == ()
        assert item["score_target"].shape == ()

    def test_multiple_shards(self, tmp_path):
        for i in range(3):
            n = 5
            write_shard(
                tmp_path / f"shard_{i:03d}.bin",
                np.random.randn(n, 50 * 81).astype(np.float32),
                np.random.randint(0, 11259, size=n).astype(np.int64),
                np.random.randint(0, 3, size=n).astype(np.int64),
                np.random.randn(n).astype(np.float32),
            )
        dataset = SLDataset(tmp_path)
        assert len(dataset) == 15
