# tests/test_sl_pipeline.py
"""Tests for the supervised learning pipeline."""

import pytest

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
