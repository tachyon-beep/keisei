# tests/test_sl_pipeline.py
"""Tests for the supervised learning pipeline."""

import pytest

from keisei.sl.parsers import (
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
