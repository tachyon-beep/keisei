# tests/test_sl_pipeline.py
"""Tests for the supervised learning pipeline."""

import numpy as np
import pytest
import torch

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

    def test_parse_multi_game_file(self, tmp_path):
        """Multiple games separated by blank lines."""
        sfen_content = (
            "result:win_black\nstartpos\n7g7f\n3c3d\n"
            "\n"
            "result:win_white\nstartpos\n2g2f\n8c8d\n"
        )
        sfen_file = tmp_path / "multi.sfen"
        sfen_file.write_text(sfen_content)

        parser = SFENParser()
        games = list(parser.parse(sfen_file))
        assert len(games) == 2
        assert games[0].outcome == GameOutcome.WIN_BLACK
        assert games[1].outcome == GameOutcome.WIN_WHITE

    def test_unknown_outcome_skipped(self, tmp_path):
        """Games with unrecognized result are silently skipped."""
        sfen_content = (
            "result:unknown_value\nstartpos\n7g7f\n"
            "\n"
            "result:win_black\nstartpos\n2g2f\n"
        )
        sfen_file = tmp_path / "mixed.sfen"
        sfen_file.write_text(sfen_content)

        parser = SFENParser()
        games = list(parser.parse(sfen_file))
        assert len(games) == 1
        assert games[0].outcome == GameOutcome.WIN_BLACK


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
        assert games[0].outcome == GameOutcome.WIN_WHITE
        assert len(games[0].moves) == 2
        assert games[0].metadata.get("player_black") == "Player1"
        assert games[0].metadata.get("player_white") == "Player2"

    def test_parse_csa_move_format(self, tmp_path):
        csa_file = tmp_path / "test.csa"
        csa_file.write_text(CSA_GAME)

        parser = CSAParser()
        games = list(parser.parse(csa_file))
        assert games[0].moves[0].move_usi == "7g7f"
        assert games[0].moves[1].move_usi == "3c3d"

    def test_time_up_result(self, tmp_path):
        """Games ending by %TIME_UP should award win to last mover."""
        csa = CSA_GAME.replace("%TORYO", "%TIME_UP")
        csa_file = tmp_path / "timeout.csa"
        csa_file.write_text(csa)

        parser = CSAParser()
        games = list(parser.parse(csa_file))
        assert len(games) == 1
        assert games[0].outcome == GameOutcome.WIN_WHITE

    def test_sennichite_result(self, tmp_path):
        """%SENNICHITE is a draw."""
        csa = CSA_GAME.replace("%TORYO", "%SENNICHITE")
        csa_file = tmp_path / "draw.csa"
        csa_file.write_text(csa)

        parser = CSAParser()
        games = list(parser.parse(csa_file))
        assert len(games) == 1
        assert games[0].outcome == GameOutcome.DRAW

    def test_chudan_skipped(self, tmp_path):
        """%CHUDAN (interrupted) games are skipped entirely."""
        csa = CSA_GAME.replace("%TORYO", "%CHUDAN")
        csa_file = tmp_path / "interrupted.csa"
        csa_file.write_text(csa)

        parser = CSAParser()
        games = list(parser.parse(csa_file))
        assert len(games) == 0

    def test_drop_move(self, tmp_path):
        """CSA drop move like +0055FU should convert to USI P*5e."""
        csa = CSA_GAME.replace("+7776FU\n-3334FU", "+0055FU")
        # Replace the result so it's still parseable (Black drops, then resigns)
        csa_file = tmp_path / "drop.csa"
        csa_file.write_text(csa)

        parser = CSAParser()
        games = list(parser.parse(csa_file))
        assert len(games) == 1
        assert games[0].moves[0].move_usi == "P*5e"


class TestSLDataset:
    def test_write_and_read_shard(self, tmp_path):
        """Write a shard with 10 positions and read them back."""
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((10, 50 * 81)).astype(np.float32)
        policy_targets = rng.integers(0, 11259, size=10).astype(np.int64)
        value_targets = rng.integers(0, 3, size=10).astype(np.int64)
        score_targets = rng.standard_normal(10).astype(np.float32)

        shard_path = tmp_path / "shard_000.bin"
        write_shard(shard_path, observations, policy_targets, value_targets, score_targets)

        dataset = SLDataset(tmp_path)
        assert len(dataset) == 10

        item = dataset[0]
        assert item["observation"].shape == (50, 9, 9)
        assert item["policy_target"].shape == ()
        assert item["value_target"].shape == ()
        assert item["score_target"].shape == ()

    def test_round_trip_values(self, tmp_path):
        """Verify exact value preservation through write/read cycle."""
        rng = np.random.default_rng(123)
        observations = rng.standard_normal((5, 50 * 81)).astype(np.float32)
        policy_targets = np.array([0, 100, 5000, 11258, 42], dtype=np.int64)
        value_targets = np.array([0, 1, 2, 0, 1], dtype=np.int64)
        score_targets = np.array([0.5, -0.3, 0.0, 1.0, -1.0], dtype=np.float32)

        write_shard(tmp_path / "shard_000.bin", observations, policy_targets,
                    value_targets, score_targets)

        dataset = SLDataset(tmp_path)
        for i in range(5):
            item = dataset[i]
            assert item["policy_target"].item() == policy_targets[i]
            assert item["value_target"].item() == value_targets[i]
            np.testing.assert_almost_equal(
                item["score_target"].item(), score_targets[i], decimal=6
            )

    def test_multiple_shards(self, tmp_path):
        rng = np.random.default_rng(99)
        for i in range(3):
            n = 5
            write_shard(
                tmp_path / f"shard_{i:03d}.bin",
                rng.standard_normal((n, 50 * 81)).astype(np.float32),
                rng.integers(0, 11259, size=n).astype(np.int64),
                rng.integers(0, 3, size=n).astype(np.int64),
                rng.standard_normal(n).astype(np.float32),
            )
        dataset = SLDataset(tmp_path)
        assert len(dataset) == 15

    def test_cross_shard_boundary_access(self, tmp_path):
        """Access items at shard boundaries to catch off-by-one in bisect."""
        rng = np.random.default_rng(77)
        for i in range(3):
            n = 5
            policy = np.full(n, i * 100 + np.arange(n), dtype=np.int64)
            write_shard(
                tmp_path / f"shard_{i:03d}.bin",
                rng.standard_normal((n, 50 * 81)).astype(np.float32),
                policy,
                rng.integers(0, 3, size=n).astype(np.int64),
                rng.standard_normal(n).astype(np.float32),
            )
        dataset = SLDataset(tmp_path)
        # Last item of shard 0
        _ = dataset[4]
        # First item of shard 1
        _ = dataset[5]
        # Last item of shard 2
        _ = dataset[14]
        # Out of range
        with pytest.raises(IndexError):
            _ = dataset[15]


class TestSLTrainer:
    @pytest.fixture
    def small_model(self):
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        return SEResNetModel(params)

    def test_train_one_epoch(self, small_model, tmp_path):
        from keisei.sl.trainer import SLConfig, SLTrainer

        n = 16
        rng = np.random.default_rng(42)
        write_shard(
            tmp_path / "shard_000.bin",
            rng.standard_normal((n, 50 * 81)).astype(np.float32),
            rng.integers(0, 11259, size=n).astype(np.int64),
            rng.integers(0, 3, size=n).astype(np.int64),
            rng.standard_normal(n).astype(np.float32),
        )

        config = SLConfig(
            data_dir=str(tmp_path), batch_size=8, learning_rate=1e-3, total_epochs=1
        )
        trainer = SLTrainer(small_model, config)

        # Capture initial weights
        initial_params = {
            name: p.clone() for name, p in small_model.named_parameters()
        }

        metrics = trainer.train_epoch()

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "score_loss" in metrics
        assert all(not np.isnan(v) for v in metrics.values()), f"NaN: {metrics}"

        # Verify at least one parameter changed (model actually trained)
        changed = any(
            not torch.equal(initial_params[name], p)
            for name, p in small_model.named_parameters()
        )
        assert changed, "No parameters changed after training epoch"

    def test_train_empty_dataset(self, small_model, tmp_path):
        """Training on empty dataset should return zero metrics without error."""
        from keisei.sl.trainer import SLConfig, SLTrainer

        config = SLConfig(
            data_dir=str(tmp_path), batch_size=8, learning_rate=1e-3, total_epochs=1
        )
        trainer = SLTrainer(small_model, config)
        metrics = trainer.train_epoch()

        assert metrics["policy_loss"] == 0.0
        assert metrics["value_loss"] == 0.0
        assert metrics["score_loss"] == 0.0
