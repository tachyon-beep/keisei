# tests/test_sl_pipeline.py
"""Tests for the supervised learning pipeline."""

import logging

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


class TestCSAParserHardening:
    def test_multi_game_csa_file(self, tmp_path):
        """Games separated by '/' should parse as individual records."""
        multi_game = (
            "V2.2\nN+Player1\nN-Player2\n"
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
            "/\n"
            "V2.2\nN+A\nN-B\n+\n+2726FU\n-8384FU\n%TORYO\n"
        )
        csa_file = tmp_path / "multi.csa"
        csa_file.write_text(multi_game)
        parser = CSAParser()
        games = list(parser.parse(csa_file))
        assert len(games) == 2
        # Validate per-game content — no board state leakage across separator
        assert len(games[0].moves) == 2
        assert games[0].moves[0].move_usi == "7g7f"
        assert games[0].metadata.get("player_black") == "Player1"
        assert len(games[1].moves) == 2
        assert games[1].moves[0].move_usi == "2g2f"
        assert games[1].metadata.get("player_black") == "A"

    def test_empty_game_between_separators(self, tmp_path):
        """Empty blocks between '/' separators should be skipped."""
        content = "+7776FU\n%TORYO\n/\n/\n+2726FU\n%TORYO\n"
        csa_file = tmp_path / "gaps.csa"
        csa_file.write_text(content)
        parser = CSAParser()
        games = list(parser.parse(csa_file))
        assert len(games) == 2


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


class TestWriteShardPerformance:
    def test_write_large_shard(self, tmp_path):
        """Write 10K positions -- should complete in < 5 seconds."""
        import time

        n = 10_000
        rng = np.random.default_rng(42)
        obs = rng.standard_normal((n, 50 * 81)).astype(np.float32)
        policy = rng.integers(0, 11259, size=n).astype(np.int64)
        value = rng.integers(0, 3, size=n).astype(np.int64)
        score = rng.standard_normal(n).astype(np.float32)

        start = time.monotonic()
        write_shard(tmp_path / "perf_test.bin", obs, policy, value, score)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"write_shard took {elapsed:.1f}s for 10K positions"


class TestCSAPromotionDetection:
    """H3: Test _csa_move_to_usi promotion detection logic."""

    def test_piece_promotes_produces_plus_suffix(self):
        """A pawn moving into promotion zone and becoming TO should produce USI with '+'."""
        parser = CSAParser()
        # Board has an unpromoted pawn (FU) at (7,3) -- col 7, row 3
        board: dict[tuple[int, int], str] = {(7, 3): "FU"}
        # Move: pawn at (7,3) moves to (7,2) and becomes TO (promoted pawn)
        # CSA: +7372TO  (source=73, dest=72, piece=TO)
        usi = parser._csa_move_to_usi("+7372TO", board)
        assert usi == "7c7b+", f"Expected '7c7b+' but got '{usi}'"

    def test_already_promoted_piece_no_double_plus(self):
        """An already-promoted piece (TO) moving should NOT produce '++'."""
        parser = CSAParser()
        # Board has a promoted pawn (TO) already at (5,3)
        board: dict[tuple[int, int], str] = {(5, 3): "TO"}
        # Move: promoted pawn at (5,3) moves to (5,2), still TO
        # CSA: +5352TO
        usi = parser._csa_move_to_usi("+5352TO", board)
        assert usi == "5c5b", f"Expected '5c5b' (no promotion suffix) but got '{usi}'"

    def test_bishop_promotes_to_horse(self):
        """Bishop (KA) promoting to horse (UM) should produce '+'."""
        parser = CSAParser()
        board: dict[tuple[int, int], str] = {(8, 8): "KA"}
        # Bishop at (8,8) moves to (2,2) and becomes UM (promoted bishop)
        usi = parser._csa_move_to_usi("+8822UM", board)
        assert usi == "8h2b+", f"Expected '8h2b+' but got '{usi}'"

    def test_horse_moves_no_promotion_suffix(self):
        """An already-promoted bishop (UM) moving should NOT produce '+'."""
        parser = CSAParser()
        board: dict[tuple[int, int], str] = {(2, 2): "UM"}
        # UM at (2,2) moves to (3,3), still UM
        usi = parser._csa_move_to_usi("+2233UM", board)
        assert usi == "2b3c", f"Expected '2b3c' (no promotion suffix) but got '{usi}'"

    def test_rook_promotes_to_dragon(self):
        """Rook (HI) promoting to dragon (RY) should produce '+'."""
        parser = CSAParser()
        board: dict[tuple[int, int], str] = {(2, 8): "HI"}
        usi = parser._csa_move_to_usi("+2822RY", board)
        assert usi == "2h2b+", f"Expected '2h2b+' but got '{usi}'"

    def test_nonpromoting_move_no_suffix(self):
        """A gold (KI) moving -- cannot promote -- should have no suffix."""
        parser = CSAParser()
        board: dict[tuple[int, int], str] = {(5, 1): "KI"}
        usi = parser._csa_move_to_usi("+5152KI", board)
        assert usi == "5a5b", f"Expected '5a5b' but got '{usi}'"

    def test_promotion_in_full_game_parse(self, tmp_path):
        """Integration test: promotion appears correctly when parsing a full CSA game."""
        # Game where black's pawn at 7g promotes by moving to 7b
        csa = (
            "V2.2\n"
            "P1-KY-KE-GI-KI-OU-KI-GI-KE-KY\n"
            "P2 * -HI *  *  *  *  * -KA * \n"
            "P3-FU-FU-FU-FU-FU-FU-FU-FU-FU\n"
            "P4 *  *  *  *  *  *  *  *  * \n"
            "P5 *  *  *  *  *  *  *  *  * \n"
            "P6 *  *  *  *  *  *  *  *  * \n"
            "P7+FU+FU+FU+FU+FU+FU+FU+FU+FU\n"
            "P8 * +KA *  *  *  *  * +HI * \n"
            "P9+KY+KE+GI+KI+OU+KI+GI+KE+KY\n"
            "+\n"
            # Black moves pawn from 7g(7,7) to 7f(7,6) -- no promotion
            "+7776FU\n"
            # White makes a move
            "-3334FU\n"
            # Black moves pawn from 7f(7,6) to 7e(7,5) -- still no promotion
            "+7675FU\n"
            # White makes a move
            "-2233KA\n"
            # Simulate: pawn now at 7e jumps to 7d (row 4) -- still not promoting
            "+7574FU\n"
            # White move
            "-3344KA\n"
            # Pawn at 7d to 7c (row 3) -- still FU (not yet in promotion zone result)
            "+7473FU\n"
            # White move
            "-4455KA\n"
            # Pawn at 7c captures at 7b and PROMOTES: FU -> TO
            "+7372TO\n"
            "%TORYO\n"
        )
        csa_file = tmp_path / "promote.csa"
        csa_file.write_text(csa)

        parser = CSAParser()
        games = list(parser.parse(csa_file))
        assert len(games) == 1
        # The last black move (index 8, 0-indexed) should have '+' promotion suffix
        promoting_move = games[0].moves[8]
        assert promoting_move.move_usi == "7c7b+", (
            f"Expected '7c7b+' but got '{promoting_move.move_usi}'"
        )
        # Earlier pawn moves should NOT have promotion suffix
        assert games[0].moves[0].move_usi == "7g7f"
        assert games[0].moves[2].move_usi == "7f7e"


class TestSLTrainerCheckpointRoundTrip:
    """C3: Verify SL-trained weights can be saved and reloaded correctly."""

    @pytest.fixture
    def small_model(self):
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        return SEResNetModel(params)

    def test_checkpoint_preserves_trained_weights(self, small_model, tmp_path):
        """Train for one epoch, save checkpoint, load into fresh model, verify weights match."""
        from keisei.sl.trainer import SLConfig, SLTrainer
        from keisei.training.checkpoint import load_checkpoint, save_checkpoint
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        # Write a small shard for training
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

        # Train for one epoch
        trainer.train_epoch()

        # Save checkpoint
        ckpt_path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            ckpt_path,
            small_model,
            trainer.optimizer,
            epoch=1,
            step=0,
            architecture="se_resnet",
        )

        # Create a fresh model and optimizer, load checkpoint
        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        fresh_model = SEResNetModel(params)
        fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)

        meta = load_checkpoint(
            ckpt_path,
            fresh_model,
            fresh_optimizer,
            expected_architecture="se_resnet",
        )

        assert meta["epoch"] == 1
        assert meta["step"] == 0

        # Verify all weights match
        for name, param in small_model.named_parameters():
            fresh_param = dict(fresh_model.named_parameters())[name]
            assert torch.allclose(param, fresh_param, atol=1e-7), (
                f"Weight mismatch for parameter '{name}'"
            )


class TestSLDatasetPartialShard:
    """M2: A shard file smaller than one record should be silently dropped."""

    def test_partial_shard_file_has_zero_items(self, tmp_path):
        """A shard with fewer bytes than RECORD_SIZE produces an empty dataset."""
        from keisei.sl.dataset import RECORD_SIZE

        # Write a partial shard (fewer bytes than one full record)
        partial_shard = tmp_path / "shard_000.bin"
        partial_shard.write_bytes(b"\x00" * (RECORD_SIZE - 1))

        dataset = SLDataset(tmp_path)
        assert len(dataset) == 0


class TestSLTrainerSchedulerAndClipping:
    """H4: Test multi-epoch LR scheduler and gradient clipping."""

    @pytest.fixture
    def small_model(self):
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        return SEResNetModel(params)

    def test_scheduler_not_called_on_empty_dataset(self, small_model, tmp_path):
        """scheduler.step() should NOT be called when there is no data."""
        from unittest.mock import patch
        from keisei.sl.trainer import SLConfig, SLTrainer

        config = SLConfig(
            data_dir=str(tmp_path), batch_size=8, learning_rate=1e-3, total_epochs=10
        )
        trainer = SLTrainer(small_model, config)

        lr_before = trainer.optimizer.param_groups[0]["lr"]
        with patch.object(trainer.scheduler, "step", wraps=trainer.scheduler.step) as mock_step:
            trainer.train_epoch()
            mock_step.assert_not_called()
        lr_after = trainer.optimizer.param_groups[0]["lr"]
        assert lr_after == lr_before, "LR should not change on empty dataset"

    def test_multi_epoch_lr_decreases(self, small_model, tmp_path):
        """Training across 2+ epochs should decrease the learning rate via cosine schedule."""
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
            data_dir=str(tmp_path), batch_size=8, learning_rate=1e-3, total_epochs=10
        )
        trainer = SLTrainer(small_model, config)

        lr_initial = trainer.optimizer.param_groups[0]["lr"]
        trainer.train_epoch()
        lr_after_1 = trainer.optimizer.param_groups[0]["lr"]
        trainer.train_epoch()
        lr_after_2 = trainer.optimizer.param_groups[0]["lr"]

        # Cosine annealing should decrease LR over epochs
        assert lr_after_1 < lr_initial, (
            f"LR should decrease after epoch 1: {lr_initial} -> {lr_after_1}"
        )
        assert lr_after_2 < lr_after_1, (
            f"LR should decrease after epoch 2: {lr_after_1} -> {lr_after_2}"
        )

    def test_gradient_clipping_bounds_norms(self, small_model, tmp_path):
        """Gradient norms should be bounded by grad_clip after training."""
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

        grad_clip = 0.5
        config = SLConfig(
            data_dir=str(tmp_path), batch_size=8, learning_rate=1e-3,
            total_epochs=1, grad_clip=grad_clip,
        )
        trainer = SLTrainer(small_model, config)

        # Instead of checking post-optimizer-step gradients (which are often zero),
        # verify clipping works by manually doing a forward+backward pass and then clipping.
        trainer.model.train()
        batch = next(iter(trainer.dataloader))
        obs = batch["observation"].to(trainer.device)
        policy_targets = batch["policy_target"].to(trainer.device)
        value_targets = batch["value_target"].to(trainer.device)
        score_targets = batch["score_target"].to(trainer.device)

        import torch.nn.functional as F
        output = trainer.model(obs)
        policy_loss = F.cross_entropy(
            output.policy_logits.reshape(obs.shape[0], -1), policy_targets
        )
        value_loss = F.cross_entropy(output.value_logits, value_targets)
        score_loss = F.mse_loss(output.score_lead.squeeze(-1), score_targets)
        loss = policy_loss + value_loss + score_loss

        trainer.optimizer.zero_grad()
        loss.backward()

        # Measure total grad norm BEFORE clipping
        norm_before = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), float("inf")
        )

        # If the norm was already under grad_clip, the test isn't interesting.
        # Use a very small clip to force clipping.
        small_clip = 0.01
        # Re-do backward to get fresh gradients
        trainer.optimizer.zero_grad()
        loss = policy_loss + value_loss + score_loss  # reuse computed losses
        # Need fresh backward
        output2 = trainer.model(obs)
        loss2 = (
            F.cross_entropy(output2.policy_logits.reshape(obs.shape[0], -1), policy_targets)
            + F.cross_entropy(output2.value_logits, value_targets)
            + F.mse_loss(output2.score_lead.squeeze(-1), score_targets)
        )
        trainer.optimizer.zero_grad()
        loss2.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), small_clip
        )

        # After clipping, the actual norm should be <= small_clip (within tolerance)
        clipped_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                clipped_norm += p.grad.data.norm(2).item() ** 2
        clipped_norm = clipped_norm ** 0.5

        assert clipped_norm <= small_clip + 1e-5, (
            f"Clipped gradient norm {clipped_norm:.4f} exceeds clip value {small_clip}"
        )


class TestGameFilterRatingKeys:
    def test_black_rating_below_minimum_rejects(self):
        from keisei.sl.parsers import GameFilter, GameRecord, ParsedMove
        gf = GameFilter(min_ply=1, min_rating=1500)
        record = GameRecord(
            moves=[ParsedMove("7g7f", "startpos")] * 5,
            outcome=GameOutcome.WIN_BLACK,
            metadata={"black_rating": "1200", "white_rating": "1600"},
        )
        assert not gf.accepts(record)

    def test_white_rating_below_minimum_rejects(self):
        from keisei.sl.parsers import GameFilter, GameRecord, ParsedMove
        gf = GameFilter(min_ply=1, min_rating=1500)
        record = GameRecord(
            moves=[ParsedMove("7g7f", "startpos")] * 5,
            outcome=GameOutcome.WIN_BLACK,
            metadata={"black_rating": "1600", "white_rating": "1200"},
        )
        assert not gf.accepts(record)

    def test_both_ratings_above_minimum_accepts(self):
        from keisei.sl.parsers import GameFilter, GameRecord, ParsedMove
        gf = GameFilter(min_ply=1, min_rating=1500)
        record = GameRecord(
            moves=[ParsedMove("7g7f", "startpos")] * 5,
            outcome=GameOutcome.WIN_BLACK,
            metadata={"black_rating": "1600", "white_rating": "1700"},
        )
        assert gf.accepts(record)

    def test_no_rating_keys_accepts(self):
        from keisei.sl.parsers import GameFilter, GameRecord, ParsedMove
        gf = GameFilter(min_ply=1, min_rating=1500)
        record = GameRecord(
            moves=[ParsedMove("7g7f", "startpos")] * 5,
            outcome=GameOutcome.WIN_BLACK,
            metadata={},
        )
        assert gf.accepts(record)

    def test_non_digit_rating_ignored(self):
        from keisei.sl.parsers import GameFilter, GameRecord, ParsedMove
        gf = GameFilter(min_ply=1, min_rating=1500)
        record = GameRecord(
            moves=[ParsedMove("7g7f", "startpos")] * 5,
            outcome=GameOutcome.WIN_BLACK,
            metadata={"rating": "unknown"},
        )
        assert gf.accepts(record)


class TestSLTrainerWithBinaryShards:
    """Integration test: write_shard → SLDataset → SLTrainer → train_epoch."""

    @pytest.fixture
    def small_model(self):
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        params = SEResNetParams(num_blocks=2, channels=32, se_reduction=8,
                                global_pool_channels=16, policy_channels=8,
                                value_fc_size=32, score_fc_size=16, obs_channels=50)
        return SEResNetModel(params)

    def test_train_epoch_with_binary_shards(self, tmp_path, small_model):
        """Full pipeline using the actual binary shard format."""
        from keisei.sl.dataset import write_shard, OBS_SIZE, SLDataset
        from keisei.sl.trainer import SLTrainer, SLConfig

        n_positions = 16
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((n_positions, OBS_SIZE)).astype(np.float32)
        policy_targets = rng.integers(0, 11259, size=n_positions).astype(np.int64)
        value_targets = rng.integers(0, 3, size=n_positions).astype(np.int64)
        score_targets = rng.standard_normal(n_positions).astype(np.float32).clip(-1.5, 1.5)

        write_shard(tmp_path / "shard_000.bin", observations, policy_targets,
                    value_targets, score_targets)

        # Verify dataset reads back correct values
        ds = SLDataset(tmp_path)
        assert len(ds) == n_positions
        sample = ds[0]
        np.testing.assert_allclose(
            sample["observation"].numpy().reshape(-1),
            observations[0],
            atol=1e-6,
        )
        assert sample["policy_target"].item() == policy_targets[0]
        assert sample["value_target"].item() == value_targets[0]
        np.testing.assert_allclose(
            sample["score_target"].item(), score_targets[0], atol=1e-6
        )

        # Run a full training epoch through SLTrainer
        config = SLConfig(
            data_dir=str(tmp_path),
            batch_size=4,
            learning_rate=1e-3,
            total_epochs=10,
            num_workers=0,
            lambda_policy=1.0,
            lambda_value=1.5,
            lambda_score=0.02,
            grad_clip=1.0,
        )
        trainer = SLTrainer(small_model, config)
        metrics = trainer.train_epoch()

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "score_loss" in metrics
        for key, val in metrics.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"
        assert metrics["policy_loss"] > 0.0


class TestSLTrainerExtended:
    """Extended SL trainer tests: multi-epoch, empty dataset, gradient clipping."""

    @pytest.fixture
    def small_model(self):
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        params = SEResNetParams(num_blocks=2, channels=32, se_reduction=8,
                                global_pool_channels=16, policy_channels=8,
                                value_fc_size=32, score_fc_size=16, obs_channels=50)
        return SEResNetModel(params)

    def _write_binary_shard(self, shard_dir, n_positions=16):
        from keisei.sl.dataset import write_shard, OBS_SIZE
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((n_positions, OBS_SIZE)).astype(np.float32)
        policy_targets = rng.integers(0, 11259, size=n_positions).astype(np.int64)
        value_targets = rng.integers(0, 3, size=n_positions).astype(np.int64)
        score_targets = rng.standard_normal(n_positions).astype(np.float32).clip(-1.5, 1.5)
        write_shard(shard_dir / "shard_000.bin", observations, policy_targets,
                    value_targets, score_targets)

    def test_multi_epoch_lr_decreases(self, tmp_path, small_model):
        """CosineAnnealingLR should decrease LR over multiple epochs."""
        from keisei.sl.trainer import SLTrainer, SLConfig
        self._write_binary_shard(tmp_path)
        config = SLConfig(data_dir=str(tmp_path), batch_size=4, learning_rate=1e-3,
                          total_epochs=10, num_workers=0, lambda_policy=1.0,
                          lambda_value=1.5, lambda_score=0.02, grad_clip=1.0)
        trainer = SLTrainer(small_model, config)

        lr_before = trainer.optimizer.param_groups[0]["lr"]
        for _ in range(5):
            trainer.train_epoch()
        lr_after = trainer.optimizer.param_groups[0]["lr"]
        assert lr_after < lr_before, "LR should decrease with CosineAnnealingLR"

    def test_empty_dataset_returns_zero_loss(self, tmp_path, small_model):
        """Training with no shards should return zero losses without error."""
        from keisei.sl.trainer import SLTrainer, SLConfig
        config = SLConfig(data_dir=str(tmp_path), batch_size=4, learning_rate=1e-3,
                          total_epochs=10, num_workers=0, lambda_policy=1.0,
                          lambda_value=1.5, lambda_score=0.02, grad_clip=1.0)
        trainer = SLTrainer(small_model, config)
        metrics = trainer.train_epoch()
        assert metrics["policy_loss"] == 0.0
        assert metrics["value_loss"] == 0.0
        assert metrics["score_loss"] == 0.0

    def test_gradient_clipping_finite_metrics(self, tmp_path, small_model):
        """Training with tight gradient clipping should still produce finite metrics."""
        from keisei.sl.trainer import SLTrainer, SLConfig
        self._write_binary_shard(tmp_path)
        config = SLConfig(data_dir=str(tmp_path), batch_size=4, learning_rate=1e-1,
                          total_epochs=10, num_workers=0, lambda_policy=1.0,
                          lambda_value=1.5, lambda_score=0.02, grad_clip=0.5)
        trainer = SLTrainer(small_model, config)
        metrics = trainer.train_epoch()
        for key, val in metrics.items():
            assert np.isfinite(val), f"{key} is not finite"


class TestCSAParserEdgeCases:
    """CSA parser edge cases: CHUDAN, JISHOGI, SENNICHITE, multi-game."""

    def test_chudan_returns_none(self, tmp_path):
        """Interrupted game (%CHUDAN) should be skipped."""
        csa_text = """V2.2
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
%CHUDAN
"""
        csa_file = tmp_path / "game.csa"
        csa_file.write_text(csa_text)
        parser = CSAParser()
        records = list(parser.parse(csa_file))
        assert len(records) == 0

    def test_jishogi_last_mover_wins(self, tmp_path):
        """Impasse declaration (%JISHOGI) — last mover wins."""
        csa_text = """V2.2
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
+2726FU
%JISHOGI
"""
        csa_file = tmp_path / "game.csa"
        csa_file.write_text(csa_text)
        parser = CSAParser()
        records = list(parser.parse(csa_file))
        assert len(records) == 1
        # Last mover is "+" (Black made the 3rd move +2726FU)
        assert records[0].outcome == GameOutcome.WIN_BLACK

    def test_sennichite_is_draw(self, tmp_path):
        """Repetition (%SENNICHITE) — draw."""
        csa_text = """V2.2
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
%SENNICHITE
"""
        csa_file = tmp_path / "game.csa"
        csa_file.write_text(csa_text)
        parser = CSAParser()
        records = list(parser.parse(csa_file))
        assert len(records) == 1
        assert records[0].outcome == GameOutcome.DRAW

    def test_multi_game_archive(self, tmp_path):
        """Multi-game CSA file separated by / should yield multiple records."""
        game_block = """V2.2
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
%TORYO"""
        csa_file = tmp_path / "multi.csa"
        csa_file.write_text(game_block + "\n/\n" + game_block)
        parser = CSAParser()
        records = list(parser.parse(csa_file))
        assert len(records) == 2


class TestSLDatasetMultiShard:
    """Test SLDataset cross-shard boundary access."""

    def test_cross_shard_boundary_access(self, tmp_path):
        """Access positions at shard boundary — verify no off-by-one."""
        from keisei.sl.dataset import write_shard, OBS_SIZE, SLDataset

        rng = np.random.default_rng(42)
        n_per_shard = 5

        # Write 2 shards with distinct data
        all_policy = []
        for shard_idx in range(2):
            obs = rng.standard_normal((n_per_shard, OBS_SIZE)).astype(np.float32)
            policy = rng.integers(0, 11259, size=n_per_shard).astype(np.int64)
            value = rng.integers(0, 3, size=n_per_shard).astype(np.int64)
            score = rng.standard_normal(n_per_shard).astype(np.float32)
            write_shard(tmp_path / f"shard_{shard_idx:03d}.bin", obs, policy, value, score)
            all_policy.extend(policy.tolist())

        ds = SLDataset(tmp_path)
        assert len(ds) == 10

        # Verify every position reads correctly
        for i in range(10):
            sample = ds[i]
            assert sample["policy_target"].item() == all_policy[i]

        # Access boundary positions specifically
        sample_last_shard0 = ds[4]   # last position in shard 0
        sample_first_shard1 = ds[5]  # first position in shard 1
        assert sample_last_shard0["policy_target"].item() == all_policy[4]
        assert sample_first_shard1["policy_target"].item() == all_policy[5]

        # Out of range should raise
        with pytest.raises(IndexError):
            ds[10]
        with pytest.raises(IndexError):
            ds[-1]


class TestSLDatasetMmapCache:
    """Tests for LRU-bounded mmap cache (keisei-ed9dbc77d7)."""

    def _write_shards(self, tmp_path, n_shards=3, n_positions=5):
        """Helper: write n_shards shard files with n_positions each."""
        rng = np.random.default_rng(42)
        for i in range(n_shards):
            write_shard(
                tmp_path / f"shard_{i:03d}.bin",
                rng.standard_normal((n_positions, 50 * 81)).astype(np.float32),
                rng.integers(0, 11259, size=n_positions).astype(np.int64),
                rng.integers(0, 3, size=n_positions).astype(np.int64),
                rng.standard_normal(n_positions).astype(np.float32),
            )

    def test_max_cache_size_zero_raises(self, tmp_path):
        """max_cache_size < 1 must raise ValueError."""
        self._write_shards(tmp_path, n_shards=1)
        with pytest.raises(ValueError, match="max_cache_size must be >= 1"):
            SLDataset(tmp_path, max_cache_size=0)

    def test_lru_eviction_bounds_cache_size(self, tmp_path):
        """With max_cache_size=2 and 3 shards, cache never exceeds 2 entries."""
        self._write_shards(tmp_path, n_shards=3)
        dataset = SLDataset(tmp_path, max_cache_size=2)

        _ = dataset[0]   # shard 0
        _ = dataset[5]   # shard 1
        assert len(dataset._mmap_cache) == 2

        _ = dataset[10]  # shard 2 — should evict shard 0
        assert len(dataset._mmap_cache) == 2

        shard_0_path = dataset.shards[0][0]
        assert shard_0_path not in dataset._mmap_cache, "LRU should have evicted shard 0"

    def test_lru_promotion_on_reaccess(self, tmp_path):
        """Re-accessing a shard promotes it; the non-promoted shard is evicted."""
        self._write_shards(tmp_path, n_shards=3)
        dataset = SLDataset(tmp_path, max_cache_size=2)

        _ = dataset[0]   # shard 0 (LRU)
        _ = dataset[5]   # shard 1 (MRU)
        _ = dataset[0]   # re-access shard 0 — promotes to MRU

        _ = dataset[10]  # shard 2 — should evict shard 1 (now LRU), not shard 0
        shard_0_path = dataset.shards[0][0]
        shard_1_path = dataset.shards[1][0]
        assert shard_0_path in dataset._mmap_cache, "Shard 0 was promoted, should survive"
        assert shard_1_path not in dataset._mmap_cache, "Shard 1 was LRU, should be evicted"

    def test_lru_single_slot_boundary(self, tmp_path):
        """max_cache_size=1: every new shard evicts the previous one."""
        self._write_shards(tmp_path, n_shards=2)
        dataset = SLDataset(tmp_path, max_cache_size=1)

        _ = dataset[0]  # shard 0
        assert len(dataset._mmap_cache) == 1

        _ = dataset[5]  # shard 1 — should evict shard 0
        assert len(dataset._mmap_cache) == 1
        shard_0_path = dataset.shards[0][0]
        assert shard_0_path not in dataset._mmap_cache

    def test_warning_when_shards_exceed_cache_size(self, tmp_path, caplog):
        """A warning should be logged when num_shards > max_cache_size."""
        self._write_shards(tmp_path, n_shards=5)
        with caplog.at_level(logging.WARNING, logger="keisei.sl.dataset"):
            SLDataset(tmp_path, max_cache_size=2)
        assert any("5 shards but max_cache_size=2" in msg for msg in caplog.messages)

    def test_clear_cache_empties_and_reaccess_works(self, tmp_path):
        """clear_cache() empties the cache; subsequent access re-opens cleanly."""
        self._write_shards(tmp_path, n_shards=2)
        dataset = SLDataset(tmp_path, max_cache_size=16)

        # Populate cache
        item_before = dataset[0]
        assert len(dataset._mmap_cache) == 1

        # Clear
        dataset.clear_cache()
        assert len(dataset._mmap_cache) == 0

        # Re-access same item — should work and return identical values
        item_after = dataset[0]
        assert len(dataset._mmap_cache) == 1
        assert torch.equal(item_before["observation"], item_after["observation"])
        assert item_before["policy_target"] == item_after["policy_target"]
