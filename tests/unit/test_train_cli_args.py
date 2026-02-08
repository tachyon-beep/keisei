"""Tests for CLI argument parsing in train.py: parsers, training command, main_sync."""

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from keisei.training.train import (
    add_evaluation_arguments,
    add_training_arguments,
    create_main_parser,
    run_training_command,
)


def _parse_eval_args(*args: str) -> argparse.Namespace:
    """Parse evaluation arguments, simulating the evaluate subcommand."""
    parser = argparse.ArgumentParser()
    add_evaluation_arguments(parser)
    return parser.parse_args(list(args))


class TestEvalArgDefaults:
    """Verify that evaluation CLI defaults are None, not hardcoded values."""

    def test_strategy_default_is_none(self):
        """--strategy defaults to None when not explicitly passed."""
        args = _parse_eval_args("--agent_checkpoint", "dummy.pt")
        assert args.strategy is None

    def test_num_games_default_is_none(self):
        """--num_games defaults to None when not explicitly passed."""
        args = _parse_eval_args("--agent_checkpoint", "dummy.pt")
        assert args.num_games is None

    def test_opponent_type_default_is_none(self):
        """--opponent_type defaults to None when not explicitly passed."""
        args = _parse_eval_args("--agent_checkpoint", "dummy.pt")
        assert args.opponent_type is None


class TestEvalArgExplicit:
    """Verify that explicitly passed CLI args are captured correctly."""

    def test_explicit_strategy(self):
        """--strategy captures the value when passed explicitly."""
        args = _parse_eval_args(
            "--agent_checkpoint", "dummy.pt", "--strategy", "tournament"
        )
        assert args.strategy == "tournament"

    def test_explicit_num_games(self):
        """--num_games captures the value when passed explicitly."""
        args = _parse_eval_args(
            "--agent_checkpoint", "dummy.pt", "--num_games", "50"
        )
        assert args.num_games == 50

    def test_explicit_opponent_type(self):
        """--opponent_type captures the value when passed explicitly."""
        args = _parse_eval_args(
            "--agent_checkpoint", "dummy.pt", "--opponent_type", "heuristic"
        )
        assert args.opponent_type == "heuristic"


# ===========================================================================
# Parser structure tests
# ===========================================================================


class TestCreateMainParser:
    """Verify main parser has train/evaluate subcommands."""

    def test_parser_has_train_subcommand(self):
        """Parser accepts 'train' subcommand."""
        parser = create_main_parser()
        args = parser.parse_args(["train"])
        assert args.command == "train"

    def test_parser_has_evaluate_subcommand(self):
        """Parser accepts 'evaluate' subcommand."""
        parser = create_main_parser()
        args = parser.parse_args(["evaluate", "--agent_checkpoint", "test.pt"])
        assert args.command == "evaluate"

    def test_no_command_sets_none(self):
        """No subcommand results in command=None."""
        parser = create_main_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestAddTrainingArguments:
    """Verify training arguments are properly added to parser."""

    def _parse_train_args(self, *args):
        parser = argparse.ArgumentParser()
        add_training_arguments(parser)
        return parser.parse_args(list(args))

    def test_config_arg(self):
        """--config accepts a path string."""
        args = self._parse_train_args("--config", "my_config.yaml")
        assert args.config == "my_config.yaml"

    def test_resume_arg(self):
        """--resume accepts a checkpoint path."""
        args = self._parse_train_args("--resume", "/path/to/ckpt.pth")
        assert args.resume == "/path/to/ckpt.pth"

    def test_override_arg_appends(self):
        """--override can be specified multiple times."""
        args = self._parse_train_args(
            "--override", "training.lr=0.001",
            "--override", "training.epochs=10",
        )
        assert len(args.override) == 2

    def test_savedir_arg(self):
        """--savedir accepts a directory path."""
        args = self._parse_train_args("--savedir", "/tmp/models")
        assert args.savedir == "/tmp/models"

    def test_total_timesteps_arg(self):
        """--total-timesteps accepts an integer."""
        args = self._parse_train_args("--total-timesteps", "50000")
        assert args.total_timesteps == 50000

    def test_defaults_are_none(self):
        """All optional args default to None or empty list."""
        args = self._parse_train_args()
        assert args.config is None
        assert args.resume is None
        assert args.override == []
        assert args.savedir is None
        assert args.device is None


# ===========================================================================
# Command execution tests
# ===========================================================================


class TestRunTrainingCommand:
    """Tests for run_training_command execution."""

    @patch("keisei.training.train.Trainer")
    @patch("keisei.training.train.load_config")
    @patch("keisei.training.train.apply_wandb_sweep_config")
    @patch("keisei.training.train.build_cli_overrides")
    def test_default_args_creates_trainer(
        self, mock_build_cli, mock_sweep, mock_load, mock_trainer_cls
    ):
        """Default args create Trainer and call run_training_loop."""
        mock_sweep.return_value = {}
        mock_build_cli.return_value = {}
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        args = SimpleNamespace(
            config=None,
            override=[],
            enable_async_evaluation=False,
        )

        run_training_command(args)

        mock_load.assert_called_once()
        mock_trainer_cls.assert_called_once_with(config=mock_config, args=args)
        mock_trainer.run_training_loop.assert_called_once()

    @patch("keisei.training.train.Trainer")
    @patch("keisei.training.train.load_config")
    @patch("keisei.training.train.apply_wandb_sweep_config")
    @patch("keisei.training.train.build_cli_overrides")
    def test_config_flag_loads_specified_config(
        self, mock_build_cli, mock_sweep, mock_load, mock_trainer_cls
    ):
        """--config flag passes config path to load_config."""
        mock_sweep.return_value = {}
        mock_build_cli.return_value = {}
        mock_load.return_value = MagicMock()
        mock_trainer_cls.return_value = MagicMock()

        args = SimpleNamespace(
            config="custom_config.yaml",
            override=[],
            enable_async_evaluation=False,
        )

        run_training_command(args)

        mock_load.assert_called_once_with("custom_config.yaml", {})

    @patch("keisei.training.train.Trainer")
    @patch("keisei.training.train.load_config")
    @patch("keisei.training.train.apply_wandb_sweep_config")
    @patch("keisei.training.train.build_cli_overrides")
    def test_override_flag_merges_overrides(
        self, mock_build_cli, mock_sweep, mock_load, mock_trainer_cls
    ):
        """--override flag merges key=value pairs into config overrides."""
        mock_sweep.return_value = {}
        mock_build_cli.return_value = {}
        mock_load.return_value = MagicMock()
        mock_trainer_cls.return_value = MagicMock()

        args = SimpleNamespace(
            config=None,
            override=["training.learning_rate=0.001"],
            enable_async_evaluation=False,
        )

        run_training_command(args)

        # Verify the override was passed to load_config
        call_args = mock_load.call_args
        overrides = call_args[0][1]
        assert "training.learning_rate" in overrides
        assert overrides["training.learning_rate"] == "0.001"

    @patch("keisei.training.train.Trainer")
    @patch("keisei.training.train.load_config")
    @patch("keisei.training.train.apply_wandb_sweep_config")
    @patch("keisei.training.train.build_cli_overrides")
    def test_enables_async_evaluation(
        self, mock_build_cli, mock_sweep, mock_load, mock_trainer_cls
    ):
        """--enable-async-evaluation enables async callbacks."""
        mock_sweep.return_value = {}
        mock_build_cli.return_value = {}
        mock_load.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        args = SimpleNamespace(
            config=None,
            override=[],
            enable_async_evaluation=True,
        )

        run_training_command(args)

        mock_trainer.callback_manager.use_async_evaluation.assert_called_once()


# ===========================================================================
# main_sync tests
# ===========================================================================


class TestMainSync:
    """Tests for main_sync entry point."""

    @patch("keisei.training.train.asyncio")
    @patch("keisei.training.train.multiprocessing")
    def test_sets_multiprocessing_start_method(self, mock_mp, mock_asyncio):
        """Sets multiprocessing start method to 'spawn'."""
        from keisei.training.train import main_sync

        mock_mp.get_start_method.return_value = None

        main_sync()

        mock_mp.freeze_support.assert_called_once()
        mock_mp.set_start_method.assert_called_once_with("spawn", force=True)

    @patch("keisei.training.train.sys")
    @patch("keisei.training.train.asyncio")
    @patch("keisei.training.train.multiprocessing")
    def test_handles_keyboard_interrupt(self, mock_mp, mock_asyncio, mock_sys):
        """Handles KeyboardInterrupt gracefully."""
        from keisei.training.train import main_sync

        mock_mp.get_start_method.return_value = "spawn"
        mock_asyncio.run.side_effect = KeyboardInterrupt

        main_sync()

        mock_sys.exit.assert_called_with(1)

    @patch("keisei.training.train.sys")
    @patch("keisei.training.train.asyncio")
    @patch("keisei.training.train.multiprocessing")
    def test_handles_generic_exception(self, mock_mp, mock_asyncio, mock_sys):
        """Handles generic exception and exits with code 1."""
        from keisei.training.train import main_sync

        mock_mp.get_start_method.return_value = "spawn"
        mock_asyncio.run.side_effect = RuntimeError("fatal error")

        main_sync()

        mock_sys.exit.assert_called_with(1)
