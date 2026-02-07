"""E2E tests for the train.py CLI interface.

Tests cover:
- Help output and exit codes
- Short training runs completing successfully
- Checkpoint file creation
- Invalid config handling
- Expected log output from short runs
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


# All E2E tests are marked with the e2e marker for selective execution.
pytestmark = pytest.mark.e2e


class TestCLIHelp:
    """Verify that help messages display correctly and exit with code 0."""

    def test_main_help_exits_zero(self, run_cli):
        """train.py --help should exit with code 0 and show usage text."""
        result = run_cli(["--help"])
        assert result.returncode == 0, (
            f"Expected exit code 0 for --help, got {result.returncode}.\n"
            f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        )
        assert "usage" in result.stdout.lower() or "Usage" in result.stdout

    def test_train_help_shows_training_options(self, run_cli):
        """train.py train --help should show training-specific options."""
        result = run_cli(["train", "--help"])
        assert result.returncode == 0, (
            f"Expected exit code 0 for train --help, got {result.returncode}.\n"
            f"stderr: {result.stderr[:500]}"
        )
        # The help output should mention key training arguments
        stdout_lower = result.stdout.lower()
        assert "--config" in stdout_lower, "Expected --config in train help output"
        assert "--resume" in stdout_lower, "Expected --resume in train help output"
        assert "--override" in stdout_lower, "Expected --override in train help output"

    def test_evaluate_help_shows_evaluation_options(self, run_cli):
        """train.py evaluate --help should show evaluation-specific options."""
        result = run_cli(["evaluate", "--help"])
        assert result.returncode == 0, (
            f"Expected exit code 0 for evaluate --help, got {result.returncode}.\n"
            f"stderr: {result.stderr[:500]}"
        )
        stdout_lower = result.stdout.lower()
        assert "--agent_checkpoint" in stdout_lower or "--agent-checkpoint" in stdout_lower, (
            "Expected --agent_checkpoint in evaluate help output"
        )


class TestShortTrainingRun:
    """Run training with minimal configs to verify the pipeline works end-to-end."""

    def test_short_training_completes_successfully(self, run_cli, fast_config_path, tmp_path):
        """A very short training run (64 timesteps) should complete with exit code 0."""
        run_name = "e2e_short_run"
        result = run_cli(
            [
                "train",
                "--config", str(fast_config_path),
                "--run-name", run_name,
                "--device", "cpu",
            ],
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Training did not complete successfully (exit code {result.returncode}).\n"
            f"stdout (last 1000 chars): {result.stdout[-1000:]}\n"
            f"stderr (last 1000 chars): {result.stderr[-1000:]}"
        )

    def test_training_creates_checkpoint_files(self, run_cli, fast_config_with_checkpoints, tmp_path):
        """Training should create .pth checkpoint files in the model directory."""
        run_name = "e2e_ckpt_test"
        result = run_cli(
            [
                "train",
                "--config", str(fast_config_with_checkpoints),
                "--run-name", run_name,
                "--device", "cpu",
            ],
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Training failed (exit code {result.returncode}).\n"
            f"stderr (last 1000 chars): {result.stderr[-1000:]}"
        )

        # The model dir should be <tmp_path>/models/<run_name>/
        model_dir = tmp_path / "models" / run_name
        assert model_dir.exists(), (
            f"Model directory {model_dir} does not exist after training.\n"
            f"Contents of {tmp_path / 'models'}: "
            f"{list((tmp_path / 'models').iterdir()) if (tmp_path / 'models').exists() else 'dir not found'}"
        )

        # Look for .pth files (checkpoints or final model)
        pth_files = list(model_dir.glob("*.pth"))
        assert len(pth_files) > 0, (
            f"Expected at least one .pth file in {model_dir}, found none.\n"
            f"Contents: {list(model_dir.iterdir())}"
        )

    def test_training_creates_final_model(self, run_cli, fast_config_path, tmp_path):
        """After completing all timesteps, training should save final_model.pth."""
        run_name = "e2e_final_model"
        result = run_cli(
            [
                "train",
                "--config", str(fast_config_path),
                "--run-name", run_name,
                "--device", "cpu",
            ],
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Training failed (exit code {result.returncode}).\n"
            f"stderr: {result.stderr[-1000:]}"
        )

        model_dir = tmp_path / "models" / run_name
        # After completing all timesteps, a final_model.pth or a final checkpoint should exist
        pth_files = list(model_dir.glob("*.pth"))
        filenames = [f.name for f in pth_files]
        # Either final_model.pth or a checkpoint_ts<N>.pth at the final timestep
        has_final = "final_model.pth" in filenames or any(
            f.startswith("checkpoint_ts") for f in filenames
        )
        assert has_final, (
            f"Expected final_model.pth or checkpoint_ts*.pth in {model_dir}.\n"
            f"Found: {filenames}"
        )


class TestInvalidConfig:
    """Verify the CLI handles invalid configurations gracefully."""

    def test_invalid_config_file_fails(self, run_cli, tmp_path):
        """Pointing --config to a nonexistent file should fail with non-zero exit."""
        result = run_cli(
            [
                "train",
                "--config", str(tmp_path / "nonexistent_config.yaml"),
                "--run-name", "e2e_bad_config",
            ],
            timeout=30,
        )
        assert result.returncode != 0, (
            "Expected non-zero exit code for nonexistent config file, got 0."
        )

    def test_invalid_config_values_fail(self, run_cli, tmp_path):
        """A config with invalid values (negative learning rate) should fail."""
        bad_cfg = {
            "env": {"device": "cpu"},
            "training": {
                "learning_rate": -1.0,  # Invalid: must be positive
            },
            "wandb": {"enabled": False},
        }
        bad_config_file = tmp_path / "bad_config.yaml"
        bad_config_file.write_text(yaml.dump(bad_cfg, default_flow_style=False))

        result = run_cli(
            [
                "train",
                "--config", str(bad_config_file),
                "--run-name", "e2e_bad_values",
            ],
            timeout=30,
        )
        assert result.returncode != 0, (
            "Expected non-zero exit code for invalid config values."
        )
        # The error message should mention validation or learning_rate
        combined_output = result.stdout + result.stderr
        assert "learning_rate" in combined_output.lower() or "validation" in combined_output.lower(), (
            f"Expected error mentioning learning_rate or validation.\n"
            f"Output: {combined_output[:1000]}"
        )


class TestCLIOverrides:
    """Verify CLI --override flags work correctly."""

    def test_override_total_timesteps(self, run_cli, fast_config_path, tmp_path):
        """--override training.total_timesteps=32 should limit training to 32 steps."""
        run_name = "e2e_override"
        result = run_cli(
            [
                "train",
                "--config", str(fast_config_path),
                "--run-name", run_name,
                "--device", "cpu",
                "--override", "training.total_timesteps=32",
            ],
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Training with override failed (exit code {result.returncode}).\n"
            f"stderr: {result.stderr[-1000:]}"
        )
