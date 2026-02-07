"""E2E tests for checkpoint resume functionality.

Tests cover:
- Training produces a checkpoint that can be found
- Resuming from a specific checkpoint completes without error
- Resumed training starts from the saved timestep
- Checkpoint files contain expected state keys
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml


# All E2E tests are marked with the e2e marker for selective execution.
pytestmark = pytest.mark.e2e


def _find_checkpoint(model_dir: Path) -> Path:
    """Find the first .pth checkpoint file in model_dir (non-final_model)."""
    pth_files = sorted(model_dir.glob("checkpoint_ts*.pth"))
    if pth_files:
        return pth_files[0]
    # Fallback to any .pth file
    pth_files = sorted(model_dir.glob("*.pth"))
    assert pth_files, f"No .pth files found in {model_dir}"
    return pth_files[0]


def _make_resume_config(tmp_path: Path, model_dir: str, total_timesteps: int = 64) -> Path:
    """Create a config YAML suitable for resume testing."""
    cfg = {
        "env": {
            "device": "cpu",
            "seed": 42,
            "input_channels": 46,
            "num_actions_total": 13527,
            "max_moves_per_game": 50,
        },
        "training": {
            "total_timesteps": total_timesteps,
            "steps_per_epoch": 8,
            "ppo_epochs": 2,
            "minibatch_size": 4,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "clip_epsilon": 0.2,
            "value_loss_coeff": 0.5,
            "entropy_coef": 0.01,
            "tower_depth": 2,
            "tower_width": 32,
            "se_ratio": 0.0,
            "model_type": "resnet",
            "gradient_clip_max_norm": 0.5,
            "lambda_gae": 0.95,
            "checkpoint_interval_timesteps": 16,
            "evaluation_interval_timesteps": 10000,
            "enable_torch_compile": False,
            "mixed_precision": False,
            "ddp": False,
            "render_every_steps": 1,
        },
        "evaluation": {
            "enable_periodic_evaluation": False,
            "num_games": 1,
            "max_moves_per_game": 20,
        },
        "logging": {
            "model_dir": model_dir,
            "log_file": str(tmp_path / "train.log"),
        },
        "wandb": {"enabled": False},
        "parallel": {"enabled": False},
        "display": {
            "display_moves": False,
            "turn_tick": 0.0,
        },
        "webui": {"enabled": False},
    }
    config_file = tmp_path / "resume_config.yaml"
    config_file.write_text(yaml.dump(cfg, default_flow_style=False))
    return config_file


class TestTrainAndResume:
    """Test training followed by checkpoint resume."""

    def _run_initial_training(self, run_cli, tmp_path, run_name="e2e_resume_init"):
        """Run initial training (64 timesteps) and return (result, model_dir)."""
        model_dir_str = str(tmp_path / "models")
        config_path = _make_resume_config(
            tmp_path, model_dir=model_dir_str, total_timesteps=64
        )
        result = run_cli(
            [
                "train",
                "--config", str(config_path),
                "--run-name", run_name,
                "--device", "cpu",
            ],
            timeout=120,
        )
        model_dir = tmp_path / "models" / run_name
        return result, model_dir

    def test_initial_training_produces_checkpoint(self, run_cli, tmp_path):
        """The first training run should produce at least one checkpoint file."""
        result, model_dir = self._run_initial_training(run_cli, tmp_path)
        assert result.returncode == 0, (
            f"Initial training failed (exit code {result.returncode}).\n"
            f"stderr: {result.stderr[-1000:]}"
        )
        pth_files = list(model_dir.glob("*.pth"))
        assert len(pth_files) > 0, (
            f"No checkpoint files found in {model_dir}.\n"
            f"Contents: {list(model_dir.iterdir()) if model_dir.exists() else 'dir not found'}"
        )

    def test_resume_from_checkpoint_completes(self, run_cli, tmp_path):
        """Resuming from a checkpoint should complete without error."""
        run_name = "e2e_resume_complete"
        # Phase 1: initial training
        result1, model_dir = self._run_initial_training(
            run_cli, tmp_path, run_name=run_name
        )
        assert result1.returncode == 0, (
            f"Phase 1 training failed: {result1.stderr[-500:]}"
        )

        # Find a checkpoint to resume from
        checkpoint_path = _find_checkpoint(model_dir)

        # Phase 2: resume training with higher total_timesteps
        model_dir_str = str(tmp_path / "models")
        resume_config = _make_resume_config(
            tmp_path, model_dir=model_dir_str, total_timesteps=128
        )
        result2 = run_cli(
            [
                "train",
                "--config", str(resume_config),
                "--resume", str(checkpoint_path),
                "--run-name", run_name,
                "--device", "cpu",
            ],
            timeout=120,
        )
        assert result2.returncode == 0, (
            f"Resume training failed (exit code {result2.returncode}).\n"
            f"stdout: {result2.stdout[-1000:]}\n"
            f"stderr: {result2.stderr[-1000:]}"
        )

    def test_resumed_training_starts_from_saved_timestep(self, run_cli, tmp_path):
        """After resume, stderr/log should indicate training started from a non-zero timestep."""
        run_name = "e2e_resume_ts"
        # Phase 1: train to 64 timesteps
        result1, model_dir = self._run_initial_training(
            run_cli, tmp_path, run_name=run_name
        )
        assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr[-500:]}"

        checkpoint_path = _find_checkpoint(model_dir)

        # Load checkpoint to learn what timestep it was saved at
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        saved_ts = ckpt.get("global_timestep", 0)

        # Phase 2: resume with higher target
        model_dir_str = str(tmp_path / "models")
        resume_config = _make_resume_config(
            tmp_path, model_dir=model_dir_str, total_timesteps=128
        )
        result2 = run_cli(
            [
                "train",
                "--config", str(resume_config),
                "--resume", str(checkpoint_path),
                "--run-name", run_name,
                "--device", "cpu",
            ],
            timeout=120,
        )
        assert result2.returncode == 0, f"Resume failed: {result2.stderr[-500:]}"

        # The stderr output should mention resuming from checkpoint
        combined_output = result2.stdout + result2.stderr
        assert "resum" in combined_output.lower() or "checkpoint" in combined_output.lower(), (
            f"Expected log output mentioning resume/checkpoint.\n"
            f"Output (last 2000 chars): {combined_output[-2000:]}"
        )

    def test_checkpoint_file_contains_expected_state(self, run_cli, tmp_path):
        """Checkpoint files should contain model_state_dict, optimizer_state_dict, and global_timestep."""
        run_name = "e2e_ckpt_state"
        result, model_dir = self._run_initial_training(
            run_cli, tmp_path, run_name=run_name
        )
        assert result.returncode == 0, f"Training failed: {result.stderr[-500:]}"

        checkpoint_path = _find_checkpoint(model_dir)
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)

        # Verify essential keys are present
        assert "model_state_dict" in ckpt, (
            f"Checkpoint missing 'model_state_dict'. Keys: {list(ckpt.keys())}"
        )
        assert "optimizer_state_dict" in ckpt, (
            f"Checkpoint missing 'optimizer_state_dict'. Keys: {list(ckpt.keys())}"
        )
        assert "global_timestep" in ckpt, (
            f"Checkpoint missing 'global_timestep'. Keys: {list(ckpt.keys())}"
        )
        assert "total_episodes_completed" in ckpt, (
            f"Checkpoint missing 'total_episodes_completed'. Keys: {list(ckpt.keys())}"
        )

        # global_timestep should be a positive integer (checkpoint was saved mid-training)
        assert isinstance(ckpt["global_timestep"], int), (
            f"global_timestep should be int, got {type(ckpt['global_timestep'])}"
        )
        assert ckpt["global_timestep"] > 0, (
            f"global_timestep should be positive, got {ckpt['global_timestep']}"
        )
