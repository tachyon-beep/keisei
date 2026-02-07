"""End-to-end test tier: CLI subprocess tests.

Provides fixtures and helpers for running the train.py CLI as subprocesses
with minimal configurations for fast E2E testing.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def train_cmd() -> List[str]:
    """Base command list for invoking train.py via the current interpreter."""
    return [sys.executable, str(TRAIN_SCRIPT)]


@pytest.fixture
def cli_env() -> Dict[str, str]:
    """Environment dict with wandb disabled and deterministic settings."""
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"
    # Ensure reproducibility and avoid GPU usage in tests
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


@pytest.fixture
def run_cli(train_cmd, cli_env):
    """Return a helper that runs a CLI command and returns CompletedProcess.

    Usage inside a test::

        result = run_cli(["train", "--help"])
        assert result.returncode == 0
    """

    def _run(
        extra_args: List[str],
        timeout: int = 120,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        env = {**cli_env, **(env_overrides or {})}
        cmd = train_cmd + extra_args
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(PROJECT_ROOT),
        )

    return _run


@pytest.fixture
def fast_config_path(tmp_path) -> Path:
    """Write a minimal YAML config optimised for speed and return its path.

    Key choices:
    - total_timesteps=64, steps_per_epoch=8  (only 8 epochs of collection)
    - tiny model (tower_depth=2, tower_width=32)
    - checkpoint every 32 timesteps so at least one checkpoint is created
    - wandb disabled, webui disabled, torch compile disabled
    - display_moves off, no demo delay
    - CPU only
    """
    cfg = {
        "env": {
            "device": "cpu",
            "seed": 42,
            "input_channels": 46,
            "num_actions_total": 13527,
            "max_moves_per_game": 50,
        },
        "training": {
            "total_timesteps": 64,
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
            "checkpoint_interval_timesteps": 32,
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
            "model_dir": str(tmp_path / "models"),
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
    config_file = tmp_path / "fast_config.yaml"
    config_file.write_text(yaml.dump(cfg, default_flow_style=False))
    return config_file


@pytest.fixture
def fast_config_with_checkpoints(tmp_path) -> Path:
    """Like fast_config_path but guarantees frequent checkpoint saves.

    total_timesteps=64, checkpoint_interval_timesteps=16 => expect checkpoints
    at timesteps 16, 32, 48, 64 (plus a final checkpoint).
    """
    cfg = {
        "env": {
            "device": "cpu",
            "seed": 42,
            "input_channels": 46,
            "num_actions_total": 13527,
            "max_moves_per_game": 50,
        },
        "training": {
            "total_timesteps": 64,
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
            "model_dir": str(tmp_path / "models"),
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
    config_file = tmp_path / "fast_config_ckpt.yaml"
    config_file.write_text(yaml.dump(cfg, default_flow_style=False))
    return config_file
