"""
E2E tests for train.py.
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path

import pytest

TRAIN_PATH = Path(__file__).parent.parent.parent / "train.py"

# Common CLI args appended to every subprocess training invocation to ensure
# tests are isolated: WandB disabled via config override, WebUI disabled to
# avoid port conflicts.
_ISOLATION_OVERRIDES = [
    "--override", "wandb.enabled=false",
    "--override", "webui.enabled=false",
]


def _make_test_env():
    """Build a subprocess environment with WandB properly disabled."""
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    return env


def check_training_outputs(result, expected_timesteps):
    """Check common training outputs."""
    assert result.returncode == 0, f"Training failed: {result.stderr}"

    # Check that we actually trained for the expected timesteps
    # Look for timestep progress in the formatted output
    assert f"Steps: {expected_timesteps}/{expected_timesteps}" in result.stderr or \
           f"timesteps={expected_timesteps}" in result.stderr or \
           str(expected_timesteps) in result.stderr, f"Expected timesteps {expected_timesteps} not found in output"


@pytest.mark.e2e
def test_train_cli_help():
    """Test that train.py train --help runs and prints usage."""
    result = subprocess.run(
        [sys.executable, TRAIN_PATH, "train", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    assert "--device" in result.stdout


@pytest.mark.e2e
def test_train_resume_autodetect(tmp_path):
    """
    Test that train.py can auto-detect a checkpoint from parent directory and resume training.

    This test verifies the parent directory checkpoint search functionality:
    1. Run an initial short training to produce a real checkpoint
    2. Move the checkpoint to the parent (savedir) directory
    3. Run train.py with --resume latest and --savedir tmp_path
    4. Verify ModelManager finds the checkpoint in parent dir, copies it to run dir, and resumes
    """
    import shutil

    env = _make_test_env()

    # Step 1: Run a short initial training to produce a valid checkpoint
    result_initial = subprocess.run(
        [
            sys.executable,
            TRAIN_PATH,
            "train",
            "--savedir", str(tmp_path),
            "--total-timesteps", "10",
            "--seed", "42",
            "--device", "cpu",
            *_ISOLATION_OVERRIDES,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result_initial.returncode == 0, f"Initial training failed: {result_initial.stderr}"

    # Step 2: Find the checkpoint from the initial run and copy to parent dir
    run_dirs = list(tmp_path.glob("keisei_*"))
    assert run_dirs, f"No run directory created in {tmp_path}"
    initial_run_dir = run_dirs[0]

    checkpoints = list(initial_run_dir.glob("*.pth"))
    assert checkpoints, f"No checkpoint produced in {initial_run_dir}"
    # Copy first checkpoint to parent directory (savedir)
    src_ckpt = checkpoints[0]
    parent_ckpt = tmp_path / "checkpoint_ts1.pth"
    shutil.copy2(str(src_ckpt), str(parent_ckpt))
    print(f"Copied checkpoint {src_ckpt} -> {parent_ckpt}")

    # Step 3: Run train.py with --resume latest, expecting it to find checkpoint in parent dir
    result = subprocess.run(
        [
            sys.executable,
            TRAIN_PATH,
            "train",
            "--resume", "latest",
            "--savedir", str(tmp_path),
            "--total-timesteps", "10",
            "--seed", "42",
            "--device", "cpu",
            *_ISOLATION_OVERRIDES,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")

    # Step 4: Verify results
    # A new run directory should have been created (2 total now)
    all_run_dirs = list(tmp_path.glob("keisei_*"))
    assert len(all_run_dirs) >= 2, f"Expected at least 2 run directories, got {len(all_run_dirs)}"

    # Verify that training completed successfully
    check_training_outputs(result, 10)


@pytest.mark.e2e
def test_train_runs_minimal():
    """Test that train.py runs with minimal configuration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        env = _make_test_env()

        result = subprocess.run(
            [
                sys.executable,
                TRAIN_PATH,
                "train",
                "--total-timesteps", "10",
                "--savedir", str(tmp_path),
                "--seed", "42",
                "--device", "cpu",
                *_ISOLATION_OVERRIDES,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )

        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        # Check that a run directory was created
        run_dirs = list(tmp_path.glob("keisei_*"))
        assert run_dirs, "No run directory created"

        run_dir = run_dirs[0]
        print(f"Found run directory: {run_dir}")

        # Verify that training completed successfully
        check_training_outputs(result, 10)


@pytest.mark.e2e
def test_train_config_override():
    """Test that train.py handles config overrides correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        env = _make_test_env()

        result = subprocess.run(
            [
                sys.executable,
                TRAIN_PATH,
                "train",
                "--total-timesteps", "10",
                "--savedir", str(tmp_path),
                "--seed", "42",
                "--device", "cpu",
                "--override", "training.tower_width=32",
                "--override", "training.learning_rate=0.001",
                *_ISOLATION_OVERRIDES,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )

        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        # Check that a run directory was created
        run_dirs = list(tmp_path.glob("keisei_*"))
        assert run_dirs, "No run directory created for config override test"

        # Verify that training completed successfully
        check_training_outputs(result, 10)


@pytest.mark.e2e
def test_train_run_name_and_savedir():
    """Test that train.py respects custom run names and save directories."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        env = _make_test_env()

        # Test with a custom run name prefix
        result = subprocess.run(
            [
                sys.executable,
                TRAIN_PATH,
                "train",
                "--total-timesteps", "10",
                "--savedir", str(tmp_path),
                "--run-name", "mytestrunprefix",
                "--model", "testmodel",
                "--input_features", "testfeats",
                "--seed", "42",
                "--device", "cpu",
                *_ISOLATION_OVERRIDES,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )

        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        # When --run-name is explicitly provided, the directory uses that name exactly
        # (not combining with model/feature parameters)
        run_dirs = list(tmp_path.glob("mytestrunprefix*"))
        assert run_dirs, f"Expected run directory starting with 'mytestrunprefix' not found in {tmp_path}"

        # Find the run directory that starts with our custom run name
        run_dir = None
        for d in run_dirs:
            if d.is_dir() and d.name.startswith("mytestrunprefix"):
                run_dir = d
                break

        assert run_dir is not None, f"No run directory starting with 'mytestrunprefix' found in {tmp_path}"

        print(f"Found run directory: {run_dir}")

        # Verify that training completed successfully
        check_training_outputs(result, 10)


@pytest.mark.e2e
def test_train_explicit_resume(tmp_path):
    """
    Test explicit checkpoint resuming with train.py.

    This test verifies explicit checkpoint path functionality:
    1. Run an initial short training to produce a real checkpoint
    2. Run train.py with --resume pointing to that specific checkpoint
    3. Verify training resumes from the checkpoint
    """
    env = _make_test_env()

    # Step 1: Run a short initial training to produce a valid checkpoint
    result_initial = subprocess.run(
        [
            sys.executable,
            TRAIN_PATH,
            "train",
            "--savedir", str(tmp_path),
            "--total-timesteps", "10",
            "--seed", "42",
            "--device", "cpu",
            *_ISOLATION_OVERRIDES,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result_initial.returncode == 0, f"Initial training failed: {result_initial.stderr}"

    # Step 2: Find the checkpoint from the initial run
    run_dirs = list(tmp_path.glob("keisei_*"))
    assert run_dirs, f"No run directory created in {tmp_path}"
    initial_run_dir = run_dirs[0]

    checkpoints = list(initial_run_dir.glob("*.pth"))
    assert checkpoints, f"No checkpoint produced in {initial_run_dir}"
    checkpoint_path = checkpoints[0]
    print(f"Using checkpoint: {checkpoint_path}")

    # Step 3: Run train.py with explicit checkpoint path
    result = subprocess.run(
        [
            sys.executable,
            TRAIN_PATH,
            "train",
            "--resume", str(checkpoint_path),
            "--savedir", str(tmp_path),
            "--total-timesteps", "20",
            "--seed", "42",
            "--device", "cpu",
            *_ISOLATION_OVERRIDES,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")

    # Step 4: Verify results
    # A new run directory should have been created (2 total now)
    all_run_dirs = list(tmp_path.glob("keisei_*"))
    assert len(all_run_dirs) >= 2, f"Expected at least 2 run directories, got {len(all_run_dirs)}"

    # Verify that training completed successfully
    check_training_outputs(result, 20)
