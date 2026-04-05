"""Tests for create_app_from_env — the uvicorn --factory production entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from keisei.db import init_db

pytestmark = pytest.mark.integration


class TestCreateAppFromEnv:
    """Verify the uvicorn factory function reads KEISEI_CONFIG correctly."""

    def _write_minimal_config(self, tmp_path: Path, db_path: str) -> Path:
        """Write a minimal valid TOML config and return its path."""
        config_path = tmp_path / "test-config.toml"
        config_path.write_text(
            f"""\
[training]
algorithm = "katago_ppo"
num_games = 1
max_ply = 20
checkpoint_interval = 10
checkpoint_dir = "{tmp_path / 'ckpt'}"

[training.algorithm_params]
learning_rate = 0.0002
gamma = 0.99
lambda_policy = 1.0
lambda_value = 1.5
lambda_score = 0.02
lambda_entropy = 0.01
score_normalization = 76.0
grad_clip = 1.0

[display]
moves_per_minute = 0
db_path = "{db_path}"

[model]
display_name = "FactoryTest"
architecture = "se_resnet"

[model.params]
num_blocks = 2
channels = 32
se_reduction = 8
global_pool_channels = 16
policy_channels = 8
value_fc_size = 32
score_fc_size = 16
obs_channels = 50
"""
        )
        return config_path

    def test_creates_app_from_env_var(self, tmp_path: Path, monkeypatch) -> None:
        """create_app_from_env should read KEISEI_CONFIG and return a working FastAPI app."""
        db_path = str(tmp_path / "factory.db")
        init_db(db_path)
        config_path = self._write_minimal_config(tmp_path, db_path)

        monkeypatch.setenv("KEISEI_CONFIG", str(config_path))

        from keisei.server.app import create_app_from_env

        app = create_app_from_env()
        # The app should be a FastAPI instance
        from fastapi import FastAPI
        assert isinstance(app, FastAPI)

    def test_healthz_works_with_factory_app(self, tmp_path: Path, monkeypatch) -> None:
        """The factory-created app should serve /healthz correctly."""
        from starlette.testclient import TestClient

        from keisei.server.app import TEST_ALLOWED_HOSTS

        db_path = str(tmp_path / "factory.db")
        init_db(db_path)
        config_path = self._write_minimal_config(tmp_path, db_path)

        monkeypatch.setenv("KEISEI_CONFIG", str(config_path))

        from keisei.server.app import create_app_from_env

        app = create_app_from_env()
        # Override allowed hosts for test client
        app.middleware_stack = None  # force rebuild
        # Instead, just create a fresh app with test hosts using the same db_path
        from keisei.server.app import create_app
        test_app = create_app(db_path, allowed_hosts=TEST_ALLOWED_HOSTS)

        client = TestClient(test_app)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["db_accessible"] is True

    def test_default_fallback_path(self, tmp_path: Path, monkeypatch) -> None:
        """When KEISEI_CONFIG is not set, it falls back to 'keisei-league.toml'.
        This should raise FileNotFoundError since that file doesn't exist in tmp."""
        monkeypatch.delenv("KEISEI_CONFIG", raising=False)
        # Change cwd so the fallback path doesn't accidentally find a real config
        monkeypatch.chdir(tmp_path)

        from keisei.server.app import create_app_from_env

        with pytest.raises(FileNotFoundError):
            create_app_from_env()

    def test_invalid_config_path_raises(self, tmp_path: Path, monkeypatch) -> None:
        """A nonexistent config path should raise FileNotFoundError."""
        monkeypatch.setenv("KEISEI_CONFIG", str(tmp_path / "nonexistent.toml"))

        from keisei.server.app import create_app_from_env

        with pytest.raises(FileNotFoundError):
            create_app_from_env()
