"""Tests for TieredPool Phase 3 wiring — DynamicTrainer and FrontierPromoter integration."""

import logging
from dataclasses import replace

import pytest

from keisei.config import DynamicConfig, LeagueConfig
from keisei.db import init_db
from keisei.training.dynamic_trainer import DynamicTrainer
from keisei.training.frontier_promoter import FrontierPromoter
from keisei.training.opponent_store import OpponentStore
from keisei.training.tiered_pool import TieredPool


@pytest.fixture
def pool_with_training(tmp_path):
    db_path = str(tmp_path / "pool.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    store = OpponentStore(db_path, str(league_dir))
    config = LeagueConfig()
    assert config.dynamic.training_enabled is True
    pool = TieredPool(store, config, learner_lr=2e-4)
    return pool, store


@pytest.fixture
def pool_no_training(tmp_path):
    db_path = str(tmp_path / "pool.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    store = OpponentStore(db_path, str(league_dir))
    config = LeagueConfig(dynamic=replace(DynamicConfig(), training_enabled=False))
    pool = TieredPool(store, config, learner_lr=2e-4)
    return pool, store


class TestTieredPoolPhase3Wiring:
    def test_tiered_pool_creates_trainer_and_promoter(self, pool_with_training):
        pool, _store = pool_with_training
        assert isinstance(pool.dynamic_trainer, DynamicTrainer)
        assert isinstance(pool.frontier_manager._promoter, FrontierPromoter)

    def test_tiered_pool_no_trainer_when_training_disabled(self, pool_no_training):
        pool, _store = pool_no_training
        assert pool.dynamic_trainer is None

    def test_tiered_pool_gpu_collision_warning(self, tmp_path, caplog):
        db_path = str(tmp_path / "pool.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))
        config = LeagueConfig()
        assert config.dynamic.training_enabled is True

        with caplog.at_level(logging.WARNING, logger="keisei.training.tiered_pool"):
            TieredPool(store, config, learner_lr=2e-4)

        all_messages = " ".join(r.message for r in caplog.records)
        assert "Dynamic training enabled" in all_messages, (
            f"Expected 'Dynamic training enabled' in logs, got: {all_messages}"
        )
        assert "GPU memory contention" in all_messages, (
            f"Expected 'GPU memory contention' in logs, got: {all_messages}"
        )
