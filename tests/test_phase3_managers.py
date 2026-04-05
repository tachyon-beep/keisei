"""Phase 3 manager tests: DynamicManager training enablement and get_trainable."""

import pytest
import torch

from keisei.config import DynamicConfig
from keisei.db import init_db
from keisei.training.opponent_store import OpponentStore, Role
from keisei.training.tier_managers import DynamicManager


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "tier.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    return OpponentStore(db_path, str(league_dir))


def _add_entry(store, epoch, role=Role.UNASSIGNED, elo=1000.0):
    model = torch.nn.Linear(10, 10)
    entry = store.add_entry(model, "resnet", {}, epoch=epoch, role=role)
    if elo != 1000.0:
        store.update_elo(entry.id, elo)
    return store._get_entry(entry.id)


def test_dynamic_manager_allows_training_enabled(store):
    """Constructing DynamicManager with training_enabled=True should not raise."""
    mgr = DynamicManager(store, DynamicConfig(training_enabled=True))
    assert mgr._config.training_enabled is True


def test_dynamic_config_default_now_true():
    """DynamicConfig default for training_enabled is now True."""
    assert DynamicConfig().training_enabled is True


def test_get_trainable_returns_active_dynamic_entries(store):
    """get_trainable filters out disabled entries."""
    e1 = _add_entry(store, 1, role=Role.DYNAMIC)
    e2 = _add_entry(store, 2, role=Role.DYNAMIC)
    e3 = _add_entry(store, 3, role=Role.DYNAMIC)
    mgr = DynamicManager(store, DynamicConfig(training_enabled=True))
    result = mgr.get_trainable(disabled_entries={e2.id})
    assert len(result) == 2
    result_ids = {e.id for e in result}
    assert e1.id in result_ids
    assert e3.id in result_ids
    assert e2.id not in result_ids


def test_get_trainable_empty_when_training_disabled(store):
    """get_trainable returns empty list when training is disabled."""
    _add_entry(store, 1, role=Role.DYNAMIC)
    _add_entry(store, 2, role=Role.DYNAMIC)
    mgr = DynamicManager(store, DynamicConfig(training_enabled=False))
    result = mgr.get_trainable()
    assert result == []
