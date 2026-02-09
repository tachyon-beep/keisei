"""
Tests for lineage event emission in ModelManager.

Validates that checkpoint saves, training start, and training resume
correctly emit lineage events through the registry.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from keisei.lineage.event_schema import validate_event
from keisei.lineage.registry import LineageRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_manager(tmp_path, lineage_enabled=True):
    """Build a ModelManager with a stubbed config and optional lineage registry.

    We patch out the heavy init (feature specs, mixed precision, compilation)
    to keep tests fast and focused on lineage emission.
    """
    from keisei.training.model_manager import ModelManager

    config = MagicMock()
    config.training.mixed_precision = False
    config.training.enable_torch_compile = False
    config.training.enable_compilation_benchmarking = False
    config.training.input_features = "core46"
    config.training.model_type = "resnet"
    config.training.tower_depth = 9
    config.training.tower_width = 256
    config.training.se_ratio = 0.25
    config.env.device = "cpu"
    config.env.num_actions_total = 13527

    args = MagicMock()
    args.input_features = None
    args.model = None
    args.tower_depth = None
    args.tower_width = None
    args.se_ratio = None
    args.resume = None

    import torch

    device = torch.device("cpu")

    with patch.object(ModelManager, "_setup_feature_spec"), \
         patch.object(ModelManager, "_setup_mixed_precision"), \
         patch.object(ModelManager, "_setup_compilation_infrastructure"):
        mm = ModelManager(config, args, device, logger_func=lambda msg: None)

    if lineage_enabled:
        registry_path = tmp_path / "lineage.jsonl"
        registry = LineageRegistry(registry_path)
        mm.set_lineage_registry(registry, "test-run")
        return mm, registry, registry_path
    return mm, None, None


def _mock_agent_save(path, timestep, episodes, stats_to_save=None):
    """Simulate agent.save_model by creating an empty file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("mock checkpoint")


# ---------------------------------------------------------------------------
# set_lineage_registry
# ---------------------------------------------------------------------------


class TestSetLineageRegistry:
    def test_registry_is_wired(self, tmp_path):
        mm, registry, _ = _make_model_manager(tmp_path)
        assert mm._lineage_registry is registry
        assert mm._run_name == "test-run"

    def test_no_registry_by_default(self, tmp_path):
        mm, _, _ = _make_model_manager(tmp_path, lineage_enabled=False)
        assert mm._lineage_registry is None


# ---------------------------------------------------------------------------
# save_checkpoint emits checkpoint_created
# ---------------------------------------------------------------------------


class TestSaveCheckpointLineage:
    def test_successful_save_emits_event(self, tmp_path):
        mm, registry, registry_path = _make_model_manager(tmp_path)
        model_dir = str(tmp_path / "models")

        agent = MagicMock()
        agent.save_model = _mock_agent_save

        success, path = mm.save_checkpoint(
            agent=agent,
            model_dir=model_dir,
            timestep=5000,
            episode_count=100,
            stats={"black_wins": 10, "white_wins": 8, "draws": 2},
            run_name="test-run",
            is_wandb_active=False,
        )

        assert success is True
        assert registry.event_count == 1

        event = registry.load_all()[0]
        assert event["event_type"] == "checkpoint_created"
        assert event["payload"]["global_timestep"] == 5000
        assert event["payload"]["total_episodes"] == 100
        assert event["payload"]["checkpoint_path"] == path
        assert validate_event(event) == []

    def test_failed_save_does_not_emit(self, tmp_path):
        mm, registry, _ = _make_model_manager(tmp_path)

        agent = MagicMock()
        agent.save_model.side_effect = OSError("disk full")

        success, _ = mm.save_checkpoint(
            agent=agent,
            model_dir=str(tmp_path / "models"),
            timestep=5000,
            episode_count=100,
            stats={},
            run_name="test-run",
            is_wandb_active=False,
        )

        assert success is False
        assert registry.event_count == 0

    def test_zero_timestep_does_not_emit(self, tmp_path):
        mm, registry, _ = _make_model_manager(tmp_path)
        agent = MagicMock()

        success, _ = mm.save_checkpoint(
            agent=agent,
            model_dir=str(tmp_path / "models"),
            timestep=0,
            episode_count=0,
            stats={},
            run_name="test-run",
            is_wandb_active=False,
        )

        assert success is False
        assert registry.event_count == 0

    def test_no_registry_save_still_works(self, tmp_path):
        """Checkpoint saves work fine without a lineage registry."""
        mm, _, _ = _make_model_manager(tmp_path, lineage_enabled=False)
        model_dir = str(tmp_path / "models")

        agent = MagicMock()
        agent.save_model = _mock_agent_save

        success, path = mm.save_checkpoint(
            agent=agent,
            model_dir=model_dir,
            timestep=5000,
            episode_count=100,
            stats={},
            run_name="test-run",
            is_wandb_active=False,
        )

        assert success is True
        assert path is not None


# ---------------------------------------------------------------------------
# save_final_checkpoint emits checkpoint_created
# ---------------------------------------------------------------------------


class TestSaveFinalCheckpointLineage:
    def test_final_checkpoint_emits_event(self, tmp_path):
        mm, registry, _ = _make_model_manager(tmp_path)
        model_dir = str(tmp_path / "models")

        agent = MagicMock()
        agent.save_model = _mock_agent_save

        success, path = mm.save_final_checkpoint(
            agent=agent,
            model_dir=model_dir,
            global_timestep=50000,
            total_episodes_completed=2000,
            game_stats={"black_wins": 500, "white_wins": 400, "draws": 100},
            run_name="test-run",
            is_wandb_active=False,
        )

        assert success is True
        assert registry.event_count == 1
        event = registry.load_all()[0]
        assert event["event_type"] == "checkpoint_created"
        assert event["payload"]["global_timestep"] == 50000


# ---------------------------------------------------------------------------
# save_final_model emits checkpoint_created
# ---------------------------------------------------------------------------


class TestSaveFinalModelLineage:
    def test_final_model_emits_event(self, tmp_path):
        mm, registry, _ = _make_model_manager(tmp_path)
        model_dir = str(tmp_path / "models")

        agent = MagicMock()
        agent.save_model = _mock_agent_save

        success, path = mm.save_final_model(
            agent=agent,
            model_dir=model_dir,
            global_timestep=100000,
            total_episodes_completed=5000,
            game_stats={"black_wins": 1000, "white_wins": 900, "draws": 100},
            run_name="test-run",
            is_wandb_active=False,
        )

        assert success is True
        assert registry.event_count == 1
        event = registry.load_all()[0]
        assert event["event_type"] == "checkpoint_created"
        assert event["payload"]["global_timestep"] == 100000
        assert "final_model.pth" in event["payload"]["checkpoint_path"]


# ---------------------------------------------------------------------------
# Parent model ID tracking
# ---------------------------------------------------------------------------


class TestParentModelId:
    def test_no_parent_on_fresh_start(self, tmp_path):
        mm, registry, _ = _make_model_manager(tmp_path)
        model_dir = str(tmp_path / "models")

        agent = MagicMock()
        agent.save_model = _mock_agent_save

        mm.save_checkpoint(
            agent=agent,
            model_dir=model_dir,
            timestep=5000,
            episode_count=100,
            stats={},
            run_name="test-run",
            is_wandb_active=False,
        )

        event = registry.load_all()[0]
        assert event["payload"]["parent_model_id"] is None

    def test_parent_set_from_checkpoint_resume(self, tmp_path):
        """After resuming, new checkpoints should reference the parent."""
        mm, registry, _ = _make_model_manager(tmp_path)

        # Simulate a checkpoint file to resume from
        ckpt_path = tmp_path / "models" / "checkpoint_ts10000.pth"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_text("mock")

        agent = MagicMock()
        agent.load_model.return_value = {"global_timestep": 10000}
        agent.save_model = _mock_agent_save

        # Resume from checkpoint
        mm._handle_specific_checkpoint_resume(agent, str(ckpt_path))

        # Now save a new checkpoint
        model_dir = str(tmp_path / "models")
        mm.save_checkpoint(
            agent=agent,
            model_dir=model_dir,
            timestep=15000,
            episode_count=200,
            stats={},
            run_name="test-run",
            is_wandb_active=False,
        )

        event = registry.load_all()[0]
        assert event["payload"]["parent_model_id"] == "test-run::checkpoint_ts10000"

    def test_parent_from_non_standard_filename(self, tmp_path):
        """Checkpoint paths without _ts pattern use raw path as parent."""
        mm, registry, _ = _make_model_manager(tmp_path)

        ckpt_path = tmp_path / "models" / "some_model.pth"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_text("mock")

        agent = MagicMock()
        agent.load_model.return_value = {}

        mm._handle_specific_checkpoint_resume(agent, str(ckpt_path))

        # parent_model_id should be the raw path (fallback)
        assert mm._parent_model_id == str(ckpt_path)


# ---------------------------------------------------------------------------
# training_started / training_resumed emission
# ---------------------------------------------------------------------------


class TestTrainingLifecycleEvents:
    def test_emit_training_started(self, tmp_path):
        mm, registry, _ = _make_model_manager(tmp_path)

        mm._emit_training_started({"learning_rate": 0.001})

        assert registry.event_count == 1
        event = registry.load_all()[0]
        assert event["event_type"] == "training_started"
        assert event["payload"]["config_snapshot"] == {"learning_rate": 0.001}
        assert validate_event(event) == []

    def test_emit_training_resumed(self, tmp_path):
        mm, registry, _ = _make_model_manager(tmp_path)

        mm._emit_training_resumed("/tmp/checkpoint_ts5000.pth", 5000)

        assert registry.event_count == 1
        event = registry.load_all()[0]
        assert event["event_type"] == "training_resumed"
        assert event["payload"]["resumed_from_checkpoint"] == "/tmp/checkpoint_ts5000.pth"
        assert event["payload"]["global_timestep_at_resume"] == 5000
        assert validate_event(event) == []

    def test_no_emission_without_registry(self, tmp_path):
        """Lifecycle events are silently skipped when no registry is wired."""
        mm, _, _ = _make_model_manager(tmp_path, lineage_enabled=False)
        # These should not raise
        mm._emit_training_started({"lr": 0.01})
        mm._emit_training_resumed("/tmp/ckpt.pth", 100)


# ---------------------------------------------------------------------------
# Multiple saves accumulate events
# ---------------------------------------------------------------------------


class TestEventAccumulation:
    def test_multiple_checkpoints_produce_ordered_events(self, tmp_path):
        mm, registry, _ = _make_model_manager(tmp_path)
        model_dir = str(tmp_path / "models")

        agent = MagicMock()
        agent.save_model = _mock_agent_save

        for ts in [5000, 10000, 15000]:
            mm.save_checkpoint(
                agent=agent,
                model_dir=model_dir,
                timestep=ts,
                episode_count=ts // 50,
                stats={},
                run_name="test-run",
                is_wandb_active=False,
            )

        assert registry.event_count == 3
        events = registry.load_all()
        timesteps = [e["payload"]["global_timestep"] for e in events]
        assert timesteps == [5000, 10000, 15000]

    def test_model_ids_are_unique_per_timestep(self, tmp_path):
        mm, registry, _ = _make_model_manager(tmp_path)
        model_dir = str(tmp_path / "models")

        agent = MagicMock()
        agent.save_model = _mock_agent_save

        for ts in [5000, 10000]:
            mm.save_checkpoint(
                agent=agent,
                model_dir=model_dir,
                timestep=ts,
                episode_count=0,
                stats={},
                run_name="test-run",
                is_wandb_active=False,
            )

        events = registry.load_all()
        model_ids = [e["model_id"] for e in events]
        assert model_ids[0] != model_ids[1]
        assert "5000" in model_ids[0]
        assert "10000" in model_ids[1]


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------


class TestLineageRoundTrip:
    def test_events_survive_registry_reload(self, tmp_path):
        """Events written by ModelManager can be reloaded by a fresh registry."""
        mm, registry, registry_path = _make_model_manager(tmp_path)
        model_dir = str(tmp_path / "models")

        agent = MagicMock()
        agent.save_model = _mock_agent_save

        mm.save_checkpoint(
            agent=agent,
            model_dir=model_dir,
            timestep=5000,
            episode_count=100,
            stats={},
            run_name="test-run",
            is_wandb_active=False,
        )

        # Reload from the same file
        registry2 = LineageRegistry(registry_path)
        assert registry2.event_count == 1
        event = registry2.load_all()[0]
        assert event["event_type"] == "checkpoint_created"
        assert event["payload"]["global_timestep"] == 5000
