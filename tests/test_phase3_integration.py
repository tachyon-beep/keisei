"""Phase 3 integration tests — rollout collection from match_utils."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from keisei.config import DynamicConfig, FrontierStaticConfig
from keisei.db import init_db
from keisei.training.dynamic_trainer import DynamicTrainer, MatchRollout
from keisei.training.frontier_promoter import FrontierPromoter
from keisei.training.match_utils import play_batch, play_match
from keisei.training.opponent_store import EloColumn, EntryStatus, OpponentEntry, OpponentStore, Role
from keisei.training.tier_managers import FrontierManager
from keisei.training.tournament import LeagueTournament

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockModel(nn.Module):
    """Minimal model that returns random policy logits and a dummy value."""

    def __init__(self, action_dim: int = 11259) -> None:
        super().__init__()
        self.action_dim = action_dim
        # Need at least one parameter so PyTorch considers it a Module
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        batch = x.shape[0]
        return SimpleNamespace(
            policy_logits=torch.randn(batch, self.action_dim),
            value=torch.zeros(batch, 1),
        )


class MockVecEnv:
    """Minimal VecEnv mock for testing match_utils."""

    def __init__(self, num_envs: int = 2, obs_channels: int = 50, end_ply: int = 5) -> None:
        self.num_envs = num_envs
        self.obs_channels = obs_channels
        self.end_ply = end_ply
        self._step_count = 0
        self._current_player = 0  # alternates each step

    def reset(self) -> SimpleNamespace:
        self._step_count = 0
        self._current_player = 0
        return SimpleNamespace(
            observations=np.random.randn(
                self.num_envs, self.obs_channels, 9, 9
            ).astype(np.float32),
            legal_masks=np.ones((self.num_envs, 11259), dtype=bool),
        )

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self._step_count += 1
        old_player = self._current_player
        self._current_player = 1 - self._current_player

        terminated = self._step_count >= self.end_ply
        rewards = np.array(
            [1.0 if terminated else 0.0] * self.num_envs, dtype=np.float32
        )

        return SimpleNamespace(
            observations=np.random.randn(
                self.num_envs, self.obs_channels, 9, 9
            ).astype(np.float32),
            legal_masks=np.ones((self.num_envs, 11259), dtype=bool),
            current_players=np.full(
                self.num_envs, self._current_player, dtype=np.uint8
            ),
            rewards=rewards,
            terminated=np.full(self.num_envs, terminated),
            truncated=np.full(self.num_envs, False),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_play_batch_collects_rollout_when_requested():
    """play_batch with collect_rollout=True returns a 4-tuple with MatchRollout."""
    vecenv = MockVecEnv(num_envs=2, end_ply=5)
    model_a = _MockModel()
    model_b = _MockModel()

    result = play_batch(
        vecenv, model_a, model_b,
        device=torch.device("cpu"),
        num_envs=2,
        max_ply=20,
        collect_rollout=True,
    )

    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"
    a_wins, b_wins, draws, rollout = result
    assert isinstance(rollout, MatchRollout)
    assert rollout.observations.shape[0] > 0
    # All tensors should be on CPU
    for name in ("observations", "actions", "rewards", "dones", "legal_masks", "perspective"):
        t = getattr(rollout, name)
        assert t.device == torch.device("cpu"), f"{name} not on CPU"


def test_play_batch_no_rollout_by_default():
    """play_batch without collect_rollout returns a 3-tuple."""
    vecenv = MockVecEnv(num_envs=2, end_ply=3)
    model_a = _MockModel()
    model_b = _MockModel()

    result = play_batch(
        vecenv, model_a, model_b,
        device=torch.device("cpu"),
        num_envs=2,
        max_ply=20,
    )

    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
    a_wins, b_wins, draws = result
    assert isinstance(a_wins, int)


def test_play_batch_rollout_has_correct_perspective():
    """Perspective tensor alternates between 0 and 1 each ply."""
    vecenv = MockVecEnv(num_envs=2, end_ply=6)
    model_a = _MockModel()
    model_b = _MockModel()

    result = play_batch(
        vecenv, model_a, model_b,
        device=torch.device("cpu"),
        num_envs=2,
        max_ply=20,
        collect_rollout=True,
    )
    _, _, _, rollout = result
    perspective = rollout.perspective

    # MockVecEnv starts current_player=0 and alternates each step.
    # Perspective is captured BEFORE step, so:
    # ply 0: current_players=0 (init), ply 1: current_players=1 (after step flips), etc.
    for ply in range(perspective.shape[0]):
        expected = ply % 2
        assert (perspective[ply] == expected).all(), (
            f"Ply {ply}: expected all {expected}, got {perspective[ply]}"
        )


def test_play_batch_perspective_captured_before_step():
    """Perspective at ply 0 reflects the INITIAL current_players (before first step)."""
    vecenv = MockVecEnv(num_envs=2, end_ply=3)
    model_a = _MockModel()
    model_b = _MockModel()

    result = play_batch(
        vecenv, model_a, model_b,
        device=torch.device("cpu"),
        num_envs=2,
        max_ply=20,
        collect_rollout=True,
    )
    _, _, _, rollout = result

    # current_players is initialized to zeros before the loop.
    # Perspective should capture this BEFORE the first step() call.
    assert (rollout.perspective[0] == 0).all(), (
        f"First perspective should be 0 (initial), got {rollout.perspective[0]}"
    )


def test_play_match_collects_rollout():
    """play_match with collect_rollout=True threads through and returns 4-tuple."""
    vecenv = MockVecEnv(num_envs=2, end_ply=3)
    model_a = _MockModel()
    model_b = _MockModel()

    result = play_match(
        vecenv, model_a, model_b,
        device=torch.device("cpu"),
        num_envs=2,
        max_ply=20,
        games_target=4,
        collect_rollout=True,
    )

    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"
    a_wins, b_wins, draws, rollout = result
    assert isinstance(rollout, MatchRollout)
    assert rollout.observations.shape[0] > 0


# ---------------------------------------------------------------------------
# Task 13: Tournament training trigger tests
# ---------------------------------------------------------------------------


def _make_entry(entry_id: int, role: Role, name: str = "test") -> OpponentEntry:
    """Create a minimal OpponentEntry for testing."""
    return OpponentEntry(
        id=entry_id,
        display_name=f"{name}-{entry_id}",
        architecture="test",
        model_params={},
        checkpoint_path=f"/tmp/model_{entry_id}.pt",
        elo_rating=1500.0,
        created_epoch=0,
        games_played=0,
        created_at="2026-01-01T00:00:00",
        flavour_facts=[],
        role=role,
    )


def _make_mock_rollout() -> MatchRollout:
    """Create a minimal MatchRollout for testing."""
    n_steps = 4
    n_envs = 2
    return MatchRollout(
        observations=torch.randn(n_steps, n_envs, 50, 9, 9),
        actions=torch.randint(0, 100, (n_steps, n_envs)),
        rewards=torch.zeros(n_steps, n_envs),
        dones=torch.zeros(n_steps, n_envs, dtype=torch.bool),
        legal_masks=torch.ones(n_steps, n_envs, 11259, dtype=torch.bool),
        perspective=torch.zeros(n_steps, n_envs, dtype=torch.long),
    )


class TestTournamentTrainingTrigger:
    """Tests for tournament triggering Dynamic training after trainable matches."""

    def _make_tournament(self, dynamic_trainer=None):
        """Build a LeagueTournament with mocked store/scheduler."""
        store = MagicMock()
        scheduler = MagicMock()
        return LeagueTournament(
            store=store,
            scheduler=scheduler,
            device="cpu",
            num_envs=2,
            max_ply=20,
            games_per_match=4,
            dynamic_trainer=dynamic_trainer,
        )

    def test_is_trainable_match_dynamic_vs_dynamic(self):
        """D-vs-D is trainable."""
        t = self._make_tournament()
        a = _make_entry(1, Role.DYNAMIC)
        b = _make_entry(2, Role.DYNAMIC)
        assert t._is_trainable_match(a, b) is True

    def test_is_trainable_match_dynamic_vs_recent_fixed(self):
        """D-vs-RF is trainable."""
        t = self._make_tournament()
        a = _make_entry(1, Role.DYNAMIC)
        b = _make_entry(2, Role.RECENT_FIXED)
        assert t._is_trainable_match(a, b) is True

    def test_is_trainable_match_rf_vs_dynamic(self):
        """RF-vs-D is trainable (order reversed)."""
        t = self._make_tournament()
        a = _make_entry(1, Role.RECENT_FIXED)
        b = _make_entry(2, Role.DYNAMIC)
        assert t._is_trainable_match(a, b) is True

    def test_is_not_trainable_dynamic_vs_frontier(self):
        """D-vs-FS is NOT trainable."""
        t = self._make_tournament()
        a = _make_entry(1, Role.DYNAMIC)
        b = _make_entry(2, Role.FRONTIER_STATIC)
        assert t._is_trainable_match(a, b) is False

    def test_is_not_trainable_rf_vs_rf(self):
        """RF-vs-RF (no Dynamic) is NOT trainable."""
        t = self._make_tournament()
        a = _make_entry(1, Role.RECENT_FIXED)
        b = _make_entry(2, Role.RECENT_FIXED)
        assert t._is_trainable_match(a, b) is False

    def test_tournament_triggers_training_after_dynamic_match(self):
        """After a trainable D-vs-D match, record_match + update are called."""
        trainer = MagicMock(spec=DynamicTrainer)
        trainer.should_update.return_value = True
        trainer.is_rate_limited.return_value = False
        trainer.update.return_value = True

        t = self._make_tournament(dynamic_trainer=trainer)

        entry_a = _make_entry(1, Role.DYNAMIC)
        entry_b = _make_entry(2, Role.DYNAMIC)
        rollout = _make_mock_rollout()

        # Mock _play_match to return (wins_a, wins_b, draws, rollout)
        with patch.object(t, "_play_match", return_value=(2, 1, 1, rollout)):
            t.store.get_entry.side_effect = lambda eid: (
                entry_a if eid == 1 else entry_b
            )
            t._run_one_match(MagicMock(), entry_a, entry_b, epoch=10)

        # Both entries are Dynamic, so record_match called for both
        assert trainer.record_match.call_count == 2
        trainer.record_match.assert_any_call(1, rollout, side=0)
        trainer.record_match.assert_any_call(2, rollout, side=1)
        # should_update returns True and not rate limited => update called for both
        assert trainer.update.call_count == 2
        updated_ids = {call.args[0].id for call in trainer.update.call_args_list}
        assert updated_ids == {1, 2}, f"Expected updates for entries 1 and 2, got {updated_ids}"

    def test_tournament_skips_training_for_non_trainable_match(self):
        """D-vs-FS match does NOT trigger training."""
        trainer = MagicMock(spec=DynamicTrainer)
        t = self._make_tournament(dynamic_trainer=trainer)

        entry_a = _make_entry(1, Role.DYNAMIC)
        entry_b = _make_entry(2, Role.FRONTIER_STATIC)

        assert not t._is_trainable_match(entry_a, entry_b)

        # 3-tuple is correct: non-trainable path calls _play_match without
        # collect_rollout, so it returns (wins_a, wins_b, draws) only.
        with patch.object(t, "_play_match", return_value=(2, 1, 1)):
            t.store.get_entry.side_effect = lambda eid: (
                entry_a if eid == 1 else entry_b
            )
            t._run_one_match(MagicMock(), entry_a, entry_b, epoch=10)

        trainer.record_match.assert_not_called()
        trainer.should_update.assert_not_called()
        trainer.update.assert_not_called()

    def test_tournament_respects_rate_limit(self):
        """When is_rate_limited() returns True, update is NOT called."""
        trainer = MagicMock(spec=DynamicTrainer)
        trainer.should_update.return_value = True
        trainer.is_rate_limited.return_value = True  # rate limited!

        t = self._make_tournament(dynamic_trainer=trainer)
        entry_a = _make_entry(1, Role.DYNAMIC)
        entry_b = _make_entry(2, Role.DYNAMIC)
        rollout = _make_mock_rollout()

        with patch.object(t, "_play_match", return_value=(2, 1, 1, rollout)):
            t.store.get_entry.side_effect = lambda eid: (
                entry_a if eid == 1 else entry_b
            )
            t._run_one_match(MagicMock(), entry_a, entry_b, epoch=10)

        assert trainer.record_match.call_count == 2
        trainer.update.assert_not_called()

    def test_tournament_triggers_training_for_dynamic_side_only_in_drf_match(self):
        """In a D-vs-RF match, only the Dynamic entry gets record_match."""
        trainer = MagicMock(spec=DynamicTrainer)
        trainer.should_update.return_value = True
        trainer.is_rate_limited.return_value = False
        trainer.update.return_value = True

        t = self._make_tournament(dynamic_trainer=trainer)

        entry_a = _make_entry(1, Role.DYNAMIC)
        entry_b = _make_entry(2, Role.RECENT_FIXED)
        rollout = _make_mock_rollout()

        with patch.object(t, "_play_match", return_value=(2, 1, 1, rollout)):
            t.store.get_entry.side_effect = lambda eid: (
                entry_a if eid == 1 else entry_b
            )
            t._run_one_match(MagicMock(), entry_a, entry_b, epoch=10)

        # Only the Dynamic entry (entry_a, side=0) should get record_match
        trainer.record_match.assert_called_once_with(1, rollout, side=0)
        # update called once (for Dynamic entry only)
        trainer.update.assert_called_once()

    def test_tournament_triggers_training_for_dynamic_side_only_in_rfd_match(self):
        """In a RF-vs-D match, only the Dynamic entry (side=1) gets record_match."""
        trainer = MagicMock(spec=DynamicTrainer)
        trainer.should_update.return_value = True
        trainer.is_rate_limited.return_value = False
        trainer.update.return_value = True

        t = self._make_tournament(dynamic_trainer=trainer)

        entry_a = _make_entry(1, Role.RECENT_FIXED)
        entry_b = _make_entry(2, Role.DYNAMIC)
        rollout = _make_mock_rollout()

        with patch.object(t, "_play_match", return_value=(2, 1, 1, rollout)):
            t.store.get_entry.side_effect = lambda eid: (
                entry_a if eid == 1 else entry_b
            )
            t._run_one_match(MagicMock(), entry_a, entry_b, epoch=10)

        # Only the Dynamic entry (entry_b, side=1) should get record_match
        trainer.record_match.assert_called_once_with(2, rollout, side=1)
        trainer.update.assert_called_once()

    def test_tournament_no_training_when_trainer_is_none(self):
        """dynamic_trainer=None should not crash."""
        t = self._make_tournament(dynamic_trainer=None)
        entry_a = _make_entry(1, Role.DYNAMIC)
        entry_b = _make_entry(2, Role.DYNAMIC)

        assert t._is_trainable_match(entry_a, entry_b) is True

        # 3-tuple is correct despite D-vs-D being trainable by _is_trainable_match:
        # with dynamic_trainer=None, is_trainable evaluates to False in _run_one_match,
        # so _play_match is called without collect_rollout and returns 3-tuple.
        with patch.object(t, "_play_match", return_value=(2, 1, 1)):
            t.store.get_entry.side_effect = lambda eid: (
                entry_a if eid == 1 else entry_b
            )
            # Should not raise
            t._run_one_match(MagicMock(), entry_a, entry_b, epoch=10)


# ---------------------------------------------------------------------------
# Helpers for end-to-end tests (Task 15) — shared via conftest.py
# ---------------------------------------------------------------------------

from tests._helpers import TinyModel as _TinyModel, make_rollout as _make_synthetic_rollout


# ---------------------------------------------------------------------------
# Test 1: Full Dynamic training cycle (Task 15)
# ---------------------------------------------------------------------------


class TestFullDynamicTrainingCycle:
    """End-to-end: tournament match -> DynamicTrainer update -> weights change."""

    def test_full_dynamic_training_cycle(self, tmp_path: Path) -> None:
        # --- Setup: DB + store ---
        db_path = str(tmp_path / "test.db")
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        init_db(db_path)

        store = OpponentStore(db_path=db_path, league_dir=str(league_dir))

        # Create two Dynamic entries with small test models
        model_a = _TinyModel()
        model_b = _TinyModel()

        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=_TinyModel(),
        ):
            entry_a = store.add_entry(
                model=model_a,
                architecture="tiny",
                model_params={},
                epoch=1,
                role=Role.DYNAMIC,
            )
            entry_b = store.add_entry(
                model=model_b,
                architecture="tiny",
                model_params={},
                epoch=2,
                role=Role.DYNAMIC,
            )

        # --- Snapshot weights before update ---
        ckpt_a_path = Path(entry_a.checkpoint_path)
        before_hash = hashlib.md5(ckpt_a_path.read_bytes()).hexdigest()
        before_mtime = ckpt_a_path.stat().st_mtime

        # --- DynamicTrainer with immediate update ---
        config = DynamicConfig(update_every_matches=1)
        trainer = DynamicTrainer(store=store, config=config, learner_lr=1e-3)

        # --- Record a match and update ---
        rollout = _make_synthetic_rollout(steps=10, num_envs=1, side=0)
        trainer.record_match(entry_a.id, rollout, 0)

        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=_TinyModel(),
        ):
            result = trainer.update(entry_a, device="cpu")

        assert result is True

        # --- Assertions ---
        # 1. Weights changed
        after_hash = hashlib.md5(ckpt_a_path.read_bytes()).hexdigest()
        assert before_hash != after_hash, "Checkpoint file should differ after update"

        # 2. update_count incremented
        refreshed_a = store.get_entry(entry_a.id)
        assert refreshed_a is not None
        assert refreshed_a.update_count == 1

        # 3. last_train_at set
        assert refreshed_a.last_train_at is not None

        # 4. Checkpoint file on disk was updated (hash check above is authoritative;
        #    mtime may equal before_mtime on fast filesystems with coarse timestamps)
        assert ckpt_a_path.exists()

        store.close()


# ---------------------------------------------------------------------------
# Test 2: Full Frontier promotion cycle (Task 15)
# ---------------------------------------------------------------------------


class TestFullFrontierPromotionCycle:
    """End-to-end: Dynamic entry meets criteria -> promoted to Frontier."""

    def test_full_frontier_promotion_cycle(self, tmp_path: Path) -> None:
        # --- Setup: DB + store ---
        db_path = str(tmp_path / "test.db")
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        init_db(db_path)

        store = OpponentStore(db_path=db_path, league_dir=str(league_dir))

        dummy_model = _TinyModel()

        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=_TinyModel(),
        ):
            # --- Create 5 Frontier entries with descending Elo ---
            frontier_elos = [1200.0, 1150.0, 1100.0, 1050.0, 1000.0]
            frontier_ids: list[int] = []
            for i, elo in enumerate(frontier_elos):
                entry = store.add_entry(
                    model=dummy_model,
                    architecture="tiny",
                    model_params={},
                    epoch=i + 1,
                    role=Role.FRONTIER_STATIC,
                )
                store.update_elo(entry.id, elo)
                store.update_role_elo(entry.id, EloColumn.FRONTIER, elo)
                frontier_ids.append(entry.id)

            # --- Create 10 Dynamic entries ---
            dynamic_ids: list[int] = []
            for i in range(10):
                entry = store.add_entry(
                    model=dummy_model,
                    architecture="tiny",
                    model_params={},
                    epoch=100 + i,
                    role=Role.DYNAMIC,
                )
                store.update_elo(entry.id, 1050.0 + i * 10)
                store.update_role_elo(entry.id, EloColumn.FRONTIER, 1050.0 + i * 10)
                # Record enough games to meet min_games_for_promotion
                store.record_result(
                    epoch=100,
                    learner_id=entry.id,
                    opponent_id=frontier_ids[0],
                    wins=100,
                    losses=100,
                    draws=0,
                )
                dynamic_ids.append(entry.id)

            # --- Best Dynamic entry: Elo 1250, unique lineage ---
            best_dynamic_id = dynamic_ids[-1]
            store.update_elo(best_dynamic_id, 1250.0)
            store.update_role_elo(best_dynamic_id, EloColumn.FRONTIER, 1250.0)
            store._conn.execute(
                "UPDATE league_entries SET lineage_group = ? WHERE id = ?",
                ("lineage-new", best_dynamic_id),
            )
            store._conn.commit()

        # --- FrontierPromoter with streak_epochs=1 (fast for testing) ---
        frontier_config = FrontierStaticConfig(
            slots=5,
            streak_epochs=1,
            min_games_for_promotion=100,
            min_tenure_epochs=1,
            promotion_margin_elo=50.0,
        )
        promoter = FrontierPromoter(config=frontier_config)
        frontier_manager = FrontierManager(
            store=store, config=frontier_config, promoter=promoter,
        )
        # Verify the manager uses our promoter instance (streak state continuity)
        assert frontier_manager._promoter is promoter

        # --- Build streak: two consecutive evaluate() calls ---
        dynamic_entries = store.list_by_role(Role.DYNAMIC)
        frontier_entries = store.list_by_role(Role.FRONTIER_STATIC)

        # First eval registers in top-K; streak not yet met
        candidate = promoter.evaluate(dynamic_entries, frontier_entries, epoch=100)
        assert candidate is None or candidate.id == best_dynamic_id

        # Second eval at epoch 101: streak_epochs=1 satisfied (101-100 >= 1)
        candidate = promoter.evaluate(dynamic_entries, frontier_entries, epoch=101)
        assert candidate is not None
        assert candidate.id == best_dynamic_id

        # --- Trigger the full review cycle ---
        frontier_manager.review(epoch=101)

        # --- Assertions ---
        frontier_after = store.list_by_role(Role.FRONTIER_STATIC)
        frontier_ids_after = {e.id for e in frontier_after}

        # 1. Weakest frontier entry (Elo 1000) was retired
        weakest_id = frontier_ids[4]
        weakest_entry = store.get_entry(weakest_id)
        assert weakest_entry is not None
        assert weakest_entry.status == EntryStatus.RETIRED, (
            f"Weakest frontier entry should be retired, got {weakest_entry.status}"
        )

        # 2. Total active Frontier entries == 5 (one added, one retired)
        assert len(frontier_after) == 5, (
            f"Expected 5 active frontier entries, got {len(frontier_after)}"
        )

        # 3. The promoted Dynamic entry still exists as Dynamic
        best_dynamic = store.get_entry(best_dynamic_id)
        assert best_dynamic is not None
        assert best_dynamic.role == Role.DYNAMIC
        assert best_dynamic.status == EntryStatus.ACTIVE

        # 4. A new Frontier Static entry was created (clone of the promoted Dynamic)
        new_frontier_ids = frontier_ids_after - set(frontier_ids)
        assert len(new_frontier_ids) >= 1, "Expected at least one new frontier entry"

        new_frontier_entry = store.get_entry(new_frontier_ids.pop())
        assert new_frontier_entry is not None
        assert new_frontier_entry.parent_entry_id == best_dynamic_id

        store.close()
