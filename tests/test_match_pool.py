"""Tests for ConcurrentMatchPool — partitioned concurrent match execution."""

from __future__ import annotations

import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from keisei.config import ConcurrencyConfig
from keisei.training.concurrent_matches import ConcurrentMatchPool, MatchResult, _MatchSlot
from keisei.training.opponent_store import OpponentEntry, Role
from tests._helpers import TinyModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockVecEnv:
    """Deterministic mock that terminates envs after N steps with reward +1.0."""

    def __init__(self, num_envs: int, terminate_after: int = 3) -> None:
        self.num_envs = num_envs
        self.terminate_after = terminate_after
        self._ply = np.zeros(num_envs, dtype=int)

    def reset(self) -> SimpleNamespace:
        self._ply = np.zeros(self.num_envs, dtype=int)
        return SimpleNamespace(
            observations=np.random.randn(self.num_envs, 50, 9, 9).astype(np.float32),
            legal_masks=np.ones((self.num_envs, 11259), dtype=bool),
        )

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self._ply += 1
        terminated = self._ply >= self.terminate_after
        rewards = np.where(terminated, 1.0, 0.0).astype(np.float32)
        self._ply[terminated] = 0
        return SimpleNamespace(
            observations=np.random.randn(self.num_envs, 50, 9, 9).astype(np.float32),
            legal_masks=np.ones((self.num_envs, 11259), dtype=bool),
            current_players=np.zeros(self.num_envs, dtype=np.uint8),
            rewards=rewards,
            terminated=terminated,
            truncated=np.zeros(self.num_envs, dtype=bool),
        )


def _make_entry(entry_id: int, name: str = "test") -> OpponentEntry:
    """Create a minimal OpponentEntry for testing."""
    return OpponentEntry(
        id=entry_id,
        display_name=f"{name}-{entry_id}",
        architecture="resnet",
        model_params={},
        checkpoint_path="/fake/path.pt",
        elo_rating=1000.0,
        created_epoch=0,
        games_played=0,
        created_at="2026-01-01T00:00:00",
        flavour_facts=[],
        role=Role.FRONTIER_STATIC,
    )


def _make_pool(
    parallel_matches: int = 2,
    envs_per_match: int = 2,
    total_envs: int = 4,
    max_resident_models: int = 4,
) -> ConcurrentMatchPool:
    config = ConcurrencyConfig(
        parallel_matches=parallel_matches,
        envs_per_match=envs_per_match,
        total_envs=total_envs,
        max_resident_models=max_resident_models,
    )
    return ConcurrentMatchPool(config)


# ---------------------------------------------------------------------------
# TestMatchResult
# ---------------------------------------------------------------------------


class TestMatchResult:
    def test_match_result_fields(self) -> None:
        entry_a = _make_entry(1)
        entry_b = _make_entry(2)
        result = MatchResult(
            entry_a=entry_a,
            entry_b=entry_b,
            a_wins=3,
            b_wins=1,
            draws=0,
            rollout=None,
        )
        assert result.entry_a is entry_a
        assert result.entry_b is entry_b
        assert result.a_wins == 3
        assert result.b_wins == 1
        assert result.draws == 0
        assert result.rollout is None


# ---------------------------------------------------------------------------
# TestMatchSlot
# ---------------------------------------------------------------------------


class TestMatchSlot:
    def test_games_completed(self) -> None:
        slot = _MatchSlot(index=0, env_start=0, env_end=2)
        assert slot.games_completed == 0
        slot.a_wins = 2
        slot.b_wins = 1
        slot.draws = 1
        assert slot.games_completed == 4

    def test_reset_for_pairing(self) -> None:
        entry_a = _make_entry(1)
        entry_b = _make_entry(2)
        model_a = TinyModel()
        model_b = TinyModel()
        slot = _MatchSlot(index=0, env_start=0, env_end=2)
        slot.reset_for_pairing(
            entry_a=entry_a,
            entry_b=entry_b,
            model_a=model_a,
            model_b=model_b,
            games_target=4,
            collect_rollout=False,
        )
        assert slot.entry_a is entry_a
        assert slot.model_a is model_a
        assert slot.games_target == 4
        assert slot.active is True
        assert slot.a_wins == 0

    def test_to_result(self) -> None:
        entry_a = _make_entry(1)
        entry_b = _make_entry(2)
        slot = _MatchSlot(index=0, env_start=0, env_end=2)
        slot.reset_for_pairing(
            entry_a=entry_a,
            entry_b=entry_b,
            model_a=TinyModel(),
            model_b=TinyModel(),
            games_target=4,
            collect_rollout=False,
        )
        slot.a_wins = 3
        slot.b_wins = 1
        result = slot.to_result()
        assert isinstance(result, MatchResult)
        assert result.a_wins == 3
        assert result.b_wins == 1
        assert result.rollout is None


# ---------------------------------------------------------------------------
# TestRunRound
# ---------------------------------------------------------------------------


class TestRunRound:
    def test_processes_all_pairings(self) -> None:
        """2 parallel slots, 2 pairings -> 2 results, each with >= 4 games."""
        pool = _make_pool(parallel_matches=2, envs_per_match=2, total_envs=4)
        vecenv = MockVecEnv(num_envs=4, terminate_after=3)
        entries = [_make_entry(i) for i in range(4)]
        pairings = [
            (entries[0], entries[1]),
            (entries[2], entries[3]),
        ]

        results = pool.run_round(
            vecenv,
            pairings,
            load_fn=lambda entry: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=4,
        )
        assert len(results) == 2
        for r in results:
            assert isinstance(r, MatchResult)
            total_games = r.a_wins + r.b_wins + r.draws
            assert total_games >= 4

    def test_more_pairings_than_parallel_slots(self) -> None:
        """1 slot, 3 pairings -> 3 results in correct order with games completed."""
        pool = _make_pool(parallel_matches=1, envs_per_match=2, total_envs=2, max_resident_models=2)
        vecenv = MockVecEnv(num_envs=2, terminate_after=2)
        entries = [_make_entry(i) for i in range(6)]
        pairings = [
            (entries[0], entries[1]),
            (entries[2], entries[3]),
            (entries[4], entries[5]),
        ]

        results = pool.run_round(
            vecenv,
            pairings,
            load_fn=lambda entry: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=4,
        )
        assert len(results) == 3
        # Check ordering matches pairing order
        assert results[0].entry_a.id == 0
        assert results[0].entry_b.id == 1
        assert results[1].entry_a.id == 2
        assert results[1].entry_b.id == 3
        assert results[2].entry_a.id == 4
        assert results[2].entry_b.id == 5
        # Each pairing should have completed the target number of games
        for r in results:
            total = r.a_wins + r.b_wins + r.draws
            assert total >= 4, f"Pairing {r.entry_a.id}v{r.entry_b.id} only completed {total} games"

    def test_empty_pairings(self) -> None:
        """[] -> []"""
        pool = _make_pool()
        vecenv = MockVecEnv(num_envs=4)
        results = pool.run_round(
            vecenv,
            [],
            load_fn=lambda entry: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
        )
        assert results == []

    def test_stop_event_interrupts(self) -> None:
        """Pre-set stop -> returns empty list without loading any models."""
        pool = _make_pool()
        vecenv = MockVecEnv(num_envs=4)
        entries = [_make_entry(i) for i in range(4)]
        pairings = [(entries[0], entries[1]), (entries[2], entries[3])]
        stop_event = threading.Event()
        stop_event.set()  # Set before calling

        load_calls = []
        results = pool.run_round(
            vecenv,
            pairings,
            load_fn=lambda entry: load_calls.append(entry.id) or TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            stop_event=stop_event,
        )
        assert len(results) == 0
        assert len(load_calls) == 0, "load_fn should never be called when stop is pre-set"

    def test_rollout_collection_for_trainable(self) -> None:
        """trainable_fn=True -> rollout not None, observations.ndim==5 (steps, envs, C, 9, 9)."""
        pool = _make_pool(parallel_matches=1, envs_per_match=2, total_envs=2, max_resident_models=2)
        vecenv = MockVecEnv(num_envs=2, terminate_after=2)
        entries = [_make_entry(i) for i in range(2)]
        pairings = [(entries[0], entries[1])]

        results = pool.run_round(
            vecenv,
            pairings,
            load_fn=lambda entry: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=4,
            trainable_fn=lambda ea, eb: True,
        )
        assert len(results) == 1
        assert results[0].rollout is not None
        # MatchRollout.observations shape: (steps, num_envs, channels, 9, 9) = ndim 5
        assert results[0].rollout.observations.ndim == 5

    def test_no_rollout_when_trainable_fn_false(self) -> None:
        """trainable_fn=False -> rollout is None."""
        pool = _make_pool(parallel_matches=1, envs_per_match=2, total_envs=2, max_resident_models=2)
        vecenv = MockVecEnv(num_envs=2, terminate_after=2)
        entries = [_make_entry(i) for i in range(2)]
        pairings = [(entries[0], entries[1])]

        results = pool.run_round(
            vecenv,
            pairings,
            load_fn=lambda entry: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=4,
            trainable_fn=lambda ea, eb: False,
        )
        assert len(results) == 1
        assert results[0].rollout is None


# ---------------------------------------------------------------------------
# TestMaxResidentModels
# ---------------------------------------------------------------------------


class TestMaxResidentModels:
    def test_concurrent_slots_capped_by_max_resident(self) -> None:
        """With 4 parallel slots but max_resident=4 (2 slots worth),
        only 2 slots should be active at once."""
        config = ConcurrencyConfig(
            parallel_matches=4,
            envs_per_match=2,
            total_envs=8,
            max_resident_models=4,  # only 2 slots can be active (4 // 2)
        )
        pool = ConcurrentMatchPool(config)
        vecenv = MockVecEnv(num_envs=8, terminate_after=2)
        entries = [_make_entry(i) for i in range(8)]
        pairings = [
            (entries[0], entries[1]),
            (entries[2], entries[3]),
            (entries[4], entries[5]),
            (entries[6], entries[7]),
        ]

        # Track peak concurrent model loads
        loaded_models: list[object] = []
        peak_loaded = 0

        def tracking_load(entry: OpponentEntry) -> TinyModel:
            nonlocal peak_loaded
            m = TinyModel()
            loaded_models.append(m)
            peak_loaded = max(peak_loaded, len(loaded_models))
            return m

        def tracking_release(ma: object, mb: object) -> None:
            if ma in loaded_models:
                loaded_models.remove(ma)
            if mb in loaded_models:
                loaded_models.remove(mb)

        results = pool.run_round(
            vecenv,
            pairings,
            load_fn=tracking_load,
            release_fn=tracking_release,
            device="cpu",
            games_per_match=4,
        )

        assert len(results) == 4  # all pairings completed
        assert peak_loaded <= 4, (
            f"Peak loaded models {peak_loaded} exceeded max_resident_models 4"
        )


# ---------------------------------------------------------------------------
# TestRunRoundStopEventMidGame
# ---------------------------------------------------------------------------


class TestRunRoundStopEventMidGame:
    def test_run_round_stop_event_mid_game(self) -> None:
        """Set stop_event after N step calls; verify partial results and models released."""
        pool = _make_pool(parallel_matches=1, envs_per_match=2, total_envs=2, max_resident_models=2)
        stop_event = threading.Event()
        step_count = 0

        class StoppingVecEnv(MockVecEnv):
            """MockVecEnv that sets stop_event after a few steps."""

            def step(self, actions: np.ndarray) -> SimpleNamespace:
                nonlocal step_count
                step_count += 1
                if step_count >= 2:
                    stop_event.set()
                return super().step(actions)

        vecenv = StoppingVecEnv(num_envs=2, terminate_after=100)  # never terminates naturally
        entries = [_make_entry(i) for i in range(2)]
        pairings = [(entries[0], entries[1])]

        released_models: list[object] = []

        def track_release(ma: object, mb: object) -> None:
            released_models.append(ma)
            released_models.append(mb)

        results = pool.run_round(
            vecenv,
            pairings,
            load_fn=lambda entry: TinyModel(),
            release_fn=track_release,
            device="cpu",
            games_per_match=1000,  # won't complete
            stop_event=stop_event,
        )
        # Partial results — game didn't finish, so no completed results
        assert len(results) == 0
        # Models should still be released (cleanup for active slots on stop_event)
        assert len(released_models) == 2, (
            f"Expected 2 released models (model_a + model_b), got {len(released_models)}"
        )


# ---------------------------------------------------------------------------
# TestRunRoundModelBLoadFailure
# ---------------------------------------------------------------------------


class TestRunRoundModelBLoadFailure:
    def test_run_round_model_b_load_failure(self) -> None:
        """model_a loads, model_b raises -> model_a must not be leaked, pool continues."""
        # Use 2 parallel slots so both pairings get assigned during initial assignment.
        # Pairing 0 -> slot 0 (will fail), pairing 1 -> slot 1 (should succeed).
        pool = _make_pool(parallel_matches=2, envs_per_match=2, total_envs=4, max_resident_models=4)
        vecenv = MockVecEnv(num_envs=4, terminate_after=2)
        entries = [_make_entry(i) for i in range(4)]
        pairings = [
            (entries[0], entries[1]),  # will fail on model_b
            (entries[2], entries[3]),  # should succeed
        ]

        load_call_count = 0

        def failing_load(entry: OpponentEntry) -> TinyModel:
            nonlocal load_call_count
            load_call_count += 1
            # First call: model_a for pairing 0 -> OK
            # Second call: model_b for pairing 0 -> FAIL
            # Third+ calls: pairing 1 -> OK
            if load_call_count == 2:
                raise RuntimeError("Simulated model_b load failure")
            return TinyModel()

        results = pool.run_round(
            vecenv,
            pairings,
            load_fn=failing_load,
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=4,
        )

        # First pairing skipped due to load failure, second should complete
        assert len(results) == 1
        assert results[0].entry_a.id == 2
        assert results[0].entry_b.id == 3


# ---------------------------------------------------------------------------
# TestRunRoundPlyCeiling
# ---------------------------------------------------------------------------


class TestRunRoundPlyCeiling:
    def test_run_round_ply_ceiling_warning(self) -> None:
        """Very low max_ply should trigger ply ceiling and yield partial results."""
        pool = _make_pool(parallel_matches=1, envs_per_match=2, total_envs=2, max_resident_models=2)
        # terminate_after=100 ensures games don't finish naturally
        vecenv = MockVecEnv(num_envs=2, terminate_after=100)
        entries = [_make_entry(i) for i in range(2)]
        pairings = [(entries[0], entries[1])]

        results = pool.run_round(
            vecenv,
            pairings,
            load_fn=lambda entry: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=1000,  # unreachable
            max_ply=1,  # very low ceiling
        )
        # Should get a result (partial) due to ply ceiling
        assert len(results) == 1
        r = results[0]
        total = r.a_wins + r.b_wins + r.draws
        assert total < 1000, "Should not have completed all games"


# ---------------------------------------------------------------------------
# TestPartitionRange
# ---------------------------------------------------------------------------


class TestPartitionRange:
    def test_partition_range_valid(self) -> None:
        """Index 0 returns (0, envs_per_match); last index returns correct end."""
        pool = _make_pool(parallel_matches=4, envs_per_match=3, total_envs=12)
        assert pool.partition_range(0) == (0, 3)
        assert pool.partition_range(1) == (3, 6)
        assert pool.partition_range(2) == (6, 9)
        assert pool.partition_range(3) == (9, 12)

    def test_partition_range_negative_raises(self) -> None:
        """index=-1 should raise ValueError."""
        pool = _make_pool(parallel_matches=2, envs_per_match=2, total_envs=4)
        with pytest.raises(ValueError, match="out of range"):
            pool.partition_range(-1)

    def test_partition_range_too_large_raises(self) -> None:
        """index >= parallel_matches should raise ValueError."""
        pool = _make_pool(parallel_matches=2, envs_per_match=2, total_envs=4)
        with pytest.raises(ValueError, match="out of range"):
            pool.partition_range(2)
        with pytest.raises(ValueError, match="out of range"):
            pool.partition_range(10)
