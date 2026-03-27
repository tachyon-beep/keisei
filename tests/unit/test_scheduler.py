"""Unit tests for ContinuousMatchScheduler."""

import json
import tempfile

import pytest
from pathlib import Path
from collections import Counter

pytestmark = pytest.mark.unit


class TestMatchSelection:
    """Weighted random matchup selection by Elo proximity."""

    def _make_scheduler(self, checkpoints, ratings=None):
        """Create a scheduler with mock pool and ratings."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler

        scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
        scheduler._pool_paths = [Path(c) for c in checkpoints]
        scheduler._games_played = Counter()
        scheduler._elo_registry = None

        # Mock ratings
        scheduler._get_rating = lambda name: (ratings or {}).get(name, 1500.0)
        return scheduler

    def test_pick_returns_two_distinct_paths(self):
        scheduler = self._make_scheduler(["a.pth", "b.pth", "c.pth"])
        a, b = scheduler._pick_matchup()
        assert a != b
        assert a in scheduler._pool_paths
        assert b in scheduler._pool_paths

    def test_pick_raises_with_fewer_than_two_models(self):
        scheduler = self._make_scheduler(["a.pth"])
        with pytest.raises(ValueError, match="at least 2"):
            scheduler._pick_matchup()

    def test_pick_favors_close_ratings(self):
        """Models with close Elo should be matched more often."""
        ratings = {"a.pth": 1500.0, "b.pth": 1510.0, "c.pth": 1900.0}
        scheduler = self._make_scheduler(["a.pth", "b.pth", "c.pth"], ratings)

        pair_counts = Counter()
        for _ in range(1000):
            a, b = scheduler._pick_matchup()
            pair = tuple(sorted([a.name, b.name]))
            pair_counts[pair] += 1

        # a vs b (10 Elo apart) should be picked much more than a vs c (400 apart)
        ab_count = pair_counts.get(("a.pth", "b.pth"), 0)
        ac_count = pair_counts.get(("a.pth", "c.pth"), 0)
        assert ab_count > ac_count * 2, f"Close-rated pair {ab_count} should dominate distant {ac_count}"

    def test_new_models_get_uncertainty_boost(self):
        """Models with fewer games get higher uncertainty weight (1/sqrt(games))."""
        ratings = {"a.pth": 1500.0, "b.pth": 1500.0, "c.pth": 1500.0}
        scheduler = self._make_scheduler(["a.pth", "b.pth", "c.pth"], ratings)
        # a and b have many games, c is new — uncertainty weight favors c
        scheduler._games_played["a.pth"] = 100
        scheduler._games_played["b.pth"] = 100
        scheduler._games_played["c.pth"] = 1

        pair_counts = Counter()
        for _ in range(1000):
            a, b = scheduler._pick_matchup()
            pair = tuple(sorted([a.name, b.name]))
            pair_counts[pair] += 1

        # c should appear in more pairs than expected (boosted)
        c_appearances = sum(v for k, v in pair_counts.items() if "c.pth" in k)
        assert c_appearances > 400, f"New model should appear often, got {c_appearances}/1000"


from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import numpy as np
import torch


class TestGameExecution:
    """Scheduler runs games between two models."""

    @pytest.mark.asyncio
    async def test_run_game_loop_returns_result(self):
        """Game loop runs until done and returns winner info."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler
        from keisei.shogi.shogi_core_definitions import Color

        scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
        scheduler.max_moves_per_game = 10
        scheduler.move_delay = 0
        scheduler.num_spectated = 0
        scheduler._active_matches = {}
        scheduler._publish_state = MagicMock()
        scheduler._policy_mapper = MagicMock()
        scheduler._policy_mapper.get_legal_mask.return_value = torch.zeros(13527)

        game = MagicMock()
        game.get_legal_moves.return_value = [(0, 0, 1, 1, False)]
        game.get_observation.return_value = np.zeros((46, 9, 9))
        game.to_sfen.return_value = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

        player_cycle = [Color.BLACK, Color.WHITE] * 5
        type(game).current_player = property(
            lambda self, _cycle=iter(player_cycle): next(_cycle)
        )

        obs = np.zeros((46, 9, 9))
        game.make_move.side_effect = [
            (obs, 0.0, False, {}),
        ] * 9 + [(obs, 0.0, True, {"winner": "black"})]

        agent_a = MagicMock()
        agent_a.select_action.return_value = ((0, 0, 1, 1, False), 0, 0.0, 0.0)
        agent_b = MagicMock()
        agent_b.select_action.return_value = ((0, 0, 1, 1, False), 0, 0.0, 0.0)

        result = await scheduler._run_game_loop(
            game, agent_a, agent_b, spectated=False, slot=0
        )
        assert result.done is True
        assert result.winner == 0  # Black wins

    @pytest.mark.asyncio
    async def test_run_game_loop_draw_on_max_moves(self):
        """Game returns draw when max_moves reached without winner."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler, MatchResult
        from keisei.shogi.shogi_core_definitions import Color

        scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
        scheduler.max_moves_per_game = 3
        scheduler.move_delay = 0
        scheduler.num_spectated = 0
        scheduler._active_matches = {}
        scheduler._publish_state = MagicMock()
        scheduler._policy_mapper = MagicMock()
        scheduler._policy_mapper.get_legal_mask.return_value = torch.zeros(13527)

        game = MagicMock()
        game.get_legal_moves.return_value = [(0, 0, 1, 1, False)]
        game.get_observation.return_value = np.zeros((46, 9, 9))
        game.to_sfen.return_value = "startpos"

        player_cycle = [Color.BLACK, Color.WHITE] * 5
        type(game).current_player = property(
            lambda self, _cycle=iter(player_cycle): next(_cycle)
        )

        obs = np.zeros((46, 9, 9))
        game.make_move.return_value = (obs, 0.0, False, {})

        agent_a = MagicMock()
        agent_a.select_action.return_value = ((0, 0, 1, 1, False), 0, 0.0, 0.0)
        agent_b = MagicMock()
        agent_b.select_action.return_value = ((0, 0, 1, 1, False), 0, 0.0, 0.0)

        result = await scheduler._run_game_loop(
            game, agent_a, agent_b, spectated=False, slot=0
        )
        assert result.done is True
        assert result.winner is None
        assert result.move_count == 3
        assert result.reason == "max_moves"

    @pytest.mark.asyncio
    async def test_run_match_handles_missing_checkpoint(self):
        """_run_match logs error and doesn't crash on missing file."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler, SchedulerConfig
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SchedulerConfig(
                checkpoint_dir=Path(tmpdir),
                elo_registry_path=Path(tmpdir) / "elo.json",
                device="cpu",
                num_concurrent=1,
                state_path=Path(tmpdir) / "state.json",
            )
            scheduler = ContinuousMatchScheduler(config)

            await scheduler._run_match(
                0, Path("/nonexistent/a.pth"), Path("/nonexistent/b.pth")
            )
            # Should not raise — error is caught and logged


class TestEloUpdate:
    """Verify Elo ratings are updated correctly after a match."""

    @pytest.mark.asyncio
    async def test_elo_updated_after_win(self):
        """Winner gains Elo, loser loses Elo after a completed match."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler, SchedulerConfig
        from keisei.shogi.shogi_core_definitions import Color

        with tempfile.TemporaryDirectory() as tmpdir:
            elo_path = Path(tmpdir) / "elo.json"
            state_path = Path(tmpdir) / "state.json"

            config = SchedulerConfig(
                checkpoint_dir=Path(tmpdir),
                elo_registry_path=elo_path,
                device="cpu",
                num_concurrent=1,
                state_path=state_path,
            )
            scheduler = ContinuousMatchScheduler(config)

            # Seed initial ratings
            scheduler._elo_registry.get_rating("model_a.pth")  # 1500
            scheduler._elo_registry.get_rating("model_b.pth")  # 1500
            old_a = scheduler._elo_registry.get_rating("model_a.pth")
            old_b = scheduler._elo_registry.get_rating("model_b.pth")

            # Mock _run_game_loop to return a win for model_a (Black)
            from keisei.evaluation.scheduler import MatchResult

            async def fake_game_loop(game, agent_a, agent_b, spectated, slot):
                return MatchResult(done=True, winner=0, move_count=10, reason="game_over")

            scheduler._run_game_loop = fake_game_loop

            # Mock load_evaluation_agent to return dummy agents
            agent_mock = MagicMock()
            with patch(
                "keisei.evaluation.scheduler.load_evaluation_agent",
                return_value=agent_mock,
            ) as mock_load, patch(
                "keisei.shogi.shogi_game.ShogiGame"
            ) as game_cls:
                game_cls.return_value.to_sfen.return_value = "startpos"

                await scheduler._run_match(
                    0, Path(tmpdir) / "model_a.pth", Path(tmpdir) / "model_b.pth"
                )

            new_a = scheduler._elo_registry.get_rating("model_a.pth")
            new_b = scheduler._elo_registry.get_rating("model_b.pth")

            # Winner should gain, loser should lose (equal starting Elo)
            assert new_a > old_a, f"Winner Elo should increase: {old_a} -> {new_a}"
            assert new_b < old_b, f"Loser Elo should decrease: {old_b} -> {new_b}"
            # Games played should be tracked
            assert scheduler._games_played["model_a.pth"] == 1
            assert scheduler._games_played["model_b.pth"] == 1


class TestStatePublishing:
    """Scheduler writes atomic JSON state for dashboard."""

    def test_publish_state_creates_json(self):
        from keisei.evaluation.scheduler import ContinuousMatchScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
            scheduler._state_path = Path(tmpdir) / "state.json"
            scheduler._active_matches = {
                0: {"match_id": "test", "status": "in_progress", "spectated": True,
                    "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                    "model_a": {}, "model_b": {}, "move_count": 0}
            }
            scheduler._recent_results = [{"winner": "model_a"}]
            scheduler._elo_registry = MagicMock()
            scheduler._elo_registry.ratings = {"model_a": 1520.0, "model_b": 1480.0}
            scheduler._games_played = Counter({"model_a": 10, "model_b": 5})

            scheduler._publish_state()

            assert scheduler._state_path.exists()
            data = json.loads(scheduler._state_path.read_text())
            assert "matches" in data
            assert "leaderboard" in data
            assert "recent_results" in data
            assert data["schema_version"] == "ladder-v1"

    def test_leaderboard_sorted_by_elo(self):
        from keisei.evaluation.scheduler import ContinuousMatchScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
            scheduler._state_path = Path(tmpdir) / "state.json"
            scheduler._active_matches = {}
            scheduler._recent_results = []
            scheduler._elo_registry = MagicMock()
            scheduler._elo_registry.ratings = {
                "weak": 1400.0, "strong": 1600.0, "mid": 1500.0
            }
            scheduler._games_played = Counter({"weak": 5, "strong": 5, "mid": 5})

            scheduler._publish_state()

            data = json.loads(scheduler._state_path.read_text())
            elos = [entry["elo"] for entry in data["leaderboard"]]
            assert elos == sorted(elos, reverse=True)


class TestRunLoop:
    """Scheduler run loop manages concurrent game slots."""

    @pytest.mark.asyncio
    async def test_run_starts_and_can_be_cancelled(self):
        """Scheduler starts, runs briefly, and stops on cancellation."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            ckpt_dir.mkdir()
            (ckpt_dir / "checkpoint_ts1000.pth").write_bytes(b"fake")
            (ckpt_dir / "checkpoint_ts2000.pth").write_bytes(b"fake")

            elo_path = Path(tmpdir) / "elo.json"
            state_path = Path(tmpdir) / "state.json"

            from keisei.evaluation.scheduler import SchedulerConfig

            config = SchedulerConfig(
                checkpoint_dir=ckpt_dir,
                elo_registry_path=elo_path,
                device="cpu",
                num_concurrent=1,
                num_spectated=0,
                move_delay=0,
                poll_interval=1.0,
                max_moves_per_game=5,
                state_path=state_path,
            )
            scheduler = ContinuousMatchScheduler(config)

            match_started = asyncio.Event()
            match_count = 0

            async def fake_run_match(slot, a, b):
                nonlocal match_count
                match_count += 1
                match_started.set()
                await asyncio.sleep(0.05)

            scheduler._run_match = fake_run_match

            task = asyncio.create_task(scheduler.run())
            # Wait for at least one match to start (event-based, no timing dependency)
            await asyncio.wait_for(match_started.wait(), timeout=5.0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            assert match_count > 0, "Scheduler should have dispatched at least one match"


class TestSchedulerExport:
    """Scheduler is importable from the evaluation package."""

    def test_importable_from_package(self):
        from keisei.evaluation import ContinuousMatchScheduler
        assert ContinuousMatchScheduler is not None
