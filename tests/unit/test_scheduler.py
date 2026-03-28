"""Unit tests for ContinuousMatchScheduler."""

import asyncio
import itertools
import json
import tempfile
from collections import Counter, deque
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest
import torch

from keisei.evaluation.scheduler import MatchOutcome, MatchResult

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
        scheduler._failed_checkpoints = set()

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


class TestGameExecution:
    """Scheduler runs games between two models."""

    @pytest.mark.asyncio
    async def test_run_game_loop_returns_result(self):
        """Game loop runs until done and returns winner info."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler
        from keisei.shogi.shogi_core_definitions import Color

        scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
        mock_config = MagicMock()
        mock_config.max_moves_per_game = 10
        mock_config.move_delay = 0
        mock_config.move_timeout = 30.0
        scheduler._config = mock_config
        scheduler._active_matches = {}
        scheduler._publish_state = AsyncMock()
        scheduler._policy_mapper = MagicMock()
        scheduler._policy_mapper.get_legal_mask.return_value = torch.zeros(13527)
        scheduler._inference_semaphore = asyncio.Semaphore(1)

        game = MagicMock()
        game.get_legal_moves.return_value = [(0, 0, 1, 1, False)]
        game.get_observation.return_value = np.zeros((46, 9, 9))
        game.to_sfen.return_value = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

        _cycle = itertools.cycle([Color.BLACK, Color.WHITE])
        type(game).current_player = property(lambda self: next(_cycle))

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
        assert result.winner == MatchOutcome.BLACK_WIN

    @pytest.mark.asyncio
    async def test_run_game_loop_draw_on_max_moves(self):
        """Game returns draw when max_moves reached without winner."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler
        from keisei.shogi.shogi_core_definitions import Color

        scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
        mock_config = MagicMock()
        mock_config.max_moves_per_game = 3
        mock_config.move_delay = 0
        mock_config.move_timeout = 30.0
        scheduler._config = mock_config
        scheduler._active_matches = {}
        scheduler._publish_state = AsyncMock()
        scheduler._policy_mapper = MagicMock()
        scheduler._policy_mapper.get_legal_mask.return_value = torch.zeros(13527)
        scheduler._inference_semaphore = asyncio.Semaphore(1)

        game = MagicMock()
        game.get_legal_moves.return_value = [(0, 0, 1, 1, False)]
        game.get_observation.return_value = np.zeros((46, 9, 9))
        game.to_sfen.return_value = "startpos"

        _cycle = itertools.cycle([Color.BLACK, Color.WHITE])
        type(game).current_player = property(lambda self: next(_cycle))

        obs = np.zeros((46, 9, 9))
        game.make_move.return_value = (obs, 0.0, False, {})

        agent_a = MagicMock()
        agent_a.select_action.return_value = ((0, 0, 1, 1, False), 0, 0.0, 0.0)
        agent_b = MagicMock()
        agent_b.select_action.return_value = ((0, 0, 1, 1, False), 0, 0.0, 0.0)

        result = await scheduler._run_game_loop(
            game, agent_a, agent_b, spectated=False, slot=0
        )
        assert result.winner == MatchOutcome.DRAW
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
                num_spectated=0,
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
                num_spectated=0,
                state_path=state_path,
            )
            scheduler = ContinuousMatchScheduler(config)

            # Seed initial ratings
            scheduler._elo_registry.get_rating("model_a.pth")  # 1500
            scheduler._elo_registry.get_rating("model_b.pth")  # 1500
            old_a = scheduler._elo_registry.get_rating("model_a.pth")
            old_b = scheduler._elo_registry.get_rating("model_b.pth")

            # Mock _run_game_loop to return a win for model_a (Black)
            async def fake_game_loop(game, agent_a, agent_b, spectated, slot):
                return MatchResult(winner=MatchOutcome.BLACK_WIN, move_count=10, reason="game_over")

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
        from keisei.evaluation.scheduler import ActiveMatchState, ContinuousMatchScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
            scheduler._state_path = Path(tmpdir) / "state.json"
            scheduler._active_matches = {
                0: ActiveMatchState(
                    match_id="test", status="in_progress", spectated=True,
                    sfen="lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                    model_a={}, model_b={}, move_count=0, move_log=[],
                )
            }
            scheduler._recent_results = [{"winner": "model_a"}]
            scheduler._elo_registry = MagicMock()
            scheduler._elo_registry.get_all_ratings.return_value = {"model_a": 1520.0, "model_b": 1480.0}
            scheduler._games_played = Counter({"model_a": 10, "model_b": 5})
            scheduler._wins = Counter({"model_a": 7, "model_b": 2})

            # Test the sync components directly (async _publish_state
            # delegates to these).
            state = scheduler._build_state_snapshot()
            scheduler._write_state_sync(state)

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
            scheduler._elo_registry.get_all_ratings.return_value = {
                "weak": 1400.0, "strong": 1600.0, "mid": 1500.0
            }
            scheduler._games_played = Counter({"weak": 5, "strong": 5, "mid": 5})
            scheduler._wins = Counter({"weak": 1, "strong": 4, "mid": 2})

            state = scheduler._build_state_snapshot()
            scheduler._write_state_sync(state)

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


class TestBlacklist:
    """Blacklisting excludes failed checkpoints from matchup selection."""

    def _make_scheduler(self, checkpoints, ratings=None):
        from keisei.evaluation.scheduler import ContinuousMatchScheduler

        scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
        scheduler._pool_paths = [Path(c) for c in checkpoints]
        scheduler._games_played = Counter()
        scheduler._elo_registry = None
        scheduler._failed_checkpoints = set()
        scheduler._get_rating = lambda name: (ratings or {}).get(name, 1500.0)
        return scheduler

    def test_pick_excludes_blacklisted_checkpoints(self):
        """Blacklisted checkpoints should never appear in matchup results."""
        scheduler = self._make_scheduler(["a.pth", "b.pth", "c.pth"])
        scheduler._failed_checkpoints.add(Path("c.pth"))

        for _ in range(100):
            a, b = scheduler._pick_matchup()
            assert a != Path("c.pth"), "Blacklisted checkpoint appeared as model_a"
            assert b != Path("c.pth"), "Blacklisted checkpoint appeared as model_b"

    def test_pick_raises_when_all_but_one_blacklisted(self):
        """Should raise ValueError if blacklisting leaves fewer than 2 models."""
        scheduler = self._make_scheduler(["a.pth", "b.pth", "c.pth"])
        scheduler._failed_checkpoints.add(Path("b.pth"))
        scheduler._failed_checkpoints.add(Path("c.pth"))

        with pytest.raises(ValueError, match="at least 2"):
            scheduler._pick_matchup()


class TestCircuitBreakerAndBlacklist:
    """_run_match error paths properly update circuit breaker and blacklist."""

    @pytest.mark.asyncio
    async def test_missing_checkpoint_blacklists_only_missing_file(self):
        """Only the actually-missing checkpoint should be blacklisted."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler, SchedulerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            existing = Path(tmpdir) / "exists.pth"
            existing.write_bytes(b"fake")
            missing = Path(tmpdir) / "missing.pth"

            config = SchedulerConfig(
                checkpoint_dir=Path(tmpdir),
                elo_registry_path=Path(tmpdir) / "elo.json",
                device="cpu",
                num_concurrent=1,
                num_spectated=0,
                state_path=Path(tmpdir) / "state.json",
            )
            scheduler = ContinuousMatchScheduler(config)

            # Load missing first so FileNotFoundError is raised for it
            await scheduler._run_match(0, missing, existing)

            assert missing in scheduler._failed_checkpoints
            assert existing not in scheduler._failed_checkpoints
            assert scheduler._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_consecutive_failures_reset_on_success(self):
        """A successful match should reset _consecutive_failures to 0."""
        from keisei.evaluation.scheduler import (
            ContinuousMatchScheduler,
            MatchResult,
            SchedulerConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SchedulerConfig(
                checkpoint_dir=Path(tmpdir),
                elo_registry_path=Path(tmpdir) / "elo.json",
                device="cpu",
                num_concurrent=1,
                num_spectated=0,
                state_path=Path(tmpdir) / "state.json",
            )
            scheduler = ContinuousMatchScheduler(config)
            scheduler._consecutive_failures = 3  # pre-set some failures

            async def fake_game_loop(game, agent_a, agent_b, spectated, slot):
                return MatchResult(winner=MatchOutcome.BLACK_WIN, move_count=5, reason="game_over")

            scheduler._run_game_loop = fake_game_loop

            agent_mock = MagicMock()
            with patch(
                "keisei.evaluation.scheduler.load_evaluation_agent",
                return_value=agent_mock,
            ), patch("keisei.shogi.shogi_game.ShogiGame") as game_cls:
                game_cls.return_value.to_sfen.return_value = "startpos"
                scheduler._elo_registry.get_rating("a.pth")
                scheduler._elo_registry.get_rating("b.pth")

                await scheduler._run_match(0, Path(tmpdir) / "a.pth", Path(tmpdir) / "b.pth")

            assert scheduler._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_value_error_increments_circuit_breaker(self):
        """ValueError during match should increment _consecutive_failures."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler, SchedulerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SchedulerConfig(
                checkpoint_dir=Path(tmpdir),
                elo_registry_path=Path(tmpdir) / "elo.json",
                device="cpu",
                num_concurrent=1,
                num_spectated=0,
                state_path=Path(tmpdir) / "state.json",
            )
            scheduler = ContinuousMatchScheduler(config)

            with patch(
                "keisei.evaluation.scheduler.load_evaluation_agent",
                side_effect=ValueError("bad config"),
            ):
                await scheduler._run_match(0, Path("/a.pth"), Path("/b.pth"))

            assert scheduler._consecutive_failures == 1


class TestGameLoopEdgePaths:
    """Tests for _run_game_loop error/edge paths (I1)."""

    def _make_scheduler_for_game_loop(self):
        """Create a minimal scheduler for game loop tests."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler, ActiveMatchState

        scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
        mock_config = MagicMock()
        mock_config.max_moves_per_game = 100
        mock_config.move_delay = 0
        mock_config.move_timeout = 0.1  # Short timeout for tests
        scheduler._config = mock_config
        scheduler._active_matches = {}
        scheduler._publish_state = AsyncMock()
        scheduler._policy_mapper = MagicMock()
        scheduler._policy_mapper.get_legal_mask.return_value = torch.zeros(13527)
        scheduler._inference_semaphore = asyncio.Semaphore(1)
        return scheduler

    @pytest.mark.asyncio
    async def test_no_legal_moves_returns_opponent_win(self):
        """No legal moves (checkmate) awards win to opponent."""
        from keisei.shogi.shogi_core_definitions import Color

        scheduler = self._make_scheduler_for_game_loop()

        game = MagicMock()
        game.get_legal_moves.return_value = []  # No legal moves
        game.to_sfen.return_value = "checkmate_pos"
        game.current_player = Color.BLACK

        agent_a = MagicMock()
        agent_b = MagicMock()

        result = await scheduler._run_game_loop(
            game, agent_a, agent_b, spectated=False, slot=0
        )
        assert result.winner == MatchOutcome.WHITE_WIN
        assert result.reason == "no_legal_moves"
        assert result.move_count == 0

    @pytest.mark.asyncio
    async def test_no_legal_moves_white_turn_returns_black_win(self):
        """No legal moves on White's turn awards win to Black."""
        from keisei.shogi.shogi_core_definitions import Color

        scheduler = self._make_scheduler_for_game_loop()

        game = MagicMock()
        game.get_legal_moves.return_value = []
        game.to_sfen.return_value = "checkmate_pos"
        game.current_player = Color.WHITE

        result = await scheduler._run_game_loop(
            game, MagicMock(), MagicMock(), spectated=False, slot=0
        )
        assert result.winner == MatchOutcome.BLACK_WIN

    @pytest.mark.asyncio
    async def test_inference_timeout_returns_opponent_win(self):
        """Agent inference timeout awards win to opponent."""
        from keisei.shogi.shogi_core_definitions import Color

        scheduler = self._make_scheduler_for_game_loop()

        game = MagicMock()
        game.get_legal_moves.return_value = [(0, 0, 1, 1, False)]
        game.get_observation.return_value = np.zeros((46, 9, 9))
        game.current_player = Color.BLACK

        agent_a = MagicMock()
        # select_action blocks forever — will trigger timeout
        import time
        agent_a.select_action.side_effect = lambda *a, **kw: time.sleep(10)
        agent_b = MagicMock()

        result = await scheduler._run_game_loop(
            game, agent_a, agent_b, spectated=False, slot=0
        )
        assert result.winner == MatchOutcome.WHITE_WIN
        assert result.reason == "inference_timeout"

    @pytest.mark.asyncio
    async def test_none_action_returns_opponent_win(self):
        """Agent returning None action awards win to opponent."""
        from keisei.shogi.shogi_core_definitions import Color

        scheduler = self._make_scheduler_for_game_loop()

        game = MagicMock()
        game.get_legal_moves.return_value = [(0, 0, 1, 1, False)]
        game.get_observation.return_value = np.zeros((46, 9, 9))
        game.current_player = Color.BLACK

        agent_a = MagicMock()
        agent_a.select_action.return_value = (None, 0, 0.0, 0.0)
        agent_b = MagicMock()

        result = await scheduler._run_game_loop(
            game, agent_a, agent_b, spectated=False, slot=0
        )
        assert result.winner == MatchOutcome.WHITE_WIN
        assert result.reason == "no_action_selected"


class TestSpectatedGameLoop:
    """Tests for spectated code path in _run_game_loop (I2)."""

    @pytest.mark.asyncio
    async def test_spectated_publishes_state_each_move(self):
        """Spectated games publish state after every move."""
        from keisei.evaluation.scheduler import (
            ContinuousMatchScheduler, ActiveMatchState,
        )
        from keisei.shogi.shogi_core_definitions import Color

        scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
        mock_config = MagicMock()
        mock_config.max_moves_per_game = 5
        mock_config.move_delay = 0
        mock_config.move_timeout = 30.0
        scheduler._config = mock_config
        scheduler._publish_state = AsyncMock()
        scheduler._policy_mapper = MagicMock()
        scheduler._policy_mapper.get_legal_mask.return_value = torch.zeros(13527)
        scheduler._inference_semaphore = asyncio.Semaphore(1)

        # Pre-populate active match state for slot 0
        scheduler._active_matches = {
            0: ActiveMatchState(
                match_id="test", status="waiting", spectated=True,
                sfen="startpos", model_a={}, model_b={},
                move_count=0, move_log=[],
            )
        }

        game = MagicMock()
        game.get_legal_moves.return_value = [(0, 0, 1, 1, False)]
        game.get_observation.return_value = np.zeros((46, 9, 9))
        game.to_sfen.return_value = "updated_sfen"

        _cycle = itertools.cycle([Color.BLACK, Color.WHITE])
        type(game).current_player = property(lambda self: next(_cycle))

        obs = np.zeros((46, 9, 9))
        game.make_move.side_effect = [
            (obs, 0.0, False, {}),
            (obs, 0.0, True, {"winner": "black"}),
        ]

        agent_a = MagicMock()
        agent_a.select_action.return_value = ((0, 0, 1, 1, False), 0, 0.0, 0.0)
        agent_b = MagicMock()
        agent_b.select_action.return_value = ((0, 0, 1, 1, False), 0, 0.0, 0.0)

        result = await scheduler._run_game_loop(
            game, agent_a, agent_b, spectated=True, slot=0
        )

        assert result.winner == MatchOutcome.BLACK_WIN
        # _publish_state called once per move (2 moves)
        assert scheduler._publish_state.call_count == 2
        # Match state was updated
        match_state = scheduler._active_matches[0]
        assert match_state.sfen == "updated_sfen"
        assert match_state.move_count == 2

    @pytest.mark.asyncio
    async def test_spectated_publish_failure_does_not_abort_game(self):
        """A failed _publish_state in the game loop should not kill the game."""
        from keisei.evaluation.scheduler import (
            ContinuousMatchScheduler, ActiveMatchState,
        )
        from keisei.shogi.shogi_core_definitions import Color

        scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
        mock_config = MagicMock()
        mock_config.max_moves_per_game = 5
        mock_config.move_delay = 0
        mock_config.move_timeout = 30.0
        scheduler._config = mock_config
        # _publish_state raises on every call
        scheduler._publish_state = AsyncMock(side_effect=OSError("disk full"))
        scheduler._policy_mapper = MagicMock()
        scheduler._policy_mapper.get_legal_mask.return_value = torch.zeros(13527)
        scheduler._inference_semaphore = asyncio.Semaphore(1)

        scheduler._active_matches = {
            0: ActiveMatchState(
                match_id="test", status="waiting", spectated=True,
                sfen="startpos", model_a={}, model_b={},
                move_count=0, move_log=[],
            )
        }

        game = MagicMock()
        game.get_legal_moves.return_value = [(0, 0, 1, 1, False)]
        game.get_observation.return_value = np.zeros((46, 9, 9))
        game.to_sfen.return_value = "sfen"

        _cycle = itertools.cycle([Color.BLACK, Color.WHITE])
        type(game).current_player = property(lambda self: next(_cycle))

        obs = np.zeros((46, 9, 9))
        game.make_move.side_effect = [
            (obs, 0.0, True, {"winner": "white"}),
        ]

        agent_a = MagicMock()
        agent_a.select_action.return_value = ((0, 0, 1, 1, False), 0, 0.0, 0.0)
        agent_b = MagicMock()

        # Should NOT raise despite publish failures
        result = await scheduler._run_game_loop(
            game, agent_a, agent_b, spectated=True, slot=0
        )
        assert result.winner == MatchOutcome.WHITE_WIN


class TestSchedulerConfigValidation:
    """Tests for SchedulerConfig Pydantic validators (I3)."""

    def test_rejects_invalid_device_string(self):
        """Invalid device strings should be rejected."""
        from keisei.evaluation.scheduler import SchedulerConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Invalid device"):
            SchedulerConfig(
                checkpoint_dir=Path("/tmp"),
                elo_registry_path=Path("/tmp/elo.json"),
                device="mps",
            )

    def test_rejects_spectated_exceeding_concurrent(self):
        """num_spectated > num_concurrent should be rejected."""
        from keisei.evaluation.scheduler import SchedulerConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="num_spectated"):
            SchedulerConfig(
                checkpoint_dir=Path("/tmp"),
                elo_registry_path=Path("/tmp/elo.json"),
                device="cpu",
                num_concurrent=2,
                num_spectated=5,
            )

    def test_accepts_valid_cuda_device(self):
        """Valid CUDA device strings should be accepted."""
        from keisei.evaluation.scheduler import SchedulerConfig

        config = SchedulerConfig(
            checkpoint_dir=Path("/tmp"),
            elo_registry_path=Path("/tmp/elo.json"),
            device="cuda:0",
        )
        assert config.device == "cuda:0"

    def test_accepts_cpu_device(self):
        """'cpu' should be accepted."""
        from keisei.evaluation.scheduler import SchedulerConfig

        config = SchedulerConfig(
            checkpoint_dir=Path("/tmp"),
            elo_registry_path=Path("/tmp/elo.json"),
            device="cpu",
        )
        assert config.device == "cpu"

    def test_spectated_equals_concurrent_is_valid(self):
        """num_spectated == num_concurrent is the boundary case and should pass."""
        from keisei.evaluation.scheduler import SchedulerConfig

        config = SchedulerConfig(
            checkpoint_dir=Path("/tmp"),
            elo_registry_path=Path("/tmp/elo.json"),
            device="cpu",
            num_concurrent=3,
            num_spectated=3,
        )
        assert config.num_spectated == 3


class TestRuntimeErrorBlacklisting:
    """RuntimeError in _run_match should blacklist both checkpoints (C2)."""

    @pytest.mark.asyncio
    async def test_runtime_error_blacklists_both_models(self):
        """Architecture mismatch (RuntimeError) blacklists both checkpoints."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler, SchedulerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SchedulerConfig(
                checkpoint_dir=Path(tmpdir),
                elo_registry_path=Path(tmpdir) / "elo.json",
                device="cpu",
                num_concurrent=1,
                num_spectated=0,
                state_path=Path(tmpdir) / "state.json",
            )
            scheduler = ContinuousMatchScheduler(config)

            model_a = Path(tmpdir) / "a.pth"
            model_b = Path(tmpdir) / "b.pth"

            with patch(
                "keisei.evaluation.scheduler.load_evaluation_agent",
                side_effect=RuntimeError("size mismatch for layer.weight"),
            ):
                await scheduler._run_match(0, model_a, model_b)

            assert model_a in scheduler._failed_checkpoints
            assert model_b in scheduler._failed_checkpoints
            assert scheduler._consecutive_failures == 1


class TestWinRateInLeaderboard:
    """Leaderboard entries include win_rate computed from wins/games_played."""

    def test_win_rate_computed_correctly(self):
        from keisei.evaluation.scheduler import ContinuousMatchScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
            scheduler._state_path = Path(tmpdir) / "state.json"
            scheduler._active_matches = {}
            scheduler._recent_results = []
            scheduler._elo_registry = MagicMock()
            scheduler._elo_registry.get_all_ratings.return_value = {
                "model_a": 1600.0, "model_b": 1400.0
            }
            scheduler._games_played = Counter({"model_a": 10, "model_b": 10})
            scheduler._wins = Counter({"model_a": 7, "model_b": 3})

            state = scheduler._build_state_snapshot()
            lb = {e["name"]: e for e in state["leaderboard"]}
            assert lb["model_a"]["win_rate"] == 0.7
            assert lb["model_b"]["win_rate"] == 0.3

    def test_win_rate_zero_when_no_games(self):
        from keisei.evaluation.scheduler import ContinuousMatchScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
            scheduler._state_path = Path(tmpdir) / "state.json"
            scheduler._active_matches = {}
            scheduler._recent_results = []
            scheduler._elo_registry = MagicMock()
            scheduler._elo_registry.get_all_ratings.return_value = {"new_model": 1500.0}
            scheduler._games_played = Counter()
            scheduler._wins = Counter()

            state = scheduler._build_state_snapshot()
            assert state["leaderboard"][0]["win_rate"] == 0.0


class TestFastTrack:
    """New checkpoints get fast-tracked for initial rating games."""

    def _make_scheduler(self, checkpoints, fast_track_games=8):
        from keisei.evaluation.scheduler import ContinuousMatchScheduler

        scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
        scheduler._pool_paths = [Path(c) for c in checkpoints]
        scheduler._games_played = Counter()
        scheduler._wins = Counter()
        scheduler._elo_registry = None
        scheduler._failed_checkpoints = set()
        scheduler._fast_track_queue = deque()
        scheduler._config = MagicMock()
        scheduler._config.fast_track_games = fast_track_games
        scheduler._get_rating = lambda name: 1500.0
        return scheduler

    def test_pick_fast_track_returns_pair(self):
        """Fast-track queue returns a valid pair with the new model."""
        scheduler = self._make_scheduler(["a.pth", "b.pth", "c.pth"])
        scheduler._fast_track_queue.append((Path("c.pth"), 3))

        pair = scheduler._pick_fast_track_matchup()
        assert pair is not None
        a, b = pair
        # c.pth must be one of the pair
        assert Path("c.pth") in (a, b)

    def test_fast_track_decrements_remaining(self):
        """Each call decrements the remaining games counter."""
        scheduler = self._make_scheduler(["a.pth", "b.pth"])
        scheduler._fast_track_queue.append((Path("b.pth"), 3))

        scheduler._pick_fast_track_matchup()
        assert len(scheduler._fast_track_queue) == 1
        _, remaining = scheduler._fast_track_queue[0]
        assert remaining == 2

    def test_fast_track_removes_when_exhausted(self):
        """Model is removed from queue when remaining hits 0."""
        scheduler = self._make_scheduler(["a.pth", "b.pth"])
        scheduler._fast_track_queue.append((Path("b.pth"), 1))

        scheduler._pick_fast_track_matchup()
        assert len(scheduler._fast_track_queue) == 0

    def test_fast_track_returns_none_when_empty(self):
        """Returns None when the fast-track queue is empty."""
        scheduler = self._make_scheduler(["a.pth", "b.pth"])
        assert scheduler._pick_fast_track_matchup() is None

    def test_fast_track_skips_blacklisted(self):
        """Blacklisted models are skipped in the fast-track queue."""
        scheduler = self._make_scheduler(["a.pth", "b.pth", "c.pth"])
        scheduler._fast_track_queue.append((Path("b.pth"), 5))
        scheduler._failed_checkpoints.add(Path("b.pth"))

        pair = scheduler._pick_fast_track_matchup()
        assert pair is None
        assert len(scheduler._fast_track_queue) == 0

    def test_refresh_pool_enqueues_new_models(self):
        """_refresh_pool adds new checkpoints to the fast-track queue."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler, SchedulerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            ckpt_dir.mkdir()
            (ckpt_dir / "checkpoint_ts1000.pth").write_bytes(b"fake")

            config = SchedulerConfig(
                checkpoint_dir=ckpt_dir,
                elo_registry_path=Path(tmpdir) / "elo.json",
                device="cpu",
                num_concurrent=1,
                num_spectated=0,
                state_path=Path(tmpdir) / "state.json",
                fast_track_games=5,
            )
            scheduler = ContinuousMatchScheduler(config)
            # Seed the pool with the existing checkpoint
            scheduler._refresh_pool()
            # First refresh enqueues the initial checkpoint
            initial_count = len(scheduler._fast_track_queue)
            assert initial_count == 1  # checkpoint_ts1000.pth

            # Drain the initial fast-track entry
            scheduler._fast_track_queue.clear()

            # Add a second checkpoint and refresh
            (ckpt_dir / "checkpoint_ts2000.pth").write_bytes(b"fake2")
            scheduler._refresh_pool()

            assert len(scheduler._fast_track_queue) == 1
            path, remaining = scheduler._fast_track_queue[0]
            assert path.name == "checkpoint_ts2000.pth"
            assert remaining == 5


class TestSchedulerExport:
    """Scheduler is importable from the evaluation package."""

    def test_importable_from_package(self):
        from keisei.evaluation import ContinuousMatchScheduler
        assert ContinuousMatchScheduler is not None
