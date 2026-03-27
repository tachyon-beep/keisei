"""Unit tests for ContinuousMatchScheduler."""

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

    def test_new_models_get_weight_boost(self):
        """Models with <5 games get 3x weight boost."""
        ratings = {"a.pth": 1500.0, "b.pth": 1500.0, "c.pth": 1500.0}
        scheduler = self._make_scheduler(["a.pth", "b.pth", "c.pth"], ratings)
        # a and b have many games, c is new
        scheduler._games_played["a.pth"] = 20
        scheduler._games_played["b.pth"] = 20
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
