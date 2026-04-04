"""Tests for the keisei-evaluate head-to-head CLI."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from keisei.training.evaluate import EvalResult, _play_evaluation_games, run_evaluation


class TestEvalResult:
    def test_win_rate(self):
        result = EvalResult(wins=60, losses=30, draws=10)
        assert result.total_games == 100
        assert abs(result.win_rate - 0.65) < 1e-6  # (60 + 5) / 100

    def test_elo_delta_positive(self):
        result = EvalResult(wins=60, losses=30, draws=10)
        assert result.elo_delta() > 0

    def test_elo_delta_negative(self):
        result = EvalResult(wins=30, losses=60, draws=10)
        assert result.elo_delta() < 0

    def test_elo_delta_perfect(self):
        result = EvalResult(wins=100, losses=0, draws=0)
        assert result.elo_delta() == float("inf")

    def test_elo_delta_zero_wins(self):
        result = EvalResult(wins=0, losses=100, draws=0)
        assert result.elo_delta() == float("-inf")

    def test_confidence_interval(self):
        result = EvalResult(wins=200, losses=150, draws=50)
        low, high = result.win_rate_ci(confidence=0.95)
        assert low < result.win_rate
        assert high > result.win_rate
        assert high - low < 0.15  # 400 games -> CI < +/-7.5%

    def test_empty_result(self):
        result = EvalResult(wins=0, losses=0, draws=0)
        assert result.total_games == 0
        assert result.win_rate == 0.0
        low, high = result.win_rate_ci()
        assert low == 0.0
        assert high == 1.0


class TestRunEvaluation:
    def test_returns_eval_result(self):
        with patch("keisei.training.evaluate._play_evaluation_games") as mock_play:
            mock_play.return_value = EvalResult(wins=5, losses=3, draws=2)
            result = run_evaluation(
                checkpoint_a="/fake/a.pt", arch_a="resnet",
                checkpoint_b="/fake/b.pt", arch_b="se_resnet",
                games=10, max_ply=100,
            )
            assert result.total_games == 10
            assert result.wins == 5


# ---------------------------------------------------------------------------
# Helpers for mocking the VecEnv and game loop internals
# ---------------------------------------------------------------------------


@dataclass
class _FakeResetResult:
    observations: np.ndarray
    legal_masks: np.ndarray


@dataclass
class _FakeStepResult:
    observations: np.ndarray
    legal_masks: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    current_players: np.ndarray


def _make_mock_env(reward: float, steps_before_done: int = 1):
    """Return a mock VecEnv that ends after *steps_before_done* steps with *reward*.

    observations shape: (1, 46, 9, 9)  — katago observation
    legal_masks shape: (1, 11259)       — spatial action mask (9*9*139)

    current_players returns the *next* player (post-step convention):
    after step N, current_players = [N % 2].
    """
    obs = np.zeros((1, 46, 9, 9), dtype=np.float32)
    mask = np.ones((1, 11259), dtype=bool)

    env = MagicMock()
    call_count = {"n": 0}

    def _reset():
        call_count["n"] = 0
        return _FakeResetResult(observations=obs, legal_masks=mask)

    def _step(actions):
        call_count["n"] += 1
        done = call_count["n"] >= steps_before_done
        return _FakeStepResult(
            observations=obs,
            legal_masks=mask,
            rewards=np.array([reward if done else 0.0]),
            terminated=np.array([done]),
            truncated=np.array([False]),
            current_players=np.array([call_count["n"] % 2]),
        )

    env.reset.side_effect = _reset
    env.step.side_effect = _step
    return env


def _make_mock_model():
    """Return a mock model whose forward() returns a (policy, value) tuple."""
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.load_state_dict.return_value = None

    def _forward(obs):
        batch = obs.shape[0]
        policy = torch.zeros(batch, 11259)
        value = torch.zeros(batch, 1)
        return (policy, value)

    model.side_effect = _forward
    model.__call__ = _forward
    return model


# Shared patch targets
_PATCH_BUILD = "keisei.training.evaluate.build_model"
_PATCH_LOAD = "keisei.training.evaluate.torch.load"
_PATCH_VECENV = "keisei.training.evaluate.VecEnv"


def _run_games_with_mock(
    games: int,
    reward: float,
    steps: int = 1,
):
    """Helper that runs _play_evaluation_games with fully mocked deps."""
    import sys

    mock_model_a = _make_mock_model()
    mock_model_b = _make_mock_model()
    models = iter([mock_model_a, mock_model_b])

    mock_env = _make_mock_env(reward=reward, steps_before_done=steps)

    # shogi_gym may not be installed; inject a fake module so the
    # ``from shogi_gym import VecEnv`` inside _play_evaluation_games works.
    fake_shogi_gym = MagicMock()
    fake_shogi_gym.VecEnv.return_value = mock_env
    had_module = "shogi_gym" in sys.modules
    old_module = sys.modules.get("shogi_gym")
    sys.modules["shogi_gym"] = fake_shogi_gym

    try:
        with (
            patch(_PATCH_BUILD, side_effect=lambda *a, **kw: next(models)),
            patch(_PATCH_LOAD, return_value={"model_state_dict": {}}),
        ):
            result = _play_evaluation_games(
                checkpoint_a="/fake/a.pt", arch_a="resnet",
                checkpoint_b="/fake/b.pt", arch_b="resnet",
                games=games, max_ply=500,
                params_a={}, params_b={},
                device="cpu",
            )
    finally:
        if had_module:
            sys.modules["shogi_gym"] = old_module
        else:
            sys.modules.pop("shogi_gym", None)

    return result


class TestPlayEvaluationGames:
    """Tests for the _play_evaluation_games loop body."""

    # ---- Alternating colour assignment ----

    def test_even_game_a_is_black(self):
        """game_i=0 (even) -> a_is_black=True, so +1 reward is a win for A."""
        result = _run_games_with_mock(games=1, reward=1.0)
        assert result.wins == 1
        assert result.losses == 0
        assert result.draws == 0

    def test_odd_game_a_is_white(self):
        """game_i=1 (odd) -> a_is_black=False, so +1 reward becomes -1 for A (loss)."""
        # With 2 games: game 0 (even, a_is_black) reward +1 -> win
        #               game 1 (odd, a_is_white) reward +1 -> loss (sign flip)
        result = _run_games_with_mock(games=2, reward=1.0)
        assert result.wins == 1
        assert result.losses == 1
        assert result.draws == 0

    # ---- Reward sign convention ----

    def test_positive_reward_a_is_black_counts_as_win(self):
        """When A plays black and reward is +1, A wins."""
        result = _run_games_with_mock(games=1, reward=1.0)
        assert result.wins == 1

    def test_negative_reward_a_is_black_counts_as_loss(self):
        """When A plays black and reward is -1, A loses."""
        result = _run_games_with_mock(games=1, reward=-1.0)
        assert result.losses == 1

    def test_positive_reward_a_is_white_counts_as_loss(self):
        """When A plays white (odd game) and reward is +1, sign flips -> A loses."""
        # 2 games: game 0 gets +1 (win), game 1 gets +1 flipped to -1 (loss)
        result = _run_games_with_mock(games=2, reward=1.0)
        assert result.wins == 1  # game 0
        assert result.losses == 1  # game 1

    def test_negative_reward_a_is_white_counts_as_win(self):
        """When A plays white (odd game) and reward is -1, sign flips -> A wins."""
        # 2 games: game 0 gets -1 (loss), game 1 gets -1 flipped to +1 (win)
        result = _run_games_with_mock(games=2, reward=-1.0)
        assert result.losses == 1  # game 0
        assert result.wins == 1  # game 1

    # ---- Draw counting ----

    def test_zero_reward_counts_as_draw(self):
        """Reward of 0 should be tallied as a draw regardless of colour."""
        result = _run_games_with_mock(games=4, reward=0.0)
        assert result.wins == 0
        assert result.losses == 0
        assert result.draws == 4

    # ---- Aggregate tallying ----

    def test_total_games_correct(self):
        """All games are counted: wins + losses + draws == games."""
        result = _run_games_with_mock(games=6, reward=1.0)
        assert result.total_games == 6
        # 3 even (a_is_black, +1 -> win) + 3 odd (a_is_white, +1 -> loss)
        assert result.wins == 3
        assert result.losses == 3
        assert result.draws == 0


class TestRewardPerspectiveCorrection:
    """Regression: reward is from last-mover perspective, not Black's.

    When White delivers checkmate, reward=+1.0 means White won.
    The old code assumed reward=+1.0 always meant Black won, flipping
    win/loss attribution whenever the terminal move was by the "wrong" side.
    """

    def _make_env_white_terminal(self, reward: float, *, truncated: bool = False):
        """Mock env where White (player 1) makes the terminal move.

        2 steps per game: Black moves (non-terminal), White moves (terminal).
        Resets call_count on env.reset() so multi-game runs work.

        current_players returns the *next* player (post-step convention),
        not the mover who just acted. The production code captures the mover
        before calling env.step(), so this convention is correct.
        """
        obs = np.zeros((1, 46, 9, 9), dtype=np.float32)
        mask = np.ones((1, 11259), dtype=bool)

        env = MagicMock()
        call_count = {"n": 0}

        def _reset():
            call_count["n"] = 0
            return _FakeResetResult(observations=obs, legal_masks=mask)

        def _step(actions):
            call_count["n"] += 1
            done = call_count["n"] >= 2  # terminates on step 2 (White's move)
            return _FakeStepResult(
                observations=obs,
                legal_masks=mask,
                rewards=np.array([reward if done else 0.0]),
                terminated=np.array([done and not truncated]),
                truncated=np.array([done and truncated]),
                current_players=np.array([call_count["n"] % 2]),
            )

        env.reset.side_effect = _reset
        env.step.side_effect = _step
        return env

    def _run_with_env(self, mock_env, games: int):
        import sys

        mock_model_a = _make_mock_model()
        mock_model_b = _make_mock_model()
        models = iter([mock_model_a, mock_model_b])

        fake_shogi_gym = MagicMock()
        fake_shogi_gym.VecEnv.return_value = mock_env
        had_module = "shogi_gym" in sys.modules
        old_module = sys.modules.get("shogi_gym")
        sys.modules["shogi_gym"] = fake_shogi_gym

        try:
            with (
                patch(_PATCH_BUILD, side_effect=lambda *a, **kw: next(models)),
                patch(_PATCH_LOAD, return_value={"model_state_dict": {}}),
            ):
                return _play_evaluation_games(
                    checkpoint_a="/fake/a.pt", arch_a="resnet",
                    checkpoint_b="/fake/b.pt", arch_b="resnet",
                    games=games, max_ply=500,
                    params_a={}, params_b={},
                    device="cpu",
                )
        finally:
            if had_module:
                sys.modules["shogi_gym"] = old_module
            else:
                sys.modules.pop("shogi_gym", None)

    def test_white_terminal_move_a_is_black_reward_positive(self):
        """White wins (reward=+1 from White's POV). A is Black -> A loses."""
        env = self._make_env_white_terminal(reward=1.0)
        result = self._run_with_env(env, games=1)
        # A is Black, White won -> A lost
        assert result.losses == 1, f"Expected A loss, got W={result.wins} L={result.losses} D={result.draws}"
        assert result.wins == 0

    def test_white_terminal_move_a_is_black_reward_negative(self):
        """White loses (reward=-1 from White's POV). A is Black -> A wins."""
        env = self._make_env_white_terminal(reward=-1.0)
        result = self._run_with_env(env, games=1)
        # A is Black, White lost -> A won
        assert result.wins == 1, f"Expected A win, got W={result.wins} L={result.losses} D={result.draws}"
        assert result.losses == 0

    def test_white_terminal_move_a_is_white_reward_positive(self):
        """White wins (reward=+1). A is White (game_i=1) -> A wins."""
        env = self._make_env_white_terminal(reward=1.0)
        result = self._run_with_env(env, games=2)
        # game 0: A=Black, White won -> loss
        # game 1: A=White, White won -> win
        assert result.wins == 1
        assert result.losses == 1

    def test_truncated_game_perspective_correct(self):
        """Truncation (max ply) should use the same mover-based perspective."""
        env = self._make_env_white_terminal(reward=0.0, truncated=True)
        result = self._run_with_env(env, games=1)
        assert result.draws == 1


class TestGetPolicyFlat:
    """Tests for the _get_policy_flat dispatch function."""

    def test_tuple_output(self):
        """BaseModel returns (policy, value) tuple -- policy is returned as-is."""
        from keisei.training.demonstrator import _get_policy_flat

        policy = torch.randn(2, 11259)
        value = torch.randn(2, 1)
        result = _get_policy_flat((policy, value), batch_size=2)
        assert torch.equal(result, policy)

    def test_object_output(self):
        """KataGoBaseModel returns an object with .policy_logits shaped (B, 9, 9, 139)."""
        from keisei.training.demonstrator import _get_policy_flat

        output = MagicMock()
        output.policy_logits = torch.randn(2, 9, 9, 139)
        result = _get_policy_flat(output, batch_size=2)
        assert result.shape == (2, 9 * 9 * 139)  # flattened to (B, 11259)


class TestRawStateDictFallback:
    """H1: When checkpoint is a raw state_dict (not wrapped in a dict with
    'model_state_dict' key), _play_evaluation_games should use it directly."""

    def test_raw_state_dict_checkpoint(self):
        """Save a checkpoint as a raw state_dict and verify evaluation loads it."""
        import sys

        mock_model_a = _make_mock_model()
        mock_model_b = _make_mock_model()
        models = iter([mock_model_a, mock_model_b])

        mock_env = _make_mock_env(reward=1.0, steps_before_done=1)

        # The raw state_dict — not wrapped in {"model_state_dict": ...}
        raw_state_dict = {"layer.weight": torch.zeros(10)}

        fake_shogi_gym = MagicMock()
        fake_shogi_gym.VecEnv.return_value = mock_env
        had_module = "shogi_gym" in sys.modules
        old_module = sys.modules.get("shogi_gym")
        sys.modules["shogi_gym"] = fake_shogi_gym

        try:
            with (
                patch(_PATCH_BUILD, side_effect=lambda *a, **kw: next(models)),
                patch(_PATCH_LOAD, return_value=raw_state_dict),
            ):
                result = _play_evaluation_games(
                    checkpoint_a="/fake/a.pt", arch_a="resnet",
                    checkpoint_b="/fake/b.pt", arch_b="resnet",
                    games=1, max_ply=500,
                    params_a={}, params_b={},
                    device="cpu",
                )
        finally:
            if had_module:
                sys.modules["shogi_gym"] = old_module
            else:
                sys.modules.pop("shogi_gym", None)

        # Verify it loaded the raw dict as state_dict (fallback path)
        mock_model_a.load_state_dict.assert_called_once_with(raw_state_dict)
        mock_model_b.load_state_dict.assert_called_once_with(raw_state_dict)
        assert result.total_games == 1

    def test_non_dict_checkpoint_used_as_state_dict(self):
        """When torch.load returns a non-dict object, it should be passed
        directly to load_state_dict (the ``else ckpt`` branch)."""
        import sys

        mock_model_a = _make_mock_model()
        mock_model_b = _make_mock_model()
        models = iter([mock_model_a, mock_model_b])

        mock_env = _make_mock_env(reward=0.0, steps_before_done=1)

        # Simulate a checkpoint that is not a dict at all (e.g. OrderedDict
        # subclass that isinstance(..., dict) returns False for, or just
        # a raw tensor — contrived but exercises the branch).
        class NotADict:
            pass

        sentinel = NotADict()

        fake_shogi_gym = MagicMock()
        fake_shogi_gym.VecEnv.return_value = mock_env
        had_module = "shogi_gym" in sys.modules
        old_module = sys.modules.get("shogi_gym")
        sys.modules["shogi_gym"] = fake_shogi_gym

        try:
            with (
                patch(_PATCH_BUILD, side_effect=lambda *a, **kw: next(models)),
                patch(_PATCH_LOAD, return_value=sentinel),
            ):
                result = _play_evaluation_games(
                    checkpoint_a="/fake/a.pt", arch_a="resnet",
                    checkpoint_b="/fake/b.pt", arch_b="resnet",
                    games=1, max_ply=500,
                    params_a={}, params_b={},
                    device="cpu",
                )
        finally:
            if had_module:
                sys.modules["shogi_gym"] = old_module
            else:
                sys.modules.pop("shogi_gym", None)

        # The non-dict object should be passed directly to load_state_dict
        mock_model_a.load_state_dict.assert_called_once_with(sentinel)
        mock_model_b.load_state_dict.assert_called_once_with(sentinel)
        assert result.draws == 1
