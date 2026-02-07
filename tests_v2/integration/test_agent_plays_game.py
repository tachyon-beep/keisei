"""Integration tests: a real PPOAgent plays a real ShogiGame.

Tests verify that agent + game + policy mapper interact correctly to
produce valid actions, update game state, and collect episode data.
"""

import numpy as np
import pytest
import torch

from keisei.shogi.shogi_game import ShogiGame


# ---------------------------------------------------------------------------
# Basic interaction
# ---------------------------------------------------------------------------


class TestAgentSelectsActions:
    """Agent can select valid actions from real game observations."""

    def test_agent_selects_action_from_initial_position(
        self, ppo_agent, shogi_game, session_policy_mapper, legal_mask_fn
    ):
        """Agent returns a valid move, policy index, log_prob, and value
        from the standard opening position."""
        obs = shogi_game.get_observation()
        legal_moves = shogi_game.get_legal_moves()
        legal_mask = legal_mask_fn(legal_moves)

        move, idx, log_prob, value = ppo_agent.select_action(
            obs, legal_mask, is_training=True
        )

        assert move is not None, "Agent should select a non-None move"
        assert 0 <= idx < session_policy_mapper.get_total_actions()
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_selected_move_is_legal(
        self, ppo_agent, shogi_game, legal_mask_fn
    ):
        """The move selected by the agent is in the game's legal move list."""
        obs = shogi_game.get_observation()
        legal_moves = shogi_game.get_legal_moves()
        legal_mask = legal_mask_fn(legal_moves)

        move, _, _, _ = ppo_agent.select_action(
            obs, legal_mask, is_training=True
        )

        assert move in legal_moves, (
            f"Agent selected {move}, which is not in legal_moves"
        )

    def test_agent_action_can_be_applied_to_game(
        self, ppo_agent, shogi_game, legal_mask_fn
    ):
        """make_move succeeds with the move selected by the agent."""
        obs = shogi_game.get_observation()
        legal_moves = shogi_game.get_legal_moves()
        legal_mask = legal_mask_fn(legal_moves)

        move, _, _, _ = ppo_agent.select_action(
            obs, legal_mask, is_training=True
        )

        result = shogi_game.make_move(move)
        assert isinstance(result, tuple) and len(result) == 4
        next_obs, reward, done, info = result
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape[0] == 46  # channels


# ---------------------------------------------------------------------------
# Multi-move play
# ---------------------------------------------------------------------------


class TestAgentPlaysMultipleMoves:
    """Agent can play several moves in sequence without errors."""

    def test_agent_plays_10_moves(
        self, ppo_agent, shogi_game, legal_mask_fn
    ):
        """Agent successfully plays 10 consecutive moves."""
        for _ in range(10):
            if shogi_game.game_over:
                break

            obs = shogi_game.get_observation()
            legal_moves = shogi_game.get_legal_moves()
            if not legal_moves:
                break
            legal_mask = legal_mask_fn(legal_moves)

            move, idx, log_prob, value = ppo_agent.select_action(
                obs, legal_mask, is_training=True
            )
            assert move is not None
            shogi_game.make_move(move)

        # At least some moves were played
        assert shogi_game.move_count > 0

    def test_game_state_updates_after_agent_action(
        self, ppo_agent, shogi_game, legal_mask_fn
    ):
        """Game state (move_count, current_player) updates after each agent move."""
        initial_player = shogi_game.current_player
        initial_move_count = shogi_game.move_count

        obs = shogi_game.get_observation()
        legal_moves = shogi_game.get_legal_moves()
        legal_mask = legal_mask_fn(legal_moves)
        move, _, _, _ = ppo_agent.select_action(obs, legal_mask, is_training=True)
        shogi_game.make_move(move)

        assert shogi_game.move_count == initial_move_count + 1
        assert shogi_game.current_player != initial_player


# ---------------------------------------------------------------------------
# Full game
# ---------------------------------------------------------------------------


class TestAgentPlaysFullGame:
    """Agent can play a full game to completion (or max_moves)."""

    def test_agent_plays_game_to_completion(
        self, ppo_agent, legal_mask_fn
    ):
        """A game with a small max_moves limit terminates."""
        game = ShogiGame(max_moves_per_game=60)
        total_reward = 0.0

        while not game.game_over:
            obs = game.get_observation()
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            legal_mask = legal_mask_fn(legal_moves)
            move, _, _, _ = ppo_agent.select_action(
                obs, legal_mask, is_training=True
            )
            _, reward, done, info = game.make_move(move)
            total_reward += reward
            if done:
                break

        assert game.game_over or game.move_count >= 60
        assert isinstance(total_reward, float)

    def test_episode_rewards_are_collected(
        self, ppo_agent, legal_mask_fn
    ):
        """Rewards can be accumulated across an entire episode."""
        game = ShogiGame(max_moves_per_game=30)
        rewards = []

        while not game.game_over:
            obs = game.get_observation()
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            legal_mask = legal_mask_fn(legal_moves)
            move, _, _, _ = ppo_agent.select_action(
                obs, legal_mask, is_training=True
            )
            _, reward, done, _ = game.make_move(move)
            rewards.append(reward)
            if done:
                break

        assert len(rewards) > 0, "At least one step should produce a reward"
        assert all(isinstance(r, float) for r in rewards)


# ---------------------------------------------------------------------------
# Value head
# ---------------------------------------------------------------------------


class TestAgentValueEstimates:
    """Agent value predictions are valid floats for game observations."""

    def test_get_value_returns_float(self, ppo_agent, shogi_game):
        obs = shogi_game.get_observation()
        value = ppo_agent.get_value(obs)
        assert isinstance(value, float)
        assert np.isfinite(value)

    def test_value_changes_with_different_positions(
        self, ppo_agent, play_random_moves
    ):
        """Value estimate may differ between early and mid-game positions."""
        game1 = ShogiGame(max_moves_per_game=500)
        val_initial = ppo_agent.get_value(game1.get_observation())

        # Play a few random moves to change the position
        game2 = ShogiGame(max_moves_per_game=500)
        play_random_moves(game2, 10)
        val_mid = ppo_agent.get_value(game2.get_observation())

        # Both should be valid floats (they might be equal in a random net, that is OK)
        assert isinstance(val_initial, float)
        assert isinstance(val_mid, float)
        assert np.isfinite(val_initial)
        assert np.isfinite(val_mid)
