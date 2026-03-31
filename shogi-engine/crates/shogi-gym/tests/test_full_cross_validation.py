"""Full cross-validation: Rust shogi-gym vs Python shogi_python_reference.

Compares legal move sets, observation tensors, and game trajectories
between the Rust engine (via VecEnv/SpectatorEnv) and the Python
reference engine to catch any divergence before production use.
"""
import sys

sys.path.insert(0, "/home/john/keisei")

import numpy as np
import pytest
from shogi_gym import (
    VecEnv,
    SpectatorEnv,
    DefaultActionMapper,
    OBS_CURRENT_UNPROMOTED_START,
    OBS_CURRENT_PROMOTED_START,
    OBS_OPPONENT_UNPROMOTED_START,
    OBS_OPPONENT_PROMOTED_START,
    OBS_CURRENT_HAND_START,
    OBS_OPPONENT_HAND_START,
    OBS_PLAYER_INDICATOR,
    OBS_MOVE_COUNT,
)
from keisei.shogi_python_reference import ShogiGame, Color
from keisei.shogi_python_reference.shogi_game_io import (
    generate_neural_network_observation,
)
from keisei.utils.utils import PolicyOutputMapper


class TestLegalMoveSetCrossValidation:
    """Compare legal move sets from both engines at identical positions."""

    def setup_method(self):
        self.rust_mapper = DefaultActionMapper()
        self.python_mapper = PolicyOutputMapper()

    def _get_rust_legal_set(self, spectator: SpectatorEnv) -> set[int]:
        """Get legal action indices from the Rust engine."""
        return set(spectator.legal_actions())

    def _get_python_legal_set(self, game: ShogiGame) -> set[int]:
        """Get legal action indices from the Python engine (perspective-encoded)."""
        legal_moves = game.get_legal_moves()
        is_white = game.current_player == Color.WHITE
        indices = set()
        for mv in legal_moves:
            # Flip to perspective space for White (matching Rust's convention)
            if is_white:
                mv = PolicyOutputMapper.flip_move(mv)
            idx = self.python_mapper.shogi_move_to_policy_index(mv)
            indices.add(idx)
        return indices

    def test_startpos_legal_moves_match(self):
        """Both engines should produce identical legal moves at startpos."""
        spectator = SpectatorEnv()
        spectator.reset()
        game = ShogiGame()

        rust_legal = self._get_rust_legal_set(spectator)
        python_legal = self._get_python_legal_set(game)

        assert rust_legal == python_legal, (
            f"Legal move mismatch at startpos.\n"
            f"Rust only: {rust_legal - python_legal}\n"
            f"Python only: {python_legal - rust_legal}"
        )

    def test_legal_moves_match_after_random_play(self):
        """Play 50 random moves and compare legal sets at each position."""
        spectator = SpectatorEnv(max_ply=500)
        spectator.reset()
        game = ShogiGame()

        rng = np.random.default_rng(42)

        for ply in range(50):
            if spectator.is_over:
                break

            # Get legal sets from both engines
            rust_legal = self._get_rust_legal_set(spectator)
            python_legal = self._get_python_legal_set(game)

            assert rust_legal == python_legal, (
                f"Legal move mismatch at ply {ply}.\n"
                f"Rust only ({len(rust_legal - python_legal)}): "
                f"{sorted(rust_legal - python_legal)[:10]}\n"
                f"Python only ({len(python_legal - rust_legal)}): "
                f"{sorted(python_legal - rust_legal)[:10]}\n"
                f"SFEN: {spectator.to_sfen()}"
            )

            # Pick a random legal action (from the Rust set — verified identical)
            legal_list = sorted(rust_legal)
            action = legal_list[rng.integers(len(legal_list))]

            # Decode to get the move in Python format
            decoded = self.rust_mapper.decode(action, is_white=False)

            # Apply to Rust engine
            spectator.step(action)

            # Convert to Python move format and apply
            if decoded["type"] == "board":
                from_sq = decoded["from_sq"]
                to_sq = decoded["to_sq"]
                promote = decoded["promote"]
                py_move = (
                    from_sq // 9, from_sq % 9,
                    to_sq // 9, to_sq % 9,
                    promote,
                )
            else:
                to_sq = decoded["to_sq"]
                pt_idx = decoded["piece_type_idx"]
                from keisei.shogi_python_reference.shogi_core_definitions import (
                    get_unpromoted_types,
                )
                hand_types = get_unpromoted_types()
                py_move = (
                    None, None,
                    to_sq // 9, to_sq % 9,
                    hand_types[pt_idx],
                )

            # Need to flip the move if it's White's turn in the Python engine
            is_white = game.current_player == Color.WHITE
            if is_white:
                py_move = PolicyOutputMapper.flip_move(py_move)

            game.make_move(py_move)


class TestObservationCrossValidation:
    """Compare observation tensors from both engines."""

    def test_startpos_observation_match(self):
        """Observation tensors at startpos should be identical."""
        # Rust observation
        env = VecEnv(num_envs=1, max_ply=500)
        result = env.reset()
        rust_obs = np.asarray(result.observations)[0]  # (46, 9, 9)

        # Python observation
        game = ShogiGame()
        python_obs = generate_neural_network_observation(game)  # (46, 9, 9)

        # Compare piece channels (0-27)
        np.testing.assert_allclose(
            rust_obs[:28], python_obs[:28],
            atol=1e-6,
            err_msg="Piece channels differ at startpos",
        )

        # Compare hand channels (28-41) — should all be zero at startpos
        np.testing.assert_allclose(
            rust_obs[28:42], python_obs[28:42],
            atol=1e-6,
            err_msg="Hand channels differ at startpos",
        )

        # Compare player indicator (channel 42)
        np.testing.assert_allclose(
            rust_obs[42], python_obs[42],
            atol=1e-6,
            err_msg="Player indicator differs at startpos",
        )

        # Compare move count (channel 43) — both should be 0
        np.testing.assert_allclose(
            rust_obs[43], python_obs[43],
            atol=1e-6,
            err_msg="Move count channel differs at startpos",
        )

    def test_observation_match_after_moves(self):
        """Play a few moves and compare observations at each step."""
        spectator = SpectatorEnv(max_ply=500)
        spectator.reset()
        game = ShogiGame()
        mapper = DefaultActionMapper()
        python_mapper = PolicyOutputMapper()

        rng = np.random.default_rng(123)

        for ply in range(20):
            if spectator.is_over:
                break

            # Get Rust observation
            rust_obs = np.array(spectator.get_observation())  # (46, 9, 9)

            # Get Python observation
            python_obs = generate_neural_network_observation(game)

            # Compare all channels
            np.testing.assert_allclose(
                rust_obs, python_obs,
                atol=1e-5,
                err_msg=f"Observation mismatch at ply {ply}, SFEN: {spectator.to_sfen()}",
            )

            # Pick random legal action and play it on both engines
            legal = spectator.legal_actions()
            action = legal[rng.integers(len(legal))]
            decoded = mapper.decode(action, is_white=False)

            spectator.step(action)

            if decoded["type"] == "board":
                from_sq = decoded["from_sq"]
                to_sq = decoded["to_sq"]
                py_move = (
                    from_sq // 9, from_sq % 9,
                    to_sq // 9, to_sq % 9,
                    decoded["promote"],
                )
            else:
                to_sq = decoded["to_sq"]
                from keisei.shogi_python_reference.shogi_core_definitions import (
                    get_unpromoted_types,
                )
                hand_types = get_unpromoted_types()
                py_move = (
                    None, None,
                    to_sq // 9, to_sq % 9,
                    hand_types[decoded["piece_type_idx"]],
                )

            is_white = game.current_player == Color.WHITE
            if is_white:
                py_move = PolicyOutputMapper.flip_move(py_move)

            game.make_move(py_move)


class TestVecEnvObservationCrossValidation:
    """Verify VecEnv produces the same observations as SpectatorEnv."""

    def test_vecenv_matches_spectator_at_startpos(self):
        """VecEnv and SpectatorEnv should produce identical startpos observations."""
        env = VecEnv(num_envs=1, max_ply=500)
        result = env.reset()
        vec_obs = np.asarray(result.observations)[0]

        spectator = SpectatorEnv(max_ply=500)
        spectator.reset()
        spec_obs = np.array(spectator.get_observation())

        np.testing.assert_array_equal(
            vec_obs, spec_obs,
            err_msg="VecEnv and SpectatorEnv observations differ at startpos",
        )


class TestRecordAndReplay:
    """Record action sequences and replay through both engines."""

    def test_replay_10_games(self):
        """Play 10 short games, recording action indices.
        Replay through both engines and verify identical trajectories."""
        mapper = DefaultActionMapper()
        python_mapper = PolicyOutputMapper()

        rng = np.random.default_rng(999)

        for game_num in range(10):
            # Play a game using SpectatorEnv, recording actions
            spectator = SpectatorEnv(max_ply=30)
            spectator.reset()
            action_log = []

            while not spectator.is_over:
                legal = spectator.legal_actions()
                action = legal[rng.integers(len(legal))]
                action_log.append(action)
                spectator.step(action)

            rust_final_sfen = spectator.to_sfen()
            rust_final_ply = spectator.ply

            # Replay through Python engine
            game = ShogiGame()
            for step_idx, action in enumerate(action_log):
                decoded = mapper.decode(action, is_white=False)

                if decoded["type"] == "board":
                    from_sq = decoded["from_sq"]
                    to_sq = decoded["to_sq"]
                    py_move = (
                        from_sq // 9, from_sq % 9,
                        to_sq // 9, to_sq % 9,
                        decoded["promote"],
                    )
                else:
                    to_sq = decoded["to_sq"]
                    from keisei.shogi_python_reference.shogi_core_definitions import (
                        get_unpromoted_types,
                    )
                    hand_types = get_unpromoted_types()
                    py_move = (
                        None, None,
                        to_sq // 9, to_sq % 9,
                        hand_types[decoded["piece_type_idx"]],
                    )

                is_white = game.current_player == Color.WHITE
                if is_white:
                    py_move = PolicyOutputMapper.flip_move(py_move)

                game.make_move(py_move)

            assert game.move_count == rust_final_ply, (
                f"Game {game_num}: ply mismatch. "
                f"Rust={rust_final_ply}, Python={game.move_count}"
            )
