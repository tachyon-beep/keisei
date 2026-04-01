"""keisei-evaluate: head-to-head evaluation between two checkpoints."""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from keisei.training.demonstrator import _get_policy_flat
from keisei.training.model_registry import build_model

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of a head-to-head evaluation."""

    wins: int
    losses: int
    draws: int

    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total_games

    def elo_delta(self) -> float:
        """Estimated Elo difference: positive means A is stronger."""
        wr = self.win_rate
        if wr <= 0.0 or wr >= 1.0:
            return float("inf") if wr >= 1.0 else float("-inf")
        return -400.0 * math.log10(1.0 / wr - 1.0)

    def win_rate_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        """Wilson score confidence interval for win rate."""
        n = self.total_games
        if n == 0:
            return (0.0, 1.0)
        p = self.win_rate
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
        return (max(0.0, centre - margin), min(1.0, centre + margin))


def run_evaluation(
    checkpoint_a: str,
    arch_a: str,
    checkpoint_b: str,
    arch_b: str,
    games: int = 400,
    max_ply: int = 500,
    params_a: dict | None = None,
    params_b: dict | None = None,
) -> EvalResult:
    """Run head-to-head evaluation between two checkpoints."""
    return _play_evaluation_games(
        checkpoint_a, arch_a, checkpoint_b, arch_b,
        games, max_ply, params_a or {}, params_b or {},
    )


def _play_evaluation_games(
    checkpoint_a: str, arch_a: str,
    checkpoint_b: str, arch_b: str,
    games: int, max_ply: int,
    params_a: dict, params_b: dict,
) -> EvalResult:
    """Play the actual games. Separated for testability (can be mocked)."""
    model_a = build_model(arch_a, params_a)
    model_a.load_state_dict(torch.load(checkpoint_a, map_location="cpu", weights_only=True))
    model_a.eval()

    model_b = build_model(arch_b, params_b)
    model_b.load_state_dict(torch.load(checkpoint_b, map_location="cpu", weights_only=True))
    model_b.eval()

    from shogi_gym import VecEnv

    wins, losses, draws = 0, 0, 0
    env = VecEnv(num_envs=1, max_ply=max_ply,
                 observation_mode="katago", action_mode="spatial")

    for game_i in range(games):
        a_is_black = game_i % 2 == 0
        models = [model_a, model_b] if a_is_black else [model_b, model_a]

        reset_result = env.reset()
        obs = torch.from_numpy(np.asarray(reset_result.observations))
        legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks))
        current_player = 0
        done = False

        while not done:
            model = models[current_player]
            with torch.no_grad():
                output = model(obs)
                flat = _get_policy_flat(output, obs.shape[0])
                masked = flat.masked_fill(~legal_masks, float("-inf"))
                probs = F.softmax(masked, dim=-1)
                action = torch.distributions.Categorical(probs).sample()

            step_result = env.step(action.tolist())
            done = bool(step_result.terminated[0] or step_result.truncated[0])
            obs = torch.from_numpy(np.asarray(step_result.observations))
            legal_masks = torch.from_numpy(np.asarray(step_result.legal_masks))
            current_player = int(step_result.current_players[0])

        reward = float(step_result.rewards[0])
        a_reward = reward if a_is_black else -reward
        if a_reward > 0:
            wins += 1
        elif a_reward < 0:
            losses += 1
        else:
            draws += 1

        if (game_i + 1) % 50 == 0:
            logger.info("Evaluation: %d/%d games (W=%d L=%d D=%d)",
                        game_i + 1, games, wins, losses, draws)

    return EvalResult(wins=wins, losses=losses, draws=draws)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Head-to-head evaluation of two checkpoints")
    parser.add_argument("--checkpoint-a", required=True, help="Path to checkpoint A")
    parser.add_argument("--arch-a", required=True, help="Architecture name for A")
    parser.add_argument("--checkpoint-b", required=True, help="Path to checkpoint B")
    parser.add_argument("--arch-b", required=True, help="Architecture name for B")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--max-ply", type=int, default=500)
    args = parser.parse_args()

    if args.games < 200:
        logger.warning("--games=%d is below 200: Elo estimates will have wide CIs", args.games)

    result = run_evaluation(
        checkpoint_a=args.checkpoint_a, arch_a=args.arch_a,
        checkpoint_b=args.checkpoint_b, arch_b=args.arch_b,
        games=args.games, max_ply=args.max_ply,
    )

    low, high = result.win_rate_ci()
    print(f"\nResults ({result.total_games} games):")
    print(f"  A wins: {result.wins}  losses: {result.losses}  draws: {result.draws}")
    print(f"  Win rate: {result.win_rate:.1%} (95% CI: [{low:.1%}, {high:.1%}])")
    print(f"  Elo delta: {result.elo_delta():+.0f}")


if __name__ == "__main__":
    main()
