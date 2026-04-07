"""GameFeatureTracker: inline per-game feature extraction during match play.

Accumulates behavioural features (captures, drops, promotions, rook/king
movement, opening sequences) from actions and StepMetadata during
play_batch() calls.  When a game ends, produces a feature row ready for
DB insertion.

All move classification is derived from the spatial action encoding
(9×9×139) without needing to replay games through SpectatorEnv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Spatial action encoding constants (from spatial_action_mapper.rs)
SPATIAL_MOVE_TYPES = 139
# Move types 64-131: sliding/knight moves WITH promotion
PROMOTION_MOVE_TYPE_MIN = 64
PROMOTION_MOVE_TYPE_MAX = 131
# Move types 132-138: drops (7 piece types)
DROP_MOVE_TYPE_MIN = 132
DROP_MOVE_TYPE_MAX = 138
# No-capture sentinel in StepMetadata.captured_piece
NO_CAPTURE = 255

# Rook starting squares (file-major: square = row * 9 + col).
# Black rook: rank 9 (row 8), file 2 (col 7) → 8*9+7 = 79
# White rook: rank 1 (row 0), file 8 (col 1) → 0*9+1 = 1
# Note: spatial encoding uses destination square for drops but source
# square for board moves.  The source square index from the action is
# perspective-relative (rotated for white), so we check against the
# black perspective square for both sides.
BLACK_ROOK_SQUARE = 79  # h1 in board coords (row=8, col=7)
# King starting squares
BLACK_KING_SQUARE = 76  # e1 in board coords (row=8, col=4)

# "Early" game threshold for drops (within first 40 plies)
EARLY_DROP_PLY_THRESHOLD = 40
# Opening sequence tracking
OPENING_SEQ_3_LEN = 3
OPENING_SEQ_6_LEN = 6
# Rook mobility window
ROOK_MOBILITY_PLY = 20
# King movement window
KING_MOVEMENT_PLY = 30


def classify_action(action_id: int) -> tuple[bool, bool, int]:
    """Classify an action from its spatial encoding.

    Returns:
        (is_drop, is_promotion, source_square)
        source_square is the perspective-relative source for board moves,
        or the destination square for drops.
    """
    source_square = action_id // SPATIAL_MOVE_TYPES
    move_type = action_id % SPATIAL_MOVE_TYPES
    is_drop = DROP_MOVE_TYPE_MIN <= move_type <= DROP_MOVE_TYPE_MAX
    is_promotion = PROMOTION_MOVE_TYPE_MIN <= move_type <= PROMOTION_MOVE_TYPE_MAX
    return is_drop, is_promotion, source_square


@dataclass
class GameFeatureAccumulator:
    """Tracks features for one game in one env slot."""

    # Opening
    actions: list[int] = field(default_factory=list)
    # Tempo and aggression
    first_capture_ply: int | None = None
    first_check_ply: int | None = None
    first_drop_ply: int | None = None
    num_checks: int = 0
    num_captures: int = 0
    # Drop and promotion
    num_drops: int = 0
    num_promotions: int = 0
    num_early_drops: int = 0
    # Positional proxies
    rook_moved_ply: int | None = None
    rook_moves_in_20: int = 0
    king_displacement_20: int = 0
    king_moves_in_30: int = 0
    num_repetitions: int = 0
    # Tracking state
    _ply: int = 0

    def reset(self) -> None:
        """Reset for a new game (after VecEnv auto-reset)."""
        self.actions.clear()
        self.first_capture_ply = None
        self.first_check_ply = None
        self.first_drop_ply = None
        self.num_checks = 0
        self.num_captures = 0
        self.num_drops = 0
        self.num_promotions = 0
        self.num_early_drops = 0
        self.rook_moved_ply = None
        self.rook_moves_in_20 = 0
        self.king_displacement_20 = 0
        self.king_moves_in_30 = 0
        self.num_repetitions = 0
        self._ply = 0


@dataclass
class GameFeatureRow:
    """A completed game's features, ready for DB insertion."""

    checkpoint_id: int
    opponent_id: int
    epoch: int
    side: str  # "black" or "white"
    result: str  # "win", "loss", "draw"
    total_plies: int
    first_action: int | None
    opening_seq_3: str | None
    opening_seq_6: str | None
    rook_moved_ply: int | None
    king_displacement_20: int
    first_capture_ply: int | None
    first_check_ply: int | None
    first_drop_ply: int | None
    num_checks: int
    num_captures: int
    num_drops: int
    num_promotions: int
    num_early_drops: int
    rook_moves_in_20: int
    king_moves_in_30: int
    num_repetitions: int
    termination_reason: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for write_game_features()."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "opponent_id": self.opponent_id,
            "epoch": self.epoch,
            "side": self.side,
            "result": self.result,
            "total_plies": self.total_plies,
            "first_action": self.first_action,
            "opening_seq_3": self.opening_seq_3,
            "opening_seq_6": self.opening_seq_6,
            "rook_moved_ply": self.rook_moved_ply,
            "king_displacement_20": self.king_displacement_20,
            "first_capture_ply": self.first_capture_ply,
            "first_check_ply": self.first_check_ply,
            "first_drop_ply": self.first_drop_ply,
            "num_checks": self.num_checks,
            "num_captures": self.num_captures,
            "num_drops": self.num_drops,
            "num_promotions": self.num_promotions,
            "num_early_drops": self.num_early_drops,
            "rook_moves_in_20": self.rook_moves_in_20,
            "king_moves_in_30": self.king_moves_in_30,
            "num_repetitions": self.num_repetitions,
            "termination_reason": self.termination_reason,
        }


class GameFeatureTracker:
    """Tracks per-game features across all envs in a batch.

    Usage::

        tracker = GameFeatureTracker(num_envs, entry_a_id, entry_b_id, epoch)
        # In play_batch loop:
        tracker.record_step(actions_np, captured_np, term_reason_np,
                            ply_count_np, current_players_np,
                            terminated_np, truncated_np, rewards_np)
        # After batch, collect completed rows:
        rows = tracker.completed_rows
    """

    def __init__(
        self,
        num_envs: int,
        entry_a_id: int,
        entry_b_id: int,
        epoch: int,
    ) -> None:
        self.num_envs = num_envs
        self.entry_a_id = entry_a_id  # player A = side 0 (black)
        self.entry_b_id = entry_b_id  # player B = side 1 (white)
        self.epoch = epoch
        self.accumulators = [GameFeatureAccumulator() for _ in range(num_envs)]
        self.completed_rows: list[GameFeatureRow] = []

    def record_step(
        self,
        actions: np.ndarray,
        captured_piece: np.ndarray,
        termination_reason: np.ndarray,
        ply_count: np.ndarray,
        pre_step_players: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        """Record one step across all envs.

        Args:
            actions: Action IDs taken this step, shape (num_envs,).
            captured_piece: From StepMetadata, 255=no capture, shape (num_envs,).
            termination_reason: From StepMetadata, shape (num_envs,).
            ply_count: From StepMetadata, shape (num_envs,).
            pre_step_players: Who moved this step (0=A/black, 1=B/white).
            terminated: Whether game ended by terminal condition.
            truncated: Whether game ended by truncation.
            rewards: Per-env rewards (from last-mover's perspective).
        """
        done = terminated | truncated

        for i in range(self.num_envs):
            acc = self.accumulators[i]
            action = int(actions[i])
            ply = int(ply_count[i])
            acc._ply = ply
            mover = int(pre_step_players[i])

            is_drop, is_promotion, source_sq = classify_action(action)

            # Track opening actions (first 6 plies per side = 12 total plies)
            if len(acc.actions) < OPENING_SEQ_6_LEN * 2:
                acc.actions.append(action)

            # Captures
            cap = int(captured_piece[i])
            if cap != NO_CAPTURE:
                acc.num_captures += 1
                if acc.first_capture_ply is None:
                    acc.first_capture_ply = ply

            # Drops
            if is_drop:
                acc.num_drops += 1
                if acc.first_drop_ply is None:
                    acc.first_drop_ply = ply
                if ply <= EARLY_DROP_PLY_THRESHOLD:
                    acc.num_early_drops += 1

            # Promotions
            if is_promotion:
                acc.num_promotions += 1

            # Rook movement (perspective-relative: source_sq is from mover's
            # perspective, so BLACK_ROOK_SQUARE works for both sides)
            if not is_drop and source_sq == BLACK_ROOK_SQUARE:
                if acc.rook_moved_ply is None:
                    acc.rook_moved_ply = ply
                if ply <= ROOK_MOBILITY_PLY:
                    acc.rook_moves_in_20 += 1

            # King movement
            if not is_drop and source_sq == BLACK_KING_SQUARE:
                if ply <= 20:
                    acc.king_displacement_20 += 1
                if ply <= KING_MOVEMENT_PLY:
                    acc.king_moves_in_30 += 1

            # Repetition detection (termination_reason == 2)
            tr = int(termination_reason[i])
            if done[i] and tr == 2:
                acc.num_repetitions += 1

            # Game ended — emit feature rows for both sides
            if done[i]:
                self._emit_game(i, ply, tr, mover, float(rewards[i]))

    def _emit_game(
        self,
        env_idx: int,
        total_plies: int,
        termination_reason: int,
        last_mover: int,
        reward: float,
    ) -> None:
        """Produce feature rows for both players of a completed game."""
        acc = self.accumulators[env_idx]

        # Determine results: reward > 0 means last_mover won
        if reward > 0:
            winner = last_mover
        elif reward < 0:
            winner = 1 - last_mover
        else:
            winner = -1  # draw

        # Build opening sequences from accumulated actions
        # Actions alternate: ply 1 = A's move, ply 2 = B's move, etc.
        a_actions = acc.actions[0::2]  # A's moves (plies 1, 3, 5, ...)
        b_actions = acc.actions[1::2]  # B's moves (plies 2, 4, 6, ...)

        def _opening_seq(acts: list[int], length: int) -> str | None:
            if len(acts) >= length:
                return ",".join(str(a) for a in acts[:length])
            return None

        for side_idx, side_name, side_actions in [
            (0, "black", a_actions),
            (1, "white", b_actions),
        ]:
            if winner == side_idx:
                result = "win"
            elif winner == -1:
                result = "draw"
            else:
                result = "loss"

            checkpoint_id = self.entry_a_id if side_idx == 0 else self.entry_b_id
            opponent_id = self.entry_b_id if side_idx == 0 else self.entry_a_id

            row = GameFeatureRow(
                checkpoint_id=checkpoint_id,
                opponent_id=opponent_id,
                epoch=self.epoch,
                side=side_name,
                result=result,
                total_plies=total_plies,
                first_action=side_actions[0] if side_actions else None,
                opening_seq_3=_opening_seq(side_actions, OPENING_SEQ_3_LEN),
                opening_seq_6=_opening_seq(side_actions, OPENING_SEQ_6_LEN),
                rook_moved_ply=acc.rook_moved_ply,
                king_displacement_20=acc.king_displacement_20,
                first_capture_ply=acc.first_capture_ply,
                first_check_ply=acc.first_check_ply,
                first_drop_ply=acc.first_drop_ply,
                num_checks=acc.num_checks,
                num_captures=acc.num_captures,
                num_drops=acc.num_drops,
                num_promotions=acc.num_promotions,
                num_early_drops=acc.num_early_drops,
                rook_moves_in_20=acc.rook_moves_in_20,
                king_moves_in_30=acc.king_moves_in_30,
                num_repetitions=acc.num_repetitions,
                termination_reason=termination_reason,
            )
            self.completed_rows.append(row)

        # Reset accumulator for next game (VecEnv auto-resets)
        acc.reset()
