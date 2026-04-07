"""Tests for GameFeatureTracker inline feature extraction."""

import numpy as np

from keisei.training.game_feature_tracker import (
    EARLY_DROP_PLY_THRESHOLD,
    NO_CAPTURE,
    SPATIAL_MOVE_TYPES,
    GameFeatureAccumulator,
    GameFeatureTracker,
    classify_action,
)


class TestClassifyAction:
    def test_regular_move(self):
        # Square 0, move type 0 (N direction, distance 1)
        is_drop, is_promotion, sq = classify_action(0)
        assert not is_drop
        assert not is_promotion
        assert sq == 0

    def test_promotion_move(self):
        # Square 0, move type 64 (first promotion type)
        action = 0 * SPATIAL_MOVE_TYPES + 64
        is_drop, is_promotion, sq = classify_action(action)
        assert not is_drop
        assert is_promotion
        assert sq == 0

    def test_drop_move(self):
        # Square 40 (center), move type 132 (first drop type)
        action = 40 * SPATIAL_MOVE_TYPES + 132
        is_drop, is_promotion, sq = classify_action(action)
        assert is_drop
        assert not is_promotion
        assert sq == 40

    def test_last_drop_type(self):
        # Square 80, move type 138 (last drop type)
        action = 80 * SPATIAL_MOVE_TYPES + 138
        is_drop, is_promotion, sq = classify_action(action)
        assert is_drop
        assert not is_promotion

    def test_last_promotion_type(self):
        # Move type 131 (last promotion type)
        action = 10 * SPATIAL_MOVE_TYPES + 131
        is_drop, is_promotion, sq = classify_action(action)
        assert not is_drop
        assert is_promotion


class TestGameFeatureAccumulator:
    def test_initial_state(self):
        acc = GameFeatureAccumulator()
        assert acc.num_drops == 0
        assert acc.num_captures == 0
        assert acc.num_promotions == 0
        assert acc.first_capture_ply is None

    def test_reset(self):
        acc = GameFeatureAccumulator()
        acc.num_drops = 5
        acc.first_capture_ply = 10
        acc.actions.append(42)
        acc.reset()
        assert acc.num_drops == 0
        assert acc.first_capture_ply is None
        assert len(acc.actions) == 0


class TestGameFeatureTracker:
    def _make_tracker(self, num_envs=2):
        return GameFeatureTracker(
            num_envs=num_envs, entry_a_id=1, entry_b_id=2, epoch=10
        )

    def _step(self, tracker, actions, captured=None, terminated=None,
              truncated=None, rewards=None, pre_step_players=None,
              term_reason=None, ply_count=None):
        n = tracker.num_envs
        if captured is None:
            captured = np.full(n, NO_CAPTURE, dtype=np.uint8)
        if terminated is None:
            terminated = np.zeros(n, dtype=bool)
        if truncated is None:
            truncated = np.zeros(n, dtype=bool)
        if rewards is None:
            rewards = np.zeros(n, dtype=np.float32)
        if pre_step_players is None:
            pre_step_players = np.zeros(n, dtype=np.uint8)
        if term_reason is None:
            term_reason = np.zeros(n, dtype=np.uint8)
        if ply_count is None:
            ply_count = np.ones(n, dtype=np.uint16)
        tracker.record_step(
            actions=np.array(actions, dtype=np.int64),
            captured_piece=captured,
            termination_reason=term_reason,
            ply_count=ply_count,
            pre_step_players=pre_step_players,
            terminated=terminated,
            truncated=truncated,
            rewards=rewards,
        )

    def test_no_games_completed(self):
        tracker = self._make_tracker()
        # Regular move, no game end
        self._step(tracker, [0, 0])
        assert len(tracker.completed_rows) == 0

    def test_capture_tracking(self):
        tracker = self._make_tracker(num_envs=1)
        # Step with a capture
        self._step(tracker, [0], captured=np.array([3], dtype=np.uint8),
                   ply_count=np.array([5], dtype=np.uint16))
        assert tracker.accumulators[0].num_captures == 1
        assert tracker.accumulators[0].first_capture_ply == 5

    def test_drop_tracking(self):
        tracker = self._make_tracker(num_envs=1)
        # Drop action: square 40, move_type 132
        drop_action = 40 * SPATIAL_MOVE_TYPES + 132
        self._step(tracker, [drop_action],
                   ply_count=np.array([3], dtype=np.uint16))
        assert tracker.accumulators[0].num_drops == 1
        assert tracker.accumulators[0].first_drop_ply == 3

    def test_early_drop_counting(self):
        tracker = self._make_tracker(num_envs=1)
        drop_action = 40 * SPATIAL_MOVE_TYPES + 133
        # Early drop (within threshold)
        self._step(tracker, [drop_action],
                   ply_count=np.array([10], dtype=np.uint16))
        assert tracker.accumulators[0].num_early_drops == 1
        # Late drop (past threshold)
        self._step(tracker, [drop_action],
                   ply_count=np.array([EARLY_DROP_PLY_THRESHOLD + 5], dtype=np.uint16))
        assert tracker.accumulators[0].num_early_drops == 1  # not incremented
        assert tracker.accumulators[0].num_drops == 2

    def test_promotion_tracking(self):
        tracker = self._make_tracker(num_envs=1)
        promo_action = 10 * SPATIAL_MOVE_TYPES + 64
        self._step(tracker, [promo_action])
        assert tracker.accumulators[0].num_promotions == 1

    def test_game_completion_produces_two_rows(self):
        tracker = self._make_tracker(num_envs=1)
        # Play a few steps then end game with A winning
        self._step(tracker, [0],
                   pre_step_players=np.array([0], dtype=np.uint8),
                   ply_count=np.array([1], dtype=np.uint16))
        self._step(
            tracker, [0],
            pre_step_players=np.array([0], dtype=np.uint8),
            terminated=np.array([True]),
            rewards=np.array([1.0], dtype=np.float32),
            ply_count=np.array([10], dtype=np.uint16),
        )
        # Should produce 2 rows: one for each side
        assert len(tracker.completed_rows) == 2
        black_row = next(r for r in tracker.completed_rows if r.side == "black")
        white_row = next(r for r in tracker.completed_rows if r.side == "white")
        assert black_row.result == "win"
        assert white_row.result == "loss"
        assert black_row.checkpoint_id == 1  # entry_a = black
        assert white_row.checkpoint_id == 2  # entry_b = white

    def test_draw_result(self):
        tracker = self._make_tracker(num_envs=1)
        self._step(
            tracker, [0],
            terminated=np.array([True]),
            rewards=np.array([0.0], dtype=np.float32),
            ply_count=np.array([50], dtype=np.uint16),
        )
        assert len(tracker.completed_rows) == 2
        assert all(r.result == "draw" for r in tracker.completed_rows)

    def test_opening_sequence_tracking(self):
        tracker = self._make_tracker(num_envs=1)
        # Simulate 6 plies of moves
        for ply in range(6):
            player = ply % 2
            self._step(
                tracker, [ply + 1],
                pre_step_players=np.array([player], dtype=np.uint8),
                ply_count=np.array([ply + 1], dtype=np.uint16),
            )
        # End game
        self._step(
            tracker, [99],
            terminated=np.array([True]),
            rewards=np.array([1.0], dtype=np.float32),
            ply_count=np.array([7], dtype=np.uint16),
        )
        black_row = next(r for r in tracker.completed_rows if r.side == "black")
        # Black moves on plies 1, 3, 5 → actions 1, 3, 5
        assert black_row.first_action == 1
        assert black_row.opening_seq_3 == "1,3,5"

    def test_to_dict(self):
        tracker = self._make_tracker(num_envs=1)
        self._step(
            tracker, [0],
            terminated=np.array([True]),
            rewards=np.array([1.0], dtype=np.float32),
            ply_count=np.array([10], dtype=np.uint16),
        )
        row = tracker.completed_rows[0]
        d = row.to_dict()
        assert "checkpoint_id" in d
        assert "side" in d
        assert "result" in d
        assert "total_plies" in d
        assert "num_drops" in d
        assert "num_promotions" in d

    def test_accumulator_resets_after_game_end(self):
        tracker = self._make_tracker(num_envs=1)
        # Capture on first game
        self._step(
            tracker, [0],
            captured=np.array([1], dtype=np.uint8),
            ply_count=np.array([5], dtype=np.uint16),
        )
        # End first game
        self._step(
            tracker, [0],
            terminated=np.array([True]),
            rewards=np.array([1.0], dtype=np.float32),
            ply_count=np.array([10], dtype=np.uint16),
        )
        # Accumulator should be reset
        assert tracker.accumulators[0].num_captures == 0
        assert tracker.accumulators[0].first_capture_ply is None

    def test_repetition_tracking(self):
        tracker = self._make_tracker(num_envs=1)
        self._step(
            tracker, [0],
            terminated=np.array([True]),
            rewards=np.array([0.0], dtype=np.float32),
            term_reason=np.array([2], dtype=np.uint8),  # Repetition
            ply_count=np.array([30], dtype=np.uint16),
        )
        rows = tracker.completed_rows
        assert rows[0].num_repetitions == 1
