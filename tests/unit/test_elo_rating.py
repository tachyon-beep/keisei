"""Unit tests for keisei.training.elo_rating: EloRatingSystem."""

import pytest

from keisei.shogi.shogi_core_definitions import Color
from keisei.training.elo_rating import EloRatingSystem


class TestEloRatingSystemInit:
    """Tests for EloRatingSystem initialization."""

    def test_default_initialization(self):
        elo = EloRatingSystem()
        assert elo.initial_rating == 1500.0
        assert elo.k_factor == 32.0
        assert elo.black_rating == 1500.0
        assert elo.white_rating == 1500.0
        assert elo.rating_history == []

    def test_custom_initialization(self):
        elo = EloRatingSystem(initial_rating=1200.0, k_factor=16.0)
        assert elo.initial_rating == 1200.0
        assert elo.k_factor == 16.0
        assert elo.black_rating == 1200.0
        assert elo.white_rating == 1200.0


class TestExpectedScore:
    """Tests for the _expected_score static method."""

    def test_equal_ratings_returns_half(self):
        score = EloRatingSystem._expected_score(1500.0, 1500.0)
        assert score == pytest.approx(0.5)

    def test_higher_rating_a_near_one(self):
        score = EloRatingSystem._expected_score(2000.0, 1000.0)
        assert score > 0.99

    def test_lower_rating_a_near_zero(self):
        score = EloRatingSystem._expected_score(1000.0, 2000.0)
        assert score < 0.01

    def test_400_point_gap(self):
        score = EloRatingSystem._expected_score(1900.0, 1500.0)
        assert score == pytest.approx(10.0 / 11.0, abs=1e-4)


class TestUpdateRatings:
    """Tests for update_ratings method."""

    def test_black_win_increases_black_rating(self):
        elo = EloRatingSystem()
        result = elo.update_ratings(Color.BLACK)
        assert result["black_rating"] > 1500.0
        assert result["white_rating"] < 1500.0

    def test_white_win_increases_white_rating(self):
        elo = EloRatingSystem()
        result = elo.update_ratings(Color.WHITE)
        assert result["white_rating"] > 1500.0
        assert result["black_rating"] < 1500.0

    def test_draw_keeps_ratings_equal(self):
        elo = EloRatingSystem()
        result = elo.update_ratings(None)
        assert result["black_rating"] == pytest.approx(1500.0)
        assert result["white_rating"] == pytest.approx(1500.0)

    def test_rating_difference_key_present(self):
        elo = EloRatingSystem()
        result = elo.update_ratings(Color.BLACK)
        assert "rating_difference" in result
        assert result["rating_difference"] == pytest.approx(
            result["black_rating"] - result["white_rating"]
        )

    def test_rating_history_tracking(self):
        elo = EloRatingSystem()
        elo.update_ratings(Color.BLACK)
        elo.update_ratings(Color.WHITE)
        assert len(elo.rating_history) == 2
        assert "black_rating" in elo.rating_history[0]
        assert "white_rating" in elo.rating_history[0]
        assert "difference" in elo.rating_history[0]

    def test_sequential_updates_accumulate(self):
        elo = EloRatingSystem()
        # Three black wins should significantly raise black's rating
        for _ in range(3):
            elo.update_ratings(Color.BLACK)
        assert elo.black_rating > 1500.0 + 30  # Should accumulate well above initial
        assert elo.white_rating < 1500.0 - 30


class TestStrengthAssessment:
    """Tests for get_strength_assessment at each threshold boundary."""

    def test_balanced(self):
        elo = EloRatingSystem()
        assert elo.get_strength_assessment() == "Balanced"

    def test_slight_advantage(self):
        elo = EloRatingSystem()
        elo.black_rating = 1560.0  # diff = 60
        assert elo.get_strength_assessment() == "Slight advantage"

    def test_clear_advantage(self):
        elo = EloRatingSystem()
        elo.black_rating = 1650.0  # diff = 150
        assert elo.get_strength_assessment() == "Clear advantage"

    def test_strong_advantage(self):
        elo = EloRatingSystem()
        elo.black_rating = 1800.0  # diff = 300
        assert elo.get_strength_assessment() == "Strong advantage"

    def test_overwhelming_advantage(self):
        elo = EloRatingSystem()
        elo.black_rating = 2000.0  # diff = 500
        assert elo.get_strength_assessment() == "Overwhelming advantage"

    def test_boundary_at_50(self):
        elo = EloRatingSystem()
        elo.black_rating = 1549.9
        assert elo.get_strength_assessment() == "Balanced"
        elo.black_rating = 1550.0
        assert elo.get_strength_assessment() == "Slight advantage"

    def test_white_advantage_also_detected(self):
        elo = EloRatingSystem()
        elo.white_rating = 1650.0  # diff = 150, within "Clear advantage" range
        assert elo.get_strength_assessment() == "Clear advantage"
