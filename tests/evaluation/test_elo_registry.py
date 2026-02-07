import json
from pathlib import Path

from keisei.evaluation.opponents.elo_registry import EloRegistry


def test_elo_registry_load_update_save(tmp_path):
    path = tmp_path / "elo.json"
    registry = EloRegistry(path)
    assert registry.get_rating("A") == 1500.0
    results = ["agent_win", "agent_win", "draw", "opponent_win"]
    registry.update_ratings("A", "B", results)
    registry.save()
    assert path.exists()
    data = json.loads(path.read_text())
    assert "A" in data["ratings"] and "B" in data["ratings"]
    assert abs(data["ratings"]["A"] - 1500.0) > 1e-6  # A's rating should have changed


def test_elo_batch_scaling(tmp_path):
    """Larger batches should produce larger rating changes than small batches."""
    # Single game
    r1 = EloRegistry(tmp_path / "elo1.json")
    r1.update_ratings("A", "B", ["agent_win"])
    delta_1 = abs(r1.get_rating("A") - 1500.0)

    # 10-game sweep (all wins)
    r10 = EloRegistry(tmp_path / "elo10.json")
    r10.update_ratings("A", "B", ["agent_win"] * 10)
    delta_10 = abs(r10.get_rating("A") - 1500.0)

    # 10 wins must shift ratings more than 1 win
    assert delta_10 > delta_1 * 5  # should be ~10x, use 5x as conservative bound


def test_elo_equal_players_draw_no_change(tmp_path):
    """Equal-rated players drawing should not change ratings."""
    registry = EloRegistry(tmp_path / "elo.json")
    registry.update_ratings("A", "B", ["draw"])
    # Expected score for equal players is 0.5, draw scores 0.5 => no change
    assert abs(registry.get_rating("A") - 1500.0) < 1e-6
    assert abs(registry.get_rating("B") - 1500.0) < 1e-6


def test_elo_empty_results_no_change(tmp_path):
    """Empty results list should not change ratings."""
    registry = EloRegistry(tmp_path / "elo.json")
    registry.update_ratings("A", "B", [])
    assert registry.get_rating("A") == 1500.0
    assert registry.get_rating("B") == 1500.0


def test_elo_symmetry(tmp_path):
    """Rating changes should be symmetric (zero-sum)."""
    registry = EloRegistry(tmp_path / "elo.json")
    registry.update_ratings("A", "B", ["agent_win", "opponent_win", "agent_win"])
    delta_a = registry.get_rating("A") - 1500.0
    delta_b = registry.get_rating("B") - 1500.0
    assert abs(delta_a + delta_b) < 1e-6  # zero-sum
