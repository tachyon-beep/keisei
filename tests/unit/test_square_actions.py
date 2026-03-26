"""Unit tests for per-square action data in the state contract."""

import pytest

pytestmark = pytest.mark.unit


class TestSquareActionContract:
    """SquareAction and square_actions field in PolicyInsight."""

    def test_square_action_typeddict_exists(self):
        """SquareAction TypedDict is importable."""
        from keisei.webui.view_contracts import SquareAction

        # TypedDict instances are just dicts
        sa: SquareAction = {"action": "7g7f", "prob": 0.23}
        assert sa["action"] == "7g7f"
        assert sa["prob"] == 0.23

    def test_policy_insight_allows_square_actions(self):
        """PolicyInsight accepts square_actions field."""
        from keisei.webui.view_contracts import PolicyInsight

        pi: PolicyInsight = {
            "action_heatmap": [[0.0] * 9 for _ in range(9)],
            "top_actions": [],
            "value_estimate": 0.0,
            "action_entropy": 0.0,
            "square_actions": {
                "5,2": [{"action": "7g7f", "prob": 0.23}],
            },
        }
        assert "5,2" in pi["square_actions"]
