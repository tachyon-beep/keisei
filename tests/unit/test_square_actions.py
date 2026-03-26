"""Unit tests for per-square action data in the state contract."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

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


class TestSquareActionExtraction:
    """extract_policy_insight produces correct square_actions."""

    def _make_mock_agent_and_mapper(self):
        """Create mocks that produce a known probability distribution."""
        agent = MagicMock()
        agent.device = torch.device("cpu")
        agent.scaler = None

        # Model that returns known logits
        model = MagicMock()
        model.training = False
        # 13527 logits, all zero (uniform) except a few
        logits = torch.zeros(1, 13527)
        logits[0, 0] = 5.0  # High prob action targeting (2, 2)
        logits[0, 1] = 3.0  # Medium prob action targeting (2, 2)
        logits[0, 2] = 2.0  # Lower prob action targeting (4, 4)
        model.return_value = (logits, torch.tensor([[0.5]]))
        model.eval = MagicMock()
        model.train = MagicMock()
        agent.model = model

        # Policy mapper with known move destinations
        idx_to_move = [(0, 0, 2, 2, False)] * 13527  # All point to (2,2) by default
        idx_to_move[0] = (0, 0, 2, 2, False)  # Action 0 -> dest (2,2)
        idx_to_move[1] = (1, 1, 2, 2, False)  # Action 1 -> dest (2,2)
        idx_to_move[2] = (3, 3, 4, 4, False)  # Action 2 -> dest (4,4)
        mapper = MagicMock()
        mapper.idx_to_move = idx_to_move
        mapper.action_idx_to_usi_move = lambda idx: f"act{idx}"

        return agent, mapper

    def test_square_actions_present_in_result(self):
        """extract_policy_insight returns square_actions key."""
        from keisei.webui.state_snapshot import extract_policy_insight

        agent, mapper = self._make_mock_agent_and_mapper()
        obs = np.zeros((46, 9, 9), dtype=np.float32)
        legal_mask = torch.ones(13527, dtype=torch.bool)

        result = extract_policy_insight(
            agent, obs, mapper, top_k=5, legal_mask=legal_mask
        )
        assert result is not None
        assert "square_actions" in result

    def test_square_actions_top3_per_square(self):
        """Each square gets at most 3 actions."""
        from keisei.webui.state_snapshot import extract_policy_insight

        agent, mapper = self._make_mock_agent_and_mapper()
        obs = np.zeros((46, 9, 9), dtype=np.float32)
        legal_mask = torch.ones(13527, dtype=torch.bool)

        result = extract_policy_insight(
            agent, obs, mapper, top_k=5, legal_mask=legal_mask
        )
        sa = result["square_actions"]
        for key, actions in sa.items():
            assert len(actions) <= 3

    def test_square_actions_key_format(self):
        """Keys use 'r,c' format with 0-indexed coordinates."""
        from keisei.webui.state_snapshot import extract_policy_insight

        agent, mapper = self._make_mock_agent_and_mapper()
        obs = np.zeros((46, 9, 9), dtype=np.float32)
        legal_mask = torch.ones(13527, dtype=torch.bool)

        result = extract_policy_insight(
            agent, obs, mapper, top_k=5, legal_mask=legal_mask
        )
        sa = result["square_actions"]
        for key in sa:
            parts = key.split(",")
            assert len(parts) == 2
            r, c = int(parts[0]), int(parts[1])
            assert 0 <= r < 9
            assert 0 <= c < 9

    def test_square_actions_sorted_by_prob(self):
        """Actions within each square are sorted descending by probability."""
        from keisei.webui.state_snapshot import extract_policy_insight

        agent, mapper = self._make_mock_agent_and_mapper()
        obs = np.zeros((46, 9, 9), dtype=np.float32)
        legal_mask = torch.ones(13527, dtype=torch.bool)

        result = extract_policy_insight(
            agent, obs, mapper, top_k=5, legal_mask=legal_mask
        )
        sa = result["square_actions"]
        for key, actions in sa.items():
            probs = [a["prob"] for a in actions]
            assert probs == sorted(probs, reverse=True)
