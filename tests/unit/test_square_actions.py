"""Unit tests for per-square action data in the state contract."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


class TestActionProbContract:
    """ActionProb and square_actions field in PolicyInsight."""

    def test_action_prob_typeddict_exists(self):
        """ActionProb TypedDict is importable."""
        from keisei.webui.view_contracts import ActionProb

        ap: ActionProb = {"action": "7g7f", "prob": 0.23}
        assert ap["action"] == "7g7f"
        assert ap["prob"] == 0.23

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


class TestPolicyInsightErrorPath:
    """extract_policy_insight returns None on error and restores model mode."""

    def test_returns_none_on_forward_pass_error(self):
        """If model() raises, extract_policy_insight returns None."""
        from keisei.webui.state_snapshot import extract_policy_insight

        agent = MagicMock()
        agent.device = torch.device("cpu")
        agent.scaler = None
        model = MagicMock()
        model.training = True
        model.side_effect = RuntimeError("CUDA error")
        model.return_value = None
        model.__call__ = MagicMock(side_effect=RuntimeError("CUDA error"))
        agent.model = model

        mapper = MagicMock()
        obs = np.zeros((46, 9, 9), dtype=np.float32)

        result = extract_policy_insight(agent, obs, mapper, top_k=5)
        assert result is None

    def test_restores_model_mode_on_error(self):
        """Model training mode is restored even if forward pass raises."""
        from keisei.webui.state_snapshot import extract_policy_insight

        agent = MagicMock()
        agent.device = torch.device("cpu")
        agent.scaler = None
        model = MagicMock()
        model.training = True
        model.__call__ = MagicMock(side_effect=RuntimeError("shape mismatch"))
        agent.model = model

        mapper = MagicMock()
        obs = np.zeros((46, 9, 9), dtype=np.float32)

        extract_policy_insight(agent, obs, mapper, top_k=5)
        # model.train(True) should have been called via finally block
        model.train.assert_called_with(True)

    def test_legal_mask_none_still_works(self):
        """extract_policy_insight produces results when legal_mask is None."""
        from keisei.webui.state_snapshot import extract_policy_insight

        agent, mapper = TestSquareActionExtraction()._make_mock_agent_and_mapper()
        obs = np.zeros((46, 9, 9), dtype=np.float32)

        result = extract_policy_insight(agent, obs, mapper, top_k=5, legal_mask=None)
        assert result is not None
        assert "action_heatmap" in result
        assert "square_actions" in result

    def test_mismatched_legal_mask_shape_uses_unmasked(self):
        """A legal_mask with wrong shape is silently ignored (unmasked softmax)."""
        from keisei.webui.state_snapshot import extract_policy_insight

        agent, mapper = TestSquareActionExtraction()._make_mock_agent_and_mapper()
        obs = np.zeros((46, 9, 9), dtype=np.float32)
        # Wrong shape: 100 instead of 13527
        bad_mask = torch.ones(100, dtype=torch.bool)

        result = extract_policy_insight(agent, obs, mapper, top_k=5, legal_mask=bad_mask)
        # Should still produce a result using unmasked probabilities
        assert result is not None
        assert "action_heatmap" in result


class TestPolicyInsightGating:
    """_build_training_view gates policy insight on config and processing state."""

    def test_insight_none_when_disabled(self):
        """policy_insight is None when config flag is off."""
        from types import SimpleNamespace

        from keisei.webui.state_snapshot import _build_training_view

        trainer = MagicMock()
        trainer.config = SimpleNamespace(
            webui=SimpleNamespace(policy_insight=False, policy_insight_top_k=10)
        )
        trainer.metrics_manager.get_stats_dict.return_value = {}
        trainer.metrics_manager.get_learning_curves.return_value = {}
        trainer.step_manager.move_log = []
        trainer.step_manager.move_history = []
        trainer.game.get_board_state_dict.return_value = {"board": []}
        trainer.last_gradient_norm = 0.0

        result = _build_training_view(trainer)
        assert result["policy_insight"] is None

    def test_insight_none_when_processing(self):
        """policy_insight is None when PPO update is in progress."""
        from types import SimpleNamespace

        from keisei.webui.state_snapshot import _build_training_view

        trainer = MagicMock()
        trainer.config = SimpleNamespace(
            webui=SimpleNamespace(policy_insight=True, policy_insight_top_k=10)
        )
        stats = {"processing": True}
        trainer.metrics_manager.get_stats_dict.return_value = stats
        trainer.metrics_manager.get_learning_curves.return_value = {}
        trainer.step_manager.move_log = []
        trainer.step_manager.move_history = []
        trainer.game.get_board_state_dict.return_value = {"board": []}
        trainer.last_gradient_norm = 0.0

        result = _build_training_view(trainer)
        assert result["policy_insight"] is None
