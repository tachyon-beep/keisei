"""Unit tests for keisei.utils.agent_loading."""

from unittest.mock import MagicMock, patch

import pytest

from keisei.utils.agent_loading import initialize_opponent, load_evaluation_agent
from keisei.utils.opponents import SimpleHeuristicOpponent, SimpleRandomOpponent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_policy_mapper():
    mapper = MagicMock()
    mapper.get_total_actions.return_value = 13527
    return mapper


# ---------------------------------------------------------------------------
# TestLoadEvaluationAgent
# ---------------------------------------------------------------------------


class TestLoadEvaluationAgent:
    def test_missing_checkpoint_raises_file_not_found(self, mock_policy_mapper):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_evaluation_agent(
                "/nonexistent/path/model.pt",
                "cpu",
                mock_policy_mapper,
                input_channels=46,
            )

    @patch("keisei.utils.agent_loading.os.path.isfile", return_value=True)
    def test_loads_checkpoint_and_sets_eval_mode(
        self, mock_isfile, mock_policy_mapper
    ):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        mock_actor_critic_cls = MagicMock(return_value=mock_model)
        mock_agent = MagicMock()
        mock_ppo_agent_cls = MagicMock(return_value=mock_agent)

        with (
            patch(
                "keisei.core.neural_network.ActorCritic",
                mock_actor_critic_cls,
            ),
            patch(
                "keisei.core.ppo_agent.PPOAgent",
                mock_ppo_agent_cls,
            ),
        ):
            result = load_evaluation_agent(
                "/fake/checkpoint.pt",
                "cpu",
                mock_policy_mapper,
                input_channels=46,
            )

        mock_agent.load_model.assert_called_once_with("/fake/checkpoint.pt")
        mock_agent.model.eval.assert_called_once()
        assert result is mock_agent

    @patch("keisei.utils.agent_loading.os.path.isfile", return_value=True)
    def test_model_placed_on_correct_device(self, mock_isfile, mock_policy_mapper):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        mock_actor_critic_cls = MagicMock(return_value=mock_model)
        mock_ppo_agent_cls = MagicMock(return_value=MagicMock())

        with (
            patch(
                "keisei.core.neural_network.ActorCritic",
                mock_actor_critic_cls,
            ),
            patch(
                "keisei.core.ppo_agent.PPOAgent",
                mock_ppo_agent_cls,
            ),
        ):
            load_evaluation_agent(
                "/fake/checkpoint.pt",
                "cpu",
                mock_policy_mapper,
                input_channels=46,
            )

        # The model's .to() should have been called with a torch.device("cpu")
        import torch

        mock_model.to.assert_called_once_with(torch.device("cpu"))

    @patch("keisei.utils.agent_loading.os.path.isfile", return_value=True)
    def test_uses_policy_mapper_total_actions(self, mock_isfile, mock_policy_mapper):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        mock_actor_critic_cls = MagicMock(return_value=mock_model)
        mock_ppo_agent_cls = MagicMock(return_value=MagicMock())

        with (
            patch(
                "keisei.core.neural_network.ActorCritic",
                mock_actor_critic_cls,
            ),
            patch(
                "keisei.core.ppo_agent.PPOAgent",
                mock_ppo_agent_cls,
            ),
        ):
            load_evaluation_agent(
                "/fake/checkpoint.pt",
                "cpu",
                mock_policy_mapper,
                input_channels=46,
            )

        mock_policy_mapper.get_total_actions.assert_called()
        # ActorCritic should have been called with (input_channels, total_actions)
        mock_actor_critic_cls.assert_called_once_with(46, 13527)

    @patch("keisei.utils.agent_loading.os.path.isfile", return_value=True)
    def test_custom_input_features_passed_to_config(
        self, mock_isfile, mock_policy_mapper
    ):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        mock_actor_critic_cls = MagicMock(return_value=mock_model)

        captured_config = {}

        def capture_ppo_init(**kwargs):
            captured_config.update(kwargs)
            return MagicMock()

        mock_ppo_agent_cls = MagicMock(side_effect=capture_ppo_init)

        with (
            patch(
                "keisei.core.neural_network.ActorCritic",
                mock_actor_critic_cls,
            ),
            patch(
                "keisei.core.ppo_agent.PPOAgent",
                mock_ppo_agent_cls,
            ),
        ):
            load_evaluation_agent(
                "/fake/checkpoint.pt",
                "cpu",
                mock_policy_mapper,
                input_channels=46,
                input_features="custom_features",
            )

        config = captured_config.get("config")
        assert config is not None
        assert config.training.input_features == "custom_features"

    @patch("keisei.utils.agent_loading.os.path.isfile", return_value=True)
    def test_default_input_features_is_core46(self, mock_isfile, mock_policy_mapper):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        mock_actor_critic_cls = MagicMock(return_value=mock_model)

        captured_config = {}

        def capture_ppo_init(**kwargs):
            captured_config.update(kwargs)
            return MagicMock()

        mock_ppo_agent_cls = MagicMock(side_effect=capture_ppo_init)

        with (
            patch(
                "keisei.core.neural_network.ActorCritic",
                mock_actor_critic_cls,
            ),
            patch(
                "keisei.core.ppo_agent.PPOAgent",
                mock_ppo_agent_cls,
            ),
        ):
            load_evaluation_agent(
                "/fake/checkpoint.pt",
                "cpu",
                mock_policy_mapper,
                input_channels=46,
            )

        config = captured_config.get("config")
        assert config is not None
        assert config.training.input_features == "core46"


# ---------------------------------------------------------------------------
# TestInitializeOpponent
# ---------------------------------------------------------------------------


class TestInitializeOpponent:
    def test_random_returns_simple_random_opponent(self, mock_policy_mapper):
        result = initialize_opponent("random", None, "cpu", mock_policy_mapper, 46)
        assert isinstance(result, SimpleRandomOpponent)

    def test_heuristic_returns_simple_heuristic_opponent(self, mock_policy_mapper):
        result = initialize_opponent("heuristic", None, "cpu", mock_policy_mapper, 46)
        assert isinstance(result, SimpleHeuristicOpponent)

    def test_ppo_without_path_raises_value_error(self, mock_policy_mapper):
        with pytest.raises(ValueError, match="Opponent path must be provided"):
            initialize_opponent("ppo", None, "cpu", mock_policy_mapper, 46)

    @patch("keisei.utils.agent_loading.load_evaluation_agent")
    def test_ppo_with_path_calls_load_evaluation_agent(
        self, mock_load, mock_policy_mapper
    ):
        mock_load.return_value = MagicMock()
        result = initialize_opponent(
            "ppo", "/fake/model.pt", "cpu", mock_policy_mapper, 46
        )
        mock_load.assert_called_once_with(
            "/fake/model.pt", "cpu", mock_policy_mapper, 46
        )
        assert result is mock_load.return_value

    def test_unknown_type_raises_value_error(self, mock_policy_mapper):
        with pytest.raises(ValueError, match="Unknown opponent type"):
            initialize_opponent("unknown", None, "cpu", mock_policy_mapper, 46)
