"""Evaluation test tier: real agents playing real games."""

import pytest
import torch

from keisei.config_schema import EvaluationConfig
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.evaluation.core.evaluation_context import AgentInfo, OpponentInfo


@pytest.fixture
def small_cnn_model(session_policy_mapper):
    """Small CNN ActorCritic model for fast evaluation tests."""
    return ActorCritic(
        input_channels=46,
        num_actions_total=session_policy_mapper.get_total_actions(),
    )


@pytest.fixture
def random_agent(small_cnn_model, app_config):
    """A real PPOAgent with random (untrained) weights on CPU."""
    return PPOAgent(
        model=small_cnn_model,
        config=app_config,
        device=torch.device("cpu"),
        name="RandomWeightAgent",
    )


@pytest.fixture
def second_random_agent(session_policy_mapper, app_config):
    """A second independent PPOAgent with random weights for multi-agent tests."""
    model = ActorCritic(
        input_channels=46,
        num_actions_total=session_policy_mapper.get_total_actions(),
    )
    return PPOAgent(
        model=model,
        config=app_config,
        device=torch.device("cpu"),
        name="SecondRandomAgent",
    )


@pytest.fixture
def eval_config_single(tmp_path):
    """EvaluationConfig for single-opponent evaluation with fast settings."""
    config = EvaluationConfig(
        num_games=2,
        max_moves_per_game=50,
        strategy="single_opponent",
        save_path=str(tmp_path / "eval_output"),
        log_level="DEBUG",
        wandb_log_eval=False,
        random_seed=42,
    )
    config.configure_for_single_opponent(
        opponent_name="random",
        play_as_both_colors=True,
        color_balance_tolerance=0.1,
    )
    return config


@pytest.fixture
def eval_config_tournament(tmp_path):
    """EvaluationConfig for tournament evaluation with fast settings."""
    config = EvaluationConfig(
        num_games=4,
        max_moves_per_game=50,
        strategy="tournament",
        save_path=str(tmp_path / "eval_output"),
        log_level="DEBUG",
        wandb_log_eval=False,
        random_seed=42,
    )
    config.configure_for_tournament(
        opponent_pool_config=[
            {"name": "random_opp_1", "type": "random", "metadata": {}},
            {"name": "random_opp_2", "type": "random", "metadata": {}},
        ],
        num_games_per_opponent=2,
    )
    return config


@pytest.fixture
def eval_config_ladder(tmp_path):
    """EvaluationConfig for ladder evaluation with fast settings."""
    config = EvaluationConfig(
        num_games=4,
        max_moves_per_game=50,
        strategy="ladder",
        save_path=str(tmp_path / "eval_output"),
        log_level="DEBUG",
        wandb_log_eval=False,
        random_seed=42,
    )
    config.configure_for_ladder(
        opponent_pool_config=[
            {
                "name": "ladder_random_1",
                "type": "random",
                "initial_rating": 1400,
            },
            {
                "name": "ladder_random_2",
                "type": "random",
                "initial_rating": 1500,
            },
        ],
        num_games_per_match=2,
        num_opponents_per_evaluation=2,
        num_opponents_to_select=5,
    )
    return config


@pytest.fixture
def agent_info(random_agent):
    """AgentInfo pointing to a real in-memory PPOAgent (no checkpoint file needed)."""
    return AgentInfo(
        name="test_agent",
        checkpoint_path=None,
        metadata={"agent_instance": random_agent},
    )


@pytest.fixture
def opponent_info_random():
    """OpponentInfo for a random opponent (no checkpoint needed)."""
    return OpponentInfo(
        name="random",
        type="random",
        checkpoint_path=None,
        metadata={},
    )
