from unittest.mock import MagicMock

import pytest  # required for raises
import torch

from keisei.evaluation.core import (
    EvaluationConfig,
    EvaluationResult,
    create_evaluation_config,
)
from keisei.evaluation.core_manager import EvaluationManager


class DummyEvaluator:
    def __init__(self, config):
        self.config = config

    def set_runtime_context(
        self,
        policy_mapper=None,
        device: str = None,
        model_dir: str = None,
        wandb_active: bool = False,
        **kwargs
    ):
        """Mock implementation of set_runtime_context."""
        pass

    async def evaluate(self, agent_info, context):
        _ = agent_info  # use argument to avoid lint complaint
        return EvaluationResult(context=context, games=[], summary_stats=MagicMock())


def test_evaluate_checkpoint(monkeypatch, tmp_path):
    cfg = create_evaluation_config(strategy="single_opponent", num_games=1)
    manager = EvaluationManager(cfg, run_name="test")

    dummy_evaluator = DummyEvaluator(cfg)
    monkeypatch.setattr(
        "keisei.evaluation.core_manager.EvaluatorFactory.create",
        lambda cfg: dummy_evaluator,
    )

    # Create valid PyTorch checkpoint file
    ckpt_path = tmp_path / "agent.ckpt"
    torch.save({"model_state_dict": {}, "optimizer_state_dict": {}}, ckpt_path)

    result = manager.evaluate_checkpoint(str(ckpt_path))
    assert isinstance(result, EvaluationResult)
    assert result.context.agent_info.checkpoint_path == str(ckpt_path)


def test_evaluate_current_agent(monkeypatch):
    cfg = create_evaluation_config(strategy="single_opponent", num_games=1)
    manager = EvaluationManager(cfg, run_name="test")

    dummy_agent = MagicMock()
    dummy_agent.model = MagicMock()

    captured = {}

    class DummyEvaluator:
        def __init__(self, config):
            self.config = config

        def set_runtime_context(
            self,
            policy_mapper=None,
            device: str = None,
            model_dir: str = None,
            wandb_active: bool = False,
            **kwargs
        ):
            """Mock implementation of set_runtime_context."""
            pass

        async def evaluate(self, agent_info, context):
            _ = context  # use both args to avoid lint
            _ = agent_info
            captured["agent"] = agent_info.metadata.get("agent_instance")
            return EvaluationResult(
                context=context, games=[], summary_stats=MagicMock()
            )

    monkeypatch.setattr(
        "keisei.evaluation.core_manager.EvaluatorFactory.create",
        lambda cfg: DummyEvaluator(cfg),
    )

    result = manager.evaluate_current_agent(dummy_agent)
    assert isinstance(result, EvaluationResult)
    assert captured["agent"] is dummy_agent


def test_evaluate_current_agent_real_integration(tmp_path):
    """Test evaluation of a real agent with actual evaluator integration."""
    from keisei.evaluation.core import (
        EvaluationResult,
        create_evaluation_config,
    )
    from keisei.evaluation.core_manager import EvaluationManager
    from tests.evaluation.factories import EvaluationTestFactory

    cfg = create_evaluation_config(
        strategy="single_opponent", 
        num_games=1, 
        opponent_name="test_opponent"
    )
    manager = EvaluationManager(
        cfg, run_name="real_integration", pool_size=1, elo_registry_path=None
    )
    # Setup runtime
    test_agent = EvaluationTestFactory.create_test_agent()
    manager.setup(
        device="cpu",
        policy_mapper=test_agent.config.env,
        model_dir=str(tmp_path),
        wandb_active=False,
    )
    result = manager.evaluate_current_agent(test_agent)
    assert isinstance(result, EvaluationResult)
    # Ensure agent passed through context
    assert result.context.agent_info.metadata.get("agent_instance") is test_agent


def test_in_memory_evaluation_full_workflow(tmp_path):
    """Test complete in-memory evaluation workflow including opponent caching."""
    import asyncio
    from pathlib import Path

    import torch

    from keisei.evaluation.core import (
        EvaluationResult,
        create_evaluation_config,
    )
    from keisei.evaluation.core_manager import EvaluationManager
    from tests.evaluation.factories import EvaluationTestFactory

    cfg = create_evaluation_config(
        strategy="single_opponent",
        num_games=1,
        enable_in_memory_evaluation=True,
        opponent_name="test_opponent",
    )
    manager = EvaluationManager(
        cfg,
        run_name="in_memory_flow",
        pool_size=1,
        elo_registry_path=str(tmp_path / "elo.json"),
    )
    # Prepare opponent checkpoint
    opp_ckpt = tmp_path / "opponent.pt"
    checkpoint = {"model_state_dict": {"w": torch.randn(2, 2)}}
    torch.save(checkpoint, opp_ckpt)
    # Setup runtime
    test_agent = EvaluationTestFactory.create_test_agent()
    manager.setup(
        device="cpu",
        policy_mapper=test_agent.config.env,
        model_dir=str(tmp_path),
        wandb_active=False,
    )
    # Run in-memory evaluation
    result = asyncio.run(
        manager.evaluate_current_agent_in_memory(test_agent, str(opp_ckpt))
    )
    assert isinstance(result, EvaluationResult)
    # Verify in-memory flag
    assert result.context.agent_info.metadata.get("in_memory_weights") is True


def test_evaluation_error_propagation(monkeypatch):
    # Using pytest for exception assertions
    """Test errors from evaluator propagate and leave manager in usable state."""
    import asyncio

    from keisei.evaluation.core import (
        EvaluationConfig,
        EvaluationResult,
        create_evaluation_config,
    )
    from keisei.evaluation.core_manager import EvaluationManager, EvaluatorFactory
    from tests.evaluation.factories import EvaluationTestFactory

    # Create failing evaluator
    class FailingEvaluator:
        def __init__(self, config):
            # no-op initializer
            self.config = config

        def set_runtime_context(
            self,
            policy_mapper=None,
            device: str = None,
            model_dir: str = None,
            wandb_active: bool = False,
            **kwargs
        ):
            """Mock implementation of set_runtime_context."""
            pass

        async def evaluate(self, agent_info, context):
            raise RuntimeError("simulated failure")

    # Monkeypatch factory
    monkeypatch.setattr(EvaluatorFactory, "create", lambda cfg: FailingEvaluator(cfg))
    cfg = create_evaluation_config(strategy="single_opponent", num_games=1)
    manager = EvaluationManager(
        cfg, run_name="error_prop", pool_size=1, elo_registry_path=None
    )
    test_agent = EvaluationTestFactory.create_test_agent()
    manager.setup(
        device="cpu",
        policy_mapper=test_agent.config.env,
        model_dir=".",
        wandb_active=False,
    )

    # Expect exception
    with pytest.raises(RuntimeError, match="simulated failure"):
        manager.evaluate_current_agent(test_agent)
    # Manager should still be functional
    # Try a simple cache operation
    from keisei.evaluation.core.model_manager import ModelWeightManager

    manager.model_weight_manager.clear_cache()
    assert hasattr(manager, "model_weight_manager")


def test_sync_eval_raises_in_async_context(monkeypatch):
    """Calling sync evaluate_current_agent from async context raises a clear error.

    Regression test for P1-14: The original code had a broad `except RuntimeError`
    that caught both "no running event loop" and genuine evaluator failures,
    silently swallowing real errors when called from within a running event loop.
    The fix rejects async-context calls with a clear message instead.
    """
    import asyncio

    from keisei.evaluation.core import create_evaluation_config
    from keisei.evaluation.core_manager import EvaluationManager, EvaluatorFactory
    from tests.evaluation.factories import EvaluationTestFactory

    class DummyEvaluator:
        def __init__(self, config):
            self.config = config

        def set_runtime_context(self, **kwargs):
            pass

        async def evaluate(self, agent_info, context):
            return None  # Should never be reached

    monkeypatch.setattr(EvaluatorFactory, "create", lambda cfg: DummyEvaluator(cfg))
    cfg = create_evaluation_config(strategy="single_opponent", num_games=1)
    manager = EvaluationManager(
        cfg, run_name="async_guard", pool_size=1, elo_registry_path=None
    )
    test_agent = EvaluationTestFactory.create_test_agent()
    manager.setup(
        device="cpu",
        policy_mapper=test_agent.config.env,
        model_dir=".",
        wandb_active=False,
    )

    async def call_from_async():
        with pytest.raises(RuntimeError, match="cannot be called from an async context"):
            manager.evaluate_current_agent(test_agent)

    asyncio.run(call_from_async())