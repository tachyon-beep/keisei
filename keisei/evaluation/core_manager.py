from __future__ import annotations

"""EvaluationManager for new evaluation system."""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

# Import strategies to ensure they register with the factory
from . import strategies  # pylint: disable=unused-import
from .core import (
    AgentInfo,
    EvaluationConfig,
    EvaluationContext,
    EvaluationResult,
    EvaluatorFactory,
    ModelWeightManager,
    OpponentInfo,
)
from .opponents import OpponentPool
from .performance_manager import EvaluationPerformanceManager


class EvaluationManager:
    """Manage evaluator creation and execution."""

    def __init__(
        self,
        config: EvaluationConfig,
        run_name: str,
        pool_size: int = 5,
        elo_registry_path: str | None = None,
    ) -> None:
        self.config = config
        self.run_name = run_name
        self.device = "cpu"
        self.policy_mapper = None
        self.model_dir: str | None = None
        self.wandb_active = False
        self.opponent_pool = OpponentPool(pool_size, elo_registry_path)

        # Initialize model weight manager for in-memory evaluation
        self.model_weight_manager = ModelWeightManager(
            device=getattr(config, "temp_agent_device", "cpu"),
            max_cache_size=getattr(config, "model_weight_cache_size", 5),
        )
        self.enable_in_memory_eval = getattr(
            config, "enable_in_memory_evaluation", True
        )

        # CRITICAL FIX: Initialize performance manager and integrate it into evaluation flow
        self.performance_manager = EvaluationPerformanceManager(
            max_concurrent=getattr(config, "max_concurrent_evaluations", 4),
            timeout_seconds=getattr(config, "evaluation_timeout_seconds", 300),
        )

        # Enable performance safeguards by default in production
        self.performance_safeguards_enabled = getattr(
            config, "enable_performance_safeguards", True
        )

        if not self.performance_safeguards_enabled:
            self.performance_manager.disable_enforcement()

    def setup(
        self,
        device: str,
        policy_mapper,
        model_dir: str,
        wandb_active: bool,
    ) -> None:
        """Configure runtime properties."""
        self.device = device
        self.policy_mapper = policy_mapper
        self.model_dir = model_dir
        self.wandb_active = wandb_active

    def evaluate_checkpoint(
        self, agent_checkpoint: str, _opponent_checkpoint: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate a saved agent checkpoint."""
        # Validate checkpoint file exists and is readable
        if not Path(agent_checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {agent_checkpoint}")

        # Try to validate the checkpoint file format
        try:
            import torch

            checkpoint_data = torch.load(
                agent_checkpoint, map_location="cpu", weights_only=True
            )
            # Basic validation - should be a dictionary with expected keys
            if not isinstance(checkpoint_data, dict):
                raise ValueError(
                    f"Checkpoint file is not a valid PyTorch checkpoint: {agent_checkpoint}"
                )
        except Exception as e:
            raise ValueError(
                f"Failed to load checkpoint file {agent_checkpoint}: {str(e)}"
            )

        agent_info = AgentInfo(name="current_agent", checkpoint_path=agent_checkpoint)
        context = EvaluationContext(
            session_id=str(uuid4()),
            timestamp=datetime.now(),
            agent_info=agent_info,
            configuration=self.config,
            environment_info={"device": self.device},
        )
        evaluator = EvaluatorFactory.create(self.config)
        # Set runtime context from training system
        evaluator.set_runtime_context(
            policy_mapper=self.policy_mapper,
            device=self.device,
            model_dir=self.model_dir,
            wandb_active=self.wandb_active,
        )
        # FIXED: Check if we're in an async context and use appropriate method
        try:
            # Check if there's a running event loop
            asyncio.get_running_loop()
            # If we get here, we're in an async context - this shouldn't happen for evaluate_checkpoint
            # which is designed for synchronous use. Return a sync wrapper.
            import sys

            if sys.version_info >= (3, 7):
                # For Python 3.7+, we can create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # CRITICAL FIX: Use performance manager for ALL evaluations
                    if self.performance_safeguards_enabled:
                        result = loop.run_until_complete(
                            self.performance_manager.run_evaluation_with_safeguards(
                                evaluator, agent_info, context
                            )
                        )
                    else:
                        result = loop.run_until_complete(
                            evaluator.evaluate(agent_info, context)
                        )
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
                return result
            else:
                # Fallback for older Python versions
                raise RuntimeError(
                    "Cannot run evaluation synchronously from within async context. "
                    "Use evaluate_checkpoint_async() instead."
                )
        except RuntimeError:
            # No running event loop, safe to use asyncio.run
            if self.performance_safeguards_enabled:
                return asyncio.run(
                    self.performance_manager.run_evaluation_with_safeguards(
                        evaluator, agent_info, context
                    )
                )
            else:
                return asyncio.run(evaluator.evaluate(agent_info, context))

    async def evaluate_checkpoint_async(
        self, agent_checkpoint: str, _opponent_checkpoint: Optional[str] = None
    ) -> EvaluationResult:
        """Async version of evaluate_checkpoint for use in async contexts."""
        # Validate checkpoint file exists and is readable
        if not Path(agent_checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {agent_checkpoint}")

        # Try to validate the checkpoint file format
        try:
            import torch

            checkpoint_data = torch.load(
                agent_checkpoint, map_location="cpu", weights_only=True
            )
            # Basic validation - should be a dictionary with expected keys
            if not isinstance(checkpoint_data, dict):
                raise ValueError(
                    f"Checkpoint file is not a valid PyTorch checkpoint: {agent_checkpoint}"
                )
        except Exception as e:
            raise ValueError(
                f"Failed to load checkpoint file {agent_checkpoint}: {str(e)}"
            )

        agent_info = AgentInfo(name="current_agent", checkpoint_path=agent_checkpoint)
        context = EvaluationContext(
            session_id=str(uuid4()),
            timestamp=datetime.now(),
            agent_info=agent_info,
            configuration=self.config,
            environment_info={"device": self.device},
        )
        evaluator = EvaluatorFactory.create(self.config)
        # Set runtime context from training system
        evaluator.set_runtime_context(
            policy_mapper=self.policy_mapper,
            device=self.device,
            model_dir=self.model_dir,
            wandb_active=self.wandb_active,
        )
        # CRITICAL FIX: Use performance manager for async evaluation
        if self.performance_safeguards_enabled:
            return await self.performance_manager.run_evaluation_with_safeguards(
                evaluator, agent_info, context
            )
        else:
            return await evaluator.evaluate(agent_info, context)

    def evaluate_current_agent(self, agent) -> EvaluationResult:
        """Evaluate an in-memory PPOAgent instance."""

        # Ensure the agent has the expected attributes
        if not hasattr(agent, "model") or agent.model is None:
            raise ValueError("Agent must have a 'model' attribute for evaluation")

        model = agent.model

        # Switch to eval mode for duration of evaluation
        if hasattr(model, "eval"):
            model.eval()

        agent_info = AgentInfo(
            name=getattr(agent, "name", "current_agent"),
            metadata={"agent_instance": agent},
        )
        context = EvaluationContext(
            session_id=str(uuid4()),
            timestamp=datetime.now(),
            agent_info=agent_info,
            configuration=self.config,
            environment_info={"device": self.device},
        )

        evaluator = EvaluatorFactory.create(self.config)
        # Set runtime context from training system
        evaluator.set_runtime_context(
            policy_mapper=self.policy_mapper,
            device=self.device,
            model_dir=self.model_dir,
            wandb_active=self.wandb_active,
        )
        # FIXED: Check if we're in an async context and use appropriate method
        try:
            # Check if there's a running event loop
            asyncio.get_running_loop()
            # If we get here, we're in an async context - this shouldn't happen for evaluate_current_agent
            # which is designed for synchronous use from callbacks. Return a sync wrapper.
            import sys

            if sys.version_info >= (3, 7):
                # For Python 3.7+, we can create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # CRITICAL FIX: Use performance manager for current agent evaluation
                    if self.performance_safeguards_enabled:
                        result = loop.run_until_complete(
                            self.performance_manager.run_evaluation_with_safeguards(
                                evaluator, agent_info, context
                            )
                        )
                    else:
                        result = loop.run_until_complete(
                            evaluator.evaluate(agent_info, context)
                        )
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
            else:
                # Fallback for older Python versions
                raise RuntimeError(
                    "Cannot run evaluation synchronously from within async context. "
                    "Use evaluate_current_agent_async() instead."
                )
        except RuntimeError:
            # No running event loop, safe to use asyncio.run
            if self.performance_safeguards_enabled:
                result = asyncio.run(
                    self.performance_manager.run_evaluation_with_safeguards(
                        evaluator, agent_info, context
                    )
                )
            else:
                result = asyncio.run(evaluator.evaluate(agent_info, context))

        # Restore training mode after evaluation
        if hasattr(model, "train"):
            model.train()

        return result

    async def evaluate_current_agent_async(self, agent) -> EvaluationResult:
        """Async version of evaluate_current_agent for use in async contexts."""

        # Ensure the agent has the expected attributes
        if not hasattr(agent, "model") or agent.model is None:
            raise ValueError("Agent must have a 'model' attribute for evaluation")

        model = agent.model

        # Switch to eval mode for duration of evaluation
        if hasattr(model, "eval"):
            model.eval()

        agent_info = AgentInfo(
            name=getattr(agent, "name", "current_agent"),
            metadata={"agent_instance": agent},
        )
        context = EvaluationContext(
            session_id=str(uuid4()),
            timestamp=datetime.now(),
            agent_info=agent_info,
            configuration=self.config,
            environment_info={"device": self.device},
        )

        evaluator = EvaluatorFactory.create(self.config)
        # Set runtime context from training system
        evaluator.set_runtime_context(
            policy_mapper=self.policy_mapper,
            device=self.device,
            model_dir=self.model_dir,
            wandb_active=self.wandb_active,
        )

        # CRITICAL FIX: Use performance manager for async current agent evaluation
        if self.performance_safeguards_enabled:
            result = await self.performance_manager.run_evaluation_with_safeguards(
                evaluator, agent_info, context
            )
        else:
            result = await evaluator.evaluate(agent_info, context)

        # Restore training mode after evaluation
        if hasattr(model, "train"):
            model.train()

        return result

    async def evaluate_current_agent_in_memory(
        self, agent, opponent_checkpoint: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate agent using in-memory weights without file I/O."""
        if not self.enable_in_memory_eval:
            return await self.evaluate_current_agent_async(
                agent
            )  # Fallback to async file-based

        try:
            # Extract current agent weights
            agent_weights = self.model_weight_manager.extract_agent_weights(agent)
            agent_info = AgentInfo(
                name=getattr(agent, "name", "current_agent"),
                model_type=type(agent).__name__,
                metadata={"agent_instance": agent, "in_memory_weights": True},
            )

            # Get opponent
            if opponent_checkpoint:
                opponent_path = Path(opponent_checkpoint)
                opponent_weights = self.model_weight_manager.cache_opponent_weights(
                    opponent_path.stem, opponent_path
                )
                opponent_info = OpponentInfo(
                    name=opponent_path.stem,
                    type="ppo",
                    checkpoint_path=str(opponent_path),
                )
            else:
                # Sample from pool
                opponent_path_obj = self.opponent_pool.sample()
                if not opponent_path_obj:
                    raise ValueError("No opponents available in pool")

                opponent_path = Path(opponent_path_obj)
                opponent_weights = self.model_weight_manager.cache_opponent_weights(
                    opponent_path.stem, opponent_path
                )
                opponent_info = OpponentInfo(
                    name=opponent_path.stem,
                    type="ppo",
                    checkpoint_path=str(opponent_path_obj),
                )

            # Create evaluation context
            context = EvaluationContext(
                session_id=f"inmem_{int(time.time())}",
                timestamp=datetime.now(),
                agent_info=agent_info,
                configuration=self.config,
                environment_info={
                    "device": self.device,
                    "evaluation_mode": "in_memory",
                    "opponent_info": opponent_info.__dict__,
                },
            )

            # Run in-memory evaluation
            return await self._run_in_memory_evaluation(
                agent_weights, opponent_weights, agent_info, opponent_info, context
            )

        except (ValueError, FileNotFoundError, RuntimeError) as e:
            # Fallback to file-based evaluation on error
            print(f"In-memory evaluation failed, falling back to async file-based: {e}")
            return await self.evaluate_current_agent_async(agent)

    async def _run_in_memory_evaluation(
        self, agent_weights, opponent_weights, agent_info, opponent_info, context
    ) -> EvaluationResult:
        """Run evaluation using in-memory weights."""
        # Get the evaluator
        evaluator = EvaluatorFactory.create(self.config)
        # Set runtime context from training system
        evaluator.set_runtime_context(
            policy_mapper=self.policy_mapper,
            device=self.device,
            model_dir=self.model_dir,
            wandb_active=self.wandb_active,
        )

        # Check if evaluator supports in-memory evaluation
        if hasattr(evaluator, "evaluate_in_memory"):
            # CRITICAL FIX: Apply performance safeguards to in-memory evaluation
            if self.performance_safeguards_enabled:
                # Create a wrapper for in-memory evaluation
                class InMemoryEvaluatorWrapper:
                    def __init__(self, eval_func, weights_args):
                        self.eval_func = eval_func
                        self.weights_args = weights_args

                    async def evaluate(self, agent_info, context):
                        return await self.eval_func(
                            agent_info, context, **self.weights_args
                        )

                wrapper = InMemoryEvaluatorWrapper(
                    evaluator.evaluate_in_memory,
                    {
                        "agent_weights": agent_weights,
                        "opponent_weights": opponent_weights,
                        "opponent_info": opponent_info,
                    },
                )

                return await self.performance_manager.run_evaluation_with_safeguards(
                    wrapper, agent_info, context
                )
            else:
                return await evaluator.evaluate_in_memory(
                    agent_info,
                    context,
                    agent_weights=agent_weights,
                    opponent_weights=opponent_weights,
                    opponent_info=opponent_info,
                )
        else:
            # Fallback to regular evaluation with performance safeguards
            if self.performance_safeguards_enabled:
                return await self.performance_manager.run_evaluation_with_safeguards(
                    evaluator, agent_info, context
                )
            else:
                return await evaluator.evaluate(agent_info, context)
