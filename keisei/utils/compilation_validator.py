"""
torch.compile validation framework for neural network optimization.

This module provides comprehensive validation and fallback mechanisms for
torch.compile integration, ensuring both performance and correctness.
"""

import warnings
import sys
from typing import Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass


import torch
import torch.nn as nn

from keisei.core.actor_critic_protocol import ActorCriticProtocol
from keisei.utils.performance_benchmarker import (
    PerformanceBenchmarker,
    ComparisonResult,
)


@dataclass
class CompilationResult:
    """Results from torch.compile validation process."""

    success: bool
    compiled_model: Optional[nn.Module]
    error_message: Optional[str]
    fallback_used: bool
    validation_passed: bool
    performance_improvement: Optional[float]
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        if self.success:
            perf_str = (
                f", {self.performance_improvement:.2f}x speedup"
                if self.performance_improvement
                else ""
            )
            return f"Compilation SUCCESS{perf_str}"
        else:
            fallback_str = " (using fallback)" if self.fallback_used else ""
            return f"Compilation FAILED: {self.error_message}{fallback_str}"


class CompilationValidator:
    """
    Comprehensive validation framework for torch.compile integration.

    Features:
    - Safe compilation with automatic fallback
    - Numerical equivalence validation
    - Performance regression detection
    - Detailed error reporting and recovery
    - Configuration-driven compilation parameters
    """

    def __init__(
        self,
        config_training,
        logger_func: Optional[Callable[[str], None]] = None,
        benchmarker: Optional[PerformanceBenchmarker] = None,
    ):
        """
        Initialize the compilation validator.

        Args:
            config_training: TrainingConfig with torch.compile settings
            logger_func: Optional logging function for status messages
            benchmarker: Optional performance benchmarker instance
        """
        self.config = config_training
        self.logger_func = logger_func or (lambda msg: None)
        self.benchmarker = benchmarker

        # Extract compilation settings
        self.enabled = getattr(config_training, "enable_torch_compile", True)
        self.mode = getattr(config_training, "torch_compile_mode", "default")
        self.dynamic = getattr(config_training, "torch_compile_dynamic", None)
        self.fullgraph = getattr(config_training, "torch_compile_fullgraph", False)
        self.backend = getattr(config_training, "torch_compile_backend", None)
        self.enable_fallback = getattr(
            config_training, "enable_compilation_fallback", True
        )
        self.validate_output = getattr(
            config_training, "validate_compiled_output", True
        )
        self.tolerance = getattr(
            config_training, "compilation_validation_tolerance", 1e-5
        )
        self.warmup_steps = getattr(config_training, "compilation_warmup_steps", 5)
        self.enable_benchmarking = getattr(
            config_training, "enable_compilation_benchmarking", True
        )

        # Compilation state tracking
        self.compilation_attempted = False
        self.last_result: Optional[CompilationResult] = None

    def compile_model(
        self,
        model: ActorCriticProtocol,
        sample_input: torch.Tensor,
        model_name: str = "model",
    ) -> CompilationResult:
        """
        Compile a model with comprehensive validation and fallback.

        Args:
            model: Model to compile (must implement ActorCriticProtocol)
            sample_input: Representative input tensor for validation
            model_name: Descriptive name for logging

        Returns:
            CompilationResult with compiled model and validation status
        """
        self.compilation_attempted = True

        if not self.enabled:
            self.logger_func("torch.compile is disabled in configuration")
            return CompilationResult(
                success=True,
                compiled_model=model,
                error_message=None,
                fallback_used=False,
                validation_passed=True,
                performance_improvement=None,
                metadata={"compilation_skipped": True},
            )

        if not self._check_torch_compile_availability():
            return self._create_fallback_result(
                model, "torch.compile not available in this PyTorch version"
            )

        self.logger_func(f"Compiling {model_name} with mode='{self.mode}'...")

        try:
            # Attempt compilation with configured parameters
            compiled_model = self._attempt_compilation(model)

            # Validate compiled model if requested
            if self.validate_output:
                validation_passed, validation_details = self._validate_compiled_model(
                    model, compiled_model, sample_input
                )
                if not validation_passed:
                    error_msg = f"Numerical validation failed: {validation_details}"
                    if self.enable_fallback:
                        self.logger_func(f"WARNING: {error_msg}, using fallback")
                        return self._create_fallback_result(model, error_msg)
                    else:
                        raise RuntimeError(error_msg)
            else:
                validation_passed = True
                validation_details = {}

            # Performance benchmarking if enabled
            performance_improvement = None
            if self.enable_benchmarking and self.benchmarker:
                performance_improvement = self._benchmark_compilation(
                    model, compiled_model, sample_input, model_name
                )

            # Successful compilation
            result = CompilationResult(
                success=True,
                compiled_model=compiled_model,
                error_message=None,
                fallback_used=False,
                validation_passed=validation_passed,
                performance_improvement=performance_improvement,
                metadata={
                    "mode": self.mode,
                    "dynamic": self.dynamic,
                    "fullgraph": self.fullgraph,
                    "backend": self.backend,
                    "validation_details": validation_details,
                },
            )

            self.last_result = result
            self.logger_func(f"Successfully compiled {model_name}: {result}")
            return result

        except Exception as e:
            error_msg = f"Compilation failed: {str(e)}"
            self.logger_func(f"ERROR: {error_msg}")

            if self.enable_fallback:
                self.logger_func("Using non-compiled model as fallback")
                return self._create_fallback_result(model, error_msg)
            else:
                raise RuntimeError(error_msg) from e

    def _check_torch_compile_availability(self) -> bool:
        """Check if torch.compile is available in current PyTorch version."""
        return hasattr(torch, "compile") and sys.version_info >= (3, 8)

    def _attempt_compilation(self, model: nn.Module) -> nn.Module:
        """Attempt to compile the model with configured parameters."""
        compile_kwargs = {"mode": self.mode, "fullgraph": self.fullgraph}

        # Add optional parameters if specified
        if self.dynamic is not None:
            compile_kwargs["dynamic"] = self.dynamic
        if self.backend is not None:
            compile_kwargs["backend"] = self.backend

        # Filter out None values and invalid combinations
        compile_kwargs = {k: v for k, v in compile_kwargs.items() if v is not None}

        self.logger_func(f"torch.compile parameters: {compile_kwargs}")
        return torch.compile(model, **compile_kwargs)

    def _validate_compiled_model(
        self,
        original_model: ActorCriticProtocol,
        compiled_model: nn.Module,
        sample_input: torch.Tensor,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate that compiled model produces equivalent outputs."""
        try:
            original_model.eval()
            compiled_model.eval()

            # Warmup compiled model
            with torch.no_grad():
                for _ in range(self.warmup_steps):
                    _ = compiled_model(sample_input)

            # Numerical equivalence validation
            with torch.no_grad():
                original_policy, original_value = original_model(sample_input)
                compiled_policy, compiled_value = compiled_model(sample_input)

                policy_diff = torch.max(
                    torch.abs(original_policy - compiled_policy)
                ).item()
                value_diff = torch.max(
                    torch.abs(original_value - compiled_value)
                ).item()
                max_diff = max(policy_diff, value_diff)

                validation_passed = max_diff <= self.tolerance

                details = {
                    "policy_max_diff": policy_diff,
                    "value_max_diff": value_diff,
                    "max_overall_diff": max_diff,
                    "tolerance": self.tolerance,
                    "passed": validation_passed,
                }

                if validation_passed:
                    self.logger_func(
                        f"Numerical validation PASSED: max_diff={max_diff:.2e}, "
                        f"tolerance={self.tolerance:.2e}"
                    )
                else:
                    self.logger_func(
                        f"Numerical validation FAILED: max_diff={max_diff:.2e}, "
                        f"tolerance={self.tolerance:.2e}"
                    )

                return validation_passed, details

        except Exception as e:
            self.logger_func(f"Validation error: {str(e)}")
            return False, {"error": str(e)}

    def _benchmark_compilation(
        self,
        original_model: ActorCriticProtocol,
        compiled_model: nn.Module,
        sample_input: torch.Tensor,
        model_name: str,
    ) -> Optional[float]:
        """Benchmark performance improvement from compilation."""
        try:
            comparison = self.benchmarker.compare_models(
                baseline_model=original_model,
                optimized_model=compiled_model,
                input_tensor=sample_input,
                baseline_name=f"{model_name}_original",
                optimized_name=f"{model_name}_compiled",
            )

            self.logger_func(f"Performance comparison: {comparison}")
            return comparison.speedup

        except Exception as e:
            self.logger_func(f"Benchmarking failed: {str(e)}")
            return None

    def _create_fallback_result(
        self, original_model: nn.Module, error_message: str
    ) -> CompilationResult:
        """Create a fallback result using the original model."""
        return CompilationResult(
            success=False,
            compiled_model=original_model,
            error_message=error_message,
            fallback_used=True,
            validation_passed=True,  # Original model is always valid
            performance_improvement=None,
            metadata={"fallback_reason": error_message},
        )

    def get_compilation_status(self) -> Dict[str, Any]:
        """Get current compilation status and configuration."""
        return {
            "enabled": self.enabled,
            "attempted": self.compilation_attempted,
            "last_result": self.last_result,
            "configuration": {
                "mode": self.mode,
                "dynamic": self.dynamic,
                "fullgraph": self.fullgraph,
                "backend": self.backend,
                "enable_fallback": self.enable_fallback,
                "validate_output": self.validate_output,
                "tolerance": self.tolerance,
                "warmup_steps": self.warmup_steps,
                "enable_benchmarking": self.enable_benchmarking,
            },
        }


def safe_compile_model(
    model: ActorCriticProtocol,
    sample_input: torch.Tensor,
    config_training,
    logger_func: Optional[Callable[[str], None]] = None,
    benchmarker: Optional[PerformanceBenchmarker] = None,
    model_name: str = "model",
) -> Tuple[nn.Module, CompilationResult]:
    """
    Convenience function for safe model compilation with validation.

    Args:
        model: Model to compile
        sample_input: Representative input for validation
        config_training: TrainingConfig with compilation settings
        logger_func: Optional logging function
        benchmarker: Optional performance benchmarker
        model_name: Descriptive name for logging

    Returns:
        Tuple of (compiled_model, compilation_result)
    """
    validator = CompilationValidator(
        config_training=config_training,
        logger_func=logger_func,
        benchmarker=benchmarker,
    )

    result = validator.compile_model(model, sample_input, model_name)
    return result.compiled_model, result
