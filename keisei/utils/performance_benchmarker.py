"""
Performance benchmarking framework for neural network optimizations.

This module provides systematic performance measurement infrastructure for:
- torch.compile optimization validation
- Model inference speed benchmarking
- Memory usage tracking
- Performance regression detection
"""

import time
import gc
import statistics
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import warnings

import torch
import torch.nn as nn

from keisei.core.actor_critic_protocol import ActorCriticProtocol


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark run."""

    name: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    memory_peak_mb: float
    memory_allocated_mb: float
    num_iterations: int
    device: str
    model_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.mean_time_ms:.2f}±{self.std_time_ms:.2f}ms "
            f"(mem: {self.memory_peak_mb:.1f}MB, n={self.num_iterations})"
        )

    def speedup_vs(self, baseline: "BenchmarkResult") -> float:
        """Calculate speedup ratio vs baseline result."""
        if baseline.mean_time_ms == 0:
            return float("inf")
        return baseline.mean_time_ms / self.mean_time_ms

    def memory_change_vs(self, baseline: "BenchmarkResult") -> float:
        """Calculate memory usage change vs baseline (MB)."""
        return self.memory_peak_mb - baseline.memory_peak_mb


@dataclass
class ComparisonResult:
    """Results from comparing two benchmark results."""

    baseline: BenchmarkResult
    optimized: BenchmarkResult
    speedup: float
    memory_change_mb: float
    is_improvement: bool

    def __str__(self) -> str:
        direction = "↑" if self.speedup > 1.0 else "↓"
        mem_direction = "↑" if self.memory_change_mb > 0 else "↓"
        return (
            f"Speedup: {self.speedup:.2f}x {direction} | "
            f"Memory: {self.memory_change_mb:+.1f}MB {mem_direction} | "
            f"Status: {'IMPROVED' if self.is_improvement else 'REGRESSED'}"
        )


class PerformanceBenchmarker:
    """
    High-precision performance benchmarking for neural network models.

    Features:
    - Multiple warmup iterations for stable measurements
    - Memory usage tracking (peak and allocated)
    - Statistical analysis of timing variations
    - torch.compile optimization validation
    - Automatic outlier detection and removal
    """

    def __init__(
        self,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        outlier_threshold: float = 2.0,
        enable_profiling: bool = False,
        logger_func: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the performance benchmarker.

        Args:
            warmup_iterations: Number of warmup runs before measurement
            benchmark_iterations: Number of timed iterations for statistics
            outlier_threshold: Standard deviations threshold for outlier removal
            enable_profiling: Enable detailed PyTorch profiling
            logger_func: Optional logging function for status messages
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.outlier_threshold = outlier_threshold
        self.enable_profiling = enable_profiling
        self.logger_func = logger_func or (lambda msg: None)

        # Results storage
        self.results: Dict[str, BenchmarkResult] = {}

    def benchmark_model(
        self,
        model: ActorCriticProtocol,
        input_tensor: torch.Tensor,
        name: str,
        model_type: str = "unknown",
        enable_grad: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """
        Benchmark a model's forward pass performance.

        Args:
            model: Model to benchmark (must implement ActorCriticProtocol)
            input_tensor: Input tensor for forward pass
            name: Descriptive name for this benchmark
            model_type: Type description for the model
            enable_grad: Whether to enable gradients during forward pass
            metadata: Additional metadata to store with results

        Returns:
            BenchmarkResult containing timing and memory statistics
        """
        self.logger_func(f"Benchmarking {name}...")

        device = str(input_tensor.device)
        metadata = metadata or {}

        # Ensure model is in eval mode for consistent results
        was_training = model.training
        model.eval()

        # Warmup phase
        self._warmup_model(model, input_tensor, enable_grad)

        # Benchmark phase
        timings = []
        memory_peaks = []
        memory_allocated = []

        for i in range(self.benchmark_iterations):
            torch.cuda.empty_cache() if device.startswith("cuda") else None
            gc.collect()

            # Memory tracking setup
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
                memory_before = torch.cuda.memory_allocated() / 1024**2
            else:
                memory_before = 0

            # Timing measurement
            start_time = time.perf_counter()

            with torch.set_grad_enabled(enable_grad):
                with self._synchronization_context(device):
                    _ = model(input_tensor)

            end_time = time.perf_counter()

            # Record measurements
            timing_ms = (end_time - start_time) * 1000
            timings.append(timing_ms)

            if device.startswith("cuda"):
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                current_memory = torch.cuda.memory_allocated() / 1024**2
                memory_peaks.append(peak_memory)
                memory_allocated.append(current_memory)
            else:
                memory_peaks.append(0)
                memory_allocated.append(0)

        # Statistical analysis with outlier removal
        filtered_timings = self._remove_outliers(timings)

        if len(filtered_timings) < self.benchmark_iterations * 0.8:
            warnings.warn(
                f"High number of outliers detected in {name} benchmark "
                f"({len(timings) - len(filtered_timings)}/{len(timings)} removed)"
            )

        result = BenchmarkResult(
            name=name,
            mean_time_ms=statistics.mean(filtered_timings),
            std_time_ms=(
                statistics.stdev(filtered_timings) if len(filtered_timings) > 1 else 0.0
            ),
            min_time_ms=min(filtered_timings),
            max_time_ms=max(filtered_timings),
            memory_peak_mb=max(memory_peaks) if memory_peaks else 0.0,
            memory_allocated_mb=(
                statistics.mean(memory_allocated) if memory_allocated else 0.0
            ),
            num_iterations=len(filtered_timings),
            device=device,
            model_type=model_type,
            metadata=metadata,
        )

        # Restore original training mode
        if was_training:
            model.train()

        self.results[name] = result
        self.logger_func(f"Completed {name}: {result}")
        return result

    def compare_models(
        self,
        baseline_model: ActorCriticProtocol,
        optimized_model: ActorCriticProtocol,
        input_tensor: torch.Tensor,
        baseline_name: str = "baseline",
        optimized_name: str = "optimized",
        speedup_threshold: float = 1.05,
    ) -> ComparisonResult:
        """
        Compare performance between baseline and optimized models.

        Args:
            baseline_model: Reference model for comparison
            optimized_model: Optimized model to test
            input_tensor: Input tensor for both models
            baseline_name: Name for baseline benchmark
            optimized_name: Name for optimized benchmark
            speedup_threshold: Minimum speedup to consider an improvement

        Returns:
            ComparisonResult with speedup and memory analysis
        """
        self.logger_func(f"Comparing {baseline_name} vs {optimized_name}...")

        # Benchmark both models
        baseline_result = self.benchmark_model(
            baseline_model, input_tensor, baseline_name
        )
        optimized_result = self.benchmark_model(
            optimized_model, input_tensor, optimized_name
        )

        # Calculate comparison metrics
        speedup = optimized_result.speedup_vs(baseline_result)
        memory_change = optimized_result.memory_change_vs(baseline_result)
        is_improvement = speedup >= speedup_threshold

        comparison = ComparisonResult(
            baseline=baseline_result,
            optimized=optimized_result,
            speedup=speedup,
            memory_change_mb=memory_change,
            is_improvement=is_improvement,
        )

        self.logger_func(f"Comparison result: {comparison}")
        return comparison

    def validate_numerical_equivalence(
        self,
        baseline_model: ActorCriticProtocol,
        optimized_model: ActorCriticProtocol,
        input_tensor: torch.Tensor,
        tolerance: float = 1e-5,
        num_samples: int = 10,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Validate that optimized model produces numerically equivalent outputs.

        Args:
            baseline_model: Reference model
            optimized_model: Model to validate against reference
            input_tensor: Input tensor for validation
            tolerance: Maximum allowed absolute difference
            num_samples: Number of validation samples to test

        Returns:
            Tuple of (is_equivalent, max_difference, detailed_differences)
        """
        self.logger_func("Validating numerical equivalence...")

        baseline_model.eval()
        optimized_model.eval()

        max_policy_diff = 0.0
        max_value_diff = 0.0
        differences = []

        with torch.no_grad():
            for i in range(num_samples):
                # Generate slightly varied inputs for robustness
                noise_factor = 0.01 * torch.randn_like(input_tensor)
                test_input = input_tensor + noise_factor

                # Get outputs from both models
                baseline_policy, baseline_value = baseline_model(test_input)
                optimized_policy, optimized_value = optimized_model(test_input)

                # Calculate differences
                policy_diff = torch.max(
                    torch.abs(baseline_policy - optimized_policy)
                ).item()
                value_diff = torch.max(
                    torch.abs(baseline_value - optimized_value)
                ).item()

                max_policy_diff = max(max_policy_diff, policy_diff)
                max_value_diff = max(max_value_diff, value_diff)
                differences.append(
                    {"sample": i, "policy_diff": policy_diff, "value_diff": value_diff}
                )

        max_difference = max(max_policy_diff, max_value_diff)
        is_equivalent = max_difference <= tolerance

        detailed_differences = {
            "max_policy_diff": max_policy_diff,
            "max_value_diff": max_value_diff,
            "max_overall_diff": max_difference,
            "tolerance": tolerance,
            "num_samples": num_samples,
        }

        status = "PASSED" if is_equivalent else "FAILED"
        self.logger_func(
            f"Numerical validation {status}: max_diff={max_difference:.2e}, "
            f"tolerance={tolerance:.2e}"
        )

        return is_equivalent, max_difference, detailed_differences

    def _warmup_model(
        self, model: ActorCriticProtocol, input_tensor: torch.Tensor, enable_grad: bool
    ) -> None:
        """Perform warmup iterations to stabilize model performance."""
        with torch.set_grad_enabled(enable_grad):
            for _ in range(self.warmup_iterations):
                with self._synchronization_context(str(input_tensor.device)):
                    _ = model(input_tensor)

    @contextmanager
    def _synchronization_context(self, device: str):
        """Context manager for proper device synchronization during timing."""
        try:
            yield
        finally:
            if device.startswith("cuda"):
                torch.cuda.synchronize()

    def _remove_outliers(self, values: List[float]) -> List[float]:
        """Remove statistical outliers from timing measurements."""
        if len(values) < 3:
            return values

        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        threshold = self.outlier_threshold * std_val

        filtered = [v for v in values if abs(v - mean_val) <= threshold]

        return filtered if filtered else values  # Return original if all filtered

    def get_result(self, name: str) -> Optional[BenchmarkResult]:
        """Get a stored benchmark result by name."""
        return self.results.get(name)

    def get_all_results(self) -> Dict[str, BenchmarkResult]:
        """Get all stored benchmark results."""
        return self.results.copy()

    def clear_results(self) -> None:
        """Clear all stored benchmark results."""
        self.results.clear()

    def export_results(self) -> Dict[str, Any]:
        """Export all results to a serializable format."""
        return {
            name: {
                "name": result.name,
                "mean_time_ms": result.mean_time_ms,
                "std_time_ms": result.std_time_ms,
                "min_time_ms": result.min_time_ms,
                "max_time_ms": result.max_time_ms,
                "memory_peak_mb": result.memory_peak_mb,
                "memory_allocated_mb": result.memory_allocated_mb,
                "num_iterations": result.num_iterations,
                "device": result.device,
                "model_type": result.model_type,
                "metadata": result.metadata,
            }
            for name, result in self.results.items()
        }


def create_benchmarker(
    config_training, logger_func: Optional[Callable[[str], None]] = None
) -> PerformanceBenchmarker:
    """
    Factory function to create a PerformanceBenchmarker from training config.

    Args:
        config_training: TrainingConfig instance with benchmarking settings
        logger_func: Optional logging function

    Returns:
        Configured PerformanceBenchmarker instance
    """
    return PerformanceBenchmarker(
        warmup_iterations=getattr(config_training, "compilation_warmup_steps", 5),
        benchmark_iterations=100,  # Fixed for consistent measurement
        outlier_threshold=2.0,
        enable_profiling=getattr(
            config_training, "enable_compilation_benchmarking", True
        ),
        logger_func=logger_func,
    )
