"""
Performance safeguards and SLA monitoring for evaluation system.
"""

import asyncio
import functools
import logging
import time

import psutil
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for evaluation."""

    evaluation_latency_ms: float
    memory_overhead_mb: float
    gpu_utilization_percent: Optional[float]
    cpu_utilization_percent: float
    start_time: datetime
    end_time: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "evaluation_latency_ms": self.evaluation_latency_ms,
            "memory_overhead_mb": self.memory_overhead_mb,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "cpu_utilization_percent": self.cpu_utilization_percent,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
        }


class ResourceMonitor:
    """Monitor system resource usage during evaluation."""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.initial_cpu_percent = None

    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        self.initial_memory = self.get_memory_usage()
        self.initial_cpu_percent = self.process.cpu_percent()

    def get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        return self.process.memory_info().rss

    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()

    def get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization if available."""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100  # Convert to percentage
        except ImportError:
            # GPUtil not available, try nvidia-ml-py
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return float(utilization.gpu)
            except (ImportError, Exception):
                pass
        return None

    def get_memory_overhead(self) -> float:
        """Get memory overhead since monitoring started."""
        if self.initial_memory is None:
            return 0.0
        current_memory = self.get_memory_usage()
        return (current_memory - self.initial_memory) / (1024 * 1024)  # Convert to MB


class EvaluationPerformanceSLA:
    """Performance service level agreement monitoring."""

    SLA_METRICS = {
        "evaluation_latency_ms": 5000,  # Max 5s per evaluation
        "memory_overhead_mb": 500,  # Max 500MB overhead
        "training_impact_percent": 5,  # Max 5% training slowdown
        "gpu_utilization_percent": 80,  # Max 80% GPU usage during eval
    }

    def validate_performance_sla(self, metrics: Dict[str, float]) -> bool:
        """Validate evaluation meets performance SLA."""
        violations = []

        for metric, threshold in self.SLA_METRICS.items():
            # CRITICAL FIX: Handle None values for GPU metrics on CPU-only deployments
            if (
                metric in metrics
                and metrics[metric] is not None
                and metrics[metric] > threshold
            ):
                violations.append(f"{metric}={metrics[metric]} > {threshold}")
                logger.error(f"SLA violation: {metric}={metrics[metric]} > {threshold}")

        if violations:
            logger.error(f"Performance SLA violations: {', '.join(violations)}")
            return False
        return True

    def log_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """Log performance metrics for monitoring."""
        metrics_dict = metrics.to_dict()
        logger.info(f"Evaluation performance metrics: {metrics_dict}")

        # Check SLA compliance
        if not self.validate_performance_sla(metrics_dict):
            logger.warning("Evaluation failed to meet performance SLA")
        else:
            logger.info("Evaluation met all performance SLA requirements")


class EvaluationPerformanceManager:
    """Performance safeguards for evaluation system."""

    def __init__(self, max_concurrent: int = 4, timeout_seconds: int = 300):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = timeout_seconds
        self._resource_monitor = None  # Lazy initialization
        self.sla_monitor = EvaluationPerformanceSLA()
        self.max_memory_mb = 500  # Maximum memory overhead allowed
        self.active = True  # Performance enforcement active

    @property
    def resource_monitor(self) -> ResourceMonitor:
        """Lazy initialization of ResourceMonitor."""
        if self._resource_monitor is None:
            self._resource_monitor = ResourceMonitor()
        return self._resource_monitor

    def enforce_resource_limits(self, metrics: PerformanceMetrics) -> bool:
        """Enforce resource limits during evaluation."""
        if not self.active:
            logger.warning(
                "Performance enforcement is DISABLED - resource limits bypassed!"
            )
            return True

        # Memory limit enforcement
        if metrics.memory_overhead_mb > self.max_memory_mb:
            logger.error(
                f"Memory limit exceeded: {metrics.memory_overhead_mb}MB > {self.max_memory_mb}MB"
            )
            raise EvaluationResourceError(
                f"Memory overhead {metrics.memory_overhead_mb}MB exceeds limit {self.max_memory_mb}MB"
            )

        # Timeout enforcement is handled by asyncio.wait_for in run_evaluation_with_safeguards

        # SLA validation (logs violations but doesn't fail evaluation)
        self.sla_monitor.validate_performance_sla(metrics.to_dict())

        return True

    async def run_evaluation_with_safeguards(self, evaluator, agent_info, context):
        """Run evaluation with performance controls."""
        if not self.active:
            logger.warning(
                "Performance safeguards are DISABLED - running evaluation without protection!"
            )
            return await evaluator.evaluate(agent_info, context)

        async with self.semaphore:  # Limit concurrency
            start_time = datetime.now()
            self.resource_monitor.start_monitoring()

            try:
                # Resource monitoring
                initial_memory = self.resource_monitor.get_memory_usage()
                initial_cpu = self.resource_monitor.get_cpu_percent()

                # CRITICAL FIX: Timeout control with proper resource enforcement
                result = await asyncio.wait_for(
                    evaluator.evaluate(agent_info, context), timeout=self.timeout
                )

                end_time = datetime.now()

                # Performance validation
                final_memory = self.resource_monitor.get_memory_usage()
                memory_overhead = self.resource_monitor.get_memory_overhead()
                cpu_percent = self.resource_monitor.get_cpu_percent()
                gpu_percent = self.resource_monitor.get_gpu_utilization()

                # Calculate latency
                latency_ms = (end_time - start_time).total_seconds() * 1000

                # Create performance metrics
                metrics = PerformanceMetrics(
                    evaluation_latency_ms=latency_ms,
                    memory_overhead_mb=memory_overhead,
                    gpu_utilization_percent=gpu_percent,
                    cpu_utilization_percent=cpu_percent,
                    start_time=start_time,
                    end_time=end_time,
                )

                # CRITICAL FIX: Actually enforce resource limits
                self.enforce_resource_limits(metrics)

                # Log and validate performance
                self.sla_monitor.log_performance_metrics(metrics)

                # Warning for high memory usage
                if memory_overhead > 500:  # 500MB
                    logger.warning(
                        f"Evaluation exceeded memory threshold: {memory_overhead:.1f}MB"
                    )

                return result

            except asyncio.TimeoutError:
                logger.error(f"Evaluation timeout after {self.timeout}s")
                raise EvaluationTimeoutError(
                    f"Evaluation timed out after {self.timeout} seconds"
                )
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                raise

    async def run_evaluation_with_monitoring(self, evaluation_func, *args, **kwargs):
        """Run evaluation function with performance monitoring."""
        return await self.run_evaluation_with_safeguards(
            MockEvaluator(evaluation_func), args, kwargs
        )

    def disable_enforcement(self):
        """Disable performance enforcement (for testing only)."""
        self.active = False
        logger.warning("Performance enforcement DISABLED - use only for testing!")

    def enable_enforcement(self):
        """Enable performance enforcement."""
        self.active = True
        logger.info("Performance enforcement ENABLED")


class MockEvaluator:
    """Mock evaluator wrapper for performance monitoring."""

    def __init__(self, evaluation_func):
        self.evaluation_func = evaluation_func

    async def evaluate(self, args, kwargs):
        """Execute the evaluation function."""
        if asyncio.iscoroutinefunction(self.evaluation_func):
            return await self.evaluation_func(*args, **kwargs)
        else:
            # Run sync function in executor (run_in_executor doesn't support **kwargs)
            loop = asyncio.get_event_loop()
            func = functools.partial(self.evaluation_func, *args, **kwargs)
            return await loop.run_in_executor(None, func)


class EvaluationTimeoutError(Exception):
    """Raised when evaluation exceeds timeout threshold."""

    pass


class EvaluationResourceError(Exception):
    """Raised when evaluation exceeds resource limits."""

    pass


