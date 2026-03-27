# keisei/evaluation/__init__.py
"""Public interface for the evaluation package."""

from .core_manager import EvaluationManager
from .scheduler import ContinuousMatchScheduler, MatchResult, SchedulerConfig

__all__ = ["EvaluationManager", "ContinuousMatchScheduler", "MatchResult", "SchedulerConfig"]

# Enhanced features (optional)
try:
    from .enhanced_manager import EnhancedEvaluationManager

    __all__.append("EnhancedEvaluationManager")
except ImportError:
    pass
