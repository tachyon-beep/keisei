"""
DRL Shogi Client - A reinforcement learning agent for Japanese chess (Shogi).

This package contains:
- Shogi game engine and rules
- Reinforcement learning agents
- Neural network models
- Training utilities
"""

from .evaluation.core_manager import EvaluationManager

# Re-export the main components for easy access
from .shogi.shogi_core_definitions import Color, MoveTuple, Piece, PieceType
from .shogi.shogi_game import ShogiGame

__all__ = [
    # Shogi core types
    "Color",
    "PieceType",
    "Piece",
    "MoveTuple",
    "ShogiGame",
    "EvaluationManager",
    # Let the other modules be imported explicitly
]
