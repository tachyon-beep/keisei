"""
This module re-exports the main classes from the refactored Shogi engine components
for backward compatibility.
"""

from .shogi_core_definitions import Color, MoveTuple, Piece, PieceType
from .shogi_game import ShogiGame

# These exports allow code that previously imported from shogi_engine.py
# to continue working without changes
__all__ = ["Color", "PieceType", "Piece", "ShogiGame", "MoveTuple"]
