"""
Shogi Python Reference Engine (archived).

Original Python Shogi game engine preserved as a reference and test oracle
during migration to the Rust engine (shogi-engine/).

This package contains all the components related to the Shogi game:
- Core definitions: piece types, colors, move tuples
- Game logic and rules
- Move execution
- Game input/output utilities
"""

# Export the main components for easy access
from .shogi_core_definitions import Color, MoveTuple, Piece, PieceType
from .shogi_game import ShogiGame

__all__ = [
    # Core types
    "Color",
    "PieceType",
    "Piece",
    "MoveTuple",
    # Game class
    "ShogiGame",
]
