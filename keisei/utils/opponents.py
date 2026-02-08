"""
opponents.py: Contains simple opponent classes for evaluation and testing.
"""

import random
from typing import List

from keisei.shogi.shogi_core_definitions import MoveTuple, PieceType
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils.utils import BaseOpponent


class SimpleRandomOpponent(BaseOpponent):
    """An opponent that selects a random legal move."""

    def __init__(self, name: str = "SimpleRandomOpponent"):
        super().__init__(name)

    def select_move(self, game_instance: ShogiGame) -> MoveTuple:
        legal_moves = game_instance.get_legal_moves()
        if not legal_moves:
            raise ValueError(
                "No legal moves available for SimpleRandomOpponent, game should be over."
            )
        return random.choice(legal_moves)  # nosec B311


class SimpleHeuristicOpponent(BaseOpponent):
    """An opponent that uses simple heuristics to select a move."""

    def __init__(self, name: str = "SimpleHeuristicOpponent"):
        super().__init__(name)

    def select_move(self, game_instance: ShogiGame) -> MoveTuple:
        legal_moves = game_instance.get_legal_moves()
        if not legal_moves:
            raise ValueError(
                "No legal moves available for SimpleHeuristicOpponent, game should be over."
            )
        capturing_moves: List[MoveTuple] = []
        non_promoting_pawn_moves: List[MoveTuple] = []
        other_moves: List[MoveTuple] = []
        for move_tuple in legal_moves:
            is_capture = False
            is_pawn_move_no_promo = False
            if (
                isinstance(move_tuple[0], int)
                and isinstance(move_tuple[1], int)
                and isinstance(move_tuple[2], int)
                and isinstance(move_tuple[3], int)
                and isinstance(move_tuple[4], bool)
            ):
                from_r = move_tuple[0]
                from_c = move_tuple[1]
                to_r = move_tuple[2]
                to_c = move_tuple[3]
                promote = move_tuple[4]
                destination_piece = game_instance.board[to_r][to_c]
                if (
                    destination_piece is not None
                    and destination_piece.color != game_instance.current_player
                ):
                    is_capture = True
                if not is_capture:
                    source_piece = game_instance.board[from_r][from_c]
                    if (
                        source_piece
                        and source_piece.type == PieceType.PAWN
                        and not promote
                    ):
                        is_pawn_move_no_promo = True
            if is_capture:
                capturing_moves.append(move_tuple)
            elif is_pawn_move_no_promo:
                non_promoting_pawn_moves.append(move_tuple)
            else:
                other_moves.append(move_tuple)
        if capturing_moves:
            return random.choice(capturing_moves)  # nosec B311
        if non_promoting_pawn_moves:
            return random.choice(non_promoting_pawn_moves)  # nosec B311
        if other_moves:
            return random.choice(other_moves)  # nosec B311
        return random.choice(legal_moves)  # nosec B311


__all__ = [
    "SimpleRandomOpponent",
    "SimpleHeuristicOpponent",
]
