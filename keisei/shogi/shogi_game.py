"""
shogi_game.py: Main ShogiGame class for DRL Shogi Client.
Orchestrates game state and delegates complex logic to helper modules.
"""

# pylint: disable=too-many-lines

import copy  # Added for __deepcopy__
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Create logger for this module
logger = logging.getLogger(__name__)

# Import helper modules
from . import shogi_game_io, shogi_move_execution, shogi_rules_logic

# Import types and fundamental classes from shogi_core_definitions
from .shogi_core_definitions import PIECE_TYPE_TO_HAND_TYPE  # Consolidated
from .shogi_core_definitions import MoveApplicationResult  # Added
from .shogi_core_definitions import TerminationReason  # Added
from .shogi_core_definitions import (
    Color,
    MoveTuple,
    Piece,
    PieceType,
    get_unpromoted_types,
)


class ShogiGame:
    """
    Represents the Shogi game state, board, and operations.
    Delegates complex rule logic, I/O, and move execution to helper modules.
    """

    def __init__(
        self, max_moves_per_game: int = 500
    ) -> None:  # Added default for max_moves_per_game
        self.board: List[List[Optional[Piece]]]
        self.hands: Dict[int, Dict[PieceType, int]]
        self.current_player: Color = Color.BLACK
        self.move_count: int = 0
        self.game_over: bool = False
        self.winner: Optional[Color] = None
        self.termination_reason: Optional[str] = None
        self.move_history: List[Dict[str, Any]] = []
        self._max_moves_this_game = max_moves_per_game
        self._initial_board_setup_done = False
        self._seed_value: Optional[Any] = None  # Added for seeding
        self.reset()

    def seed(self, seed_value: Optional[Any] = None) -> "ShogiGame":
        """
        Seeds the game's random number generators if applicable.
        Currently, this method primarily stores the seed value and logs it,
        as Shogi itself is deterministic once moves are chosen.
        This can be extended if stochastic elements are introduced.

        Args:
            seed_value: The value to seed with.

        Returns:
            self: The game instance for chaining.
        """
        self._seed_value = seed_value
        logger.debug("ShogiGame instance seeded with value: %s", seed_value)
        # If any RNGs were used directly by ShogiGame (e.g., for tie-breaking, though not typical),
        # they would be seeded here. For now, it's mainly for external components or future use.
        return self

    @property
    def max_moves_per_game(self) -> int:
        return self._max_moves_this_game

    def _setup_initial_board(self):
        """Sets up the board to the standard initial Shogi position."""
        self.board = [[None for _ in range(9)] for _ in range(9)]

        for i in range(9):
            self.board[2][i] = Piece(PieceType.PAWN, Color.WHITE)
            self.board[6][i] = Piece(PieceType.PAWN, Color.BLACK)

        self.board[0][0] = Piece(PieceType.LANCE, Color.WHITE)
        self.board[0][1] = Piece(PieceType.KNIGHT, Color.WHITE)
        self.board[0][2] = Piece(PieceType.SILVER, Color.WHITE)
        self.board[0][3] = Piece(PieceType.GOLD, Color.WHITE)
        self.board[0][4] = Piece(PieceType.KING, Color.WHITE)
        self.board[0][5] = Piece(PieceType.GOLD, Color.WHITE)
        self.board[0][6] = Piece(PieceType.SILVER, Color.WHITE)
        self.board[0][7] = Piece(PieceType.KNIGHT, Color.WHITE)
        self.board[0][8] = Piece(PieceType.LANCE, Color.WHITE)

        self.board[1][1] = Piece(PieceType.ROOK, Color.WHITE)
        self.board[1][7] = Piece(PieceType.BISHOP, Color.WHITE)

        self.board[8][0] = Piece(PieceType.LANCE, Color.BLACK)
        self.board[8][1] = Piece(PieceType.KNIGHT, Color.BLACK)
        self.board[8][2] = Piece(PieceType.SILVER, Color.BLACK)
        self.board[8][3] = Piece(PieceType.GOLD, Color.BLACK)
        self.board[8][4] = Piece(PieceType.KING, Color.BLACK)
        self.board[8][5] = Piece(PieceType.GOLD, Color.BLACK)
        self.board[8][6] = Piece(PieceType.SILVER, Color.BLACK)
        self.board[8][7] = Piece(PieceType.KNIGHT, Color.BLACK)
        self.board[8][8] = Piece(PieceType.LANCE, Color.BLACK)

        self.board[7][1] = Piece(PieceType.BISHOP, Color.BLACK)
        self.board[7][7] = Piece(PieceType.ROOK, Color.BLACK)

    def reset(self) -> np.ndarray:  # MODIFIED: Return np.ndarray
        """Resets the game to the initial state and returns the observation."""  # MODIFIED: Docstring
        self._setup_initial_board()
        self.hands = {
            Color.BLACK.value: {ptype: 0 for ptype in get_unpromoted_types()},
            Color.WHITE.value: {ptype: 0 for ptype in get_unpromoted_types()},
        }
        self.current_player = Color.BLACK
        self.move_count = 0
        self.game_over = False
        self.winner = None
        self.termination_reason = None
        self.move_history = []

        self._initial_board_setup_done = True
        return self.get_observation()  # MODIFIED: Return observation

    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        """Returns the piece at the specified position, or None if empty or out of bounds."""
        if self.is_on_board(row, col):
            return self.board[row][col]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[Piece]) -> None:
        """Sets or removes a piece at the specified position on the board."""
        if not self.is_on_board(row, col):
            raise ValueError(
                f"set_piece: coordinates ({row}, {col}) are out of bounds for 9x9 board"
            )
        self.board[row][col] = piece

    def to_string(self) -> str:
        """Returns a string representation of the current board state."""
        return shogi_game_io.convert_game_to_text_representation(self)

    def is_on_board(self, row: int, col: int) -> bool:
        return 0 <= row < 9 and 0 <= col < 9

    def __deepcopy__(self, memo: Dict[int, Any]):  # Added type hint for memo
        if id(self) in memo:
            return memo[id(self)]

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        result.board = copy.deepcopy(self.board, memo)
        result.hands = copy.deepcopy(self.hands, memo)
        result.current_player = self.current_player
        result.move_count = self.move_count
        result.game_over = self.game_over
        result.winner = self.winner
        result.termination_reason = self.termination_reason
        result.move_history = copy.deepcopy(self.move_history, memo)
        result._max_moves_this_game = self._max_moves_this_game
        result._initial_board_setup_done = self._initial_board_setup_done

        return result

    def _is_sliding_piece_type(self, piece_type: PieceType) -> bool:
        return shogi_rules_logic.is_piece_type_sliding(piece_type)

    def get_individual_piece_moves(
        self, piece: Piece, r_from: int, c_from: int
    ) -> List[Tuple[int, int]]:  # Changed from list[tuple[int, int]]
        return shogi_rules_logic.generate_piece_potential_moves(
            self, piece, r_from, c_from
        )

    def get_observation(self) -> np.ndarray:
        """
        Generates the neural network observation for the current game state.

        The observation is a multi-channel NumPy array representing the board,
        hands, and other game metadata from the current player's perspective.
        For detailed structure, see `shogi_game_io.generate_neural_network_observation`.

        Returns:
            np.ndarray: The observation array.
        """
        return shogi_game_io.generate_neural_network_observation(self)

    def get_state(self) -> np.ndarray:
        """
        Alias for get_observation() for backward compatibility.

        Returns:
            np.ndarray: The observation array.
        """
        return self.get_observation()

    def is_nifu(self, color: Color, col: int) -> bool:
        return shogi_rules_logic.check_for_nifu(self, color, col)

    def is_uchi_fu_zume(self, drop_row: int, drop_col: int, color: Color) -> bool:
        return shogi_rules_logic.check_for_uchi_fu_zume(self, drop_row, drop_col, color)

    def _is_square_attacked(self, row: int, col: int, attacker_color: Color) -> bool:
        return shogi_rules_logic.check_if_square_is_attacked(
            self, row, col, attacker_color
        )

    def get_legal_moves(self) -> List[MoveTuple]:  # Changed from List["MoveTuple"]
        moves = shogi_rules_logic.generate_all_legal_moves(self)
        return moves

    def _king_in_check_after_move(self, player_color: Color) -> bool:
        return shogi_rules_logic.is_king_in_check_after_simulated_move(
            self, player_color
        )

    def find_king(self, color: Color) -> Optional[Tuple[int, int]]:
        """Finds the king of the specified color on the board."""
        # Delegate to the rules logic function
        return shogi_rules_logic.find_king(self, color)

    def get_king_legal_moves(self, color: Color) -> int:
        """Return the number of legal moves available to the king of ``color``."""
        original_player = self.current_player
        self.current_player = color
        try:
            king_pos = self.find_king(color)
            if not king_pos:
                return 0
            legal_moves = self.get_legal_moves()
            count = 0
            for move in legal_moves:
                if move[0] is None or move[1] is None:
                    continue
                piece = self.get_piece(move[0], move[1])
                if piece and piece.type == PieceType.KING:
                    count += 1
            return count
        finally:
            self.current_player = original_player

    def is_in_check(
        self, color: Color, debug_recursion: bool = False
    ) -> bool:  # Added debug_recursion
        """Checks if the specified player is in check."""
        # Delegate to the rules logic function
        return shogi_rules_logic.is_in_check(
            self, color, debug_recursion=debug_recursion
        )  # Pass debug flag

    # --- SFEN methods now delegate to shogi_game_io.py ---

    def sfen_encode_move(
        self, move_tuple: MoveTuple
    ) -> str:  # Changed from "MoveTuple"
        """
        Encodes a move in SFEN (Shogi Forsyth-Edwards Notation) format.
        Board move: (from_r, from_c, to_r, to_c, promote_bool) -> e.g., "7g7f" or "2b3a+"
        Drop move: (None, None, to_r, to_c, piece_type) -> e.g., "P*5e"
        """
        return shogi_game_io.encode_move_to_sfen_string(move_tuple)

    def to_sfen_string(self) -> str:
        """
        Serializes the current game state to an SFEN string.
        Format: <board> <turn> <hands> <move_number>
        Example: lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
        """
        return shogi_game_io.convert_game_to_sfen_string(self)

    def to_sfen(self) -> str:
        """Alias for to_sfen_string() for convenience."""
        return self.to_sfen_string()

    @classmethod
    def from_sfen(
        cls, sfen_str: str, max_moves_for_game_instance: int = 500
    ) -> "ShogiGame":
        """Loads a game state from an SFEN string."""
        # Parse SFEN string components
        board_sfen, turn_sfen, hands_sfen, move_number_sfen = (
            shogi_game_io.parse_sfen_string_components(sfen_str)
        )

        current_player_from_sfen = Color.BLACK if turn_sfen == "b" else Color.WHITE

        try:
            move_number = int(move_number_sfen)
            if move_number < 1:
                raise ValueError("SFEN move number must be positive")
        except ValueError as e:
            if "SFEN move number must be positive" in str(e):
                raise
            raise ValueError(
                f"Invalid move number in SFEN: '{move_number_sfen}'"
            ) from e

        game = cls(max_moves_per_game=max_moves_for_game_instance)
        game.current_player = current_player_from_sfen  # Set current player before potential termination check
        game.move_count = (
            move_number - 1
        )  # SFEN move number is 1-indexed for the *next* move
        # Our move_count is 0-indexed for moves *completed*.
        # So, if SFEN says move 1, 0 moves completed.
        # If SFEN says move 5, 4 moves completed.

        game.board = [[None for _ in range(9)] for _ in range(9)]
        game.hands = {
            Color.BLACK.value: {ptype: 0 for ptype in get_unpromoted_types()},
            Color.WHITE.value: {ptype: 0 for ptype in get_unpromoted_types()},
        }
        game.move_history = []  # SFEN does not contain history

        shogi_game_io.populate_board_from_sfen_segment(game.board, board_sfen)
        shogi_game_io.populate_hands_from_sfen_segment(game.hands, hands_sfen)

        game._initial_board_setup_done = True


        # Evaluate termination conditions for the loaded position.
        # The `player_who_just_moved` for from_sfen is tricky. If the game is over,
        # it's the opponent of `game.current_player` (if checkmate/stalemate).
        # However, `_check_and_update_termination_status` expects the player whose turn it *would be*.
        # For from_sfen, `game.current_player` is already set to the player whose turn it is.
        # If this player has no moves and is in check, then the *other* player delivered checkmate.
        # So, we need to determine who would have made the move that *led* to this state.
        # This is complex if the SFEN represents a mid-game state where the last move isn't known.
        # For now, if checkmate, winner is opponent of current_player. If stalemate, winner is None.
        # The `player_who_just_moved` argument to `_check_and_update_termination_status` is primarily
        # to assign the winner correctly in case of checkmate.
        # Let's pass current_player.opponent() as a placeholder, it will be used if checkmate occurs.
        game._check_and_update_termination_status(game.current_player.opponent())

        return game

    def _board_state_hash(self) -> tuple:
        """
        Returns a hashable representation of the current board state, hands, and player to move.
        Used for checking for repetition (sennichite).
        """
        board_tuple = tuple(
            tuple((p.type.value, p.color.value) if p else None for p in row)
            for row in self.board
        )
        hands_tuple = (
            tuple(
                sorted(
                    (pt.value, count)
                    for pt, count in self.hands[Color.BLACK.value].items()
                    if count > 0
                )
            ),
            tuple(
                sorted(
                    (pt.value, count)
                    for pt, count in self.hands[Color.WHITE.value].items()
                    if count > 0
                )
            ),
        )
        return (board_tuple, hands_tuple, self.current_player.value)

    def get_board_state_hash(self) -> tuple:
        """Public interface to access the board state hash."""
        return self._board_state_hash()

    def get_reward(self, perspective_player_color: Optional[Color] = None) -> float:
        """
        Calculates the reward for the current game state from a given player's perspective.

        Args:
            perspective_player_color: The player for whom the reward is calculated.
                                      This argument is mandatory.

        Returns:
            float: 1.0 for a win, -1.0 for a loss, 0.0 for a draw or ongoing game.

        Raises:
            ValueError: If perspective_player_color is None.
        """
        if perspective_player_color is None:
            raise ValueError("perspective_player_color must be provided to get_reward.")

        if not self.game_over:
            return 0.0

        # Determine the perspective for evaluation
        eval_perspective: Color = perspective_player_color

        if self.winner is None:  # Draw
            return 0.0
        elif self.winner == eval_perspective:  # Perspective player won
            return 1.0
        else:  # Perspective player lost (or game ended with a winner not being them)
            return -1.0

    def _check_and_update_termination_status(
        self, player_who_just_moved: Color
    ) -> None:
        """
        Checks for all game termination conditions (checkmate, stalemate, max_moves, sennichite)
        and updates game_over, winner, and termination_reason if the game has ended.
        This method should be called after the current player has been switched (so self.current_player
        is the player whose turn it *would be*).
        """
        if self.game_over:  # If already marked as over, do nothing further.
            return

        # 1. Check for Checkmate or Stalemate
        #    These depend on the player whose turn it is now (self.current_player)
        #    having no legal moves.
        current_player_legal_moves = shogi_rules_logic.generate_all_legal_moves(self)
        if not current_player_legal_moves:
            try:
                in_check = shogi_rules_logic.is_in_check(self, self.current_player)
            except ValueError:
                # King missing — position is invalid, treat as game over (stalemate)
                in_check = False
            if in_check:
                self.game_over = True
                self.winner = (
                    player_who_just_moved  # The player who made the last move wins
                )
                self.termination_reason = TerminationReason.CHECKMATE.value
            else:
                self.game_over = True
                self.winner = None  # Stalemate is a draw
                self.termination_reason = TerminationReason.STALEMATE.value  # Use enum
            return  # Game ended by checkmate or stalemate

        # 2. Check for Max Moves Exceeded
        #    This uses self.move_count, which is the count *after* the last move.
        if self.move_count >= self.max_moves_per_game:
            self.game_over = True
            self.winner = None  # Typically a draw, or specific rules might apply
            self.termination_reason = TerminationReason.MAX_MOVES_EXCEEDED.value
            return  # Game ended by max moves

        # 3. Check for Sennichite (Repetition)
        #    This should be checked after other terminating conditions.
        if shogi_rules_logic.check_for_sennichite(self):
            self.game_over = True
            self.winner = None  # Sennichite is a draw
            self.termination_reason = TerminationReason.REPETITION.value

        # Other termination conditions like resignation, illegal move, time forfeit
        # are typically handled at a higher level or by external game management.

    def is_sennichite(self) -> bool:
        """
        Checks if the current position has occurred four times, resulting in a draw.
        """
        return shogi_rules_logic.check_for_sennichite(self)

    def _validate_move_tuple_format(self, move_tuple: MoveTuple) -> None:
        """Validates the basic structure and types of a move_tuple."""
        if not (
            isinstance(move_tuple, tuple)
            and len(move_tuple) == 5
            and (
                (  # Board move
                    isinstance(move_tuple[0], int)
                    and isinstance(move_tuple[1], int)
                    and isinstance(move_tuple[2], int)
                    and isinstance(move_tuple[3], int)
                    and isinstance(move_tuple[4], bool)
                )
                or (  # Drop move
                    move_tuple[0] is None
                    and move_tuple[1] is None
                    and isinstance(move_tuple[2], int)
                    and isinstance(move_tuple[3], int)
                    and isinstance(move_tuple[4], PieceType)
                )
            )
        ):
            raise ValueError(f"Invalid move_tuple format: {move_tuple}")

    def _prepare_move_details_for_history(
        self,
        move_tuple: MoveTuple,
        player_who_made_the_move: Color,
        move_count_before_move: int,
    ) -> Dict[str, Any]:
        """Initializes the dictionary for storing move details in history."""
        return {
            "move": move_tuple,
            "is_drop": False,
            "captured": None,
            "was_promoted_in_move": False,
            "original_type_before_promotion": None,
            "dropped_piece_type": None,
            "original_color_of_moved_piece": None,
            "player_who_made_the_move": player_who_made_the_move,
            "move_count_before_move": move_count_before_move,
            "original_board_state": None,
            "original_hands_state": None,
            "state_hash": None,  # Ensure state_hash is initialized
        }

    def _populate_drop_move_details(
        self, move_details: Dict[str, Any], promote_or_drop_info: PieceType
    ) -> None:
        """Populates details specific to a drop move."""
        move_details["is_drop"] = True
        if not isinstance(promote_or_drop_info, PieceType):
            raise ValueError(
                "Internal error: Drop move info not PieceType despite validation."
            )
        move_details["dropped_piece_type"] = promote_or_drop_info

    def _validate_and_populate_board_move_details(
        self,
        move_details: Dict[str, Any],
        from_r: int,
        from_c: int,
        to_r: int,
        to_c: int,
        player_who_made_the_move: Color,
    ) -> None:
        """Validates a board move and populates its specific details."""
        piece_to_move = self.get_piece(from_r, from_c)
        if piece_to_move is None:
            raise ValueError(f"Invalid move: No piece at source ({from_r},{from_c})")
        if piece_to_move.color != player_who_made_the_move:
            raise ValueError(
                f"Invalid move: Piece at ({from_r},{from_c}) does not belong to current player."
            )

        potential_squares = shogi_rules_logic.generate_piece_potential_moves(
            self, piece_to_move, from_r, from_c
        )
        if (to_r, to_c) not in potential_squares:
            raise ValueError(
                f"Illegal movement pattern: {piece_to_move.type.name} at "
                f"({from_r},{from_c}) cannot move to ({to_r},{to_c}). "
                f"Potential squares: {potential_squares}"
            )
        move_details["original_type_before_promotion"] = piece_to_move.type
        move_details["original_color_of_moved_piece"] = piece_to_move.color

    def _handle_simulation_return(
        self, move_details_for_history: Dict[str, Any]
    ) -> Dict[str, Any]:
        return move_details_for_history

    def _handle_real_move_return(
        self, player_who_made_the_move: Color
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        next_obs = self.get_observation()
        reward = 0.0
        if self.game_over:
            if self.winner == player_who_made_the_move:
                reward = 1.0
            elif self.winner is None:  # Draw
                reward = 0.0
            else:  # Opponent won
                reward = -1.0

        done = self.game_over
        info: Dict[str, Any] = {
            "reason": self.termination_reason if self.game_over else "Game ongoing"
        }
        if self.game_over and self.winner is not None:
            info["winner"] = self.winner.name
        return next_obs, reward, done, info

    def make_move(
        self, move_tuple: MoveTuple, is_simulation: bool = False
    ) -> Union[Dict[str, Any], Tuple[np.ndarray, float, bool, Dict[str, Any]]]:
        """
        Applies a move, updates history, and delegates to shogi_move_execution.
        This is the primary method for making a move.

        Args:
            move_tuple: The move to apply
            is_simulation: If True, this is a simulation move for legal move checking

        Returns:
            If is_simulation is True: move details dictionary
            Otherwise: A 4-tuple (observation, reward, done, info) for RL training
        """
        if self.game_over and not is_simulation:
            # Return early if game is over and it's not a simulation
            return self._handle_real_move_return(
                self.current_player.opponent()
            )  # Pass opponent as nominal player

        self._validate_move_tuple_format(move_tuple)  # Ensure validation is called

        player_who_made_the_move = self.current_player
        move_count_before_move = self.move_count

        move_details_for_history = self._prepare_move_details_for_history(
            move_tuple, player_who_made_the_move, move_count_before_move
        )

        if move_tuple[0] is None:  # Drop move
            # move_tuple for drop: (None, None, to_r, to_c, piece_type: PieceType)
            # _populate_drop_move_details expects (self, move_details, promote_or_drop_info: PieceType)
            drop_piece_type = move_tuple[4]
            if not isinstance(drop_piece_type, PieceType):
                raise ValueError(
                    f"Invalid piece type for drop in move_tuple: {drop_piece_type}"
                )
            self._populate_drop_move_details(move_details_for_history, drop_piece_type)
        else:  # Board move
            # move_tuple for board: (from_r, from_c, to_r, to_c, promote_bool: bool)
            # _validate_and_populate_board_move_details expects
            # (self, move_details, from_r, from_c, to_r, to_c, player_who_made_the_move)
            from_r, from_c, to_r, to_c, _ = move_tuple
            # Type assertion based on _validate_move_tuple_format and the if condition
            assert isinstance(from_r, int), "from_r must be int for a board move"
            assert isinstance(from_c, int), "from_c must be int for a board move"
            assert isinstance(to_r, int), "to_r must be int for a board move"
            assert isinstance(to_c, int), "to_c must be int for a board move"

            self._validate_and_populate_board_move_details(
                move_details_for_history,
                from_r,
                from_c,
                to_r,
                to_c,
                player_who_made_the_move,
            )

        # --- Capture state for simulation undo if needed ---
        if is_simulation:
            move_details_for_history["original_board_state"] = copy.deepcopy(self.board)
            move_details_for_history["original_hands_state"] = copy.deepcopy(self.hands)

        move_application_result = shogi_move_execution.apply_move_to_board_state(
            self.board, self.hands, move_tuple, player_who_made_the_move
        )
        move_details_for_history["captured"] = (
            move_application_result.captured_piece_type
        )
        move_details_for_history["was_promoted_in_move"] = (
            move_application_result.was_promotion
        )

        shogi_move_execution.apply_move_to_game(self, is_simulation)

        if not is_simulation:
            current_state_hash = self._board_state_hash()
            move_details_for_history["state_hash"] = current_state_hash
            self.move_history.append(move_details_for_history)
            self._check_and_update_termination_status(player_who_made_the_move)

        if is_simulation:
            return self._handle_simulation_return(move_details_for_history)
        else:
            return self._handle_real_move_return(player_who_made_the_move)

    def _restore_board_and_hands_for_undo(
        self, last_move_details: Dict[str, Any]
    ) -> None:
        """Helper to restore board and hands from move details during undo."""
        move_tuple = last_move_details["move"]
        _, _, to_r, to_c, _ = move_tuple

        if last_move_details["is_drop"]:
            dropped_piece_type = last_move_details["dropped_piece_type"]
            assert dropped_piece_type is not None, "dropped_piece_type missing for drop"

            self.board[to_r][to_c] = None
            self.hands[self.current_player.value][dropped_piece_type] = (
                self.hands[self.current_player.value].get(dropped_piece_type, 0) + 1
            )
        else:  # Board move
            from_r, from_c, _, _, _ = move_tuple  # Unpack from_r, from_c here
            original_type_at_source = last_move_details[
                "original_type_before_promotion"
            ]
            original_color_of_moved_piece = last_move_details[
                "original_color_of_moved_piece"
            ]
            assert (
                original_type_at_source is not None
            ), "original_type_before_promotion missing"
            assert (
                original_color_of_moved_piece is not None
            ), "original_color_of_moved_piece missing"

            self.board[from_r][from_c] = Piece(
                original_type_at_source, original_color_of_moved_piece
            )

            captured_piece_board_type = last_move_details["captured"]
            if captured_piece_board_type:
                captured_piece_color = self.current_player.opponent()
                self.board[to_r][to_c] = Piece(
                    captured_piece_board_type, captured_piece_color
                )

                hand_type_of_captured = PIECE_TYPE_TO_HAND_TYPE.get(
                    captured_piece_board_type
                )
                if hand_type_of_captured is None:
                    raise ValueError(
                        f"Cannot convert captured board type {captured_piece_board_type} to hand type during undo."
                    )

                player_hand = self.hands[self.current_player.value]
                if player_hand.get(hand_type_of_captured, 0) > 0:
                    player_hand[hand_type_of_captured] -= 1
                else:
                    # This implies an inconsistency
                    # Consider logging or raising a more specific error if strictness is required
                    pass
            else:
                self.board[to_r][to_c] = None

    def undo_move(
        self, simulation_undo_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Reverts the last move made, restoring the previous game state.
        Can use simulation_undo_details to undo a simulated move not in history.
        """
        if simulation_undo_details:
            original_board_state = simulation_undo_details.get("original_board_state")
            original_hands_state = simulation_undo_details.get("original_hands_state")
            original_current_player = simulation_undo_details.get(
                "player_who_made_the_move"
            )
            original_move_count = simulation_undo_details.get("move_count_before_move")

            if not all(
                isinstance(arg, expected_type)
                for arg, expected_type in [
                    (original_board_state, list),
                    (original_hands_state, dict),
                    (original_current_player, Color),
                    (original_move_count, int),
                ]
            ):
                # More detailed type error could be raised here if needed
                raise TypeError(
                    "One or more arguments from simulation_undo_details have incorrect types."
                )

            assert original_board_state is not None, "original_board_state missing"
            assert original_hands_state is not None, "original_hands_state missing"
            assert (
                original_current_player is not None
            ), "original_current_player missing"
            assert original_move_count is not None, "original_move_count missing"

            shogi_move_execution.revert_last_applied_move(
                self,
                original_board_state=original_board_state,
                original_hands_state=original_hands_state,
                original_current_player=original_current_player,
                original_move_count=original_move_count,
            )
        else:  # Undo from game history
            if not self.move_history:
                return

            last_move_details = self.move_history.pop()
            self.current_player = last_move_details["player_who_made_the_move"]
            self.move_count = last_move_details["move_count_before_move"]

            self._restore_board_and_hands_for_undo(last_move_details)

            self.game_over = False
            self.winner = None
            self.termination_reason = None

    def add_to_hand(self, captured_piece: Piece, capturing_player_color: Color) -> None:
        """
        Adds a captured piece (as unpromoted) to the capturing player's hand.
        """
        if captured_piece.type == PieceType.KING:
            return

        hand_piece_type = PIECE_TYPE_TO_HAND_TYPE.get(captured_piece.type)
        if hand_piece_type is None:
            raise ValueError(
                f"Invalid piece type {captured_piece.type} to add to hand."
            )

        self.hands[capturing_player_color.value][hand_piece_type] = (
            self.hands[capturing_player_color.value].get(hand_piece_type, 0) + 1
        )

    def remove_from_hand(
        self, piece_type: PieceType, color: Color
    ) -> bool:  # Corrected signature
        """Removes one piece of piece_type from the specified color's hand."""
        if piece_type not in get_unpromoted_types():
            # Attempting to remove a promoted type from hand, which is invalid.
            # Or, piece_type is KING, which cannot be in hand.
            # print(f\"Warning: Attempted to remove invalid piece type '{piece_type}' from hand.\")
            return False  # Or raise error

        hand_to_modify = self.hands[color.value]
        if hand_to_modify.get(piece_type, 0) > 0:
            hand_to_modify[piece_type] -= 1
            return True
        # print(f\"Warning: Attempted to remove {piece_type} from {color}\'s hand, but not available.\")
        return False

    def get_pieces_in_hand(self, color: Color) -> Dict[PieceType, int]:
        """
        Returns a copy of the pieces in hand for the specified player.
        """
        return self.hands[color.value].copy()

    def is_in_promotion_zone(self, row: int, color: Color) -> bool:
        """Checks if the specified row is in the promotion zone for the given color."""
        if color == Color.BLACK:
            return 0 <= row <= 2
        return 6 <= row <= 8

    def _check_drop_pawn_rules(
        self, row: int, col: int, player_color: Color, last_rank: int
    ) -> bool:
        """Checks specific rules for dropping a pawn."""
        if self.is_nifu(player_color, col):
            return False
        if row == last_rank:
            return False
        if self.is_uchi_fu_zume(row, col, player_color):
            return False
        return True

    def _check_drop_lance_rules(self, row: int, last_rank: int) -> bool:
        """Checks specific rules for dropping a lance."""
        # player_color removed as it was unused
        if row == last_rank:
            return False
        return True

    def _check_drop_knight_rules(
        self, row: int, last_rank: int, second_last_rank: int
    ) -> bool:
        """Checks specific rules for dropping a knight."""
        # player_color removed as it was unused
        if row == last_rank or row == second_last_rank:
            return False
        return True

    def can_drop_piece(
        self, piece_type: PieceType, row: int, col: int, player_color: Color
    ) -> bool:
        """
        Checks if a piece of the specified type can be legally dropped by the player
        at the given board coordinates.
        This includes checks for nifu (two pawns in the same file) and uchi_fu_zume (dropping a pawn for checkmate),
        and that pieces are not dropped where they have no further moves.
        """
        if (
            piece_type == PieceType.KING
            or not self.is_on_board(row, col)
            or self.hands[player_color.value].get(piece_type, 0) <= 0
            or self.board[row][col] is not None
        ):  # Square must be empty
            return False

        last_rank = 0 if player_color == Color.BLACK else 8
        second_last_rank = 1 if player_color == Color.BLACK else 7

        if piece_type == PieceType.PAWN:
            return self._check_drop_pawn_rules(row, col, player_color, last_rank)
        elif piece_type == PieceType.LANCE:
            return self._check_drop_lance_rules(row, last_rank)
        elif piece_type == PieceType.KNIGHT:
            return self._check_drop_knight_rules(row, last_rank, second_last_rank)

        return True  # Default for other pieces (Gold, Silver, Bishop, Rook)

    def test_move(self, move_tuple: MoveTuple) -> bool:
        """
        Tests if a move is valid without applying it or throwing exceptions.

        Args:
            move_tuple: The move to validate

        Returns:
            True if the move is valid, False otherwise
        """
        if self.game_over:
            return False

        try:
            # Validate move tuple format
            self._validate_move_tuple_format(move_tuple)
        except ValueError:
            return False

        try:
            if move_tuple[0] is None:  # Drop move
                return self._test_drop_move(move_tuple)
            else:  # Board move
                return self._test_board_move(move_tuple)

        except Exception:
            # If any unexpected error occurs, treat as invalid move
            return False

    def _test_drop_move(self, move_tuple: MoveTuple) -> bool:
        """Test if a drop move is valid."""
        drop_piece_type = move_tuple[4]
        if not isinstance(drop_piece_type, PieceType):
            return False

        to_r, to_c = move_tuple[2], move_tuple[3]

        # Use existing drop validation method
        return self.can_drop_piece(drop_piece_type, to_r, to_c, self.current_player)

    def _test_board_move(self, move_tuple: MoveTuple) -> bool:
        """Test if a board move is valid."""
        from_r, from_c, to_r, to_c, _ = move_tuple

        # Type checks
        if (
            not isinstance(from_r, int)
            or not isinstance(from_c, int)
            or not isinstance(to_r, int)
            or not isinstance(to_c, int)
        ):
            return False

        # Use the same validation logic as _validate_and_populate_board_move_details
        piece_to_move = self.get_piece(from_r, from_c)
        if piece_to_move is None:
            return False

        if piece_to_move.color != self.current_player:
            return False

        # Check if move follows piece movement rules
        potential_squares = shogi_rules_logic.generate_piece_potential_moves(
            self, piece_to_move, from_r, from_c
        )
        if (to_r, to_c) not in potential_squares:
            return False

        # Check if target square contains own piece
        target_piece = self.get_piece(to_r, to_c)
        if target_piece and target_piece.color == self.current_player:
            return False

        # CRITICAL: Check if move would leave king in check
        # Simulate the move and check if king is safe
        try:
            move_details = self.make_move(move_tuple, is_simulation=True)
            # If make_move succeeds in simulation, the move is valid
            # Need to undo the simulation
            self.undo_move(simulation_undo_details=move_details)
            return True
        except Exception:
            # If make_move fails, the move is invalid
            return False
