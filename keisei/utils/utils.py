"""
utils.py: Contains PolicyOutputMapper and TrainingLogger.
"""

from __future__ import annotations

import datetime
import json
import os
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    TextIO,
    cast,
)

import torch
import yaml  # type: ignore[import]
from pydantic import ValidationError
from keisei.config_schema import AppConfig
from keisei.shogi.shogi_core_definitions import (
    BoardMoveTuple,
    DropMoveTuple,
    PieceType,
    get_unpromoted_types,
)
from keisei.utils.unified_logger import (
    log_error_to_stderr,
    log_info_to_stderr,
)

# --- Config Loader Utility ---

# Mapping of flat override keys to nested config paths
FLAT_KEY_TO_NESTED = {
    # Env
    "SEED": "env.seed",
    "DEVICE": "env.device",
    "INPUT_CHANNELS": "env.input_channels",
    # Training
    "TOTAL_TIMESTEPS": "training.total_timesteps",
    "LEARNING_RATE": "training.learning_rate",
    "PPO_EPOCHS": "training.ppo_epochs",
    "MINIBATCH_SIZE": "training.minibatch_size",
    "GAMMA": "training.gamma",
    "CLIP_EPSILON": "training.clip_epsilon",
    "VALUE_LOSS_COEFF": "training.value_loss_coeff",
    "ENTROPY_COEFF": "training.entropy_coef",
    "STEPS_PER_EPOCH": "training.steps_per_epoch",
    "CHECKPOINT_INTERVAL_TIMESTEPS": "training.checkpoint_interval_timesteps",
    # Logging
    "MODEL_DIR": "logging.model_dir",
    "LOG_FILE": "logging.log_file",
    # Evaluation
    "NUM_GAMES": "evaluation.num_games",
    "OPPONENT_TYPE": "evaluation.opponent_type",
    # WandB
    "WANDB_ENABLED": "wandb.enabled",
    "WANDB_PROJECT": "wandb.project",
    "WANDB_ENTITY": "wandb.entity",
    # Display moves
    "DISPLAY_MOVES": "display.display_moves",
    "TURN_TICK": "display.turn_tick",
}


def _load_yaml_or_json(path: str) -> dict:
    if path.endswith((".yaml", ".yml")):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file type: {path}")


def _merge_overrides(config_data: dict, overrides: dict) -> None:
    for k, v in overrides.items():
        parts = k.split(".")
        d = config_data
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = v


def _map_flat_overrides(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {}
    if overrides is None:
        return mapped
    for k, v in overrides.items():
        if k.isupper() and k in FLAT_KEY_TO_NESTED:
            mapped[FLAT_KEY_TO_NESTED[k]] = v
        else:
            mapped[k] = v
    return mapped


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge overrides into base dict, preserving unspecified keys."""
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def load_config(
    config_path: Optional[str] = None, cli_overrides: Optional[Dict[str, Any]] = None
) -> AppConfig:
    """
    Loads configuration from a YAML or JSON file and applies CLI overrides.
    Always loads default_config.yaml as the base, then merges in overrides from config_path (if present), then CLI overrides.
    """
    base_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "default_config.yaml",
    )
    config_data = _load_yaml_or_json(base_config_path)
    # If config_path is provided and is not the default, treat as override file (JSON or YAML)
    if config_path and os.path.abspath(config_path) != os.path.abspath(
        base_config_path
    ):
        override_data = _load_yaml_or_json(config_path)
        top_keys = {
            "env",
            "training",
            "evaluation",
            "logging",
            "wandb",
            "demo",
            "parallel",
            "display",
        }
        if not (
            isinstance(override_data, dict) and top_keys & set(override_data.keys())
        ):
            mapped_overrides = _map_flat_overrides(override_data)
            _merge_overrides(config_data, mapped_overrides)
        else:
            _deep_merge(config_data, override_data)
    if cli_overrides:
        mapped_overrides = _map_flat_overrides(cli_overrides)
        _merge_overrides(config_data, mapped_overrides)
    try:
        config = AppConfig.model_validate(config_data)
    except ValidationError as e:
        log_error_to_stderr("Utils", "Configuration validation error:")
        log_error_to_stderr("Utils", str(e))
        raise
    return config


if TYPE_CHECKING:
    from keisei.shogi.shogi_core_definitions import MoveTuple
    from keisei.shogi.shogi_game import ShogiGame  # Added for type hinting


class BaseOpponent(ABC):
    """Abstract base class for game opponents."""

    def __init__(self, name: str = "BaseOpponent"):
        self.name = name

    @abstractmethod
    def select_move(self, game_instance: "ShogiGame") -> "MoveTuple":
        """
        Selects a move given the current game state.

        Args:
            game_instance: The current instance of the ShogiGame.

        Returns:
            A MoveTuple representing the selected move.
        """


class PolicyOutputMapper:
    """Maps Shogi moves to/from policy network output indices."""

    def __init__(self) -> None:
        """Initializes the PolicyOutputMapper by generating all possible move representations."""
        self.idx_to_move: List["MoveTuple"] = []
        self.move_to_idx: Dict["MoveTuple", int] = {}
        current_idx = 0
        self._unrecognized_moves_log_cache: Set[str] = (
            set()
        )  # Cache for logging distinct unrecognized moves
        self._unrecognized_moves_logged_count = (
            0  # Counter for logged distinct unrecognized moves
        )
        self._max_distinct_unrecognized_to_log = (
            5  # Max distinct unrecognized moves to log in detail
        )
        # Piece type mapping for USI drop characters
        self._USI_DROP_PIECE_CHARS: Dict[PieceType, str] = {
            PieceType.PAWN: "P",
            PieceType.LANCE: "L",
            PieceType.KNIGHT: "N",
            PieceType.SILVER: "S",
            PieceType.GOLD: "G",
            PieceType.BISHOP: "B",
            PieceType.ROOK: "R",
        }

        # Generate Board Moves: (from_r, from_c, to_r, to_c, promote_flag: bool)
        # Ensure from_r, from_c, to_r, to_c are treated as integers.
        for r_from in range(9):
            for c_from in range(9):
                for r_to in range(9):
                    for c_to in range(9):
                        if r_from == r_to and c_from == c_to:
                            continue  # Skip null moves

                        # Move without promotion
                        move_no_promo: BoardMoveTuple = (
                            r_from,
                            c_from,
                            r_to,
                            c_to,
                            False,
                        )
                        self.idx_to_move.append(move_no_promo)
                        self.move_to_idx[move_no_promo] = current_idx
                        current_idx += 1

                        # Move with promotion (if applicable, though ShogiGame handles legality)
                        # For policy mapping, we generate all potential promotion moves.
                        # Actual legality (e.g., can piece promote from/to square) is game logic.
                        move_promo: BoardMoveTuple = (
                            r_from,
                            c_from,
                            r_to,
                            c_to,
                            True,
                        )
                        self.idx_to_move.append(move_promo)
                        self.move_to_idx[move_promo] = current_idx
                        current_idx += 1

        # Generate Drop Moves: (None, None, to_r, to_c, piece_type_to_drop: PieceType)
        # Ensure to_r, to_c are treated as integers.
        # piece_type_to_drop should be PieceType enum, not int, if DropMoveTuple expects PieceType.
        # If DropMoveTuple expects int, then piece.value is correct.
        # Based on current DropMoveTuple = Tuple[Optional[int], Optional[int], int, int, PieceType],
        # we should use the PieceType enum directly.

        # Get unpromoted piece types that can be held in hand (excluding King)
        hand_piece_types = get_unpromoted_types()

        for r_to in range(9):
            for c_to in range(9):
                for piece_type_enum in hand_piece_types:
                    # DropMoveTuple expects PieceType enum for the piece type
                    drop_move: DropMoveTuple = (
                        None,  # from_r is None for drops
                        None,  # from_c is None for drops
                        r_to,
                        c_to,
                        piece_type_enum,  # Use the PieceType enum directly
                    )
                    self.idx_to_move.append(drop_move)
                    self.move_to_idx[drop_move] = current_idx
                    current_idx += 1

    def get_total_actions(self) -> int:
        """Returns the total number of unique actions (moves) in the policy output."""
        return len(self.idx_to_move)

    def shogi_move_to_policy_index(self, move: "MoveTuple") -> int:
        """Convert a Shogi MoveTuple to its policy index."""
        idx = self.move_to_idx.get(move)
        if idx is None:
            # Attempt to handle cases where the exact tuple might not match due to PieceType enum identity
            # This is a fallback, ideally the move objects should match directly.
            if len(move) == 5 and move[0] is None:  # Potential DropMoveTuple
                # Reconstruct DropMoveTuple with PieceType from self.idx_to_move if a similar one exists
                # This is a bit of a heuristic and might need refinement
                for stored_move in self.move_to_idx:
                    if (
                        len(stored_move) == 5
                        and stored_move[0] is None
                        and stored_move[2] == move[2]
                        and stored_move[3] == move[3]
                        and isinstance(stored_move[4], PieceType)
                        and isinstance(move[4], PieceType)
                        and stored_move[4].value == move[4].value
                    ):  # Compare PieceType by value
                        idx = self.move_to_idx.get(stored_move)
                        break
            if idx is None:  # If still not found after heuristic
                raise ValueError(
                    f"Move {move} (type: {type(move)}, element types: {[type(el) for el in move]}) "
                    f"not found in PolicyOutputMapper's known moves. Known keys example: {list(self.move_to_idx.keys())[0]}"
                )
        return idx

    def policy_index_to_shogi_move(self, idx: int) -> "MoveTuple":
        """Convert a policy index back to its Shogi MoveTuple."""
        if (
            0 <= idx < self.get_total_actions()
        ):  # MODIFIED: Changed self.total_actions to self.get_total_actions()
            return self.idx_to_move[idx]
        raise IndexError(
            f"Policy index {idx} is out of bounds (0-{self.get_total_actions() - 1})."  # MODIFIED: Changed self.total_actions to self.get_total_actions()
        )

    def get_legal_mask(
        self, legal_shogi_moves: List["MoveTuple"], device: torch.device
    ) -> torch.Tensor:
        """
        Creates a boolean mask indicating which actions in the policy output are legal.

        Args:
            legal_shogi_moves: A list of legal Shogi moves (MoveTuple).
            device: The torch device on which to create the mask tensor.

        Returns:
            A 1D boolean tensor where True indicates a legal move.
        """
        mask = torch.zeros(self.get_total_actions(), dtype=torch.bool, device=device)
        for move in legal_shogi_moves:
            try:
                idx = self.shogi_move_to_policy_index(move)
                mask[idx] = True
            except ValueError as e:
                # Hard crash on unmapped moves to prevent corrupted experiments
                raise ValueError(
                    f"CRITICAL: Legal move {move} could not be mapped to policy index. "
                    f"This indicates incomplete move coverage in PolicyOutputMapper which will corrupt experiments. "
                    f"Original error: {e}"
                ) from e

        return mask

    def _usi_sq(self, r: int, c: int) -> str:
        """Converts 0-indexed (row, col) to USI square string (e.g., (0,0) -> '9a')."""
        if not (0 <= r <= 8 and 0 <= c <= 8):
            raise ValueError(f"Invalid square coordinates: ({r}, {c})")
        file = str(9 - c)
        rank = chr(ord("a") + r)
        return f"{file}{rank}"

    def _get_usi_char_for_drop(self, piece_type: PieceType) -> str:
        """Gets the USI character for a droppable piece."""
        if piece_type not in self._USI_DROP_PIECE_CHARS:
            raise ValueError(
                f"Piece type {piece_type} cannot be dropped or is not a recognized droppable piece."
            )
        return self._USI_DROP_PIECE_CHARS[piece_type]

    def action_idx_to_shogi_move(self, action_idx: int) -> "MoveTuple":
        """Converts an action index from the policy output back to a Shogi MoveTuple."""
        if 0 <= action_idx < len(self.idx_to_move):
            return self.idx_to_move[action_idx]
        # Log distinct unrecognized moves only a few times to avoid flooding logs
        log_message = f"Action index {action_idx} is out of bounds for idx_to_move (size {len(self.idx_to_move)})."
        raise IndexError(log_message)

    def shogi_move_to_usi(self, move_tuple: "MoveTuple") -> str:
        """Converts a Shogi MoveTuple to its USI string representation."""
        if len(move_tuple) == 5 and isinstance(move_tuple[4], bool):  # BoardMoveTuple
            from_r, from_c, to_r, to_c, promote = cast(BoardMoveTuple, move_tuple)
            if not all(
                isinstance(coord, int) for coord in [from_r, from_c, to_r, to_c]
            ):
                raise ValueError(
                    "Invalid coordinates in BoardMoveTuple for USI conversion."
                )
            usi_from_sq = self._usi_sq(from_r, from_c)
            usi_to_sq = self._usi_sq(to_r, to_c)
            promo_char = "+" if promote else ""
            return f"{usi_from_sq}{usi_to_sq}{promo_char}"
        if len(move_tuple) == 5 and isinstance(
            move_tuple[4], PieceType
        ):  # DropMoveTuple
            _none1, _none2, to_r, to_c, piece_type_enum = cast(
                DropMoveTuple, move_tuple
            )
            if not all(isinstance(coord, int) for coord in [to_r, to_c]):
                raise ValueError(
                    "Invalid coordinates in DropMoveTuple for USI conversion."
                )
            if not isinstance(piece_type_enum, PieceType):
                raise ValueError(
                    "Invalid piece type in DropMoveTuple for USI conversion."
                )
            try:
                piece_usi_char = self._get_usi_char_for_drop(piece_type_enum)
            except ValueError as e:
                raise ValueError(
                    f"Invalid piece type for drop in USI conversion: {piece_type_enum.name}"
                ) from e
            usi_to_sq = self._usi_sq(to_r, to_c)
            return f"{piece_usi_char}*{usi_to_sq}"

        raise ValueError(
            f"Unrecognized move_tuple format for USI conversion: "
            f"length {len(move_tuple)}, last element type {type(move_tuple[-1]) if move_tuple else 'N/A'}"
        )

    def usi_to_shogi_move(self, usi_move_str: str) -> "MoveTuple":
        """Converts a USI string to its Shogi MoveTuple representation."""
        if not isinstance(usi_move_str, str) or len(usi_move_str) < 4:
            raise ValueError(f"Invalid USI move string format: {usi_move_str}")

        # Helper to parse USI square (e.g., '9a' -> (0,0))
        def _parse_usi_sq(sq_str: str) -> tuple[int, int]:
            if not (len(sq_str) == 2 and sq_str[0].isdigit() and sq_str[1].isalpha()):
                raise ValueError(f"Invalid USI square format: {sq_str}")
            file = int(sq_str[0])
            rank_char = sq_str[1]
            c = 9 - file
            r = ord(rank_char) - ord("a")
            if not (0 <= r <= 8 and 0 <= c <= 8):
                raise ValueError(
                    f"Square coordinates out of bounds: {sq_str} -> ({r}, {c})"
                )
            return r, c

        # Drop move (e.g., P*5e)
        if usi_move_str[1] == "*":
            if len(usi_move_str) != 4:
                raise ValueError(f"Invalid USI drop move string length: {usi_move_str}")
            piece_char = usi_move_str[0]
            to_sq_str = usi_move_str[2:]

            dropped_piece_type: Optional[PieceType] = None
            for pt, char in self._USI_DROP_PIECE_CHARS.items():
                if char == piece_char:
                    dropped_piece_type = pt
                    break
            if dropped_piece_type is None:
                raise ValueError(f"Invalid piece character for drop: {piece_char}")

            to_r, to_c = _parse_usi_sq(to_sq_str)
            # MODIFIED: Correctly instantiate DropMoveTuple
            return (None, None, to_r, to_c, dropped_piece_type)

        # Board move (e.g., 7g7f, 2b3a+, 8h2b+)
        if 4 <= len(usi_move_str) <= 5:
            pass  # Valid length, continue processing
        else:
            raise ValueError(f"Invalid USI board move string length: {usi_move_str}")

        from_sq_str = usi_move_str[0:2]
        to_sq_str = usi_move_str[2:4]
        promote = False
        if len(usi_move_str) == 5:
            if usi_move_str[4] == "+":
                promote = True
            else:
                raise ValueError(
                    f"Invalid promotion character in USI move: {usi_move_str}"
                )

        from_r, from_c = _parse_usi_sq(from_sq_str)
        to_r, to_c = _parse_usi_sq(to_sq_str)
        # MODIFIED: Correctly instantiate BoardMoveTuple
        return (from_r, from_c, to_r, to_c, promote)

    def action_idx_to_usi_move(self, action_idx: int, _board=None) -> str:
        """Converts an action index to its USI move string representation."""
        shogi_move = self.action_idx_to_shogi_move(action_idx)
        return self.shogi_move_to_usi(shogi_move)


class TrainingLogger:
    """Handles logging of training progress to a file and stderr."""

    def __init__(
        self,
        log_file_path: str,
        also_stdout: Optional[bool] = None,
        **_kwargs: Any,
    ):
        """
        Initializes the TrainingLogger.

        Args:
            log_file_path: Path to the log file.
            also_stdout: Whether to also print log messages to stderr.
            **_kwargs: Accepts (and ignores) extra keyword arguments for compatibility.
        """
        self.log_file_path = log_file_path
        self.log_file: Optional[TextIO] = None
        self.also_stdout = also_stdout if also_stdout is not None else True

    def __enter__(self) -> "TrainingLogger":
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def log(self, message: str) -> None:
        """Logs a message to the file and optionally to stderr."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"

        if self.log_file:
            self.log_file.write(full_message + "\n")
            self.log_file.flush()

        if self.also_stdout:
            log_info_to_stderr("TrainingLogger", full_message)


class EvaluationLogger:
    """Handles logging of evaluation results to a file and optionally to stdout."""

    def __init__(self, log_file_path: str, also_stdout: bool = True, **_kwargs: Any):
        """
        Initializes the EvaluationLogger.

        Args:
            log_file_path: Path to the log file.
            also_stdout: Whether to also print log messages to stdout.
            **kwargs: To catch unexpected keyword arguments.
        """
        self.log_file_path = log_file_path
        self.also_stdout = also_stdout
        self.log_file: Optional[TextIO] = None

    def __enter__(self) -> "EvaluationLogger":
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def log(self, message: str) -> None:  # MODIFIED: Simplified to a single log method
        """Logs a message to the file and optionally to stdout."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"

        if self.log_file:
            self.log_file.write(full_message + "\n")
            self.log_file.flush()

        if self.also_stdout:
            log_error_to_stderr("EvaluationLogger", full_message)


def generate_run_name(config: "AppConfig", run_name: Optional[str] = None) -> str:
    """Generates a unique run name based on config and timestamp, or returns the provided run_name if set."""
    if run_name:
        return run_name
    # Example: keisei_resnet_feats_core46_20231027_153000
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type_short = config.training.model_type[:10]  # Truncate for brevity
    feature_set_short = config.training.input_features.replace("_", "")[:15]
    run_name_parts = [
        config.wandb.run_name_prefix if config.wandb.run_name_prefix else "keisei",
        model_type_short,
        f"feats_{feature_set_short}",
        timestamp,
    ]
    return "_".join(filter(None, run_name_parts))
