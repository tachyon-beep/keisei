"""shogi-gym: Rust-powered RL environments for Shogi."""

from enum import IntEnum

from shogi_gym._native import (
    DefaultActionMapper,
    DefaultObservationGenerator,
    VecEnv,
    SpectatorEnv,
    StepResult,
    ResetResult,
    StepMetadata,
)


class TerminationReason(IntEnum):
    """Codes stored in StepMetadata.termination_reason."""

    NOT_TERMINATED = 0
    CHECKMATE = 1
    REPETITION = 2
    PERPETUAL_CHECK = 3
    IMPASSE = 4
    MAX_MOVES = 5


# Sentinel value in StepMetadata.captured_piece meaning "no capture this step"
NO_CAPTURE: int = 255

# Observation channel offsets (46-channel layout)
OBS_CURRENT_UNPROMOTED_START = 0   # channels 0-7:  current player's unpromoted pieces
OBS_CURRENT_PROMOTED_START = 8     # channels 8-13: current player's promoted pieces
OBS_OPPONENT_UNPROMOTED_START = 14 # channels 14-21: opponent's unpromoted pieces
OBS_OPPONENT_PROMOTED_START = 22   # channels 22-27: opponent's promoted pieces
OBS_CURRENT_HAND_START = 28        # channels 28-34: current player's hand (normalized)
OBS_OPPONENT_HAND_START = 35       # channels 35-41: opponent's hand (normalized)
OBS_PLAYER_INDICATOR = 42          # 1.0 if Black, 0.0 if White
OBS_MOVE_COUNT = 43                # ply / max_ply
OBS_RESERVED_START = 44            # channels 44-45: reserved (zeros)
OBS_NUM_CHANNELS = 46


__all__ = [
    "DefaultActionMapper",
    "DefaultObservationGenerator",
    "VecEnv",
    "SpectatorEnv",
    "StepResult",
    "ResetResult",
    "StepMetadata",
    "TerminationReason",
    "NO_CAPTURE",
    "OBS_CURRENT_UNPROMOTED_START",
    "OBS_CURRENT_PROMOTED_START",
    "OBS_OPPONENT_UNPROMOTED_START",
    "OBS_OPPONENT_PROMOTED_START",
    "OBS_CURRENT_HAND_START",
    "OBS_OPPONENT_HAND_START",
    "OBS_PLAYER_INDICATOR",
    "OBS_MOVE_COUNT",
    "OBS_RESERVED_START",
    "OBS_NUM_CHANNELS",
]
