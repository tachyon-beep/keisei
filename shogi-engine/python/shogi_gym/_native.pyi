"""Type stubs for shogi_gym._native (Rust PyO3 extension)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

class DefaultActionMapper:
    action_space_size: int
    def __init__(self) -> None: ...
    def encode_board_move(self, from_sq: int, to_sq: int, promote: bool, is_white: bool) -> int: ...
    def encode_drop_move(self, to_sq: int, piece_type_idx: int, is_white: bool) -> int: ...
    def decode(self, idx: int, is_white: bool) -> dict[str, object]: ...


class DefaultObservationGenerator:
    channels: int
    def __init__(self) -> None: ...


class StepMetadata:
    captured_piece: NDArray[np.uint8]
    termination_reason: NDArray[np.uint8]
    ply_count: NDArray[np.uint16]
    material_balance: NDArray[np.int32]


class StepResult:
    observations: NDArray[np.float32]
    legal_masks: NDArray[np.bool_]
    rewards: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    terminal_observations: NDArray[np.float32]
    current_players: NDArray[np.uint8]
    step_metadata: StepMetadata


class ResetResult:
    observations: NDArray[np.float32]
    legal_masks: NDArray[np.bool_]


class VecEnv:
    num_envs: int
    action_space_size: int
    observation_channels: int
    def __init__(self, num_envs: int = 512, max_ply: int = 500) -> None: ...
    def reset(self) -> ResetResult: ...
    def step(self, actions: list[int]) -> StepResult: ...


class SpectatorEnv:
    action_space_size: int
    current_player: str
    ply: int
    is_over: bool
    def __init__(self, max_ply: int = 500) -> None: ...
    def reset(self) -> dict[str, object]: ...
    def step(self, action: int) -> dict[str, object]: ...
    def to_dict(self) -> dict[str, object]: ...
    def to_sfen(self) -> str: ...
    def get_observation(self) -> list[float]: ...
    def legal_actions(self) -> list[int]: ...
