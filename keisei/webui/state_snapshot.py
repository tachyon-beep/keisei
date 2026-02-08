"""
state_snapshot.py: Clean state extraction for the Streamlit dashboard.

Builds a JSON-serializable snapshot of training state that the Streamlit app
reads via a shared state file.  Output conforms to the v1
``BroadcastStateEnvelope`` contract defined in ``view_contracts.py``.

Uses the actual game API directly — no hasattr probing or random fallbacks.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .view_contracts import (
    SCHEMA_VERSION,
    make_health_map,
    sanitize_pending_updates,
)


def extract_board_state(game: Any) -> Dict[str, Any]:
    """Extract board state from a ShogiGame instance.

    Uses the concrete Piece API: piece.type (PieceType), piece.color (Color),
    piece.is_promoted (bool), and game.hands[Color.value] (dict[PieceType, int]).
    """
    board = []
    for row in range(9):
        board_row: List[Optional[Dict[str, Any]]] = []
        for col in range(9):
            piece = game.board[row][col]
            if piece is not None:
                board_row.append(
                    {
                        "type": piece.type.name.lower(),
                        "color": piece.color.name.lower(),
                        "promoted": piece.is_promoted,
                    }
                )
            else:
                board_row.append(None)
        board.append(board_row)

    # Hands: game.hands is Dict[int, Dict[PieceType, int]]
    # Color.BLACK.value == 0, Color.WHITE.value == 1
    black_hand: Dict[str, int] = {}
    white_hand: Dict[str, int] = {}
    for pt, count in game.hands.get(0, {}).items():
        if count > 0:
            black_hand[pt.name.lower()] = count
    for pt, count in game.hands.get(1, {}).items():
        if count > 0:
            white_hand[pt.name.lower()] = count

    current_player = game.current_player.name.lower()
    winner = game.winner.name.lower() if game.winner is not None else None

    return {
        "board": board,
        "current_player": current_player,
        "move_count": game.move_count,
        "game_over": game.game_over,
        "winner": winner,
        "black_hand": black_hand,
        "white_hand": white_hand,
    }


def extract_metrics(metrics_manager: Any) -> Dict[str, Any]:
    """Extract training metrics from MetricsManager.

    Pulls history lists (last 50), win rates, and stat counters.
    """
    history = metrics_manager.history

    # Last 50 of each history list
    def tail(lst: Any, n: int = 50) -> List:
        return list(lst)[-n:]

    # Win rate history
    win_rates_history = tail(history.win_rates_history)

    return {
        "global_timestep": metrics_manager.global_timestep,
        "total_episodes": metrics_manager.total_episodes_completed,
        "black_wins": metrics_manager.black_wins,
        "white_wins": metrics_manager.white_wins,
        "draws": metrics_manager.draws,
        "processing": metrics_manager.processing,
        "learning_curves": {
            "policy_losses": tail(history.policy_losses),
            "value_losses": tail(history.value_losses),
            "entropies": tail(history.entropies),
            "kl_divergences": tail(history.kl_divergences),
            "clip_fractions": tail(history.clip_fractions),
            "learning_rates": tail(history.learning_rates),
            "episode_lengths": tail(history.episode_lengths),
            "episode_rewards": tail(history.episode_rewards),
        },
        "win_rates_history": win_rates_history,
        "hot_squares": metrics_manager.get_hot_squares(3),
    }


def extract_step_info(step_manager: Any) -> Dict[str, Any]:
    """Extract step/move info from StepManager."""
    return {
        "move_log": list(step_manager.move_log[-20:]),
        "sente_capture_count": step_manager.sente_capture_count,
        "gote_capture_count": step_manager.gote_capture_count,
        "sente_drop_count": step_manager.sente_drop_count,
        "gote_drop_count": step_manager.gote_drop_count,
        "sente_promo_count": step_manager.sente_promo_count,
        "gote_promo_count": step_manager.gote_promo_count,
    }


def extract_buffer_info(experience_buffer: Any) -> Dict[str, int]:
    """Extract experience buffer size/capacity."""
    return {
        "size": experience_buffer.size(),
        "capacity": experience_buffer.capacity(),
    }


def _resolve_mode(trainer: Any) -> str:
    """Derive the broadcast mode from the trainer's config.

    Returns ``"training_only"`` when periodic evaluation is disabled,
    otherwise the configured evaluation strategy string.
    """
    config = getattr(trainer, "config", None)
    if config is None:
        return "training_only"
    eval_cfg = getattr(config, "evaluation", None)
    if eval_cfg is None:
        return "training_only"
    if not getattr(eval_cfg, "enable_periodic_evaluation", True):
        return "training_only"
    return getattr(eval_cfg, "strategy", "training_only")


def _build_training_view(trainer: Any) -> Dict[str, Any]:
    """Assemble the ``TrainingViewState`` payload from the trainer."""
    training: Dict[str, Any] = {}

    # Board state
    if trainer.game is not None:
        training["board_state"] = extract_board_state(trainer.game)
    else:
        training["board_state"] = None

    # Metrics (always present)
    training["metrics"] = extract_metrics(trainer.metrics_manager)

    # Step info
    if trainer.step_manager is not None:
        training["step_info"] = extract_step_info(trainer.step_manager)
    else:
        training["step_info"] = None

    # Buffer info
    if trainer.experience_buffer is not None:
        training["buffer_info"] = extract_buffer_info(trainer.experience_buffer)
    else:
        training["buffer_info"] = None

    # Model info (always present)
    training["model_info"] = {
        "gradient_norm": getattr(trainer, "last_gradient_norm", 0.0),
    }

    return training


def build_snapshot(
    trainer: Any,
    speed: float = 0.0,
    pending_updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a v1 ``BroadcastStateEnvelope`` from the trainer.

    This is the single entry point called by StreamlitManager.  The output
    conforms to the contract in ``view_contracts.py`` and passes
    ``validate_envelope()``.
    """
    training = _build_training_view(trainer)

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.time(),
        "speed": speed,
        "mode": _resolve_mode(trainer),
        "active_views": ["training"],
        "health": make_health_map(training="ok"),
        "training": training,
        "pending_updates": sanitize_pending_updates(pending_updates),
    }


def write_snapshot_atomic(snapshot: Dict[str, Any], path: Path) -> None:
    """Write snapshot as JSON atomically using tempfile + os.replace().

    This prevents the Streamlit reader from seeing partial writes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(snapshot)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".state_"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp_path, str(path))
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
