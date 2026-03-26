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
        "move_log": list(step_manager.move_log[-300:]),
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


def extract_policy_insight(
    ppo_agent: Any,
    observation: Any,
    policy_mapper: Any,
    top_k: int = 10,
) -> Optional[Dict[str, Any]]:
    """Extract policy insight from the agent's current observation.

    Performs a ``torch.no_grad()`` forward pass to get action probabilities
    and the critic's value estimate.  Returns ``None`` on any error so that
    snapshot production is never interrupted.

    The ``action_heatmap`` is a 9x9 grid where each cell holds the sum of
    action probabilities whose destination square is that cell.
    """
    try:
        import torch

        if observation is None or ppo_agent is None:
            return None

        model = ppo_agent.model
        device = ppo_agent.device

        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Observation normalization (mirrors PPOAgent.select_action)
        from torch.amp import GradScaler

        scaler = getattr(ppo_agent, "scaler", None)
        if scaler is not None and not isinstance(scaler, GradScaler):
            if hasattr(scaler, "transform"):
                obs_tensor = scaler.transform(obs_tensor)
            else:
                obs_tensor = scaler(obs_tensor)

        was_training = model.training
        with torch.no_grad():
            model.eval()
            policy_logits, value = model(obs_tensor)
            model.train(was_training)

        # Softmax over the full action space
        probs = torch.softmax(policy_logits.squeeze(0), dim=0)
        value_estimate = float(value.item())

        # Action entropy
        log_probs = torch.log(probs + 1e-10)
        action_entropy = float(-(probs * log_probs).sum().item())

        probs_np = probs.cpu().numpy()

        # Build 9x9 heatmap: sum of probs per destination square
        heatmap = [[0.0] * 9 for _ in range(9)]
        idx_to_move = policy_mapper.idx_to_move
        for idx in range(len(idx_to_move)):
            p = float(probs_np[idx])
            if p < 1e-8:
                continue
            move = idx_to_move[idx]
            # Destination square is at indices [2], [3]
            to_r, to_c = move[2], move[3]
            if (
                isinstance(to_r, int)
                and isinstance(to_c, int)
                and 0 <= to_r < 9
                and 0 <= to_c < 9
            ):
                heatmap[to_r][to_c] += p

        # Top-K actions
        top_indices = probs_np.argsort()[-top_k:][::-1]
        top_actions = []
        for idx in top_indices:
            p = float(probs_np[idx])
            if p < 1e-8:
                break
            try:
                usi = policy_mapper.action_idx_to_usi_move(int(idx))
            except (IndexError, ValueError):
                usi = f"idx:{idx}"
            top_actions.append({"action": usi, "prob": p})

        return {
            "action_heatmap": heatmap,
            "top_actions": top_actions,
            "value_estimate": value_estimate,
            "action_entropy": action_entropy,
        }

    except Exception:
        # Non-fatal — snapshot production continues without insight
        return None


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

    # Policy insight (optional — gated on config and training state)
    training["policy_insight"] = None
    config = getattr(trainer, "config", None)
    webui_cfg = getattr(config, "webui", None) if config else None
    insight_enabled = getattr(webui_cfg, "policy_insight", False)
    is_processing = training.get("metrics", {}).get("processing", False)

    if insight_enabled and not is_processing:
        ppo_agent = getattr(trainer, "agent", None)
        step_mgr = getattr(trainer, "step_manager", None)
        obs = getattr(step_mgr, "_latest_obs_for_snapshot", None)
        env_mgr = getattr(trainer, "env_manager", None)
        mapper = getattr(env_mgr, "policy_mapper", None)
        top_k = getattr(webui_cfg, "policy_insight_top_k", 10)

        if ppo_agent is not None and mapper is not None:
            training["policy_insight"] = extract_policy_insight(
                ppo_agent, obs, mapper, top_k=top_k
            )

    return training


def extract_lineage_summary(
    registry: Any,
    graph: Any,
    current_model_id: Optional[str],
) -> Dict[str, Any]:
    """Extract lineage summary for the broadcast envelope.

    Parameters
    ----------
    registry
        The LineageRegistry (for event_count and recent events).
    graph
        The LineageGraph read model built from registry events.
    current_model_id
        The model_id of the currently active checkpoint, or None.
    """
    event_count = registry.event_count

    if current_model_id is None or graph.get_node(current_model_id) is None:
        return {
            "event_count": event_count,
            "latest_checkpoint_id": None,
            "parent_id": None,
            "model_id": None,
            "run_name": None,
            "generation": 0,
            "latest_rating": None,
            "recent_events": [],
            "ancestor_chain": [],
        }

    node = graph.get_node(current_model_id)
    ancestors = graph.ancestors(current_model_id)

    # Recent events: last 10 from registry, summarised
    all_events = registry.load_all()
    recent = [
        {
            "event_type": e["event_type"],
            "model_id": e["model_id"],
            "emitted_at": e.get("emitted_at", ""),
        }
        for e in all_events[-10:]
    ]

    return {
        "event_count": event_count,
        "latest_checkpoint_id": current_model_id,
        "parent_id": node.parent_model_id,
        "model_id": current_model_id,
        "run_name": node.run_name,
        "generation": len(ancestors) + 1,
        "latest_rating": node.latest_rating,
        "recent_events": recent,
        "ancestor_chain": [a.model_id for a in ancestors],
    }


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

    active_views: List[str] = ["training"]
    health_overrides: Dict[str, str] = {"training": "ok"}

    # Lineage view — only when registry is available on the trainer
    lineage_data = None
    registry = getattr(trainer, "lineage_registry", None)
    if registry is not None:
        from keisei.lineage.graph import LineageGraph

        graph = LineageGraph.from_events(registry.load_all())
        current_model_id = getattr(
            getattr(trainer, "model_manager", None), "current_model_id", None
        )
        lineage_data = extract_lineage_summary(registry, graph, current_model_id)
        active_views.append("lineage")
        health_overrides["lineage"] = "ok"

    envelope: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.time(),
        "speed": speed,
        "mode": _resolve_mode(trainer),
        "active_views": active_views,
        "health": make_health_map(**health_overrides),
        "training": training,
        "pending_updates": sanitize_pending_updates(pending_updates),
    }

    if lineage_data is not None:
        envelope["lineage"] = lineage_data

    return envelope


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
