"""Build Elo rating timelines from the match log for charting.

Reads a JSONL match log and reconstructs per-model rating progression
over time, suitable for rendering with ``st.line_chart()``.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_INITIAL_ELO = 1500.0


def build_elo_timelines(
    log_path: Path,
    top_n: int = 10,
    leaderboard: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[float]]:
    """Read a JSONL match log and build per-model Elo timelines.

    Args:
        log_path: Path to the match_log.jsonl file.
        top_n: Number of top models to include in the output.
        leaderboard: If provided, use these model names (in order) to
            determine top-N instead of computing from the log.

    Returns:
        Dict mapping model name to a list of Elo values, one per match
        in the log. Models not involved in a match get their previous
        rating forward-filled. Returns ``{}`` if the log is missing or empty.
    """
    entries = _read_log(log_path)
    if not entries:
        return {}

    current_elo: Dict[str, float] = {}
    snapshots: List[Dict[str, float]] = []

    for entry in entries:
        model_a = entry.get("model_a", "")
        model_b = entry.get("model_b", "")
        delta_a = entry.get("elo_delta_a", 0.0)
        delta_b = entry.get("elo_delta_b", 0.0)

        if model_a not in current_elo:
            current_elo[model_a] = _INITIAL_ELO
        if model_b not in current_elo:
            current_elo[model_b] = _INITIAL_ELO

        current_elo[model_a] += delta_a
        current_elo[model_b] += delta_b

        snapshots.append(dict(current_elo))

    if leaderboard:
        top_names = [e.get("name", "") for e in leaderboard[:top_n]]
        top_names = [n for n in top_names if n in current_elo]
    else:
        sorted_models = sorted(current_elo.items(), key=lambda x: x[1], reverse=True)
        top_names = [name for name, _ in sorted_models[:top_n]]

    if not top_names:
        return {}

    result: Dict[str, List[float]] = {}
    for name in top_names:
        timeline = [_INITIAL_ELO]
        for snap in snapshots:
            timeline.append(snap.get(name, timeline[-1]))
        result[name] = timeline

    return result


def _read_log(log_path: Path) -> List[Dict[str, Any]]:
    """Read and parse a JSONL match log, skipping malformed lines."""
    if not log_path.exists():
        return []

    entries: List[Dict[str, Any]] = []
    with open(log_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in %s", line_num, log_path)
    return entries
