from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

from keisei.evaluation.opponents.elo_registry import EloRegistry


@dataclass
class OpponentEntry:
    """Represents a stored opponent checkpoint."""

    path: Path


class OpponentPool:
    """Manage a pool of opponent checkpoints and their Elo ratings."""

    def __init__(
        self, pool_size: int = 5, elo_registry_path: Optional[str] = None
    ) -> None:
        # Validate pool size
        if pool_size <= 0:
            raise ValueError(f"Pool size must be positive, got: {pool_size}")
        if pool_size > 1000:  # Reasonable upper limit
            raise ValueError(f"Pool size too large (>1000), got: {pool_size}")

        self.pool_size = pool_size
        self._entries: Deque[Path] = deque(maxlen=pool_size)
        self.elo_registry: Optional[EloRegistry] = None
        if elo_registry_path:
            self.elo_registry = EloRegistry(Path(elo_registry_path))

    # Management -----------------------------------------------------
    def add_checkpoint(self, path: Path | str) -> None:
        """Add a checkpoint to the pool, evicting the oldest if full."""
        p = Path(path)

        # Validate checkpoint file exists
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {p}")

        # Validate checkpoint file is readable
        if not p.is_file():
            raise ValueError(f"Checkpoint path is not a file: {p}")

        self._entries.append(p)
        if self.elo_registry:
            # Ensure rating entry exists
            self.elo_registry.get_rating(p.name)
            self.elo_registry.save()

    def get_all(self) -> Iterable[Path]:
        """Return all checkpoints in the pool."""
        return list(self._entries)

    def known_paths(self) -> set[Path]:
        """Return the set of resolved paths currently in the pool."""
        return {p.resolve() for p in self._entries}

    def scan_directory(
        self, directory: Path | str, pattern: str = "checkpoint_ts*.pth"
    ) -> int:
        """Scan a directory for checkpoint files and add them to the pool.

        Args:
            directory: Directory to scan for checkpoints
            pattern: Glob pattern for checkpoint files

        Returns:
            Number of new checkpoints added
        """
        d = Path(directory)
        if not d.exists():
            logger.warning("Scan directory does not exist: %s", d)
            return 0

        def _safe_mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        matches = sorted(d.glob(pattern), key=_safe_mtime)
        already_known = self.known_paths()
        added = 0
        for match in matches:
            if match.resolve() not in already_known:
                try:
                    self.add_checkpoint(match)
                except (FileNotFoundError, ValueError) as e:
                    logger.debug("Skipping vanished checkpoint %s: %s", match, e)
                    continue
                already_known.add(match.resolve())
                added += 1

        logger.info(
            "Scanned %s: found %d checkpoints, added %d new",
            d,
            len(matches),
            added,
        )
        return added

    # Selection ------------------------------------------------------
    def sample(self) -> Optional[Path]:
        """Return a random opponent checkpoint from the pool."""
        if not self._entries:
            return None
        return random.choice(list(self._entries))

    def champion(self) -> Optional[Path]:
        """Return the checkpoint with the highest Elo rating."""
        if not self._entries:
            return None
        if not self.elo_registry:
            return self.sample()
        return max(self._entries, key=lambda p: self.elo_registry.get_rating(p.name))

    # Elo updates ----------------------------------------------------
    def update_ratings(
        self,
        agent_id: str,
        opponent_id: str,
        results: Sequence[str],
    ) -> None:
        """Update Elo ratings for a match and persist."""
        if not self.elo_registry:
            return
        self.elo_registry.update_ratings(agent_id, opponent_id, list(results))
        self.elo_registry.save()
