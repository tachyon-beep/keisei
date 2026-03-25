"""
Append-only JSONL registry for lineage events.

Follows the same persistence pattern as ``keisei.evaluation.opponents.elo_registry``
but uses append-only JSONL rather than overwrite-JSON so that the full event
history is always preserved and replayable.

Resilience rules:
    - Blank lines are silently skipped on load.
    - Corrupt (non-JSON) lines are logged and skipped.
    - Duplicate event IDs are deduplicated on load (first occurrence wins).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from keisei.lineage.event_schema import LineageEvent, validate_event

logger = logging.getLogger(__name__)


class LineageRegistry:
    """Append-only JSONL persistence for lineage events.

    Parameters
    ----------
    file_path:
        Path to the ``.jsonl`` event log.  Parent directories are created
        automatically on first write.  If the file exists, events are loaded
        on construction.
    """

    def __init__(self, file_path: Path) -> None:
        self._file_path = Path(file_path)
        self._events: List[LineageEvent] = []
        self._seen_ids: set[str] = set()
        self._load()

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def event_count(self) -> int:
        """Number of events currently loaded."""
        return len(self._events)

    @property
    def next_sequence_number(self) -> int:
        """Next monotonic sequence number (= current event count)."""
        return len(self._events)

    # -----------------------------------------------------------------
    # Write API
    # -----------------------------------------------------------------

    def append(self, event: Dict) -> bool:  # type: ignore[type-arg]
        """Validate and append *event* to the JSONL log.

        Returns ``True`` if the event was written, ``False`` if it was a
        duplicate (same ``event_id``).

        Raises
        ------
        ValueError
            If the event fails validation.
        """
        errors = validate_event(event)
        if errors:
            raise ValueError(
                f"Invalid lineage event: {'; '.join(errors)}"
            )

        event_id: str = event["event_id"]
        if event_id in self._seen_ids:
            logger.debug("Duplicate event_id %r skipped", event_id)
            return False

        # Ensure parent directory exists
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(event, separators=(",", ":"))
        with open(self._file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        self._events.append(event)  # type: ignore[arg-type]
        self._seen_ids.add(event_id)
        return True

    # -----------------------------------------------------------------
    # Read API
    # -----------------------------------------------------------------

    def load_all(self) -> List[LineageEvent]:
        """Return a defensive copy of all loaded events."""
        return list(self._events)

    def filter_by_type(self, event_type: str) -> List[LineageEvent]:
        """Return events matching *event_type*."""
        return [e for e in self._events if e["event_type"] == event_type]

    def filter_by_run(self, run_name: str) -> List[LineageEvent]:
        """Return events matching *run_name*."""
        return [e for e in self._events if e["run_name"] == run_name]

    def filter_by_model(self, model_id: str) -> List[LineageEvent]:
        """Return events matching *model_id*."""
        return [e for e in self._events if e["model_id"] == model_id]

    def get_latest_event(self) -> Optional[LineageEvent]:
        """Return the most recently appended event, or ``None``."""
        return self._events[-1] if self._events else None

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _load(self) -> None:
        """Load existing events from the JSONL file, if it exists."""
        if not self._file_path.exists():
            return

        loaded = 0
        skipped = 0
        with open(self._file_path, "r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue  # blank line — skip silently

                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Corrupt JSONL line %d in %s: %s",
                        line_num,
                        self._file_path,
                        exc,
                    )
                    skipped += 1
                    continue

                # Validate structure
                errors = validate_event(data)
                if errors:
                    logger.warning(
                        "Invalid event on line %d in %s: %s",
                        line_num,
                        self._file_path,
                        "; ".join(errors),
                    )
                    skipped += 1
                    continue

                # Deduplicate by event_id
                eid = data["event_id"]
                if eid in self._seen_ids:
                    logger.debug(
                        "Duplicate event_id %r on line %d — skipped", eid, line_num
                    )
                    continue

                self._events.append(data)  # type: ignore[arg-type]
                self._seen_ids.add(eid)
                loaded += 1

        if loaded or skipped:
            logger.info(
                "Loaded %d lineage events from %s (%d skipped)",
                loaded,
                self._file_path,
                skipped,
            )
