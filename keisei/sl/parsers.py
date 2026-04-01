"""Game record parsers for supervised learning data."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class GameOutcome(Enum):
    WIN_BLACK = "win_black"
    WIN_WHITE = "win_white"
    DRAW = "draw"


@dataclass
class ParsedMove:
    move_usi: str
    sfen_before: str = ""


@dataclass
class GameRecord:
    moves: list[ParsedMove]
    outcome: GameOutcome
    metadata: dict[str, str] = field(default_factory=dict)


class GameParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> Iterator[GameRecord]: ...

    @abstractmethod
    def supported_extensions(self) -> set[str]: ...


class SFENParser(GameParser):
    """Simple SFEN-based game record parser.

    Format: blocks separated by blank lines.
    First line: key:value metadata (at minimum: result:win_black|win_white|draw)
    Second line: starting position ("startpos" or SFEN string)
    Remaining lines: one USI move per line.
    """

    def supported_extensions(self) -> set[str]:
        return {".sfen"}

    def parse(self, path: Path) -> Iterator[GameRecord]:
        text = path.read_text()
        blocks = text.strip().split("\n\n")

        for block in blocks:
            lines = [
                line.strip() for line in block.strip().split("\n") if line.strip()
            ]
            if len(lines) < 2:
                continue

            # Parse metadata from first line(s)
            metadata: dict[str, str] = {}
            move_start = 0
            for i, line in enumerate(lines):
                if ":" in line and not any(
                    c.isdigit() for c in line.split(":")[0]
                ):
                    key, _, val = line.partition(":")
                    metadata[key.strip()] = val.strip()
                    move_start = i + 1
                else:
                    break

            # Parse outcome
            result_str = metadata.get("result", "")
            try:
                outcome = GameOutcome(result_str)
            except ValueError:
                continue  # skip games with unknown outcome

            # Skip position line (startpos or SFEN)
            if move_start < len(lines):
                move_start += 1

            # Parse moves
            moves = []
            for line in lines[move_start:]:
                moves.append(ParsedMove(move_usi=line))

            if moves:
                yield GameRecord(moves=moves, outcome=outcome, metadata=metadata)
