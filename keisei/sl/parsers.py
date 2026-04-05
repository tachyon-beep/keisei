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


@dataclass
class GameFilter:
    """Filter for game quality before SL encoding."""

    min_ply: int = 40
    min_rating: int | None = None

    def accepts(self, record: GameRecord) -> bool:
        if len(record.moves) < self.min_ply:
            return False
        if self.min_rating is not None:
            for key in ("rating", "black_rating", "white_rating"):
                rating_str = record.metadata.get(key, "")
                if rating_str.isdigit() and int(rating_str) < self.min_rating:
                    return False
        return True


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
        text = text.replace("\r\n", "\n").replace("\r", "\n")
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


class CSAParser(GameParser):
    """Parser for Computer Shogi Association (CSA) game record format.

    Handles V2.2 format used by Floodgate and other servers.
    Converts CSA move notation to USI notation.
    """

    # CSA row (1-9) to USI rank letter
    _ROW_TO_RANK = {
        1: "a", 2: "b", 3: "c", 4: "d", 5: "e",
        6: "f", 7: "g", 8: "h", 9: "i",
    }
    # CSA piece names to USI piece names (for drops)
    _PIECE_TO_USI = {
        "FU": "P", "KY": "L", "KE": "N", "GI": "S",
        "KI": "G", "KA": "B", "HI": "R",
        "TO": "P", "NY": "L", "NK": "N", "NG": "S",
        "UM": "B", "RY": "R", "OU": "K",
    }
    # Promoted CSA pieces
    _PROMOTED = {"TO", "NY", "NK", "NG", "UM", "RY"}

    def supported_extensions(self) -> set[str]:
        return {".csa"}

    def _csa_move_to_usi(
        self, csa_move: str, board: dict[tuple[int, int], str]
    ) -> str:
        """Convert CSA move like '+7776FU' to USI move like '7g7f'.

        CSA format: [+-]<from_col><from_row><to_col><to_row><piece>
        If from is "00", it's a drop.

        `board` tracks piece names at each (col, row) position for promotion
        detection. Updated in-place by the caller after each move.
        """
        body = csa_move[1:]

        from_col = int(body[0])
        from_row = int(body[1])
        to_col = int(body[2])
        to_row = int(body[3])
        piece = body[4:]  # piece name at DESTINATION (post-move)

        if from_col == 0 and from_row == 0:
            # Drop move: "0055FU" -> "P*5e"
            usi_piece = self._PIECE_TO_USI.get(piece, piece)
            to_file = str(to_col)
            to_rank = self._ROW_TO_RANK[to_row]
            return f"{usi_piece}*{to_file}{to_rank}"

        # Board move
        from_file = str(from_col)
        from_rank = self._ROW_TO_RANK[from_row]
        to_file = str(to_col)
        to_rank = self._ROW_TO_RANK[to_row]

        usi = f"{from_file}{from_rank}{to_file}{to_rank}"

        # Promotion detection: compare piece at source (before move) with
        # piece at destination (after move). If the destination piece is a
        # promoted type but the source piece was not, promotion happened.
        source_piece = board.get((from_col, from_row), "")
        if piece in self._PROMOTED and source_piece not in self._PROMOTED:
            usi += "+"

        return usi

    @staticmethod
    def _parse_board_from_p_lines(
        p_lines: list[str],
    ) -> dict[tuple[int, int], str]:
        """Parse CSA P1-P9 position lines into a (col, row) -> piece_name dict."""
        board: dict[tuple[int, int], str] = {}
        for line in p_lines:
            if not line.startswith("P") or len(line) < 3:
                continue
            row_char = line[1]
            if not row_char.isdigit():
                continue
            row = int(row_char)
            # Each position is 3 chars: " * " (empty) or "+FU" / "-FU"
            content = line[2:]
            for col_idx in range(9):
                start = col_idx * 3
                if start + 3 > len(content):
                    break
                cell = content[start : start + 3]
                if cell.strip() == "*" or cell.strip() == "":
                    continue
                # cell is like "+FU" or "-KY"
                piece_name = cell[1:3] if len(cell) >= 3 else cell
                actual_col = 9 - col_idx  # CSA columns are 9..1 left-to-right
                board[(actual_col, row)] = piece_name
        return board

    def parse(self, path: Path) -> Iterator[GameRecord]:
        # Try UTF-8 first, fall back to Shift-JIS for older Floodgate files.
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                import chardet
                raw = path.read_bytes()
                detected = chardet.detect(raw)
                encoding = detected.get("encoding", "shift_jis") or "shift_jis"
                text = raw.decode(encoding, errors="replace")
                logger.info("Decoded %s as %s (confidence %.0f%%)",
                            path.name, encoding, (detected.get("confidence", 0) or 0) * 100)
            except ImportError:
                text = path.read_text(encoding="shift_jis", errors="replace")
                logger.warning("Non-UTF-8 file %s decoded as Shift-JIS (chardet not available)", path.name)

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split on '/' separator lines for multi-game archives
        blocks = text.split("\n/\n")
        for block_i, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue
            try:
                record = self._parse_single_game(block)
            except Exception:
                logger.exception(
                    "Failed to parse CSA game block %d in %s — skipping",
                    block_i, path.name,
                )
                continue
            if record is not None:
                yield record

    def _parse_single_game(self, text: str) -> GameRecord | None:
        """Parse a single CSA game block into a GameRecord."""
        lines = text.split("\n")

        metadata: dict[str, str] = {}
        moves: list[ParsedMove] = []
        last_mover: str = "+"
        result_line: str = ""
        p_lines: list[str] = []

        # First pass: collect P-lines for board state initialization
        for line in lines:
            line_stripped = line.strip()
            if (
                line_stripped.startswith("P")
                and len(line_stripped) > 2
                and line_stripped[1].isdigit()
            ):
                p_lines.append(line_stripped)

        # Initialize board state from position definition
        board = self._parse_board_from_p_lines(p_lines) if p_lines else {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith("'"):
                continue  # skip empty lines and comments
            if line.startswith("V"):
                continue  # version
            if line.startswith("N+"):
                metadata["player_black"] = line[2:]
            elif line.startswith("N-"):
                metadata["player_white"] = line[2:]
            elif line.startswith("$"):
                key, _, val = line[1:].partition(":")
                metadata[key.lower()] = val.strip()
            elif line.startswith("P"):
                continue  # position definition lines (already parsed)
            elif line == "+" or line == "-":
                continue  # side to move indicator
            elif line.startswith("+") or line.startswith("-"):
                if "%" in line:
                    # Embedded resign/result (e.g., "+%TORYO"): do NOT update last_mover.
                    # The side that writes %TORYO is the resigning side (the side-to-move),
                    # NOT the side that played the last board move. last_mover must remain
                    # as the previous move's player so the outcome logic correctly awards
                    # the win to last_mover (the opponent of the resigner).
                    result_line = line[1:]
                else:
                    body = line[1:]
                    # CSA move must be at least 5 chars: 4 digits + piece name
                    if len(body) < 5:
                        logger.warning(
                            "Skipping malformed CSA move (too short): %r", line,
                        )
                        return None
                    last_mover = line[0]
                    usi_move = self._csa_move_to_usi(line, board)
                    moves.append(ParsedMove(move_usi=usi_move))

                    # Update board state
                    from_col, from_row = int(body[0]), int(body[1])
                    to_col, to_row = int(body[2]), int(body[3])
                    piece = body[4:]
                    if from_col != 0 or from_row != 0:
                        board.pop((from_col, from_row), None)
                    board[(to_col, to_row)] = piece
            elif line.startswith("%"):
                result_line = line

        if not moves:
            return None

        # Determine outcome from result line.
        # %TORYO, %TIME_UP, %ILLEGAL_MOVE: side-to-move loses (last_mover wins).
        # %JISHOGI, %KACHI: impasse/declaration, last mover wins.
        # %SENNICHITE, %HIKIWAKE: draws.
        # %CHUDAN: interrupted game — skip (no valid outcome).
        _last_mover_wins = {"%TORYO", "%TIME_UP", "%ILLEGAL_MOVE", "%JISHOGI", "%KACHI"}
        _draws = {"%SENNICHITE", "%HIKIWAKE"}

        if result_line in _last_mover_wins:
            outcome = (
                GameOutcome.WIN_BLACK
                if last_mover == "+"
                else GameOutcome.WIN_WHITE
            )
        elif result_line in _draws:
            outcome = GameOutcome.DRAW
        elif result_line == "%CHUDAN":
            return None  # interrupted game, skip
        else:
            logger.warning("Unknown CSA result '%s', skipping game", result_line)
            return None

        return GameRecord(moves=moves, outcome=outcome, metadata=metadata)
