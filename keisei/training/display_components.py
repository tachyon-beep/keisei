"""Reusable Rich display components for the training TUI."""

from __future__ import annotations

import math
from collections import Counter, deque
from time import monotonic
from typing import Dict, List, Optional, Sequence

from rich import box
from rich.align import Align
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from keisei.shogi.shogi_core_definitions import Color
from keisei.utils import _coords_to_square_name


class HorizontalSeparator:
    """A reusable horizontal separator bar for visual organization."""

    def __init__(self, width_ratio: float = 0.9, style: str = "dim", char: str = "─"):
        """
        Args:
            width_ratio: Ratio of available width to use (0.0 to 1.0)
            style: Rich style to apply to the separator
            char: Character to use for the separator line
        """
        self.width_ratio = width_ratio
        self.style = style
        self.char = char

    def render(self, available_width: int = 50) -> RenderableType:
        """
        Render a horizontal separator line.

        Args:
            available_width: The available width for the separator
        """
        separator_width = max(1, int(available_width * self.width_ratio))
        separator_line = self.char * separator_width
        text = Text(separator_line, style=self.style)
        return Align.center(text)


class ShogiBoard:
    """Rich-ified representation of the current Shogi board state using a grid of Panels."""

    def __init__(
        self,
        use_unicode: bool = True,
        cell_width: int = 5,
        cell_height: int = 3,
    ) -> None:
        self.use_unicode = use_unicode
        self.cell_width = cell_width
        self.cell_height = cell_height

    def _piece_to_symbol(self, piece) -> str:
        """
        Turn your internal piece-object into a single-string symbol.
        This version is robust against case-sensitivity and unknown piece types.
        """
        if not piece:
            return "・"

        # This handles your non-unicode fallback if you have it
        if not self.use_unicode:
            return getattr(piece, "symbol", lambda: "?")()

        # --- Definitive Unicode Symbol Lookup ---
        symbols = {
            "PAWN": "歩",
            "LANCE": "香",
            "KNIGHT": "桂",
            "SILVER": "銀",
            "GOLD": "金",
            "BISHOP": "角",
            "ROOK": "飛",
            "KING": "王",
            "PROMOTED_PAWN": "と",
            "PROMOTED_LANCE": "杏",
            "PROMOTED_KNIGHT": "圭",
            "PROMOTED_SILVER": "全",
            "PROMOTED_BISHOP": "馬",
            "PROMOTED_ROOK": "龍",
        }

        try:
            # 1. Get the piece type name safely.
            piece_type_name_attr = getattr(piece, "type", None)
            if not piece_type_name_attr:
                return "?"  # The object has no 'type' attribute

            # 2. Make the lookup case-insensitive.
            lookup_key = str(piece_type_name_attr.name).upper()

            # 3. Return the symbol, or the first letter of the key as a fallback for debugging.
            return symbols.get(lookup_key, lookup_key[0])

        except (AttributeError, IndexError):
            # This is a final catch-all if the piece object is malformed.
            from rich.console import Console

            console = Console(stderr=True, style="bold red")
            console.print(
                f"[ShogiBoard] Error: Invalid object passed as a piece: {piece}"
            )
            return "!"

    def _colorize(self, symbol: str, piece) -> Text:
        """Apply per-piece colouring and bold styling for heavier appearance."""
        if not piece:
            return Text(symbol)
        if piece.color == Color.BLACK:
            return Text(symbol, style="bold bright_white")
        return Text(symbol, style="bold bright_blue")

    def _get_shogi_notation(self, row: int, col: int) -> str:
        return _coords_to_square_name(row, col)

    def _create_cell_panel(
        self, piece, r_idx: int, c_idx: int, hot_squares: Optional[set]
    ) -> Panel:
        """Creates a single styled Panel for a board square."""
        light_bg_color = "#EEC28A"
        dark_bg_color = "#C19A55"
        light_bg = Style(bgcolor=light_bg_color)
        dark_bg = Style(bgcolor=dark_bg_color)

        is_light = (r_idx + c_idx) % 2 == 0
        bg_style = light_bg if is_light else dark_bg

        symbol = self._piece_to_symbol(piece)
        if piece:
            styled_text = self._colorize(symbol, piece)
        else:
            dot_color = dark_bg_color if is_light else light_bg_color
            styled_text = Text(symbol, style=Style(color=dot_color))

        centered_content = Align.center(styled_text, vertical="middle")

        # Apply the background style to the Panel itself, not the text
        cell_panel = Panel(
            centered_content,
            box=box.SIMPLE,
            width=self.cell_width,
            height=self.cell_height,
            style=bg_style,
        )

        board_col = 8 - c_idx
        sq_name = self._get_shogi_notation(r_idx, board_col)
        if hot_squares and sq_name in hot_squares:
            cell_panel.border_style = "bold dark_red"

        return cell_panel

    def _generate_board_grid(
        self, board_state, hot_squares: Optional[set] = None
    ) -> Table:
        """Create a 9x9 grid of Panels representing the board."""
        board_grid = Table.grid(expand=False)
        for _ in range(9):
            board_grid.add_column()

        for r_idx, row in enumerate(board_state.board):
            row_renderables = [
                self._create_cell_panel(piece, r_idx, c_idx, hot_squares)
                for c_idx, piece in enumerate(reversed(row))
            ]
            board_grid.add_row(*row_renderables)

        return board_grid

    def render(
        self, board_state=None, highlight_squares: Optional[set] = None, **_kwargs
    ) -> RenderableType:
        """Returns a final Panel containing the board grid and coordinate labels."""
        if not board_state:
            return Panel(
                Text("No active game", justify="center"),
                title="Main Board",
                border_style="blue",
            )

        # 1. Generate the core 9x9 board of panels.
        board_grid = self._generate_board_grid(
            board_state, hot_squares=highlight_squares
        )

        # 2. Create the outer layout to hold the board and coordinates.
        layout_grid = Table.grid(expand=False, padding=0)
        layout_grid.add_column(width=2, justify="center")  # For rank labels (a-i)
        layout_grid.add_column()  # For the board itself

        # 3. Add the top file labels (9 to 1) in their own sub-grid.
        file_labels = [
            Text(str(n), justify="center", style="bold") for n in range(9, 0, -1)
        ]
        top_label_grid = Table.grid(expand=False)
        for _ in range(9):
            top_label_grid.add_column(width=self.cell_width)
        top_label_grid.add_row(*file_labels)

        # Add an empty top-left cell, then the file labels.
        layout_grid.add_row(Text(" "), top_label_grid)

        # 4. Create rank labels and combine them with the board grid.
        # This is the corrected logic that fixes the "no attribute 'renderable'" error.
        rank_labels = Group(
            *(
                Align.center(
                    Text(chr(ord("a") + i), style="bold"),
                    vertical="middle",
                    height=self.cell_height,
                )
                for i in range(9)
            )
        )
        layout_grid.add_row(rank_labels, board_grid)

        return Panel(Align.center(layout_grid), title="Main Board", border_style="blue")


class RecentMovesPanel:
    def __init__(
        self, max_moves: int = 20, newest_on_top: bool = True, flash_ms: int = 0
    ):
        self.max_moves = max_moves
        self.newest_on_top = newest_on_top
        self.flash_ms = flash_ms
        self._last_move: str | None = None
        self._flash_deadline: float = 0.0

    def _stylise(self, move: str) -> Text:

        # The flashing logic for the newest move
        is_newest = self.newest_on_top and move == self._last_move
        is_flashing = is_newest and monotonic() < self._flash_deadline

        style = "bold green" if is_flashing else ""
        return Text(f" {move}", style=style)

    def render(
        self,
        move_strings: Optional[List[str]] = None,
        ply_per_sec: float = 0.0,
        **_kwargs,  # Absorb unused kwargs like available_height
    ) -> RenderableType:
        moves = move_strings or []

        # Update the flash timer if a new move has arrived
        if moves and moves[-1] != self._last_move:
            self._last_move = moves[-1]
            if self.flash_ms > 0:
                self._flash_deadline = (
                    __import__("time").monotonic() + self.flash_ms / 1000
                )

        # 1. Slice the list to the configured max_moves. No more capacity logic.
        slice_ = moves[-self.max_moves :]

        # 2. Reverse the list if needed.
        if self.newest_on_top:
            slice_.reverse()

        # 3. Create the text object from the sliced moves. No more manual padding.
        body = Text("\n").join(self._stylise(m) for m in slice_)

        # 4. Create the panel and let the Layout manager handle sizing.
        title = (
            f"Recent Moves ({len(moves)} | {ply_per_sec:.1f} ply/s)"
            if ply_per_sec
            else f"Recent Moves ({len(moves)})"
        )
        return Panel(body, title=title, border_style="yellow", expand=True)


class PieceStandPanel:
    """Renders the captured pieces (komadai) for each player."""

    def _format_hand(self, hand: Dict[str, int]) -> str:
        symbols = {
            "PAWN": "歩",
            "LANCE": "香",
            "KNIGHT": "桂",
            "SILVER": "銀",
            "GOLD": "金",
            "BISHOP": "角",
            "ROOK": "飛",
        }
        parts = [
            f"{symbols.get(getattr(k, 'name', k), '?')}x{v}"
            for k, v in hand.items()
            if v > 0
        ]
        return " ".join(parts) or ""

    def render(self, game) -> RenderableType:
        """Renders the captured pieces (komadai) for each player using a Table."""
        if not game:
            return Panel("...", title="Captured Pieces", border_style="yellow")

        sente_hand_str = self._format_hand(
            getattr(game, "hands", {}).get(Color.BLACK.value, {})
        )
        gote_hand_str = self._format_hand(
            getattr(game, "hands", {}).get(Color.WHITE.value, {})
        )

        # Use a simple Table for clean, two-column alignment.
        hand_table = Table.grid(padding=(0, 1))
        hand_table.add_column(style="bold", justify="right", width=7)  # e.g., "Sente: "
        hand_table.add_column()

        # Add rows for each player, applying color and bold styling to the pieces.
        hand_table.add_row(
            "Sente:", Text(sente_hand_str or "None", style="bold bright_red")
        )
        hand_table.add_row(
            "Gote:", Text(gote_hand_str or "None", style="bold bright_blue")
        )

        return Panel(hand_table, title="Captured Pieces", border_style="yellow")


class Sparkline:
    """Simple Unicode sparkline generator for metric trends."""

    def __init__(self, width: int = 20) -> None:
        self.width = width
        self.chars = "▁▂▃▄▅▆▇"

    def generate(
        self,
        values: Sequence[float],
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
    ) -> str:
        if not values:
            return " " * self.width

        # Filter out infinite and NaN values
        finite_values = [v for v in values if not (math.isnan(v) or math.isinf(v))]

        if not finite_values:
            return " " * self.width

        if len(finite_values) == 1:
            # Single value should fill the width with the same character
            return "▄" * self.width

        min_v = range_min if range_min is not None else min(finite_values)
        max_v = range_max if range_max is not None else max(finite_values)

        # Handle infinity in range bounds
        if math.isinf(min_v) or math.isinf(max_v):
            return " " * self.width

        clipped = [
            min(max(v if not (math.isnan(v) or math.isinf(v)) else min_v, min_v), max_v)
            for v in values
        ]
        if max_v == min_v:
            normalized = [4] * len(clipped)
        else:
            rng = max_v - min_v
            if rng == 0 or math.isnan(rng) or math.isinf(rng):
                normalized = [4] * len(clipped)
            else:
                normalized = [int((v - min_v) / rng * 6) for v in clipped]

        recent = normalized[-self.width :]
        spark = "".join(self.chars[n] for n in recent)
        if len(spark) < self.width:
            spark = "▁" * (self.width - len(spark)) + spark
        return spark


class RollingAverageCalculator:
    """Compute a rolling average and simple trend direction."""

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.values: deque[float] = deque(maxlen=window_size)

    def add_value(self, value: float) -> float:
        self.values.append(value)
        return sum(self.values) / len(self.values)

    def get_trend_direction(self) -> str:
        if len(self.values) < 2:
            return "→"
        if self.values[-1] > self.values[0]:
            return "↑"
        if self.values[-1] < self.values[0]:
            return "↓"
        return "→"


class MultiMetricSparkline:
    """Render multiple metric sparklines in one panel."""

    def __init__(self, width: int, metrics: List[str]) -> None:
        self.width = width
        self.metrics = metrics
        self.data: Dict[str, deque] = {m: deque(maxlen=width) for m in metrics}
        self.spark = Sparkline(width=width)

    def add_data_point(self, metric_name: str, value: float) -> None:
        if metric_name in self.data:
            self.data[metric_name].append(value)

    def render_with_trendlines(self) -> RenderableType:
        lines: List[str] = []
        for name in self.metrics:
            values = self.data.get(name, deque())
            spark = self.spark.generate(list(values))
            lines.append(f"{name}: {spark}")
        return Text("\n".join(lines), style="cyan")


class GameStatisticsPanel:
    """Renders detailed statistics about the current game and session."""

    def _calculate_material(self, game_object, color):
        piece_values = {
            "PAWN": 1,
            "LANCE": 3,
            "KNIGHT": 3,
            "SILVER": 5,
            "GOLD": 6,
            "BISHOP": 8,
            "ROOK": 10,
        }
        total_value = 0
        for row in game_object.board:
            for piece in row:
                if piece and piece.color == color:
                    key = piece.type.name.replace("PROMOTED_", "")
                    total_value += piece_values.get(key, 0)
        return total_value

    def _format_hand(self, hand: Dict[str, int]) -> str:
        symbols = {
            "PAWN": "歩",
            "LANCE": "香",
            "KNIGHT": "桂",
            "SILVER": "銀",
            "GOLD": "金",
            "BISHOP": "角",
            "ROOK": "飛",
        }
        parts = [
            f"{symbols.get(getattr(k, 'name', k), '?')}x{v}"
            for k, v in hand.items()
            if v > 0
        ]
        return " ".join(parts) or ""

    def _format_opening_name(self, move_str: str) -> str:
        """Translates a raw shogi move string into a more readable format."""
        if not move_str or len(move_str) < 2:
            return move_str  # Return as-is if invalid or empty

        piece_map = {
            "P": "Pawn",
            "L": "Lance",
            "N": "Knight",
            "S": "Silver",
            "G": "Gold",
            "B": "Bishop",
            "R": "Rook",
            "K": "King",
        }

        # Case 1: It's a drop move (e.g., 'P*2c')
        if "*" in move_str:
            piece_char = move_str[0].upper()
            piece_name = piece_map.get(piece_char, "Piece")
            destination = move_str[2:]
            return f"{piece_name} drop to {destination}"

        # Case 2: It's a regular board move (e.g., '7g7f' or '2c3d+')
        promotion_text = ""
        if move_str.endswith("+"):
            promotion_text = " with promotion"
            move_str = move_str[:-1]  # Remove the '+' for parsing

        if len(move_str) == 4:
            start_sq = move_str[0:2]
            end_sq = move_str[2:4]
            # We can't know the piece name from '7g7f', so we describe the action.
            return f"Move from {start_sq} to {end_sq}{promotion_text}"

        # Fallback for any other format
        return move_str

    def render(
        self,
        game,
        move_history: Optional[List[str]] = None,
        metrics_manager=None,
        sente_best_capture: Optional[str] = None,
        gote_best_capture: Optional[str] = None,
        sente_captures: int = 0,
        gote_captures: int = 0,
        sente_drops: int = 0,
        gote_drops: int = 0,
        sente_promos: int = 0,
        gote_promos: int = 0,
    ) -> RenderableType:
        if not game or not move_history or not metrics_manager:
            return Panel(
                "Waiting for game to start...",
                title="Game Statistics",
                border_style="green",
            )

        # --- Calculate In-Game Stats ---
        sente_material = self._calculate_material(game, Color.BLACK)
        gote_material = self._calculate_material(game, Color.WHITE)
        material_adv = sente_material - gote_material

        is_in_check = False  # Default to False
        # Check if the game object has the necessary attributes before calling them
        if hasattr(game, "current_player") and hasattr(game, "is_in_check"):
            # Call the method with the required 'color' argument
            is_in_check = game.is_in_check(game.current_player)

        hot_squares = metrics_manager.get_hot_squares(top_n=3)

        # --- Get Session Stats (now pre-formatted) ---
        sente_openings = metrics_manager.sente_opening_history
        gote_openings = metrics_manager.gote_opening_history
        fav_sente_opening_raw = (
            Counter(sente_openings).most_common(1)[0][0] if sente_openings else "N/A"
        )
        fav_gote_opening_raw = (
            Counter(gote_openings).most_common(1)[0][0] if gote_openings else "N/A"
        )

        # Use the new helper to format the names before displaying them
        fav_sente_opening_formatted = self._format_opening_name(fav_sente_opening_raw)
        fav_gote_opening_formatted = self._format_opening_name(fav_gote_opening_raw)

        # --- Activity Counters are provided by StepManager ---

        # --- Moves Since Last Capture ---
        moves_since_capture = len(move_history)
        for idx in range(len(move_history) - 1, -1, -1):
            if "captur" in move_history[idx].lower():
                moves_since_capture = len(move_history) - idx - 1
                break

        # --- King Safety ---
        sente_king_moves = game.get_king_legal_moves(Color.BLACK)
        gote_king_moves = game.get_king_legal_moves(Color.WHITE)

        # --- Create Table ---
        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()

        table.add_row("Check Status:", "[red]CHECK[/]" if is_in_check else "Clear")
        table.add_row("Material Adv:", f"{material_adv:+.1f} (Sente)")
        table.add_row(
            "King Safety:",
            f"Sente: {sente_king_moves} moves | Gote: {gote_king_moves} moves",
        )
        table.add_row(
            "Sente Activity:",
            f"{sente_captures} Captures, {sente_drops} Drops, {sente_promos} Promotions",
        )
        table.add_row(
            "Gote Activity:",
            f"{gote_captures} Captures, {gote_drops} Drops, {gote_promos} Promotions",
        )
        table.add_row("Sente Best Capture:", sente_best_capture or "-")
        table.add_row("Gote Best Capture:", gote_best_capture or "-")
        table.add_row("Moves Since Capture:", str(moves_since_capture))
        table.add_row("Hot Squares:", ", ".join(hot_squares) or "N/A")
        table.add_row("Sente's Favourite Opening:", fav_sente_opening_formatted)
        table.add_row("Gote's Favourite Opening:", fav_gote_opening_formatted)
        table.add_row(
            "",
            "",
        )
        sente_hand_str = self._format_hand(game.hands.get(Color.BLACK.value, {}))
        gote_hand_str = self._format_hand(game.hands.get(Color.WHITE.value, {}))
        table.add_row("Sente's Hand:", sente_hand_str)
        table.add_row("Gote's Hand:", gote_hand_str)

        return Panel(Group(table), title="Game Statistics", border_style="green")
