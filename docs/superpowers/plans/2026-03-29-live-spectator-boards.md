# Live Spectator Boards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live game board rendering (1 primary + 2 secondary) to the Ladder tab in the Streamlit dashboard, reading match state from the existing ladder state file.

**Architecture:** New `sfen_utils.py` module provides SFEN→board_state conversion. Four new helper functions in `streamlit_app.py` handle match selection (with manual pin override) and board rendering. The existing `render_board()` function renders all boards. Session state tracks the user's pinned match.

**Tech Stack:** Python 3.13, Streamlit, existing `render_board()` HTML table renderer

**Spec:** `docs/superpowers/specs/2026-03-29-live-spectator-boards-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `keisei/webui/sfen_utils.py` | Create | SFEN string → board_state dict conversion |
| `keisei/webui/streamlit_app.py` | Modify | Add `select_display_matches()`, `set_primary_match()`, `render_primary_board()`, `render_secondary_board()`, extend `render_ladder_tab()` |
| `tests/unit/test_sfen_utils.py` | Create | SFEN parser unit tests |
| `tests/unit/test_ladder_dashboard.py` | Modify | Add `TestSelectDisplayMatches` test class |

---

### Task 1: SFEN Parser — Core Piece Mapping Tests

**Files:**
- Create: `tests/unit/test_sfen_utils.py`
- Create: `keisei/webui/sfen_utils.py`

- [ ] **Step 1: Write the starting position test**

```python
"""Tests for SFEN-to-board_state conversion."""

import pytest

pytestmark = pytest.mark.unit


class TestSfenToBoardState:
    """Tests for sfen_to_board_state()."""

    STARTPOS = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

    def test_starting_position_dimensions(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        assert len(result["board"]) == 9
        assert all(len(row) == 9 for row in result["board"])

    def test_starting_position_current_player(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        assert result["current_player"] == "black"

    def test_starting_position_move_count(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        assert result["move_count"] == 1

    def test_starting_position_not_game_over(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        assert result["game_over"] is False
        assert result["winner"] is None

    def test_starting_position_empty_hands(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        assert result["black_hand"] == {}
        assert result["white_hand"] == {}

    def test_starting_position_black_pieces_row9(self):
        """Row 9 (index 8): LNSGKGSNL — all black pieces."""
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        row = result["board"][8]
        expected_types = [
            "lance", "knight", "silver", "gold", "king",
            "gold", "silver", "knight", "lance",
        ]
        for col, expected_type in enumerate(expected_types):
            assert row[col] is not None, f"col {col} should not be empty"
            assert row[col]["type"] == expected_type
            assert row[col]["color"] == "black"
            assert row[col]["promoted"] is False

    def test_starting_position_white_pieces_row1(self):
        """Row 1 (index 0): lnsgkgsnl — all white pieces."""
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        row = result["board"][0]
        expected_types = [
            "lance", "knight", "silver", "gold", "king",
            "gold", "silver", "knight", "lance",
        ]
        for col, expected_type in enumerate(expected_types):
            assert row[col] is not None, f"col {col} should not be empty"
            assert row[col]["type"] == expected_type
            assert row[col]["color"] == "white"
            assert row[col]["promoted"] is False

    def test_starting_position_empty_middle_rows(self):
        """Rows 4-6 (indices 3-5) should be all None."""
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        for r in range(3, 6):
            assert all(cell is None for cell in result["board"][r]), f"row {r}"

    def test_starting_position_bishops_and_rooks(self):
        """Row 2 (index 1): white bishop at col 7, rook at col 1."""
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        # White rook at row 1, col 1
        assert result["board"][1][1]["type"] == "rook"
        assert result["board"][1][1]["color"] == "white"
        # White bishop at row 1, col 7
        assert result["board"][1][7]["type"] == "bishop"
        assert result["board"][1][7]["color"] == "white"
        # Black bishop at row 7, col 1
        assert result["board"][7][1]["type"] == "bishop"
        assert result["board"][7][1]["color"] == "black"
        # Black rook at row 7, col 7
        assert result["board"][7][7]["type"] == "rook"
        assert result["board"][7][7]["color"] == "black"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sfen_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'keisei.webui.sfen_utils'`

- [ ] **Step 3: Implement the SFEN parser**

```python
"""Convert SFEN (Shogi Forsyth-Edwards Notation) strings to board_state dicts.

The output matches the shape produced by ``state_snapshot.extract_board_state()``,
making it directly consumable by ``render_board()`` in the Streamlit dashboard.
"""

from typing import Any, Dict, List, Optional

# SFEN letter → (piece type name, color)
# Uppercase = black (sente), lowercase = white (gote)
_SFEN_PIECE_MAP: Dict[str, tuple] = {
    "K": ("king", "black"),
    "R": ("rook", "black"),
    "B": ("bishop", "black"),
    "G": ("gold", "black"),
    "S": ("silver", "black"),
    "N": ("knight", "black"),
    "L": ("lance", "black"),
    "P": ("pawn", "black"),
    "k": ("king", "white"),
    "r": ("rook", "white"),
    "b": ("bishop", "white"),
    "g": ("gold", "white"),
    "s": ("silver", "white"),
    "n": ("knight", "white"),
    "l": ("lance", "white"),
    "p": ("pawn", "white"),
}

# SFEN hand letter → hand piece key name (always lowercase base type)
_SFEN_HAND_MAP: Dict[str, str] = {
    "P": "pawn",
    "L": "lance",
    "N": "knight",
    "S": "silver",
    "G": "gold",
    "B": "bishop",
    "R": "rook",
}

# Promoted type names (matching PieceType.name.lower() output)
_PROMOTED_PREFIX = "promoted_"


def sfen_to_board_state(sfen: str) -> Dict[str, Any]:
    """Parse an SFEN string into a board_state dict.

    Args:
        sfen: Standard SFEN string, e.g.
            ``"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"``

    Returns:
        Dict matching the shape of ``extract_board_state()``::

            {
                "board": [[Optional[{"type", "color", "promoted"}], ...], ...],
                "current_player": "black" | "white",
                "move_count": int,
                "game_over": False,
                "winner": None,
                "black_hand": {"pawn": 2, ...},
                "white_hand": {"knight": 1, ...},
            }

    Raises:
        ValueError: If the SFEN string is malformed.
    """
    parts = sfen.strip().split()
    if len(parts) < 3:
        raise ValueError(
            f"SFEN must have at least 3 fields (board, turn, hand), got {len(parts)}: {sfen!r}"
        )

    board_str = parts[0]
    turn = parts[1]
    hand_str = parts[2]
    move_count = int(parts[3]) if len(parts) >= 4 else 1

    board = _parse_board(board_str)
    current_player = _parse_turn(turn)
    black_hand, white_hand = _parse_hand(hand_str)

    return {
        "board": board,
        "current_player": current_player,
        "move_count": move_count,
        "game_over": False,
        "winner": None,
        "black_hand": black_hand,
        "white_hand": white_hand,
    }


def _parse_board(board_str: str) -> List[List[Optional[Dict[str, Any]]]]:
    """Parse the board portion of an SFEN string into a 9x9 grid."""
    rows = board_str.split("/")
    if len(rows) != 9:
        raise ValueError(
            f"SFEN board must have 9 rows separated by '/', got {len(rows)}"
        )

    board: List[List[Optional[Dict[str, Any]]]] = []
    for rank_idx, row_str in enumerate(rows):
        row: List[Optional[Dict[str, Any]]] = []
        promoted = False
        for ch in row_str:
            if ch == "+":
                promoted = True
                continue
            if ch.isdigit():
                if promoted:
                    raise ValueError(
                        f"'+' followed by digit '{ch}' in row {rank_idx + 1}"
                    )
                row.extend([None] * int(ch))
                continue
            if ch not in _SFEN_PIECE_MAP:
                raise ValueError(
                    f"Unknown piece character '{ch}' in row {rank_idx + 1}"
                )
            base_type, color = _SFEN_PIECE_MAP[ch]
            if promoted:
                piece_type = _PROMOTED_PREFIX + base_type
                row.append({"type": piece_type, "color": color, "promoted": True})
                promoted = False
            else:
                row.append({"type": base_type, "color": color, "promoted": False})

        if len(row) != 9:
            raise ValueError(
                f"SFEN row {rank_idx + 1} has {len(row)} squares, expected 9"
            )
        board.append(row)

    return board


def _parse_turn(turn: str) -> str:
    """Parse the turn field into 'black' or 'white'."""
    if turn == "b":
        return "black"
    elif turn == "w":
        return "white"
    else:
        raise ValueError(f"SFEN turn must be 'b' or 'w', got {turn!r}")


def _parse_hand(hand_str: str) -> tuple:
    """Parse the hand field into (black_hand, white_hand) dicts.

    Returns:
        Tuple of (black_hand, white_hand) where each is {piece_name: count}.
    """
    black_hand: Dict[str, int] = {}
    white_hand: Dict[str, int] = {}

    if hand_str == "-":
        return black_hand, white_hand

    count_str = ""
    for ch in hand_str:
        if ch.isdigit():
            count_str += ch
            continue
        count = int(count_str) if count_str else 1
        count_str = ""
        upper = ch.upper()
        if upper not in _SFEN_HAND_MAP:
            raise ValueError(f"Unknown hand piece character '{ch}'")
        piece_name = _SFEN_HAND_MAP[upper]
        if ch.isupper():
            black_hand[piece_name] = black_hand.get(piece_name, 0) + count
        else:
            white_hand[piece_name] = white_hand.get(piece_name, 0) + count

    return black_hand, white_hand
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sfen_utils.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/webui/sfen_utils.py tests/unit/test_sfen_utils.py
git commit -m "feat(webui): add SFEN-to-board_state parser for spectator boards"
```

---

### Task 2: SFEN Parser — Promotions, Hands, and Error Cases

**Files:**
- Modify: `tests/unit/test_sfen_utils.py`

- [ ] **Step 1: Write promotion and hand tests**

Add to `TestSfenToBoardState` in `tests/unit/test_sfen_utils.py`:

```python
    def test_promoted_pieces(self):
        """SFEN with +R (dragon) and +B (horse)."""
        from keisei.webui.sfen_utils import sfen_to_board_state

        sfen = "lnsgkgsnl/1+R5+b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        result = sfen_to_board_state(sfen)
        # Black promoted rook at row 1, col 1
        assert result["board"][1][1]["type"] == "promoted_rook"
        assert result["board"][1][1]["color"] == "black"
        assert result["board"][1][1]["promoted"] is True
        # White promoted bishop at row 1, col 7
        assert result["board"][1][7]["type"] == "promoted_bishop"
        assert result["board"][1][7]["color"] == "white"
        assert result["board"][1][7]["promoted"] is True

    def test_white_turn(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1"
        result = sfen_to_board_state(sfen)
        assert result["current_player"] == "white"

    def test_hand_pieces_multiple_types(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        sfen = "4k4/9/9/9/9/9/9/9/4K4 b 2P1L3Nrb 10"
        result = sfen_to_board_state(sfen)
        assert result["black_hand"] == {"pawn": 2, "lance": 1, "knight": 3}
        assert result["white_hand"] == {"rook": 1, "bishop": 1}

    def test_hand_single_piece_no_count(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        sfen = "4k4/9/9/9/9/9/9/9/4K4 b Pp 1"
        result = sfen_to_board_state(sfen)
        assert result["black_hand"] == {"pawn": 1}
        assert result["white_hand"] == {"pawn": 1}

    def test_empty_hand_dash(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        sfen = "4k4/9/9/9/9/9/9/9/4K4 b - 1"
        result = sfen_to_board_state(sfen)
        assert result["black_hand"] == {}
        assert result["white_hand"] == {}

    def test_move_count_parsed(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        sfen = "4k4/9/9/9/9/9/9/9/4K4 b - 42"
        result = sfen_to_board_state(sfen)
        assert result["move_count"] == 42
```

Add a new class for error cases:

```python
class TestSfenErrors:
    """Tests for SFEN parser error handling."""

    def test_too_few_fields(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        with pytest.raises(ValueError, match="at least 3 fields"):
            sfen_to_board_state("lnsgkgsnl/9/9/9/9/9/9/9/LNSGKGSNL b")

    def test_wrong_row_count(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        with pytest.raises(ValueError, match="9 rows"):
            sfen_to_board_state("4k4/9/9/9/9/9/9/4K4 b - 1")

    def test_row_too_short(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        with pytest.raises(ValueError, match="8 squares"):
            sfen_to_board_state("4k3/9/9/9/9/9/9/9/4K4 b - 1")

    def test_unknown_piece_char(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        with pytest.raises(ValueError, match="Unknown piece character 'x'"):
            sfen_to_board_state("4k3x/9/9/9/9/9/9/9/4K4 b - 1")

    def test_invalid_turn(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        with pytest.raises(ValueError, match="turn must be 'b' or 'w'"):
            sfen_to_board_state("4k4/9/9/9/9/9/9/9/4K4 x - 1")

    def test_unknown_hand_piece(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        with pytest.raises(ValueError, match="Unknown hand piece"):
            sfen_to_board_state("4k4/9/9/9/9/9/9/9/4K4 b X 1")
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sfen_utils.py -v`
Expected: All 22 tests PASS (10 from Task 1 + 12 new)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_sfen_utils.py
git commit -m "test(webui): add SFEN promotion, hand, and error case tests"
```

---

### Task 3: Match Selection Logic — Tests

**Files:**
- Modify: `tests/unit/test_ladder_dashboard.py`

- [ ] **Step 1: Write tests for select_display_matches**

Add to `tests/unit/test_ladder_dashboard.py`:

```python
class TestSelectDisplayMatches:
    """Tests for select_display_matches helper."""

    def _make_match(self, match_id, elo_a, elo_b, spectated=True, sfen="4k4/9/9/9/9/9/9/9/4K4 b - 1"):
        return {
            "match_id": match_id,
            "model_a": {"name": f"model_a_{match_id}", "elo": elo_a},
            "model_b": {"name": f"model_b_{match_id}", "elo": elo_b},
            "spectated": spectated,
            "sfen": sfen,
            "move_count": 10,
            "move_log": [],
            "status": "in_progress",
            "slot": 0,
        }

    def test_empty_matches(self):
        from keisei.webui.streamlit_app import select_display_matches

        primary, secondaries, pin = select_display_matches([], None)
        assert primary is None
        assert secondaries == []
        assert pin is None

    def test_single_match_becomes_primary(self):
        from keisei.webui.streamlit_app import select_display_matches

        matches = [self._make_match("m1", 1600, 1500)]
        primary, secondaries, pin = select_display_matches(matches, None)
        assert primary["match_id"] == "m1"
        assert secondaries == []
        assert pin is None

    def test_highest_combined_elo_is_primary(self):
        from keisei.webui.streamlit_app import select_display_matches

        matches = [
            self._make_match("low", 1400, 1400),
            self._make_match("high", 1700, 1600),
            self._make_match("mid", 1500, 1500),
        ]
        primary, secondaries, pin = select_display_matches(matches, None)
        assert primary["match_id"] == "high"
        assert len(secondaries) == 2
        assert secondaries[0]["match_id"] == "mid"
        assert secondaries[1]["match_id"] == "low"

    def test_pinned_match_becomes_primary(self):
        from keisei.webui.streamlit_app import select_display_matches

        matches = [
            self._make_match("high", 1700, 1600),
            self._make_match("low", 1400, 1400),
        ]
        primary, secondaries, pin = select_display_matches(matches, "low")
        assert primary["match_id"] == "low"
        assert secondaries[0]["match_id"] == "high"
        assert pin == "low"

    def test_stale_pin_clears(self):
        from keisei.webui.streamlit_app import select_display_matches

        matches = [self._make_match("m1", 1600, 1500)]
        primary, secondaries, pin = select_display_matches(matches, "gone")
        assert primary["match_id"] == "m1"
        assert pin is None

    def test_non_spectated_filtered(self):
        from keisei.webui.streamlit_app import select_display_matches

        matches = [
            self._make_match("spectated", 1500, 1500, spectated=True),
            self._make_match("background", 1700, 1700, spectated=False),
        ]
        primary, secondaries, pin = select_display_matches(matches, None)
        assert primary["match_id"] == "spectated"
        assert secondaries == []

    def test_missing_sfen_filtered(self):
        from keisei.webui.streamlit_app import select_display_matches

        matches = [
            self._make_match("has_sfen", 1500, 1500),
            self._make_match("no_sfen", 1700, 1700, sfen=None),
        ]
        primary, secondaries, pin = select_display_matches(matches, None)
        assert primary["match_id"] == "has_sfen"
        assert secondaries == []

    def test_two_matches(self):
        from keisei.webui.streamlit_app import select_display_matches

        matches = [
            self._make_match("m1", 1600, 1500),
            self._make_match("m2", 1400, 1400),
        ]
        primary, secondaries, pin = select_display_matches(matches, None)
        assert primary["match_id"] == "m1"
        assert len(secondaries) == 1
        assert secondaries[0]["match_id"] == "m2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_ladder_dashboard.py::TestSelectDisplayMatches -v`
Expected: FAIL — `ImportError: cannot import name 'select_display_matches'`

- [ ] **Step 3: Commit tests**

```bash
git add tests/unit/test_ladder_dashboard.py
git commit -m "test(webui): add select_display_matches tests"
```

---

### Task 4: Match Selection Logic — Implementation

**Files:**
- Modify: `keisei/webui/streamlit_app.py` (add after `format_elo_arrow` function, before `render_ladder_tab`)

- [ ] **Step 1: Implement select_display_matches and set_primary_match**

Add these functions to `keisei/webui/streamlit_app.py` between `format_elo_arrow()` (line ~1110) and `render_ladder_tab()` (line ~1113):

```python
def select_display_matches(
    matches: List[Dict[str, Any]],
    pinned_match_id: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """Select primary and secondary matches for the spectator board display.

    Returns:
        (primary, secondaries, effective_pin) where effective_pin is the
        pinned_match_id if still valid, or None if cleared.
    """
    # Filter to spectated matches with valid SFEN
    eligible = [
        m
        for m in matches
        if m.get("spectated") and m.get("sfen")
    ]

    if not eligible:
        return None, [], None

    # Sort by combined Elo descending
    def combined_elo(m: Dict[str, Any]) -> float:
        a = m.get("model_a", {})
        b = m.get("model_b", {})
        elo_a = a.get("elo", 1500.0) if isinstance(a, dict) else 1500.0
        elo_b = b.get("elo", 1500.0) if isinstance(b, dict) else 1500.0
        return elo_a + elo_b

    sorted_matches = sorted(eligible, key=combined_elo, reverse=True)

    # Check if pinned match is still active
    effective_pin = None
    if pinned_match_id is not None:
        pinned = [m for m in sorted_matches if m.get("match_id") == pinned_match_id]
        if pinned:
            primary = pinned[0]
            effective_pin = pinned_match_id
            secondaries = [m for m in sorted_matches if m.get("match_id") != pinned_match_id]
            return primary, secondaries[:2], effective_pin

    # Auto-select: highest combined Elo
    primary = sorted_matches[0]
    secondaries = sorted_matches[1:3]
    return primary, secondaries, effective_pin


def set_primary_match(match_id: Optional[str]) -> None:
    """Pin a match as primary, or pass None to reset to auto-selection.

    This is the promotion API — the same entry point that UI buttons
    and future integrations (e.g., Twitch voting) use.
    """
    st.session_state["ladder_primary_match_id"] = match_id
```

Also add `Tuple` to the typing imports at the top of the file if not already present.

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_ladder_dashboard.py::TestSelectDisplayMatches -v`
Expected: All 8 tests PASS

- [ ] **Step 3: Commit**

```bash
git add keisei/webui/streamlit_app.py
git commit -m "feat(webui): add match selection logic with manual pin override"
```

---

### Task 5: Board Rendering Helpers and Ladder Tab Extension

**Files:**
- Modify: `keisei/webui/streamlit_app.py`

- [ ] **Step 1: Add the rendering helpers**

Add these functions after `set_primary_match()` and before `render_ladder_tab()`:

```python
def render_primary_board(match: Dict[str, Any]) -> None:
    """Render the primary spectator board with full match details."""
    from keisei.webui.sfen_utils import sfen_to_board_state

    model_a = match.get("model_a", {})
    model_b = match.get("model_b", {})
    name_a = model_a.get("name", "?") if isinstance(model_a, dict) else "?"
    name_b = model_b.get("name", "?") if isinstance(model_b, dict) else "?"
    elo_a = model_a.get("elo", 1500) if isinstance(model_a, dict) else 1500
    elo_b = model_b.get("elo", 1500) if isinstance(model_b, dict) else 1500

    # Player header
    st.markdown(
        f"**{name_a}** ({elo_a:.0f}) vs **{name_b}** ({elo_b:.0f})"
    )

    # Board
    board_state = sfen_to_board_state(match["sfen"])
    render_board(board_state, cell_size=40)

    # Hand pieces
    black_hand = board_state.get("black_hand", {})
    white_hand = board_state.get("white_hand", {})
    if black_hand or white_hand:
        bh = ", ".join(f"{v}\u00d7{k[0].upper()}" for k, v in black_hand.items()) or "\u2014"
        wh = ", ".join(f"{v}\u00d7{k[0].upper()}" for k, v in white_hand.items()) or "\u2014"
        st.caption(f"\u2617 Hand: {bh} \u2003\u2003 \u2616 Hand: {wh}")

    # Move log
    move_log = match.get("move_log", [])
    if move_log:
        recent = move_log[-10:]
        start_idx = max(1, len(move_log) - len(recent) + 1)
        moves_text = " \u2003 ".join(
            f"{i}. {m}" for i, m in enumerate(recent, start_idx)
        )
        st.text(moves_text)

    # Status
    move_count = match.get("move_count", board_state.get("move_count", 0))
    current = board_state["current_player"].capitalize()
    st.caption(f"Move {move_count} \u00b7 {current} to play")


def render_secondary_board(match: Dict[str, Any]) -> None:
    """Render a compact secondary spectator board."""
    from keisei.webui.sfen_utils import sfen_to_board_state

    model_a = match.get("model_a", {})
    model_b = match.get("model_b", {})
    name_a = model_a.get("name", "?") if isinstance(model_a, dict) else "?"
    name_b = model_b.get("name", "?") if isinstance(model_b, dict) else "?"
    elo_a = model_a.get("elo", 1500) if isinstance(model_a, dict) else 1500
    elo_b = model_b.get("elo", 1500) if isinstance(model_b, dict) else 1500

    st.markdown(
        f"**{name_a}** ({elo_a:.0f}) vs **{name_b}** ({elo_b:.0f})"
    )

    board_state = sfen_to_board_state(match["sfen"])
    render_board(board_state, cell_size=32)

    move_count = match.get("move_count", board_state.get("move_count", 0))
    current = board_state["current_player"].capitalize()
    col_status, col_btn = st.columns([3, 1])
    col_status.caption(f"Move {move_count} \u00b7 {current} to play")
    match_id = match.get("match_id", "")
    col_btn.button(
        "Watch",
        key=f"watch_{match_id}",
        on_click=set_primary_match,
        args=(match_id,),
    )
```

- [ ] **Step 2: Extend render_ladder_tab with Live Games section**

In `render_ladder_tab()`, add the following after the timestamp caption (after line ~1175):

```python
    # --- Live Games section ---
    st.markdown("---")
    st.subheader("Live Games")

    pinned = st.session_state.get("ladder_primary_match_id")
    primary, secondaries, effective_pin = select_display_matches(matches, pinned)

    if effective_pin != pinned:
        st.session_state["ladder_primary_match_id"] = effective_pin

    if primary is None:
        st.info("No live games \u2014 waiting for spectated matches")
    else:
        if secondaries:
            primary_col, secondary_col = st.columns([3, 2])
        else:
            primary_col = st.container()
            secondary_col = None

        with primary_col:
            render_primary_board(primary)

        if secondary_col is not None:
            with secondary_col:
                for match in secondaries:
                    render_secondary_board(match)
                if st.session_state.get("ladder_primary_match_id"):
                    st.button(
                        "\u21bb Auto-select primary",
                        on_click=set_primary_match,
                        args=(None,),
                    )
```

- [ ] **Step 3: Run all ladder dashboard tests**

Run: `uv run pytest tests/unit/test_ladder_dashboard.py tests/unit/test_sfen_utils.py -v`
Expected: All tests PASS

- [ ] **Step 4: Run the full test suite to check for regressions**

Run: `uv run pytest tests/unit/ -v --tb=short`
Expected: All existing tests PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/webui/streamlit_app.py
git commit -m "feat(webui): add live spectator boards to Ladder tab"
```

---

### Task 6: Manual Smoke Test and Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Run linting**

Run: `uv run flake8 keisei/webui/sfen_utils.py keisei/webui/streamlit_app.py --max-line-length=120`
Expected: No errors

- [ ] **Step 2: Run type checking**

Run: `uv run mypy keisei/webui/sfen_utils.py --ignore-missing-imports`
Expected: No errors

- [ ] **Step 3: Run formatting**

Run: `uv run black --check keisei/webui/sfen_utils.py keisei/webui/streamlit_app.py`
Expected: Already formatted (or run `uv run black` to fix)

- [ ] **Step 4: Run full test suite one final time**

Run: `uv run pytest tests/unit/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Final commit if any formatting changes**

```bash
git add -u
git commit -m "style: format spectator board code"
```
