"""Tests for shogi_core_definitions.py: enums, Piece class, constants, and helpers.

Covers: Color enum, PieceType enum, Piece class lifecycle (construction,
promotion, equality, hashing, deepcopy), TerminationReason enum,
MoveApplicationResult dataclass, mapping constants, observation plane
indices, and symbol-conversion helper functions.
"""

import copy

import pytest

from keisei.shogi.shogi_core_definitions import (
    BASE_TO_PROMOTED_TYPE,
    Color,
    KIF_PIECE_SYMBOLS,
    MoveApplicationResult,
    OBS_CURR_PLAYER_HAND_START,
    OBS_CURR_PLAYER_INDICATOR,
    OBS_CURR_PLAYER_PROMOTED_START,
    OBS_CURR_PLAYER_UNPROMOTED_START,
    OBS_MOVE_COUNT,
    OBS_OPP_PLAYER_HAND_START,
    OBS_OPP_PLAYER_PROMOTED_START,
    OBS_OPP_PLAYER_UNPROMOTED_START,
    OBS_PROMOTED_ORDER,
    OBS_UNPROMOTED_ORDER,
    PIECE_TYPE_TO_HAND_TYPE,
    PROMOTED_TO_BASE_TYPE,
    PROMOTED_TYPES_SET,
    SYMBOL_TO_PIECE_TYPE,
    Piece,
    PieceType,
    TerminationReason,
    get_piece_type_from_symbol,
    get_unpromoted_types,
)


# ===========================================================================
# 1. Color enum (~5 tests)
# ===========================================================================


class TestColorEnum:
    """Verify Color enum values and the opponent() method."""

    def test_black_value_is_zero(self):
        assert Color.BLACK.value == 0

    def test_white_value_is_one(self):
        assert Color.WHITE.value == 1

    def test_opponent_of_black_is_white(self):
        assert Color.BLACK.opponent() == Color.WHITE

    def test_opponent_of_white_is_black(self):
        assert Color.WHITE.opponent() == Color.BLACK

    def test_iteration_covers_both_colors(self):
        colors = list(Color)
        assert len(colors) == 2
        assert Color.BLACK in colors
        assert Color.WHITE in colors


# ===========================================================================
# 2. PieceType enum (~8 tests)
# ===========================================================================


class TestPieceTypeEnum:
    """Verify PieceType enum values and the to_usi_char() method."""

    def test_all_fourteen_values_exist(self):
        assert len(PieceType) == 14

    def test_unpromoted_values_range_zero_to_seven(self):
        expected = {
            "PAWN": 0,
            "LANCE": 1,
            "KNIGHT": 2,
            "SILVER": 3,
            "GOLD": 4,
            "BISHOP": 5,
            "ROOK": 6,
            "KING": 7,
        }
        for name, val in expected.items():
            assert PieceType[name].value == val

    def test_promoted_values_range_eight_to_thirteen(self):
        expected = {
            "PROMOTED_PAWN": 8,
            "PROMOTED_LANCE": 9,
            "PROMOTED_KNIGHT": 10,
            "PROMOTED_SILVER": 11,
            "PROMOTED_BISHOP": 12,
            "PROMOTED_ROOK": 13,
        }
        for name, val in expected.items():
            assert PieceType[name].value == val

    @pytest.mark.parametrize(
        "piece_type, expected_char",
        [
            (PieceType.PAWN, "P"),
            (PieceType.LANCE, "L"),
            (PieceType.KNIGHT, "N"),
            (PieceType.SILVER, "S"),
            (PieceType.GOLD, "G"),
            (PieceType.BISHOP, "B"),
            (PieceType.ROOK, "R"),
        ],
    )
    def test_to_usi_char_valid_droppable_pieces(self, piece_type, expected_char):
        assert piece_type.to_usi_char() == expected_char

    def test_to_usi_char_raises_for_king(self):
        with pytest.raises(ValueError, match="KING"):
            PieceType.KING.to_usi_char()

    @pytest.mark.parametrize(
        "promoted_type",
        [
            PieceType.PROMOTED_PAWN,
            PieceType.PROMOTED_LANCE,
            PieceType.PROMOTED_KNIGHT,
            PieceType.PROMOTED_SILVER,
            PieceType.PROMOTED_BISHOP,
            PieceType.PROMOTED_ROOK,
        ],
    )
    def test_to_usi_char_raises_for_promoted_types(self, promoted_type):
        with pytest.raises(ValueError, match="cannot be dropped"):
            promoted_type.to_usi_char()

    def test_piece_types_have_unique_values(self):
        values = [pt.value for pt in PieceType]
        assert len(values) == len(set(values))

    def test_piece_types_are_contiguous_integers(self):
        values = sorted(pt.value for pt in PieceType)
        assert values == list(range(14))


# ===========================================================================
# 3. Piece class (~10 tests)
# ===========================================================================


class TestPieceClass:
    """Verify Piece construction, attributes, promotion, equality, and copy."""

    def test_construction_sets_type_and_color(self):
        p = Piece(PieceType.PAWN, Color.BLACK)
        assert p.type == PieceType.PAWN
        assert p.color == Color.BLACK

    def test_is_promoted_false_for_base_type(self):
        p = Piece(PieceType.ROOK, Color.WHITE)
        assert p.is_promoted is False

    def test_is_promoted_true_for_promoted_type(self):
        p = Piece(PieceType.PROMOTED_ROOK, Color.BLACK)
        assert p.is_promoted is True

    def test_symbol_uppercase_for_black(self):
        p = Piece(PieceType.PAWN, Color.BLACK)
        assert p.symbol() == "P"

    def test_symbol_lowercase_for_white(self):
        p = Piece(PieceType.PAWN, Color.WHITE)
        assert p.symbol() == "p"

    def test_symbol_promoted_black_has_plus_prefix(self):
        p = Piece(PieceType.PROMOTED_ROOK, Color.BLACK)
        assert p.symbol() == "+R"

    def test_symbol_promoted_white_lowercase_with_plus_prefix(self):
        p = Piece(PieceType.PROMOTED_BISHOP, Color.WHITE)
        assert p.symbol() == "+b"

    def test_promote_changes_promotable_piece(self):
        p = Piece(PieceType.PAWN, Color.BLACK)
        p.promote()
        assert p.type == PieceType.PROMOTED_PAWN
        assert p.is_promoted is True

    def test_promote_noop_for_non_promotable_piece(self):
        """Gold and King cannot promote; promote() should be a no-op."""
        gold = Piece(PieceType.GOLD, Color.BLACK)
        gold.promote()
        assert gold.type == PieceType.GOLD
        assert gold.is_promoted is False

        king = Piece(PieceType.KING, Color.WHITE)
        king.promote()
        assert king.type == PieceType.KING
        assert king.is_promoted is False

    def test_promote_noop_for_already_promoted_piece(self):
        p = Piece(PieceType.PROMOTED_SILVER, Color.WHITE)
        p.promote()
        assert p.type == PieceType.PROMOTED_SILVER
        assert p.is_promoted is True

    def test_unpromote_reverts_promoted_piece(self):
        p = Piece(PieceType.PROMOTED_LANCE, Color.BLACK)
        p.unpromote()
        assert p.type == PieceType.LANCE
        assert p.is_promoted is False

    def test_unpromote_noop_for_base_piece(self):
        p = Piece(PieceType.SILVER, Color.WHITE)
        p.unpromote()
        assert p.type == PieceType.SILVER
        assert p.is_promoted is False

    def test_eq_same_type_and_color(self):
        p1 = Piece(PieceType.BISHOP, Color.BLACK)
        p2 = Piece(PieceType.BISHOP, Color.BLACK)
        assert p1 == p2

    def test_eq_different_color(self):
        p1 = Piece(PieceType.BISHOP, Color.BLACK)
        p2 = Piece(PieceType.BISHOP, Color.WHITE)
        assert p1 != p2

    def test_eq_returns_not_implemented_for_non_piece(self):
        p = Piece(PieceType.PAWN, Color.BLACK)
        assert p.__eq__("not a piece") is NotImplemented

    def test_hash_equal_pieces_have_same_hash(self):
        p1 = Piece(PieceType.ROOK, Color.WHITE)
        p2 = Piece(PieceType.ROOK, Color.WHITE)
        assert hash(p1) == hash(p2)

    def test_hash_usable_as_dict_key(self):
        p = Piece(PieceType.KNIGHT, Color.BLACK)
        d = {p: "test_value"}
        lookup = Piece(PieceType.KNIGHT, Color.BLACK)
        assert d[lookup] == "test_value"

    def test_repr_format(self):
        p = Piece(PieceType.GOLD, Color.WHITE)
        assert repr(p) == "Piece(GOLD, WHITE)"

    def test_deepcopy_creates_independent_copy(self):
        original = Piece(PieceType.SILVER, Color.BLACK)
        copied = copy.deepcopy(original)
        assert original == copied
        assert original is not copied
        # Mutating the copy does not affect the original
        copied.promote()
        assert copied.type == PieceType.PROMOTED_SILVER
        assert original.type == PieceType.SILVER

    def test_constructor_raises_type_error_for_invalid_piece_type(self):
        with pytest.raises(TypeError, match="piece_type must be an instance of PieceType"):
            Piece(0, Color.BLACK)  # type: ignore[arg-type]

    def test_constructor_raises_type_error_for_invalid_color(self):
        with pytest.raises(TypeError, match="color must be an instance of Color"):
            Piece(PieceType.PAWN, "BLACK")  # type: ignore[arg-type]


# ===========================================================================
# 4. TerminationReason enum (~3 tests)
# ===========================================================================


class TestTerminationReason:
    """Verify TerminationReason enum values and __str__ behaviour."""

    def test_all_ten_values_exist(self):
        assert len(TerminationReason) == 10

    def test_expected_values(self):
        expected = {
            "CHECKMATE": "Tsumi",
            "STALEMATE": "stalemate",
            "REPETITION": "Sennichite",
            "MAX_MOVES_EXCEEDED": "Max moves reached",
            "RESIGNATION": "resignation",
            "TIME_FORFEIT": "time_forfeit",
            "ILLEGAL_MOVE": "illegal_move",
            "AGREEMENT": "agreement",
            "IMPASSE": "impasse",
            "NO_CONTEST": "no_contest",
        }
        for name, val in expected.items():
            assert TerminationReason[name].value == val

    def test_str_returns_value(self):
        assert str(TerminationReason.CHECKMATE) == "Tsumi"
        assert str(TerminationReason.MAX_MOVES_EXCEEDED) == "Max moves reached"
        assert str(TerminationReason.REPETITION) == "Sennichite"


# ===========================================================================
# 5. MoveApplicationResult dataclass (~3 tests)
# ===========================================================================


class TestMoveApplicationResult:
    """Verify MoveApplicationResult dataclass construction and defaults."""

    def test_default_construction(self):
        result = MoveApplicationResult()
        assert result.captured_piece_type is None
        assert result.was_promotion is False

    def test_construction_with_values(self):
        result = MoveApplicationResult(
            captured_piece_type=PieceType.GOLD,
            was_promotion=True,
        )
        assert result.captured_piece_type == PieceType.GOLD
        assert result.was_promotion is True

    def test_field_access_and_mutation(self):
        result = MoveApplicationResult()
        result.captured_piece_type = PieceType.ROOK
        result.was_promotion = True
        assert result.captured_piece_type == PieceType.ROOK
        assert result.was_promotion is True


# ===========================================================================
# 6. Constants (~5 tests)
# ===========================================================================


class TestConstants:
    """Verify mapping constants and observation plane indices."""

    def test_promoted_types_set_has_six_entries(self):
        assert len(PROMOTED_TYPES_SET) == 6
        for pt in PROMOTED_TYPES_SET:
            assert pt.value >= 8  # All promoted values are 8-13

    def test_base_to_promoted_roundtrip(self):
        """Every entry in BASE_TO_PROMOTED_TYPE should reverse via PROMOTED_TO_BASE_TYPE."""
        assert len(BASE_TO_PROMOTED_TYPE) == 6
        assert len(PROMOTED_TO_BASE_TYPE) == 6
        for base, promoted in BASE_TO_PROMOTED_TYPE.items():
            assert PROMOTED_TO_BASE_TYPE[promoted] == base

    def test_piece_type_to_hand_type_demotes_promoted_pieces(self):
        """Promoted pieces should map to their unpromoted (hand) equivalents."""
        assert PIECE_TYPE_TO_HAND_TYPE[PieceType.PROMOTED_PAWN] == PieceType.PAWN
        assert PIECE_TYPE_TO_HAND_TYPE[PieceType.PROMOTED_ROOK] == PieceType.ROOK
        assert PIECE_TYPE_TO_HAND_TYPE[PieceType.PROMOTED_BISHOP] == PieceType.BISHOP
        # Base pieces should map to themselves
        assert PIECE_TYPE_TO_HAND_TYPE[PieceType.GOLD] == PieceType.GOLD
        assert PIECE_TYPE_TO_HAND_TYPE[PieceType.SILVER] == PieceType.SILVER

    def test_observation_plane_indices_are_sequential(self):
        """Observation channels should be laid out in ascending order."""
        assert OBS_CURR_PLAYER_UNPROMOTED_START == 0
        assert OBS_CURR_PLAYER_PROMOTED_START == 8
        assert OBS_OPP_PLAYER_UNPROMOTED_START == 14
        assert OBS_OPP_PLAYER_PROMOTED_START == 22
        assert OBS_CURR_PLAYER_HAND_START == 28
        assert OBS_OPP_PLAYER_HAND_START == 35
        assert OBS_CURR_PLAYER_INDICATOR == 42
        assert OBS_MOVE_COUNT == 43

    def test_symbol_to_piece_type_has_fourteen_entries(self):
        assert len(SYMBOL_TO_PIECE_TYPE) == 14
        # Spot-check a few entries
        assert SYMBOL_TO_PIECE_TYPE["P"] == PieceType.PAWN
        assert SYMBOL_TO_PIECE_TYPE["K"] == PieceType.KING
        assert SYMBOL_TO_PIECE_TYPE["+R"] == PieceType.PROMOTED_ROOK

    def test_obs_unpromoted_order_has_eight_entries_including_king(self):
        assert len(OBS_UNPROMOTED_ORDER) == 8
        assert PieceType.KING in OBS_UNPROMOTED_ORDER

    def test_obs_promoted_order_has_six_entries(self):
        assert len(OBS_PROMOTED_ORDER) == 6
        assert all(pt in PROMOTED_TYPES_SET for pt in OBS_PROMOTED_ORDER)


# ===========================================================================
# 7. Helper functions (~8 tests)
# ===========================================================================


class TestHelperFunctions:
    """Verify get_unpromoted_types() and get_piece_type_from_symbol()."""

    def test_get_unpromoted_types_returns_seven(self):
        result = get_unpromoted_types()
        assert len(result) == 7

    def test_get_unpromoted_types_excludes_king(self):
        result = get_unpromoted_types()
        assert PieceType.KING not in result

    def test_get_unpromoted_types_contains_all_hand_pieces(self):
        result = get_unpromoted_types()
        expected = {
            PieceType.PAWN,
            PieceType.LANCE,
            PieceType.KNIGHT,
            PieceType.SILVER,
            PieceType.GOLD,
            PieceType.BISHOP,
            PieceType.ROOK,
        }
        assert set(result) == expected

    def test_get_piece_type_from_symbol_uppercase(self):
        assert get_piece_type_from_symbol("P") == PieceType.PAWN
        assert get_piece_type_from_symbol("K") == PieceType.KING
        assert get_piece_type_from_symbol("R") == PieceType.ROOK

    def test_get_piece_type_from_symbol_lowercase(self):
        assert get_piece_type_from_symbol("p") == PieceType.PAWN
        assert get_piece_type_from_symbol("r") == PieceType.ROOK
        assert get_piece_type_from_symbol("b") == PieceType.BISHOP

    def test_get_piece_type_from_symbol_promoted_uppercase(self):
        assert get_piece_type_from_symbol("+R") == PieceType.PROMOTED_ROOK
        assert get_piece_type_from_symbol("+P") == PieceType.PROMOTED_PAWN
        assert get_piece_type_from_symbol("+B") == PieceType.PROMOTED_BISHOP

    def test_get_piece_type_from_symbol_promoted_lowercase(self):
        assert get_piece_type_from_symbol("+r") == PieceType.PROMOTED_ROOK
        assert get_piece_type_from_symbol("+p") == PieceType.PROMOTED_PAWN
        assert get_piece_type_from_symbol("+s") == PieceType.PROMOTED_SILVER

    def test_get_piece_type_from_symbol_raises_for_unknown(self):
        with pytest.raises(ValueError, match="Unknown piece symbol"):
            get_piece_type_from_symbol("X")

    def test_get_piece_type_from_symbol_raises_for_empty_string(self):
        with pytest.raises(ValueError, match="Unknown piece symbol"):
            get_piece_type_from_symbol("")

    def test_get_piece_type_from_symbol_raises_for_invalid_promoted(self):
        with pytest.raises(ValueError, match="Unknown piece symbol"):
            get_piece_type_from_symbol("+X")


class TestKifPieceSymbols:
    """Verify the KIF_PIECE_SYMBOLS mapping covers all 14 piece types."""

    def test_kif_piece_symbols_has_fourteen_entries(self):
        assert len(KIF_PIECE_SYMBOLS) == 14

    def test_kif_piece_symbols_covers_all_piece_types(self):
        for pt in PieceType:
            assert pt in KIF_PIECE_SYMBOLS, f"{pt.name} missing from KIF_PIECE_SYMBOLS"

    def test_kif_piece_symbols_values_are_two_char_strings(self):
        for pt, symbol in KIF_PIECE_SYMBOLS.items():
            assert isinstance(symbol, str)
            assert len(symbol) == 2, f"KIF symbol for {pt.name} should be 2 chars, got '{symbol}'"
