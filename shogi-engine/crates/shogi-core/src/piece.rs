use std::num::NonZeroU8;
use crate::types::{Color, PieceType};

/// A Shogi piece packed into a NonZeroU8.
///
/// Bit layout: [5]=promoted, [4]=color (0=Black, 1=White), [3:0]=piece_type (1-8).
///
/// PieceType values span 1-8, requiring 4 bits.  Color occupies bit 4 and the
/// promoted flag occupies bit 5.  Since piece_type >= 1, the underlying value
/// is always non-zero, so `Option<Piece>` is 1 byte via niche optimization.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Piece(NonZeroU8);

impl Piece {
    pub fn new(piece_type: PieceType, color: Color, promoted: bool) -> Piece {
        let mut val = piece_type as u8;          // bits [3:0]
        if color == Color::White {
            val |= 0x10;                          // bit 4
        }
        if promoted {
            val |= 0x20;                          // bit 5
        }
        Piece(NonZeroU8::new(val).unwrap())
    }

    #[inline]
    pub fn to_u8(self) -> u8 {
        self.0.get()
    }

    #[inline]
    pub fn from_u8(val: u8) -> Option<Piece> {
        NonZeroU8::new(val).map(Piece)
    }

    pub fn piece_type(self) -> PieceType {
        PieceType::from_u8(self.0.get() & 0x0F)
            .expect("invalid piece type in Piece encoding")
    }

    pub fn color(self) -> Color {
        if self.0.get() & 0x10 != 0 {
            Color::White
        } else {
            Color::Black
        }
    }

    pub fn is_promoted(self) -> bool {
        self.0.get() & 0x20 != 0
    }

    pub fn promote(self) -> Piece {
        debug_assert!(self.piece_type().can_promote());
        debug_assert!(!self.is_promoted());
        Piece(NonZeroU8::new(self.0.get() | 0x20).unwrap())
    }

    pub fn unpromote(self) -> Piece {
        Piece(NonZeroU8::new(self.0.get() & !0x20).unwrap())
    }
}

impl std::fmt::Debug for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prom = if self.is_promoted() { "+" } else { "" };
        write!(f, "{}{:?}({:?})", prom, self.piece_type(), self.color())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Color, PieceType};

    #[test]
    fn test_option_piece_is_one_byte() {
        assert_eq!(std::mem::size_of::<Option<Piece>>(), 1);
    }

    #[test]
    fn test_piece_roundtrip() {
        let colors = [Color::Black, Color::White];
        let piece_types = [
            PieceType::Pawn,
            PieceType::Lance,
            PieceType::Knight,
            PieceType::Silver,
            PieceType::Gold,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::King,
        ];
        for &color in &colors {
            for &pt in &piece_types {
                for &promoted in &[false, true] {
                    // King and Gold cannot be promoted — skip invalid combos
                    if promoted && !pt.can_promote() {
                        continue;
                    }
                    let piece = Piece::new(pt, color, promoted);
                    assert_eq!(piece.piece_type(), pt);
                    assert_eq!(piece.color(), color);
                    assert_eq!(piece.is_promoted(), promoted);
                }
            }
        }
    }

    #[test]
    fn test_piece_u8_roundtrip() {
        let piece = Piece::new(PieceType::Rook, Color::White, true);
        let val = piece.to_u8();
        let decoded = Piece::from_u8(val).expect("should decode");
        assert_eq!(decoded, piece);
    }

    #[test]
    fn test_piece_from_u8_zero_is_none() {
        assert_eq!(Piece::from_u8(0), None);
    }

    #[test]
    fn test_promote_unpromote() {
        for &pt in &[
            PieceType::Pawn,
            PieceType::Lance,
            PieceType::Knight,
            PieceType::Silver,
            PieceType::Bishop,
            PieceType::Rook,
        ] {
            for &color in &[Color::Black, Color::White] {
                let base = Piece::new(pt, color, false);
                let promoted = base.promote();
                assert!(promoted.is_promoted());
                assert_eq!(promoted.piece_type(), pt);
                assert_eq!(promoted.color(), color);

                let back = promoted.unpromote();
                assert_eq!(back, base);
                assert!(!back.is_promoted());
            }
        }
    }

    #[test]
    fn test_all_pieces_nonzero() {
        let colors = [Color::Black, Color::White];
        let piece_types = [
            PieceType::Pawn,
            PieceType::Lance,
            PieceType::Knight,
            PieceType::Silver,
            PieceType::Gold,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::King,
        ];
        for &color in &colors {
            for &pt in &piece_types {
                for &promoted in &[false, true] {
                    if promoted && !pt.can_promote() {
                        continue;
                    }
                    let piece = Piece::new(pt, color, promoted);
                    assert_ne!(piece.to_u8(), 0, "piece {:?} encoded as zero", piece);
                }
            }
        }
    }
}
