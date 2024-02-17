#pragma once

#include "Piece.h"

/*
Compact (16bit) move representation to preserve memory during search.

The format is as follows (ffff tttttt ssssss)
Bits 0-5: start square index
Bits 6-11: target square index
Bits 12-15: flag (promotion type, etc)
*/
namespace Chess {
const class Move {
    // 16bit move value
public:
    const unsigned short moveValue = 0;

    // Flags
    static constexpr int NoFlag = 0b0000;
    static constexpr int EnPassantCaptureFlag = 0b0001;
    static constexpr int CastleFlag = 0b0010;
    static constexpr int PawnTwoUpFlag = 0b0011;

    static constexpr int PromoteToQueenFlag = 0b0100;
    static constexpr int PromoteToKnightFlag = 0b0101;
    static constexpr int PromoteToRookFlag = 0b0110;
    static constexpr int PromoteToBishopFlag = 0b0111;

    // Masks
    static constexpr unsigned short startSquareMask = 0b0000000000111111;
    static constexpr unsigned short targetSquareMask = 0b0000111111000000;
    static constexpr unsigned short flagMask = 0b1111000000000000;

    inline constexpr Move(unsigned short moveValue) : moveValue(moveValue) {}

    inline constexpr Move(int startSquare, int targetSquare)
        : moveValue(static_cast<unsigned short>(startSquare | targetSquare << 6)) {}

    inline constexpr Move(int startSquare, int targetSquare, int flag)
        : moveValue(static_cast<unsigned short>(startSquare | targetSquare << 6 | flag << 12)) {}

    inline constexpr Move &operator=(const Move &other) { return *this = Move(other.moveValue); }

    inline constexpr unsigned short Value() const { return moveValue; }
    inline constexpr bool IsNull() const { return moveValue == 0; }

    inline constexpr int StartSquare() const { return moveValue & startSquareMask; }
    inline constexpr int TargetSquare() const { return (moveValue & targetSquareMask) >> 6; }
    inline constexpr bool IsPromotion() const { return MoveFlag() >= PromoteToQueenFlag; }
    inline constexpr int MoveFlag() const { return moveValue >> 12; }

    inline constexpr int PromotionPieceType() const {
        switch (MoveFlag()) {
        case PromoteToRookFlag:
            return Piece::Rook;
        case PromoteToKnightFlag:
            return Piece::Knight;
        case PromoteToBishopFlag:
            return Piece::Bishop;
        case PromoteToQueenFlag:
            return Piece::Queen;
        default:
            return Piece::None;
        }
    }

    inline constexpr bool operator==(Move other) const { return moveValue == other.moveValue; }
    inline constexpr bool operator!=(Move other) const { return moveValue != other.moveValue; }

    static inline constexpr Move NullMove() { return Move(0); }

    static inline constexpr bool SameMove(Move a, Move b) { return a.moveValue == b.moveValue; }

    Move() = default;
};
} // namespace Chess
