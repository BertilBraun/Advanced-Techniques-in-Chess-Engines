#pragma once

#include "Engine/Board/Coord.h"
#include <array>
#include <memory>
#include <vector>

namespace Chess {
class BitBoardUtility final {
public:
    static inline constexpr unsigned long long FileA = 0x101010101010101;

    static inline constexpr unsigned long long Rank1 = 0b11111111;
    static inline constexpr unsigned long long Rank2 = Rank1 << 8;
    static inline constexpr unsigned long long Rank3 = Rank2 << 8;
    static inline constexpr unsigned long long Rank4 = Rank3 << 8;
    static inline constexpr unsigned long long Rank5 = Rank4 << 8;
    static inline constexpr unsigned long long Rank6 = Rank5 << 8;
    static inline constexpr unsigned long long Rank7 = Rank6 << 8;
    static inline constexpr unsigned long long Rank8 = Rank7 << 8;

    static inline constexpr unsigned long long notAFile = ~FileA;
    static inline constexpr unsigned long long notHFile = ~(FileA << 7);

    static inline std::array<unsigned long long, 64> KnightAttacks;
    static inline std::array<unsigned long long, 64> KingMoves;
    static inline std::array<unsigned long long, 64> WhitePawnAttacks;
    static inline std::array<unsigned long long, 64> BlackPawnAttacks;

    // Get index of least significant set bit in given 64bit value. Also clears the bit to zero.
    static inline constexpr int PopLSB(unsigned long long &b) {
        int i = std::countr_zero(b);
        b &= (b - 1);
        return i;
    }

    static inline constexpr void SetSquare(unsigned long long &bitboard, int squareIndex) {
        bitboard |= 1ULL << squareIndex;
    }

    static inline constexpr void ClearSquare(unsigned long long &bitboard, int squareIndex) {
        bitboard &= ~(1ULL << squareIndex);
    }

    static inline constexpr void ToggleSquare(unsigned long long &bitboard, int squareIndex) {
        bitboard ^= 1ULL << squareIndex;
    }

    static inline constexpr void ToggleSquares(unsigned long long &bitboard, int squareA, int squareB) {
        bitboard ^= (1ULL << squareA | 1ULL << squareB);
    }

    static inline constexpr bool ContainsSquare(unsigned long long bitboard, int square) {
        return ((bitboard >> square) & 1) != 0;
    }

    static inline constexpr unsigned long long PawnAttacks(unsigned long long pawnBitboard, bool isWhite) {
        // Pawn attacks are calculated like so: (example given with white to move)

        // The first half of the attacks are calculated by shifting all pawns north-east: northEastAttacks =
        // pawnBitboard << 9 Note that pawns on the h file will be wrapped around to the a file, so then mask out the a
        // file: northEastAttacks &= notAFile (Any pawns that were originally on the a file will have been shifted to
        // the b file, so a file should be empty).

        // The other half of the attacks are calculated by shifting all pawns north-west. This time the h file must be
        // masked out. Combine the two halves to get a bitboard with all the pawn attacks: northEastAttacks |
        // northWestAttacks

        if (isWhite) {
            return ((pawnBitboard << 9) & notAFile) | ((pawnBitboard << 7) & notHFile);
        }

        return ((pawnBitboard >> 7) & notAFile) | ((pawnBitboard >> 9) & notHFile);
    }

    static inline constexpr unsigned long long Shift(unsigned long long bitboard, int numSquaresToShift) {
        if (numSquaresToShift > 0) {
            return bitboard << numSquaresToShift;
        } else {
            return bitboard >> -numSquaresToShift;
        }
    }

public:
    class StaticConstructor {
    public:
        static constexpr std::array<Coord, 4> orthoDir = {Coord(-1, 0), Coord(0, 1), Coord(1, 0), Coord(0, -1)};
        static constexpr std::array<Coord, 4> diagDir = {Coord(-1, -1), Coord(-1, 1), Coord(1, 1), Coord(1, -1)};
        static constexpr std::array<Coord, 8> knightJumps = {Coord(-2, -1), Coord(-2, 1), Coord(-1, 2), Coord(1, 2),
                                                             Coord(2, 1),   Coord(2, -1), Coord(1, -2), Coord(-1, -2)};
        static constexpr std::array<int, 2> pawnOffsets = {-1, 1};

        StaticConstructor() {

            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    ProcessSquare(x, y);
                }
            }
        }

        static constexpr void ProcessSquare(int x, int y) {
            Coord square = Coord(x, y);

            for (int dirIndex = 0; dirIndex < 4; dirIndex++) {
                // Orthogonal and diagonal directions
                for (int dst = 1; dst < 8; dst++) {
                    Coord ortho = square + orthoDir[dirIndex] * dst;
                    Coord diag = square + diagDir[dirIndex] * dst;

                    if (ortho.IsValidSquare()) {
                        if (dst == 1) {
                            KingMoves[square.SquareIndex()] |= 1ul << ortho.SquareIndex();
                        }
                    }

                    if (diag.IsValidSquare()) {
                        if (dst == 1) {
                            KingMoves[square.SquareIndex()] |= 1ul << diag.SquareIndex();
                        }
                    }
                }

                // Knight jumps
                for (int i = 0; i < knightJumps.size(); i++) {
                    Coord knight = square + knightJumps[i];
                    if (knight.IsValidSquare()) {
                        KnightAttacks[square.SquareIndex()] |= 1ul << knight.SquareIndex();
                    }
                }

                // Pawn attacks
                for (int i = 0; i < pawnOffsets.size(); i++) {
                    Coord pawnWhite = square + Coord(pawnOffsets[i], 1);
                    Coord pawnBlack = square + Coord(pawnOffsets[i], -1);

                    if (pawnWhite.IsValidSquare()) {
                        WhitePawnAttacks[square.SquareIndex()] |= 1ul << pawnWhite.SquareIndex();
                    }

                    if (pawnBlack.IsValidSquare()) {
                        BlackPawnAttacks[square.SquareIndex()] |= 1ul << pawnBlack.SquareIndex();
                    }
                }
            }
        }
    };

public:
    static inline BitBoardUtility::StaticConstructor staticConstructor;
};
} // namespace Chess
