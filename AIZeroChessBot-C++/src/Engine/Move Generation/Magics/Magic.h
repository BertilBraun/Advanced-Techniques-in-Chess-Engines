#pragma once

#include "MagicHelper.h"
#include "PrecomputedMagics.h"
#include <memory>
#include <vector>

namespace Chess {
// Helper class for magic bitboards.
// This is a technique where bishop and rook moves are precomputed
// for any configuration of origin square and blocking pieces.
class Magic final {
    // Rook and bishop mask bitboards for each origin square.
    // A mask is simply the legal moves available to the piece from the origin square
    // (on an empty board), except that the moves stop 1 square before the edge of the board.
public:
    static inline std::array<unsigned long long, 64> RookMask;
    static inline std::array<unsigned long long, 64> BishopMask;

    static inline std::array<std::vector<unsigned long long>, 64> RookAttacks;
    static inline std::array<std::vector<unsigned long long>, 64> BishopAttacks;

    static unsigned long long GetSliderAttacks(int square, unsigned long long blockers,
                                               bool ortho) {
        return ortho ? GetRookAttacks(square, blockers) : GetBishopAttacks(square, blockers);
    }

    static unsigned long long GetRookAttacks(int square, unsigned long long blockers) {
        unsigned long long key =
            ((blockers & RookMask[square]) * PrecomputedMagics::RookMagics[square]) >>
            PrecomputedMagics::RookShifts[square];
        return RookAttacks[square][key];
    }

    static unsigned long long GetBishopAttacks(int square, unsigned long long blockers) {
        unsigned long long key =
            ((blockers & BishopMask[square]) * PrecomputedMagics::BishopMagics[square]) >>
            PrecomputedMagics::BishopShifts[square];
        return BishopAttacks[square][key];
    }

public:
    class StaticConstructor {
    public:
        StaticConstructor() {
            for (int squareIndex = 0; squareIndex < 64; squareIndex++) {
                RookMask[squareIndex] = MagicHelper::CreateMovementMask(squareIndex, true);
                BishopMask[squareIndex] = MagicHelper::CreateMovementMask(squareIndex, false);
            }

            for (int i = 0; i < 64; i++) {
                RookAttacks[i] = CreateTable(i, true, PrecomputedMagics::RookMagics[i],
                                             PrecomputedMagics::RookShifts[i]);
                BishopAttacks[i] = CreateTable(i, false, PrecomputedMagics::BishopMagics[i],
                                               PrecomputedMagics::BishopShifts[i]);
            }
        }

        static constexpr std::vector<unsigned long long>
        CreateTable(int square, bool rook, unsigned long long magic, int leftShift) {
            int numBits = 64 - leftShift;
            int lookupSize = 1 << numBits;
            std::vector<unsigned long long> table;
            table.resize(lookupSize);

            unsigned long long movementMask = MagicHelper::CreateMovementMask(square, rook);

            for (unsigned long long pattern :
                 MagicHelper::CreateAllBlockerBitboards(movementMask)) {
                unsigned long long index = (pattern * magic) >> leftShift;
                unsigned long long moves =
                    MagicHelper::LegalMoveBitboardFromBlockers(square, pattern, rook);
                table[index] = moves;
            }

            return table;
        }
    };

public:
    static inline Magic::StaticConstructor staticConstructor;
};
} // namespace Chess
