#pragma once

#include "Engine/Board/Coord.h"
#include "Engine/Helpers/BoardHelper.h"
#include "Engine/Move Generation/Bitboards/BitBoardUtility.h"
#include <vector>

namespace Chess {
class MagicHelper final {
public:
    static std::vector<unsigned long long> CreateAllBlockerBitboards(unsigned long long movementMask) {
        // Create a list of the indices of the bits that are set in the movement mask
        std::vector<int> moveSquareIndices;
        for (int i = 0; i < 64; i++) {
            if (((movementMask >> i) & 1) == 1) {
                moveSquareIndices.push_back(i);
            }
        }

        // Calculate total number of different bitboards (one for each possible arrangement of pieces)
        int numPatterns = 1 << moveSquareIndices.size(); // 2^n
        std::vector<unsigned long long> blockerBitboards(numPatterns);

        // Create all bitboards
        for (int patternIndex = 0; patternIndex < numPatterns; patternIndex++) {
            for (int bitIndex = 0; bitIndex < moveSquareIndices.size(); bitIndex++) {
                int bit = (patternIndex >> bitIndex) & 1;
                blockerBitboards[patternIndex] |= static_cast<unsigned long long>(bit) << moveSquareIndices[bitIndex];
            }
        }

        return blockerBitboards;
    }

    static unsigned long long CreateMovementMask(int squareIndex, bool ortho) {
        unsigned long long mask = 0;
        std::array directions = ortho ? BoardHelper::RookDirections : BoardHelper::BishopDirections;
        Coord startCoord = Coord(squareIndex);

        for (auto dir : directions) {
            for (int dst = 1; dst < 8; dst++) {
                Coord coord = startCoord + dir * dst;
                Coord nextCoord = startCoord + dir * (dst + 1);

                if (nextCoord.IsValidSquare()) {
                    BitBoardUtility::SetSquare(mask, coord.SquareIndex());
                } else {
                    break;
                }
            }
        }
        return mask;
    }

    static unsigned long long LegalMoveBitboardFromBlockers(int startSquare, unsigned long long blockerBitboard,
                                                            bool ortho) {
        unsigned long long bitboard = 0;

        std::array directions = ortho ? BoardHelper::RookDirections : BoardHelper::BishopDirections;
        Coord startCoord = Coord(startSquare);

        for (auto dir : directions) {
            for (int dst = 1; dst < 8; dst++) {
                Coord coord = startCoord + dir * dst;

                if (coord.IsValidSquare()) {
                    BitBoardUtility::SetSquare(bitboard, coord.SquareIndex());
                    if (BitBoardUtility::ContainsSquare(blockerBitboard, coord.SquareIndex())) {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        return bitboard;
    }
};
} // namespace Chess
