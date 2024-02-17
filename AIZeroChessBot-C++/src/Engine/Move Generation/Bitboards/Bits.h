#pragma once

#include "BitBoardUtility.h"
#include "Engine/Helpers/BoardHelper.h"
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace Chess {
// A collection of precomputed bitboards for use during move gen, search, etc.
class Bits final {
public:
    static inline constexpr unsigned long long FileA = 0x101010101010101;

    static inline constexpr unsigned long long WhiteKingsideMask = 1ULL << BoardHelper::f1 | 1ULL << BoardHelper::g1;
    static inline constexpr unsigned long long BlackKingsideMask = 1ULL << BoardHelper::f8 | 1ULL << BoardHelper::g8;

    static inline constexpr unsigned long long WhiteQueensideMask2 = 1ULL << BoardHelper::d1 | 1ULL << BoardHelper::c1;
    static inline constexpr unsigned long long BlackQueensideMask2 = 1ULL << BoardHelper::d8 | 1ULL << BoardHelper::c8;

    static inline constexpr unsigned long long WhiteQueensideMask = WhiteQueensideMask2 | 1ULL << BoardHelper::b1;
    static inline constexpr unsigned long long BlackQueensideMask = BlackQueensideMask2 | 1ULL << BoardHelper::b8;

    static inline std::array<unsigned long long, 64> WhitePassedPawnMask;
    static inline std::array<unsigned long long, 64> BlackPassedPawnMask;

    // A pawn on 'e4' for example, is considered supported by any pawn on
    // squares: d3, d4, f3, f4
    static inline std::array<unsigned long long, 64> WhitePawnSupportMask;
    static inline std::array<unsigned long long, 64> BlackPawnSupportMask;

    static inline std::array<unsigned long long, 8> FileMask;
    static inline std::array<unsigned long long, 8> AdjacentFileMasks;

    static inline std::array<unsigned long long, 64> KingSafetyMask;

    // Mask of 'forward' square. For example, from e4 the forward squares for white are: [e5, e6, e7, e8]
    static inline std::array<unsigned long long, 64> WhiteForwardFileMask;
    static inline std::array<unsigned long long, 64> BlackForwardFileMask;

    // Mask of three consecutive files centred at given file index.
    // For example, given file '3', the mask would contains files [2,3,4].
    // Note that for edge files, such as file 0, it would contain files [0,1,2]
    static inline std::array<unsigned long long, 8> TripleFileMask;

public:
    class StaticConstructor {
    public:
        StaticConstructor() {
            for (int i = 0; i < 8; i++) {
                FileMask[i] = FileA << i;
                unsigned long long left = i > 0 ? FileA << (i - 1) : 0;
                unsigned long long right = i < 7 ? FileA << (i + 1) : 0;
                AdjacentFileMasks[i] = left | right;
            }

            for (int i = 0; i < 8; i++) {
                int clampedFile = std::clamp(i, 1, 6);
                TripleFileMask[i] = FileMask[clampedFile] | AdjacentFileMasks[clampedFile];
            }

            for (int square = 0; square < 64; square++) {
                int file = BoardHelper::FileIndex(square);
                int rank = BoardHelper::RankIndex(square);
                unsigned long long adjacentFiles = FileA << std::max(0, file - 1) | FileA << std::min(7, file + 1);
                // Passed pawn mask
                unsigned long long whiteForwardMask =
                    ~(std::numeric_limits<unsigned long long>::max() >> (64 - 8 * (rank + 1)));
                unsigned long long blackForwardMask = ((1ULL << 8 * rank) - 1);

                WhitePassedPawnMask[square] = (FileA << file | adjacentFiles) & whiteForwardMask;
                BlackPassedPawnMask[square] = (FileA << file | adjacentFiles) & blackForwardMask;
                // Pawn support mask
                unsigned long long adjacent = (1ULL << (square - 1) | 1ULL << (square + 1)) & adjacentFiles;
                WhitePawnSupportMask[square] = adjacent | BitBoardUtility::Shift(adjacent, -8);
                BlackPawnSupportMask[square] = adjacent | BitBoardUtility::Shift(adjacent, +8);

                WhiteForwardFileMask[square] = whiteForwardMask & FileMask[file];
                BlackForwardFileMask[square] = blackForwardMask & FileMask[file];
            }

            for (int i = 0; i < 64; i++) {
                KingSafetyMask[i] = BitBoardUtility::KingMoves[i] | (1ULL << i);
            }
        }
    };

public:
    static inline Bits::StaticConstructor staticConstructor;
};
} // namespace Chess
