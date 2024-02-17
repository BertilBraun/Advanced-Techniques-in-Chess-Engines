#pragma once

#include "Engine/Helpers/BoardHelper.h"

namespace Chess {
// Structure for representing squares on the chess board as file/rank integer
// pairs. (0, 0) = a1, (7, 7) = h8. Coords can also be used as offsets. For
// example, while a Coord of (-1, 0) is not a valid square, it can be used to
// represent the concept of moving 1 square left.

const class Coord {
public:
    const int fileIndex = 0;
    const int rankIndex = 0;

    inline constexpr Coord(int fileIndex, int rankIndex)
        : fileIndex(fileIndex), rankIndex(rankIndex) {}

    inline constexpr Coord(int squareIndex)
        : fileIndex(BoardHelper::FileIndex(squareIndex)),
          rankIndex(BoardHelper::RankIndex(squareIndex)) {}

    inline constexpr bool IsLightSquare() const { return (fileIndex + rankIndex) % 2 != 0; }

    inline constexpr bool operator==(Coord other) const {
        return fileIndex == other.fileIndex && rankIndex == other.rankIndex;
    }
    inline constexpr bool operator!=(Coord other) const {
        return fileIndex != other.fileIndex || rankIndex != other.rankIndex;
    }
    inline constexpr bool operator<(Coord other) const {
        return fileIndex < other.fileIndex ||
               (fileIndex == other.fileIndex && rankIndex < other.rankIndex);
    }
    inline constexpr bool operator>(Coord other) const {
        return fileIndex > other.fileIndex ||
               (fileIndex == other.fileIndex && rankIndex > other.rankIndex);
    }

    inline constexpr Coord operator+(Coord b) const {
        return Coord(this->fileIndex + b.fileIndex, this->rankIndex + b.rankIndex);
    }
    inline constexpr Coord operator-(Coord b) const {
        return Coord(this->fileIndex - b.fileIndex, this->rankIndex - b.rankIndex);
    }
    inline constexpr Coord operator*(int m) const {
        return Coord(this->fileIndex * m, this->rankIndex * m);
    }
    friend inline constexpr Coord operator*(int m, Coord a) { return a * m; }

    inline constexpr bool IsValidSquare() const {
        return fileIndex >= 0 && fileIndex < 8 && rankIndex >= 0 && rankIndex < 8;
    }

    inline constexpr int SquareIndex() const { return BoardHelper::IndexFromCoord(*this); }

    inline constexpr Coord() = default;
};
} // namespace Chess
