#pragma once

#include "Engine/Board/Board.h"
#include "Engine/Board/Coord.h"
#include "Engine/Board/Piece.h"
#include <array>
#include <format>
#include <memory>
#include <string>
#include <vector>

namespace Chess {
class BoardHelper {
public:
    static inline const std::array<Coord, 4> RookDirections = {Coord(-1, 0), Coord(1, 0),
                                                               Coord(0, 1), Coord(0, -1)};
    static inline const std::array<Coord, 4> BishopDirections = {Coord(-1, 1), Coord(1, 1),
                                                                 Coord(1, -1), Coord(-1, -1)};

    static inline const std::string fileNames = "abcdefgh";
    static inline const std::string rankNames = "12345678";

    static inline constexpr int a1 = 0;
    static inline constexpr int b1 = 1;
    static inline constexpr int c1 = 2;
    static inline constexpr int d1 = 3;
    static inline constexpr int e1 = 4;
    static inline constexpr int f1 = 5;
    static inline constexpr int g1 = 6;
    static inline constexpr int h1 = 7;

    static inline constexpr int a8 = 56;
    static inline constexpr int b8 = 57;
    static inline constexpr int c8 = 58;
    static inline constexpr int d8 = 59;
    static inline constexpr int e8 = 60;
    static inline constexpr int f8 = 61;
    static inline constexpr int g8 = 62;
    static inline constexpr int h8 = 63;

    // Rank (0 to 7) of square
    static inline constexpr int RankIndex(int squareIndex) { return squareIndex >> 3; }

    // File (0 to 7) of square
    static inline constexpr int FileIndex(int squareIndex) { return squareIndex & 0b000111; }

    static inline constexpr int IndexFromCoord(int fileIndex, int rankIndex) {
        return rankIndex * 8 + fileIndex;
    }

    static inline constexpr int IndexFromCoord(Coord coord) {
        return IndexFromCoord(coord.fileIndex, coord.rankIndex);
    }

    static inline constexpr Coord CoordFromIndex(int squareIndex) {
        return Coord(FileIndex(squareIndex), RankIndex(squareIndex));
    }

    static inline constexpr bool LightSquare(int fileIndex, int rankIndex) {
        return (fileIndex + rankIndex) % 2 != 0;
    }

    static inline constexpr bool LightSquare(int squareIndex) {
        return LightSquare(FileIndex(squareIndex), RankIndex(squareIndex));
    }

    static inline constexpr std::string SquareNameFromCoordinate(int fileIndex, int rankIndex) {
        return fileNames[fileIndex] + std::to_string(rankIndex + 1);
    }

    static inline constexpr std::string SquareNameFromIndex(int squareIndex) {
        return SquareNameFromCoordinate(CoordFromIndex(squareIndex));
    }

    static inline constexpr std::string SquareNameFromCoordinate(Coord coord) {
        return SquareNameFromCoordinate(coord.fileIndex, coord.rankIndex);
    }

    static inline constexpr int SquareIndexFromName(const std::string &name) {
        char fileName = name[0];
        char rankName = name[1];
        int fileIndex = (int) fileNames.find(fileName);
        int rankIndex = (int) rankNames.find(rankName);
        return IndexFromCoord(fileIndex, rankIndex);
    }

    static inline constexpr bool IsValidCoordinate(int x, int y) {
        return x >= 0 && x < 8 && y >= 0 && y < 8;
    }

    /// <summary>
    /// Creates an ASCII-diagram of the current position.
    /// </summary>
    static std::string CreateDiagram(const Board &board, bool blackAtTop = true) {

        std::string result;
        for (int y = 0; y < 8; y++) {
            int rankIndex = blackAtTop ? 7 - y : y;
            result.append("+---+---+---+---+---+---+---+---+");

            for (int x = 0; x < 8; x++) {
                int fileIndex = blackAtTop ? x : 7 - x;
                int squareIndex = IndexFromCoord(fileIndex, rankIndex);
                int piece = board.Square[squareIndex];

                result.append(std::format("| {0:s} ", Piece::GetSymbol(piece)));

                if (x == 7) {
                    // Show rank number
                    result.append(std::format("| {0:s}", rankIndex + 1));
                    result.append("\n");
                }
            }

            if (y == 7) {
                // Show file names
                result.append("+---+---+---+---+---+---+---+---+");
                result.append("\n");
                const std::string fileNames = "  a   b   c   d   e   f   g   h  ";
                const std::string fileNamesRev = "  h   g   f   e   d   c   b   a  ";
                result.append(blackAtTop ? fileNames : fileNamesRev);
                result.append("\n");
                result.append("\n");
            }
        }

        return result;
    }
};
} // namespace Chess
