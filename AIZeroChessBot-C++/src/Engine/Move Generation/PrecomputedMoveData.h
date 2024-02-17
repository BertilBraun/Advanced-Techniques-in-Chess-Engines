#pragma once

#include "Engine/Board/Board.h"
#include "Engine/Board/Coord.h"
#include "Engine/Helpers/BoardHelper.h"
#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace Chess {

class PrecomputedMoveData final {
    template <typename T> static constexpr int sign(T val) { return (T(0) < val) - (val < T(0)); }

public:
    static inline std::array<std::array<unsigned long long, 64>, 64> alignMask;
    static inline std::array<std::array<unsigned long long, 64>, 8> dirRayMask;

    // First 4 are orthogonal, last 4 are diagonals (N, S, W, E, NW, SE, NE, SW)
    static inline constexpr std::array<int, 8> directionOffsets = {8, -8, -1, 1, 7, -7, 9, -9};

    static inline constexpr std::array<Coord, 8> dirOffsets2D = {
        Coord(0, 1),  Coord(0, -1), Coord(-1, 0), Coord(1, 0),
        Coord(-1, 1), Coord(1, -1), Coord(1, 1),  Coord(-1, -1)};

    // Stores number of moves available in each of the 8 directions for every square on the board
    // Order of directions is: N, S, W, E, NW, SE, NE, SW
    // So for example, if availableSquares[0][1] == 7...
    // that means that there are 7 squares to the north of b1 (the square with index 1 in board
    // array)
    static inline std::array<std::array<int, 8>, 64> numSquaresToEdge;

    // Stores array of indices for each square a knight can land on from any square on the board
    // So for example, knightMoves[0] is equal to {10, 17}, meaning a knight on a1 can jump to c2
    // and b3
    static inline std::array<std::vector<unsigned char>, 64> knightMoves;
    static inline std::array<std::vector<unsigned char>, 64> kingMoves;

    // Pawn attack directions for white and black (NW, NE; SW SE)
    static inline constexpr std::array<std::array<unsigned char, 2>, 2> pawnAttackDirections = {
        {{{4, 6}}, {{7, 5}}}};

    static inline std::array<std::vector<int>, 64> pawnAttacksWhite;
    static inline std::array<std::vector<int>, 64> pawnAttacksBlack;
    static inline std::array<int, 127> directionLookup;

    static inline std::array<unsigned long long, 64> kingAttackBitboards;
    static inline std::array<unsigned long long, 64> knightAttackBitboards;
    static inline std::array<std::array<unsigned long long, 2>, 64> pawnAttackBitboards;

    static inline std::array<unsigned long long, 64> rookMoves;
    static inline std::array<unsigned long long, 64> bishopMoves;
    static inline std::array<unsigned long long, 64> queenMoves;

    // Aka manhattan distance (answers how many moves for a rook to get from square a to square b)
    static inline std::array<std::array<int, 64>, 64> OrthogonalDistance;

    // Aka chebyshev distance (answers how many moves for a king to get from square a to square b)
    static inline std::array<std::array<int, 64>, 64> kingDistance;
    static inline std::array<int, 64> CentreManhattanDistance;

    static int NumRookMovesToReachSquare(int startSquare, int targetSquare) {
        return OrthogonalDistance[startSquare][targetSquare];
    }

    static int NumKingMovesToReachSquare(int startSquare, int targetSquare) {
        return kingDistance[startSquare][targetSquare];
    }

    // Initialize lookup data
private:
    class StaticConstructor {
    public:
        constexpr StaticConstructor() {
            // Calculate knight jumps and available squares for each square on the board.
            // See comments by variable definitions for more info.
            std::vector<int> allKnightJumps = {15, 17, -17, -15, 10, -6, 6, -10};

            for (int squareIndex = 0; squareIndex < 64; squareIndex++) {

                int y = squareIndex / 8;
                int x = squareIndex - y * 8;

                int north = 7 - y;
                int south = y;
                int west = x;
                int east = 7 - x;
                numSquaresToEdge[squareIndex][0] = north;
                numSquaresToEdge[squareIndex][1] = south;
                numSquaresToEdge[squareIndex][2] = west;
                numSquaresToEdge[squareIndex][3] = east;
                numSquaresToEdge[squareIndex][4] = std::min(north, west);
                numSquaresToEdge[squareIndex][5] = std::min(south, east);
                numSquaresToEdge[squareIndex][6] = std::min(north, east);
                numSquaresToEdge[squareIndex][7] = std::min(south, west);

                // Calculate all squares knight can jump to from current square
                auto legalKnightJumps = std::vector<unsigned char>();
                unsigned long long knightBitboard = 0;
                for (auto knightJumpDelta : allKnightJumps) {
                    int knightJumpSquare = squareIndex + knightJumpDelta;
                    if (knightJumpSquare >= 0 && knightJumpSquare < 64) {
                        int knightSquareY = knightJumpSquare / 8;
                        int knightSquareX = knightJumpSquare - knightSquareY * 8;
                        // Ensure knight has moved max of 2 squares on x/y axis (to reject indices
                        // that have wrapped around side of board)
                        int maxCoordMoveDst =
                            std::max(std::abs(x - knightSquareX), std::abs(y - knightSquareY));
                        if (maxCoordMoveDst == 2) {
                            legalKnightJumps.push_back(
                                static_cast<unsigned char>(knightJumpSquare));
                            knightBitboard |= 1ULL << knightJumpSquare;
                        }
                    }
                }
                knightMoves[squareIndex] = legalKnightJumps;
                knightAttackBitboards[squareIndex] = knightBitboard;

                // Calculate all squares king can move to from current square (not including
                // castling)
                auto legalKingMoves = std::vector<unsigned char>();
                for (auto kingMoveDelta : directionOffsets) {
                    int kingMoveSquare = squareIndex + kingMoveDelta;
                    if (kingMoveSquare >= 0 && kingMoveSquare < 64) {
                        int kingSquareY = kingMoveSquare / 8;
                        int kingSquareX = kingMoveSquare - kingSquareY * 8;
                        // Ensure king has moved max of 1 square on x/y axis (to reject indices that
                        // have wrapped around side of board)
                        int maxCoordMoveDst =
                            std::max(std::abs(x - kingSquareX), std::abs(y - kingSquareY));
                        if (maxCoordMoveDst == 1) {
                            legalKingMoves.push_back(static_cast<unsigned char>(kingMoveSquare));
                            kingAttackBitboards[squareIndex] |= 1ULL << kingMoveSquare;
                        }
                    }
                }
                kingMoves[squareIndex] = legalKingMoves;

                // Calculate legal pawn captures for white and black
                if (x > 0) {
                    if (y < 7) {
                        pawnAttacksWhite[squareIndex].push_back(squareIndex + 7);
                        pawnAttackBitboards[squareIndex][Board::WhiteIndex] |= 1ULL
                                                                               << (squareIndex + 7);
                    }
                    if (y > 0) {
                        pawnAttacksBlack[squareIndex].push_back(squareIndex - 9);
                        pawnAttackBitboards[squareIndex][Board::BlackIndex] |= 1ULL
                                                                               << (squareIndex - 9);
                    }
                }
                if (x < 7) {
                    if (y < 7) {
                        pawnAttacksWhite[squareIndex].push_back(squareIndex + 9);
                        pawnAttackBitboards[squareIndex][Board::WhiteIndex] |= 1ul
                                                                               << (squareIndex + 9);
                    }
                    if (y > 0) {
                        pawnAttacksBlack[squareIndex].push_back(squareIndex - 7);
                        pawnAttackBitboards[squareIndex][Board::BlackIndex] |= 1ul
                                                                               << (squareIndex - 7);
                    }
                }

                // Rook moves
                for (int directionIndex = 0; directionIndex < 4; directionIndex++) {
                    int currentDirOffset = directionOffsets[directionIndex];
                    for (int n = 0; n < numSquaresToEdge[squareIndex][directionIndex]; n++) {
                        int targetSquare = squareIndex + currentDirOffset * (n + 1);
                        rookMoves[squareIndex] |= 1ul << targetSquare;
                    }
                }
                // Bishop moves
                for (int directionIndex = 4; directionIndex < 8; directionIndex++) {
                    int currentDirOffset = directionOffsets[directionIndex];
                    for (int n = 0; n < numSquaresToEdge[squareIndex][directionIndex]; n++) {
                        int targetSquare = squareIndex + currentDirOffset * (n + 1);
                        bishopMoves[squareIndex] |= 1ul << targetSquare;
                    }
                }
                queenMoves[squareIndex] = rookMoves[squareIndex] | bishopMoves[squareIndex];
            }

            for (int i = 0; i < 127; i++) {
                int offset = i - 63;
                int absOffset = std::abs(offset);
                int absDir = 1;
                if (absOffset % 9 == 0) {
                    absDir = 9;
                } else if (absOffset % 8 == 0) {
                    absDir = 8;
                } else if (absOffset % 7 == 0) {
                    absDir = 7;
                }

                directionLookup[i] = absDir * sign(offset);
            }

            // Distance lookup
            for (int squareA = 0; squareA < 64; squareA++) {
                Coord coordA = BoardHelper::CoordFromIndex(squareA);
                int fileDstFromCentre = std::max(3 - coordA.fileIndex, coordA.fileIndex - 4);
                int rankDstFromCentre = std::max(3 - coordA.rankIndex, coordA.rankIndex - 4);
                CentreManhattanDistance[squareA] = fileDstFromCentre + rankDstFromCentre;

                for (int squareB = 0; squareB < 64; squareB++) {

                    Coord coordB = BoardHelper::CoordFromIndex(squareB);
                    int rankDistance = std::abs(coordA.rankIndex - coordB.rankIndex);
                    int fileDistance = std::abs(coordA.fileIndex - coordB.fileIndex);
                    OrthogonalDistance[squareA][squareB] = fileDistance + rankDistance;
                    kingDistance[squareA][squareB] = std::max(fileDistance, rankDistance);
                }
            }

            for (int squareA = 0; squareA < 64; squareA++) {
                for (int squareB = 0; squareB < 64; squareB++) {
                    Coord cA = BoardHelper::CoordFromIndex(squareA);
                    Coord cB = BoardHelper::CoordFromIndex(squareB);
                    Coord delta = cB - cA;
                    Coord dir = Coord(sign(delta.fileIndex), sign(delta.rankIndex));
                    // Coord dirOffset = dirOffsets2D[dirIndex];

                    for (int i = -8; i < 8; i++) {
                        Coord coord = BoardHelper::CoordFromIndex(squareA) + dir * i;
                        if (coord.IsValidSquare()) {
                            alignMask[squareA][squareB] |= 1ul
                                                           << (BoardHelper::IndexFromCoord(coord));
                        }
                    }
                }
            }

            for (int dirIndex = 0; dirIndex < dirOffsets2D.size(); dirIndex++) {
                for (int squareIndex = 0; squareIndex < 64; squareIndex++) {
                    Coord square = BoardHelper::CoordFromIndex(squareIndex);

                    for (int i = 0; i < 8; i++) {
                        Coord coord = square + dirOffsets2D[dirIndex] * i;
                        if (coord.IsValidSquare()) {
                            dirRayMask[dirIndex][squareIndex] |=
                                1ul << (BoardHelper::IndexFromCoord(coord));
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    };
};
} // namespace Chess
