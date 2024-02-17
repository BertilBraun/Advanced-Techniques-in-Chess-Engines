#pragma once

#include <array>
#include <memory>
#include <vector>

namespace Chess {
class PieceList {

    // Indices of squares occupied by given piece type (only elements up to Count are valid, the
    // rest are unused/garbage)
public:
    std::vector<int> occupiedSquares;
    // Map to go from index of a square, to the index in the occupiedSquares array where that square
    // is stored
    std::array<int, 64> map = {};

    PieceList(int maxPieceCount = 16) { occupiedSquares.reserve(maxPieceCount); }

    int Count() const { return occupiedSquares.size(); }

    void AddPieceAtSquare(int square) {
        map[square] = occupiedSquares.size();
        occupiedSquares.push_back(square);
    }

    void RemovePieceAtSquare(int square) {
        int pieceIndex = map[square]; // get the index of this element in the occupiedSquares array
        occupiedSquares[pieceIndex] =
            occupiedSquares
                .back(); // move last element in array to the place of the removed element
        map[occupiedSquares[pieceIndex]] =
            pieceIndex; // update map to point to the moved element's location in the array
        occupiedSquares.pop_back(); // remove the last element in the array
    }

    void MovePiece(int startSquare, int targetSquare) {
        int pieceIndex =
            map[startSquare]; // get the index of this element in the occupiedSquares array
        occupiedSquares[pieceIndex] = targetSquare;
        map[targetSquare] = pieceIndex;
    }

    int operator[](int index) { return occupiedSquares[index]; }
};
} // namespace Chess
