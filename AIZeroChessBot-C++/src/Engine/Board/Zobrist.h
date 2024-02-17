#pragma once

#include "Board.h"
#include "Piece.h"
#include <array>
#include <memory>
#include <random>
#include <vector>

namespace Chess {
// Helper class for the calculation of zobrist hash.
// This is a single 64bit value that (non-uniquely) represents the current state of the game.

// It is mainly used for quickly detecting positions that have already been evaluated, to avoid
// potentially performing lots of duplicate work during game search.

class Zobrist {
    // Random numbers are generated for each aspect of the game state, and are used for calculating
    // the hash:

    // piece type, colour, square index
public:
    static std::array<std::array<unsigned long long, 64>, Piece::TotalTypesOfPieces> piecesArray;
    // Each player has 4 possible castling right states: none, queenside, kingside, both.
    // So, taking both sides into account, there are 16 possible states.
    static std::array<unsigned long long, 16> castlingRights;
    // En passant file (0 = no ep).
    //  Rank does not need to be specified since side to move is included in key
    static std::array<unsigned long long, 9> enPassantFile;
    static inline unsigned long long sideToMove = 0;

private:
    class StaticConstructor {
    public:
        StaticConstructor() {
            constexpr int seed = 29426028;
            std::mt19937_64 gen = std::mt19937_64(std::random_device{}());

            for (int squareIndex = 0; squareIndex < 64; squareIndex++) {
                for (auto piece : Piece::PieceIndices) {
                    piecesArray[piece][squareIndex] = RandomUnsigned64BitNumber(gen);
                }
            }

            for (int i = 0; i < castlingRights.size(); i++) {
                castlingRights[i] = RandomUnsigned64BitNumber(gen);
            }

            for (int i = 0; i < enPassantFile.size(); i++) {
                enPassantFile[i] = i == 0 ? 0 : RandomUnsigned64BitNumber(gen);
            }

            sideToMove = RandomUnsigned64BitNumber(gen);
        }

        // Generates a random 64-bit unsigned integer
        static unsigned long long RandomUnsigned64BitNumber(std::mt19937_64 &rng) {
            std::uniform_int_distribution<unsigned long long> dis(
                std::numeric_limits<unsigned long long>::min(),
                std::numeric_limits<unsigned long long>::max());
            return dis(rng);
        }
    };

    static inline Zobrist::StaticConstructor staticConstructor;

public:
    // Calculate zobrist key from current board position.
    // NOTE: this function is slow and should only be used when the board is initially set up from
    // fen. During search, the key should be updated incrementally instead.
    static unsigned long long CalculateZobristKey(const Board &board) {
        unsigned long long zobristKey = 0;

        for (int squareIndex = 0; squareIndex < 64; squareIndex++) {
            int piece = board.Square[squareIndex];

            if (Piece::PieceType(piece) != Piece::None) {
                zobristKey ^= piecesArray[piece][squareIndex];
            }
        }

        zobristKey ^= enPassantFile[board.CurrentGameState.enPassantFile];

        if (board.MoveColour() == Piece::Black) {
            zobristKey ^= sideToMove;
        }

        zobristKey ^= castlingRights[board.CurrentGameState.castlingRights];

        return zobristKey;
    }
};
} // namespace Chess
