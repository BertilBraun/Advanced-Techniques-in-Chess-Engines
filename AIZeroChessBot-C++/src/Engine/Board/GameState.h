#pragma once

namespace Chess {
const class GameState {
public:
    const int capturedPieceType = 0;
    const int enPassantFile = 0;
    const int castlingRights = 0;
    const int fiftyMoveCounter = 0;
    const unsigned long long zobristKey = 0;

    static constexpr int ClearWhiteKingsideMask = 0b1110;
    static constexpr int ClearWhiteQueensideMask = 0b1101;
    static constexpr int ClearBlackKingsideMask = 0b1011;
    static constexpr int ClearBlackQueensideMask = 0b0111;

    GameState(int capturedPieceType, int enPassantFile, int castlingRights, int fiftyMoveCounter,
              unsigned long long zobristKey)
        : capturedPieceType(capturedPieceType), enPassantFile(enPassantFile),
          castlingRights(castlingRights), fiftyMoveCounter(fiftyMoveCounter),
          zobristKey(zobristKey) {}

    inline GameState &operator=(const GameState &other) {
        return *this = GameState(other.capturedPieceType, other.enPassantFile, other.castlingRights,
                                 other.fiftyMoveCounter, other.zobristKey);
    }

    bool HasKingsideCastleRight(bool white) const {
        int mask = white ? 1 : 4;
        return (castlingRights & mask) != 0;
    }

    bool HasQueensideCastleRight(bool white) const {
        int mask = white ? 2 : 8;
        return (castlingRights & mask) != 0;
    }

    GameState() = default;
};
} // namespace Chess
