#pragma once

#include "position.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace Stockfish;

class Board {
public:
    static constexpr std::uint16_t MAX_REVERSIBLE_HISTORY_PLIES = 100;

    explicit Board(
        const std::string &fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Board(const Board &) = default;
    Board(Board &&) noexcept = default;
    Board &operator=(const Board &) = default;
    Board &operator=(Board &&) noexcept = default;

    /// Returns +1 if White to move, -1 if Black to move.
    [[nodiscard]] int currentPlayer() const { return (m_pos.side_to_move() == WHITE) ? +1 : -1; }

    /// Pushes a move onto the internal Position. Assumes move is legal.
    void makeMove(Move m);

    /// Returns true if no legal moves are available (checkmate or stalemate).
    [[nodiscard]] bool isGameOver() const;

    /**
     * If the game is over and ended in checkmate, returns the winner:
     *   +1 if White won, -1 if Black won.
     * For stalemate or other draws, returns std::nullopt.
     */
    [[nodiscard]] std::optional<int> checkWinner() const;

    /// Generates every legal move, including all promotion choices.
    [[nodiscard]] std::vector<Move> validMoves() const;

    /// Returns a 64‐bit Zobrist hash of the current position.
    [[nodiscard]] std::uint64_t quickHash() const { return m_pos.key(); }

    /**
     * Returns a material‐based heuristic score in [-1, +1]:
     *   +1 means White is up maximum material,
     *   -1 means Black is up maximum material,
     *   0 means material is equal.
     */
    [[nodiscard]] double approximateResultScore() const;

    /// Returns the FEN string representation of the current position.
    [[nodiscard]] std::string fen() const { return m_pos.fen(); }

    /// Sets the internal Position from a FEN string.
    void setFen(const std::string &fen);

    /// Returns a reference to the internal Position object.
    [[nodiscard]] const Position &position() const { return m_pos; }

    /// Returns the number of earlier occurrences of the current position.
    [[nodiscard]] int repetitionCount() const;

    /**
     * Produces an ASCII representation of the board similar to the Python version:
     *   '  a b c d e f g h'
     *   '1 r n b q k b n r 1'
     *   '2 p p p p p p p p 2'
     *   '3 . . . . . . . . 3'
     *   ...
     *   '8 R N B Q K B N R 8'
     *   '  a b c d e f g h'
     *
     * Ranks are printed with rank 1 at the top down to rank 8 at the bottom,
     * to match the Python code's iteration over i=0..7 as rank i+1.
     */
    [[nodiscard]] std::string repr() const;

private:
    struct PositionHistory {
        PositionHistory(const std::uint64_t positionKey,
                        std::shared_ptr<const PositionHistory> previousPosition)
            : key(positionKey), previous(std::move(previousPosition)),
              retainedPreviousPositions(
                  previous == nullptr ? 0 : previous->retainedPreviousPositions + 1) {}

        const std::uint64_t key;
        const std::shared_ptr<const PositionHistory> previous;
        const std::uint16_t retainedPreviousPositions;
    };

    Position m_pos;
    std::shared_ptr<const PositionHistory> m_history;

    [[nodiscard]] static constexpr const char *pieceSymbol(PieceType pt, Color c);

    [[nodiscard]] bool drawByInsufficientMaterial() const;
    [[nodiscard]] std::uint8_t castlingRightsMask() const;
    void validateHistory() const;
};

// Overload operator<< so you can do: std::cout << board << "\n";
inline std::ostream &operator<<(std::ostream &os, const Board &board) {
    os << board.repr();
    return os;
}
