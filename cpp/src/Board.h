#pragma once

#include "bitboard.h"
#include "movegen.h"
#include "position.h"
#include "util/ChunkedList.h"

using namespace Stockfish;

static inline constexpr std::array COLORS = {Color::WHITE, Color::BLACK};
static inline constexpr std::array PIECE_TYPES = {
    PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK, PieceType::QUEEN, PieceType::KING
};


// Material values matching the Python implementation
static constexpr int PIECE_VALUE[PIECE_TYPE_NB] = {
    0, // No piece
    1, // Pawn
    3, // Knight
    3, // Bishop
    5, // Rook
    9, // Queen
    0  // King (value treated as 0 for evaluation)
};

// Maximum total material on the board (white minus black denominator)
static constexpr int MAX_MATERIAL_VALUE = PIECE_VALUE[PAWN] * 8 + PIECE_VALUE[KNIGHT] * 2 +
                                          PIECE_VALUE[BISHOP] * 2 + PIECE_VALUE[ROOK] * 2 +
                                          PIECE_VALUE[QUEEN] * 1;

class Board {
public:
    Board(const std::string &fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") {
        // Initialize Position and StateInfo with the standard start position.
        setFen(fen);
    }

    /// Returns +1 if White to move, -1 if Black to move.
    int current_player() const { return (pos.side_to_move() == WHITE) ? +1 : -1; }

    /// Pushes a move onto the internal Position. Assumes move is legal.
    void make_move(Move m) {
        // Optionally, one could assert that 'm' is legal:
        // MoveList<LEGAL> ml(pos, st);
        // bool found = false;
        // for (Move mv : ml) if (mv == m) { found = true; break; }
        // assert(found && "Attempting to push an illegal move");
        moveHistory.add({m, StateInfo()});
        pos.do_move(m, moveHistory[moveHistory.size()-1].second, nullptr);
    }

    /// Returns true if no legal moves are available (checkmate or stalemate).
    bool is_game_over() const {
        // 3-fold-repetition and 50 move rul draw is handled outside move generation
        if (can_claim_3fold_repetition() || is_50_move_rule_draw() ||
            draw_by_insufficient_material()) {
            return true;
        }

        // Generate all legal moves from the current position.
        return MoveList<LEGAL>(pos).size() == 0;
    }

    /**
     * If the game is over and ended in checkmate, returns the winner:
     *   +1 if White won, -1 if Black won.
     * For stalemate or other draws, returns std::nullopt.
     */
    std::optional<int> check_winner() const {
        // If there is at least one legal move, the game is not over.
        if (MoveList<LEGAL>(pos).size() > 0) {
            return std::nullopt;
        }

        // No legal moves → either checkmate or stalemate.
        // pos.checkers() yields a bitboard of pieces that are giving check to the side to move.
        if (pos.checkers() != 0ULL) {
            // Side to move is in check and has no legal reply ⇒ checkmate.
            // If White was to move, Black delivered mate ⇒ Black wins (−1).
            // If Black was to move, White delivered mate ⇒ White wins (+1).
            return (pos.side_to_move() == WHITE) ? -1 : +1;
        } else {
            // No legal move and NOT in check ⇒ stalemate/draw.
            return std::nullopt;
        }
    }

    /**
     * Generates all legal moves, but filters out underpromotions (i.e., only allows
     * promotions to Queen, or non‐promotion moves).
     */
    std::vector<Move> get_valid_moves() const {
        const MoveList<LEGAL> ml(pos);
        std::vector<Move> filtered;
        filtered.reserve(ml.size());
        for (Move m : ml) {
            if (m.type_of() != PROMOTION || m.promotion_type() == QUEEN) {
                // If it's a promotion, only keep Queen promotions.
                filtered.push_back(m);
            }
        }
        return filtered;
    }

    /// Returns a 64‐bit Zobrist hash of the current position.
    std::uint64_t quick_hash() const { return pos.key(); }

    /**
     * Returns a material‐based heuristic score in [-1, +1]:
     *   +1 means White is up maximum material,
     *   -1 means Black is up maximum material,
     *   0 means material is equal.
     */
    double get_approximate_result_score() const {
        int mat_value = 0;
        // For each piece type, count white pieces minus black pieces.
        for (PieceType pt : PIECE_TYPES) {
            const Bitboard white_bb = pos.pieces(pt, WHITE);
            const Bitboard black_bb = pos.pieces(pt, BLACK);

            const int white_count = popcount(white_bb);
            const int black_count = popcount(black_bb);

            mat_value += (white_count - black_count) * PIECE_VALUE[pt];
        }
        return static_cast<double>(mat_value) / static_cast<double>(MAX_MATERIAL_VALUE);
    }

    std::string fen() const {
        // Returns the FEN string representation of the current position.
        return pos.fen();
    }

    /// Sets the internal Position from a FEN string.
    void setFen(const std::string &fen) {
        moveHistory.clear();
        pos.set(fen, false, &st);
    }

    const Position& position() const {
        // Returns a reference to the internal Position object.
        return pos;
    }

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
     *
     * White pieces are uppercase, black pieces lowercase, empty squares as '.'
     */
    std::string repr() const {
        std::ostringstream oss;
        constexpr int BOARD_LENGTH = 8;
        // File header
        oss << "  a b c d e f g h\n";
        // For i = 0..7, treat i as rank index 0=rank1, 7=rank8
        for (int i = 0; i < BOARD_LENGTH; ++i) {
            int rank = i + 1;
            oss << rank << " ";
            for (int file = 0; file < BOARD_LENGTH; ++file) {
                const Piece pc = pos.piece_on(square(file, i));
                if (pc != NO_PIECE) {
                    oss << piece_symbol(type_of(pc), color_of(pc));
                } else {
                    oss << ".";
                }
                if (file < BOARD_LENGTH - 1)
                    oss << " ";
            }
            oss << " " << rank << "\n";
        }
        // File footer
        oss << "  a b c d e f g h";
        return oss.str();
    }

private:
    Position pos;
    StateInfo st;
    ChunkedList<std::pair<Move, StateInfo>> moveHistory;

    static constexpr const char* piece_symbol(const PieceType pt, const Color c) {
        switch (pt) {
        case PAWN:
            return reinterpret_cast<const char*>(c == WHITE ? u8"♟" : u8"♙");
        case KNIGHT:
            return reinterpret_cast<const char*>(c == WHITE ? u8"♞" : u8"♘");
        case BISHOP:
            return reinterpret_cast<const char*>(c == WHITE ? u8"♝" : u8"♗");
        case ROOK:
            return reinterpret_cast<const char*>(c == WHITE ? u8"♜" : u8"♖");
        case QUEEN:
            return reinterpret_cast<const char*>(c == WHITE ? u8"♛" : u8"♕");
        case KING:
            return reinterpret_cast<const char*>(c == WHITE ? u8"♚" : u8"♔");
        default:
            return ".";
        }
    }

    bool draw_by_insufficient_material() const {
        // default early stopping
        if (pos.count<ALL_PIECES>() > 4) {
            return false;
        }

        // check for chess and atomic
        return (pos.count<ALL_PIECES>() == 2) ||                             // 1) KK
               (pos.count<ALL_PIECES>() == 3 && pos.count<BISHOP>() == 1) || // 2) KB vs K
               (pos.count<ALL_PIECES>() == 3 && pos.count<KNIGHT>() == 1) || // 3) KN vs K
               (pos.count<ALL_PIECES>() == 4 &&
                (pos.count<KNIGHT>(WHITE) == 2 || pos.count<KNIGHT>(BLACK) == 2)); // 4) KNN vs K
    }

    bool can_claim_3fold_repetition() const {
        // The repetition info stores the ply distance to the next previous
        // occurrence of the same position.
        // It is negative in the 3-fold case, or zero if the position was not repeated.
        return st.repetition < 0;
    }

    bool is_50_move_rule_draw() const {
        if (st.rule50 > 99 && (!pos.checkers() || MoveList<LEGAL>(pos).size())) {
            return true;
        }
        return false;
    }
};

// Overload operator<< so you can do: std::cout << board << "\n";
inline std::ostream &operator<<(std::ostream &os, const Board &board) {
    os << board.repr();
    return os;
}