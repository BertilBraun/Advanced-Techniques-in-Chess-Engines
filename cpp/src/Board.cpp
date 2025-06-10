#include "Board.h"

#include "bitboard.h"
#include "movegen.h"

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

Board::Board(const std::string &fen) {
    // Initialize Position and StateInfo with the standard start position.
    setFen(fen);
}

void Board::makeMove(Move m) {
    TimeItGuard timer("Board::makeMove");

    // Optionally, one could assert that 'm' is legal:
    assert(contains(validMoves(), m) && "Attempting to push an illegal move");
    m_pos.do_move(m);
}

bool Board::isGameOver() const {
    TimeItGuard timer("Board::isGameOver");

    // 3-fold-repetition and 50 move rul draw is handled outside move generation
    if (m_pos.is_draw() || drawByInsufficientMaterial()) {
        return true;
    }

    // Generate all legal moves from the current position.
    return MoveList<LEGAL>(m_pos).size() == 0;
}

std::optional<int> Board::checkWinner() const {
    TimeItGuard timer("Board::checkWinner");

    // If there is at least one legal move, the game is not over.
    if (MoveList<LEGAL>(m_pos).size() > 0) {
        return std::nullopt;
    }

    // No legal moves → either checkmate or stalemate.
    // pos.checkers() yields a bitboard of pieces that are giving check to the side to move.
    if (m_pos.checkers() != 0ULL) {
        // Side to move is in check and has no legal reply ⇒ checkmate.
        // If White was to move, Black delivered mate ⇒ Black wins (−1).
        // If Black was to move, White delivered mate ⇒ White wins (+1).
        return (m_pos.side_to_move() == WHITE) ? -1 : +1;
    } else {
        // No legal move and NOT in check ⇒ stalemate/draw.
        return std::nullopt;
    }
}
std::vector<Move> Board::validMoves() const {
    TimeItGuard timer("Board::validMoves");

    const MoveList<LEGAL> ml(m_pos);
    std::vector<Move> filtered;
    filtered.reserve(ml.size());
    for (auto m : ml) {
        const Move move(m.raw()); // Create a Move out of the ExtMove object just to be safe.
        if (move.type_of() != PROMOTION || move.promotion_type() == QUEEN) {
            // If it's a promotion, only keep Queen promotions.
            filtered.push_back(move);
        }
    }
    return filtered;
}

double Board::approximateResultScore() const {
    int materialValue = 0;
    // For each piece type, count white pieces minus black pieces.
    for (PieceType pt : PIECE_TYPES) {
        const Bitboard whiteBB = m_pos.pieces(pt, WHITE);
        const Bitboard blackBB = m_pos.pieces(pt, BLACK);

        const int white_count = popcount(whiteBB);
        const int black_count = popcount(blackBB);

        materialValue += (white_count - black_count) * PIECE_VALUE[pt];
    }
    return static_cast<double>(materialValue) / static_cast<double>(MAX_MATERIAL_VALUE);
}

std::string Board::repr() const {
    std::ostringstream oss;
    // File header
    oss << "  a b c d e f g h\n";
    // For i = 0..7, treat i as rank index 0=rank1, 7=rank8
    for (int i = 0; i < BOARD_LENGTH; ++i) {
        int rank = i + 1;
        oss << rank << " ";
        for (int file = 0; file < BOARD_LENGTH; ++file) {
            const Piece pc = m_pos.piece_on(square(file, i));
            if (pc != NO_PIECE) {
                oss << pieceSymbol(type_of(pc), color_of(pc));
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
bool Board::drawByInsufficientMaterial() const {
    // default early stopping
    if (m_pos.count<ALL_PIECES>() > 4) {
        return false;
    }

    // check for chess and atomic
    return (m_pos.count<ALL_PIECES>() == 2) ||                               // 1) KK
           (m_pos.count<ALL_PIECES>() == 3 && m_pos.count<BISHOP>() == 1) || // 2) KB vs K
           (m_pos.count<ALL_PIECES>() == 3 && m_pos.count<KNIGHT>() == 1) || // 3) KN vs K
           (m_pos.count<ALL_PIECES>() == 4 &&
            (m_pos.count<KNIGHT>(WHITE) == 2 || m_pos.count<KNIGHT>(BLACK) == 2)); // 4) KNN vs K
}

constexpr const char *Board::pieceSymbol(const PieceType pt, const Color c) {
    switch (pt) {
    case PAWN:
        return reinterpret_cast<const char *>(c == WHITE ? u8"♟" : u8"♙");
    case KNIGHT:
        return reinterpret_cast<const char *>(c == WHITE ? u8"♞" : u8"♘");
    case BISHOP:
        return reinterpret_cast<const char *>(c == WHITE ? u8"♝" : u8"♗");
    case ROOK:
        return reinterpret_cast<const char *>(c == WHITE ? u8"♜" : u8"♖");
    case QUEEN:
        return reinterpret_cast<const char *>(c == WHITE ? u8"♛" : u8"♕");
    case KING:
        return reinterpret_cast<const char *>(c == WHITE ? u8"♚" : u8"♔");
    default:
        return ".";
    }
}