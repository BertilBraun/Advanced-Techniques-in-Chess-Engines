#include "Board.h"

#include "bitboard.h"
#include "movegen.h"

#include "common.hpp"

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
static constexpr Bitboard DARK_SQUARES = 0xAA55AA55AA55AA55ULL;

Board::Board(const std::string &fen) {
    // Initialize Position and StateInfo with the standard start position.
    setFen(fen);
}

Board::Board(const Board &other) : m_pos(other.m_pos), m_history(other.m_history) {}

Board &Board::operator=(const Board &other) {
    if (this != &other) {
        m_pos = other.m_pos;
        m_history = other.m_history;
        m_validMoves.reset();
    }
    return *this;
}

void Board::makeMove(Move m) {
    TIMEIT("Board::makeMove");

    assert(contains(validMoves(), m) && "Attempting to push an illegal move");
    const bool pawnMove = type_of(m_pos.moved_piece(m)) == PAWN;
    const bool capture = m_pos.capture(m);
    const std::uint8_t castlingRightsBeforeMove = castlingRightsMask();
    m_pos.do_move(m);
    const bool castlingRightsChanged = castlingRightsMask() != castlingRightsBeforeMove;
    const bool resetsRepetitionHistory = pawnMove || capture || castlingRightsChanged;
    const bool historyIsAtCapacity =
        m_history->retainedPreviousPositions >= MAX_REVERSIBLE_HISTORY_PLIES;
    std::shared_ptr<const PositionHistory> previous =
        resetsRepetitionHistory || historyIsAtCapacity ? nullptr : m_history;
    m_history =
        std::make_shared<const PositionHistory>(m_pos.repetition_key(), std::move(previous));
    m_validMoves.reset();
    validateHistory();
}

void Board::setFen(const std::string &fen) {
    m_pos.set(fen, false);
    m_history = std::make_shared<const PositionHistory>(m_pos.repetition_key(), nullptr);
    m_validMoves.reset();
    validateHistory();
}

int Board::repetitionCount() const {
    validateHistory();
    int occurrences = 0;
    std::uint16_t inspectedPositions = 0;
    for (auto entry = m_history->previous;
         entry != nullptr && inspectedPositions < MAX_REVERSIBLE_HISTORY_PLIES;
         entry = entry->previous, ++inspectedPositions) {
        if (entry->key == m_history->key) {
            ++occurrences;
        }
    }
    return occurrences;
}

std::uint8_t Board::castlingRightsMask() const {
    return static_cast<std::uint8_t>(
        static_cast<std::uint8_t>(m_pos.can_castle(CastlingRights::WHITE_OO)) |
        (static_cast<std::uint8_t>(m_pos.can_castle(CastlingRights::WHITE_OOO)) << 1) |
        (static_cast<std::uint8_t>(m_pos.can_castle(CastlingRights::BLACK_OO)) << 2) |
        (static_cast<std::uint8_t>(m_pos.can_castle(CastlingRights::BLACK_OOO)) << 3));
}

void Board::validateHistory() const {
    assert(m_history != nullptr && "Board history must be initialized");
    assert(m_history->key == m_pos.repetition_key() &&
           "Board history head must describe the current position");
    assert(m_history->retainedPreviousPositions <= MAX_REVERSIBLE_HISTORY_PLIES &&
           "Board history exceeded its bounded reversible window");
    assert((m_history->previous == nullptr) == (m_history->retainedPreviousPositions == 0) &&
           "Board history depth is inconsistent");
}

bool Board::isGameOver() const {
    TIMEIT("Board::isGameOver");

    const std::vector<Move> &legalMoves = validMoves();
    const bool fiftyMoveDraw =
        m_pos.rule50_count() >= 100 && (m_pos.checkers() == 0ULL || !legalMoves.empty());
    if (fiftyMoveDraw || repetitionCount() >= 2 || drawByInsufficientMaterial()) {
        return true;
    }

    return legalMoves.empty();
}

std::optional<int> Board::checkWinner() const {
    TIMEIT("Board::checkWinner");

    // If there is at least one legal move, the game is not over.
    if (!validMoves().empty()) {
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
const std::vector<Move> &Board::validMoves() const {
    TIMEIT("Board::validMoves");

    if (m_validMoves.has_value()) {
        return *m_validMoves;
    }

    const MoveList<LEGAL> ml(m_pos);
    std::vector<Move> legalMoves;
    legalMoves.reserve(ml.size());
    for (auto m : ml) {
        legalMoves.emplace_back(m.raw());
    }
    m_validMoves.emplace(std::move(legalMoves));
    return *m_validMoves;
}

double Board::approximateResultScore() const {
    int materialValue = 0;
    // For each piece type, count white pieces minus black pieces.
    for (const PieceType pt : PIECE_TYPES) {
        const Bitboard whiteBB = m_pos.pieces(WHITE, pt);
        const Bitboard blackBB = m_pos.pieces(BLACK, pt);

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
    if (m_pos.count<PAWN>() > 0 || m_pos.count<ROOK>() > 0 || m_pos.count<QUEEN>() > 0) {
        return false;
    }

    const int knightCount = m_pos.count<KNIGHT>();
    const int bishopCount = m_pos.count<BISHOP>();
    if (bishopCount == 0) {
        return knightCount == 0 || (knightCount <= 2 && (m_pos.count<KNIGHT>(WHITE) == 0 ||
                                                         m_pos.count<KNIGHT>(BLACK) == 0));
    }
    if (knightCount > 0) {
        return false;
    }

    const Bitboard bishops = m_pos.pieces(BISHOP);
    return (bishops & DARK_SQUARES) == 0 || (bishops & ~DARK_SQUARES) == 0;
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
