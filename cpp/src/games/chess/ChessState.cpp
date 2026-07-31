#include "games/chess/ChessState.hpp"

#include "games/chess/ChessEncoding.hpp"

#include <algorithm>
#include <bitboard.h>
#include <limits>
#include <movegen.h>
#include <mutex>

namespace az::games::chess {
namespace {

constexpr uint64 FNV_OFFSET = 14695981039346656037ULL;
constexpr uint64 FNV_PRIME = 1099511628211ULL;
constexpr Stockfish::Bitboard DARK_SQUARES = 0xAA55AA55AA55AA55ULL;

void hashUint64(uint64 &hash, uint64 value) {
    for (int32 byte = 0; byte < 8; ++byte) {
        hash ^= static_cast<uint8>(value >> (byte * 8));
        hash *= FNV_PRIME;
    }
}

} // namespace

void ChessState::initializeStockfish() {
    static std::once_flag initialized;
    std::call_once(initialized, []() {
        Stockfish::Bitboards::init();
        Stockfish::Position::init();
    });
}

void ChessState::validateRules(const ChessRules &rules) {
    if (rules.startingFen.empty()) {
        throw std::invalid_argument("chess starting FEN must not be empty");
    }
    if (rules.halfmoveDrawPlyCount < 1 || rules.halfmoveDrawPlyCount >= MAXIMUM_HISTORY_POSITIONS) {
        throw std::invalid_argument("chess halfmove draw count must be in [1, 150]");
    }
    if (rules.safetyPlyCap < 1) {
        throw std::invalid_argument("chess safety ply cap must be positive");
    }
}

ChessState::ChessState(ChessRules rules) : _rules(std::move(rules)) {
    validateRules(_rules);
    initializeStockfish();
    _position.set(_rules.startingFen, false);
    _repetitionKeys[0] = _position.repetition_key();
    _historyCount = 1;
}

ChessState::ChessState(const ChessState &other)
    : _rules(other._rules), _position(other._position), _repetitionKeys(other._repetitionKeys),
      _historyCount(other._historyCount), _ply(other._ply) {}

ChessState &ChessState::operator=(const ChessState &other) {
    if (this != &other) {
        _rules = other._rules;
        _position = other._position;
        _repetitionKeys = other._repetitionKeys;
        _historyCount = other._historyCount;
        _ply = other._ply;
    }
    return *this;
}

const ChessRules &ChessState::rules() const { return _rules; }

int32 ChessState::actionCount() const { return CHESS_ACTION_COUNT; }

std::vector<Stockfish::Move> ChessState::legalMoves() const {
    const Stockfish::MoveList<Stockfish::LEGAL> moves(_position);
    std::vector<Stockfish::Move> result;
    result.reserve(moves.size());
    for (const Stockfish::Move move : moves) {
        result.emplace_back(move.raw());
    }
    return result;
}

std::vector<int32> ChessState::legalActions() const {
    if (isTerminal()) {
        return {};
    }
    const std::vector<Stockfish::Move> moves = legalMoves();
    std::vector<int32> actions;
    actions.reserve(moves.size());
    for (const Stockfish::Move move : moves) {
        actions.push_back(encodeMove(move, _position.side_to_move()));
    }
    return actions;
}

bool ChessState::isLegal(int32 action) const {
    return !isTerminal() && decodeLegalMove(action, _position).has_value();
}

uint8 ChessState::castlingRightsMask() const {
    return static_cast<uint8>(
        static_cast<uint8>(_position.can_castle(Stockfish::WHITE_OO)) |
        (static_cast<uint8>(_position.can_castle(Stockfish::WHITE_OOO)) << 1U) |
        (static_cast<uint8>(_position.can_castle(Stockfish::BLACK_OO)) << 2U) |
        (static_cast<uint8>(_position.can_castle(Stockfish::BLACK_OOO)) << 3U));
}

void ChessState::appendHistory(bool resetHistory) {
    if (resetHistory) {
        _historyCount = 1;
        _repetitionKeys[0] = _position.repetition_key();
        return;
    }
    assert(_historyCount < MAXIMUM_HISTORY_POSITIONS);
    _repetitionKeys[_historyCount++] = _position.repetition_key();
}

void ChessState::apply(int32 action) {
    if (isTerminal()) {
        throw std::invalid_argument("cannot apply an action to a terminal chess state");
    }
    const std::optional<Stockfish::Move> move = decodeLegalMove(action, _position);
    if (!move.has_value()) {
        throw std::invalid_argument("illegal chess action");
    }
    const bool pawnMove = Stockfish::type_of(_position.moved_piece(*move)) == Stockfish::PAWN;
    const bool capture = _position.capture(*move);
    const uint8 castlingBefore = castlingRightsMask();
    _position.do_move(*move);
    const bool castlingChanged = castlingBefore != castlingRightsMask();
    appendHistory(pawnMove || capture || castlingChanged);
    ++_ply;
}

Player ChessState::currentPlayer() const {
    return _position.side_to_move() == Stockfish::WHITE ? Player::White : Player::Black;
}

int32 ChessState::ply() const { return _ply; }

int32 ChessState::repetitionCount() const {
    int32 repetitions = 0;
    const uint64 current = _repetitionKeys[_historyCount - 1U];
    for (uint16 index = 0; index + 1U < _historyCount; ++index) {
        if (_repetitionKeys[index] == current) {
            ++repetitions;
        }
    }
    return repetitions;
}

bool ChessState::hasInsufficientMaterial() const {
    if (_position.count<Stockfish::PAWN>() > 0 || _position.count<Stockfish::ROOK>() > 0 ||
        _position.count<Stockfish::QUEEN>() > 0) {
        return false;
    }
    const int32 knights = _position.count<Stockfish::KNIGHT>();
    const int32 bishops = _position.count<Stockfish::BISHOP>();
    if (bishops == 0) {
        return knights <= 1;
    }
    if (knights > 0) {
        return false;
    }
    const Stockfish::Bitboard bishopSquares = _position.pieces(Stockfish::BISHOP);
    return (bishopSquares & DARK_SQUARES) == 0 || (bishopSquares & ~DARK_SQUARES) == 0;
}

TerminationReason ChessState::terminationReason() const {
    const std::vector<Stockfish::Move> moves = legalMoves();
    if (moves.empty()) {
        return _position.checkers() != 0 ? TerminationReason::Checkmate
                                         : TerminationReason::Stalemate;
    }
    if (repetitionCount() >= 2) {
        return TerminationReason::ThreefoldRepetition;
    }
    if (_position.rule50_count() >= _rules.halfmoveDrawPlyCount) {
        return TerminationReason::HalfmoveRule;
    }
    if (hasInsufficientMaterial()) {
        return TerminationReason::InsufficientMaterial;
    }
    if (_ply >= _rules.safetyPlyCap) {
        return TerminationReason::SafetyPlyCap;
    }
    return TerminationReason::Ongoing;
}

bool ChessState::isTerminal() const { return terminationReason() != TerminationReason::Ongoing; }

TerminalResult ChessState::terminalResult() const {
    const TerminationReason reason = terminationReason();
    if (reason != TerminationReason::Checkmate) {
        return TerminalResult{.reason = reason, .winner = std::nullopt};
    }
    const Player winner = currentPlayer() == Player::White ? Player::Black : Player::White;
    return TerminalResult{.reason = reason, .winner = winner};
}

uint64 ChessState::stateHash() const {
    uint64 hash = FNV_OFFSET;
    hashUint64(hash, _position.key());
    hashUint64(hash, static_cast<uint64>(_position.rule50_count()));
    hashUint64(hash, static_cast<uint64>(_rules.halfmoveDrawPlyCount));
    hashUint64(hash, static_cast<uint64>(_rules.safetyPlyCap));
    hashUint64(hash, static_cast<uint64>(_ply));
    for (uint16 index = 0; index < _historyCount; ++index) {
        hashUint64(hash, _repetitionKeys[index]);
    }
    return hash;
}

const Stockfish::Position &ChessState::position() const { return _position; }

bool ChessState::operator==(const ChessState &other) const {
    return _rules == other._rules && _position.fen() == other._position.fen() &&
           _historyCount == other._historyCount && _ply == other._ply &&
           std::equal(_repetitionKeys.begin(), _repetitionKeys.begin() + _historyCount,
                      other._repetitionKeys.begin());
}

} // namespace az::games::chess
