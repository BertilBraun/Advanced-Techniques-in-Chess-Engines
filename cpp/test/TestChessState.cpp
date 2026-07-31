#include "core/SessionConfiguration.hpp"
#include "games/GameDefinition.hpp"
#include "games/chess/ChessDefinition.hpp"
#include "games/chess/ChessPolicy.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <movegen.h>
#include <set>
#include <vector>

using az::games::chess::ChessDefinition;
using az::games::chess::ChessEncoding;
using az::games::chess::ChessRules;
using az::games::chess::ChessState;
using az::games::chess::Player;
using az::games::chess::TerminationReason;

static_assert(az::games::GameDefinition<ChessDefinition>);

namespace {

constexpr const char *STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

ChessRules rules(std::string fen = STARTING_FEN, int32 halfmoveDrawPlies = 100) {
    return ChessRules{
        .startingFen = std::move(fen),
        .halfmoveDrawPlyCount = halfmoveDrawPlies,
        .safetyPlyCap = 1024,
    };
}

void applyMove(ChessState &state, Stockfish::Square from, Stockfish::Square to,
               Stockfish::PieceType promotion = Stockfish::NO_PIECE_TYPE) {
    for (const Stockfish::Move move : Stockfish::MoveList<Stockfish::LEGAL>(state.position())) {
        const Stockfish::PieceType actualPromotion = move.type_of() == Stockfish::PROMOTION
                                                         ? move.promotion_type()
                                                         : Stockfish::NO_PIECE_TYPE;
        if (move.from_sq() == from && move.to_sq() == to && actualPromotion == promotion) {
            state.apply(az::games::chess::encodeMove(move, state.position().side_to_move()));
            return;
        }
    }
    throw std::logic_error("fixture move is not legal: " + std::to_string(from) + "-" +
                           std::to_string(to));
}

void testInitialPositionAndEncoding() {
    ChessState state(rules());
    const std::vector<int32> legal = state.legalActions();
    const std::set<int32> expectedActions{
        44,  46,  168, 170, 194, 195, 218, 219, 245, 246,
        274, 275, 303, 304, 332, 333, 361, 362, 388, 389,
    };
    assert(legal.size() == 20);
    assert(std::set<int32>(legal.begin(), legal.end()) == expectedActions);
    assert(state.currentPlayer() == Player::White);
    assert(!state.isTerminal());
    assert(state.actionCount() == 1880);

    const ChessEncoding encoding = state.canonicalEncoding();
    for (int32 column = 0; column < 8; ++column) {
        assert(encoding.at(0, 1, column) == 1);
        assert(encoding.at(6, 6, column) == 1);
        assert(encoding.at(16, 1, column) == 1);
        assert(encoding.at(17, 6, column) == 1);
    }
    assert(encoding.at(12, 0, 0) == 1);
    assert(encoding.at(13, 0, 0) == 1);
    assert(encoding.at(14, 0, 0) == 1);
    assert(encoding.at(15, 0, 0) == 1);
    assert(encoding.at(28, 0, 0) == 0);

    applyMove(state, Stockfish::SQ_E2, Stockfish::SQ_E4);
    const ChessEncoding blackEncoding = state.canonicalEncoding();
    assert(state.currentPlayer() == Player::Black);
    for (int32 column = 0; column < 8; ++column) {
        assert(blackEncoding.at(0, 1, column) == 1);
    }
}

void testRepetitionAndHistoryHash() {
    ChessState state(rules());
    const uint64 initialHash = state.stateHash();
    for (int32 cycle = 0; cycle < 2; ++cycle) {
        applyMove(state, Stockfish::SQ_G1, Stockfish::SQ_F3);
        applyMove(state, Stockfish::SQ_G8, Stockfish::SQ_F6);
        applyMove(state, Stockfish::SQ_F3, Stockfish::SQ_G1);
        applyMove(state, Stockfish::SQ_F6, Stockfish::SQ_G8);
    }
    assert(state.repetitionCount() == 2);
    assert(state.terminationReason() == TerminationReason::ThreefoldRepetition);
    assert(state.stateHash() != initialHash);
    assert(ChessDefinition::terminalValue(state) == 0.0);
}

void testSpecialMoves() {
    ChessState castling(rules("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"));
    applyMove(castling, Stockfish::SQ_E1, Stockfish::SQ_H1);
    assert(castling.position().piece_on(Stockfish::SQ_G1) == Stockfish::W_KING);
    assert(castling.position().piece_on(Stockfish::SQ_F1) == Stockfish::W_ROOK);

    ChessState enPassant(rules("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"));
    applyMove(enPassant, Stockfish::SQ_E5, Stockfish::SQ_D6);
    assert(enPassant.position().piece_on(Stockfish::SQ_D6) == Stockfish::W_PAWN);
    assert(enPassant.position().piece_on(Stockfish::SQ_D5) == Stockfish::NO_PIECE);

    ChessState promotion(rules("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"));
    applyMove(promotion, Stockfish::SQ_A7, Stockfish::SQ_A8, Stockfish::KNIGHT);
    assert(promotion.position().piece_on(Stockfish::SQ_A8) == Stockfish::W_KNIGHT);
}

void testTerminalSemantics() {
    ChessState checkmate(rules("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"));
    assert(checkmate.terminationReason() == TerminationReason::Checkmate);
    assert(checkmate.terminalResult().winner == Player::White);
    assert(ChessDefinition::terminalValue(checkmate) == -1.0);

    ChessState stalemate(rules("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"));
    assert(stalemate.terminationReason() == TerminationReason::Stalemate);
    assert(ChessDefinition::terminalValue(stalemate) == 0.0);

    ChessState insufficient(rules("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
    assert(insufficient.terminationReason() == TerminationReason::InsufficientMaterial);

    ChessState halfmove(rules("4k2r/8/8/8/8/8/8/R3K3 w - - 100 1"));
    assert(halfmove.terminationReason() == TerminationReason::HalfmoveRule);

    ChessState seventyFiveMove(rules("4k2r/8/8/8/8/8/8/R3K3 w - - 150 1", 150));
    assert(seventyFiveMove.terminationReason() == TerminationReason::HalfmoveRule);
}

} // namespace

int main() {
    testInitialPositionAndEncoding();
    testRepetitionAndHistoryHash();
    testSpecialMoves();
    testTerminalSemantics();
    return 0;
}
