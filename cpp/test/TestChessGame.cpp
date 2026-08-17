#include "TestRunner.hpp"
#include "games/chess/ChessGame.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testSearchStateIdentity() {
    const Board initial;
    const Board sameInitial;
    require(ChessGame::statesEqual(initial, sameInitial),
            "Equal chess rule states must share graph identity");
    require(ChessGame::stateHash(initial) == ChessGame::stateHash(sameInitial),
            "Equal chess rule states must have equal graph hashes");
    const Board differentFullmove("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 99");
    require(ChessGame::statesEqual(initial, differentFullmove),
            "Non-semantic chess fullmove display state must not split graph identity");

    const Board otherSide("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    require(!ChessGame::statesEqual(initial, otherSide),
            "Chess graph identity must include the side to move");

    const Board castling("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    const Board noCastling("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1");
    require(!ChessGame::statesEqual(castling, noCastling),
            "Chess graph identity must include castling rights");

    const Board enPassant("8/8/8/3pP3/8/8/4K2k/8 w - d6 0 1");
    const Board noEnPassant("8/8/8/3pP3/8/8/4K2k/8 w - - 0 1");
    require(!ChessGame::statesEqual(enPassant, noEnPassant),
            "Chess graph identity must include en-passant state");

    const Board halfmoveZero("8/8/8/8/8/8/4K2k/8 w - - 0 1");
    const Board halfmoveNinetyNine("8/8/8/8/8/8/4K2k/8 w - - 99 1");
    require(!ChessGame::statesEqual(halfmoveZero, halfmoveNinetyNine),
            "Chess graph identity must include the fifty-move clock");

    const Board repeated = Board::replay(initial.fen(), {"g1f3", "g8f6", "f3g1", "f6g8"});
    const Board historyBlind(repeated.fen());
    require(repeated.fen() == historyBlind.fen(), "Chess history fixture must share its FEN");
    require(!ChessGame::statesEqual(repeated, historyBlind),
            "Chess graph identity must include repetition-relevant history");

    const Board knightOrderA = Board::replay(initial.fen(), {"g1f3", "g8f6", "b1c3", "b8c6"});
    const Board knightOrderB = Board::replay(initial.fen(), {"b1c3", "b8c6", "g1f3", "g8f6"});
    require(knightOrderA.fen() == knightOrderB.fen(),
            "Commuting knight orders must reach the same current chess position");
    require(!ChessGame::statesEqual(knightOrderA, knightOrderB),
            "Different reversible histories must not share graph identity");

    const Board resetOrderA =
        Board::replay(initial.fen(), {"g1f3", "g8f6", "b1c3", "b8c6", "e2e4"});
    const Board resetOrderB =
        Board::replay(initial.fen(), {"b1c3", "b8c6", "g1f3", "g8f6", "e2e4"});
    require(ChessGame::statesEqual(resetOrderA, resetOrderB),
            "The same irreversible move must merge histories after their rule-state reset");
    require(ChessGame::stateHash(resetOrderA) == ChessGame::stateHash(resetOrderB),
            "Equal post-reset chess states must have equal graph hashes");
}
} // namespace

int runChessGameTests() {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();

    const Board initial;
    testSearchStateIdentity();
    require(!ChessGame::isTerminal(initial), "Initial chess position must not be terminal");
    require(static_cast<int>(ChessGame::legalActions(initial).size()) == 20,
            "Initial chess position must expose 20 legal actions");

    const CompressedEncodedBoard encoded = encodeBoard(initial);
    require(encoded.binary_planes.size() ==
                static_cast<std::size_t>(ChessRepresentationDimensions::binary_channel_count),
            "Chess contract binary channel count disagrees with the encoded position");
    require(encoded.scalar_planes.size() ==
                static_cast<std::size_t>(ChessRepresentationDimensions::scalar_channel_count),
            "Chess contract scalar channel count disagrees with the encoded position");

    const ChessAction firstAction = ChessGame::legalActions(initial).front();
    const int actionId = ChessGame::Encoding::actionId(firstAction, initial);
    require(0 <= actionId && actionId < ChessRepresentationDimensions::action_count,
            "Chess contract action id is outside the configured action space");
    require(ChessEncoding::decodeAction(actionId, initial) == firstAction,
            "Chess action decode did not invert encoding");
    const int mirroredActionId = ChessEncoding::mirrorActionId(actionId);
    require(ChessEncoding::mirrorActionId(mirroredActionId) == actionId,
            "Mirroring a chess action twice did not restore it");

    const Board child = ChessGame::childState(initial, firstAction);
    require(child.fen() != initial.fen(), "Chess child position must differ from its parent");

    const Board terminal = Board::replay(Board{}.fen(), {"f2f3", "e7e5", "g2g4", "d8h4"});
    require(ChessGame::isTerminal(terminal), "Checkmate sequence must produce a terminal position");
    require(ChessGame::terminalValue(terminal) == -1.0F,
            "Terminal chess value must preserve the existing result");

    const Board stalemate("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    require(ChessGame::isTerminal(stalemate), "Stalemate must be terminal");
    require(ChessGame::terminalValue(stalemate) == 0.0F, "Stalemate must be a draw");

    const Board fiftyMoveDraw("8/8/8/8/8/8/6k1/K5R1 w - - 100 75");
    require(ChessGame::isTerminal(fiftyMoveDraw), "The fifty-move boundary must be terminal");
    require(ChessGame::terminalValue(fiftyMoveDraw) == 0.0F,
            "The fifty-move boundary must be a draw");

    const Board insufficientMaterial("8/8/8/8/8/8/6k1/K7 w - - 0 1");
    require(ChessGame::isTerminal(insufficientMaterial), "Insufficient material must be terminal");
    require(ChessGame::terminalValue(insufficientMaterial) == 0.0F,
            "Insufficient material must be a draw");

    const Board promotedWhiteMaterial("QQQQQQQQ/8/8/8/8/8/8/4K2k w - - 0 1");
    require(promotedWhiteMaterial.approximateResultScore() == 1.0,
            "Promoted white material must remain within the WDL scalar range");
    const Board promotedBlackMaterial("4k2K/8/8/8/8/8/8/qqqqqqqq b - - 0 1");
    require(promotedBlackMaterial.approximateResultScore() == -1.0,
            "Promoted black material must remain within the WDL scalar range");
    std::cout << "Chess game tests passed\n";
    return 0;
}
