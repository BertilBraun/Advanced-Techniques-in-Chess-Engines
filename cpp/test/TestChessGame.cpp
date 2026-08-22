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
} // namespace

int runChessGameTests() {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();

    const Board initial;
    require(!ChessGame::isTerminal(initial), "Initial chess position must not be terminal");
    require(static_cast<int>(ChessGame::legalActions(initial).size()) == 20,
            "Initial chess position must expose 20 legal actions");

    const CompressedEncodedBoard encoded = encodeBoard(initial);
    require(encoded.binaryPlanes.size() ==
                static_cast<std::size_t>(ChessRepresentationDimensions::binary_channel_count),
            "Chess contract binary channel count disagrees with the encoded position");
    require(encoded.scalarPlanes.size() ==
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
    require(ChessGame::isTerminal(insufficientMaterial),
            "Insufficient material must be terminal");
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
