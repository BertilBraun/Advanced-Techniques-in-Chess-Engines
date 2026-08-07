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

    const Board child = ChessGame::childState(initial, firstAction);
    require(child.fen() != initial.fen(), "Chess child position must differ from its parent");

    const Board terminal = Board::replay(Board{}.fen(), {"f2f3", "e7e5", "g2g4", "d8h4"});
    require(ChessGame::isTerminal(terminal), "Checkmate sequence must produce a terminal position");
    require(ChessGame::terminalValue(terminal) == -1.0F,
            "Terminal chess value must preserve the existing result");
    std::cout << "Chess game tests passed\n";
    return 0;
}
