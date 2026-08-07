#include "games/chess/ChessGameContract.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
} // namespace

int main() {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();

    const Board initial = ChessGameContract::initialPosition();
    require(!ChessGameContract::isTerminal(initial), "Initial chess position must not be terminal");
    require(static_cast<int>(ChessGameContract::legalActions(initial).size()) == 20,
            "Initial chess position must expose 20 legal actions");

    const CompressedEncodedBoard encoded = ChessGameContract::encodeInput(initial);
    require(encoded.bits.size() ==
                static_cast<std::size_t>(ChessRepresentationDimensions::binary_channel_count),
            "Chess contract binary channel count disagrees with the encoded position");
    require(encoded.scal.size() ==
                static_cast<std::size_t>(ChessRepresentationDimensions::scalar_channel_count),
            "Chess contract scalar channel count disagrees with the encoded position");

    const ChessAction firstAction = ChessGameContract::legalActions(initial).front();
    const int actionId = ChessGameContract::actionId(firstAction, initial);
    require(0 <= actionId && actionId < ChessRepresentationDimensions::action_count,
            "Chess contract action id is outside the configured action space");
    const std::vector<ChessAction> decodedActions =
        ChessGameContract::decodeActions({actionId}, initial);
    require(decodedActions.size() == 1, "Chess contract failed to decode the encoded action");
    require(decodedActions.front() == firstAction,
            "Chess contract changed the encoded chess action");

    const Board child = ChessGameContract::childPosition(initial, firstAction);
    require(child.fen() != initial.fen(), "Chess child position must differ from its parent");

    const Board terminal =
        ChessGameContract::replayPosition(Board{}.fen(), {"f2f3", "e7e5", "g2g4", "d8h4"});
    require(ChessGameContract::isTerminal(terminal),
            "Checkmate sequence must produce a terminal position");
    require(ChessGameContract::terminalResult(terminal) == -1.0F,
            "Checkmate must score as a loss for the side to move");
    require(ChessGameContract::terminalValue(terminal) == -1.0F,
            "Terminal chess value must preserve the existing result");
    require(!ChessGameContract::terminalValue(initial).has_value(),
            "Non-terminal chess positions must not expose a terminal value");

    std::cout << "Chess game contract tests passed\n";
    return 0;
}
