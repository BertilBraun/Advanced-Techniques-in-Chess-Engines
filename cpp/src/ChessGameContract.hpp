#pragma once

#include "Board.h"
#include "BoardEncoding.hpp"
#include "GameHistory.hpp"
#include "MoveEncoding.hpp"

struct ChessRepresentationDimensions {
    static constexpr int board_length = BOARD_LEN;
    static constexpr int channel_count = BOARD_C;
    static constexpr int binary_channel_count = BINARY_C;
    static constexpr int scalar_channel_count = SCALAR_C;
    static constexpr int action_count = ACTION_SIZE;
};

class ChessGameContract {
public:
    using Position = Board;
    using Action = Move;
    using EncodedPosition = CompressedEncodedBoard;

    [[nodiscard]] static Position initialPosition() { return Position{}; }

    [[nodiscard]] static Position replayPosition(const std::string &startingFen,
                                                const std::vector<std::string> &movesUci) {
        return replayMoves(startingFen, movesUci);
    }

    [[nodiscard]] static Position childPosition(const Position &parent, const Action action) {
        Position child(parent);
        child.makeMove(action);
        return child;
    }

    [[nodiscard]] static const std::vector<Action> &legalActions(const Position &position) {
        return position.validMoves();
    }

    [[nodiscard]] static bool isTerminal(const Position &position) { return position.isGameOver(); }

    [[nodiscard]] static float terminalResult(const Position &position) {
        return getBoardResultScore(position);
    }

    [[nodiscard]] static int actionId(const Action action, const Position &position) {
        return encodeMove(action, &position);
    }

    [[nodiscard]] static std::vector<Action> decodeActions(const std::vector<int> &actionIds,
                                                           const Position &position) {
        return decodeMoves(actionIds, &position);
    }

    [[nodiscard]] static EncodedPosition encodeInput(const Position &position) {
        return encodeBoard(&position);
    }

    static void encodeInputInto(const Position &position, int8 *destination) {
        encodeBoardInto(position, destination);
    }
};
