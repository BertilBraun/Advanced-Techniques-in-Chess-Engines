#pragma once

#include "search/InferenceTypes.hpp"
#include "games/chess/ChessAction.hpp"
#include "games/chess/ChessBoard.hpp"
#include "games/chess/ChessEncoding.hpp"

#include <optional>

class ChessGameContract {
public:
    using Position = Board;
    using Action = ChessAction;
    using EncodedPosition = EncodedChessPosition;

    [[nodiscard]] static constexpr InferenceDimensions inferenceDimensions() noexcept {
        return {BOARD_C, BOARD_LEN, BOARD_LEN, ACTION_SIZE, 3};
    }

    [[nodiscard]] static Position initialPosition() { return Position{}; }

    [[nodiscard]] static Position replayPosition(const std::string &starting_fen,
                                                 const std::vector<std::string> &moves_uci) {
        return Board::replay(starting_fen, moves_uci);
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

    [[nodiscard]] static std::optional<float> terminalValue(const Position &position) {
        if (!isTerminal(position)) {
            return std::nullopt;
        }
        return terminalResult(position);
    }

    [[nodiscard]] static int actionId(const Action action, const Position &position) {
        return chess_action_id(action, position);
    }

    [[nodiscard]] static std::vector<Action> decodeActions(const std::vector<int> &action_ids,
                                                           const Position &position) {
        return decode_chess_actions(action_ids, position);
    }

    [[nodiscard]] static EncodedPosition encodeInput(const Position &position) {
        return encode_chess_position(position);
    }

    static void encodeInputInto(const Position &position, int8 *destination) {
        encode_chess_position_into(position, destination);
    }
};
