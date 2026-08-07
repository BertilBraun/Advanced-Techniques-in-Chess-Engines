#pragma once

#include "games/chess/ChessAction.hpp"
#include "games/chess/ChessBoard.hpp"
#include "games/chess/ChessEncoding.hpp"
#include "search/InferenceTypes.hpp"

#include <optional>

class ChessGameContract {
public:
    using Position = Board;
    using Action = ChessAction;
    using EncodedPosition = CompressedEncodedBoard;

    [[nodiscard]] static constexpr InferenceDimensions inferenceDimensions() noexcept {
        return {
            .channels = ChessRepresentationDimensions::channel_count,
            .rows = ChessRepresentationDimensions::board_length,
            .columns = ChessRepresentationDimensions::board_length,
            .actions = ChessAction::action_count,
            .outcomes = 3,
        };
    }

    [[nodiscard]] static constexpr float searchTurnDiscount() noexcept { return 0.99F; }

    [[nodiscard]] static Position initialPosition() { return Position{}; }

    [[nodiscard]] static Position replayPosition(const std::string &starting_fen,
                                                 const std::vector<std::string> &moves_uci) {
        return Board::replay(starting_fen, moves_uci);
    }

    [[nodiscard]] static Position childPosition(const Position &parent, const Action action) {
        Position child(parent);
        child.makeMove(action.move);
        return child;
    }

    [[nodiscard]] static std::vector<Action> legalActions(const Position &position) {
        std::vector<Action> actions;
        actions.reserve(position.validMoves().size());
        for (const Stockfish::Move move : position.validMoves()) {
            actions.emplace_back(move);
        }
        return actions;
    }

    [[nodiscard]] static bool isTerminal(const Position &position) { return position.isGameOver(); }

    [[nodiscard]] static float terminalResult(const Position &position) {
        return chessTerminalValue(position);
    }

    [[nodiscard]] static std::optional<float> terminalValue(const Position &position) {
        if (!isTerminal(position)) {
            return std::nullopt;
        }
        return terminalResult(position);
    }

    [[nodiscard]] static int actionId(const Action action, const Position &position) {
        return ChessActionCodec::encode(action, position);
    }

    [[nodiscard]] static std::vector<Action> decodeActions(const std::vector<int> &action_ids,
                                                           const Position &position) {
        return ChessActionCodec::decode(action_ids, position);
    }

    [[nodiscard]] static EncodedPosition encodeInput(const Position &position) {
        return encodeBoard(position);
    }

    static void encodeInputInto(const Position &position, std::int8_t *destination) {
        encodeBoardInto(position, destination);
    }
};
