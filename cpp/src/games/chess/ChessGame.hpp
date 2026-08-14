#pragma once

#include "games/GameConcepts.hpp"
#include "games/chess/encoding/ChessEncoding.hpp"
#include "games/chess/implementation/ChessAction.hpp"
#include "games/chess/implementation/ChessBoard.hpp"

#include <cassert>

struct ChessGame {
    using State = Board;
    using Action = ChessAction;
    using Encoding = ChessEncoding;

    [[nodiscard]] static State childState(const State &parent, const Action action) {
        State child(parent);
        child.makeMove(action.move);
        return child;
    }

    [[nodiscard]] static std::vector<Action> legalActions(const State &state) {
        std::vector<Action> actions;
        actions.reserve(state.validMoves().size());
        for (const Stockfish::Move move : state.validMoves()) {
            actions.emplace_back(move);
        }
        return actions;
    }

    [[nodiscard]] static bool isTerminal(const State &state) { return state.isGameOver(); }

    [[nodiscard]] static float terminalValue(const State &state) {
        assert(isTerminal(state));
        const std::optional<int> winner = state.checkWinner();
        if (!winner.has_value()) {
            return 0.0F;
        }
        return *winner == state.currentPlayer() ? 1.0F : -1.0F;
    }
};

static_assert(SearchGame<ChessGame>);
