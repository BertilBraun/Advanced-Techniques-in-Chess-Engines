#pragma once

#include "games/GameConcepts.hpp"
#include "games/go/encoding/GoEncoding.hpp"
#include "games/go/implementation/GoPosition.hpp"

#include <cassert>
#include <cstddef>
#include <vector>

template <std::size_t BoardSize, std::size_t HistoryLength = 8> struct GoGame {
    using State = GoPosition<BoardSize, HistoryLength>;
    using Action = GoAction<BoardSize>;
    using Encoding = GoEncoding<BoardSize, HistoryLength>;

    [[nodiscard]] static State childState(const State &parent, const Action action) {
        return parent.child(action);
    }
    [[nodiscard]] static std::vector<Action> legalActions(const State &state) {
        return state.legal_actions();
    }
    [[nodiscard]] static bool isTerminal(const State &state) noexcept {
        return state.is_terminal();
    }
    [[nodiscard]] static float terminalValue(const State &state) {
        assert(isTerminal(state));
        const GoTerminalResult result = state.terminal_result();
        if (!result.winner.has_value()) {
            return 0.0F;
        }
        return *result.winner == state.player() ? 1.0F : -1.0F;
    }
};

using Go7Game = GoGame<7, 8>;
using Go9Game = GoGame<9, 8>;

static_assert(SearchGame<Go7Game>);
static_assert(SearchGame<Go9Game>);
