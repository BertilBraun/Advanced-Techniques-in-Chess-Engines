#include "games/chess/ChessDefinition.hpp"

namespace az::games::chess {

core::GameIdentity ChessDefinition::identity() { return core::GameIdentity::Chess; }

uint32 ChessDefinition::gameSchemaVersion() { return 1; }

uint32 ChessDefinition::replaySchemaVersion() { return 1; }

std::string_view ChessDefinition::policySpaceIdentity() {
    return "chess_canonical_1880_move_map_v1";
}

ChessDefinition::State ChessDefinition::createInitialState(const Rules &rules) {
    return State(rules);
}

int32 ChessDefinition::actionCount(const Rules &rules) {
    static_cast<void>(State(rules));
    return CHESS_ACTION_COUNT;
}

core::TensorSpecification ChessDefinition::inputSpecification(const Rules &rules) {
    static_cast<void>(State(rules));
    return core::TensorSpecification{
        .dataType = core::TensorDataType::Int8,
        .layout = core::TensorLayout::ChannelsFirst,
        .dimensions = {CHESS_ENCODING_PLANES, CHESS_BOARD_SIZE, CHESS_BOARD_SIZE},
    };
}

int32 ChessDefinition::actionToPolicy(Action action, const Rules &rules) {
    static_cast<void>(rules);
    if (action < 0 || action >= CHESS_ACTION_COUNT) {
        throw std::invalid_argument("chess action is outside the policy space");
    }
    return action;
}

ChessDefinition::Action ChessDefinition::policyToAction(int32 policyIndex, const Rules &rules) {
    return actionToPolicy(policyIndex, rules);
}

std::span<const ChessDefinition::Symmetry> ChessDefinition::validSymmetries() { return SYMMETRIES; }

ChessDefinition::Action ChessDefinition::transformAction(Action action, const Rules &rules,
                                                         Symmetry symmetry) {
    static_cast<void>(symmetry);
    return actionToPolicy(action, rules);
}

ChessDefinition::Encoding ChessDefinition::transformEncoding(const Encoding &encoding,
                                                             Symmetry symmetry) {
    static_cast<void>(symmetry);
    return encoding;
}

ChessDefinition::ReplayPayload ChessDefinition::replayPayload(const State &state) {
    return state.canonicalEncoding();
}

std::optional<double> ChessDefinition::terminalValue(const State &state) {
    const TerminationReason reason = state.terminationReason();
    if (reason == TerminationReason::Ongoing || reason == TerminationReason::SafetyPlyCap) {
        return std::nullopt;
    }
    const TerminalResult result = state.terminalResult();
    if (!result.winner.has_value()) {
        return 0.0;
    }
    return *result.winner == state.currentPlayer() ? 1.0 : -1.0;
}

} // namespace az::games::chess
