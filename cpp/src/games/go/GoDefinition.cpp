#include "games/go/GoDefinition.hpp"

#include <stdexcept>

namespace az::games::go {

core::GameIdentity GoDefinition::identity() { return core::GameIdentity::Go; }

uint32 GoDefinition::gameSchemaVersion() { return 1; }

uint32 GoDefinition::replaySchemaVersion() { return 1; }

std::string_view GoDefinition::policySpaceIdentity() { return "go_points_row_major_with_pass_v1"; }

GoDefinition::State GoDefinition::createInitialState(const Rules &rules) { return State(rules); }

int32 GoDefinition::actionCount(const Rules &rules) { return State(rules).actionCount(); }

core::TensorSpecification GoDefinition::inputSpecification(const Rules &rules) {
    const State state(rules);
    const Encoding encoding = state.canonicalEncoding();
    return core::TensorSpecification{
        .dataType = core::TensorDataType::Int8,
        .layout = core::TensorLayout::ChannelsFirst,
        .dimensions = {encoding.planes, encoding.boardSize, encoding.boardSize},
    };
}

int32 GoDefinition::actionToPolicy(Action action, const Rules &rules) {
    const int32 count = actionCount(rules);
    if (action < 0 || action >= count) {
        throw std::invalid_argument("Go action is outside the policy space");
    }
    return action;
}

GoDefinition::Action GoDefinition::policyToAction(int32 policyIndex, const Rules &rules) {
    return actionToPolicy(policyIndex, rules);
}

std::span<const GoDefinition::Symmetry> GoDefinition::validSymmetries() { return SYMMETRIES; }

GoDefinition::Action GoDefinition::transformAction(Action action, const Rules &rules,
                                                   Symmetry symmetry) {
    if (action < 0 || action >= actionCount(rules)) {
        throw std::invalid_argument("Go action is outside the policy space");
    }
    return go::transformAction(action, rules.boardSize, symmetry);
}

GoDefinition::Encoding GoDefinition::transformEncoding(const Encoding &encoding,
                                                       Symmetry symmetry) {
    return go::transformEncoding(encoding, symmetry);
}

GoDefinition::ReplayPayload GoDefinition::replayPayload(const State &state) {
    return state.canonicalEncoding();
}

std::optional<double> GoDefinition::terminalValue(const State &state) {
    if (!state.isTerminal() || state.terminationReason() == TerminationReason::SafetyPlyCap) {
        return std::nullopt;
    }
    const TerminalResult result = state.terminalResult();
    if (!result.winner.has_value()) {
        return 0.0;
    }
    return result.winner == state.currentPlayer() ? 1.0 : -1.0;
}

} // namespace az::games::go
