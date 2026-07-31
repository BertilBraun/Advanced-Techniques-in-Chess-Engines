#include "core/SessionConfiguration.hpp"
#include "games/GameDefinition.hpp"
#include "games/go/GoDefinition.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

using az::core::GameIdentity;
using az::core::InferenceArtifactMetadata;
using az::core::TensorDataType;
using az::core::TensorLayout;
using az::core::TensorSpecification;
using az::games::go::GoDefinition;
using az::games::go::GoRules;
using az::games::go::Symmetry;

static_assert(az::games::GameDefinition<GoDefinition>);

namespace {

GoRules rules(int32 boardSize) {
    return GoRules{
        .boardSize = boardSize,
        .komiHalfPoints = 15,
        .safetyPlyCap = boardSize * boardSize * 4,
        .historyLength = 4,
    };
}

InferenceArtifactMetadata artifact(const GoRules &goRules) {
    return InferenceArtifactMetadata{
        .game = GameIdentity::Go,
        .artifactSchemaVersion = 1,
        .gameSchemaVersion = GoDefinition::gameSchemaVersion(),
        .modelGeneration = 7,
        .artifactIdentity = "go-test-artifact",
        .modelFamily = "go-residual-test",
        .policySpaceIdentity = std::string(GoDefinition::policySpaceIdentity()),
        .contentChecksum = "test-checksum",
        .input = GoDefinition::inputSpecification(goRules),
        .policyOutput =
            TensorSpecification{
                .dataType = TensorDataType::Float32,
                .layout = TensorLayout::Flat,
                .dimensions = {GoDefinition::actionCount(goRules)},
            },
        .valueOutput =
            TensorSpecification{
                .dataType = TensorDataType::Float32,
                .layout = TensorLayout::Flat,
                .dimensions = {1},
            },
    };
}

template <typename Operation> void expectInvalidArgument(Operation operation) {
    bool threw = false;
    try {
        operation();
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
}

void testFixedShapeSessions() {
    for (const int32 boardSize : {3, 7, 9, 13}) {
        const GoRules goRules = rules(boardSize);
        const auto session =
            az::core::resolveSessionConfiguration<GoDefinition>(goRules, artifact(goRules));
        assert(session.metadata.game == GameIdentity::Go);
        assert(session.metadata.actionCount == boardSize * boardSize + 1);
        assert(session.metadata.canonicalInput.dataType == TensorDataType::Int8);
        assert(session.metadata.canonicalInput.layout == TensorLayout::ChannelsFirst);
        assert(session.metadata.canonicalInput.dimensions ==
               std::vector<int64>({9, boardSize, boardSize}));
        assert(session.rules == goRules);
    }
}

void testGameMappingsAndSymmetries() {
    const GoRules goRules = rules(7);
    const auto initialState = GoDefinition::createInitialState(goRules);
    assert(GoDefinition::actionToPolicy(initialState.passAction(), goRules) ==
           initialState.passAction());
    assert(GoDefinition::policyToAction(10, goRules) == 10);
    assert(GoDefinition::validSymmetries().size() == 8);
    assert(GoDefinition::transformAction(initialState.passAction(), goRules, Symmetry::Rotate90) ==
           initialState.passAction());
    assert(GoDefinition::replayPayload(initialState) == initialState.canonicalEncoding());
    assert(!GoDefinition::terminalValue(initialState).has_value());
}

void testArtifactMismatchFailsBeforeSessionCreation() {
    const GoRules goRules = rules(7);
    InferenceArtifactMetadata wrongGame = artifact(goRules);
    wrongGame.game = GameIdentity::Chess;
    expectInvalidArgument([&goRules, &wrongGame]() {
        static_cast<void>(az::core::resolveSessionConfiguration<GoDefinition>(goRules, wrongGame));
    });

    InferenceArtifactMetadata wrongPolicyShape = artifact(goRules);
    wrongPolicyShape.policyOutput.dimensions = {49};
    expectInvalidArgument([&goRules, &wrongPolicyShape]() {
        static_cast<void>(
            az::core::resolveSessionConfiguration<GoDefinition>(goRules, wrongPolicyShape));
    });
}

} // namespace

int main() {
    testFixedShapeSessions();
    testGameMappingsAndSymmetries();
    testArtifactMismatchFailsBeforeSessionCreation();
    return 0;
}
