#pragma once

#include "search/InferenceTypes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename Game> struct SearchInferenceResult {
    using Action = typename Game::Action;

    std::vector<std::pair<Action, float>> actions;
    WdlPrediction outcome;

    [[nodiscard]] float value() const noexcept { return outcome.expectedValue(); }
};

template <typename Game>
[[nodiscard]] SearchInferenceResult<Game>
processSearchInference(const float *policy, const float *outcome,
                       const typename Game::Position &position) {
    const float win = outcome[static_cast<std::size_t>(WdlIndex::Win)];
    const float draw = outcome[static_cast<std::size_t>(WdlIndex::Draw)];
    const float loss = outcome[static_cast<std::size_t>(WdlIndex::Loss)];
    if (!std::isfinite(win) || !std::isfinite(draw) || !std::isfinite(loss) || win < 0.0F ||
        draw < 0.0F || loss < 0.0F || std::abs(win + draw + loss - 1.0F) > 1e-2F) {
        throw std::runtime_error("Inference model WDL output must be three probabilities");
    }

    const std::vector<typename Game::Action> legalActions = Game::legalActions(position);
    std::vector<std::pair<typename Game::Action, float>> actions;
    actions.reserve(legalActions.size());
    float legalPolicySum = 0.0F;
    for (const typename Game::Action action : legalActions) {
        const int actionId = Game::actionId(action, position);
        if (actionId < 0 || actionId >= Game::inferenceDimensions().actions) {
            throw std::logic_error("Game contract produced an action outside its policy space");
        }
        const float probability = policy[actionId];
        if (!std::isfinite(probability) || probability < 0.0F) {
            throw std::runtime_error("Inference model policy must contain finite probabilities");
        }
        actions.emplace_back(action, probability);
        legalPolicySum += probability;
    }
    std::ranges::sort(actions, {}, [&position](const auto &actionProbability) {
        return Game::actionId(actionProbability.first, position);
    });
    if (!actions.empty() && legalPolicySum < 1e-5F) {
        const float uniformProbability = 1.0F / static_cast<float>(actions.size());
        for (auto &[action, probability] : actions) {
            static_cast<void>(action);
            probability = uniformProbability;
        }
    } else if (legalPolicySum >= 1e-5F) {
        for (auto &[action, probability] : actions) {
            static_cast<void>(action);
            probability /= legalPolicySum;
        }
        std::erase_if(actions, [](const auto &actionProbability) {
            return !(actionProbability.second > 0.0F);
        });
    }
    return {std::move(actions), {win, draw, loss}};
}
