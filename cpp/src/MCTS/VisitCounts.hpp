#pragma once

#include "common.hpp"

typedef std::pair<int, int> VisitCount;
typedef std::vector<VisitCount> VisitCounts;

typedef std::array<float, ACTION_SIZE> ActionProbabilities;

template <typename CountType>
ActionProbabilities actionProbabilities(const std::vector<std::pair<int, CountType>> &visitCounts) {
    // Ensure CountType is arithmetic (e.g. int, float, etc.)
    static_assert(std::is_arithmetic<CountType>::value, "CountType must be arithmetic.");

    ActionProbabilities probabilities = {0.0f};
    float totalVisits = 0.0f;

    for (const auto &[move, count] : visitCounts) {
        totalVisits += static_cast<float>(count);
    }

    assert(totalVisits > 0.0f);

    for (const auto &[move, count] : visitCounts) {
        probabilities[move] = static_cast<float>(count) / totalVisits;
    }

    return probabilities;
}