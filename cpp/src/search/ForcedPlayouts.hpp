#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Keeps root exploration separate from the policy target produced by that exploration.

struct RootChildSearchStatistics {
    float prior;
    std::uint32_t visits;
    float child_mean_value;
};

[[nodiscard]] inline float forcedRootVisitThreshold(const float prior,
                                                    const std::uint32_t rootVisits,
                                                    const float coefficient) {
    return std::sqrt(coefficient * prior * static_cast<float>(rootVisits));
}

[[nodiscard]] inline bool requiresForcedRootVisit(const RootChildSearchStatistics child,
                                                  const std::uint32_t rootVisits,
                                                  const float coefficient) {
    return coefficient > 0.0F && child.prior > 0.0F &&
           static_cast<float>(child.visits) <
               forcedRootVisitThreshold(child.prior, rootVisits, coefficient);
}

[[nodiscard]] inline float rootPuctScore(const RootChildSearchStatistics child,
                                         const std::uint32_t rootVisits,
                                         const float explorationConstant) {
    const float parentScale = std::sqrt(static_cast<float>(std::max(1U, rootVisits)));
    return -child.child_mean_value + explorationConstant * child.prior * parentScale /
                                         (1.0F + static_cast<float>(child.visits));
}

[[nodiscard]] inline std::vector<std::uint32_t>
prunedRootPolicyVisits(const std::span<const RootChildSearchStatistics> children,
                       const std::uint32_t rootVisits, const float explorationConstant,
                       const bool enabled) {
    std::vector<std::uint32_t> result;
    result.reserve(children.size());
    for (const RootChildSearchStatistics child : children) {
        result.push_back(child.visits);
    }
    if (!enabled || children.empty()) {
        return result;
    }

    std::size_t bestIndex = 0;
    for (std::size_t index = 1; index < children.size(); ++index) {
        if (children[index].visits > children[bestIndex].visits) {
            bestIndex = index;
        }
    }
    const float bestScore = rootPuctScore(children[bestIndex], rootVisits, explorationConstant);
    const float parentScale = std::sqrt(static_cast<float>(std::max(1U, rootVisits)));
    for (std::size_t index = 0; index < children.size(); ++index) {
        if (index == bestIndex || children[index].visits == 0) {
            continue;
        }
        const RootChildSearchStatistics child = children[index];
        const float denominator = bestScore + child.child_mean_value;
        if (denominator <= 0.0F) {
            continue;
        }
        const float supportedVisits =
            explorationConstant * child.prior * parentScale / denominator - 1.0F;
        if (!std::isfinite(supportedVisits) ||
            supportedVisits >= static_cast<float>(child.visits)) {
            continue;
        }
        result[index] = static_cast<std::uint32_t>(std::ceil(std::max(0.0F, supportedVisits)));
    }
    return result;
}
