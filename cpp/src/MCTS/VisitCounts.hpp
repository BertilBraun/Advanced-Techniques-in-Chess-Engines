#pragma once

#include "common.hpp"

typedef std::array<float, ACTION_SIZE> ActionProbabilities;

struct VisitCounts {
    struct VisitCount {
        int move;
        int count;
    };

    std::vector<VisitCount> visits;

    ActionProbabilities actionProbabilities() const {
        ActionProbabilities actionProbabilities = {0};

        for (const auto &visit : visits) {
            actionProbabilities[visit.move] = visit.count;
        }

        float totalVisits = sum(actionProbabilities);

        for (const auto &visit : visits) {
            actionProbabilities[visit.move] /= totalVisits;
        }

        return actionProbabilities;
    }
};
