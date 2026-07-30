#pragma once

#include "common.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace az::v2::search {

inline constexpr int32 MAXIMUM_SIMULATION_COUNT = std::numeric_limits<int32>::max();

struct RootNoiseConfiguration {
    bool enabled;
    double alpha;
    double fraction;
};

struct FixedPuctConfiguration {
    int32 simulationCap;
    double explorationConstant;
    double backupDiscount;
    double noVisitedChildValue;
    double actionTemperature;
    uint64 rootNoiseSeed;
    uint64 actionSamplingSeed;
    RootNoiseConfiguration rootNoise;
    bool treeReuse;

    void validate() const {
        if (simulationCap <= 0) {
            throw std::invalid_argument("simulationCap must be positive");
        }
        if (!std::isfinite(explorationConstant) || !(explorationConstant > 0.0)) {
            throw std::invalid_argument("explorationConstant must be finite and positive");
        }
        if (!std::isfinite(backupDiscount) || !(backupDiscount > 0.0 && backupDiscount <= 1.0)) {
            throw std::invalid_argument("backupDiscount must be finite and in (0, 1]");
        }
        if (!std::isfinite(noVisitedChildValue) ||
            !(noVisitedChildValue >= -1.0 && noVisitedChildValue <= 1.0)) {
            throw std::invalid_argument("noVisitedChildValue must be finite and in [-1, 1]");
        }
        if (!std::isfinite(actionTemperature) || !(actionTemperature >= 0.0)) {
            throw std::invalid_argument("actionTemperature must be finite and non-negative");
        }
        if (!std::isfinite(rootNoise.alpha) || !(rootNoise.alpha > 0.0) ||
            !std::isfinite(rootNoise.fraction) ||
            !(rootNoise.fraction >= 0.0 && rootNoise.fraction <= 1.0)) {
            throw std::invalid_argument(
                "root noise requires finite positive alpha and finite fraction in [0, 1]");
        }
        if (treeReuse) {
            throw std::invalid_argument("tree reuse is not implemented for fixed PUCT");
        }
    }
};

} // namespace az::v2::search
