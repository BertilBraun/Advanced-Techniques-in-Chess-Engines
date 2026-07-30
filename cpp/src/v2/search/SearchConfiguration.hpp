#pragma once

#include "common.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace az::v2::search {

inline constexpr int32 MAXIMUM_SIMULATION_COUNT = std::numeric_limits<int32>::max();

enum class SearchBudgetClass : int8 {
    Fixed = 0,
    ProgressiveStage = 1,
    MixedFast = 2,
    MixedFull = 3
};

struct RootNoiseConfiguration {
    bool enabled;
    double alpha;
    double fraction;
};

enum class FpuPolicy : int8 { ParentValue = 0, ReducedParentValue = 1, VisitedChildMean = 2 };

struct AdaptiveStoppingConfiguration {
    bool enabled = false;
    int32 minimumSimulations = 1;
    int32 checkIntervalSimulations = 1;
    double requiredTopVisitFraction = 1.0;
    double requiredTopTwoMargin = 1.0;
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
    FpuPolicy fpuPolicy = FpuPolicy::VisitedChildMean;
    double fpuReduction = 0.0;
    AdaptiveStoppingConfiguration adaptiveStopping{};
    SearchBudgetClass budgetClass = SearchBudgetClass::Fixed;
    double policyTargetWeight = 1.0;

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
        if (!std::isfinite(fpuReduction) || fpuReduction < 0.0) {
            throw std::invalid_argument("FPU reduction must be finite and non-negative");
        }
        if (!std::isfinite(policyTargetWeight) || policyTargetWeight < 0.0) {
            throw std::invalid_argument("policy target weight must be finite and non-negative");
        }
        if (adaptiveStopping.enabled) {
            if (adaptiveStopping.minimumSimulations <= 0 ||
                adaptiveStopping.minimumSimulations >= simulationCap) {
                throw std::invalid_argument(
                    "adaptive minimum simulations must be positive and below the cap");
            }
            if (adaptiveStopping.checkIntervalSimulations <= 0) {
                throw std::invalid_argument("adaptive check interval must be positive");
            }
            if (adaptiveStopping.checkIntervalSimulations > simulationCap) {
                throw std::invalid_argument("adaptive check interval cannot exceed the cap");
            }
            if (!std::isfinite(adaptiveStopping.requiredTopVisitFraction) ||
                adaptiveStopping.requiredTopVisitFraction <= 0.5 ||
                adaptiveStopping.requiredTopVisitFraction > 1.0) {
                throw std::invalid_argument(
                    "adaptive top visit fraction must be finite and in (0.5, 1]");
            }
            if (!std::isfinite(adaptiveStopping.requiredTopTwoMargin) ||
                adaptiveStopping.requiredTopTwoMargin < 0.0 ||
                adaptiveStopping.requiredTopTwoMargin > 1.0) {
                throw std::invalid_argument("adaptive top-two margin must be finite and in [0, 1]");
            }
        }
    }
};

} // namespace az::v2::search
