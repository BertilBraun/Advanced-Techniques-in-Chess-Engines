#pragma once

#include <cmath>
#include <stdexcept>

enum class FirstPlayUrgencyKind { Zero, ParentValue, ReducedParentValue };

struct FirstPlayUrgencyParameters {
    FirstPlayUrgencyKind kind;
    float reduction;

    FirstPlayUrgencyParameters(const FirstPlayUrgencyKind urgencyKind,
                               const float urgencyReduction = 0.0F)
        : kind(urgencyKind), reduction(urgencyReduction) {
        if (!std::isfinite(reduction) || reduction < 0.0F) {
            throw std::invalid_argument("FPU reduction must be finite and nonnegative");
        }
        if (kind != FirstPlayUrgencyKind::ReducedParentValue && reduction != 0.0F) {
            throw std::invalid_argument("Only reduced-parent FPU accepts a reduction");
        }
        if (kind == FirstPlayUrgencyKind::ReducedParentValue && reduction == 0.0F) {
            throw std::invalid_argument("Reduced-parent FPU requires a positive reduction");
        }
    }

    [[nodiscard]] float unvisitedParentValue(const float parentMean) const {
        switch (kind) {
        case FirstPlayUrgencyKind::Zero:
            return 0.0F;
        case FirstPlayUrgencyKind::ParentValue:
            return parentMean;
        case FirstPlayUrgencyKind::ReducedParentValue:
            return parentMean - reduction;
        }
        throw std::logic_error("Unknown FPU kind");
    }
};

struct TreeSearchParameters {
    float exploration_constant;
    FirstPlayUrgencyParameters first_play_urgency;
    float forced_playout_coefficient;
    float value_discount_per_ply;

    TreeSearchParameters(const float explorationConstant,
                         FirstPlayUrgencyParameters firstPlayUrgency,
                         const float forcedPlayoutCoefficient, const float valueDiscountPerPly)
        : exploration_constant(explorationConstant), first_play_urgency(firstPlayUrgency),
          forced_playout_coefficient(forcedPlayoutCoefficient),
          value_discount_per_ply(valueDiscountPerPly) {
        if (!std::isfinite(exploration_constant) || exploration_constant <= 0.0F) {
            throw std::invalid_argument("Exploration constant must be finite and positive");
        }
        if (!std::isfinite(forced_playout_coefficient) || forced_playout_coefficient < 0.0F) {
            throw std::invalid_argument(
                "Forced-playout coefficient must be finite and nonnegative");
        }
        if (!std::isfinite(value_discount_per_ply) || value_discount_per_ply <= 0.0F ||
            value_discount_per_ply > 1.0F) {
            throw std::invalid_argument("Value discount per ply must be finite and in (0, 1]");
        }
    }
};
