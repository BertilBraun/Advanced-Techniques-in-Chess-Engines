#pragma once

#include "common.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace az::v2::search {

enum class SearchBudgetClass : int8 { Fixed = 0 };
enum class SearchStopReason : int8 { FullBudget = 0, TerminalRoot = 1 };

template <typename Action> struct RootChildStatistics {
    Action action;
    double prior;
    int32 visits;
    double actionValue;
};

struct SearchTelemetry {
    int32 configuredCap;
    int32 actualSimulations;
    SearchBudgetClass budgetClass;
    SearchStopReason stopReason;
    bool policyTargetEligible;
    double policyTargetWeight;
    int32 rootVisitCount;
    int64 rootInferenceRequests;
    int64 leafInferenceRequests;
    int64 totalInferenceRequests;
    double rootEntropy;
    double topTwoVisitMargin;
};

template <typename Action> struct SearchResult {
    std::optional<Action> selectedAction;
    std::vector<double> rootPolicy;
    std::vector<int32> rootVisits;
    std::optional<double> rootValue;
    std::vector<RootChildStatistics<Action>> rootChildren;
    SearchTelemetry telemetry;
};

} // namespace az::v2::search
