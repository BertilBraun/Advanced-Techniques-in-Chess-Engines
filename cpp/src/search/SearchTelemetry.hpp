#pragma once

#include "common.hpp"
#include "search/SearchConfiguration.hpp"

#include <optional>
#include <vector>

namespace az::search {

enum class SearchStopReason : int8 { FullBudget = 0, TerminalRoot = 1, AdaptiveConfidence = 2 };

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
    double initialRootFpu;
};

struct SearchTraceSnapshot {
    int32 simulations;
    std::vector<double> rootPolicy;
    std::vector<int32> rootVisits;
    double rootValue;
};

template <typename Action> struct SearchResult {
    std::optional<Action> selectedAction;
    std::vector<double> rootPolicy;
    std::vector<int32> rootVisits;
    std::optional<double> rootValue;
    std::vector<RootChildStatistics<Action>> rootChildren;
    SearchTelemetry telemetry;
    std::vector<SearchTraceSnapshot> prefixTrace;
};

} // namespace az::search
