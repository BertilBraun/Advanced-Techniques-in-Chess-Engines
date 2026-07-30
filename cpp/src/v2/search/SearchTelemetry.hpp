#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace az::v2::search {

enum class SearchBudgetClass : std::int8_t { Fixed = 0 };
enum class SearchStopReason : std::int8_t { FullBudget = 0, TerminalRoot = 1 };

template <typename Action> struct RootChildStatistics {
    Action action;
    double prior;
    std::int32_t visits;
    double action_value;
};

struct SearchTelemetry {
    std::int32_t configured_cap;
    std::int32_t actual_simulations;
    SearchBudgetClass budget_class;
    SearchStopReason stop_reason;
    bool policy_target_eligible;
    double policy_target_weight;
    std::int32_t root_visit_count;
    std::int64_t root_inference_requests;
    std::int64_t leaf_inference_requests;
    std::int64_t total_inference_requests;
    double root_entropy;
    double top_two_visit_margin;
};

template <typename Action> struct SearchResult {
    std::optional<Action> selected_action;
    std::vector<double> root_policy;
    std::vector<std::int32_t> root_visits;
    std::optional<double> root_value;
    std::vector<RootChildStatistics<Action>> root_children;
    SearchTelemetry telemetry;
};

} // namespace az::v2::search
