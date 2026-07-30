#include "bindings/search_bindings.hpp"

#include "games/go/GoEncoding.hpp"
#include "games/go/GoState.hpp"
#include "inference/InferenceTypes.hpp"
#include "search/FixedPuct.hpp"
#include "search/SearchConfiguration.hpp"
#include "search/SearchTelemetry.hpp"

#include <cstdint>
#include <optional>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <utility>

namespace az::v2::bindings {
namespace {

namespace py = pybind11;
using games::go::GoEncoding;
using games::go::GoState;
using games::go::TerminationReason;
using inference::InferenceRequest;
using inference::InferenceResult;
using search::FixedPuctConfiguration;
using search::RootChildStatistics;
using search::RootNoiseConfiguration;
using search::SearchBudgetClass;
using search::SearchResult;
using search::SearchStopReason;
using search::SearchTelemetry;

class PythonGoEvaluator {
public:
    explicit PythonGoEvaluator(py::function callback) : callback_(std::move(callback)) {}

    [[nodiscard]] InferenceResult evaluate(const InferenceRequest<GoEncoding> &request) const {
        py::gil_scoped_acquire acquire;
        return callback_(request).cast<InferenceResult>();
    }

private:
    py::function callback_;
};

[[nodiscard]] std::optional<double> go_terminal_value(const GoState &state) {
    const auto result = state.terminal_result();
    if (result.reason == TerminationReason::SafetyPlyCap) {
        return std::nullopt;
    }
    if (!result.winner.has_value()) {
        return 0.0;
    }
    return result.winner.value() == state.current_player() ? 1.0 : -1.0;
}

} // namespace

void bind_search(pybind11::module_ &module) {
    module.attr("MAXIMUM_SIMULATION_COUNT") = search::maximum_simulation_count;

    py::enum_<SearchBudgetClass>(module, "SearchBudgetClass")
        .value("FIXED", SearchBudgetClass::Fixed);
    py::enum_<SearchStopReason>(module, "SearchStopReason")
        .value("FULL_BUDGET", SearchStopReason::FullBudget)
        .value("TERMINAL_ROOT", SearchStopReason::TerminalRoot);

    py::class_<InferenceRequest<GoEncoding>>(module, "GoInferenceRequest")
        .def_readonly("request_id", &InferenceRequest<GoEncoding>::request_id)
        .def_readonly("encoding", &InferenceRequest<GoEncoding>::encoding)
        .def_readonly("action_count", &InferenceRequest<GoEncoding>::action_count);
    py::class_<InferenceResult>(module, "InferenceResult")
        .def(py::init<std::uint64_t, std::vector<double>, double>(), py::arg("request_id"),
             py::arg("policy"), py::arg("value"))
        .def_readonly("request_id", &InferenceResult::request_id)
        .def_readonly("policy", &InferenceResult::policy)
        .def_readonly("value", &InferenceResult::value);

    py::class_<RootNoiseConfiguration>(module, "RootNoiseConfiguration")
        .def(py::init<bool, double, double>(), py::arg("enabled"), py::arg("alpha"),
             py::arg("fraction"))
        .def_readonly("enabled", &RootNoiseConfiguration::enabled)
        .def_readonly("alpha", &RootNoiseConfiguration::alpha)
        .def_readonly("fraction", &RootNoiseConfiguration::fraction);
    py::class_<FixedPuctConfiguration>(module, "FixedPuctConfiguration")
        .def(py::init<std::int32_t, double, double, double, double, std::uint64_t,
                      RootNoiseConfiguration, bool>(),
             py::arg("simulation_cap"), py::arg("exploration_constant"), py::arg("backup_discount"),
             py::arg("no_visited_child_value"), py::arg("action_temperature"), py::arg("seed"),
             py::arg("root_noise"), py::arg("tree_reuse"))
        .def_readonly("simulation_cap", &FixedPuctConfiguration::simulation_cap)
        .def_readonly("exploration_constant", &FixedPuctConfiguration::exploration_constant)
        .def_readonly("backup_discount", &FixedPuctConfiguration::backup_discount)
        .def_readonly("no_visited_child_value", &FixedPuctConfiguration::no_visited_child_value)
        .def_readonly("action_temperature", &FixedPuctConfiguration::action_temperature)
        .def_readonly("seed", &FixedPuctConfiguration::seed)
        .def_readonly("root_noise", &FixedPuctConfiguration::root_noise)
        .def_readonly("tree_reuse", &FixedPuctConfiguration::tree_reuse);
    py::class_<RootChildStatistics<std::int32_t>>(module, "RootChildStatistics")
        .def_readonly("action", &RootChildStatistics<std::int32_t>::action)
        .def_readonly("prior", &RootChildStatistics<std::int32_t>::prior)
        .def_readonly("visits", &RootChildStatistics<std::int32_t>::visits)
        .def_readonly("action_value", &RootChildStatistics<std::int32_t>::action_value);
    py::class_<SearchTelemetry>(module, "SearchTelemetry")
        .def_readonly("configured_cap", &SearchTelemetry::configured_cap)
        .def_readonly("actual_simulations", &SearchTelemetry::actual_simulations)
        .def_readonly("budget_class", &SearchTelemetry::budget_class)
        .def_readonly("stop_reason", &SearchTelemetry::stop_reason)
        .def_readonly("policy_target_eligible", &SearchTelemetry::policy_target_eligible)
        .def_readonly("policy_target_weight", &SearchTelemetry::policy_target_weight)
        .def_readonly("root_visit_count", &SearchTelemetry::root_visit_count)
        .def_readonly("root_inference_requests", &SearchTelemetry::root_inference_requests)
        .def_readonly("leaf_inference_requests", &SearchTelemetry::leaf_inference_requests)
        .def_readonly("total_inference_requests", &SearchTelemetry::total_inference_requests)
        .def_readonly("root_entropy", &SearchTelemetry::root_entropy)
        .def_readonly("top_two_visit_margin", &SearchTelemetry::top_two_visit_margin);
    py::class_<SearchResult<std::int32_t>>(module, "SearchResult")
        .def_readonly("selected_action", &SearchResult<std::int32_t>::selected_action)
        .def_readonly("root_policy", &SearchResult<std::int32_t>::root_policy)
        .def_readonly("root_visits", &SearchResult<std::int32_t>::root_visits)
        .def_readonly("root_value", &SearchResult<std::int32_t>::root_value)
        .def_readonly("root_children", &SearchResult<std::int32_t>::root_children)
        .def_readonly("telemetry", &SearchResult<std::int32_t>::telemetry);

    module.def(
        "search_go_fixed",
        [](const GoState &state, py::function evaluator,
           const FixedPuctConfiguration &configuration) {
            PythonGoEvaluator native_evaluator(std::move(evaluator));
            py::gil_scoped_release release;
            return search::FixedPuctSearch<GoState>::run(state, native_evaluator, go_terminal_value,
                                                         configuration);
        },
        py::arg("state"), py::arg("evaluator"), py::arg("configuration"));
}

} // namespace az::v2::bindings
