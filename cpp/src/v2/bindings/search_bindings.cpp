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
    explicit PythonGoEvaluator(py::function callback) : _callback(std::move(callback)) {}

    [[nodiscard]] InferenceResult evaluate(const InferenceRequest<GoEncoding> &request) const {
        py::gil_scoped_acquire acquire;
        return _callback(request).cast<InferenceResult>();
    }

private:
    py::function _callback;
};

[[nodiscard]] std::optional<double> goTerminalValue(const GoState &state) {
    const auto result = state.terminalResult();
    if (result.reason == TerminationReason::SafetyPlyCap) {
        return std::nullopt;
    }
    if (!result.winner.has_value()) {
        return 0.0;
    }
    return result.winner.value() == state.currentPlayer() ? 1.0 : -1.0;
}

} // namespace

void bindSearch(pybind11::module_ &module) {
    module.attr("MAXIMUM_SIMULATION_COUNT") = search::MAXIMUM_SIMULATION_COUNT;

    py::enum_<SearchBudgetClass>(module, "SearchBudgetClass")
        .value("FIXED", SearchBudgetClass::Fixed);
    py::enum_<SearchStopReason>(module, "SearchStopReason")
        .value("FULL_BUDGET", SearchStopReason::FullBudget)
        .value("TERMINAL_ROOT", SearchStopReason::TerminalRoot);

    py::class_<InferenceRequest<GoEncoding>>(module, "GoInferenceRequest")
        .def_readonly("request_id", &InferenceRequest<GoEncoding>::requestId)
        .def_readonly("encoding", &InferenceRequest<GoEncoding>::encoding)
        .def_readonly("action_count", &InferenceRequest<GoEncoding>::actionCount);
    py::class_<InferenceResult>(module, "InferenceResult")
        .def(py::init<uint64, std::vector<double>, double>(), py::arg("request_id"),
             py::arg("policy"), py::arg("value"))
        .def_readonly("request_id", &InferenceResult::requestId)
        .def_readonly("policy", &InferenceResult::policy)
        .def_readonly("value", &InferenceResult::value);

    py::class_<RootNoiseConfiguration>(module, "RootNoiseConfiguration")
        .def(py::init<bool, double, double>(), py::arg("enabled"), py::arg("alpha"),
             py::arg("fraction"))
        .def_readonly("enabled", &RootNoiseConfiguration::enabled)
        .def_readonly("alpha", &RootNoiseConfiguration::alpha)
        .def_readonly("fraction", &RootNoiseConfiguration::fraction);
    py::class_<FixedPuctConfiguration>(module, "FixedPuctConfiguration")
        .def(py::init<int32, double, double, double, double, uint64, uint64, RootNoiseConfiguration,
                      bool>(),
             py::arg("simulation_cap"), py::arg("exploration_constant"), py::arg("backup_discount"),
             py::arg("no_visited_child_value"), py::arg("action_temperature"),
             py::arg("root_noise_seed"), py::arg("action_sampling_seed"), py::arg("root_noise"),
             py::arg("tree_reuse"))
        .def_readonly("simulation_cap", &FixedPuctConfiguration::simulationCap)
        .def_readonly("exploration_constant", &FixedPuctConfiguration::explorationConstant)
        .def_readonly("backup_discount", &FixedPuctConfiguration::backupDiscount)
        .def_readonly("no_visited_child_value", &FixedPuctConfiguration::noVisitedChildValue)
        .def_readonly("action_temperature", &FixedPuctConfiguration::actionTemperature)
        .def_readonly("root_noise_seed", &FixedPuctConfiguration::rootNoiseSeed)
        .def_readonly("action_sampling_seed", &FixedPuctConfiguration::actionSamplingSeed)
        .def_readonly("root_noise", &FixedPuctConfiguration::rootNoise)
        .def_readonly("tree_reuse", &FixedPuctConfiguration::treeReuse);
    py::class_<RootChildStatistics<int32>>(module, "RootChildStatistics")
        .def_readonly("action", &RootChildStatistics<int32>::action)
        .def_readonly("prior", &RootChildStatistics<int32>::prior)
        .def_readonly("visits", &RootChildStatistics<int32>::visits)
        .def_readonly("action_value", &RootChildStatistics<int32>::actionValue);
    py::class_<SearchTelemetry>(module, "SearchTelemetry")
        .def_readonly("configured_cap", &SearchTelemetry::configuredCap)
        .def_readonly("actual_simulations", &SearchTelemetry::actualSimulations)
        .def_readonly("budget_class", &SearchTelemetry::budgetClass)
        .def_readonly("stop_reason", &SearchTelemetry::stopReason)
        .def_readonly("policy_target_eligible", &SearchTelemetry::policyTargetEligible)
        .def_readonly("policy_target_weight", &SearchTelemetry::policyTargetWeight)
        .def_readonly("root_visit_count", &SearchTelemetry::rootVisitCount)
        .def_readonly("root_inference_requests", &SearchTelemetry::rootInferenceRequests)
        .def_readonly("leaf_inference_requests", &SearchTelemetry::leafInferenceRequests)
        .def_readonly("total_inference_requests", &SearchTelemetry::totalInferenceRequests)
        .def_readonly("root_entropy", &SearchTelemetry::rootEntropy)
        .def_readonly("top_two_visit_margin", &SearchTelemetry::topTwoVisitMargin);
    py::class_<SearchResult<int32>>(module, "SearchResult")
        .def_readonly("selected_action", &SearchResult<int32>::selectedAction)
        .def_readonly("root_policy", &SearchResult<int32>::rootPolicy)
        .def_readonly("root_visits", &SearchResult<int32>::rootVisits)
        .def_readonly("root_value", &SearchResult<int32>::rootValue)
        .def_readonly("root_children", &SearchResult<int32>::rootChildren)
        .def_readonly("telemetry", &SearchResult<int32>::telemetry);

    module.def(
        "search_go_fixed",
        [](const GoState &state, py::function evaluator,
           const FixedPuctConfiguration &configuration) {
            PythonGoEvaluator nativeEvaluator(std::move(evaluator));
            py::gil_scoped_release release;
            return search::FixedPuctSearch<GoState>::run(state, nativeEvaluator, goTerminalValue,
                                                         configuration);
        },
        py::arg("state"), py::arg("evaluator"), py::arg("configuration"));
}

} // namespace az::v2::bindings
