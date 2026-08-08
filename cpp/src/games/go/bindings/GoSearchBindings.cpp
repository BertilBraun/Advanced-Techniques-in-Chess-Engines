#include "games/go/GoGame.hpp"
#include "search/Analysis.hpp"
#include "search/SelfPlay.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {
template <std::size_t BoardSize>
void bindGoSearch(py::module_ &module, const char *rootName, const char *searchName,
                  const char *requestName, const char *resultName, const char *batchName,
                  const char *analysisName) {
    using Contract = GoGame<BoardSize, 8>;
    using Position = typename Contract::State;
    using Root = GameSearchRoot<Contract>;
    using Search = GameSelfPlaySearch<Contract>;
    using Request = SelfPlaySearchRequest<Contract>;
    using Result = SelfPlaySearchResult<Contract>;
    using Batch = SelfPlaySearchBatch<Contract>;
    using Analysis = GameAnalysis<Contract>;
    constexpr InferenceDimensions dimensions = Contract::Encoding::inferenceDimensions();

    py::class_<Root>(module, rootName)
        .def_property_readonly("position", &Root::position)
        .def_property_readonly("is_terminal", &Root::isTerminal)
        .def_property_readonly("visits", &Root::visits)
        .def_property_readonly("live_nodes", &Root::liveNodeCount)
        .def_property_readonly(
            "children",
            [](const Root &root) {
                std::vector<std::pair<int, std::uint32_t>> children;
                for (const auto &edge : root.tree().root().children) {
                    children.emplace_back(
                        Contract::Encoding::actionId(edge.action, root.position()), edge.visits);
                }
                return children;
            })
        .def("play", &Root::play, py::arg("action_id"))
        .def("reset", &Root::reset)
        .def(
            "discount",
            [](Root &root, const float retainedFraction) {
                root.tree().discount(retainedFraction);
            },
            py::arg("retained_fraction"));

    py::class_<Request>(module, requestName)
        .def(py::init<Root, bool>(), py::arg("root"), py::arg("full_search"))
        .def_readonly("root", &Request::root)
        .def_readonly("full_search", &Request::full_search);
    py::class_<Result>(module, resultName)
        .def_readonly("root_value", &Result::root_value)
        .def_readonly("visits", &Result::visits)
        .def_readonly("root", &Result::root);
    py::class_<Batch>(module, batchName)
        .def_readonly("results", &Batch::results)
        .def_readonly("statistics", &Batch::statistics)
        .def_readonly("simulations_completed", &Batch::simulations_completed);

    py::class_<Search>(module, searchName)
        .def(py::init<const InferenceConfiguration &, const SelfPlaySearchParameters &,
                      BatchedInferenceParameters, std::uint64_t>(),
             py::arg("runtime_parameters"), py::arg("search_parameters"),
             py::arg("inference_parameters"), py::arg("initial_model_generation") = 0)
        .def_static("inference_dimensions", []() { return dimensions; })
        .def("request", [](const Search &, Root root, const bool fullSearch) {
            return Request(std::move(root), fullSearch);
        }, py::arg("root"), py::arg("full_search"))
        .def("new_root", [](Search &search, const Position &position) {
            return search.newRoot(position);
        }, py::arg("position"))
        .def("search", &Search::search, py::arg("requests"), py::arg("collect_statistics") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("refresh_model", &Search::refreshModel, py::arg("model_generation"),
             py::arg("model_path"), py::call_guard<py::gil_scoped_release>())
        .def("update_search_schedule", &Search::updateSearchSchedule,
             py::arg("search_parameters"))
        .def_property_readonly("model_generation", &Search::modelGeneration)
        .def("inference_statistics", &Search::inferenceStatistics);

    py::class_<Analysis, std::shared_ptr<Analysis>>(module, analysisName)
        .def(py::init<const InferenceConfiguration &, const AnalysisParameters &>(),
             py::arg("runtime_parameters"), py::arg("parameters"))
        .def(
            "new_root",
            [](Analysis &analysis, const GoRules rules) {
                return analysis.newRoot(Position(rules));
            },
            py::arg("rules"))
        .def("analyze_policy", &Analysis::analyzePolicy, py::arg("root"),
             py::call_guard<py::gil_scoped_release>())
        .def("analyze_counted", &Analysis::analyzeCounted, py::arg("root"), py::arg("searches"),
             py::call_guard<py::gil_scoped_release>())
        .def(
            "analyze_timed",
            [](Analysis &analysis, Root &root, const int seconds) {
                return analysis.analyzeTimed(root, std::chrono::seconds(seconds));
            },
            py::arg("root"), py::arg("seconds"), py::call_guard<py::gil_scoped_release>())
        .def("inference_statistics", &Analysis::inferenceStatistics);
}
} // namespace

void bind_go_search(py::module_ &module) {
    bindGoSearch<7>(module, "GoSearchRoot7", "GoSelfPlaySearch7", "GoSelfPlaySearchRequest7",
                    "GoSelfPlaySearchResult7", "GoSelfPlaySearchBatch7", "GoAnalysis7");
    bindGoSearch<9>(module, "GoSearchRoot9", "GoSelfPlaySearch9", "GoSelfPlaySearchRequest9",
                    "GoSelfPlaySearchResult9", "GoSelfPlaySearchBatch9", "GoAnalysis9");
}
