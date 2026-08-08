#include "games/go/GoGame.hpp"
#include "search/Analysis.hpp"
#include "search/SelfPlayBindings.hpp"

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
    using Analysis = GameAnalysis<Contract>;

    auto bindings = bindSelfPlay<Contract>(
        module, {.root = rootName,
                 .search = searchName,
                 .request = requestName,
                 .result = resultName,
                 .batch = batchName});
    bindings.root
        .def_property_readonly(
            "children",
            [](const Root &root) {
                std::vector<std::pair<int, std::uint32_t>> children;
                for (const auto &edge : root.tree().root().children) {
                    children.emplace_back(
                        Contract::Encoding::actionId(edge.action, root.position()), edge.visits);
                }
                return children;
            });

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
