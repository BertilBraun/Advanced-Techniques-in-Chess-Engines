#include "games/chess/ChessAnalysis.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_chess_analysis(py::module_ &module) {
    py::class_<ChessAnalysisCandidate>(module, "CandidateAnalysis")
        .def_readonly("move_uci", &ChessAnalysisCandidate::move_uci)
        .def_readonly("policy_prior", &ChessAnalysisCandidate::policy_prior)
        .def_readonly("visits", &ChessAnalysisCandidate::visits)
        .def_readonly("visit_share", &ChessAnalysisCandidate::visit_share)
        .def_readonly("mean_value", &ChessAnalysisCandidate::mean_value);

    py::class_<ChessAnalysisResult>(module, "AnalysisResult")
        .def_readonly("chosen_move_uci", &ChessAnalysisResult::chosen_move_uci)
        .def_readonly("value", &ChessAnalysisResult::value)
        .def_readonly("outcome", &ChessAnalysisResult::outcome)
        .def_readonly("candidates", &ChessAnalysisResult::candidates)
        .def_readonly("searches", &ChessAnalysisResult::searches)
        .def_readonly("maximum_depth", &ChessAnalysisResult::maximum_depth)
        .def_readonly("elapsed_milliseconds", &ChessAnalysisResult::elapsed_milliseconds)
        .def_readonly("principal_variation", &ChessAnalysisResult::principal_variation);

    py::class_<ChessAnalysis, std::shared_ptr<ChessAnalysis>>(module, "ChessAnalysis")
        .def(py::init<const InferenceConfiguration &, const AnalysisParameters &>(),
             py::arg("runtime_parameters"), py::arg("parameters"))
        .def(
            "new_session",
            [](const std::shared_ptr<ChessAnalysis> &analysis, const std::string &startingFen,
               const std::vector<std::string> &movesUci) {
                return std::make_shared<ChessAnalysisSession>(analysis, startingFen, movesUci);
            },
            py::arg("starting_fen"), py::arg("moves_uci"))
        .def("inference_statistics", &ChessAnalysis::inferenceStatistics);

    py::class_<ChessAnalysisSession, std::shared_ptr<ChessAnalysisSession>>(module,
                                                                            "ChessAnalysisSession")
        .def("apply_move", &ChessAnalysisSession::applyMove, py::arg("move_uci"))
        .def("analyze", &ChessAnalysisSession::analyze, py::arg("mode"),
             py::arg("time_limit_seconds") = std::nullopt, py::arg("search_limit") = std::nullopt,
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("fen", &ChessAnalysisSession::fen)
        .def_property_readonly("starting_fen", &ChessAnalysisSession::startingFen)
        .def_property_readonly("moves_uci", &ChessAnalysisSession::movesUci)
        .def_property_readonly("root_visits", &ChessAnalysisSession::rootVisits);
}
