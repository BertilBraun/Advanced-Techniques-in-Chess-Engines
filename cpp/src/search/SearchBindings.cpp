#include "search/SearchBindings.hpp"

#include "search/Analysis.hpp"
#include "search/InferenceConfiguration.hpp"
#include "search/SearchTypes.hpp"
#include "search/SelfPlay.hpp"
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_search(py::module_ &module) {
    py::enum_<AnalysisMode>(module, "AnalysisMode")
        .value("POLICY", AnalysisMode::Policy)
        .value("MCTS", AnalysisMode::Mcts);

    py::enum_<InferenceDevice>(module, "InferenceDevice")
        .value("AUTO", InferenceDevice::Auto)
        .value("CPU", InferenceDevice::Cpu)
        .value("CUDA", InferenceDevice::Cuda);

    py::enum_<FirstPlayUrgencyKind>(module, "FirstPlayUrgencyKind")
        .value("ZERO", FirstPlayUrgencyKind::Zero)
        .value("PARENT_VALUE", FirstPlayUrgencyKind::ParentValue)
        .value("REDUCED_PARENT_VALUE", FirstPlayUrgencyKind::ReducedParentValue);

    py::class_<FirstPlayUrgencyParameters>(module, "FirstPlayUrgencyParameters")
        .def(py::init<FirstPlayUrgencyKind, float>(), py::arg("kind"), py::arg("reduction") = 0.0F)
        .def_readonly("kind", &FirstPlayUrgencyParameters::kind)
        .def_readonly("reduction", &FirstPlayUrgencyParameters::reduction);
    py::class_<TreeSearchParameters>(module, "TreeSearchParameters")
        .def(py::init<float, FirstPlayUrgencyParameters, float>(), py::arg("exploration_constant"),
             py::arg("first_play_urgency"), py::arg("forced_playout_coefficient"))
        .def_readonly("exploration_constant", &TreeSearchParameters::exploration_constant)
        .def_readonly("first_play_urgency", &TreeSearchParameters::first_play_urgency)
        .def_readonly("forced_playout_coefficient",
                      &TreeSearchParameters::forced_playout_coefficient);

    py::class_<InferenceConfiguration>(module, "InferenceConfiguration")
        .def(py::init<int, std::string, InferenceDevice>(), py::arg("device_id"),
             py::arg("model_path"), py::arg("device") = InferenceDevice::Auto)
        .def_readwrite("device_id", &InferenceConfiguration::device_id)
        .def_readwrite("model_path", &InferenceConfiguration::model_path)
        .def_readwrite("device", &InferenceConfiguration::device);

    py::class_<InferenceDimensions>(module, "InferenceDimensions")
        .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>(),
             py::arg("channels"), py::arg("rows"), py::arg("columns"), py::arg("actions"),
             py::arg("outcomes"))
        .def_readonly("channels", &InferenceDimensions::channels)
        .def_readonly("rows", &InferenceDimensions::rows)
        .def_readonly("columns", &InferenceDimensions::columns)
        .def_readonly("actions", &InferenceDimensions::actions)
        .def_readonly("outcomes", &InferenceDimensions::outcomes)
        .def(py::self == py::self);

    py::class_<InferenceStatistics>(module, "InferenceStatistics")
        .def(py::init<>())
        .def_readonly("evaluations", &InferenceStatistics::evaluations)
        .def_readonly("modelInferenceCalls", &InferenceStatistics::modelInferenceCalls)
        .def_readonly("modelInferencePositions", &InferenceStatistics::modelInferencePositions)
        .def_readonly("modelBatchSizeHistogram", &InferenceStatistics::modelBatchSizeHistogram)
        .def_readonly("averageNumberOfPositionsInInferenceCall",
                      &InferenceStatistics::averageNumberOfPositionsInInferenceCall)
        .def_readonly("treeSelectionNanoseconds", &InferenceStatistics::treeSelectionNanoseconds)
        .def_readonly("boardEncodingNanoseconds", &InferenceStatistics::boardEncodingNanoseconds)
        .def_readonly("resultProcessingNanoseconds",
                      &InferenceStatistics::resultProcessingNanoseconds)
        .def_readonly("treeBackupNanoseconds", &InferenceStatistics::treeBackupNanoseconds)
        .def_readonly("treeOwnerWaitNanoseconds", &InferenceStatistics::treeOwnerWaitNanoseconds)
        .def_readonly("inferenceNanoseconds", &InferenceStatistics::inferenceNanoseconds)
        .def_readonly("workerUtilization", &InferenceStatistics::workerUtilization);

    py::class_<WdlPrediction>(module, "WdlPrediction")
        .def_readonly("win", &WdlPrediction::win)
        .def_readonly("draw", &WdlPrediction::draw)
        .def_readonly("loss", &WdlPrediction::loss)
        .def_property_readonly("value", &WdlPrediction::expectedValue);
    module.attr("OutcomeProbabilities") = module.attr("WdlPrediction");

    py::class_<GameSearchVisit>(module, "GameSearchVisit")
        .def_readonly("action_id", &GameSearchVisit::action_id)
        .def_readonly("visit_count", &GameSearchVisit::visit_count);
    py::class_<GameSearchResult>(module, "GameSearchResult")
        .def_readonly("root_value", &GameSearchResult::root_value)
        .def_readonly("highest_visited_child_action_id",
                      &GameSearchResult::highest_visited_child_action_id)
        .def_readonly("highest_visited_child_visit_count",
                      &GameSearchResult::highest_visited_child_visit_count)
        .def_readonly("highest_visited_child_q", &GameSearchResult::highest_visited_child_q)
        .def_readonly("visits", &GameSearchResult::visits)
        .def_readonly("policy_target_visits", &GameSearchResult::policy_target_visits);
    py::class_<BatchedSearchParameters>(module, "BatchedSearchParameters")
        .def(py::init<std::uint32_t, TreeSearchParameters, float, float, std::size_t>(),
             py::arg("parallel_searches"), py::arg("tree_search"), py::arg("dirichlet_alpha"),
             py::arg("dirichlet_epsilon"), py::arg("tree_capacity"));
    py::class_<BatchedInferenceParameters>(module, "BatchedInferenceParameters")
        .def(py::init<std::size_t, std::size_t, std::size_t>(), py::arg("workers"),
             py::arg("batch_size"), py::arg("outstanding_batches_per_worker"))
        .def_readonly("workers", &BatchedInferenceParameters::workers)
        .def_readonly("batch_size", &BatchedInferenceParameters::batch_size)
        .def_readonly("outstanding_batches_per_worker",
                      &BatchedInferenceParameters::outstanding_batches_per_worker);
    py::class_<AnalysisParameters>(module, "AnalysisParameters")
        .def(py::init<std::uint32_t, float, BatchedInferenceParameters>(),
             py::arg("parallel_searches"), py::arg("exploration_constant"), py::arg("inference"))
        .def_readonly("parallel_searches", &AnalysisParameters::parallel_searches)
        .def_readonly("exploration_constant", &AnalysisParameters::exploration_constant)
        .def_readonly("inference", &AnalysisParameters::inference);
    py::class_<SelfPlaySearchParameters>(module, "SelfPlaySearchParameters")
        .def(py::init<std::uint32_t, std::uint32_t, std::uint32_t, TreeSearchParameters, float,
                      float>(),
             py::arg("parallel_searches"), py::arg("full_searches"), py::arg("fast_searches"),
             py::arg("tree_search"), py::arg("dirichlet_alpha"), py::arg("dirichlet_epsilon"))
        .def_readwrite("parallel_searches", &SelfPlaySearchParameters::parallel_searches)
        .def_readwrite("full_searches", &SelfPlaySearchParameters::full_searches)
        .def_readwrite("fast_searches", &SelfPlaySearchParameters::fast_searches)
        .def_readwrite("tree_search", &SelfPlaySearchParameters::tree_search)
        .def_readwrite("dirichlet_alpha", &SelfPlaySearchParameters::dirichlet_alpha)
        .def_readwrite("dirichlet_epsilon", &SelfPlaySearchParameters::dirichlet_epsilon);
    py::class_<SelfPlaySearchStatistics>(module, "SelfPlaySearchStatistics")
        .def_readonly("average_depth", &SelfPlaySearchStatistics::average_depth)
        .def_readonly("average_entropy", &SelfPlaySearchStatistics::average_entropy)
        .def_readonly("average_kl_divergence", &SelfPlaySearchStatistics::average_kl_divergence)
        .def_readonly("average_policy_search_kl_divergence",
                      &SelfPlaySearchStatistics::average_policy_search_kl_divergence)
        .def_readonly("top_action_disagreement", &SelfPlaySearchStatistics::top_action_disagreement)
        .def_readonly("selected_action_prior_rank",
                      &SelfPlaySearchStatistics::selected_action_prior_rank);
    py::class_<AnalysisCandidate>(module, "ActionAnalysisCandidate")
        .def_readonly("action_id", &AnalysisCandidate::action_id)
        .def_readonly("policy_prior", &AnalysisCandidate::policy_prior)
        .def_readonly("visits", &AnalysisCandidate::visits)
        .def_readonly("visit_share", &AnalysisCandidate::visit_share)
        .def_readonly("mean_value", &AnalysisCandidate::mean_value);
    py::class_<GameAnalysisResult>(module, "GameAnalysisResult")
        .def_readonly("chosen_action_id", &GameAnalysisResult::chosen_action_id)
        .def_readonly("value", &GameAnalysisResult::value)
        .def_readonly("outcome", &GameAnalysisResult::outcome)
        .def_readonly("candidates", &GameAnalysisResult::candidates)
        .def_readonly("searches", &GameAnalysisResult::searches)
        .def_readonly("maximum_depth", &GameAnalysisResult::maximum_depth)
        .def_readonly("elapsed_milliseconds", &GameAnalysisResult::elapsed_milliseconds)
        .def_readonly("principal_variation", &GameAnalysisResult::principal_variation);
}
