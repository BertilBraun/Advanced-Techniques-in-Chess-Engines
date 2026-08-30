#pragma once

#include "search/SelfPlay.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

[[nodiscard]] inline std::pair<std::vector<int>, std::vector<std::uint32_t>>
visitColumns(const std::vector<GameSearchVisit> &visits) {
    std::vector<int> actionIds;
    std::vector<std::uint32_t> visitCounts;
    actionIds.reserve(visits.size());
    visitCounts.reserve(visits.size());
    for (const GameSearchVisit &visit : visits) {
        actionIds.push_back(visit.action_id);
        visitCounts.push_back(visit.visit_count);
    }
    return {std::move(actionIds), std::move(visitCounts)};
}

struct SelfPlayBindingNames {
    const char *root;
    const char *search;
    const char *request;
    const char *result;
    const char *batch;
};

template <SearchGame Game> struct BoundSelfPlayClasses {
    using Root = GameSearchRoot<Game>;
    using Search = GameSelfPlaySearch<Game>;

    py::class_<Root> root;
    py::class_<Search> search;
};

template <SearchGame Game>
BoundSelfPlayClasses<Game> bindSelfPlay(py::module_ &module, const SelfPlayBindingNames names) {
    using Position = typename Game::State;
    using Root = GameSearchRoot<Game>;
    using Search = GameSelfPlaySearch<Game>;
    using Request = SelfPlaySearchRequest<Game>;
    using Result = SelfPlaySearchResult<Game>;
    using Batch = SelfPlaySearchBatch<Game>;
    static constexpr InferenceDimensions dimensions = Game::Encoding::inferenceDimensions();

    py::class_<Root> root(module, names.root);
    root.def_property_readonly("position", &Root::position)
        .def_property_readonly("is_terminal", &Root::isTerminal)
        .def_property_readonly("visits", &Root::visits)
        .def_property_readonly("live_nodes", &Root::liveNodeCount)
        .def("play", &Root::play, py::arg("action_id"))
        .def("reset", &Root::reset)
        .def(
            "discount",
            [](Root &selectedRoot, const float retainedFraction) {
                selectedRoot.tree().discount(retainedFraction);
            },
            py::arg("retained_fraction"));

    py::class_<Request>(module, names.request)
        .def(py::init(
                 [](Root selectedRoot, const std::optional<std::uint32_t> assignedAdditionalVisits,
                    std::vector<std::uint32_t> policyCheckpointVisits,
                    const std::optional<std::uint32_t> parallelSearches, const bool addRootNoise,
                    const bool forceRootPlayouts, const SearchCheckpointDetail checkpointDetail) {
                     return Request{
                         .root = std::move(selectedRoot),
                         .assigned_additional_visits = assignedAdditionalVisits,
                         .policy_checkpoint_visits = std::move(policyCheckpointVisits),
                         .parallel_searches = parallelSearches,
                         .add_root_noise = addRootNoise,
                         .force_root_playouts = forceRootPlayouts,
                         .checkpoint_detail = checkpointDetail,
                     };
                 }),
             py::arg("root"), py::arg("assigned_additional_visits") = std::nullopt,
             py::arg("policy_checkpoint_visits") = std::vector<std::uint32_t>{},
             py::arg("parallel_searches") = std::nullopt, py::arg("add_root_noise") = true,
             py::arg("force_root_playouts") = true,
             py::arg_v("checkpoint_detail", SearchCheckpointDetail::Scalars,
                       "SearchCheckpointDetail.SCALARS"))
        .def_readonly("root", &Request::root)
        .def_readonly("assigned_additional_visits", &Request::assigned_additional_visits)
        .def_readonly("policy_checkpoint_visits", &Request::policy_checkpoint_visits)
        .def_readonly("parallel_searches", &Request::parallel_searches)
        .def_readonly("add_root_noise", &Request::add_root_noise)
        .def_readonly("force_root_playouts", &Request::force_root_playouts)
        .def_readonly("checkpoint_detail", &Request::checkpoint_detail);
    py::class_<Result>(module, names.result)
        .def_readonly("root_value", &Result::root_value)
        .def_readonly("highest_visited_child_action_id", &Result::highest_visited_child_action_id)
        .def_readonly("highest_visited_child_visit_count",
                      &Result::highest_visited_child_visit_count)
        .def_readonly("highest_visited_child_q", &Result::highest_visited_child_q)
        .def_readonly("search_visits", &Result::search_visits)
        .def_readonly("policy_target_visits", &Result::policy_target_visits)
        // Columns instead of a list of bound visit objects: the worker reads these once per ply
        // per game, and one Python object per root child dominated the advance loop.
        .def_property_readonly(
            "search_visit_columns",
            [](const Result &result) { return visitColumns(result.search_visits); })
        .def_property_readonly(
            "policy_target_columns",
            [](const Result &result) { return visitColumns(result.policy_target_visits); })
        .def_readonly("network_root_value", &Result::network_root_value)
        .def_readonly("policy_correction", &Result::policy_correction)
        .def_readonly("value_correction", &Result::value_correction)
        .def_readonly("predicted_budget_curve", &Result::predicted_budget_curve)
        .def_readonly("selected_budget_index", &Result::selected_budget_index)
        .def_readonly("assigned_additional_visits", &Result::assigned_additional_visits)
        .def_readonly("parallel_searches", &Result::parallel_searches)
        .def_readonly("spend_residual", &Result::spend_residual)
        .def_readonly("starting_visits", &Result::starting_visits)
        .def_readonly("final_visits", &Result::final_visits)
        .def_readonly("stop_reason", &Result::stop_reason)
        .def_readonly("checkpoints", &Result::checkpoints)
        .def_readonly("root", &Result::root);
    py::class_<Batch>(module, names.batch)
        .def_readonly("results", &Batch::results)
        .def_readonly("statistics", &Batch::statistics)
        .def_readonly("simulations_completed", &Batch::simulations_completed);

    py::class_<Search> search(module, names.search);
    search
        .def(py::init<const InferenceConfiguration &, const SelfPlaySearchParameters &,
                      BatchedInferenceParameters, std::uint64_t>(),
             py::arg("runtime_parameters"), py::arg("search_parameters"),
             py::arg("inference_parameters"), py::arg("initial_model_generation") = 0)
        .def_static("inference_dimensions", []() { return dimensions; })
        .def(
            "new_root",
            [](Search &selectedSearch, const Position &position,
               const std::size_t maximumCapacity) {
                return selectedSearch.newRoot(position, maximumCapacity);
            },
            py::arg("position"), py::arg("maximum_capacity") = 0)
        .def(
            "request",
            [](const Search &, Root selectedRoot,
               const std::optional<std::uint32_t> assignedAdditionalVisits,
               std::vector<std::uint32_t> policyCheckpointVisits,
               const std::optional<std::uint32_t> parallelSearches, const bool addRootNoise,
               const bool forceRootPlayouts, const SearchCheckpointDetail checkpointDetail) {
                return Request{
                    .root = std::move(selectedRoot),
                    .assigned_additional_visits = assignedAdditionalVisits,
                    .policy_checkpoint_visits = std::move(policyCheckpointVisits),
                    .parallel_searches = parallelSearches,
                    .add_root_noise = addRootNoise,
                    .force_root_playouts = forceRootPlayouts,
                    .checkpoint_detail = checkpointDetail,
                };
            },
            py::arg("root"), py::arg("assigned_additional_visits") = std::nullopt,
            py::arg("policy_checkpoint_visits") = std::vector<std::uint32_t>{},
            py::arg("parallel_searches") = std::nullopt, py::arg("add_root_noise") = true,
            py::arg("force_root_playouts") = true,
            py::arg_v("checkpoint_detail", SearchCheckpointDetail::Scalars,
                      "SearchCheckpointDetail.SCALARS"))
        .def("search", &Search::search, py::arg("requests"), py::arg("collect_statistics") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("refresh_model", &Search::refreshModel, py::arg("model_generation"),
             py::arg("model_path"), py::call_guard<py::gil_scoped_release>())
        .def("update_search_schedule", &Search::updateSearchSchedule, py::arg("search_parameters"))
        .def("reset_spend_residual", &Search::resetSpendResidual)
        .def_property_readonly("spend_residual", &Search::spendResidual)
        .def_property_readonly("model_generation", &Search::modelGeneration)
        .def("inference_statistics", &Search::inferenceStatistics);

    return {.root = root, .search = search};
}
