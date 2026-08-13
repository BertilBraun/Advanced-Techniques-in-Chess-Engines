#pragma once

#include "search/SelfPlay.hpp"

#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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
    constexpr InferenceDimensions dimensions = Game::Encoding::inferenceDimensions();

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
        .def(py::init<Root, bool>(), py::arg("root"), py::arg("full_search"))
        .def_readonly("root", &Request::root)
        .def_readonly("full_search", &Request::full_search);
    py::class_<Result>(module, names.result)
        .def_readonly("root_value", &Result::root_value)
        .def_readonly("highest_visited_child_action_id", &Result::highest_visited_child_action_id)
        .def_readonly("highest_visited_child_visit_count", &Result::highest_visited_child_visit_count)
        .def_readonly("highest_visited_child_q", &Result::highest_visited_child_q)
        .def_readonly("search_visits", &Result::search_visits)
        .def_readonly("policy_target_visits", &Result::policy_target_visits)
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
            [](Search &selectedSearch, const Position &position) {
                return selectedSearch.newRoot(position);
            },
            py::arg("position"))
        .def(
            "request",
            [](const Search &, Root selectedRoot, const bool fullSearch) {
                return Request(std::move(selectedRoot), fullSearch);
            },
            py::arg("root"), py::arg("full_search"))
        .def("search", &Search::search, py::arg("requests"), py::arg("collect_statistics") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("refresh_model", &Search::refreshModel, py::arg("model_generation"),
             py::arg("model_path"), py::call_guard<py::gil_scoped_release>())
        .def("update_search_schedule", &Search::updateSearchSchedule, py::arg("search_parameters"))
        .def_property_readonly("model_generation", &Search::modelGeneration)
        .def("inference_statistics", &Search::inferenceStatistics);

    return {.root = root, .search = search};
}
