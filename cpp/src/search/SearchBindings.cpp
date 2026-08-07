#include "search/SearchBindings.hpp"

#include "search/InferenceConfiguration.hpp"
#include "search/SearchTypes.hpp"
#include "util/TimeItGuard.h"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_search(py::module_ &module) {
    py::enum_<InferenceDevice>(module, "InferenceDevice")
        .value("AUTO", InferenceDevice::Auto)
        .value("CPU", InferenceDevice::Cpu)
        .value("CUDA", InferenceDevice::Cuda);

    py::class_<InferenceRuntimeParameters>(module, "InferenceRuntimeParameters")
        .def(py::init<int, std::string, InferenceDevice>(), py::arg("device_id"),
             py::arg("model_path"), py::arg("device") = InferenceDevice::Auto)
        .def_readwrite("device_id", &InferenceRuntimeParameters::device_id)
        .def_readwrite("model_path", &InferenceRuntimeParameters::model_path)
        .def_readwrite("device", &InferenceRuntimeParameters::device);

    py::class_<InferenceDimensions>(module, "InferenceDimensions")
        .def(py::init<int, int, int, int, int>(), py::arg("channels"), py::arg("rows"),
             py::arg("columns"), py::arg("actions"), py::arg("outcomes"))
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
        .def_readonly("directInferenceNanoseconds",
                      &InferenceStatistics::directInferenceNanoseconds)
        .def_readonly("directWorkerUtilization", &InferenceStatistics::directWorkerUtilization);

    py::class_<FunctionTimeInfo>(module, "FunctionTimeInfo")
        .def_readonly("name", &FunctionTimeInfo::name)
        .def_readonly("percent", &FunctionTimeInfo::percent)
        .def_readonly("total", &FunctionTimeInfo::total)
        .def_readonly("invocations", &FunctionTimeInfo::invocations);

    py::class_<TimeInfo>(module, "TimeInfo")
        .def_readonly("totalTime", &TimeInfo::totalTime)
        .def_readonly("percentRecorded", &TimeInfo::percentRecorded)
        .def_readonly("functionTimes", &TimeInfo::functionTimes);

    py::class_<GameSearchVisit>(module, "GameSearchVisit")
        .def_readonly("action_id", &GameSearchVisit::action_id)
        .def_readonly("visit_count", &GameSearchVisit::visit_count);
    py::class_<GameSearchResult>(module, "GameSearchResult")
        .def_readonly("root_value", &GameSearchResult::root_value)
        .def_readonly("visits", &GameSearchResult::visits);
    py::class_<BatchedSearchParameters>(module, "BatchedSearchParameters")
        .def(py::init<std::uint32_t, float, std::uint32_t, float, float, std::size_t>(),
             py::arg("parallel_searches"), py::arg("exploration_constant"),
             py::arg("minimum_root_visits"), py::arg("dirichlet_alpha"),
             py::arg("dirichlet_epsilon"), py::arg("tree_capacity"));
    py::class_<BatchedInferenceParameters>(module, "BatchedInferenceParameters")
        .def(py::init<std::size_t, std::size_t, std::size_t>(), py::arg("workers"),
             py::arg("batch_size"), py::arg("outstanding_batches_per_worker"));
}
