#include "bindings/go_bindings.hpp"
#include "bindings/search_bindings.hpp"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(az_go_native, module) {
    module.doc() = "Typed native Go correctness core";
    az::bindings::bindGo(module);
    az::bindings::bindSearch(module);
}
