#pragma once

#include <pybind11/pybind11.h>

namespace az::v2::bindings {

void bind_search(pybind11::module_ &module);

} // namespace az::v2::bindings
