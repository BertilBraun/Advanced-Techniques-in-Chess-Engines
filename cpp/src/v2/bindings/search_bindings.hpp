#pragma once

#include <pybind11/pybind11.h>

namespace az::v2::bindings {

void bindSearch(pybind11::module_ &module);

} // namespace az::v2::bindings
