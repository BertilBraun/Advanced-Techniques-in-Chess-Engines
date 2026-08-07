#include "common.hpp"

#include "games/chess/ChessBindings.hpp"
#include "games/go/GoBindings.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
void initializeRuntime() {
    Bitboards::init();
    Position::init();
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("MKL_NUM_THREADS", "1", 1);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
}
} // namespace

PYBIND11_MODULE(AlphaZeroCpp, module) {
    module.doc() = "Native game search and inference";
    initializeRuntime();
    bind_chess_game(module);
    bind_go_game(module);
}
