#include "games/chess/bindings/ChessBindings.hpp"
#include "games/go/bindings/GoBindings.hpp"
#include "search/SearchBindings.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(AlphaZeroCpp, module) {
    module.doc() = "Native game search and inference";
    bind_search(module);
    bind_chess_game(module);
    bind_go_game(module);
}
