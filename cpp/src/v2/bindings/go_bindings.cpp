#include "bindings/go_bindings.hpp"

#include "games/go/GoEncoding.hpp"
#include "games/go/GoState.hpp"
#include "games/go/GoSymmetry.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace az::v2::bindings {

void bindGo(pybind11::module_ &module) {
    namespace py = pybind11;
    using games::go::AreaScore;
    using games::go::Coordinate;
    using games::go::GoEncoding;
    using games::go::GoRules;
    using games::go::GoState;
    using games::go::Player;
    using games::go::Stone;
    using games::go::Symmetry;
    using games::go::TerminalResult;
    using games::go::TerminationReason;

    module.attr("MAXIMUM_HISTORY_LENGTH") = games::go::MAXIMUM_HISTORY_LENGTH;

    py::enum_<Stone>(module, "Stone")
        .value("EMPTY", Stone::Empty)
        .value("BLACK", Stone::Black)
        .value("WHITE", Stone::White);
    py::enum_<Player>(module, "Player").value("BLACK", Player::Black).value("WHITE", Player::White);
    py::enum_<TerminationReason>(module, "TerminationReason")
        .value("ONGOING", TerminationReason::Ongoing)
        .value("TWO_PASSES", TerminationReason::TwoPasses)
        .value("SAFETY_PLY_CAP", TerminationReason::SafetyPlyCap);
    py::enum_<Symmetry>(module, "Symmetry")
        .value("IDENTITY", Symmetry::Identity)
        .value("ROTATE_90", Symmetry::Rotate90)
        .value("ROTATE_180", Symmetry::Rotate180)
        .value("ROTATE_270", Symmetry::Rotate270)
        .value("REFLECT", Symmetry::Reflect)
        .value("REFLECT_ROTATE_90", Symmetry::ReflectRotate90)
        .value("REFLECT_ROTATE_180", Symmetry::ReflectRotate180)
        .value("REFLECT_ROTATE_270", Symmetry::ReflectRotate270);

    py::class_<Coordinate>(module, "Coordinate")
        .def_readonly("row", &Coordinate::row)
        .def_readonly("column", &Coordinate::column)
        .def(py::self == py::self);

    py::class_<GoRules>(module, "GoRules")
        .def(py::init<int32, int32, int32, int32>(), py::arg("board_size"),
             py::arg("komi_half_points"), py::arg("safety_ply_cap"), py::arg("history_length"))
        .def_readonly("board_size", &GoRules::boardSize)
        .def_readonly("komi_half_points", &GoRules::komiHalfPoints)
        .def_readonly("safety_ply_cap", &GoRules::safetyPlyCap)
        .def_readonly("history_length", &GoRules::historyLength)
        .def(py::self == py::self);

    py::class_<AreaScore>(module, "AreaScore")
        .def_readonly("black_twice", &AreaScore::blackTwice)
        .def_readonly("white_twice", &AreaScore::whiteTwice)
        .def_property_readonly("winner", &AreaScore::winner)
        .def(py::self == py::self);

    py::class_<TerminalResult>(module, "TerminalResult")
        .def_readonly("reason", &TerminalResult::reason)
        .def_readonly("score", &TerminalResult::score)
        .def_readonly("winner", &TerminalResult::winner)
        .def(py::self == py::self);

    py::class_<GoEncoding>(module, "GoEncoding")
        .def_readonly("planes", &GoEncoding::planes)
        .def_readonly("board_size", &GoEncoding::boardSize)
        .def_readonly("values", &GoEncoding::values)
        .def("at", &GoEncoding::at)
        .def(py::self == py::self);

    py::class_<GoState>(module, "GoState")
        .def(py::init<GoRules>())
        .def_static("restore", &GoState::restore, py::arg("rules"), py::arg("board"),
                    py::arg("current_player"), py::arg("ply"), py::arg("consecutive_passes"),
                    py::arg("position_history"))
        .def_property_readonly("rules", &GoState::rules)
        .def_property_readonly("board_size", &GoState::boardSize)
        .def_property_readonly("action_count", &GoState::actionCount)
        .def_property_readonly("pass_action", &GoState::passAction)
        .def_property_readonly("current_player", &GoState::currentPlayer)
        .def_property_readonly("ply", &GoState::ply)
        .def_property_readonly("consecutive_passes", &GoState::consecutivePasses)
        .def_property_readonly("board", &GoState::board)
        .def("is_legal", &GoState::isLegal)
        .def("legal_actions", &GoState::legalActions)
        .def("apply", &GoState::apply)
        .def_property_readonly("termination_reason", &GoState::terminationReason)
        .def_property_readonly("is_terminal", &GoState::isTerminal)
        .def("terminal_result", &GoState::terminalResult)
        .def("area_score", &GoState::areaScore)
        .def("canonical_encoding", &GoState::canonicalEncoding)
        .def("copy", [](const GoState &state) { return GoState(state); })
        .def("state_hash", &GoState::stateHash)
        .def(py::self == py::self);

    module.def("inverse_symmetry", &games::go::inverseSymmetry);
    module.def("transform_action", &games::go::transformAction);
    module.def("transform_coordinate", &games::go::transformCoordinate);
    module.def("transform_encoding", &games::go::transformEncoding);
}

} // namespace az::v2::bindings
