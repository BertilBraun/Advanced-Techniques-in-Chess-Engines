#include "bindings/go_bindings.hpp"

#include "games/go/GoEncoding.hpp"
#include "games/go/GoState.hpp"
#include "games/go/GoSymmetry.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace az::v2::bindings {

void bind_go(pybind11::module_ &module) {
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

    module.attr("MAXIMUM_HISTORY_LENGTH") = games::go::maximum_history_length;

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
        .def(py::init<std::int32_t, std::int32_t, std::int32_t, std::int32_t>(),
             py::arg("board_size"), py::arg("komi_half_points"), py::arg("safety_ply_cap"),
             py::arg("history_length"))
        .def_readonly("board_size", &GoRules::board_size)
        .def_readonly("komi_half_points", &GoRules::komi_half_points)
        .def_readonly("safety_ply_cap", &GoRules::safety_ply_cap)
        .def_readonly("history_length", &GoRules::history_length)
        .def(py::self == py::self);

    py::class_<AreaScore>(module, "AreaScore")
        .def_readonly("black_twice", &AreaScore::black_twice)
        .def_readonly("white_twice", &AreaScore::white_twice)
        .def_property_readonly("winner", &AreaScore::winner)
        .def(py::self == py::self);

    py::class_<TerminalResult>(module, "TerminalResult")
        .def_readonly("reason", &TerminalResult::reason)
        .def_readonly("score", &TerminalResult::score)
        .def_readonly("winner", &TerminalResult::winner)
        .def(py::self == py::self);

    py::class_<GoEncoding>(module, "GoEncoding")
        .def_readonly("planes", &GoEncoding::planes)
        .def_readonly("board_size", &GoEncoding::board_size)
        .def_readonly("values", &GoEncoding::values)
        .def("at", &GoEncoding::at)
        .def(py::self == py::self);

    py::class_<GoState>(module, "GoState")
        .def(py::init<GoRules>())
        .def_static("restore", &GoState::restore, py::arg("rules"), py::arg("board"),
                    py::arg("current_player"), py::arg("ply"), py::arg("consecutive_passes"),
                    py::arg("position_history"))
        .def_property_readonly("rules", &GoState::rules)
        .def_property_readonly("board_size", &GoState::board_size)
        .def_property_readonly("action_count", &GoState::action_count)
        .def_property_readonly("pass_action", &GoState::pass_action)
        .def_property_readonly("current_player", &GoState::current_player)
        .def_property_readonly("ply", &GoState::ply)
        .def_property_readonly("consecutive_passes", &GoState::consecutive_passes)
        .def_property_readonly("board", &GoState::board)
        .def("is_legal", &GoState::is_legal)
        .def("legal_actions", &GoState::legal_actions)
        .def("apply", &GoState::apply)
        .def_property_readonly("termination_reason", &GoState::termination_reason)
        .def_property_readonly("is_terminal", &GoState::is_terminal)
        .def("terminal_result", &GoState::terminal_result)
        .def("area_score", &GoState::area_score)
        .def("canonical_encoding", &GoState::canonical_encoding)
        .def("copy", [](const GoState &state) { return GoState(state); })
        .def("state_hash", &GoState::state_hash)
        .def(py::self == py::self);

    module.def("inverse_symmetry", &games::go::inverse_symmetry);
    module.def("transform_action", &games::go::transform_action);
    module.def("transform_coordinate", &games::go::transform_coordinate);
    module.def("transform_encoding", &games::go::transform_encoding);
}

} // namespace az::v2::bindings
