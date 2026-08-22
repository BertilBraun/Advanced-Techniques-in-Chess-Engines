#include "games/go/bindings/GoBindings.hpp"

#include "games/go/GoGame.hpp"
#include "util/py.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_go_search(py::module_ &module);

namespace {

template <std::size_t BoardSize> std::vector<int> pointIds(const BitBoard<BoardSize> &board) {
    std::vector<int> points;
    points.reserve(board.count());
    for (const auto point : board.setBits()) {
        points.push_back(static_cast<int>(BitBoard<BoardSize>::index(point)));
    }
    return points;
}

template <std::size_t BoardSize>
void setPoints(BitBoard<BoardSize> &board, const std::vector<int> &points) {
    for (const int point : points) {
        if (point < 0 || point >= static_cast<int>(BitBoard<BoardSize>::bitCount)) {
            throw std::invalid_argument("Bound Go point is outside the board");
        }
        board.set(static_cast<std::size_t>(point));
    }
}

template <std::size_t BoardSize, std::size_t HistoryLength>
GoPosition<BoardSize, HistoryLength>
restorePosition(const std::vector<std::vector<int>> &blackHistory,
                const std::vector<std::vector<int>> &whiteHistory, const GoPlayer player,
                const std::optional<int> koPoint, const int consecutivePasses, const int moveNumber,
                const GoRules rules) {
    if (blackHistory.size() > HistoryLength || whiteHistory.size() > HistoryLength) {
        throw std::invalid_argument("Bound Go history exceeds the retained history length");
    }
    std::array<GoBoard<BoardSize>, HistoryLength> history{};
    for (const auto offset : range(blackHistory.size())) {
        setPoints(history[offset].black, blackHistory[offset]);
    }
    for (const auto offset : range(whiteHistory.size())) {
        setPoints(history[offset].white, whiteHistory[offset]);
    }
    std::optional<typename BitBoard<BoardSize>::Point> typedKo;
    if (koPoint.has_value()) {
        if (*koPoint < 0 || *koPoint >= static_cast<int>(BitBoard<BoardSize>::bitCount)) {
            throw std::invalid_argument("Bound Go ko point is outside the board");
        }
        typedKo = BitBoard<BoardSize>::point(static_cast<std::size_t>(*koPoint));
    }
    return GoPosition<BoardSize, HistoryLength>::restore(history, player, typedKo,
                                                         consecutivePasses, moveNumber, rules);
}

template <std::size_t BoardSize, std::size_t HistoryLength>
void bindPosition(py::module_ &module, const char *name) {
    using Contract = GoGame<BoardSize, HistoryLength>;
    using Position = typename Contract::State;
    using Encoded = EncodedGoPosition<BoardSize, HistoryLength>;

    py::class_<Position>(module, name)
        .def(py::init<GoRules>())
        .def_static("restore", &restorePosition<BoardSize, HistoryLength>, py::arg("black_history"),
                    py::arg("white_history"), py::arg("player"), py::arg("ko_point"),
                    py::arg("consecutive_passes"), py::arg("move_number"), py::arg("rules"))
        .def_property_readonly("board_size", [](const Position &) { return BoardSize; })
        .def_property_readonly("history_length", [](const Position &) { return HistoryLength; })
        .def_property_readonly("player", &Position::player)
        .def_property_readonly("ko_point",
                               [](const Position &position) -> std::optional<int> {
                                   if (!position.koPoint().has_value()) {
                                       return std::nullopt;
                                   }
                                   return static_cast<int>(
                                       BitBoard<BoardSize>::index(*position.koPoint()));
                               })
        .def_property_readonly("consecutive_passes", &Position::consecutivePasses)
        .def_property_readonly("move_number", &Position::moveNumber)
        .def_property_readonly("rules", &Position::rules)
        .def("black_points",
             [](const Position &position, const std::size_t historyOffset) {
                 if (historyOffset >= HistoryLength) {
                     throw std::out_of_range("Go history offset is outside the retained history");
                 }
                 return pointIds(position.history()[historyOffset].black);
             })
        .def("white_points",
             [](const Position &position, const std::size_t historyOffset) {
                 if (historyOffset >= HistoryLength) {
                     throw std::out_of_range("Go history offset is outside the retained history");
                 }
                 return pointIds(position.history()[historyOffset].white);
             })
        .def("is_legal",
             [](const Position &position, const int actionId) {
                 return position.isLegal(GoAction<BoardSize>(actionId));
             })
        .def("legal_actions",
             [](const Position &position) {
                 std::vector<int> ids;
                 for (const GoAction<BoardSize> action : position.legalActions()) {
                     ids.push_back(action.id);
                 }
                 return ids;
             })
        .def("child",
             [](const Position &position, const int actionId) {
                 return position.child(GoAction<BoardSize>(actionId));
             })
        .def_property_readonly("termination_reason", &Position::terminationReason)
        .def_property_readonly("is_terminal", &Position::isTerminal)
        .def("area_score", &Position::areaScore)
        .def("terminal_result", &Position::terminalResult)
        .def("terminal_value",
             [](const Position &position) { return Contract::terminalValue(position); })
        .def("state_hash", &Position::hash)
        .def("packed_encoding",
             [](const Position &position) {
                 const Encoded encoded = encodeGoPosition(position);
                 std::string payload(Encoded::packedBytes, '\0');
                 encoded.writePackedInto(
                     std::span(reinterpret_cast<std::int8_t *>(payload.data()), payload.size()));
                 return py::bytes(payload);
             })
        .def("tensor_encoding",
             [](const Position &position) {
                 std::vector<std::int8_t> values(
                     GoRepresentationDimensions<BoardSize, HistoryLength>::channelCount *
                     BitBoard<BoardSize>::bitCount);
                 Contract::Encoding::encodeInputInto(position, values.data());
                 return values;
             })
        .def(py::self == py::self);
}

} // namespace

void bind_go_game(py::module_ &module) {
    py::enum_<GoPlayer>(module, "GoPlayer")
        .value("BLACK", GoPlayer::black)
        .value("WHITE", GoPlayer::white);
    py::enum_<GoTerminationReason>(module, "GoTerminationReason")
        .value("ONGOING", GoTerminationReason::ongoing)
        .value("TWO_PASSES", GoTerminationReason::two_passes)
        .value("MAXIMUM_MOVES", GoTerminationReason::maximum_moves);
    py::class_<GoRules>(module, "GoRules")
        .def(py::init<int, int>(), py::arg("komi_half_points"), py::arg("maximum_moves"))
        .def_readonly("komi_half_points", &GoRules::komi_half_points)
        .def_readonly("maximum_moves", &GoRules::maximum_moves)
        .def(py::self == py::self);
    py::class_<GoAreaScore>(module, "GoAreaScore")
        .def_readonly("black_half_points", &GoAreaScore::black_half_points)
        .def_readonly("white_half_points", &GoAreaScore::white_half_points)
        .def_property_readonly("winner", &GoAreaScore::winner)
        .def(py::self == py::self);
    py::class_<GoTerminalResult>(module, "GoTerminalResult")
        .def_readonly("reason", &GoTerminalResult::reason)
        .def_readonly("score", &GoTerminalResult::score)
        .def_readonly("winner", &GoTerminalResult::winner)
        .def(py::self == py::self);
    bindPosition<7, 8>(module, "GoPosition7");
    bindPosition<9, 8>(module, "GoPosition9");
    bind_go_search(module);
}
