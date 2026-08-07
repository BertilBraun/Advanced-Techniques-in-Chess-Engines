#include "games/go/GoBindings.hpp"

#include "games/go/GoGameContract.hpp"
#include "search/Analysis.hpp"
#include "search/SearchEngine.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

template <std::size_t BoardSize> std::vector<int> point_ids(const BitBoard<BoardSize> &board) {
    std::vector<int> points;
    points.reserve(board.count());
    for (const auto point : board.set_bits()) {
        points.push_back(static_cast<int>(BitBoard<BoardSize>::index(point)));
    }
    return points;
}

template <std::size_t BoardSize>
void set_points(BitBoard<BoardSize> &board, const std::vector<int> &points) {
    for (const int point : points) {
        if (point < 0 || point >= static_cast<int>(BitBoard<BoardSize>::bit_count)) {
            throw std::invalid_argument("Bound Go point is outside the board");
        }
        board.set(static_cast<std::size_t>(point));
    }
}

template <std::size_t BoardSize, std::size_t HistoryLength>
GoPosition<BoardSize, HistoryLength>
restore_position(const std::vector<std::vector<int>> &black_history,
                 const std::vector<std::vector<int>> &white_history, const GoPlayer player,
                 const std::optional<int> ko_point, const int consecutive_passes,
                 const int move_number, const GoRules rules) {
    if (black_history.size() > HistoryLength || white_history.size() > HistoryLength) {
        throw std::invalid_argument("Bound Go history exceeds the retained history length");
    }
    std::array<GoBoard<BoardSize>, HistoryLength> history{};
    for (std::size_t offset = 0; offset < black_history.size(); ++offset) {
        set_points(history[offset].black, black_history[offset]);
    }
    for (std::size_t offset = 0; offset < white_history.size(); ++offset) {
        set_points(history[offset].white, white_history[offset]);
    }
    std::optional<typename BitBoard<BoardSize>::Point> typed_ko;
    if (ko_point.has_value()) {
        if (*ko_point < 0 || *ko_point >= static_cast<int>(BitBoard<BoardSize>::bit_count)) {
            throw std::invalid_argument("Bound Go ko point is outside the board");
        }
        typed_ko = BitBoard<BoardSize>::point(static_cast<std::size_t>(*ko_point));
    }
    return GoPosition<BoardSize, HistoryLength>::restore(history, player, typed_ko,
                                                         consecutive_passes, move_number, rules);
}

template <std::size_t BoardSize>
void bind_go_search(py::module_ &module, const char *rootName, const char *searchName,
                    const char *analysisName) {
    using Contract = GoGameContract<BoardSize, 8>;
    using Root = GameSearchRoot<Contract>;
    using Search = BatchedGameSearch<Contract>;
    using Analysis = GameAnalysis<Contract>;
    constexpr InferenceDimensions dimensions = Contract::inferenceDimensions();

    py::class_<Root>(module, rootName)
        .def_property_readonly("position", &Root::position)
        .def_property_readonly("is_terminal", &Root::isTerminal)
        .def_property_readonly("visits", &Root::visits)
        .def_property_readonly("live_nodes", &Root::liveNodeCount)
        .def_property_readonly("children",
                               [](const Root &root) {
                                   std::vector<std::pair<int, std::uint32_t>> children;
                                   for (const auto &edge : root.tree().root().children) {
                                       children.emplace_back(
                                           Contract::actionId(edge.action, root.position()),
                                           edge.visits);
                                   }
                                   return children;
                               })
        .def("play", &Root::play, py::arg("action_id"));

    py::class_<Search>(module, searchName)
        .def(py::init<const std::string &, InferenceDevice, int, BatchedInferenceParameters,
                      BatchedSearchParameters, std::uint64_t>(),
             py::arg("model_path"), py::arg("device"), py::arg("device_id"),
             py::arg("inference_parameters"), py::arg("search_parameters"),
             py::arg("model_generation"))
        .def_static("inference_dimensions", []() { return dimensions; })
        .def(
            "new_root",
            [](Search &search, const GoRules rules) {
                return search.newRoot(Contract::initialPosition(rules));
            },
            py::arg("rules"))
        .def(
            "search",
            [](Search &search, std::vector<Root> roots, const std::uint32_t simulations) {
                return search.search(roots, simulations);
            },
            py::arg("roots"), py::arg("simulations"), py::call_guard<py::gil_scoped_release>())
        .def("refresh_model", &Search::refreshModel, py::arg("model_generation"),
             py::arg("model_path"), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("model_generation", &Search::modelGeneration);

    py::class_<Analysis, std::shared_ptr<Analysis>>(module, analysisName)
        .def(py::init<const InferenceRuntimeParameters &, const AnalysisParameters &>(),
             py::arg("runtime_parameters"), py::arg("parameters"))
        .def(
            "new_root",
            [](Analysis &analysis, const GoRules rules) {
                return analysis.newRoot(Contract::initialPosition(rules));
            },
            py::arg("rules"))
        .def("analyze_policy", &Analysis::analyzePolicy, py::arg("root"),
             py::call_guard<py::gil_scoped_release>())
        .def("analyze_counted", &Analysis::analyzeCounted, py::arg("root"),
             py::arg("searches"), py::call_guard<py::gil_scoped_release>())
        .def(
            "analyze_timed",
            [](Analysis &analysis, Root &root, const int seconds) {
                return analysis.analyzeTimed(root, std::chrono::seconds(seconds));
            },
            py::arg("root"), py::arg("seconds"), py::call_guard<py::gil_scoped_release>())
        .def("inference_statistics", &Analysis::inferenceStatistics);
}

template <std::size_t BoardSize, std::size_t HistoryLength>
void bind_position(py::module_ &module, const char *name) {
    using Contract = GoGameContract<BoardSize, HistoryLength>;
    using Position = typename Contract::Position;
    using Encoded = typename Contract::EncodedPosition;

    py::class_<Position>(module, name)
        .def(py::init<GoRules>())
        .def_static("restore", &restore_position<BoardSize, HistoryLength>,
                    py::arg("black_history"), py::arg("white_history"), py::arg("player"),
                    py::arg("ko_point"), py::arg("consecutive_passes"), py::arg("move_number"),
                    py::arg("rules"))
        .def_property_readonly("board_size", [](const Position &) { return BoardSize; })
        .def_property_readonly("history_length", [](const Position &) { return HistoryLength; })
        .def_property_readonly("player", &Position::player)
        .def_property_readonly("ko_point",
                               [](const Position &position) -> std::optional<int> {
                                   if (!position.ko_point().has_value()) {
                                       return std::nullopt;
                                   }
                                   return static_cast<int>(
                                       BitBoard<BoardSize>::index(*position.ko_point()));
                               })
        .def_property_readonly("consecutive_passes", &Position::consecutive_passes)
        .def_property_readonly("move_number", &Position::move_number)
        .def_property_readonly("rules", &Position::rules)
        .def("black_points",
             [](const Position &position, const std::size_t history_offset) {
                 if (history_offset >= HistoryLength) {
                     throw std::out_of_range("Go history offset is outside the retained history");
                 }
                 return point_ids(position.history()[history_offset].black);
             })
        .def("white_points",
             [](const Position &position, const std::size_t history_offset) {
                 if (history_offset >= HistoryLength) {
                     throw std::out_of_range("Go history offset is outside the retained history");
                 }
                 return point_ids(position.history()[history_offset].white);
             })
        .def("is_legal",
             [](const Position &position, const int action_id) {
                 return position.is_legal(GoAction<BoardSize>(action_id));
             })
        .def("legal_actions",
             [](const Position &position) {
                 std::vector<int> ids;
                 for (const GoAction<BoardSize> action : position.legal_actions()) {
                     ids.push_back(action.id);
                 }
                 return ids;
             })
        .def("child",
             [](const Position &position, const int action_id) {
                 return position.child(GoAction<BoardSize>(action_id));
             })
        .def_property_readonly("termination_reason", &Position::termination_reason)
        .def_property_readonly("is_terminal", &Position::is_terminal)
        .def("area_score", &Position::area_score)
        .def("terminal_result", &Position::terminal_result)
        .def("terminal_value",
             [](const Position &position) { return Contract::terminalValue(position); })
        .def("action_id",
             [](const Position &position, const int action_id) {
                 return Contract::actionId(GoAction<BoardSize>(action_id), position);
             })
        .def("decode_actions",
             [](const Position &position, const std::vector<int> &action_ids) {
                 std::vector<int> decoded;
                 for (const GoAction<BoardSize> action :
                      Contract::decodeActions(action_ids, position)) {
                     decoded.push_back(action.id);
                 }
                 return decoded;
             })
        .def("state_hash", &Position::hash)
        .def("packed_encoding",
             [](const Position &position) {
                 const Encoded encoded = Contract::encodeInput(position);
                 std::string payload(Encoded::packed_bytes, '\0');
                 Contract::writePackedInput(encoded,
                                            reinterpret_cast<std::int8_t *>(payload.data()));
                 return py::bytes(payload);
             })
        .def("tensor_encoding",
             [](const Position &position) {
                 std::vector<std::int8_t> values(
                     GoRepresentationDimensions<BoardSize, HistoryLength>::channel_count *
                     BitBoard<BoardSize>::bit_count);
                 Contract::encodeInputInto(position, values.data());
                 return values;
             })
        .def_static(
            "transform_action",
            [](const int action_id, const GoSymmetry symmetry) {
                return Contract::transformAction(GoAction<BoardSize>(action_id), symmetry).id;
            })
        .def_static("inverse_symmetry", &Contract::inverseSymmetry)
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
    py::enum_<GoSymmetry>(module, "GoSymmetry")
        .value("IDENTITY", GoSymmetry::identity)
        .value("ROTATE_90", GoSymmetry::rotate_90)
        .value("ROTATE_180", GoSymmetry::rotate_180)
        .value("ROTATE_270", GoSymmetry::rotate_270)
        .value("REFLECT", GoSymmetry::reflect)
        .value("REFLECT_ROTATE_90", GoSymmetry::reflect_rotate_90)
        .value("REFLECT_ROTATE_180", GoSymmetry::reflect_rotate_180)
        .value("REFLECT_ROTATE_270", GoSymmetry::reflect_rotate_270);
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
    bind_position<7, 8>(module, "GoPosition7");
    bind_position<9, 8>(module, "GoPosition9");
    bind_go_search<7>(module, "GoSearchRoot7", "GoBatchedSearch7", "GoAnalysis7");
    bind_go_search<9>(module, "GoSearchRoot9", "GoBatchedSearch9", "GoAnalysis9");
}
