#include "games/chess/bindings/ChessBindings.hpp"

#include "games/chess/ChessGame.hpp"
#include "games/chess/encoding/ChessEncoding.hpp"

#include "bitboard.h"
#include "position.h"

#include <cstdint>
#include <span>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_chess_search(py::module_ &module);
void bind_chess_analysis(py::module_ &module);

void bind_chess_game(py::module_ &module) {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();

    py::class_<Board>(module, "ChessPosition")
        .def(py::init<>())
        .def(py::init<const std::string &>(), py::arg("fen"))
        .def_property_readonly("current_player", &Board::currentPlayer)
        .def_property_readonly("is_terminal", &Board::isGameOver)
        .def_property_readonly("fen", &Board::fen)
        .def(
            "legal_actions",
            [](const Board &position) {
                std::vector<int> actionIds;
                actionIds.reserve(position.validMoves().size());
                for (const Stockfish::Move move : position.validMoves()) {
                    actionIds.push_back(ChessEncoding::actionId(ChessAction(move), position));
                }
                return actionIds;
            })
        .def(
            "child",
            [](const Board &position, const int actionId) {
                if (actionId < 0 || actionId >= ChessEncoding::action_count) {
                    throw std::invalid_argument("Chess action ID is outside the action space");
                }
                for (const Stockfish::Move move : position.validMoves()) {
                    if (ChessEncoding::actionId(ChessAction(move), position) == actionId) {
                        Board child(position);
                        child.makeMove(move);
                        return child;
                    }
                }
                throw std::invalid_argument("Chess action ID is illegal in this position");
            },
            py::arg("action_id"))
        .def(
            "terminal_value",
            [](const Board &position) {
                if (!position.isGameOver()) {
                    throw std::invalid_argument("Terminal value requires a terminal chess position");
                }
                return ChessGame::terminalValue(position);
            })
        .def("approximate_result_score", &Board::approximateResultScore)
        .def(
            "packed_encoding",
            [](const Board &position) {
                const CompressedEncodedBoard encoded = encodeBoard(position);
                std::string payload(CompressedEncodedBoard::packed_bytes, '\0');
                encoded.writePackedInto(
                    std::span(reinterpret_cast<std::int8_t *>(payload.data()), payload.size()));
                return py::bytes(payload);
            })
        .def("__repr__", &Board::repr);

    module.def(
        "mirror_chess_action_id",
        [](const int actionId) {
            if (actionId < 0 || actionId >= ChessEncoding::action_count) {
                throw std::invalid_argument("Chess action ID is outside the action space");
            }
            return ChessEncoding::mirrorActionId(actionId);
        },
        py::arg("action_id"));

    module.def(
        "encode_board_packed_bytes",
        [](const std::string &fen) {
            const Board board(fen);
            const CompressedEncodedBoard encoded = encodeBoard(board);
            std::string payload(CompressedEncodedBoard::packed_bytes, '\0');
            encoded.writePackedInto(
                std::span(reinterpret_cast<std::int8_t *>(payload.data()), payload.size()));
            return py::bytes(payload);
        },
        py::arg("fen"),
        R"pbdoc(Encode a FEN into the canonical packed plane-major byte layout.)pbdoc");

    bind_chess_search(module);
    bind_chess_analysis(module);
}
