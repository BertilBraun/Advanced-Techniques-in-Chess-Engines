#pragma once

#include "common.hpp"

#include "games/chess/ChessAction.hpp"

static void testMoveEncoding() {
    for (int _ : range(100)) {
        Board board;

        while (!board.isGameOver()) {
            const std::vector<Stockfish::Move> &moves = board.validMoves();
            // play a random move
            if (moves.empty()) {
                std::cout << "No valid moves available. Game over." << std::endl;
                break;
            }

            std::vector<int> moveIndices;
            for (const auto &move : moves) {
                moveIndices.push_back(ChessActionCodec::encode(ChessAction(move), board));
                if (moveIndices.back() < 0 || moveIndices.back() >= ChessAction::action_count) {
                    std::cerr << "Encoded move index out of bounds: " << moveIndices.back()
                              << " for move: " << ChessActionCodec::toUci(ChessAction(move))
                              << std::endl;
                }
            }
            // Decode the moves to get the actual Move objects
            const auto decodedActions = ChessActionCodec::decode(moveIndices, board);

            bool anyMismatch = false;
            for (auto [real, decoded] : zip(moves, decodedActions)) {
                if (real.raw() != decoded.move.raw()) {
                    anyMismatch = true;
                }
            }

            if (anyMismatch) {
                std::cerr << "Missmatch of decoded moves in Board: " << board.fen() << std::endl;
                std::cerr << "Valid moves:\n";
                for (const auto &move : moves) {
                    std::cerr << "  " << ChessActionCodec::toUci(ChessAction(move)) << " ("
                              << ChessActionCodec::encode(ChessAction(move), board) << ")\n";
                }
                std::cerr << "Decoded moves:\n";
                for (const ChessAction decoded : decodedActions) {
                    std::cerr << "  " << ChessActionCodec::toUci(decoded) << " ("
                              << ChessActionCodec::encode(decoded, board) << ")\n";
                }
            }

            board.makeMove(moves[rand() % moves.size()]);
        }
        std::cout << "Game over. Result: ";
        const auto winner = board.checkWinner();
        if (winner.has_value()) {
            std::cout << (winner.value() == 1 ? "White wins!" : "Black wins!") << std::endl;
        } else {
            std::cout << "Stalemate or draw." << std::endl;
        }
    }
}
