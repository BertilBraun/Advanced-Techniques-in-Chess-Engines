#pragma once

#include "common.hpp"

#include "Subprocess.hpp"

float normalizeEvaluation(int score_cp) {
    float score = static_cast<float>(score_cp) / 1000.0f;
    return std::max(-1.0f, std::min(1.0f, score));
}

class StockfishEvaluator {
public:
    StockfishEvaluator(const std::string &pathToStockfish) : stockfish(pathToStockfish) {
        stockfish << "uci";
        stockfish << "isready";

        std::string output;
        while (output.find("readyok") == std::string::npos)
            stockfish >> output;
    }

    ~StockfishEvaluator() { stockfish << "quit"; }

    float evaluatePosition(const std::string &fen, int depth = 15) {
        stockfish << "position fen " + fen;
        stockfish << "go depth " + std::to_string(depth);

        int boardEvaluation = 0;

        std::string line;
        stockfish >> line;

        while (line.find("bestmove") == std::string::npos) {
            if (line.find("score cp") != std::string::npos) {
                std::istringstream iss(line);
                std::string token;
                while (iss >> token) {
                    if (token == "cp") {
                        iss >> boardEvaluation;
                        break;
                    }
                }
            }

            if (line.find("mate") != std::string::npos) {
                std::istringstream iss(line);
                std::string token;
                while (iss >> token) {
                    if (token == "mate") {
                        int mateInMoves;
                        iss >> mateInMoves;
                        boardEvaluation = mateInMoves > 0 ? 1000 : -1000;
                        break;
                    }
                }
            }

            stockfish >> line;
        }

        // Normalize and clamp the board evaluation to [-1, 1]
        return normalizeEvaluation(boardEvaluation);
    }

private:
    Subprocess stockfish;
};
