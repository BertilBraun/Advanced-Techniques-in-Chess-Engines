#pragma once

#include "common.hpp"

#include "Subprocess.hpp"

float normalizeEvaluation(int score_cp) {
    float score = static_cast<float>(score_cp) / 1000.0f;
    return std::max(-1.0f, std::min(1.0f, score));
}

class StockfishEvaluator {
public:
    StockfishEvaluator(const std::string &pathToStockfish) : stockfish(pathToStockfish, "r+") {
        stockfish << "uci";
        stockfish << "isready";

        std::string output;
        while (output.find("readyok") == std::string::npos)
            stockfish >> output;
    }

    ~StockfishEvaluator() { stockfish << "quit"; }

    float evaluatePosition(const std::string &fen) {
        stockfish << "position fen " + fen;
        stockfish << "go depth 20"; // Adjust depth as needed

        int boardEvaluation = 0;

        for (auto line : stockfish.readLines()) {
            if (line.find("bestmove") == std::string::npos)
                break;

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
        }

        // Normalize and clamp the board evaluation to [-1, 1]
        return normalizeEvaluation(boardEvaluation);
    }

private:
    Subprocess stockfish;
};
