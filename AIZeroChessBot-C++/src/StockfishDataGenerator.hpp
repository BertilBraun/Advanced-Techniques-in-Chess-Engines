#pragma once

#include "common.hpp"

#include "BoardEncoding.hpp"
#include "MoveEncoding.hpp"

#include "Dataset.hpp"
#include "SelfPlayWriter.hpp"

#include "Stockfish.hpp"

#include "json.hpp"

using json = nlohmann::json;

/*
Reason for this class:

We currently have a problem, that the self-play games at the beginning are not very good. This is
because the model is not trained yet and therefore predicts bad moves as well as bad evaluations.
This means, that many of the expanded nodes in the MCTS are evaluated by the model instead of the
endgame score. This means, that we are training the model with random data, which is not very
useful. AlphaZero solves this problem by simply searching more iterations per move, which more often
leads to the endgame score being used. However, this is not viable for us, because we are using
way less computational resources than AlphaZero.

My idea to overcome this problem is to use grandmaster games and stockfish evaluations to generate
the training data for the first few iterations. This way, we can train the model with good data from
the beginning and therefore improve the self-play games. This should lead to a better model and
therefore better self-play games. After a few iterations, the model should be good enough to
generate good training data by itself, so that the self improvement loop can start.

Grandmaster games can be found here: https://database.nikonoel.fr/
Lichess evaluations can be found here: https://database.lichess.org/#evals
Stockfish can be found here: https://stockfishchess.org/download/

(We are using Stockfish 8 on the cluster, because it is the only version that compiles there)

*/

class StockfishDataGenerator {
public:
    StockfishDataGenerator(const TrainingArgs &args) : m_selfPlayWriter(args) {}

    void generateDataFromEliteGames(const std::string &pathToEliteGames) {
        // Generate training data from elite games
        // These can be found here: https://database.nikonoel.fr/
        // These should be preprocessed by PreprocessGenerationData.py
        // The format should then be: "score move1uci move2uci move3uci ..."

        if (hasGenerated("elite_games_generated"))
            return;

        std::ifstream file(pathToEliteGames);

        size_t numGames = 0;
        std::string line;
        while (std::getline(file, line)) {
            std::vector<std::string> tokens = split(line, ' ');

            float score = std::stof(tokens[0]);
            std::vector<Move> moves;
            for (size_t i = 1; i < tokens.size(); ++i) {
                moves.push_back(Move::fromUci(tokens[i]));
            }
            writeLine(Board(), moves, score);

            numGames++;
            if (numGames % 1000 == 0)
                reportProgress(numGames, 1000000, "Elite games"); // TODO
        }

        markGenerated("elite_games_generated");
    }

    void generateDataFromLichessEval(const std::string &pathToLichessEvals, bool createLines,
                                     size_t rank, size_t numProcesses) {
        // Generate training data from lichess evals
        // These can be found here: https://database.lichess.org/#evals

        if (hasGenerated("lichess_evals_generated"))
            return;

        std::ifstream file(pathToLichessEvals);

        size_t totalLines = 21158953;
        size_t linesPerProcess = totalLines / numProcesses;
        size_t start = rank * linesPerProcess;

        size_t numGames = 0;

        std::string line;
        for (size_t i = 0; i < start; ++i) {
            std::getline(file, line);
        }
        while (std::getline(file, line) && numGames < linesPerProcess) {
            json eval = json::parse(line);
            auto board = Board::fromFEN(eval["fen"]);
            auto lines = parseLichessEvalPolicy(eval["evals"], board);

            if (createLines) {
                for (auto &[moves, value] : lines) {
                    writeLine(board, moves, value);
                }
            }

            numGames++;
            if (numGames % 1000 == 0)
                reportProgress(numGames, linesPerProcess, "Evaluations");
        }

        markGenerated("lichess_evals_generated");
    }

    void generateDataThroughStockfishSelfPlay(const std::string &pathToStockfish,
                                              size_t numGames = 1000, size_t numMoves = 10) {
        // Create a new board
        // For the top numMoves moves, evaluate the position and write the data
        // Recurse on the new position until the game is over
        // Do this for numGames games
        StockfishEvaluator evaluator(pathToStockfish);

        for (size_t iteration = 0; tqdm(iteration, numGames, "Stockfish Self Play"); ++iteration) {
            Board board;
            stockfishSelfPlay(evaluator, board, numMoves);
        }
    }

private:
    SelfPlayWriter m_selfPlayWriter;
    size_t m_written = 0;
    double m_valueSum = 0.0;

    void reportProgress(size_t current, size_t total, const std::string &message) {
        double average = m_written == 0 ? 0 : m_valueSum / m_written;

        std::string desc = message + " " + std::to_string(m_written) + " written " +
                           std::to_string(average) + " average value";
        tqdm(current, total, desc);
    }

    bool hasGenerated(const std::string &name) {
        std::ifstream file(name);
        return file.good();
    }

    void markGenerated(const std::string &name) {
        std::ofstream file(name);
        file.close();
    }

    std::vector<std::pair<std::vector<Move>, float>> parseLichessEvalPolicy(const json &evals,
                                                                            Board &board) {
        std::vector<PolicyMove> policy;
        std::vector<std::pair<std::vector<Move>, float>> lines;

        float boardScore = -1.0f;

        for (const auto &eval : evals) {
            for (const auto &pv : eval["pvs"]) {
                try {
                    float score = parseLichessPvScore(pv);
                    std::vector<Move> moves = parseLichessPvMoves(pv);

                    policy.emplace_back(moves[0], score);
                    lines.emplace_back(moves, score);

                    boardScore = std::max(score, boardScore);
                } catch (std::runtime_error &e) {
                    continue;
                }
            }
        }

        normalizePolicy(policy);

        write(board, policy, board.turn == Color::WHITE ? boardScore : -boardScore);

        return lines;
    }

    void writeLine(const Board &board, const std::vector<Move> &line, float value) {
        Board copy = board.copy();
        for (const Move &move : line) {
            copy.push(move);
            write(copy, {{move, 1.0f}}, value);
        }
    }

    float parseLichessPvScore(const json &pv) const {
        if (pv.contains("cp")) {
            return normalizeEvaluation(pv["cp"].get<int>());
        } else if (pv.contains("mate")) {
            // The higher the mate score, the worse the score should be
            // But in any case, the score should be very close to -1 or 1
            int mateMoves = pv["mate"].get<int>();
            return std::max(-1.0f, std::min(1.0f, (float) mateMoves));
        }
        throw std::runtime_error("Invalid pv");
    }

    std::vector<Move> parseLichessPvMoves(const json &pv) const {
        std::string line = pv["line"].get<std::string>();
        auto moveUCIs = split(line, ' ');

        std::vector<Move> moves(moveUCIs.size());
        for (size_t i = 0; i < moveUCIs.size(); ++i) {
            moves[i] = Move::fromUci(moveUCIs[i]);
        }

        return moves;
    }

    void write(const Board &board, const std::vector<PolicyMove> &policy, float value) {
        torch::Tensor encodedBoard = encodeBoard(board);
        torch::Tensor encodedPolicy = encodeMoves(policy);

        if (m_selfPlayWriter.write(encodedBoard, encodedPolicy, value)) [[likely]] {
            m_written++;
            m_valueSum += (double) value;
        }
    }

    void stockfishSelfPlay(StockfishEvaluator &evaluator, Board &board, size_t numMoves = 10) {
        std::vector<PolicyMove> policy = generateAndWriteDataSample(evaluator, board);

        // Sort the moves by probability and take the top numMoves
        std::sort(policy.begin(), policy.end(),
                  [](const PolicyMove &a, const PolicyMove &b) { return a.second > b.second; });

        // Remove all but the top numMoves moves to reduce the memory footprint
        policy.resize(std::min(numMoves, policy.size()));

        for (auto &[move, _] : policy) {
            Board copy = board.copy();
            copy.push(move);
            if (!copy.isGameOver())
                stockfishSelfPlay(evaluator, copy, numMoves);
        }
    }

    std::vector<PolicyMove> generateAndWriteDataSample(StockfishEvaluator &evaluator,
                                                       Board &board) {
        std::vector<PolicyMove> policy = evaluateMovesPolicy(evaluator, board);
        float value = evaluator.evaluatePosition(board.fen());

        write(board, policy, value);

        return policy;
    }

    std::vector<PolicyMove> evaluateMovesPolicy(StockfishEvaluator &evaluator, Board &board) {
        std::vector<PolicyMove> policy;

        for (const Move &move : board.legalMoves()) {
            Board copy = board.copy();
            copy.push(move);
            float score = evaluator.evaluatePosition(board.fen());
            // Note: The score is negated, because the evaluator is from the perspective of the
            // current player, but the policy should be from the perspective of the last player
            policy.emplace_back(move, -score);
        }

        normalizePolicy(policy);

        return policy;
    }

    void normalizePolicy(std::vector<PolicyMove> &policy) {
        // The policy should be -1.0f <= x <= 1.0f based on the pv scores

        float min = std::numeric_limits<float>::max();
        for (PolicyMove &move : policy) {
            min = std::min(min, move.second);
        }

        for (PolicyMove &move : policy) {
            move.second -= min - 0.01f;
        }

        float sum = 0.0f;
        for (PolicyMove &move : policy) {
            sum += move.second;
        }

        for (PolicyMove &move : policy) {
            move.second /= sum;
        }
    }
};