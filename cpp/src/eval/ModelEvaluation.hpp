#pragma once

#include "common.hpp"

#include "MCTS/VisitCounts.hpp"

struct Results {
    int wins = 0;
    int losses = 0;
    int draws = 0;

    void update(std::optional<chess::Color> result, chess::Color main_player) {
        if (!result) {
            ++draws;
        } else if (*result == main_player) {
            ++wins;
        } else {
            ++losses;
        }
    }

    Results &operator+=(Results const &o) {
        wins += o.wins;
        losses += o.losses;
        draws += o.draws;
        return *this;
    }

    Results operator-() const { return Results{losses, wins, draws}; }
};

class ModelEvaluation {
public:
    ModelEvaluation(int iteration, std::string const &save_path, int num_games,
                    TensorBoardLogger *logger)
        : m_iteration(iteration), m_savePath(save_path), m_numGames(num_games), m_logger(logger) {}

    /// Exposed to Python as play_two_models_search(model_path)
    Results playTwoModelsSearch(std::string const &model_path);

private:
    using BatchEvaluator = std::function<std::vector<VisitCounts>(std::vector<chess::Board> &)>;

    std::string getJitModelPath(std::string const &p) const {
        if (p.ends_with(".jit.pt"))
            return p;
        if (p.ends_with(".pt"))
            return p.substr(0, p.size() - 3) + ".jit.pt";
        throw std::runtime_error("Invalid model path: " + p);
    }

    Results playVsEvaluationModel(BatchEvaluator const &opponent_eval, std::string const &name);

    Results playTwoModelsSearchInternal(BatchEvaluator const &eval1, BatchEvaluator const &eval2,
                                        int num_games, std::string const &name);

    int m_iteration;
    std::string m_savePath;
    int m_numGames;
    TensorBoardLogger *m_logger;
};
