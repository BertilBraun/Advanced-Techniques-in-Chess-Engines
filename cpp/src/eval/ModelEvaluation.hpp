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
        : iteration_(iteration), save_path_(save_path), num_games_(num_games), logger_(logger) {}

    /// Exposed to Python as play_two_models_search(model_path)
    Results play_two_models_search(std::string const &model_path);

private:
    using BatchEvaluator = std::function<std::vector<VisitCounts>(std::vector<chess::Board> &)>;

    std::string get_jit_model_path(std::string const &p) const {
        if (p.ends_with(".jit.pt"))
            return p;
        if (p.ends_with(".pt"))
            return p.substr(0, p.size() - 3) + ".jit.pt";
        throw std::runtime_error("Invalid model path: " + p);
    }

    Results play_vs_evaluation_model(BatchEvaluator const &opponent_eval, std::string const &name);

    Results play_two_models_search_internal(BatchEvaluator const &eval1,
                                            BatchEvaluator const &eval2, int num_games,
                                            std::string const &name);

    int iteration_;
    std::string save_path_;
    int num_games_;
    TensorBoardLogger *logger_;
};
