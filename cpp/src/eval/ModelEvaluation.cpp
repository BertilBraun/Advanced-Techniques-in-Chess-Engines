#include "ModelEvaluation.hpp"

#include "MoveEncoding.hpp"
#include "main.hpp"

Results ModelEvaluation::play_two_models_search(std::string const &model_path) {
    // build the opponent evaluator
    BatchEvaluator opponent_eval;

    if (model_path == "random") {
        // random move policy
        std::mt19937_64 rng{std::random_device{}()};
        opponent_eval = [rng](auto &boards) mutable {
            std::vector<VisitCounts> out;
            out.reserve(boards.size());
            for (auto &b : boards) {
                auto moves = b.legalMoves();
                std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
                out.push_back({{encodeMove(moves[dist(rng)]), 1}});
            }
            return out;
        };
    } else {
        // neural MCTS policy
        auto jit_path = get_jit_model_path(model_path);
        opponent_eval = [jit_path](auto &boards) {
            std::vector<std::string> fens;
            fens.reserve(boards.size());
            for (auto const &b : boards)
                fens.push_back(b.fen());

            auto results = boardInferenceMain(jit_path, fens);
            std::vector<VisitCounts> out;
            out.reserve(results.size());
            for (auto const &[score, visit_counts] : results)
                out.emplace_back(visit_counts);
            return out;
        };
    }

    return play_vs_evaluation_model(opponent_eval, model_path);
}

Results ModelEvaluation::play_vs_evaluation_model(BatchEvaluator const &opponent_eval,
                                                  std::string const &name) {
    // current model evaluator
    BatchEvaluator self_eval = [this](auto &boards) {
        auto [p, _] = get_latest_iteration_save_path(save_path_);
        std::vector<std::string> fens;
        fens.reserve(boards.size());
        for (auto const &b : boards)
            fens.push_back(b.fen());

        auto results = boardInferenceMain(p, fens);
        std::vector<VisitCounts> out;
        out.reserve(results.size());
        for (auto const &[score, visit_counts] : results)
            out.emplace_back(visit_counts);
        return out;
    };

    int half = num_games_ / 2;
    Results r;
    r += play_two_models_search_internal(self_eval, opponent_eval, half, name + "_vs_current");
    r += -play_two_models_search_internal(opponent_eval, self_eval, half, "current_vs_" + name);
    return r;
}

std::optional<chess::Color> check_winner(chess::Board &board) {
    auto result = board.result();
    if (result == "1-0")
        return chess::WHITE;
    if (result == "0-1")
        return chess::BLACK;
    return std::nullopt;
}

Results ModelEvaluation::play_two_models_search_internal(BatchEvaluator const &eval1,
                                                         BatchEvaluator const &eval2, int num_games,
                                                         std::string const &name) {
    Results results;

    // 1) initialize boards and histories
    std::vector<chess::Board> games;
    games.reserve(num_games);
    std::vector<std::vector<std::string>> histories(num_games);

    const std::vector<std::string> opening_fens = {
        "rnbqkb1r/pppp1ppp/5n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3", // Ruy-Lopez (Spanish
                                                                            // Game)
        "rnbqkb1r/pppp1ppp/5n2/4p3/B3P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",  // Italian Game
        "rnbqkb1r/pppp1ppp/5n2/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 3 3",   // Scotch Game
        "rnbqkb1r/pp1ppppp/5n2/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 2 2",  // Sicilian Defense
        "rnbqkb1r/pppp1ppp/4pn2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 2 3",    // French Defense
        "rnbqkb1r/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR w KQkq c6 2 2",    // Caro-Kann Defense
        "rnbqkb1r/ppp1pppp/3p4/8/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",   // Pirc Defense
        "rnbqkb1r/ppp1pppp/3p4/8/4P3/3P1N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",  // Modern Defense
        "rnbqkb1r/pppp1ppp/8/4p3/3Pn3/5N2/PPP2PPP/RNBQKB1R w KQkq - 3 3",   // Alekhine’s Defense
        "rnbqkb1r/pppp1ppp/5n2/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3", // King's Indian Defense
        "rnbqkb1r/pppp1ppp/5n2/4p3/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 2 3", // Grünfeld Defense
        "rnbqkb1r/pp1ppppp/5n2/2p5/3PP3/8/PPP2PPP/RNBQKBNR w KQkq c6 2 2",  // Queen’s Gambit
                                                                            // Declined
        "rnbqkb1r/pp1ppppp/5n2/8/2pPP3/8/PP3PPP/RNBQKBNR w KQkq - 0 3", // Queen’s Gambit Accepted
        "rnbqkb1r/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR w KQkq c6 2 2",     // Slav Defense
        "rnbqkb1r/pppp1ppp/4pn2/8/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq - 2 3",   // Nimzo-Indian Defense
        "rnbqkb1r/pppp1ppp/5n2/4p3/2P5/5NP1/PP1PPP1P/RNBQKB1R b KQkq - 2 3", // Catalan Opening
        "rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 2 2",      // English Opening
        "rnbqkb1r/ppppppp1/5n2/8/4P2p/8/PPPP1PPP/RNBQKBNR w KQkq - 2 2",     // Dutch Defense
        "rnbqkb1r/pppppppp/5n2/8/3PP3/4B3/PPP2PPP/RN1QKBNR b KQkq - 3 3",    // London System
        "rnbqkb1r/pppppppp/5n2/8/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 2",    // Réti Opening
    };
    for (int i = 0; i < num_games; ++i) {
        auto b = chess::Board::fromFEN(opening_fens[i % opening_fens.size()]);
        assert(b.is_valid());
        games.push_back(std::move(b));
        histories[i] = {"FEN\"" + games[i].fen() + "\""};
    }

    // 2) play until all games finish
    while (!games.empty()) {
        int N = games.size();

        // split indices by turn
        std::vector<int> white_idx, black_idx;
        white_idx.reserve(N);
        black_idx.reserve(N);
        for (int i = 0; i < N; ++i) {
            if (games[i].turn == chess::Color::WHITE)
                white_idx.push_back(i);
            else
                black_idx.push_back(i);
        }

        // build position lists
        std::vector<chess::Board> white_boards, black_boards;
        white_boards.reserve(white_idx.size());
        black_boards.reserve(black_idx.size());
        for (int i : white_idx)
            white_boards.push_back(games[i]);
        for (int i : black_idx)
            black_boards.push_back(games[i]);

        // 3) get raw visit‐counts from both evaluators
        auto raw1 = eval1(white_boards); // vector<vector<pair<move_i, count>>>
        auto raw2 = eval2(black_boards);

        // 4) for each side, apply penalty, pick best move, push & record in history
        auto apply_moves = [&](auto &raw, auto const &idxs) {
            for (size_t k = 0; k < idxs.size(); ++k) {
                int game_i = idxs[k];
                auto &board = games[game_i];
                auto visits = raw[k]; // vector<pair<move_index,int>>

                // **NEW**: build a full size‐ACTION_SIZE probability array
                auto probs = actionProbabilities(visits);

                // penalty for last up to 10 moves
                auto &hist = histories[game_i];
                int L = hist.size();
                for (int j = 0; j < std::min(10, L); ++j) {
                    if (hist[L - 1 - j].size() > 6) // skip FEN
                        continue;
                    int enc = encodeMove(chess::Move::fromUci(hist[L - 1 - j]));
                    probs[enc] /= (j + 1);
                }

                // normalize
                double probsSum = sum(probs);
                for (auto &p : probs)
                    p /= probsSum;

                // pick the best move
                int best = argmax(probs);
                auto move = decodeMove(best);
                board.push(move);

                // record in our separate history
                histories[game_i].push_back(move.uci());
            }
        };

        apply_moves(raw1, white_idx);
        apply_moves(raw2, black_idx);

        // 5) collect finished games, log them, keep the rest
        std::vector<chess::Board> next_games;
        std::vector<std::vector<std::string>> next_histories;
        next_games.reserve(N);
        next_histories.reserve(N);

        for (int i = 0; i < N; ++i) {
            auto &board = games[i];
            if (board.isGameOver()) {
                auto winner = check_winner(board);
                results.update(winner, chess::Color::WHITE);

                // join history + final FEN
                std::string moves = "";
                for (size_t m = 0; m < histories[i].size(); ++m) {
                    if (m)
                        moves += ",";
                    moves += histories[i][m];
                }

                // prefix: "W:" / "B:" / "D:"
                char prefix = winner ? (*winner == chess::Color::WHITE ? 'W' : 'B') : 'D';
                std::string message = std::string(1, prefix) + ":" + moves;
                logger_->add_text("evaluation_moves/" + std::to_string(iteration_) + "/" + name,
                                  current_time_step(), message.c_str());
            } else {
                next_games.push_back(board);
                next_histories.push_back(histories[i]);
            }
        }

        games.swap(next_games);
        histories.swap(next_histories);
    }

    return results;
}
