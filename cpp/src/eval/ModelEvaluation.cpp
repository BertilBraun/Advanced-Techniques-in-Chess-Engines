#include "ModelEvaluation.hpp"

#include "MoveEncoding.hpp"
#include "main.hpp"

Results ModelEvaluation::playTwoModelsSearch(std::string const &model_path) {
    // build the opponent evaluator
    BatchEvaluator opponentEval;

    if (model_path == "random") {
        // random move policy
        std::mt19937_64 rng{std::random_device{}()};
        opponentEval = [rng](auto &boards) mutable {
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
        auto jitPath = getJitModelPath(model_path);
        opponentEval = [jitPath](auto &boards) {
            std::vector<std::string> fens;
            fens.reserve(boards.size());
            for (auto const &b : boards)
                fens.push_back(b.fen());

            auto results = boardInferenceMain(jitPath, fens);
            std::vector<VisitCounts> out;
            out.reserve(results.size());
            for (auto const &[score, visit_counts] : results)
                out.emplace_back(visit_counts);
            return out;
        };
    }

    return playVsEvaluationModel(opponentEval, model_path);
}

Results ModelEvaluation::playVsEvaluationModel(BatchEvaluator const &opponent_eval,
                                               std::string const &name) {
    // current model evaluator
    BatchEvaluator selfEval = [this](auto &boards) {
        auto [p, _] = getLatestIterationSavePath(m_savePath);
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

    int half = m_numGames / 2;
    Results r;
    r += playTwoModelsSearchInternal(selfEval, opponent_eval, half, name + "_vs_current");
    r += -playTwoModelsSearchInternal(opponent_eval, selfEval, half, "current_vs_" + name);
    return r;
}

std::optional<chess::Color> checkWinner(chess::Board &board) {
    auto result = board.result();
    if (result == "1-0")
        return chess::WHITE;
    if (result == "0-1")
        return chess::BLACK;
    return std::nullopt;
}

Results ModelEvaluation::playTwoModelsSearchInternal(BatchEvaluator const &eval1,
                                                     BatchEvaluator const &eval2, int num_games,
                                                     std::string const &name) {
    Results results;

    // 1) initialize boards and histories
    std::vector<chess::Board> games;
    games.reserve(num_games);
    std::vector<std::vector<std::string>> histories(num_games);

    const std::vector<std::string> openingFens = {
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
        auto b = chess::Board::fromFEN(openingFens[i % openingFens.size()]);
        assert(b.is_valid());
        games.push_back(std::move(b));
        histories[i] = {"FEN\"" + games[i].fen() + "\""};
    }

    // 2) play until all games finish
    while (!games.empty()) {
        int n = games.size();

        // split indices by turn
        std::vector<int> whiteIdx, blackIdx;
        whiteIdx.reserve(n);
        blackIdx.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (games[i].turn == chess::Color::WHITE)
                whiteIdx.push_back(i);
            else
                blackIdx.push_back(i);
        }

        // build position lists
        std::vector<chess::Board> whiteBoards, blackBoards;
        whiteBoards.reserve(whiteIdx.size());
        blackBoards.reserve(blackIdx.size());
        for (int i : whiteIdx)
            whiteBoards.push_back(games[i]);
        for (int i : blackIdx)
            blackBoards.push_back(games[i]);

        // 3) get raw visit‐counts from both evaluators
        auto raw1 = eval1(whiteBoards); // vector<vector<pair<move_i, count>>>
        auto raw2 = eval2(blackBoards);

        // 4) for each side, apply penalty, pick best move, push & record in history
        auto applyMoves = [&](auto &raw, auto const &idxs) {
            for (size_t k = 0; k < idxs.size(); ++k) {
                int gameI = idxs[k];
                auto &board = games[gameI];
                auto visits = raw[k]; // vector<pair<move_index,int>>

                // **NEW**: build a full size‐ACTION_SIZE probability array
                auto probs = actionProbabilities(visits);

                // penalty for last up to 10 moves
                auto &hist = histories[gameI];
                int l = hist.size();
                for (int j = 0; j < std::min(10, l); ++j) {
                    if (hist[l - 1 - j].size() > 6) // skip FEN
                        continue;
                    int enc = encodeMove(chess::Move::fromUci(hist[l - 1 - j]));
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
                histories[gameI].push_back(move.uci());
            }
        };

        applyMoves(raw1, whiteIdx);
        applyMoves(raw2, blackIdx);

        // 5) collect finished games, log them, keep the rest
        std::vector<chess::Board> nextGames;
        std::vector<std::vector<std::string>> nextHistories;
        nextGames.reserve(n);
        nextHistories.reserve(n);

        for (int i = 0; i < n; ++i) {
            auto &board = games[i];
            if (board.isGameOver()) {
                auto winner = checkWinner(board);
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
                m_logger->addText("evaluation_moves/" + std::to_string(m_iteration) + "/" + name,
                                  currentTimeStep(), message.c_str());
            } else {
                nextGames.push_back(board);
                nextHistories.push_back(histories[i]);
            }
        }

        games.swap(nextGames);
        histories.swap(nextHistories);
    }

    return results;
}
