#pragma once

#include "common.hpp"

#include "BoardEncoding.hpp"
#include "MCTS/MCTS.hpp"
#include "TrainingArgs.hpp"

struct SelfPlayGameMemory {
    Board board;
    VisitCounts visitCounts;
    float result;
};

struct SelfPlayGame {
    Board board;
    std::vector<SelfPlayGameMemory> memory;
    std::vector<Move> playedMoves;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

    SelfPlayGame() : board(Board()), startTime(std::chrono::high_resolution_clock::now()) {}

    SelfPlayGame expand(Move move) {
        SelfPlayGame newGame = this->copy();
        newGame.board.makeMove(move);
        newGame.playedMoves.push_back(move);
        return newGame;
    }

    SelfPlayGame copy() const {
        SelfPlayGame newGame = SelfPlayGame(board);
        newGame.memory = memory;
        newGame.playedMoves = playedMoves;
        newGame.startTime = startTime;
        return newGame;
    }

    // define hash function for SelfPlayGame by hashing the board fen
    friend std::hash<SelfPlayGame>;
};

namespace std {
template <> struct hash<SelfPlayGame> {
    std::size_t operator()(const SelfPlayGame &game) const {
        return std::hash<std::string>{}(game.board.fen());
    }
};
} // namespace std

class SelfPlayWriter {
    SelfPlayParams m_args;

public:
    void write(const SelfPlayGame &game, float outcome, bool resignation, bool tooLong) {
        // TODO: Implement
        /*
        for mem in spg.memory[::-1]:
            encoded_board = CurrentGame.get_canonical_board(mem.board)
            turn_game_outcome = game_outcome if mem.board.current_player == spg.board.current_player
    else -game_outcome

            for board, visit_counts in CurrentGame.symmetric_variations(encoded_board,
    mem.visit_counts): self.dataset.add_sample( board.astype(np.int8).copy(),
                    self._preprocess_visit_counts(visit_counts),
                    clamp(turn_game_outcome + self.args.result_score_weight * mem.result_score, -1,
    1), # lerp(turn_game_outcome, mem.result_score, self.args.result_score_weight),
                )

            game_outcome *= 0.997  # discount the game outcome for each move

    def _preprocess_visit_counts(self, visit_counts: list[tuple[int, int]]) -> list[tuple[int,
    int]]: total_visits = sum(visit_count for _, visit_count in visit_counts) # Remove moves with
    less than 1% of the total visits visit_counts = [(move, count) for move, count in visit_counts
    if count >= total_visits * 0.01]

        return visit_counts
*/

        for (const auto &mem : std::ranges::reverse(game.memory)) {
            auto encodedBoard = encodeBoard(mem.board);
            auto turnGameOutcome = game.board.turn == mem.board.turn ? outcome : -outcome;
            auto resultScore =
                clamp(turnGameOutcome + m_args.result_score_weight * mem.result, -1.0, 1.0);

            for (const auto &[board, visitCounts] :
                 symmetricVariations(encodedBoard, mem.visitCounts)) {
                _addSample(board, _preprocessVisitCounts(visitCounts), resultScore);
            }

            outcome *= 0.997; // discount the game outcome for each move
        }
    }

private:
        VisitCounts _preprocessVisitCounts(const VisitCounts &visitCounts) {
        int totalVisits = 0;
        for (const auto &[_, visitCount] : visitCounts.visits) {
            totalVisits += visitCount;
        }

        VisitCounts newVisitCounts;
        for (const auto &[move, count] : visitCounts.visits) {
            if (count >= totalVisits * 0.01) {
                newVisitCounts.visits.push_back({move, count});
            }
        }

        return newVisitCounts;
    }

    void _addSample(const torch::Tensor &board, const VisitCounts &visitCounts, float resultScore) {
    }
};

class SelfPlay {
private:
    InferenceClient *m_inferenceClient;
    SelfPlayWriter *m_writer;
    SelfPlayParams m_args;

    std::unordered_map<SelfPlayGame, int> m_selfPlayGames;

    MCST m_mcst;

public:
    SelfPlay(InferenceClient *inferenceClient, SelfPlayWriter *writer, SelfPlayParams args)
        : m_inferenceClient(inferenceClient), m_writer(writer), m_args(args),
          m_mcst(inferenceClient, args.mcts) {
        m_selfPlayGames[SelfPlayGame()] = args.num_parallel_games;
    }

    void selfPlay() {
        std::vector<Board> boards;
        boards.reserve(m_selfPlayGames.size());
        for (const auto &[game, _] : m_selfPlayGames) {
            boards.push_back(game.board);
        }

        std::vector<MCTSResult> results = m_mcts.search(boards);

        std::unordered_map<SelfPlayGame, int> currentSelfPlayGames = m_selfPlayGames;

        int spg_counts = 0;
        for (const auto &[_, count] : currentSelfPlayGames) {
            assert(count > 0);
            spg_counts += count;
        }
        assert(spg_counts == m_args.num_parallel_games);

        for (const auto &[spg_count, mcts_result] : zip(currentSelfPlayGames, results)) {

            const auto &[game, count] = spg_count;
            game.memory.emplace_back(game.board.copy(), mcts_result.visitCounts,
                                     mcts_result.result);

            m_selfPlayGames.erase(game);

            if (mcts_result.result < m_args.resign_threshold) {
                m_writer->write(game, mcts_result.result, true, false);
                m_selfPlayGames[SelfPlayGame()] += count;
                continue;
            }

            if (game.playedMoves.size() >= m_args.max_moves) {
                _handleTooLongGame(game);
                m_selfPlayGames[SelfPlayGame()] += count;
                continue;
            }

            for (auto _ : range(count)) {
                auto gameActionProbabilities = mcts_result.visits.actionProbabilities();

                while (gameActionProbabilities.sum().item<float>() > 0.0) {
                    const auto [newGame, move] = _sampleSPG(game, gameActionProbabilities);

                    bool isDuplicate = m_selfPlayGames.find(newGame) != m_selfPlayGames.end();
                    // if move was played in the last 16 moves, then it is a repeated move
                    bool isRepeatedMove = false;
                    for (int i : range(1, 16)) {
                        if (game.playedMoves.size() < i)
                            break;
                        if (game.playedMoves[game.playedMoves.size() - i] == move) {
                            isRepeatedMove = true;
                            break;
                        }
                    }
                    if (!isDuplicate && !isRepeatedMove) {
                        m_selfPlayGames[newGame] = 1;
                    } else {
                        gameActionProbabilities[move] = 0.0;
                    }
                }
                if (gameActionProbabilities.sum().item<float>() == 0.0) {
                    const auto [newGame, move] =
                        _sampleSPG(game, mcst_result.visits.actionProbabilities());
                    m_selfPlayGames[newGame] += 1;
                }
            }
        }
    }

private:
    void _handleTooLongGame(const SelfPlayGame &game) {
        int numWhitePieces = game.board.countPieces(WHITE);
        int numBlackPieces = game.board.countPieces(BLACK);

        if (numWhitePieces < 4 || numBlackPieces < 4) {
            // Find out which player has better value pieces remaining.
            float materialScore = getMaterialScore(game.board);
            float outcome = game.board.turn == WHITE ? materialScore : -materialScore;
            m_writer->write(game, outcome, false, true);
        }
    }

    std::pair<SelfPlayGame, Move> _sampleSPG(const SelfPlayGame &game,
                                             const torch::Tensor &actionProbabilities) {
        Move move = _sampleMove(game.playedMoves.size(), actionProbabilities);

        SelfPlayGame newGame = game.expand(move);

        if (!newGame.board.isGameOver()) {
            return {newGame, move};
        }

        // Game is over, write the result
        float result = getBoardResultScore(newGame.board);
        assert(result.has_value());
        m_writer->write(newGame, result.value(), false, false);

        return {SelfPlayGame(), move};
    }

    Move _sampleMove(int numMoves, const torch::Tensor &actionProbabilities) {
        if (numMoves >= m_args.num_moves_after_which_to_play_greedy) {
            return decodeMove(torch::argmax(actionProbabilities).item<int>());
        } else {
            assert(m_args.temperature > 0.0);
            torch::Tensor temperature = actionProbabilities * *(1.0 / m_args.temperature);
            temperature /= temperature.sum();
            return decodeMove(torch::multinomial(temperature, 1).item<int>());
        }
    }
};