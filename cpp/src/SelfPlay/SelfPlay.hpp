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

    SelfPlayGame expand(Move move) const {
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
        for (const auto &mem : std::ranges::reverse(game.memory)) {
            auto encodedBoard = encodeBoard(mem.board);
            auto turnGameOutcome = game.board.turn == mem.board.turn ? outcome : -outcome;
            auto resultScore =
                clamp(turnGameOutcome + m_args.result_score_weight * mem.result, -1.0, 1.0);

            for (const auto &[board, visitCounts] :
                 _symmetricVariations(encodedBoard, mem.visitCounts)) {
                _addSample(board, _preprocessVisitCounts(visitCounts), resultScore);
            }

            outcome *= 0.997; // discount the game outcome for each move
        }
    }

private:
    VisitCounts _preprocessVisitCounts(const VisitCounts &visitCounts) const {
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

    void _addSample(const CompressedEncodedBoard &board, const VisitCounts &visitCounts,
                    float resultScore) {
        // TODO have a mutex lock since all threads will be writing to the same dataset
    }
};

class SelfPlay {
private:
    InferenceClient *m_inferenceClient;
    SelfPlayWriter *m_writer;
    SelfPlayParams m_args;

    std::unordered_map<SelfPlayGame, int> m_selfPlayGames;

    MCTS m_mcts;

public:
    SelfPlay(InferenceClient *inferenceClient, SelfPlayWriter *writer, SelfPlayParams args)
        : m_inferenceClient(inferenceClient), m_writer(writer), m_args(args),
          m_mcts(inferenceClient, args.mcts) {
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

        for (auto &[spg_count, mcts_result] : zip(currentSelfPlayGames, results)) {

            auto &[game, count] = spg_count;
            game.memory.emplace_back(game.board.copy(), mcts_result.visits, mcts_result.result);

            m_selfPlayGames.erase(game);

            if (mcts_result.result < m_args.resignation_threshold) {
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

                while (sum(gameActionProbabilities) > 0.0) {
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
                        gameActionProbabilities[encodeMove(move)] = 0.0;
                    }
                }
                if (sum(gameActionProbabilities) == 0.0) {
                    const auto [newGame, move] =
                        _sampleSPG(game, mcts_result.visits.actionProbabilities());
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
                                             const ActionProbabilities &actionProbabilities) {
        Move move = _sampleMove(game.playedMoves.size(), actionProbabilities);

        SelfPlayGame newGame = game.expand(move);

        if (!newGame.board.isGameOver()) {
            return {newGame, move};
        }

        // Game is over, write the result
        std::optional<float> result = getBoardResultScore(newGame.board);
        assert(result.has_value());
        m_writer->write(newGame, result.value(), false, false);

        return {SelfPlayGame(), move};
    }

    Move _sampleMove(int numMoves, const ActionProbabilities &actionProbabilities) const {
        int moveIndex = -1;
        // ActionProbabilities is a std::array<float, ACTION_SIZE>

        if (numMoves >= m_args.num_moves_after_which_to_play_greedy) {
            moveIndex = argmax(actionProbabilities);
        } else {
            assert(m_args.temperature > 0.0);
            auto temperature = pow(actionProbabilities, 1.0 / m_args.temperature);
            temperature = div(temperature, sum(temperature));

            moveIndex = multinomial(temperature);
        }

        return decodeMove(moveIndex);
    }
};