#include "SelfPlay.hpp"

SelfPlay::SelfPlay(InferenceClient *inferenceClient, SelfPlayWriter *writer, SelfPlayParams args,
                   TensorBoardLogger *logger)
    : m_writer(writer), m_args(args), m_mcts(inferenceClient, args.mcts, logger) {
    // Initialize the self-play games with the number of parallel games.
    m_selfPlayGames.resize(args.num_parallel_games);
}

void SelfPlay::selfPlay() {
    TimeItGuard timer("SelfPlay");
    std::vector<Board> boards;
    boards.reserve(m_selfPlayGames.size());
    for (const SelfPlayGame &game : m_selfPlayGames) {
        boards.push_back(game.board);
    }

    const std::vector<MCTSResult> results = m_mcts.search(boards);

    assert(m_selfPlayGames.size() == m_args.num_parallel_games);

    for (auto i : range(m_selfPlayGames.size())) {
        SelfPlayGame &game = m_selfPlayGames[i];
        const MCTSResult &mctsResult = results[i];

        game.memory.emplace_back(game.board.copy(), mctsResult.visits, mctsResult.result);

        if (mctsResult.result < m_args.resignation_threshold) {
            m_writer->write(game, mctsResult.result, true, false);
            m_selfPlayGames[i] = SelfPlayGame();
            continue;
        }

        if ((int) game.playedMoves.size() >= m_args.max_moves) {
            handleTooLongGame(game);
            m_selfPlayGames[i] = SelfPlayGame();
            continue;
        }

        ActionProbabilities gameActionProbabilities = actionProbabilities(mctsResult.visits);
        if (sum(gameActionProbabilities) == 0.0) {
            log("WARNING: No valid moves found for game ", game.board.fen(), " with result ",
                mctsResult.result);
            m_selfPlayGames[i] = SelfPlayGame();
            continue;
        }

        while (sum(gameActionProbabilities) > 0.0) {
            const auto [newGame, move] = sampleSpg(game, gameActionProbabilities);

            bool isDuplicate = false;
            for (const SelfPlayGame &existingGame : m_selfPlayGames) {
                if (existingGame.board == newGame.board) {
                    isDuplicate = true;
                    break;
                }
            }
            // if move was played in the last 16 moves, then it is a repeated move
            bool isRepeatedMove = false;
            for (int i : range(1, 16)) {
                if ((int) game.playedMoves.size() < i)
                    break;
                if (game.playedMoves[game.playedMoves.size() - i] == move) {
                    isRepeatedMove = true;
                    break;
                }
            }
            if (!isDuplicate && !isRepeatedMove) {
                m_selfPlayGames[i] = newGame;
                break;
            } else {
                gameActionProbabilities[encodeMove(move)] = 0.0;
            }
        }
        if (sum(gameActionProbabilities) == 0.0) {
            const auto [newGame, move] = sampleSpg(game, actionProbabilities(mctsResult.visits));
            m_selfPlayGames[i] = newGame;
        }
    }
}

void SelfPlay::handleTooLongGame(const SelfPlayGame &game) {
    const int numWhitePieces = countPieces(game.board, chess::WHITE);
    const int numBlackPieces = countPieces(game.board, chess::BLACK);

    if (numWhitePieces < 4 || numBlackPieces < 4) {
        // Find out which player has better value pieces remaining.
        const float materialScore = getMaterialScore(game.board);
        const float outcome = game.board.turn == chess::WHITE ? materialScore : -materialScore;
        m_writer->write(game, outcome, false, true);
    }
}

int SelfPlay::countPieces(const Board &board, chess::Color color) const {
    int count = 0;
    for (chess::PieceType pieceType : chess::PIECE_TYPES) {
        count += board.pieces(pieceType, color).size();
    }
    return count;
}

std::pair<SelfPlayGame, Move> SelfPlay::sampleSpg(const SelfPlayGame &game,
                                                  const ActionProbabilities &actionProbabilities) {
    const Move move = sampleMove(game.playedMoves.size(), actionProbabilities);

    SelfPlayGame newGame = game.expand(move);

    if (!newGame.board.isGameOver()) {
        return {newGame, move};
    }

    // Game is over, write the result
    const std::optional<float> result = getBoardResultScore(newGame.board);
    assert(result.has_value());
    m_writer->write(newGame, result.value(), false, false);

    return {SelfPlayGame(), move};
}

Move SelfPlay::sampleMove(int numMoves, const ActionProbabilities &actionProbabilities) const {
    int moveIndex = -1;

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
